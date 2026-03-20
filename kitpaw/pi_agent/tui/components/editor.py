from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from ..autocomplete import AutocompleteItem, AutocompleteProvider
from ..keybindings import get_editor_keybindings
from ..keys import decode_kitty_printable
from ..kill_ring import KillRing
from ..tui import CURSOR_MARKER
from ..undo_stack import UndoStack
from ..utils import get_segmenter, is_punctuation_char, is_whitespace_char, slice_by_column, visible_width


@dataclass(slots=True)
class _EditorState:
    lines: list[str]
    cursor_line: int
    cursor_col: int


class EditorTheme(Protocol):
    def border_color(self, text: str) -> str: ...


@dataclass(slots=True)
class EditorOptions:
    padding_x: int = 0


class Editor:
    _PASTE_MARKER_RE = re.compile(
        r"\[paste #(?P<id>\d+)( (?:(?P<line_prefix>\+)(?P<lines>\d+) lines|(?P<chars>\d+) chars))?\]"
    )

    def __init__(self, tui, theme, options=None) -> None:
        self.state = _EditorState(lines=[""], cursor_line=0, cursor_col=0)
        self.focused = False
        self.tui = tui
        self.theme = theme
        self.options = options if isinstance(options, EditorOptions) else EditorOptions(**(options or {}))
        self.border_color = lambda text: text
        self.autocomplete_provider: AutocompleteProvider | None = None
        self.autocomplete_list = None
        self.autocomplete_state: str | None = None
        self.autocomplete_prefix = ""
        self.autocomplete_max_visible = 5
        self.pastes = {}
        self.paste_counter = 0
        self.paste_buffer = ""
        self.is_in_paste = False
        self.history: list[str] = []
        self.history_index = -1
        self.kill_ring = KillRing()
        self.last_action: str | None = None
        self.undo_stack: UndoStack[_EditorState] = UndoStack()
        self.on_submit = None
        self.on_change = None
        self.sticky_col: int | None = None
        self.autocomplete_items: list[AutocompleteItem] = []
        self.autocomplete_index = 0
        self.autocomplete_force_mode = False
        self._pending_jump_direction: str | None = None
        self._layout_width = 80
        self._preferred_visual_col: int | None = None
        self._last_yank_range: tuple[int, int, int, int] | None = None
        self._history_reentry_after_set_text = False

    def get_text(self) -> str:
        return "\n".join(self.state.lines)

    def get_expanded_text(self) -> str:
        return self._expand_text(self.get_text())

    def _replace_text(self, text: str, *, reset_history_index: bool) -> None:
        self.state.lines = text.split("\n") if text else [""]
        self.state.cursor_line = len(self.state.lines) - 1
        self.state.cursor_col = len(self.state.lines[-1])
        self.sticky_col = None
        self._preferred_visual_col = None
        self.last_action = None
        self._last_yank_range = None
        if reset_history_index:
            self.history_index = -1
        self._notify_change()

    def set_text(self, text: str) -> None:
        if text != self.get_text():
            self._push_undo()
        self._replace_text(text, reset_history_index=True)
        self._history_reentry_after_set_text = True
        self._update_autocomplete()

    def add_to_history(self, text: str) -> None:
        if not text.strip():
            return
        if self.history and self.history[-1] == text:
            return
        self.history.append(text)
        self.history = self.history[-100:]

    def set_autocomplete_provider(self, provider: AutocompleteProvider) -> None:
        self.autocomplete_provider = provider
        self._hide_autocomplete()

    def set_padding_x(self, padding: int) -> None:
        self.options.padding_x = padding

    def set_autocomplete_max_visible(self, max_visible: int) -> None:
        self.autocomplete_max_visible = max_visible

    def insert_text_at_cursor(self, text: str) -> None:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        self._history_reentry_after_set_text = False
        self.history_index = -1
        self.sticky_col = None
        self._preferred_visual_col = None
        self._hide_autocomplete()
        self._push_undo()
        line = self.state.lines[self.state.cursor_line]
        before = line[: self.state.cursor_col]
        after = line[self.state.cursor_col :]
        parts = text.split("\n")
        if len(parts) == 1:
            self.state.lines[self.state.cursor_line] = before + text + after
            self.state.cursor_col += len(text)
        else:
            self.state.lines[self.state.cursor_line] = before + parts[0]
            insert_at = self.state.cursor_line + 1
            for part in parts[1:-1]:
                self.state.lines.insert(insert_at, part)
                insert_at += 1
            self.state.lines.insert(insert_at, parts[-1] + after)
            self.state.cursor_line = insert_at
            self.state.cursor_col = len(parts[-1])
        self._notify_change()
        self.last_action = None

    def handle_input(self, data: str) -> None:
        if "\x1b[200~" in data and "\x1b[201~" in data:
            pasted = data.replace("\x1b[200~", "").replace("\x1b[201~", "")
            self._insert_paste(pasted)
            return
        kb = get_editor_keybindings()
        if kb.matches(data, "tab"):
            if self._handle_tab():
                return
        if data != "\n" and kb.matches(data, "submit"):
            line = self.state.lines[self.state.cursor_line]
            if data == "\r" and self.state.cursor_col > 0 and line[self.state.cursor_col - 1 : self.state.cursor_col] == "\\":
                self._push_undo()
                self.state.lines[self.state.cursor_line] = line[: self.state.cursor_col - 1] + line[self.state.cursor_col :]
                self.state.cursor_col -= 1
                self._new_line(skip_undo=True)
                return
            if self.is_showing_autocomplete():
                if self._accept_autocomplete_on_enter():
                    return
            if self.on_submit:
                self.on_submit(self.get_expanded_text())
                self._replace_text("", reset_history_index=True)
            self.undo_stack.clear()
            self._hide_autocomplete()
            return
        if kb.matches(data, "selectCancel") and self.on_submit is None:
            self._pending_jump_direction = None
            return
        if kb.matches(data, "cursorLeft"):
            self._move_left()
            return
        if kb.matches(data, "cursorRight"):
            self._move_right()
            return
        if kb.matches(data, "cursorWordLeft"):
            self._move_word_left()
            return
        if kb.matches(data, "cursorWordRight"):
            self._move_word_right()
            return
        if kb.matches(data, "cursorUp"):
            self._move_up()
            return
        if kb.matches(data, "cursorDown"):
            self._move_down()
            return
        if kb.matches(data, "cursorLineStart"):
            self.state.cursor_col = 0
            self.sticky_col = None
            self._preferred_visual_col = None
            self.last_action = None
            self._hide_autocomplete()
            return
        if kb.matches(data, "cursorLineEnd"):
            self.state.cursor_col = len(self.state.lines[self.state.cursor_line])
            self.sticky_col = None
            self._preferred_visual_col = None
            self.last_action = None
            self._hide_autocomplete()
            return
        if kb.matches(data, "deleteCharBackward"):
            self._backspace()
            return
        if kb.matches(data, "deleteCharForward"):
            self._delete_forward()
            return
        if kb.matches(data, "deleteToLineStart"):
            self._delete_to_line_start()
            return
        if kb.matches(data, "deleteToLineEnd"):
            self._delete_to_line_end()
            return
        if kb.matches(data, "deleteWordBackward"):
            self._delete_word_backward()
            return
        if kb.matches(data, "deleteWordForward"):
            self._delete_word_forward()
            return
        if kb.matches(data, "jumpForward"):
            if self._pending_jump_direction == "forward":
                self._pending_jump_direction = None
                return
            self._pending_jump_direction = "forward"
            return
        if kb.matches(data, "jumpBackward"):
            if self._pending_jump_direction == "backward":
                self._pending_jump_direction = None
                return
            self._pending_jump_direction = "backward"
            return
        if kb.matches(data, "yank"):
            self._yank()
            return
        if kb.matches(data, "yankPop"):
            self._yank_pop()
            return
        if kb.matches(data, "undo"):
            self.undo()
            return
        kitty_printable = decode_kitty_printable(data)
        if kitty_printable is not None:
            self._insert_typed_text(kitty_printable)
            return
        if data == "\r":
            data = "\n"
        if data == "\n":
            self._new_line()
            return
        if not any(ord(ch) < 32 or ord(ch) == 127 for ch in data):
            if self._pending_jump_direction is not None:
                self._perform_jump(data, self._pending_jump_direction)
                self._pending_jump_direction = None
                return
            self._insert_typed_text(data)
            self._update_autocomplete()

    def _push_undo(self) -> None:
        self.undo_stack.push(_EditorState(lines=list(self.state.lines), cursor_line=self.state.cursor_line, cursor_col=self.state.cursor_col))

    def undo(self) -> None:
        state = self.undo_stack.pop()
        if state:
            self.state = state
            self.sticky_col = None
            self._preferred_visual_col = None
            self.history_index = -1
            self.last_action = None
            self._update_autocomplete()
            self._notify_change()

    def _notify_change(self) -> None:
        if self.on_change:
            self.on_change(self.get_text())

    def _move_left(self) -> None:
        self.last_action = None
        self.sticky_col = None
        self._preferred_visual_col = None
        if self.state.cursor_col > 0:
            line = self.state.lines[self.state.cursor_line]
            marker = self._marker_ending_at(line, self.state.cursor_col)
            if marker is not None:
                self.state.cursor_col = marker.start()
            else:
                self.state.cursor_col = self._previous_grapheme_boundary(line, self.state.cursor_col)
        elif self.state.cursor_line > 0:
            self.state.cursor_line -= 1
            self.state.cursor_col = len(self.state.lines[self.state.cursor_line])

    def _move_right(self) -> None:
        self.last_action = None
        self.sticky_col = None
        line = self.state.lines[self.state.cursor_line]
        if self.state.cursor_col < len(line):
            self._preferred_visual_col = None
            marker = self._marker_starting_at(line, self.state.cursor_col)
            if marker is not None:
                self.state.cursor_col = marker.end()
            else:
                self.state.cursor_col = self._next_grapheme_boundary(line, self.state.cursor_col)
        elif self.state.cursor_line < len(self.state.lines) - 1:
            self._preferred_visual_col = None
            self.state.cursor_line += 1
            self.state.cursor_col = 0
        else:
            self._preferred_visual_col = self._visual_col_for_cursor()

    def _move_up(self) -> None:
        self.last_action = None
        if self.sticky_col is None:
            self.sticky_col = self.state.cursor_col
        if self._preferred_visual_col is None:
            self._preferred_visual_col = self._visual_col_for_cursor()
        at_history_navigation_point = self.state.cursor_line == len(self.state.lines) - 1 and self.state.cursor_col == len(self.state.lines[-1])
        text_is_empty = len(self.state.lines) == 1 and self.state.lines[0] == ""
        single_line_set_text_reentry = self._history_reentry_after_set_text and len(self.state.lines) == 1 and not text_is_empty
        if self.history and self.history_index >= 0:
            if self.state.cursor_line > 0:
                self._move_visual_up()
                return
            if self.history_index > 0:
                self.history_index -= 1
                self._replace_text(self.history[self.history_index], reset_history_index=False)
                return
        if self.history and at_history_navigation_point and (text_is_empty or single_line_set_text_reentry):
            if self.history_index == -1:
                self._push_undo()
                self.history_index = len(self.history) - 1
                self._history_reentry_after_set_text = False
            if self.history_index >= 0:
                self._replace_text(self.history[self.history_index], reset_history_index=False)
                return
        self._move_visual_up()

    def _move_down(self) -> None:
        self.last_action = None
        if self.sticky_col is None:
            self.sticky_col = self.state.cursor_col
        if self._preferred_visual_col is None:
            self._preferred_visual_col = self._visual_col_for_cursor()
        if self.history_index >= 0:
            if self.state.cursor_line < len(self.state.lines) - 1:
                self._move_visual_down()
                return
            if self.history_index < len(self.history) - 1:
                self.history_index += 1
                self._replace_text(self.history[self.history_index], reset_history_index=False)
                return
            self.history_index = -1
            self._replace_text("", reset_history_index=False)
            return
        self._move_visual_down()

    def _backspace(self) -> None:
        self.last_action = None
        self.sticky_col = None
        self._preferred_visual_col = None
        if self.state.cursor_col > 0:
            self._push_undo()
            line = self.state.lines[self.state.cursor_line]
            marker = self._marker_ending_at(line, self.state.cursor_col)
            start = marker.start() if marker is not None else self._previous_grapheme_boundary(line, self.state.cursor_col)
            self.state.lines[self.state.cursor_line] = line[:start] + line[self.state.cursor_col :]
            self.state.cursor_col = start
            self._notify_change()
            self._update_autocomplete()
        elif self.state.cursor_line > 0:
            self._push_undo()
            prev = self.state.lines[self.state.cursor_line - 1]
            cur = self.state.lines.pop(self.state.cursor_line)
            self.state.cursor_line -= 1
            self.state.cursor_col = len(prev)
            self.state.lines[self.state.cursor_line] = prev + cur
            self._notify_change()
            self._update_autocomplete()

    def _delete_forward(self) -> None:
        self.last_action = None
        self.sticky_col = None
        self._preferred_visual_col = None
        self._history_reentry_after_set_text = False
        line = self.state.lines[self.state.cursor_line]
        if self.state.cursor_col < len(line):
            self._push_undo()
            marker = self._marker_starting_at(line, self.state.cursor_col)
            end = marker.end() if marker is not None else self._next_grapheme_boundary(line, self.state.cursor_col)
            self.state.lines[self.state.cursor_line] = line[: self.state.cursor_col] + line[end:]
            self._notify_change()
            self._update_autocomplete()

    def _delete_to_line_start(self) -> None:
        self.sticky_col = None
        self._preferred_visual_col = None
        self._history_reentry_after_set_text = False
        line = self.state.lines[self.state.cursor_line]
        if self.state.cursor_col > 0:
            self._push_undo()
            deleted = line[: self.state.cursor_col]
            self.state.lines[self.state.cursor_line] = line[self.state.cursor_col :]
            self.state.cursor_col = 0
            self._push_kill(deleted, prepend=True, action="delete-to-line-start")
            self._notify_change()
            self._update_autocomplete()
        elif self.state.cursor_line > 0:
            self._push_undo()
            previous = self.state.lines[self.state.cursor_line - 1]
            current = self.state.lines.pop(self.state.cursor_line)
            self.state.cursor_line -= 1
            self.state.cursor_col = len(previous)
            self.state.lines[self.state.cursor_line] = previous + current
            self._push_kill("\n", prepend=True, action="delete-to-line-start")
            self._notify_change()
            self._update_autocomplete()

    def _delete_to_line_end(self) -> None:
        self.sticky_col = None
        self._preferred_visual_col = None
        self._history_reentry_after_set_text = False
        line = self.state.lines[self.state.cursor_line]
        if self.state.cursor_col < len(line):
            self._push_undo()
            deleted = line[self.state.cursor_col :]
            self.state.lines[self.state.cursor_line] = line[: self.state.cursor_col]
            self._push_kill(deleted, prepend=False, action="delete-to-line-end")
            self._notify_change()
            self._update_autocomplete()
        elif self.state.cursor_line < len(self.state.lines) - 1:
            self._push_undo()
            next_line = self.state.lines.pop(self.state.cursor_line + 1)
            self.state.lines[self.state.cursor_line] = line + next_line
            self._push_kill("\n", prepend=False, action="delete-to-line-end")
            self._notify_change()
            self._update_autocomplete()

    def _delete_word_backward(self) -> None:
        self.sticky_col = None
        self._preferred_visual_col = None
        self._history_reentry_after_set_text = False
        line = self.state.lines[self.state.cursor_line]
        if self.state.cursor_col == 0:
            if self.state.cursor_line == 0:
                return
            self._push_undo()
            deleted = "\n"
            previous = self.state.lines[self.state.cursor_line - 1]
            current = self.state.lines.pop(self.state.cursor_line)
            self.state.cursor_line -= 1
            self.state.cursor_col = len(previous)
            self.state.lines[self.state.cursor_line] = previous + current
            self._push_kill(deleted, prepend=True, action="delete-word-backward")
            self._notify_change()
            self._update_autocomplete()
            return

        start = self.state.cursor_col
        while start > 0 and line[start - 1].isspace():
            start -= 1
        marker = self._marker_ending_at(line, start)
        if marker is not None:
            start = marker.start()
        elif start > 0 and is_punctuation_char(line[start - 1]):
            while start > 0 and is_punctuation_char(line[start - 1]):
                start -= 1
        else:
            while start > 0 and not line[start - 1].isspace() and not is_punctuation_char(line[start - 1]):
                start -= 1
        if start == self.state.cursor_col:
            return
        self._push_undo()
        deleted = line[start : self.state.cursor_col]
        self.state.lines[self.state.cursor_line] = line[:start] + line[self.state.cursor_col :]
        self.state.cursor_col = start
        self._push_kill(deleted, prepend=True, action="delete-word-backward")
        self._notify_change()
        self._update_autocomplete()

    def _delete_word_forward(self) -> None:
        self.sticky_col = None
        self._preferred_visual_col = None
        self._history_reentry_after_set_text = False
        line = self.state.lines[self.state.cursor_line]
        if self.state.cursor_col == len(line):
            if self.state.cursor_line >= len(self.state.lines) - 1:
                return
            self._push_undo()
            deleted = "\n"
            next_line = self.state.lines.pop(self.state.cursor_line + 1)
            self.state.lines[self.state.cursor_line] = line + next_line
            self._push_kill(deleted, prepend=False, action="delete-word-forward")
            self._notify_change()
            self._update_autocomplete()
            return

        end = self.state.cursor_col
        while end < len(line) and line[end].isspace():
            end += 1
        while end < len(line) and not line[end].isspace():
            end += 1
        if end == self.state.cursor_col:
            return
        self._push_undo()
        deleted = line[self.state.cursor_col : end]
        self.state.lines[self.state.cursor_line] = line[: self.state.cursor_col] + line[end:]
        self._push_kill(deleted, prepend=False, action="delete-word-forward")
        self._notify_change()
        self._update_autocomplete()

    def _yank(self) -> None:
        text = self.kill_ring.peek()
        if text:
            start_line = self.state.cursor_line
            start_col = self.state.cursor_col
            self.insert_text_at_cursor(text)
            self._last_yank_range = (start_line, start_col, self.state.cursor_line, self.state.cursor_col)
            self.last_action = "yank"

    def _yank_pop(self) -> None:
        if self.last_action == "yank" and self.kill_ring.length > 1 and self._last_yank_range is not None:
            start_line, start_col, end_line, end_col = self._last_yank_range
            self._replace_range(start_line, start_col, end_line, end_col, "")
            self.kill_ring.rotate()
            text = self.kill_ring.peek() or ""
            self.state.cursor_line = start_line
            self.state.cursor_col = start_col
            self.insert_text_at_cursor(text)
            self._last_yank_range = (start_line, start_col, self.state.cursor_line, self.state.cursor_col)
            self.last_action = "yank"

    def _new_line(self, skip_undo: bool = False) -> None:
        self._history_reentry_after_set_text = False
        self.history_index = -1
        self.sticky_col = None
        self._preferred_visual_col = None
        self._last_yank_range = None
        if not skip_undo:
            self._push_undo()
        line = self.state.lines[self.state.cursor_line]
        before = line[: self.state.cursor_col]
        after = line[self.state.cursor_col :]
        self.state.lines[self.state.cursor_line] = before
        self.state.lines.insert(self.state.cursor_line + 1, after)
        self.state.cursor_line += 1
        self.state.cursor_col = 0
        self._notify_change()
        self.last_action = None
        self._update_autocomplete()

    def _insert_typed_text(self, text: str) -> None:
        self._history_reentry_after_set_text = False
        self.history_index = -1
        self.sticky_col = None
        self._preferred_visual_col = None
        self._last_yank_range = None
        self._hide_autocomplete_if_needed()
        action = self._typing_action_for_text(text)
        if self._should_push_undo_for_typed_action(action):
            self._push_undo()
        line = self.state.lines[self.state.cursor_line]
        before = line[: self.state.cursor_col]
        after = line[self.state.cursor_col :]
        self.state.lines[self.state.cursor_line] = before + text + after
        self.state.cursor_col += len(text)
        self._notify_change()
        if action == "type-word" and self.last_action == "type-space":
            self.last_action = "type-space-word"
        else:
            self.last_action = action

    def _typing_action_for_text(self, text: str) -> str:
        if len(text) != 1:
            return "type-other"
        if text.isspace():
            return "type-space"
        if is_punctuation_char(text):
            return "type-punct"
        return "type-word"

    def _should_push_undo_for_typed_action(self, action: str) -> bool:
        if self.last_action is None:
            return True
        if action == "type-word":
            return self.last_action not in {"type-word", "type-space", "type-space-word"}
        if action == "type-space":
            return True
        return self.last_action != action

    def _move_word_left(self) -> None:
        self.last_action = None
        self.sticky_col = None
        self._preferred_visual_col = None
        line = self.state.lines[self.state.cursor_line]
        col = self.state.cursor_col
        while col > 0 and line[col - 1].isspace():
            col -= 1
        marker = self._marker_ending_at(line, col)
        if marker is not None:
            col = marker.start()
        elif col > 0 and is_punctuation_char(line[col - 1]):
            while col > 0 and is_punctuation_char(line[col - 1]):
                col -= 1
        else:
            while col > 0 and not line[col - 1].isspace() and not is_punctuation_char(line[col - 1]):
                col -= 1
        self.state.cursor_col = col

    def _move_word_right(self) -> None:
        self.last_action = None
        self.sticky_col = None
        self._preferred_visual_col = None
        line = self.state.lines[self.state.cursor_line]
        col = self.state.cursor_col
        while col < len(line) and line[col].isspace():
            col += 1
        marker = self._marker_starting_at(line, col)
        if marker is not None:
            col = marker.end()
        elif col < len(line) and is_punctuation_char(line[col]):
            while col < len(line) and is_punctuation_char(line[col]):
                col += 1
        else:
            while col < len(line) and not line[col].isspace() and not is_punctuation_char(line[col]):
                col += 1
        self.state.cursor_col = col

    def _insert_paste(self, text: str) -> None:
        self._history_reentry_after_set_text = False
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        current_line = self.state.lines[self.state.cursor_line] if self.state.lines else ""
        if normalized and normalized[0] in "/~.":
            char_before_cursor = current_line[self.state.cursor_col - 1] if self.state.cursor_col > 0 else ""
            if char_before_cursor and (char_before_cursor.isalnum() or char_before_cursor == "_"):
                normalized = " " + normalized
        line_count = normalized.count("\n") + 1 if normalized else 1
        total_chars = len(normalized)
        if line_count > 10 or total_chars > 1000:
            self.paste_counter += 1
            if line_count > 10:
                marker = f"[paste #{self.paste_counter} +{line_count} lines]"
            else:
                marker = f"[paste #{self.paste_counter} {total_chars} chars]"
            self.pastes[self.paste_counter] = normalized
            self.insert_text_at_cursor(marker)
            return
        self.insert_text_at_cursor(normalized)

    def _push_kill(self, text: str, *, prepend: bool, action: str) -> None:
        self.kill_ring.push(text, prepend=prepend, accumulate=self.last_action == action)
        self.last_action = action
        self._last_yank_range = None

    def _replace_range(self, start_line: int, start_col: int, end_line: int, end_col: int, replacement: str) -> None:
        before = self.state.lines[start_line][:start_col]
        after = self.state.lines[end_line][end_col:]
        replacement_lines = replacement.split("\n")
        new_lines = self.state.lines[:start_line]
        if len(replacement_lines) == 1:
            new_lines.append(before + replacement_lines[0] + after)
        else:
            new_lines.append(before + replacement_lines[0])
            new_lines.extend(replacement_lines[1:-1])
            new_lines.append(replacement_lines[-1] + after)
        new_lines.extend(self.state.lines[end_line + 1 :])
        self.state.lines = new_lines or [""]

    def _expand_text(self, text: str) -> str:
        def replace(match: re.Match[str]) -> str:
            paste_id = int(match.group("id"))
            return self.pastes.get(paste_id, match.group(0))

        return self._PASTE_MARKER_RE.sub(replace, text)

    def _marker_starting_at(self, line: str, col: int) -> re.Match[str] | None:
        match = self._PASTE_MARKER_RE.match(line, col)
        if match is None:
            return None
        paste_id = int(match.group("id"))
        return match if paste_id in self.pastes else None

    def _marker_ending_at(self, line: str, col: int) -> re.Match[str] | None:
        for match in self._PASTE_MARKER_RE.finditer(line):
            if match.end() == col and int(match.group("id")) in self.pastes:
                return match
        return None

    def _segment_with_markers(self, text: str) -> list[tuple[str, int, int]]:
        if not self.pastes or "[paste #" not in text:
            return [(match.group(0), match.start(), match.end()) for match in get_segmenter().finditer(text)]

        markers: list[tuple[int, int]] = []
        for match in self._PASTE_MARKER_RE.finditer(text):
            paste_id = int(match.group("id"))
            if paste_id in self.pastes:
                markers.append((match.start(), match.end()))
        if not markers:
            return [(match.group(0), match.start(), match.end()) for match in get_segmenter().finditer(text)]

        result: list[tuple[str, int, int]] = []
        marker_idx = 0
        for seg in get_segmenter().finditer(text):
            while marker_idx < len(markers) and markers[marker_idx][1] <= seg.start():
                marker_idx += 1
            marker = markers[marker_idx] if marker_idx < len(markers) else None
            if marker and marker[0] <= seg.start() < marker[1]:
                if seg.start() == marker[0]:
                    result.append((text[marker[0] : marker[1]], marker[0], marker[1]))
            else:
                result.append((seg.group(0), seg.start(), seg.end()))
        return result

    def _word_wrap_line(
        self,
        line: str,
        max_width: int,
        segments: list[tuple[str, int, int]] | None = None,
    ) -> list[tuple[str, int, int]]:
        if not line or max_width <= 0:
            return [("", 0, 0)]
        if visible_width(line) <= max_width:
            return [(line, 0, len(line))]

        chunks: list[tuple[str, int, int]] = []
        segs = segments or self._segment_with_markers(line)
        current_width = 0
        chunk_start = 0
        wrap_opp_index = -1
        wrap_opp_width = 0

        for idx, (grapheme, start, end) in enumerate(segs):
            g_width = visible_width(grapheme)
            is_ws = not self._marker_starting_at(line, start) and is_whitespace_char(grapheme)

            if current_width + g_width > max_width:
                if wrap_opp_index >= 0 and current_width - wrap_opp_width + g_width <= max_width:
                    chunks.append((line[chunk_start:wrap_opp_index], chunk_start, wrap_opp_index))
                    chunk_start = wrap_opp_index
                    current_width -= wrap_opp_width
                elif chunk_start < start:
                    chunks.append((line[chunk_start:start], chunk_start, start))
                    chunk_start = start
                    current_width = 0
                wrap_opp_index = -1

            if g_width > max_width:
                sub_segments = [(match.group(0), match.start(), match.end()) for match in get_segmenter().finditer(grapheme)]
                sub_chunks = self._word_wrap_line(grapheme, max_width, sub_segments)
                for sub_text, sub_start, sub_end in sub_chunks[:-1]:
                    chunks.append((sub_text, start + sub_start, start + sub_end))
                last_text, last_start, _last_end = sub_chunks[-1]
                chunk_start = start + last_start
                current_width = visible_width(last_text)
                wrap_opp_index = -1
                continue

            current_width += g_width
            next_seg = segs[idx + 1] if idx + 1 < len(segs) else None
            if is_ws and next_seg and (self._marker_starting_at(line, next_seg[1]) is not None or not is_whitespace_char(next_seg[0])):
                wrap_opp_index = next_seg[1]
                wrap_opp_width = current_width

        chunks.append((line[chunk_start:], chunk_start, len(line)))
        return chunks

    def _cursor_chunk_position(self, line: str, width: int, cursor_col: int) -> tuple[int, int]:
        chunks = self._visual_chunks_for_line(line, width)
        for row, (_text, start, end) in enumerate(chunks):
            is_last = row == len(chunks) - 1
            if (is_last and cursor_col >= start) or (start <= cursor_col < end) or (
                not is_last and cursor_col == end and end > start
            ):
                return row, visible_width(line[start:cursor_col])
        last_text, _start, _end = chunks[-1]
        return len(chunks) - 1, visible_width(last_text)

    def _cursor_col_from_visual(self, line: str, width: int, row: int, preferred_col: int) -> int:
        chunks = self._visual_chunks_for_line(line, width)
        if not chunks:
            return 0
        row = max(0, min(row, len(chunks) - 1))
        _text, start, end = chunks[row]
        if preferred_col <= 0:
            return start
        col = start
        for grapheme, seg_start, seg_end in self._segment_with_markers(line[start:end]):
            _ = seg_start
            g_width = visible_width(grapheme)
            if visible_width(line[start:col]) + g_width > preferred_col:
                break
            col = start + (seg_end)
        return min(col, end)

    def _visual_chunks_for_line(self, line: str, width: int) -> list[tuple[str, int, int]]:
        return self._word_wrap_line(line, max(1, width), self._segment_with_markers(line))

    def _max_visual_col_for_chunk(self, chunks: list[tuple[str, int, int]], row: int) -> int:
        row = max(0, min(row, len(chunks) - 1))
        text, _start, _end = chunks[row]
        chunk_width = visible_width(text)
        is_last = row == len(chunks) - 1
        return chunk_width if is_last else max(0, chunk_width - 1)

    def _compute_vertical_move_column(self, current_visual_col: int, source_max_visual_col: int, target_max_visual_col: int) -> int:
        has_preferred = self._preferred_visual_col is not None
        cursor_in_middle = current_visual_col < source_max_visual_col
        target_too_short = target_max_visual_col < current_visual_col

        if not has_preferred or cursor_in_middle:
            if target_too_short:
                self._preferred_visual_col = current_visual_col
                return target_max_visual_col
            self._preferred_visual_col = None
            return current_visual_col

        preferred = self._preferred_visual_col if self._preferred_visual_col is not None else current_visual_col
        target_cant_fit_preferred = target_max_visual_col < preferred
        if target_too_short or target_cant_fit_preferred:
            return target_max_visual_col

        self._preferred_visual_col = None
        return preferred

    def _hide_autocomplete(self) -> None:
        self.autocomplete_state = None
        self.autocomplete_items = []
        self.autocomplete_index = 0
        self.autocomplete_prefix = ""
        self.autocomplete_force_mode = False

    def _hide_autocomplete_if_needed(self) -> None:
        if self.autocomplete_force_mode:
            return
        self._hide_autocomplete()

    def is_showing_autocomplete(self) -> bool:
        return bool(self.autocomplete_state and self.autocomplete_items)

    def _update_autocomplete(self) -> None:
        if self.autocomplete_provider is None:
            self._hide_autocomplete()
            return
        current_prefix = self.state.lines[self.state.cursor_line][: self.state.cursor_col]
        should_show = self.autocomplete_force_mode or self.is_showing_autocomplete() or current_prefix.startswith("/")
        if not should_show:
            self._hide_autocomplete()
            return
        result = self.autocomplete_provider.get_suggestions(
            self.state.lines,
            self.state.cursor_line,
            self.state.cursor_col,
        )
        if not result:
            self._hide_autocomplete()
            return
        self.autocomplete_state = "open"
        self.autocomplete_items = list(result["items"])
        self.autocomplete_index = 0
        self.autocomplete_prefix = result["prefix"]

    def _apply_autocomplete_item(self, item: AutocompleteItem, prefix: str) -> None:
        if self.autocomplete_provider is None:
            return
        accepted_value = item.value
        self._push_undo()
        result = self.autocomplete_provider.apply_completion(
            self.state.lines,
            self.state.cursor_line,
            self.state.cursor_col,
            item,
            prefix,
        )
        self.state.lines = list(result["lines"])
        self.state.cursor_line = int(result["cursor_line"])
        self.state.cursor_col = int(result["cursor_col"])
        self.sticky_col = None
        self.history_index = -1
        self.last_action = None
        self._notify_change()
        self._hide_autocomplete()
        self._update_autocomplete()
        if self.is_showing_autocomplete() and self.autocomplete_prefix == accepted_value:
            self._hide_autocomplete()

    def _handle_tab(self) -> bool:
        if self.is_showing_autocomplete():
            item = self.autocomplete_items[self.autocomplete_index]
            self._apply_autocomplete_item(item, self.autocomplete_prefix)
            return True
        if self.autocomplete_provider is None:
            return False
        force_getter = getattr(self.autocomplete_provider, "get_force_file_suggestions", None)
        if callable(force_getter):
            result = force_getter(self.state.lines, self.state.cursor_line, self.state.cursor_col)
            self.autocomplete_force_mode = bool(result)
        else:
            result = self.autocomplete_provider.get_suggestions(
                self.state.lines,
                self.state.cursor_line,
                self.state.cursor_col,
            )
            self.autocomplete_force_mode = False
        if not result:
            return False
        items = list(result["items"])
        if self.autocomplete_force_mode and len(items) == 1:
            self._apply_autocomplete_item(items[0], result["prefix"])
            return True
        self.autocomplete_state = "open"
        self.autocomplete_items = items
        self.autocomplete_index = 0
        self.autocomplete_prefix = result["prefix"]
        return True

    def _accept_autocomplete_on_enter(self) -> bool:
        if not self.autocomplete_items:
            return False
        prefix = self.autocomplete_prefix
        exact_item = next((item for item in self.autocomplete_items if item.value == prefix), None)
        if exact_item is not None:
            self._hide_autocomplete()
            return False
        prefix_matches = [item for item in self.autocomplete_items if item.value.startswith(prefix)]
        if prefix_matches:
            self._apply_autocomplete_item(prefix_matches[0], prefix)
            return True
        self._apply_autocomplete_item(self.autocomplete_items[self.autocomplete_index], prefix)
        return True

    def _perform_jump(self, target: str, direction: str) -> None:
        if len(target) != 1:
            return
        self.last_action = None
        self.sticky_col = None
        self._preferred_visual_col = None
        if direction == "forward":
            for line_index in range(self.state.cursor_line, len(self.state.lines)):
                line = self.state.lines[line_index]
                start = self.state.cursor_col + 1 if line_index == self.state.cursor_line else 0
                found = line.find(target, start)
                if found != -1:
                    self.state.cursor_line = line_index
                    self.state.cursor_col = found
                    return
        else:
            for line_index in range(self.state.cursor_line, -1, -1):
                line = self.state.lines[line_index]
                end = self.state.cursor_col - 1 if line_index == self.state.cursor_line else len(line) - 1
                if end < 0:
                    continue
                found = line.rfind(target, 0, end + 1)
                if found != -1:
                    self.state.cursor_line = line_index
                    self.state.cursor_col = found
                    return

    def get_cursor(self) -> dict[str, int]:
        return {"line": self.state.cursor_line, "col": self.state.cursor_col}

    def _previous_grapheme_boundary(self, text: str, col: int) -> int:
        boundary = 0
        for match in get_segmenter().finditer(text):
            if match.end() >= col:
                break
            boundary = match.end()
        return boundary

    def _next_grapheme_boundary(self, text: str, col: int) -> int:
        for match in get_segmenter().finditer(text):
            if match.end() > col:
                return match.end()
        return len(text)

    def get_lines(self) -> list[str]:
        return list(self.state.lines)

    def invalidate(self) -> None:
        return None

    def render(self, width: int) -> list[str]:
        content_width = max(1, width - self.options.padding_x * 2)
        self._layout_width = max(1, content_width - (0 if self.options.padding_x else 1))
        result: list[str] = []
        for i, line in enumerate(self.state.lines):
            if self.focused and i == self.state.cursor_line:
                result.extend(self._wrap_line_with_cursor(line, self._layout_width, min(self.state.cursor_col, len(line))))
                continue
            result.extend(self._wrap_rendered_line(line, self._layout_width))
        return result or [""]

    def _visual_col_for_cursor(self) -> int:
        line = self.state.lines[self.state.cursor_line]
        _row, col = self._cursor_chunk_position(line, self._layout_width, self.state.cursor_col)
        return col

    def _visual_row_for_cursor(self, line: str) -> int:
        row, _col = self._cursor_chunk_position(line, self._layout_width, self.state.cursor_col)
        return row

    def _visual_row_count(self, line: str) -> int:
        return max(1, len(self._word_wrap_line(line, self._layout_width, self._segment_with_markers(line))))

    def _move_visual_up(self) -> None:
        width = max(1, self._layout_width)
        line = self.state.lines[self.state.cursor_line]
        chunks = self._visual_chunks_for_line(line, width)
        visual_row = self._visual_row_for_cursor(line)
        current_visual_col = self._visual_col_for_cursor()
        source_max_visual_col = self._max_visual_col_for_chunk(chunks, visual_row)
        if visual_row > 0:
            target_chunks = chunks
            target_row = visual_row - 1
            target_max_visual_col = self._max_visual_col_for_chunk(target_chunks, target_row)
            move_to_visual_col = self._compute_vertical_move_column(
                current_visual_col, source_max_visual_col, target_max_visual_col
            )
            self.state.cursor_col = self._cursor_col_from_visual(line, width, target_row, move_to_visual_col)
            return
        if self.state.cursor_line == 0:
            return
        self.state.cursor_line -= 1
        previous = self.state.lines[self.state.cursor_line]
        previous_chunks = self._visual_chunks_for_line(previous, width)
        target_row = len(previous_chunks) - 1
        target_max_visual_col = self._max_visual_col_for_chunk(previous_chunks, target_row)
        move_to_visual_col = self._compute_vertical_move_column(
            current_visual_col, source_max_visual_col, target_max_visual_col
        )
        self.state.cursor_col = self._cursor_col_from_visual(previous, width, target_row, move_to_visual_col)

    def _move_visual_down(self) -> None:
        width = max(1, self._layout_width)
        line = self.state.lines[self.state.cursor_line]
        chunks = self._visual_chunks_for_line(line, width)
        visual_row = self._visual_row_for_cursor(line)
        current_visual_col = self._visual_col_for_cursor()
        source_max_visual_col = self._max_visual_col_for_chunk(chunks, visual_row)
        total_rows = len(chunks)
        if visual_row < total_rows - 1:
            target_row = visual_row + 1
            target_max_visual_col = self._max_visual_col_for_chunk(chunks, target_row)
            move_to_visual_col = self._compute_vertical_move_column(
                current_visual_col, source_max_visual_col, target_max_visual_col
            )
            self.state.cursor_col = self._cursor_col_from_visual(line, width, target_row, move_to_visual_col)
            return
        if self.state.cursor_line >= len(self.state.lines) - 1:
            return
        self.state.cursor_line += 1
        next_line = self.state.lines[self.state.cursor_line]
        next_chunks = self._visual_chunks_for_line(next_line, width)
        target_max_visual_col = self._max_visual_col_for_chunk(next_chunks, 0)
        move_to_visual_col = self._compute_vertical_move_column(
            current_visual_col, source_max_visual_col, target_max_visual_col
        )
        self.state.cursor_col = self._cursor_col_from_visual(next_line, width, 0, move_to_visual_col)

    def _wrap_rendered_line(self, rendered: str, width: int) -> list[str]:
        if width <= 0:
            return [""]
        if "\x1b" not in rendered and CURSOR_MARKER not in rendered:
            wrapped = []
            for chunk_text, _start, _end in self._word_wrap_line(rendered, width, self._segment_with_markers(rendered)):
                wrapped.append(chunk_text + " " * max(0, width - visible_width(chunk_text)))
            return wrapped or [" " * width]
        line_width = max(visible_width(rendered), 1)
        wrapped: list[str] = []
        start = 0
        while start < line_width:
            clipped = slice_by_column(rendered, start, start + width, True)
            wrapped.append(clipped + " " * max(0, width - visible_width(clipped)))
            start += width
        return wrapped or [" " * width]

    def _wrap_line_with_cursor(self, line: str, width: int, cursor_col: int) -> list[str]:
        if width <= 0:
            return [""]
        segments = self._segment_with_markers(line)
        chunks = self._word_wrap_line(line, width, segments)
        wrapped: list[str] = []

        for chunk_text, start, end in chunks:
            chunk_width = visible_width(chunk_text)
            is_last = end == len(line)
            if start <= cursor_col < end or (not is_last and cursor_col == end and end > start):
                local_segments = self._segment_with_markers(chunk_text)
                chunk_rendered = ""
                for text, seg_start, _seg_end in local_segments:
                    absolute_start = start + seg_start
                    if absolute_start == cursor_col:
                        chunk_rendered += f"{CURSOR_MARKER}\x1b[7m{text}\x1b[27m"
                    else:
                        chunk_rendered += text
                wrapped.append(chunk_rendered + " " * max(0, width - chunk_width))
            else:
                wrapped.append(chunk_text + " " * max(0, width - chunk_width))

        if cursor_col == len(line):
            if wrapped:
                last_text, _start, _end = chunks[-1]
                current_width = visible_width(last_text)
                if current_width < width:
                    wrapped[-1] = last_text + f"{CURSOR_MARKER}\x1b[7m \x1b[27m" + " " * max(0, width - current_width - 1)
                else:
                    wrapped.append(f"{CURSOR_MARKER}\x1b[7m \x1b[27m" + " " * max(0, width - 1))
            else:
                wrapped.append(f"{CURSOR_MARKER}\x1b[7m \x1b[27m" + " " * max(0, width - 1))

        return wrapped or [" " * width]
