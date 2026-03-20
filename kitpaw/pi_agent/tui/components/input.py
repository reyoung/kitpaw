from __future__ import annotations

from dataclasses import dataclass

from ..keybindings import get_editor_keybindings
from ..keys import decode_kitty_printable
from ..kill_ring import KillRing
from ..tui import CURSOR_MARKER
from ..undo_stack import UndoStack
from ..utils import get_segmenter, is_punctuation_char, is_whitespace_char, slice_by_column, visible_width


@dataclass(slots=True)
class _InputState:
    value: str
    cursor: int


class Input:
    def __init__(self) -> None:
        self.value = ""
        self.cursor = 0
        self.on_submit = None
        self.on_escape = None
        self.focused = False
        self.paste_buffer = ""
        self.is_in_paste = False
        self.kill_ring = KillRing()
        self.last_action: str | None = None
        self.last_yank_range: tuple[int, int] | None = None
        self.undo_stack: UndoStack[_InputState] = UndoStack()

    def get_value(self) -> str:
        return self.value

    def set_value(self, value: str) -> None:
        self.value = value
        self.cursor = min(self.cursor, len(value))
        self.last_action = None
        self.last_yank_range = None

    def handle_input(self, data: str) -> None:
        if "\x1b[200~" in data:
            self.is_in_paste = True
            self.paste_buffer = ""
            data = data.replace("\x1b[200~", "")
        if self.is_in_paste:
            self.paste_buffer += data
            end_index = self.paste_buffer.find("\x1b[201~")
            if end_index != -1:
                self._handle_paste(self.paste_buffer[:end_index])
                self.is_in_paste = False
                remaining = self.paste_buffer[end_index + 6 :]
                self.paste_buffer = ""
                if remaining:
                    self.handle_input(remaining)
            return
        kb = get_editor_keybindings()
        if kb.matches(data, "selectCancel"):
            if self.on_escape:
                self.on_escape()
            return
        if kb.matches(data, "undo"):
            self.undo()
            return
        if kb.matches(data, "submit") or data == "\n":
            if self.on_submit:
                self.on_submit(self.value)
            return
        if kb.matches(data, "deleteCharBackward"):
            self._handle_backspace()
            return
        if kb.matches(data, "deleteCharForward"):
            self._handle_forward_delete()
            return
        if kb.matches(data, "deleteWordBackward"):
            self._delete_word_backwards()
            return
        if kb.matches(data, "deleteWordForward"):
            self._delete_word_forward()
            return
        if kb.matches(data, "deleteToLineStart"):
            self._delete_to_line_start()
            return
        if kb.matches(data, "deleteToLineEnd"):
            self._delete_to_line_end()
            return
        if kb.matches(data, "yank"):
            self._yank()
            return
        if kb.matches(data, "yankPop"):
            self._yank_pop()
            return
        if kb.matches(data, "cursorLeft"):
            self.last_action = None
            if self.cursor > 0:
                self.cursor -= self._last_grapheme_length(self.value[: self.cursor])
            return
        if kb.matches(data, "cursorRight"):
            self.last_action = None
            if self.cursor < len(self.value):
                self.cursor += self._first_grapheme_length(self.value[self.cursor :])
            return
        if kb.matches(data, "cursorLineStart"):
            self.last_action = None
            self.cursor = 0
            return
        if kb.matches(data, "cursorLineEnd"):
            self.last_action = None
            self.cursor = len(self.value)
            return
        if kb.matches(data, "cursorWordLeft"):
            self._move_word_backwards()
            return
        if kb.matches(data, "cursorWordRight"):
            self._move_word_forwards()
            return
        kitty_printable = decode_kitty_printable(data)
        if kitty_printable is not None:
            self._insert_character(kitty_printable)
            return
        if not any(ord(ch) < 32 or ord(ch) == 127 for ch in data):
            self._insert_character(data)

    def _push_undo(self) -> None:
        self.undo_stack.push(_InputState(value=self.value, cursor=self.cursor))

    def undo(self) -> None:
        state = self.undo_stack.pop()
        if state:
            self.value = state.value
            self.cursor = state.cursor
            self.last_action = None
            self.last_yank_range = None

    def _insert_character(self, char: str) -> None:
        self.last_yank_range = None
        if is_whitespace_char(char) or self.last_action != "type-word":
            self._push_undo()
        self.last_action = "type-word"
        self.value = self.value[: self.cursor] + char + self.value[self.cursor :]
        self.cursor += len(char)

    def _handle_backspace(self) -> None:
        self.last_action = None
        self.last_yank_range = None
        if self.cursor > 0:
            self._push_undo()
            grapheme_length = self._last_grapheme_length(self.value[: self.cursor])
            self.value = self.value[: self.cursor - grapheme_length] + self.value[self.cursor :]
            self.cursor -= grapheme_length

    def _handle_forward_delete(self) -> None:
        self.last_action = None
        self.last_yank_range = None
        if self.cursor < len(self.value):
            self._push_undo()
            grapheme_length = self._first_grapheme_length(self.value[self.cursor :])
            self.value = self.value[: self.cursor] + self.value[self.cursor + grapheme_length :]

    def _delete_to_line_start(self) -> None:
        self.last_yank_range = None
        if self.cursor > 0:
            self._push_undo()
            deleted = self.value[: self.cursor]
            self._push_kill(deleted, prepend=True)
            self.value = self.value[self.cursor :]
            self.cursor = 0

    def _delete_to_line_end(self) -> None:
        self.last_yank_range = None
        if self.cursor < len(self.value):
            self._push_undo()
            deleted = self.value[self.cursor :]
            self._push_kill(deleted, prepend=False)
            self.value = self.value[: self.cursor]

    def _delete_word_backwards(self) -> None:
        self.last_yank_range = None
        if self.cursor <= 0:
            return
        was_kill = self.last_action == "kill"
        old_cursor = self.cursor
        self._move_word_backwards()
        start = self.cursor
        self.cursor = old_cursor
        self._push_undo()
        deleted = self.value[start : self.cursor]
        self.kill_ring.push(deleted, prepend=True, accumulate=was_kill)
        self.last_action = "kill"
        self.last_yank_range = None
        self.value = self.value[:start] + self.value[self.cursor :]
        self.cursor = start

    def _delete_word_forward(self) -> None:
        self.last_yank_range = None
        if self.cursor >= len(self.value):
            return
        was_kill = self.last_action == "kill"
        old_cursor = self.cursor
        self._move_word_forwards()
        end = self.cursor
        self.cursor = old_cursor
        self._push_undo()
        deleted = self.value[self.cursor:end]
        self.kill_ring.push(deleted, prepend=False, accumulate=was_kill)
        self.last_action = "kill"
        self.last_yank_range = None
        self.value = self.value[: self.cursor] + self.value[end:]

    def _yank(self) -> None:
        text = self.kill_ring.peek()
        if not text:
            return
        self._push_undo()
        start = self.cursor
        self.value = self.value[: self.cursor] + text + self.value[self.cursor :]
        self.cursor += len(text)
        self.last_action = "yank"
        self.last_yank_range = (start, self.cursor)

    def _yank_pop(self) -> None:
        if self.last_action != "yank" or self.kill_ring.length <= 1 or self.last_yank_range is None:
            return
        self.kill_ring.rotate()
        text = self.kill_ring.peek() or ""
        self._push_undo()
        start, end = self.last_yank_range
        self.value = self.value[:start] + text + self.value[end:]
        self.cursor = start + len(text)
        self.last_yank_range = (start, self.cursor)

    def _handle_paste(self, content: str) -> None:
        self.last_action = None
        self.last_yank_range = None
        self._push_undo()
        clean = content.replace("\r\n", "").replace("\r", "").replace("\n", "").replace("\t", "    ")
        self.value = self.value[: self.cursor] + clean + self.value[self.cursor :]
        self.cursor += len(clean)

    def _push_kill(self, text: str, *, prepend: bool) -> None:
        self.kill_ring.push(text, prepend=prepend, accumulate=self.last_action == "kill")
        self.last_action = "kill"
        self.last_yank_range = None

    def _last_grapheme_length(self, text: str) -> int:
        length = 1
        for match in get_segmenter().finditer(text):
            length = len(match.group(0))
        return length

    def _first_grapheme_length(self, text: str) -> int:
        match = get_segmenter().search(text)
        return len(match.group(0)) if match else 1

    def _move_word_backwards(self) -> None:
        if self.cursor <= 0:
            return
        self.last_action = None
        graphemes = [match.group(0) for match in get_segmenter().finditer(self.value[: self.cursor])]
        while graphemes and is_whitespace_char(graphemes[-1]):
            self.cursor -= len(graphemes.pop())
        if not graphemes:
            return
        if is_punctuation_char(graphemes[-1]):
            while graphemes and is_punctuation_char(graphemes[-1]):
                self.cursor -= len(graphemes.pop())
            return
        while graphemes and not is_whitespace_char(graphemes[-1]) and not is_punctuation_char(graphemes[-1]):
            self.cursor -= len(graphemes.pop())

    def _move_word_forwards(self) -> None:
        if self.cursor >= len(self.value):
            return
        self.last_action = None
        graphemes = [match.group(0) for match in get_segmenter().finditer(self.value[self.cursor :])]
        index = 0
        while index < len(graphemes) and is_whitespace_char(graphemes[index]):
            self.cursor += len(graphemes[index])
            index += 1
        if index >= len(graphemes):
            return
        if is_punctuation_char(graphemes[index]):
            while index < len(graphemes) and is_punctuation_char(graphemes[index]):
                self.cursor += len(graphemes[index])
                index += 1
            return
        while index < len(graphemes) and not is_whitespace_char(graphemes[index]) and not is_punctuation_char(graphemes[index]):
            self.cursor += len(graphemes[index])
            index += 1

    def invalidate(self) -> None:
        return None

    def render(self, width: int) -> list[str]:
        prompt = "> "
        available_width = width - len(prompt)
        if available_width <= 0:
            return [prompt]

        visible_text = self.value
        cursor_display = self.cursor
        total_width = visible_width(self.value)

        if total_width >= available_width:
            scroll_width = available_width - 1 if self.cursor == len(self.value) else available_width
            cursor_col = visible_width(self.value[: self.cursor])
            if scroll_width > 0:
                half_width = scroll_width // 2
                if cursor_col < half_width:
                    start_col = 0
                elif cursor_col > total_width - half_width:
                    start_col = max(0, total_width - scroll_width)
                else:
                    start_col = max(0, cursor_col - half_width)
                visible_text = slice_by_column(self.value, start_col, scroll_width, True)
                before_cursor = slice_by_column(self.value, start_col, max(0, cursor_col - start_col), True)
                cursor_display = len(before_cursor)
            else:
                visible_text = ""
                cursor_display = 0

        before = visible_text[:cursor_display]
        at = visible_text[cursor_display : cursor_display + 1] or " "
        after = visible_text[cursor_display + len(at) :]
        cursor_marker = CURSOR_MARKER if self.focused else ""
        line = f"{prompt}{before}{cursor_marker}\x1b[7m{at}\x1b[27m{after}"
        return [line + " " * max(0, width - visible_width(line))]
