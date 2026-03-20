from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import Callable, Protocol, runtime_checkable

from .keys import is_key_release, matches_key
from .terminal import Terminal
from .terminal_image import CellDimensions, get_capabilities, is_image_line, set_cell_dimensions
from .utils import extract_segments, slice_by_column, slice_with_width, visible_width


@runtime_checkable
class Component(Protocol):
    def render(self, width: int) -> list[str]: ...

    def handle_input(self, data: str) -> None: ...

    def invalidate(self) -> None: ...


@runtime_checkable
class Focusable(Protocol):
    focused: bool


CURSOR_MARKER = "\x1b_pi:c\x07"


def is_focusable(component: Component | None) -> bool:
    return component is not None and hasattr(component, "focused")


OverlayAnchor = str


@dataclass(slots=True)
class OverlayMargin:
    top: int | None = None
    right: int | None = None
    bottom: int | None = None
    left: int | None = None


SizeValue = int | str


@dataclass(slots=True)
class OverlayOptions:
    width: SizeValue | None = None
    min_width: int | None = None
    max_height: SizeValue | None = None
    anchor: OverlayAnchor | None = None
    offset_x: int | None = None
    offset_y: int | None = None
    row: SizeValue | None = None
    col: SizeValue | None = None
    margin: OverlayMargin | int | None = None
    visible: Callable[[int, int], bool] | None = None
    non_capturing: bool = False


@dataclass(slots=True)
class OverlayHandle:
    hide: Callable[[], None]
    set_hidden: Callable[[bool], None]
    is_hidden: Callable[[], bool]
    focus: Callable[[], None]
    unfocus: Callable[[], None]
    is_focused: Callable[[], bool]


class Container:
    def __init__(self) -> None:
        self.children: list[Component] = []

    def add_child(self, component: Component) -> None:
        self.children.append(component)

    def remove_child(self, component: Component) -> None:
        if component in self.children:
            self.children.remove(component)

    def clear(self) -> None:
        self.children = []

    def invalidate(self) -> None:
        for child in self.children:
            if hasattr(child, "invalidate"):
                child.invalidate()

    def render(self, width: int) -> list[str]:
        lines: list[str] = []
        for child in self.children:
            lines.extend(child.render(width))
        return lines


class TUI(Container):
    def __init__(self, terminal: Terminal, show_hardware_cursor: bool | None = None) -> None:
        super().__init__()
        self.terminal = terminal
        self.previous_lines: list[str] = []
        self.previous_width = 0
        self.previous_height = 0
        self.focused_component: Component | None = None
        self.input_listeners: set[Callable[[str], object | None]] = set()
        self.on_debug: Callable[[], None] | None = None
        self.render_requested = False
        self._scheduled_render_handle: asyncio.Handle | None = None
        self.cursor_row = 0
        self.hardware_cursor_row = 0
        self.input_buffer = ""
        self.cell_size_query_pending = False
        self.show_hardware_cursor = bool(show_hardware_cursor) if show_hardware_cursor is not None else False
        self.clear_on_shrink = False
        self.max_lines_rendered = 0
        self.previous_viewport_top = 0
        self.full_redraw_count = 0
        self.stopped = False
        self.focus_order_counter = 0
        self.overlay_stack: list[dict[str, object]] = []

    @property
    def full_redraws(self) -> int:
        return self.full_redraw_count

    def set_clear_on_shrink(self, enabled: bool) -> None:
        self.clear_on_shrink = enabled

    def set_focus(self, component: Component | None) -> None:
        if is_focusable(self.focused_component):
            self.focused_component.focused = False  # type: ignore[attr-defined]
        self.focused_component = component
        if is_focusable(component):
            component.focused = True  # type: ignore[attr-defined]

    def show_overlay(self, component: Component, options: OverlayOptions | None = None) -> OverlayHandle:
        entry = {
            "component": component,
            "options": options,
            "pre_focus": self.focused_component,
            "hidden": False,
            "focus_order": self.focus_order_counter + 1,
        }
        self.focus_order_counter += 1
        self.overlay_stack.append(entry)
        if not (options and options.non_capturing) and self._is_overlay_visible(entry):
            self.set_focus(component)
        self.terminal.hide_cursor()
        self.request_render()

        return OverlayHandle(
            hide=lambda: self._hide_entry(entry),
            set_hidden=lambda hidden: self._set_hidden(entry, hidden),
            is_hidden=lambda: bool(entry["hidden"]),
            focus=lambda: self._focus_entry(entry),
            unfocus=lambda: self._unfocus_entry(entry),
            is_focused=lambda: self.focused_component is component,
        )

    def hide_overlay(self) -> None:
        overlay = self.overlay_stack.pop() if self.overlay_stack else None
        if not overlay:
            return
        if self.focused_component is overlay["component"]:
            top_visible = self._get_topmost_visible_overlay()
            self.set_focus(top_visible["component"] if top_visible else overlay["pre_focus"])  # type: ignore[arg-type]
        if not self.overlay_stack:
            self.terminal.hide_cursor()
        self.request_render()

    def has_overlay(self) -> bool:
        return any(self._is_overlay_visible(o) for o in self.overlay_stack)

    def _is_overlay_visible(self, entry: dict[str, object]) -> bool:
        if entry["hidden"]:
            return False
        options = entry["options"]
        if options and getattr(options, "visible", None):
            return bool(options.visible(self.terminal.columns, self.terminal.rows))  # type: ignore[union-attr]
        return True

    def _get_topmost_visible_overlay(self) -> dict[str, object] | None:
        for entry in reversed(self.overlay_stack):
            options = entry["options"]
            if options and getattr(options, "non_capturing", False):
                continue
            if self._is_overlay_visible(entry):
                return entry
        return None

    def invalidate(self) -> None:
        super().invalidate()
        for overlay in self.overlay_stack:
            component = overlay["component"]
            if hasattr(component, "invalidate"):
                component.invalidate()  # type: ignore[union-attr]

    def start(self) -> None:
        self.stopped = False
        self.terminal.start(self._handle_input, lambda: self.request_render())
        self.terminal.hide_cursor()
        self.query_cell_size()
        self.request_render()

    def add_input_listener(self, listener: Callable[[str], object | None]) -> Callable[[], None]:
        self.input_listeners.add(listener)

        def unsubscribe() -> None:
            self.input_listeners.discard(listener)

        return unsubscribe

    def remove_input_listener(self, listener: Callable[[str], object | None]) -> None:
        self.input_listeners.discard(listener)

    def query_cell_size(self) -> None:
        if not get_capabilities().images:
            return
        self.cell_size_query_pending = True
        self.terminal.write("\x1b[16t")

    def stop(self) -> None:
        self.stopped = True
        if self.previous_lines:
            target_row = len(self.previous_lines)
            line_diff = target_row - self.hardware_cursor_row
            if line_diff > 0:
                self.terminal.write(f"\x1b[{line_diff}B")
            elif line_diff < 0:
                self.terminal.write(f"\x1b[{-line_diff}A")
            self.terminal.write("\r\n")
        self.terminal.show_cursor()
        self.terminal.stop()

    def request_render(self, force: bool = False) -> None:
        if force:
            self.previous_lines = []
            self.previous_width = -1
            self.previous_height = -1
            self.cursor_row = 0
            self.hardware_cursor_row = 0
            self.max_lines_rendered = 0
            self.previous_viewport_top = 0
        if self.render_requested:
            return
        self.render_requested = True
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self.render_requested = False
            self.do_render()
            return
        self._scheduled_render_handle = loop.call_soon(self._run_scheduled_render)

    def _run_scheduled_render(self) -> None:
        self._scheduled_render_handle = None
        self.render_requested = False
        self.do_render()

    def _handle_input(self, data: str) -> None:
        current = data
        for listener in list(self.input_listeners):
            result = listener(current)
            if result is True:
                return
            if isinstance(result, str):
                current = result
                continue
            if isinstance(result, dict):
                if result.get("consume"):
                    return
                if "data" in result and isinstance(result["data"], str):
                    current = result["data"]
        if not current:
            return
        data = current
        if self.cell_size_query_pending:
            self.input_buffer += data
            filtered = self._parse_cell_size_response()
            if not filtered:
                return
            data = filtered
        if self.on_debug and matches_key(data, "shift+ctrl+d"):
            self.on_debug()
            return
        focused_overlay = next((o for o in self.overlay_stack if o["component"] is self.focused_component), None)
        if focused_overlay and not self._is_overlay_visible(focused_overlay):
            top_visible = self._get_topmost_visible_overlay()
            self.set_focus(top_visible["component"] if top_visible else focused_overlay["pre_focus"])  # type: ignore[arg-type]
        if self.focused_component and hasattr(self.focused_component, "handle_input"):
            if is_key_release(data) and not bool(getattr(self.focused_component, "wantsKeyRelease", False)):
                return
            self.focused_component.handle_input(data)  # type: ignore[union-attr]
            self.request_render()

    def _parse_cell_size_response(self) -> str:
        response_pattern = re.compile(r"\x1b\[6;(\d+);(\d+)t")
        match = response_pattern.search(self.input_buffer)
        if match:
            height_px = int(match.group(1))
            width_px = int(match.group(2))
            if height_px > 0 and width_px > 0:
                set_cell_dimensions(CellDimensions(width_px=width_px, height_px=height_px))
                self.invalidate()
                self.request_render()
            self.input_buffer = response_pattern.sub("", self.input_buffer, count=1)
            self.cell_size_query_pending = False

        partial_pattern = re.compile(r"\x1b(\[6?;?[\d;]*)?$")
        if self.input_buffer and partial_pattern.search(self.input_buffer):
            last_char = self.input_buffer[-1]
            if not re.match(r"[a-zA-Z~]", last_char):
                return ""

        result = self.input_buffer
        self.input_buffer = ""
        self.cell_size_query_pending = False
        return result

    def _resolve_overlay_layout(self, options: OverlayOptions | None, overlay_height: int, term_width: int, term_height: int) -> dict[str, int | None]:
        opt = options or OverlayOptions()
        if isinstance(opt.margin, int):
            margin_top = margin_right = margin_bottom = margin_left = max(0, opt.margin)
        elif opt.margin:
            margin_top = max(0, opt.margin.top or 0)
            margin_right = max(0, opt.margin.right or 0)
            margin_bottom = max(0, opt.margin.bottom or 0)
            margin_left = max(0, opt.margin.left or 0)
        else:
            margin_top = margin_right = margin_bottom = margin_left = 0
        avail_width = max(1, term_width - margin_left - margin_right)
        avail_height = max(1, term_height - margin_top - margin_bottom)

        def parse_size(value: SizeValue | None, reference: int) -> int | None:
            if value is None:
                return None
            if isinstance(value, int):
                return value
            if value.endswith("%"):
                try:
                    return int(reference * float(value[:-1]) / 100)
                except ValueError:
                    return None
            return None

        width = parse_size(opt.width, term_width) or min(80, avail_width)
        if opt.min_width is not None:
            width = max(width, opt.min_width)
        width = max(1, min(width, avail_width))
        max_height = parse_size(opt.max_height, term_height)
        if max_height is not None:
            max_height = max(1, min(max_height, avail_height))
        effective_height = min(overlay_height, max_height) if max_height is not None else overlay_height
        if opt.row is not None:
            if isinstance(opt.row, str) and opt.row.endswith("%"):
                max_row = max(0, avail_height - effective_height)
                row = margin_top + int(max_row * float(opt.row[:-1]) / 100)
            elif isinstance(opt.row, int):
                row = opt.row
            else:
                row = self._resolve_anchor_row(opt.anchor or "center", effective_height, avail_height, margin_top)
        else:
            row = self._resolve_anchor_row(opt.anchor or "center", effective_height, avail_height, margin_top)
        if opt.col is not None:
            if isinstance(opt.col, str) and opt.col.endswith("%"):
                max_col = max(0, avail_width - width)
                col = margin_left + int(max_col * float(opt.col[:-1]) / 100)
            elif isinstance(opt.col, int):
                col = opt.col
            else:
                col = self._resolve_anchor_col(opt.anchor or "center", width, avail_width, margin_left)
        else:
            col = self._resolve_anchor_col(opt.anchor or "center", width, avail_width, margin_left)
        if opt.offset_y is not None:
            row += opt.offset_y
        if opt.offset_x is not None:
            col += opt.offset_x
        row = max(margin_top, min(row, term_height - margin_bottom - effective_height))
        col = max(margin_left, min(col, term_width - margin_right - width))
        return {"width": width, "row": row, "col": col, "max_height": max_height}

    def _resolve_anchor_row(self, anchor: str, height: int, avail_height: int, margin_top: int) -> int:
        if anchor in {"top-left", "top-center", "top-right"}:
            return margin_top
        if anchor in {"bottom-left", "bottom-center", "bottom-right"}:
            return margin_top + avail_height - height
        return margin_top + (avail_height - height) // 2

    def _resolve_anchor_col(self, anchor: str, width: int, avail_width: int, margin_left: int) -> int:
        if anchor in {"top-left", "left-center", "bottom-left"}:
            return margin_left
        if anchor in {"top-right", "right-center", "bottom-right"}:
            return margin_left + avail_width - width
        return margin_left + (avail_width - width) // 2

    def _composite_overlays(self, lines: list[str], term_width: int, term_height: int) -> list[str]:
        if not self.overlay_stack:
            return lines
        result = list(lines)
        rendered: list[dict[str, object]] = []
        min_lines_needed = len(result)
        visible_entries = [e for e in self.overlay_stack if self._is_overlay_visible(e)]
        visible_entries.sort(key=lambda e: int(e["focus_order"]))
        for entry in visible_entries:
            component = entry["component"]
            options = entry["options"]
            layout = self._resolve_overlay_layout(options, 0, term_width, term_height)
            overlay_lines = component.render(int(layout["width"]))  # type: ignore[arg-type]
            max_height = layout["max_height"]
            if max_height is not None and len(overlay_lines) > max_height:
                overlay_lines = overlay_lines[: int(max_height)]
            layout = self._resolve_overlay_layout(options, len(overlay_lines), term_width, term_height)
            rendered.append({"overlay_lines": overlay_lines, **layout})
            min_lines_needed = max(min_lines_needed, int(layout["row"]) + len(overlay_lines))
        working_height = max(self.max_lines_rendered, min_lines_needed)
        while len(result) < working_height:
            result.append("")
        viewport_start = max(0, working_height - term_height)
        for item in rendered:
            overlay_lines = item["overlay_lines"]
            row = int(item["row"])
            col = int(item["col"])
            w = int(item["width"])
            for i, overlay_line in enumerate(overlay_lines):
                idx = viewport_start + row + i
                if 0 <= idx < len(result):
                    truncated = slice_by_column(overlay_line, 0, w, True) if visible_width(overlay_line) > w else overlay_line
                    result[idx] = self._composite_line_at(result[idx], truncated, col, w, term_width)
        return result

    def _composite_line_at(self, base_line: str, overlay_line: str, start_col: int, overlay_width: int, total_width: int) -> str:
        if is_image_line(base_line):
            return base_line
        after_start = start_col + overlay_width
        base = extract_segments(base_line, start_col, after_start, total_width - after_start, True)
        overlay = slice_with_width(overlay_line, 0, overlay_width, True)
        before_pad = max(0, start_col - int(base["beforeWidth"]))
        overlay_pad = max(0, overlay_width - int(overlay["width"]))
        actual_before_width = max(start_col, int(base["beforeWidth"]))
        actual_overlay_width = max(overlay_width, int(overlay["width"]))
        after_target = max(0, total_width - actual_before_width - actual_overlay_width)
        after_pad = max(0, after_target - int(base["afterWidth"]))
        result = (
            str(base["before"])
            + " " * before_pad
            + "\x1b[0m\x1b]8;;\x07"
            + str(overlay["text"])
            + " " * overlay_pad
            + "\x1b[0m\x1b]8;;\x07"
            + str(base["after"])
            + " " * after_pad
        )
        if visible_width(result) <= total_width:
            return result
        return slice_by_column(result, 0, total_width, True)

    def _apply_line_resets(self, lines: list[str]) -> list[str]:
        reset = "\x1b[0m\x1b]8;;\x07"
        return [line if is_image_line(line) else line + reset for line in lines]

    def _position_hardware_cursor(self, cursor_pos: dict[str, int] | None, total_lines: int) -> None:
        if not cursor_pos or total_lines <= 0:
            self.terminal.hide_cursor()
            return

        target_row = max(0, min(cursor_pos["row"], total_lines - 1))
        target_col = max(0, cursor_pos["col"])
        row_delta = target_row - self.hardware_cursor_row
        if row_delta:
            self.terminal.move_by(row_delta)
        self.terminal.write(f"\x1b[{target_col + 1}G")
        self.hardware_cursor_row = target_row
        if self.show_hardware_cursor:
            self.terminal.show_cursor()
        else:
            self.terminal.hide_cursor()

    def _extract_cursor_position(self, lines: list[str], height: int) -> dict[str, int] | None:
        viewport_top = max(0, len(lines) - height)
        for row in range(len(lines) - 1, viewport_top - 1, -1):
            marker_index = lines[row].find(CURSOR_MARKER)
            if marker_index != -1:
                before_marker = lines[row][:marker_index]
                col = visible_width(before_marker)
                lines[row] = lines[row][:marker_index] + lines[row][marker_index + len(CURSOR_MARKER) :]
                return {"row": row, "col": col}
        return None

    def do_render(self) -> None:
        if self.stopped:
            return
        width = self.terminal.columns
        height = self.terminal.rows
        viewport_top = max(0, self.max_lines_rendered - height)
        prev_viewport_top = self.previous_viewport_top
        hardware_cursor_row = self.hardware_cursor_row

        def compute_line_diff(target_row: int) -> int:
            current_screen_row = hardware_cursor_row - prev_viewport_top
            target_screen_row = target_row - viewport_top
            return target_screen_row - current_screen_row

        new_lines = self.render(width)
        if self.overlay_stack:
            new_lines = self._composite_overlays(new_lines, width, height)
        cursor_pos = self._extract_cursor_position(new_lines, height)
        new_lines = self._apply_line_resets(new_lines)
        width_changed = self.previous_width != 0 and self.previous_width != width
        height_changed = self.previous_height != 0 and self.previous_height != height

        def full_render(clear: bool) -> None:
            self.full_redraw_count += 1
            buffer = "\x1b[?2026h"
            if clear:
                buffer += "\x1b[2J\x1b[H\x1b[3J"
            for idx, line in enumerate(new_lines):
                if idx > 0:
                    buffer += "\r\n"
                buffer += line
            buffer += "\x1b[?2026l"
            self.terminal.write(buffer)
            self.cursor_row = max(0, len(new_lines) - 1)
            self.hardware_cursor_row = self.cursor_row
            if clear:
                self.max_lines_rendered = len(new_lines)
            else:
                self.max_lines_rendered = max(self.max_lines_rendered, len(new_lines))
            self.previous_viewport_top = max(0, self.max_lines_rendered - height)
            self._position_hardware_cursor(cursor_pos, len(new_lines))
            self.previous_lines = new_lines
            self.previous_width = width
            self.previous_height = height

        if not self.previous_lines and not width_changed and not height_changed:
            full_render(False)
            return

        if width_changed or height_changed:
            full_render(True)
            return

        if self.clear_on_shrink and len(new_lines) < self.max_lines_rendered and not self.overlay_stack:
            full_render(True)
            return

        first_changed = -1
        last_changed = -1
        max_lines = max(len(new_lines), len(self.previous_lines))
        for idx in range(max_lines):
            old_line = self.previous_lines[idx] if idx < len(self.previous_lines) else ""
            new_line = new_lines[idx] if idx < len(new_lines) else ""
            if old_line != new_line:
                if first_changed == -1:
                    first_changed = idx
                last_changed = idx

        appended_lines = len(new_lines) > len(self.previous_lines)
        if appended_lines:
            if first_changed == -1:
                first_changed = len(self.previous_lines)
            last_changed = len(new_lines) - 1
        append_start = appended_lines and first_changed == len(self.previous_lines) and first_changed > 0

        if first_changed == -1:
            self._position_hardware_cursor(cursor_pos, len(new_lines))
            self.previous_viewport_top = max(0, self.max_lines_rendered - height)
            self.previous_height = height
            return

        if first_changed >= len(new_lines):
            if len(self.previous_lines) > len(new_lines):
                buffer = "\x1b[?2026h"
                target_row = max(0, len(new_lines) - 1)
                line_diff = compute_line_diff(target_row)
                if line_diff > 0:
                    buffer += f"\x1b[{line_diff}B"
                elif line_diff < 0:
                    buffer += f"\x1b[{-line_diff}A"
                buffer += "\r"
                extra_lines = len(self.previous_lines) - len(new_lines)
                if extra_lines > height:
                    full_render(True)
                    return
                if extra_lines > 0:
                    buffer += "\x1b[1B"
                for idx in range(extra_lines):
                    buffer += "\r\x1b[2K"
                    if idx < extra_lines - 1:
                        buffer += "\x1b[1B"
                if extra_lines > 0:
                    buffer += f"\x1b[{extra_lines}A"
                buffer += "\x1b[?2026l"
                self.terminal.write(buffer)
                self.cursor_row = target_row
                self.hardware_cursor_row = target_row
            self._position_hardware_cursor(cursor_pos, len(new_lines))
            self.previous_lines = new_lines
            self.previous_width = width
            self.previous_height = height
            self.previous_viewport_top = max(0, self.max_lines_rendered - height)
            return

        previous_content_viewport_top = max(0, len(self.previous_lines) - height)
        if first_changed < previous_content_viewport_top:
            full_render(True)
            return

        buffer = "\x1b[?2026h"
        prev_viewport_bottom = prev_viewport_top + height - 1
        move_target_row = first_changed - 1 if append_start else first_changed
        if move_target_row > prev_viewport_bottom:
            current_screen_row = max(0, min(height - 1, hardware_cursor_row - prev_viewport_top))
            move_to_bottom = height - 1 - current_screen_row
            if move_to_bottom > 0:
                buffer += f"\x1b[{move_to_bottom}B"
            scroll = move_target_row - prev_viewport_bottom
            buffer += "\r\n" * scroll
            prev_viewport_top += scroll
            viewport_top += scroll
            hardware_cursor_row = move_target_row

        line_diff = compute_line_diff(move_target_row)
        if line_diff > 0:
            buffer += f"\x1b[{line_diff}B"
        elif line_diff < 0:
            buffer += f"\x1b[{-line_diff}A"
        buffer += "\r\n" if append_start else "\r"

        render_end = min(last_changed, len(new_lines) - 1)
        for idx in range(first_changed, render_end + 1):
            if idx > first_changed:
                buffer += "\r\n"
            buffer += "\x1b[2K"
            line = new_lines[idx]
            if not is_image_line(line) and visible_width(line) > width:
                raise ValueError(f"Rendered line {idx} exceeds terminal width ({visible_width(line)} > {width}).")
            buffer += line

        final_cursor_row = render_end
        if len(self.previous_lines) > len(new_lines):
            if render_end < len(new_lines) - 1:
                move_down = len(new_lines) - 1 - render_end
                buffer += f"\x1b[{move_down}B"
                final_cursor_row = len(new_lines) - 1
            extra_lines = len(self.previous_lines) - len(new_lines)
            for _idx in range(len(new_lines), len(self.previous_lines)):
                buffer += "\r\n\x1b[2K"
            buffer += f"\x1b[{extra_lines}A"

        buffer += "\x1b[?2026l"
        self.terminal.write(buffer)
        self.cursor_row = max(0, len(new_lines) - 1)
        self.hardware_cursor_row = final_cursor_row
        self.max_lines_rendered = max(self.max_lines_rendered, len(new_lines))
        self.previous_viewport_top = max(0, self.max_lines_rendered - height)
        self._position_hardware_cursor(cursor_pos, len(new_lines))
        self.previous_lines = new_lines
        self.previous_width = width
        self.previous_height = height

    # overlay handle helpers
    def _hide_entry(self, entry: dict[str, object]) -> None:
        if entry not in self.overlay_stack:
            return
        self.overlay_stack.remove(entry)
        if self.focused_component is entry["component"]:
            top_visible = self._get_topmost_visible_overlay()
            self.set_focus(top_visible["component"] if top_visible else entry["pre_focus"])  # type: ignore[arg-type]
        self.request_render()

    def _set_hidden(self, entry: dict[str, object], hidden: bool) -> None:
        if entry not in self.overlay_stack:
            return
        if entry["hidden"] == hidden:
            return
        entry["hidden"] = hidden
        if hidden and self.focused_component is entry["component"]:
            top_visible = self._get_topmost_visible_overlay()
            self.set_focus(top_visible["component"] if top_visible else entry["pre_focus"])  # type: ignore[arg-type]
        elif not hidden and not getattr(entry["options"], "non_capturing", False) and self._is_overlay_visible(entry):
            entry["focus_order"] = self.focus_order_counter + 1
            self.focus_order_counter += 1
            self.set_focus(entry["component"])  # type: ignore[arg-type]
        self.request_render()

    def _focus_entry(self, entry: dict[str, object]) -> None:
        if entry not in self.overlay_stack or not self._is_overlay_visible(entry):
            return
        if self.focused_component is not entry["component"]:
            self.set_focus(entry["component"])  # type: ignore[arg-type]
        entry["focus_order"] = self.focus_order_counter + 1
        self.focus_order_counter += 1
        self.request_render()

    def _unfocus_entry(self, entry: dict[str, object]) -> None:
        if self.focused_component is not entry["component"]:
            return
        top_visible = self._get_topmost_visible_overlay()
        self.set_focus(top_visible["component"] if top_visible and top_visible is not entry else entry["pre_focus"])  # type: ignore[arg-type]
        self.request_render()
