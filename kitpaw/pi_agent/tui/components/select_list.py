from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

from ..keybindings import get_editor_keybindings
from ..utils import truncate_to_width, visible_width

DEFAULT_PRIMARY_COLUMN_WIDTH = 32
PRIMARY_COLUMN_GAP = 2
MIN_DESCRIPTION_WIDTH = 10


def _normalize_to_single_line(text: str) -> str:
    return " ".join(text.splitlines()).strip()


def _clamp(value: int, min_value: int, max_value: int) -> int:
    return max(min_value, min(value, max_value))


@dataclass(slots=True)
class SelectItem:
    value: str
    label: str
    description: str | None = None


@dataclass(slots=True)
class SelectListTruncatePrimaryContext:
    text: str
    max_width: int
    column_width: int
    item: SelectItem
    is_selected: bool


@dataclass(slots=True)
class SelectListLayoutOptions:
    min_primary_column_width: int | None = None
    max_primary_column_width: int | None = None
    truncate_primary: Callable[[SelectListTruncatePrimaryContext], str] | None = None


class SelectListTheme(Protocol):
    def selected_prefix(self, text: str) -> str: ...
    def selected_text(self, text: str) -> str: ...
    def description(self, text: str) -> str: ...
    def scroll_info(self, text: str) -> str: ...
    def no_match(self, text: str) -> str: ...


def _theme_call(theme: SelectListTheme | dict[str, object], key: str, text: str) -> str:
    fn = getattr(theme, key, None)
    if fn is None and isinstance(theme, dict):
        fn = theme.get(key)
    if callable(fn):
        return str(fn(text))
    return text


class SelectList:
    def __init__(
        self,
        items: list[SelectItem],
        max_visible: int,
        theme: SelectListTheme | dict[str, object],
        layout: SelectListLayoutOptions | dict[str, Any] | None = None,
    ) -> None:
        self.items = items
        self.filtered_items = items
        self.selected_index = 0
        self.max_visible = max_visible
        self.theme = theme
        self.layout = self._coerce_layout(layout)
        self.on_select = None
        self.on_cancel = None
        self.on_selection_change = None

    def _coerce_layout(self, layout: SelectListLayoutOptions | dict[str, Any] | None) -> SelectListLayoutOptions:
        if layout is None:
            return SelectListLayoutOptions()
        if isinstance(layout, SelectListLayoutOptions):
            return layout
        return SelectListLayoutOptions(
            min_primary_column_width=layout.get("min_primary_column_width"),
            max_primary_column_width=layout.get("max_primary_column_width"),
            truncate_primary=layout.get("truncate_primary"),
        )

    def set_filter(self, filter_text: str) -> None:
        self.filtered_items = [item for item in self.items if item.value.lower().startswith(filter_text.lower())]
        self.selected_index = 0

    def set_selected_index(self, index: int) -> None:
        self.selected_index = max(0, min(index, len(self.filtered_items) - 1))

    def get_selected_item(self) -> SelectItem | None:
        if not self.filtered_items:
            return None
        return self.filtered_items[self.selected_index]

    def invalidate(self) -> None:
        return None

    def render(self, width: int) -> list[str]:
        if not self.filtered_items:
            return [_theme_call(self.theme, "no_match", "  No matching commands")]

        lines: list[str] = []
        primary_column_width = self._get_primary_column_width()
        start_index = max(0, min(self.selected_index - self.max_visible // 2, len(self.filtered_items) - self.max_visible))
        end_index = min(start_index + self.max_visible, len(self.filtered_items))

        for i in range(start_index, end_index):
            item = self.filtered_items[i]
            is_selected = i == self.selected_index
            description = _normalize_to_single_line(item.description) if item.description else None
            lines.append(self._render_item(item, is_selected, width, description, primary_column_width))

        if start_index > 0 or end_index < len(self.filtered_items):
            scroll_text = f"  ({self.selected_index + 1}/{len(self.filtered_items)})"
            lines.append(_theme_call(self.theme, "scroll_info", truncate_to_width(scroll_text, width - 2, "")))

        return lines

    def _render_item(
        self,
        item: SelectItem,
        is_selected: bool,
        width: int,
        description_single_line: str | None,
        primary_column_width: int,
    ) -> str:
        prefix = "→ " if is_selected else "  "
        prefix_width = visible_width(prefix)

        if description_single_line and width > 40:
            effective_primary_column_width = max(1, min(primary_column_width, width - prefix_width - 4))
            max_primary_width = max(1, effective_primary_column_width - PRIMARY_COLUMN_GAP)
            truncated_value = self._truncate_primary(item, is_selected, max_primary_width, effective_primary_column_width)
            truncated_value_width = visible_width(truncated_value)
            spacing = " " * max(1, effective_primary_column_width - truncated_value_width)
            description_start = prefix_width + truncated_value_width + len(spacing)
            remaining_width = width - description_start - 2

            if remaining_width > MIN_DESCRIPTION_WIDTH:
                truncated_desc = truncate_to_width(description_single_line, remaining_width, "")
                if is_selected:
                    return _theme_call(
                        self.theme,
                        "selected_text",
                        f"{prefix}{truncated_value}{spacing}{truncated_desc}",
                    )
                return prefix + truncated_value + _theme_call(self.theme, "description", f"{spacing}{truncated_desc}")

        max_width = width - prefix_width - 2
        truncated_value = self._truncate_primary(item, is_selected, max_width, max_width)
        if is_selected:
            return _theme_call(self.theme, "selected_text", f"{prefix}{truncated_value}")
        return prefix + truncated_value

    def _get_primary_column_width(self) -> int:
        bounds = self._get_primary_column_bounds()
        widest_primary = 0
        for item in self.filtered_items:
            widest_primary = max(widest_primary, visible_width(self._get_display_value(item)) + PRIMARY_COLUMN_GAP)
        return _clamp(widest_primary, bounds["min"], bounds["max"])

    def _get_primary_column_bounds(self) -> dict[str, int]:
        raw_min = (
            self.layout.min_primary_column_width
            if self.layout.min_primary_column_width is not None
            else self.layout.max_primary_column_width
            if self.layout.max_primary_column_width is not None
            else DEFAULT_PRIMARY_COLUMN_WIDTH
        )
        raw_max = (
            self.layout.max_primary_column_width
            if self.layout.max_primary_column_width is not None
            else self.layout.min_primary_column_width
            if self.layout.min_primary_column_width is not None
            else DEFAULT_PRIMARY_COLUMN_WIDTH
        )
        return {
            "min": max(1, min(raw_min, raw_max)),
            "max": max(1, max(raw_min, raw_max)),
        }

    def _truncate_primary(self, item: SelectItem, is_selected: bool, max_width: int, column_width: int) -> str:
        display_value = self._get_display_value(item)
        if self.layout.truncate_primary:
            truncated_value = self.layout.truncate_primary(
                SelectListTruncatePrimaryContext(
                    text=display_value,
                    max_width=max_width,
                    column_width=column_width,
                    item=item,
                    is_selected=is_selected,
                )
            )
        else:
            truncated_value = truncate_to_width(display_value, max_width, "")
        return truncate_to_width(truncated_value, max_width, "")

    def _get_display_value(self, item: SelectItem) -> str:
        return item.label or item.value

    def _notify_selection_change(self) -> None:
        if self.filtered_items and self.on_selection_change:
            self.on_selection_change(self.filtered_items[self.selected_index])

    def handle_input(self, key_data: str) -> None:
        kb = get_editor_keybindings()
        if kb.matches(key_data, "selectUp"):
            self.selected_index = self.selected_index - 1 if self.selected_index > 0 else len(self.filtered_items) - 1
            self._notify_selection_change()
        elif kb.matches(key_data, "selectDown"):
            self.selected_index = 0 if self.selected_index == len(self.filtered_items) - 1 else self.selected_index + 1
            self._notify_selection_change()
        elif kb.matches(key_data, "selectConfirm"):
            if self.filtered_items and self.on_select:
                self.on_select(self.filtered_items[self.selected_index])
        elif kb.matches(key_data, "selectCancel") and self.on_cancel:
            self.on_cancel()
