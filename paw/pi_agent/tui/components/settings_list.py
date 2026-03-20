from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

from ..fuzzy import fuzzy_filter
from ..keybindings import get_editor_keybindings
from ..utils import truncate_to_width, visible_width, wrap_text_with_ansi
from .input import Input


@dataclass(slots=True)
class SettingItem:
    id: str
    label: str
    current_value: str
    description: str | None = None
    values: list[str] | None = None
    submenu: Callable[[str, Callable[[str | None], None]], object] | None = None


class SettingsListTheme(Protocol):
    def label(self, text: str, selected: bool) -> str: ...
    def value(self, text: str, selected: bool) -> str: ...
    def description(self, text: str) -> str: ...
    cursor: str
    def hint(self, text: str) -> str: ...


@dataclass(slots=True)
class SettingsListOptions:
    enable_search: bool = False


def _theme_call(theme: SettingsListTheme | dict[str, object], key: str, *args: object) -> str:
    fn = getattr(theme, key, None)
    if fn is None and isinstance(theme, dict):
        fn = theme.get(key)
    if callable(fn):
        return str(fn(*args))
    if key == "cursor":
        return str(fn) if fn is not None else "→ "
    return str(args[0]) if args else ""


class SettingsList:
    def __init__(
        self,
        items: list[SettingItem],
        max_visible: int,
        theme: SettingsListTheme | dict[str, object],
        on_change: Callable[[str, str], None],
        on_cancel: Callable[[], None],
        options: SettingsListOptions | dict[str, Any] | None = None,
    ) -> None:
        self.items = items
        self.filtered_items = items
        self.theme = theme
        self.selected_index = 0
        self.max_visible = max_visible
        self.on_change = on_change
        self.on_cancel = on_cancel
        self.search_enabled = self._coerce_options(options).enable_search
        self.search_input = Input() if self.search_enabled else None
        self.submenu_component = None
        self.submenu_item_index: int | None = None

    def _coerce_options(self, options: SettingsListOptions | dict[str, Any] | None) -> SettingsListOptions:
        if options is None:
            return SettingsListOptions()
        if isinstance(options, SettingsListOptions):
            return options
        return SettingsListOptions(enable_search=bool(options.get("enable_search", False)))

    def update_value(self, setting_id: str, new_value: str) -> None:
        item = next((item for item in self.items if item.id == setting_id), None)
        if item:
            item.current_value = new_value

    def invalidate(self) -> None:
        if self.submenu_component and hasattr(self.submenu_component, "invalidate"):
            self.submenu_component.invalidate()

    def render(self, width: int) -> list[str]:
        if self.submenu_component:
            return self.submenu_component.render(width)
        return self._render_main_list(width)

    def _render_main_list(self, width: int) -> list[str]:
        lines: list[str] = []
        if self.search_enabled and self.search_input:
            lines.extend(self.search_input.render(width))
            lines.append("")

        if not self.items:
            lines.append(_theme_call(self.theme, "hint", "  No settings available"))
            if self.search_enabled:
                self._add_hint_line(lines, width)
            return lines

        display_items = self.filtered_items if self.search_enabled else self.items
        if not display_items:
            lines.append(truncate_to_width(_theme_call(self.theme, "hint", "  No matching settings"), width))
            self._add_hint_line(lines, width)
            return lines

        start_index = max(0, min(self.selected_index - self.max_visible // 2, len(display_items) - self.max_visible))
        end_index = min(start_index + self.max_visible, len(display_items))
        max_label_width = min(30, max(visible_width(item.label) for item in self.items))

        for i in range(start_index, end_index):
            item = display_items[i]
            is_selected = i == self.selected_index
            prefix = _theme_call(self.theme, "cursor") if is_selected else "  "
            prefix_width = visible_width(prefix)
            label_padded = item.label + " " * max(0, max_label_width - visible_width(item.label))
            label_text = _theme_call(self.theme, "label", label_padded, is_selected)
            used_width = prefix_width + max_label_width + visible_width("  ")
            value_max_width = width - used_width - 2
            value_text = _theme_call(
                self.theme,
                "value",
                truncate_to_width(item.current_value, value_max_width, ""),
                is_selected,
            )
            lines.append(truncate_to_width(prefix + label_text + "  " + value_text, width))

        if start_index > 0 or end_index < len(display_items):
            scroll_text = f"  ({self.selected_index + 1}/{len(display_items)})"
            lines.append(_theme_call(self.theme, "hint", truncate_to_width(scroll_text, width - 2, "")))

        selected_item = display_items[self.selected_index]
        if selected_item.description:
            lines.append("")
            wrapped = wrap_text_with_ansi(selected_item.description, width - 4)
            for line in wrapped:
                lines.append(_theme_call(self.theme, "description", f"  {line}"))

        self._add_hint_line(lines, width)
        return lines

    def handle_input(self, data: str) -> None:
        if self.submenu_component and hasattr(self.submenu_component, "handle_input"):
            self.submenu_component.handle_input(data)
            return

        kb = get_editor_keybindings()
        display_items = self.filtered_items if self.search_enabled else self.items
        if kb.matches(data, "selectUp"):
            if not display_items:
                return
            self.selected_index = self.selected_index - 1 if self.selected_index > 0 else len(display_items) - 1
        elif kb.matches(data, "selectDown"):
            if not display_items:
                return
            self.selected_index = 0 if self.selected_index == len(display_items) - 1 else self.selected_index + 1
        elif kb.matches(data, "selectConfirm") or data == " " or data == "\r":
            self._activate_item()
        elif kb.matches(data, "selectCancel"):
            self.on_cancel()
        elif self.search_enabled and self.search_input:
            sanitized = data.replace(" ", "")
            if not sanitized:
                return
            self.search_input.handle_input(sanitized)
            self._apply_filter(self.search_input.get_value())

    def _activate_item(self) -> None:
        display_items = self.filtered_items if self.search_enabled else self.items
        item = display_items[self.selected_index] if display_items else None
        if item is None:
            return
        if item.submenu is not None:
            self.submenu_item_index = self.selected_index

            def done(selected_value: str | None = None) -> None:
                if selected_value is not None:
                    item.current_value = selected_value
                    self.on_change(item.id, selected_value)
                self._close_submenu()

            self.submenu_component = item.submenu(item.current_value, done)
            return
        if item.values:
            current_index = item.values.index(item.current_value) if item.current_value in item.values else -1
            next_index = (current_index + 1) % len(item.values)
            new_value = item.values[next_index]
            item.current_value = new_value
            self.on_change(item.id, new_value)

    def _close_submenu(self) -> None:
        self.submenu_component = None
        if self.submenu_item_index is not None:
            self.selected_index = self.submenu_item_index
            self.submenu_item_index = None

    def _apply_filter(self, query: str) -> None:
        self.filtered_items = fuzzy_filter(self.items, query, lambda item: item.label)
        self.selected_index = 0

    def _add_hint_line(self, lines: list[str], width: int) -> None:
        lines.append("")
        hint = "  Type to search · Enter/Space to change · Esc to cancel" if self.search_enabled else "  Enter/Space to change · Esc to cancel"
        lines.append(truncate_to_width(_theme_call(self.theme, "hint", hint), width))
