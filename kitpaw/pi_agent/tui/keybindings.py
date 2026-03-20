from __future__ import annotations

from dataclasses import dataclass, field

from .keys import matches_key

EditorAction = str


DEFAULT_EDITOR_KEYBINDINGS: dict[EditorAction, list[str]] = {
    "cursorUp": ["up"],
    "cursorDown": ["down"],
    "cursorLeft": ["left", "ctrl+b"],
    "cursorRight": ["right", "ctrl+f"],
    "cursorWordLeft": ["alt+left", "ctrl+left", "alt+b"],
    "cursorWordRight": ["alt+right", "ctrl+right", "alt+f"],
    "cursorLineStart": ["home", "ctrl+a"],
    "cursorLineEnd": ["end", "ctrl+e"],
    "jumpForward": ["ctrl+]"],
    "jumpBackward": ["ctrl+alt+]"],
    "pageUp": ["pageUp"],
    "pageDown": ["pageDown"],
    "deleteCharBackward": ["backspace"],
    "deleteCharForward": ["delete", "ctrl+d"],
    "deleteWordBackward": ["ctrl+w", "alt+backspace"],
    "deleteWordForward": ["alt+d", "alt+delete"],
    "deleteToLineStart": ["ctrl+u"],
    "deleteToLineEnd": ["ctrl+k"],
    "newLine": ["shift+enter"],
    "submit": ["enter"],
    "tab": ["tab"],
    "selectUp": ["up"],
    "selectDown": ["down"],
    "selectPageUp": ["pageUp"],
    "selectPageDown": ["pageDown"],
    "selectConfirm": ["enter"],
    "selectCancel": ["escape", "ctrl+c"],
    "copy": ["ctrl+c"],
    "yank": ["ctrl+y"],
    "yankPop": ["alt+y"],
    "undo": ["ctrl+-"],
    "expandTools": ["ctrl+o"],
    "treeFoldOrUp": ["ctrl+left", "alt+left"],
    "treeUnfoldOrDown": ["ctrl+right", "alt+right"],
    "toggleSessionPath": ["ctrl+p"],
    "toggleSessionSort": ["ctrl+s"],
    "renameSession": ["ctrl+r"],
    "deleteSession": ["ctrl+d"],
    "deleteSessionNoninvasive": ["ctrl+backspace"],
}


@dataclass(slots=True)
class EditorKeybindingsManager:
    config: dict[EditorAction, list[str]] = field(default_factory=dict)
    _action_to_keys: dict[EditorAction, list[str]] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        self.build_maps(self.config)

    def build_maps(self, config: dict[EditorAction, list[str]]) -> None:
        self._action_to_keys = {action: keys[:] for action, keys in DEFAULT_EDITOR_KEYBINDINGS.items()}
        for action, keys in config.items():
            self._action_to_keys[action] = keys[:] if isinstance(keys, list) else [keys]

    def matches(self, data: str, action: EditorAction) -> bool:
        return any(matches_key(data, key) for key in self._action_to_keys.get(action, []))

    def get_keys(self, action: EditorAction) -> list[str]:
        return self._action_to_keys.get(action, [])

    def set_config(self, config: dict[EditorAction, list[str]]) -> None:
        self.build_maps(config)


_GLOBAL_EDITOR_KEYBINDINGS: EditorKeybindingsManager | None = None


def get_editor_keybindings() -> EditorKeybindingsManager:
    global _GLOBAL_EDITOR_KEYBINDINGS
    if _GLOBAL_EDITOR_KEYBINDINGS is None:
        _GLOBAL_EDITOR_KEYBINDINGS = EditorKeybindingsManager()
    return _GLOBAL_EDITOR_KEYBINDINGS


def set_editor_keybindings(manager: EditorKeybindingsManager) -> None:
    global _GLOBAL_EDITOR_KEYBINDINGS
    _GLOBAL_EDITOR_KEYBINDINGS = manager
