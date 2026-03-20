from __future__ import annotations

import re
from dataclasses import dataclass

_kitty_protocol_active = False

_KITTY_CSI_U_RE = re.compile(
    r"^\x1b\[(?P<code>\d+)(?::(?P<shifted>\d*))?(?::(?P<base>\d*))?(?:;(?P<mods>\d+)(?::(?P<event>\d+))?)?u$"
)
_MODIFY_OTHER_KEYS_RE = re.compile(r"^\x1b\[27;(?P<mods>\d+);(?P<code>\d+)~$")

_RXVT_MAP = {
    "\x1b[a": "shift+up",
    "\x1b[b": "shift+down",
    "\x1b[c": "shift+right",
    "\x1b[d": "shift+left",
    "\x1bOa": "ctrl+up",
    "\x1bOb": "ctrl+down",
    "\x1bOc": "ctrl+right",
    "\x1bOd": "ctrl+left",
    "\x1b[2$": "shift+insert",
    "\x1b[2^": "ctrl+insert",
    "\x1b[7$": "shift+home",
}

_LEGACY_MAP = {
    "\x1b[A": "up",
    "\x1b[B": "down",
    "\x1b[C": "right",
    "\x1b[D": "left",
    "\x1b[Z": "shift+tab",
    "\x1b[H": "home",
    "\x1b[F": "end",
    "\x1b[3~": "delete",
    "\x1b[[5~": "pageUp",
    "\x1bOP": "f1",
    "\x1b[24~": "f12",
    "\x1b[E": "clear",
    "\x1bOA": "up",
    "\x1bOB": "down",
    "\x1bOC": "right",
    "\x1bOD": "left",
    "\x1bOH": "home",
    "\x1bOF": "end",
    "\x1bOM": "enter",
    "\x7f": "backspace",
    "\x08": "ctrl+backspace",
    "\x1bp": "alt+up",
    "\x1bn": "alt+down",
    "\x1bb": "alt+left",
    "\x1bf": "alt+right",
    "\x1by": "alt+y",
}

_CSI_MODIFIER_KEY_MAP = {
    ("1", 5, "D"): "ctrl+left",
    ("1", 5, "C"): "ctrl+right",
    ("1", 5, "A"): "ctrl+up",
    ("1", 5, "B"): "ctrl+down",
}


def set_kitty_protocol_active(active: bool) -> None:
    global _kitty_protocol_active
    _kitty_protocol_active = active


def is_kitty_protocol_active() -> bool:
    return _kitty_protocol_active


@dataclass(frozen=True, slots=True)
class Key:
    escape: str = "escape"
    esc: str = "esc"
    enter: str = "enter"
    return_: str = "return"
    tab: str = "tab"
    space: str = "space"
    backspace: str = "backspace"
    delete: str = "delete"
    insert: str = "insert"
    clear: str = "clear"
    home: str = "home"
    end: str = "end"
    page_up: str = "pageUp"
    page_down: str = "pageDown"
    up: str = "up"
    down: str = "down"
    left: str = "left"
    right: str = "right"
    f1: str = "f1"
    f2: str = "f2"
    f3: str = "f3"
    f4: str = "f4"
    f5: str = "f5"
    f6: str = "f6"
    f7: str = "f7"
    f8: str = "f8"
    f9: str = "f9"
    f10: str = "f10"
    f11: str = "f11"
    f12: str = "f12"

    @staticmethod
    def ctrl(key: str) -> str:
        return f"ctrl+{key}"

    @staticmethod
    def shift(key: str) -> str:
        return f"shift+{key}"

    @staticmethod
    def alt(key: str) -> str:
        return f"alt+{key}"

    @staticmethod
    def ctrl_shift(key: str) -> str:
        return f"ctrl+shift+{key}"

    @staticmethod
    def shift_ctrl(key: str) -> str:
        return f"shift+ctrl+{key}"

    @staticmethod
    def ctrl_alt(key: str) -> str:
        return f"ctrl+alt+{key}"

    @staticmethod
    def alt_ctrl(key: str) -> str:
        return f"alt+ctrl+{key}"

    @staticmethod
    def shift_alt(key: str) -> str:
        return f"shift+alt+{key}"

    @staticmethod
    def alt_shift(key: str) -> str:
        return f"alt+shift+{key}"

    @staticmethod
    def ctrl_shift_alt(key: str) -> str:
        return f"ctrl+shift+alt+{key}"


def _apply_modifiers(key: str, mods: int) -> str | None:
    mapping = {
        1: [],
        2: ["shift"],
        3: ["alt"],
        4: ["shift", "alt"],
        5: ["ctrl"],
        6: ["ctrl", "shift"],
        7: ["ctrl", "alt"],
        8: ["ctrl", "shift", "alt"],
    }
    modifiers = mapping.get(mods)
    if modifiers is None:
        return None
    if not modifiers:
        return key
    return "+".join([*modifiers, key])


def _decode_codepoint(codepoint: int) -> str | None:
    special = {
        9: "tab",
        13: "enter",
        27: "escape",
        32: "space",
        127: "backspace",
    }
    if codepoint in special:
        return special[codepoint]
    try:
        return chr(codepoint)
    except ValueError:
        return None


def _parse_kitty_csi_u(data: str) -> tuple[str | None, int | None]:
    match = _KITTY_CSI_U_RE.match(data)
    if not match:
        return None, None

    codepoint = int(match.group("code"))
    base = match.group("base")
    mods = int(match.group("mods") or "1")
    event_type = int(match.group("event") or "1")

    key = _decode_codepoint(codepoint)
    if key is None:
        return None, event_type

    if (len(key) == 1 and key.isascii() and key.isprintable()) or not base:
        chosen = key
    else:
        chosen = _decode_codepoint(int(base)) or key

    return _apply_modifiers(chosen, mods), event_type


def _parse_modify_other_keys(data: str) -> str | None:
    match = _MODIFY_OTHER_KEYS_RE.match(data)
    if not match:
        return None

    mods = int(match.group("mods"))
    codepoint = int(match.group("code"))
    key = _decode_codepoint(codepoint)
    if key is None:
        return None
    return _apply_modifiers(key, mods)


def _parse_csi_modifier_key(data: str) -> str | None:
    if not data.startswith("\x1b[") or len(data) < 5:
        return None
    final = data[-1]
    body = data[2:-1]
    parts = body.split(";")
    if len(parts) != 2:
        return None
    key = _CSI_MODIFIER_KEY_MAP.get((parts[0], int(parts[1]), final))
    return key


def _parse_legacy_ctrl_char(data: str) -> str | None:
    if len(data) != 1:
        return None
    code = ord(data)
    special = {
        0: "ctrl+space",
        8: "ctrl+backspace",
        28: "ctrl+\\",
        29: "ctrl+]",
        31: "ctrl+-",
    }
    if code in special:
        return special[code]
    if 1 <= code <= 26:
        return f"ctrl+{chr(code + 96)}"
    return None


def _parse_alt_prefixed(data: str) -> str | None:
    if len(data) != 2 or not data.startswith("\x1b"):
        return None
    second = data[1]
    if second == "\x1b":
        return "ctrl+alt+["
    if second == "\r":
        return None if _kitty_protocol_active else "alt+enter"
    if second == "\x7f" or second == "\b":
        return "alt+backspace"
    if _kitty_protocol_active:
        return None
    ctrl = _parse_legacy_ctrl_char(second)
    if ctrl:
        return ctrl.replace("ctrl+", "ctrl+alt+", 1)
    if second == " ":
        return "alt+space"
    if second == "B":
        return "alt+left"
    if second == "F":
        return "alt+right"
    if second.isprintable():
        return f"alt+{second}"
    return None


def parse_key(data: str) -> str | None:
    if data == "\r":
        return "enter"
    if data == "\n":
        return "shift+enter" if _kitty_protocol_active else "enter"
    if data == "\t":
        return "tab"
    if data == "\x1b":
        return "escape"
    if data == " ":
        return "space"

    if parsed := _parse_kitty_csi_u(data)[0]:
        return parsed
    if parsed := _parse_modify_other_keys(data):
        return parsed
    if parsed := _parse_csi_modifier_key(data):
        return parsed
    if parsed := _RXVT_MAP.get(data):
        return parsed
    if parsed := _LEGACY_MAP.get(data):
        return parsed
    if parsed := _parse_alt_prefixed(data):
        return parsed
    if parsed := _parse_legacy_ctrl_char(data):
        return parsed
    if len(data) == 1 and data.isprintable():
        return data
    return None


def matches_key(data: str, key_id: str) -> bool:
    parsed = parse_key(data)
    if parsed is None:
        return False
    if key_id in {"enter", "return"}:
        return parsed == "enter"
    if key_id == "shift+enter":
        return parsed == "shift+enter"
    if key_id == "tab":
        return parsed == "tab"
    if key_id in {"escape", "esc"}:
        return parsed == "escape"
    if key_id == "ctrl+_":
        return parsed == "ctrl+-"
    if key_id == "ctrl+alt+_":
        return parsed == "ctrl+alt+-"
    if key_id == "ctrl+h":
        return parsed == "ctrl+backspace"
    return parsed == key_id


def is_key_release(data: str) -> bool:
    _, event_type = _parse_kitty_csi_u(data)
    return event_type == 3


def is_key_repeat(data: str) -> bool:
    _, event_type = _parse_kitty_csi_u(data)
    return event_type == 2


def decode_kitty_printable(data: str) -> str | None:
    parsed, _ = _parse_kitty_csi_u(data)
    if parsed and len(parsed) == 1 and parsed.isprintable():
        return parsed
    return None
