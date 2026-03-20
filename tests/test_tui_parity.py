from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from kitpaw.pi_agent.tui import (
    Input,
    Markdown,
    OverlayOptions,
    SelectItem,
    SelectList,
    SettingItem,
    SettingsList,
    Text,
)

ROOT = Path(__file__).resolve().parent.parent
NODE = "node"
TSX_LOADER = ROOT / "ref/pi-mono/node_modules/tsx/dist/loader.mjs"
SNAPSHOT_SCRIPT = ROOT / "tests/tui_parity/snapshot.ts"
REFERENCE_SCRIPT = ROOT / "tests/tui_parity/reference.ts"


SELECT_THEME = {
    "selected_prefix": lambda text: text,
    "selected_text": lambda text: text,
    "description": lambda text: text,
    "scroll_info": lambda text: text,
    "no_match": lambda text: text,
}

MARKDOWN_THEME = {
    "heading": lambda text: text,
    "link": lambda text: text,
    "link_url": lambda text: text,
    "code": lambda text: text,
    "code_block": lambda text: text,
    "code_block_border": lambda text: text,
    "quote": lambda text: text,
    "quote_border": lambda text: text,
    "hr": lambda text: text,
    "list_bullet": lambda text: text,
    "bold": lambda text: text,
    "italic": lambda text: text,
    "strikethrough": lambda text: text,
    "underline": lambda text: text,
}

STYLED_MARKDOWN_THEME = {
    "heading": lambda text: text,
    "link": lambda text: text,
    "link_url": lambda text: f"\x1b[2m{text}\x1b[0m",
    "code": lambda text: f"\x1b[33m{text}\x1b[0m",
    "code_block": lambda text: text,
    "code_block_border": lambda text: text,
    "quote": lambda text: f"\x1b[3m{text}\x1b[0m",
    "quote_border": lambda text: text,
    "hr": lambda text: text,
    "list_bullet": lambda text: text,
    "bold": lambda text: f"\x1b[1m{text}\x1b[0m",
    "italic": lambda text: f"\x1b[3m{text}\x1b[0m",
    "strikethrough": lambda text: text,
    "underline": lambda text: f"\x1b[4m{text}\x1b[0m",
}

SETTINGS_THEME = {
    "label": lambda text, selected: text,
    "value": lambda text, selected: text,
    "description": lambda text: text,
    "cursor": "→ ",
    "hint": lambda text: text,
}


@dataclass
class RecordingTerminal:
    columns: int = 20
    rows: int = 6
    writes: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.writes = []

    def start(self, on_input, on_resize) -> None:  # noqa: ANN001
        self.on_input = on_input
        self.on_resize = on_resize

    def stop(self) -> None:
        return None

    async def drain_input(self, max_ms: int = 1000, idle_ms: int = 50) -> None:  # noqa: ARG002
        return None

    def write(self, data: str) -> None:
        self.writes.append(data)

    @property
    def kitty_protocol_active(self) -> bool:
        return True

    def move_by(self, lines: int) -> None:
        if lines > 0:
            self.write(f"\x1b[{lines}B")
        elif lines < 0:
            self.write(f"\x1b[{-lines}A")

    def hide_cursor(self) -> None:
        self.write("\x1b[?25l")

    def show_cursor(self) -> None:
        self.write("\x1b[?25h")

    def clear_line(self) -> None:
        self.write("\x1b[K")

    def clear_from_cursor(self) -> None:
        self.write("\x1b[J")

    def clear_screen(self) -> None:
        self.write("\x1b[2J\x1b[H")

    def set_title(self, title: str) -> None:
        self.write(f"\x1b]0;{title}\x07")


def run_node_snapshot(writes: list[str], columns: int = 20, rows: int = 6) -> dict[str, object]:
    payload = {"writes": writes, "columns": columns, "rows": rows}
    result = subprocess.run(
        [NODE, "--import", str(TSX_LOADER), str(SNAPSHOT_SCRIPT)],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        check=True,
        cwd=str(ROOT),
    )
    return json.loads(result.stdout)


def run_node_reference(scenario: str) -> dict[str, object]:
    result = subprocess.run(
        [NODE, "--import", str(TSX_LOADER), str(REFERENCE_SCRIPT), scenario],
        text=True,
        capture_output=True,
        check=True,
        cwd=str(ROOT),
    )
    return json.loads(result.stdout)


def render_python_text() -> dict[str, object]:
    terminal = RecordingTerminal()
    from kitpaw.pi_agent.tui import TUI

    tui = TUI(terminal)
    tui.add_child(Text("hello"))
    tui.start()
    return run_node_snapshot(terminal.writes, terminal.columns, terminal.rows)


def render_python_input() -> dict[str, object]:
    terminal = RecordingTerminal()
    from kitpaw.pi_agent.tui import TUI

    input_widget = Input()
    input_widget.focused = True
    input_widget.set_value("hello")
    tui = TUI(terminal)
    tui.add_child(input_widget)
    tui.start()
    return run_node_snapshot(terminal.writes, terminal.columns, terminal.rows)


def render_python_select_list() -> dict[str, object]:
    terminal = RecordingTerminal(columns=80)
    from kitpaw.pi_agent.tui import TUI

    list_widget = SelectList(
        [
            SelectItem(value="short", label="short", description="short description"),
            SelectItem(
                value="very-long-command-name-that-needs-truncation",
                label="very-long-command-name-that-needs-truncation",
                description="long description",
            ),
        ],
        5,
        SELECT_THEME,
    )
    tui = TUI(terminal)
    tui.add_child(list_widget)
    tui.start()
    return run_node_snapshot(terminal.writes, terminal.columns, terminal.rows)


def render_python_overlay() -> dict[str, object]:
    terminal = RecordingTerminal(columns=20, rows=6)
    from kitpaw.pi_agent.tui import TUI

    tui = TUI(terminal)
    tui.add_child(Text("Line 1\nLine 2\nLine 3"))
    tui.show_overlay(Text("OVERLAY"), OverlayOptions(anchor="center", width=10))
    tui.start()
    return run_node_snapshot(terminal.writes, terminal.columns, terminal.rows)


def render_python_overlay_top_left() -> dict[str, object]:
    terminal = RecordingTerminal(columns=20, rows=6)
    from kitpaw.pi_agent.tui import TUI

    tui = TUI(terminal)
    tui.add_child(Text("base"))
    tui.show_overlay(Text("TOP-LEFT"), OverlayOptions(anchor="top-left", width=10))
    tui.start()
    return run_node_snapshot(terminal.writes, terminal.columns, terminal.rows)


def render_python_overlay_bottom_right() -> dict[str, object]:
    terminal = RecordingTerminal(columns=20, rows=6)
    from kitpaw.pi_agent.tui import TUI

    tui = TUI(terminal)
    tui.add_child(Text("base"))
    tui.show_overlay(Text("BTM-RIGHT"), OverlayOptions(anchor="bottom-right", width=10))
    tui.start()
    return run_node_snapshot(terminal.writes, terminal.columns, terminal.rows)


def render_python_overlay_visible_rule() -> dict[str, object]:
    terminal = RecordingTerminal(columns=20, rows=6)
    from kitpaw.pi_agent.tui import TUI

    tui = TUI(terminal)
    tui.add_child(Text("base"))
    tui.show_overlay(
        Text("HIDDEN"),
        OverlayOptions(anchor="center", width=10, visible=lambda columns, _rows: columns >= 30),
    )
    tui.start()
    return run_node_snapshot(terminal.writes, terminal.columns, terminal.rows)


def render_python_overlay_width_percent_min() -> dict[str, object]:
    terminal = RecordingTerminal(columns=20, rows=6)
    from kitpaw.pi_agent.tui import TUI

    tui = TUI(terminal)
    tui.add_child(Text("base"))
    tui.show_overlay(
        Text("XXXXXXXXXXXXXXX"),
        OverlayOptions(anchor="center", width="50%", min_width=12),
    )
    tui.start()
    return run_node_snapshot(terminal.writes, terminal.columns, terminal.rows)


def render_python_overlay_short_content() -> dict[str, object]:
    terminal = RecordingTerminal(columns=20, rows=6)
    from kitpaw.pi_agent.tui import TUI

    tui = TUI(terminal)
    tui.add_child(Text("Line 1\nLine 2\nLine 3"))
    tui.show_overlay(Text("OVERLAY_TOP\nOVERLAY_MID\nOVERLAY_BOT"))
    tui.start()
    return run_node_snapshot(terminal.writes, terminal.columns, terminal.rows)


def render_python_overlay_style_reset() -> dict[str, object]:
    terminal = RecordingTerminal(columns=20, rows=6)
    from kitpaw.pi_agent.tui import TUI

    component = _MutableComponent(["\x1b[3mXXXXXXXXXXXXXXXXXXXX\x1b[23m", "INPUT"])
    tui = TUI(terminal)
    tui.add_child(component)
    tui.show_overlay(Text("OVR"), OverlayOptions(row=0, col=5, width=3))
    tui.start()
    return run_node_snapshot(terminal.writes, terminal.columns, terminal.rows)


def render_python_markdown_list() -> dict[str, object]:
    terminal = RecordingTerminal(columns=20, rows=6)
    from kitpaw.pi_agent.tui import TUI

    tui = TUI(terminal)
    tui.add_child(
        Markdown(
            "- Item 1\n  - Nested 1.1\n  - Nested 1.2\n- Item 2",
            0,
            0,
            MARKDOWN_THEME,
        )
    )
    tui.start()
    return run_node_snapshot(terminal.writes, terminal.columns, terminal.rows)


def render_python_markdown_explicit_link() -> dict[str, object]:
    terminal = RecordingTerminal(columns=20, rows=6)
    from kitpaw.pi_agent.tui import TUI

    tui = TUI(terminal)
    tui.add_child(
        Markdown(
            "[click here](https://example.com)",
            0,
            0,
            MARKDOWN_THEME,
        )
    )
    tui.start()
    return run_node_snapshot(terminal.writes, terminal.columns, terminal.rows)


def render_python_markdown_blockquote_wrap() -> dict[str, object]:
    terminal = RecordingTerminal(columns=20, rows=6)
    from kitpaw.pi_agent.tui import TUI

    tui = TUI(terminal)
    tui.add_child(
        Markdown(
            "> This is a very long blockquote line that should wrap to multiple lines when rendered",
            0,
            0,
            MARKDOWN_THEME,
        )
    )
    tui.start()
    return run_node_snapshot(terminal.writes, terminal.columns, terminal.rows)


def render_python_markdown_table() -> dict[str, object]:
    terminal = RecordingTerminal(columns=20, rows=6)
    from kitpaw.pi_agent.tui import TUI

    tui = TUI(terminal)
    tui.add_child(
        Markdown(
            "| Name | Value |\n| --- | --- |\n| Foo | Bar |\n| Baz | Qux |",
            0,
            0,
            MARKDOWN_THEME,
        )
    )
    tui.start()
    return run_node_snapshot(terminal.writes, terminal.columns, terminal.rows)


def render_python_markdown_code_fence() -> dict[str, object]:
    terminal = RecordingTerminal(columns=20, rows=6)
    from kitpaw.pi_agent.tui import TUI

    tui = TUI(terminal)
    tui.add_child(
        Markdown(
            "```py\nprint('hello')\nprint('world')\n```",
            0,
            0,
            MARKDOWN_THEME,
        )
    )
    tui.start()
    return run_node_snapshot(terminal.writes, terminal.columns, terminal.rows)


def render_python_markdown_style_reset() -> dict[str, object]:
    terminal = RecordingTerminal(columns=20, rows=6)
    from kitpaw.pi_agent.tui import TUI

    tui = TUI(terminal)
    tui.add_child(
        Markdown(
            "This is thinking with `inline code`",
            1,
            0,
            STYLED_MARKDOWN_THEME,
            {
                "color": lambda text: f"\x1b[90m{text}\x1b[0m",
                "italic": True,
            },
        )
    )
    tui.add_child(Text("INPUT"))
    tui.start()
    return run_node_snapshot(terminal.writes, terminal.columns, terminal.rows)


def render_python_settings_list() -> dict[str, object]:
    terminal = RecordingTerminal(columns=20, rows=6)
    from kitpaw.pi_agent.tui import TUI

    tui = TUI(terminal)
    tui.add_child(
        SettingsList(
            [
                SettingItem(id="theme", label="Theme", current_value="light", description="UI theme", values=["light", "dark"]),
                SettingItem(id="model", label="Model", current_value="gpt-5"),
            ],
            5,
            SETTINGS_THEME,
            lambda _id, _value: None,
            lambda: None,
        )
    )
    tui.start()
    return run_node_snapshot(terminal.writes, terminal.columns, terminal.rows)


class _ParitySubmenu:
    def __init__(self, current_value: str) -> None:
        self.current_value = current_value

    def render(self, width: int) -> list[str]:  # noqa: ARG002
        return [f"submenu:{self.current_value}"]

    def handle_input(self, data: str) -> None:  # noqa: ARG002
        return None

    def invalidate(self) -> None:
        return None


def render_python_settings_list_submenu_open() -> dict[str, object]:
    terminal = RecordingTerminal(columns=20, rows=6)
    from kitpaw.pi_agent.tui import TUI

    settings = SettingsList(
        [
            SettingItem(
                id="theme",
                label="Theme",
                current_value="light",
                submenu=lambda current_value, done: _ParitySubmenu(current_value),
            )
        ],
        5,
        SETTINGS_THEME,
        lambda _id, _value: None,
        lambda: None,
    )
    settings.handle_input("\r")

    tui = TUI(terminal)
    tui.add_child(settings)
    tui.start()
    return run_node_snapshot(terminal.writes, terminal.columns, terminal.rows)


class _MutableComponent:
    def __init__(self, lines: list[str]) -> None:
        self.lines = lines

    def render(self, _width: int) -> list[str]:
        return list(self.lines)

    def handle_input(self, _data: str) -> None:
        return None

    def invalidate(self) -> None:
        return None


def render_python_tui_middle_line_change() -> dict[str, object]:
    terminal = RecordingTerminal(columns=20, rows=6)
    from kitpaw.pi_agent.tui import TUI

    component = _MutableComponent(["Header", "Working...", "Footer"])
    tui = TUI(terminal)
    tui.add_child(component)
    tui.start()
    component.lines = ["Header", "Working /", "Footer"]
    tui.request_render()
    return run_node_snapshot(terminal.writes, terminal.columns, terminal.rows)


def render_python_tui_clear_then_content() -> dict[str, object]:
    terminal = RecordingTerminal(columns=20, rows=6)
    from kitpaw.pi_agent.tui import TUI

    component = _MutableComponent(["Line 0", "Line 1", "Line 2"])
    tui = TUI(terminal)
    tui.add_child(component)
    tui.start()
    component.lines = []
    tui.request_render()
    component.lines = ["New Line 0", "New Line 1"]
    tui.request_render()
    return run_node_snapshot(terminal.writes, terminal.columns, terminal.rows)


def render_python_tui_style_reset() -> dict[str, object]:
    terminal = RecordingTerminal(columns=20, rows=6)
    from kitpaw.pi_agent.tui import TUI

    component = _MutableComponent(["\x1b[3mItalic", "Plain"])
    tui = TUI(terminal)
    tui.add_child(component)
    tui.start()
    return run_node_snapshot(terminal.writes, terminal.columns, terminal.rows)


def test_parity_text_basic() -> None:
    python_snapshot = render_python_text()
    ts_snapshot = run_node_reference("text-basic")
    assert python_snapshot["viewport"] == ts_snapshot["viewport"]
    assert python_snapshot["cursor"] == ts_snapshot["cursor"]


def test_parity_input_focus() -> None:
    python_snapshot = render_python_input()
    ts_snapshot = run_node_reference("input-focus")
    assert python_snapshot["viewport"] == ts_snapshot["viewport"]
    assert python_snapshot["cursor"] == ts_snapshot["cursor"]


def test_parity_select_list_basic() -> None:
    python_snapshot = render_python_select_list()
    ts_snapshot = run_node_reference("select-list-basic")
    assert python_snapshot["viewport"] == ts_snapshot["viewport"]


def test_parity_overlay_centered() -> None:
    python_snapshot = render_python_overlay()
    ts_snapshot = run_node_reference("overlay-centered")
    assert python_snapshot["viewport"] == ts_snapshot["viewport"]


def test_parity_overlay_top_left() -> None:
    python_snapshot = render_python_overlay_top_left()
    ts_snapshot = run_node_reference("overlay-top-left")
    assert python_snapshot["viewport"] == ts_snapshot["viewport"]


def test_parity_overlay_bottom_right() -> None:
    python_snapshot = render_python_overlay_bottom_right()
    ts_snapshot = run_node_reference("overlay-bottom-right")
    assert python_snapshot["viewport"] == ts_snapshot["viewport"]


def test_parity_overlay_visible_rule() -> None:
    python_snapshot = render_python_overlay_visible_rule()
    ts_snapshot = run_node_reference("overlay-visible-rule")
    assert python_snapshot["viewport"] == ts_snapshot["viewport"]


def test_parity_overlay_width_percent_min() -> None:
    python_snapshot = render_python_overlay_width_percent_min()
    ts_snapshot = run_node_reference("overlay-width-percent-min")
    assert python_snapshot["viewport"] == ts_snapshot["viewport"]


def test_parity_overlay_short_content() -> None:
    python_snapshot = render_python_overlay_short_content()
    ts_snapshot = run_node_reference("overlay-short-content")
    assert python_snapshot["viewport"] == ts_snapshot["viewport"]


def test_parity_overlay_style_reset() -> None:
    python_snapshot = render_python_overlay_style_reset()
    ts_snapshot = run_node_reference("overlay-style-reset")
    assert python_snapshot["viewport"] == ts_snapshot["viewport"]
    assert python_snapshot["italic"] == ts_snapshot["italic"]


def test_parity_markdown_list() -> None:
    python_snapshot = render_python_markdown_list()
    ts_snapshot = run_node_reference("markdown-list")
    assert python_snapshot["viewport"] == ts_snapshot["viewport"]


def test_parity_markdown_explicit_link() -> None:
    python_snapshot = render_python_markdown_explicit_link()
    ts_snapshot = run_node_reference("markdown-explicit-link")
    assert python_snapshot["viewport"] == ts_snapshot["viewport"]


def test_parity_markdown_blockquote_wrap() -> None:
    python_snapshot = render_python_markdown_blockquote_wrap()
    ts_snapshot = run_node_reference("markdown-blockquote-wrap")
    assert python_snapshot["viewport"] == ts_snapshot["viewport"]


def test_parity_markdown_table() -> None:
    python_snapshot = render_python_markdown_table()
    ts_snapshot = run_node_reference("markdown-table")
    assert python_snapshot["viewport"] == ts_snapshot["viewport"]


def test_parity_markdown_code_fence() -> None:
    python_snapshot = render_python_markdown_code_fence()
    ts_snapshot = run_node_reference("markdown-code-fence")
    assert python_snapshot["viewport"] == ts_snapshot["viewport"]


def test_parity_markdown_style_reset() -> None:
    python_snapshot = render_python_markdown_style_reset()
    ts_snapshot = run_node_reference("markdown-style-reset")
    assert python_snapshot["viewport"] == ts_snapshot["viewport"]
    assert python_snapshot["italic"] == ts_snapshot["italic"]


def test_parity_settings_list_basic() -> None:
    python_snapshot = render_python_settings_list()
    ts_snapshot = run_node_reference("settings-list-basic")
    assert python_snapshot["viewport"] == ts_snapshot["viewport"]


def test_parity_settings_list_submenu_open() -> None:
    python_snapshot = render_python_settings_list_submenu_open()
    ts_snapshot = run_node_reference("settings-list-submenu-open")
    assert python_snapshot["viewport"] == ts_snapshot["viewport"]


def test_parity_tui_middle_line_change() -> None:
    python_snapshot = render_python_tui_middle_line_change()
    ts_snapshot = run_node_reference("tui-middle-line-change")
    assert python_snapshot["viewport"] == ts_snapshot["viewport"]


def test_parity_tui_clear_then_content() -> None:
    python_snapshot = render_python_tui_clear_then_content()
    ts_snapshot = run_node_reference("tui-clear-then-content")
    assert python_snapshot["viewport"] == ts_snapshot["viewport"]


def test_parity_tui_style_reset() -> None:
    python_snapshot = render_python_tui_style_reset()
    ts_snapshot = run_node_reference("tui-style-reset")
    assert python_snapshot["viewport"] == ts_snapshot["viewport"]
    assert python_snapshot["italic"] == ts_snapshot["italic"]
