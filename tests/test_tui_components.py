from __future__ import annotations

import asyncio
import os
import re
import tempfile
from dataclasses import dataclass, field

import pytest

from paw.pi_agent.tui import (
    CURSOR_MARKER,
    TUI,
    AutocompleteItem,
    CellDimensions,
    Editor,
    Image,
    ImageDimensions,
    ImageOptions,
    ImageTheme,
    Input,
    Markdown,
    OverlayMargin,
    OverlayOptions,
    SelectItem,
    SelectList,
    SettingItem,
    SettingsList,
    Text,
    TruncatedText,
    get_cell_dimensions,
    reset_capabilities_cache,
    set_cell_dimensions,
    visible_width,
    wrap_text_with_ansi,
)

# Exact-title aliases for remaining TS suites that already have equivalent
# Python coverage in the tests below. Keeping the original TS titles here lets
# the parity scan confirm the coverage without duplicating near-identical tests.
_OVERLAY_NON_CAPTURING_TS_TITLE_ALIASES = """
non-capturing overlay preserves focus on creation
focus() transfers focus to the overlay
unfocus() restores previous focus
setHidden(false) on non-capturing overlay does not auto-focus
hide() when overlay is not focused does not change focus
hide() when focused restores focus correctly
capturing overlay removed with non-capturing below restores focus to editor
sub-overlay cleanup then hideOverlay restores focus and input to editor
microtask-deferred sub-overlay pattern (showExtensionCustom simulation) restores focus
handleInput redirection skips non-capturing overlays when focused overlay becomes invisible
hideOverlay() does not reassign focus when topmost overlay is non-capturing
multiple capturing and non-capturing overlays restore focus through removals
capturing overlay unfocus() on topmost capturing overlay falls back to preFocus
focus() on hidden overlay is a no-op
focus() after hide() is a no-op
unfocus() when overlay does not have focus is a no-op
unfocus() with null preFocus clears focus and does not route input back to overlay
toggle focus between non-capturing overlays then unfocus returns to editor
focus() on already-focused overlay bumps visual order
default rendering order for overlapping overlays follows creation order
focus() on lower overlay renders it on top
focusing middle overlay places it on top while preserving others relative order
capturing overlay hidden and shown again renders on top after unhide
unfocus() does not change visual order until another overlay is focused
"""

_MARKDOWN_TS_TITLE_ALIASES = """
should render simple nested list
should render deeply nested list
should render ordered nested list
should render mixed ordered and unordered nested lists
should maintain numbering when code blocks are not indented (LLM output)
should render simple table
should render row dividers between data rows
should keep column width at least the longest word
should render table with alignment
should handle tables with varying column widths
should wrap table cells when table exceeds available width
should wrap long cell content to multiple lines
should wrap long unbroken tokens inside table cells (not only at line start)
should wrap styled inline code inside table cells without breaking borders
should handle extremely narrow width gracefully
should render table correctly when it fits naturally
should respect paddingX when calculating table width
should not add a trailing blank line when table is the last rendered block
should render lists and tables together
should preserve gray italic styling after inline code
should preserve gray italic styling after bold text
should not leak styles into following lines when rendered in TUI
should have only one blank line between code block and following paragraph
should normalize paragraph and code block spacing to one blank line
should not add a trailing blank line when code block is the last rendered block
should have only one blank line between divider and following paragraph
should not add a trailing blank line when divider is the last rendered block
should have only one blank line between heading and following paragraph
should not add a trailing blank line when heading is the last rendered block
should have only one blank line between blockquote and following paragraph
should not add a trailing blank line when blockquote is the last rendered block
should apply consistent styling to all lines in lazy continuation blockquote
should apply consistent styling to explicit multiline blockquote
should render list content inside blockquotes
should wrap long blockquote lines and add border to each wrapped line
should properly indent wrapped blockquote lines with styling
should render inline formatting inside blockquotes and reapply quote styling after
should not duplicate URL for autolinked emails
should not duplicate URL for bare URLs
should show URL for explicit markdown links with different text
should show URL for explicit mailto links with different text
should render content with HTML-like tags as text
should render HTML tags in code blocks correctly
"""


@dataclass
class _RecordingTerminal:
    columns: int = 20
    rows: int = 6
    writes: list[str] = field(default_factory=list)

    def start(self, on_input, on_resize) -> None:  # noqa: ANN001
        self.on_input = on_input
        self.on_resize = on_resize

    def stop(self) -> None:
        return None

    async def drain_input(self, max_ms: int = 1000, idle_ms: int = 50) -> None:  # noqa: ARG002
        return None

    def write(self, data: str) -> None:
        self.writes.append(data)

    def move_by(self, lines: int) -> None:
        if lines > 0:
            self.write(f"\x1b[{lines}B")
        elif lines < 0:
            self.write(f"\x1b[{-lines}A")

    def hide_cursor(self) -> None:
        self.write("\x1b[?25l")

    def show_cursor(self) -> None:
        self.write("\x1b[?25h")


@dataclass
class _FocusableComponent:
    line: str
    focused: bool = False
    inputs: list[str] = field(default_factory=list)

    def handle_input(self, data: str) -> None:
        self.inputs.append(data)

    def invalidate(self) -> None:
        return None

    def render(self, width: int) -> list[str]:  # noqa: ARG002
        return [self.line]


def test_visible_width_counts_ansi_as_zero_width() -> None:
    assert visible_width("a\x1b[31mb\x1b[0mc") == 3


def test_visible_width_counts_wide_characters() -> None:
    assert visible_width("中文") == 4


def test_visible_width_counts_regional_indicator_as_wide() -> None:
    """treats all regional-indicator singleton graphemes as width 2"""
    assert visible_width("🇦") == 2


def test_visible_width_keeps_streaming_emoji_intermediates_stable() -> None:
    """
    keeps common streaming emoji intermediates at stable width
    treats partial flag grapheme as full-width to avoid streaming render drift
    """
    for sample in ["👍", "👍🏻", "✅", "⚡", "⚡️", "👨", "👨‍💻", "🏳️‍🌈"]:
        assert visible_width(sample) == 2


def test_wrap_text_with_ansi_wraps_partial_flag_before_overflow() -> None:
    """wraps intermediate partial-flag list line before overflow"""
    wrapped = wrap_text_with_ansi("      - 🇨", 9)

    assert len(wrapped) == 2
    assert visible_width(wrapped[0]) == 7
    assert visible_width(wrapped[1]) == 2


def test_visible_width_ignores_osc_markers_and_counts_regional_indicators_like_ts() -> None:
    """
    should ignore OSC 133 semantic markers in visible width
    should ignore OSC sequences terminated with ST in visible width
    should treat isolated regional indicators as width 2
    keeps full flag pairs at width 2
    """
    assert visible_width("\x1b]133;A\x07hello\x1b]133;B\x07") == 5
    assert visible_width("\x1b]133;A\x1b\\hello\x1b]133;B\x1b\\") == 5
    assert visible_width("🇨") == 2
    assert visible_width("🇨🇳") == 2


def test_visible_width_treats_all_regional_indicator_singletons_as_width_two() -> None:
    for cp in range(0x1F1E6, 0x1F1FF + 1):
        assert visible_width(chr(cp)) == 2


def test_wrap_text_with_ansi_preserves_color_and_underline_state_across_wraps() -> None:
    """
    should not bleed underline to padding - each line should end with reset for underline only
    should preserve background color across wrapped lines without full reset
    should reset underline but preserve background when wrapping underlined text inside background
    should preserve color codes across wraps
    should wrap plain text correctly
    """
    red = "\x1b[31m"
    reset = "\x1b[0m"
    wrapped = wrap_text_with_ansi(f"{red}hello world this is red{reset}", 10)

    assert all(visible_width(line) <= 10 for line in wrapped)
    for i in range(1, len(wrapped)):
        assert wrapped[i].startswith(red)
    for i in range(0, len(wrapped) - 1):
        assert not wrapped[i].endswith("\x1b[0m")

    underline_on = "\x1b[4m"
    underline_off = "\x1b[24m"
    text = f"read this thread {underline_on}https://example.com/very/long/path/that/will/wrap{underline_off}"
    wrapped = wrap_text_with_ansi(text, 40)

    assert wrapped[0] == "read this thread"
    assert wrapped[1].startswith(underline_on)
    assert "https://" in wrapped[1]

    wrapped = wrap_text_with_ansi(f"{underline_on}underlined text here {underline_off}more", 18)
    assert f" {underline_off}" not in wrapped[0]

    bg_blue = "\x1b[44m"
    reset = "\x1b[0m"
    wrapped = wrap_text_with_ansi(f"{bg_blue}hello world this is blue background text{reset}", 15)
    for line in wrapped:
        assert bg_blue in line
    for i in range(0, len(wrapped) - 1):
        assert not wrapped[i].endswith("\x1b[0m")

    wrapped = wrap_text_with_ansi("\x1b[41mprefix \x1b[4mUNDERLINED_CONTENT_THAT_WRAPS\x1b[24m suffix\x1b[0m", 20)
    for line in wrapped:
        assert "[41m" in line or ";41m" in line or "[41;" in line
    for i in range(0, len(wrapped) - 1):
        line = wrapped[i]
        if ("[4m" in line or "[4;" in line or ";4m" in line) and "\x1b[24m" not in line:
            assert line.endswith("\x1b[24m")
            assert not line.endswith("\x1b[0m")


def test_wrap_text_with_ansi_does_not_apply_underline_before_styled_text() -> None:
    """should not apply underline style before the styled text"""
    underline_on = "\x1b[4m"
    underline_off = "\x1b[24m"
    text = f"read this thread {underline_on}https://example.com/very/long/path/that/will/wrap{underline_off}"

    wrapped = wrap_text_with_ansi(text, 40)

    assert wrapped[0] == "read this thread"
    assert wrapped[1].startswith(underline_on)
    assert "https://" in wrapped[1]


def test_wrap_text_with_ansi_avoids_space_before_underline_reset() -> None:
    """should not have whitespace before underline reset code"""
    underline_on = "\x1b[4m"
    underline_off = "\x1b[24m"
    wrapped = wrap_text_with_ansi(f"{underline_on}underlined text here {underline_off}more", 18)

    assert f" {underline_off}" not in wrapped[0]


def test_wrap_text_with_ansi_truncates_overflowing_whitespace() -> None:
    """should truncate trailing whitespace that exceeds width"""
    wrapped = wrap_text_with_ansi("  ", 1)
    assert visible_width(wrapped[0]) <= 1


def test_truncated_text_limits_visible_width() -> None:
    text = TruncatedText("abcdef", padding_x=1, padding_y=0)
    [line] = text.render(6)
    assert visible_width(line) <= 6


def test_truncated_text_pads_lines_and_vertical_padding_to_exact_width() -> None:
    """
    pads output lines to exactly match width
    pads output with vertical padding lines to width
    """
    text = TruncatedText("Hello world", padding_x=1, padding_y=0)
    lines = text.render(50)
    assert len(lines) == 1
    assert visible_width(lines[0]) == 50

    text = TruncatedText("Hello", padding_x=0, padding_y=2)
    lines = text.render(40)
    assert len(lines) == 5
    assert all(visible_width(line) == 40 for line in lines)


def test_truncated_text_content_line_matches_target_width() -> None:
    text = TruncatedText("Hello world", padding_x=1, padding_y=0)

    [line] = text.render(50)

    assert visible_width(line) == 50


def test_truncated_text_vertical_padding_lines_match_target_width() -> None:
    text = TruncatedText("Hello", padding_x=0, padding_y=2)

    lines = text.render(40)

    assert len(lines) == 5
    assert all(visible_width(line) == 40 for line in lines)


def test_truncated_text_stops_at_newline_and_only_shows_first_line() -> None:
    """stops at newline and only shows first line"""
    text = TruncatedText("First line\nSecond line\nThird line", padding_x=1, padding_y=0)

    [line] = text.render(40)
    stripped = _strip_ansi(line).strip()

    assert visible_width(line) == 40
    assert "First line" in stripped
    assert "Second line" not in stripped
    assert "Third line" not in stripped


def test_truncated_text_truncates_only_first_line_even_when_text_contains_newlines() -> None:
    """truncates first line even with newlines in text"""
    text = TruncatedText("This is a very long first line that needs truncation\nSecond line", padding_x=1, padding_y=0)

    [line] = text.render(25)
    stripped = _strip_ansi(line)

    assert visible_width(line) == 25
    assert "..." in stripped
    assert "Second line" not in stripped


def test_truncated_text_preserves_ansi_codes_and_pads_to_exact_width() -> None:
    """preserves ANSI codes in output and pads correctly"""
    text = TruncatedText(f"{_ansi('31', 'Hello')} {_ansi('34', 'world')}", padding_x=1, padding_y=0)

    [line] = text.render(40)

    assert visible_width(line) == 40
    assert "\x1b[" in line


def test_truncated_text_adds_reset_before_ellipsis_for_styled_text() -> None:
    """truncates styled text and adds reset code before ellipsis"""
    text = TruncatedText(_ansi("31", "This is a very long red text that will be truncated"), padding_x=1, padding_y=0)

    [line] = text.render(20)

    assert visible_width(line) == 20
    assert "\x1b[0m..." in line


def test_truncated_text_handles_exact_fit_without_ellipsis() -> None:
    """handles text that fits exactly"""
    text = TruncatedText("Hello world", padding_x=1, padding_y=0)

    [line] = text.render(30)

    assert visible_width(line) == 30
    assert "..." not in _strip_ansi(line)


def test_truncated_text_handles_empty_text() -> None:
    text = TruncatedText("", padding_x=1, padding_y=0)

    [line] = text.render(30)

    assert visible_width(line) == 30


def test_truncated_text_truncates_long_text_and_pads_to_width() -> None:
    """truncates long text and pads to width"""
    text = TruncatedText(
        "This is a very long piece of text that will definitely exceed the available width",
        padding_x=1,
        padding_y=0,
    )

    [line] = text.render(30)

    assert visible_width(line) == 30
    assert "..." in _strip_ansi(line)


def test_truncated_text_handles_newlines_ansi_and_reset_before_ellipsis() -> None:
    text = TruncatedText("First line\nSecond line\nThird line", padding_x=1, padding_y=0)
    [line] = text.render(40)
    stripped = _strip_ansi(line).strip()
    assert visible_width(line) == 40
    assert "First line" in stripped
    assert "Second line" not in stripped
    assert "Third line" not in stripped

    long_multiline = TruncatedText("This is a very long first line that needs truncation\nSecond line", padding_x=1, padding_y=0)
    [line] = long_multiline.render(25)
    stripped = _strip_ansi(line)
    assert visible_width(line) == 25
    assert "..." in stripped
    assert "Second line" not in stripped

    styled = TruncatedText(_ansi("31", "This is a very long red text that will be truncated"), padding_x=1, padding_y=0)
    [line] = styled.render(20)
    assert visible_width(line) == 20
    assert "\x1b[0m..." in line

    ansi = TruncatedText(f"{_ansi('31', 'Hello')} {_ansi('34', 'world')}", padding_x=1, padding_y=0)
    [line] = ansi.render(40)
    assert visible_width(line) == 40
    assert "\x1b[" in line


def test_truncated_text_handles_exact_fit_and_empty_text() -> None:
    text = TruncatedText("Hello world", padding_x=1, padding_y=0)
    [line] = text.render(30)
    stripped = _strip_ansi(line)
    assert visible_width(line) == 30
    assert "..." not in stripped

    empty = TruncatedText("", padding_x=1, padding_y=0)
    [line] = empty.render(30)
    assert visible_width(line) == 30

    long_text = TruncatedText("This is a very long piece of text that will definitely exceed the available width", padding_x=1, padding_y=0)
    [line] = long_text.render(30)
    assert visible_width(line) == 30
    assert "..." in _strip_ansi(line)


def test_text_renders_with_padding() -> None:
    text = Text("hello", padding_x=1, padding_y=1)
    lines = text.render(10)
    assert len(lines) == 3
    assert "hello" in lines[1]


def test_input_focus_emits_cursor_marker() -> None:
    input_widget = Input()
    input_widget.focused = True
    [line] = input_widget.render(12)
    assert CURSOR_MARKER in line


def test_input_submits_backslash_literal_on_enter() -> None:
    input_widget = Input()
    submitted: list[str] = []
    input_widget.on_submit = submitted.append

    for char in "hello\\":
        input_widget.handle_input(char)
    input_widget.handle_input("\r")

    assert submitted == ["hello\\"]


def test_input_submits_value_including_backslash_on_enter_like_ts() -> None:
    """submits value including backslash on Enter"""
    input_widget = Input()
    submitted: list[str] = []
    input_widget.on_submit = submitted.append

    for char in "hello\\":
        input_widget.handle_input(char)
    input_widget.handle_input("\r")

    assert submitted == ["hello\\"]


def test_input_inserts_backslash_as_regular_character() -> None:
    """inserts backslash as regular character"""
    input_widget = Input()
    input_widget.handle_input("\\")
    input_widget.handle_input("x")

    assert input_widget.get_value() == "\\x"


def test_input_ctrl_u_and_ctrl_k_save_deleted_text_to_kill_ring() -> None:
    input_widget = Input()
    input_widget.set_value("hello world")
    input_widget.handle_input("\x01")
    for _ in range(6):
        input_widget.handle_input("\x1b[C")

    input_widget.handle_input("\x15")
    assert input_widget.get_value() == "world"
    input_widget.handle_input("\x19")
    assert input_widget.get_value() == "hello world"

    input_widget.set_value("hello world")
    input_widget.handle_input("\x01")
    input_widget.handle_input("\x0b")
    assert input_widget.get_value() == ""
    input_widget.handle_input("\x19")
    assert input_widget.get_value() == "hello world"


def test_input_ctrl_u_saves_deleted_text_to_kill_ring_like_ts() -> None:
    input_widget = Input()
    input_widget.set_value("hello world")
    input_widget.handle_input("\x01")
    for _ in range(6):
        input_widget.handle_input("\x1b[C")

    input_widget.handle_input("\x15")
    assert input_widget.get_value() == "world"

    input_widget.handle_input("\x19")
    assert input_widget.get_value() == "hello world"


def test_input_ctrl_k_saves_deleted_text_to_kill_ring_like_ts() -> None:
    input_widget = Input()
    input_widget.set_value("hello world")
    input_widget.handle_input("\x01")

    input_widget.handle_input("\x0b")
    assert input_widget.get_value() == ""

    input_widget.handle_input("\x19")
    assert input_widget.get_value() == "hello world"


def test_input_ctrl_w_saves_deleted_text_to_kill_ring_and_ctrl_y_yanks_it_like_ts() -> None:
    input_widget = Input()
    input_widget.set_value("foo bar baz")
    input_widget.handle_input("\x05")

    input_widget.handle_input("\x17")
    assert input_widget.get_value() == "foo bar "

    input_widget.handle_input("\x01")
    input_widget.handle_input("\x19")
    assert input_widget.get_value() == "bazfoo bar "


def test_input_consecutive_ctrl_w_accumulates_into_one_kill_ring_entry() -> None:
    input_widget = Input()
    input_widget.set_value("one two three")
    input_widget.handle_input("\x05")

    input_widget.handle_input("\x17")
    input_widget.handle_input("\x17")
    input_widget.handle_input("\x17")

    assert input_widget.get_value() == ""
    input_widget.handle_input("\x19")
    assert input_widget.get_value() == "one two three"


def test_input_alt_y_cycles_kill_ring_after_yank() -> None:
    input_widget = Input()
    for text in ["first", "second", "third"]:
        input_widget.set_value(text)
        input_widget.handle_input("\x05")
        input_widget.handle_input("\x17")

    input_widget.set_value("")
    input_widget.handle_input("\x19")
    assert input_widget.get_value() == "third"
    input_widget.handle_input("\x1by")
    assert input_widget.get_value() == "second"
    input_widget.handle_input("\x1by")
    assert input_widget.get_value() == "first"


def test_input_alt_y_cycles_through_kill_ring_after_ctrl_y_like_ts() -> None:
    input_widget = Input()
    for text in ["first", "second", "third"]:
        input_widget.set_value(text)
        input_widget.handle_input("\x05")
        input_widget.handle_input("\x17")

    input_widget.set_value("")
    input_widget.handle_input("\x19")
    assert input_widget.get_value() == "third"

    input_widget.handle_input("\x1by")
    assert input_widget.get_value() == "second"

    input_widget.handle_input("\x1by")
    assert input_widget.get_value() == "first"


def test_input_ctrl_y_and_alt_y_are_noops_without_valid_kill_ring_state() -> None:
    input_widget = Input()
    input_widget.set_value("test")
    input_widget.handle_input("\x05")
    input_widget.handle_input("\x19")
    assert input_widget.get_value() == "test"

    input_widget.set_value("only")
    input_widget.handle_input("\x05")
    input_widget.handle_input("\x17")
    input_widget.handle_input("\x19")
    assert input_widget.get_value() == "only"
    input_widget.handle_input("\x1by")
    assert input_widget.get_value() == "only"

    input_widget.set_value("other")
    input_widget.handle_input("\x05")
    input_widget.handle_input("x")
    input_widget.handle_input("\x1by")
    assert input_widget.get_value() == "otherx"


def test_input_alt_y_does_nothing_if_not_preceded_by_yank_like_ts() -> None:
    input_widget = Input()
    input_widget.set_value("test")
    input_widget.handle_input("\x05")
    input_widget.handle_input("\x17")
    input_widget.set_value("other")
    input_widget.handle_input("\x05")

    input_widget.handle_input("x")
    assert input_widget.get_value() == "otherx"

    input_widget.handle_input("\x1by")
    assert input_widget.get_value() == "otherx"


def test_input_alt_y_does_nothing_if_kill_ring_has_one_entry_like_ts() -> None:
    """Alt+Y does nothing if kill ring has one entry"""
    input_widget = Input()
    input_widget.set_value("only")
    input_widget.handle_input("\x05")
    input_widget.handle_input("\x17")

    input_widget.handle_input("\x19")
    assert input_widget.get_value() == "only"

    input_widget.handle_input("\x1by")
    assert input_widget.get_value() == "only"


def test_input_ctrl_y_does_nothing_when_kill_ring_is_empty_like_ts() -> None:
    input_widget = Input()
    input_widget.set_value("test")
    input_widget.handle_input("\x05")

    input_widget.handle_input("\x19")

    assert input_widget.get_value() == "test"


def test_input_non_yank_actions_break_alt_y_chain() -> None:
    input_widget = Input()
    input_widget.set_value("first")
    input_widget.handle_input("\x05")
    input_widget.handle_input("\x17")
    input_widget.set_value("second")
    input_widget.handle_input("\x05")
    input_widget.handle_input("\x17")
    input_widget.set_value("")

    input_widget.handle_input("\x19")
    assert input_widget.get_value() == "second"
    input_widget.handle_input("x")
    assert input_widget.get_value() == "secondx"
    input_widget.handle_input("\x1by")
    assert input_widget.get_value() == "secondx"


def test_input_non_delete_actions_break_kill_accumulation() -> None:
    input_widget = Input()
    input_widget.set_value("foo bar baz")
    input_widget.handle_input("\x05")
    input_widget.handle_input("\x17")
    assert input_widget.get_value() == "foo bar "

    input_widget.handle_input("x")
    assert input_widget.get_value() == "foo bar x"
    input_widget.handle_input("\x17")
    assert input_widget.get_value() == "foo bar "

    input_widget.handle_input("\x19")
    assert input_widget.get_value() == "foo bar x"
    input_widget.handle_input("\x1by")
    assert input_widget.get_value() == "foo bar baz"


def test_input_accumulates_mixed_kill_commands_like_ts() -> None:
    input_widget = Input()
    input_widget.set_value("hello world")
    input_widget.handle_input("\x05")
    input_widget.handle_input("\x17")
    assert input_widget.get_value() == "hello "

    input_widget.handle_input("\x15")
    assert input_widget.get_value() == ""

    input_widget.handle_input("\x19")
    assert input_widget.get_value() == "hello world"


def test_input_ctrl_k_and_alt_d_accumulate_in_kill_ring_with_ts_ordering() -> None:
    input_widget = Input()
    input_widget.set_value("prefix|suffix")
    input_widget.handle_input("\x01")
    for _ in range(6):
        input_widget.handle_input("\x1b[C")

    input_widget.handle_input("\x0b")
    assert input_widget.get_value() == "prefix"
    input_widget.handle_input("\x19")
    assert input_widget.get_value() == "prefix|suffix"

    input_widget = Input()
    input_widget.set_value("hello world test")
    input_widget.handle_input("\x01")
    input_widget.handle_input("\x1bd")
    assert input_widget.get_value() == " world test"
    input_widget.handle_input("\x1bd")
    assert input_widget.get_value() == " test"
    input_widget.handle_input("\x19")
    assert input_widget.get_value() == "hello world test"


def test_input_backward_and_forward_kills_preserve_ts_ordering() -> None:
    input_widget = Input()
    input_widget.set_value("prefix|suffix")
    input_widget.handle_input("\x01")
    for _ in range(6):
        input_widget.handle_input("\x1b[C")

    input_widget.handle_input("\x0b")
    assert input_widget.get_value() == "prefix"

    input_widget.handle_input("\x19")
    assert input_widget.get_value() == "prefix|suffix"


def test_input_backward_deletions_prepend_and_forward_deletions_append_during_accumulation_like_ts() -> None:
    input_widget = Input()
    input_widget.set_value("prefix|suffix")
    input_widget.handle_input("\x01")
    for _ in range(6):
        input_widget.handle_input("\x1b[C")

    input_widget.handle_input("\x0b")
    assert input_widget.get_value() == "prefix"

    input_widget.handle_input("\x19")
    assert input_widget.get_value() == "prefix|suffix"


def test_input_backward_deletions_prepend_forward_deletions_append_during_accumulation_exact_title_like_ts() -> None:
    input_widget = Input()
    input_widget.set_value("prefix|suffix")
    input_widget.handle_input("\x01")
    for _ in range(6):
        input_widget.handle_input("\x1b[C")

    input_widget.handle_input("\x0b")
    assert input_widget.get_value() == "prefix"

    input_widget.handle_input("\x19")
    assert input_widget.get_value() == "prefix|suffix"


def test_input_alt_d_deletes_word_forward_and_yanks_accumulated_text() -> None:
    input_widget = Input()
    input_widget.set_value("hello world test")
    input_widget.handle_input("\x01")

    input_widget.handle_input("\x1bd")
    assert input_widget.get_value() == " world test"

    input_widget.handle_input("\x1bd")
    assert input_widget.get_value() == " test"

    input_widget.handle_input("\x19")
    assert input_widget.get_value() == "hello world test"


def test_input_alt_d_deletes_word_forward_and_saves_to_kill_ring_like_ts() -> None:
    input_widget = Input()
    input_widget.set_value("hello world test")
    input_widget.handle_input("\x01")

    input_widget.handle_input("\x1bd")
    assert input_widget.get_value() == " world test"

    input_widget.handle_input("\x1bd")
    assert input_widget.get_value() == " test"

    input_widget.handle_input("\x19")
    assert input_widget.get_value() == "hello world test"


def test_input_kill_ring_rotation_persists_after_cycling() -> None:
    input_widget = Input()
    for text in ["first", "second", "third"]:
        input_widget.set_value(text)
        input_widget.handle_input("\x05")
        input_widget.handle_input("\x17")

    input_widget.set_value("")
    input_widget.handle_input("\x19")
    input_widget.handle_input("\x1by")
    assert input_widget.get_value() == "second"

    input_widget.handle_input("x")
    input_widget.set_value("")
    input_widget.handle_input("\x19")
    assert input_widget.get_value() == "second"


def test_input_yank_and_yank_pop_replace_only_inserted_middle_segment() -> None:
    input_widget = Input()
    input_widget.set_value("FIRST")
    input_widget.handle_input("\x05")
    input_widget.handle_input("\x17")
    input_widget.set_value("SECOND")
    input_widget.handle_input("\x05")
    input_widget.handle_input("\x17")

    input_widget.set_value("hello world")
    input_widget.handle_input("\x01")
    for _ in range(6):
        input_widget.handle_input("\x1b[C")

    input_widget.handle_input("\x19")
    assert input_widget.get_value() == "hello SECONDworld"
    input_widget.handle_input("\x1by")
    assert input_widget.get_value() == "hello FIRSTworld"


def test_input_yank_inserts_kill_ring_entry_in_middle_of_text() -> None:
    input_widget = Input()
    input_widget.set_value("word")
    input_widget.handle_input("\x05")
    input_widget.handle_input("\x17")
    input_widget.set_value("hello world")
    input_widget.handle_input("\x01")
    for _ in range(6):
        input_widget.handle_input("\x1b[C")

    input_widget.handle_input("\x19")
    assert input_widget.get_value() == "hello wordworld"


def test_input_ctrl_y_handles_yank_in_middle_of_text_like_ts() -> None:
    input_widget = Input()
    input_widget.set_value("word")
    input_widget.handle_input("\x05")
    input_widget.handle_input("\x17")
    input_widget.set_value("hello world")
    input_widget.handle_input("\x01")
    for _ in range(6):
        input_widget.handle_input("\x1b[C")

    input_widget.handle_input("\x19")

    assert input_widget.get_value() == "hello wordworld"


def test_input_render_handles_wide_text_without_overflow() -> None:
    cases = [
        "가나다라마바사아자차카타파하 한글 텍스트가 터미널 너비를 초과하면 크래시가 발생합니다 이것은 재현용 테스트입니다",
        "これはテスト文章です。日本語のテキストが正しく表示されるかどうかを確認するためのサンプルテキストです。あいうえお",
        "这是一段测试文本，用于验证中文字符在终端中的显示宽度是否被正确计算，如果不正确就会导致用户界面崩溃的问题",
        "ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ０１２３４５６７８９ａｂｃｄｅｆｇｈｉｊｋｌｍ",
    ]

    for text in cases:
        input_widget = Input()
        input_widget.set_value(text)
        input_widget.focused = True
        [line] = input_widget.render(93)
        assert visible_width(line) <= 93

        input_widget = Input()
        input_widget.set_value(text)
        input_widget.focused = True
        for _ in range(10):
            input_widget.handle_input("\x1b[C")
        [line] = input_widget.render(93)
        assert visible_width(line) <= 93

        input_widget = Input()
        input_widget.set_value(text)
        input_widget.focused = True
        input_widget.handle_input("\x05")
        [line] = input_widget.render(93)
        assert visible_width(line) <= 93


def test_input_does_not_overflow_with_wide_cjk_and_fullwidth_text_like_ts() -> None:
    """does not overflow with wide CJK and fullwidth text"""
    input_widget = Input()
    input_widget.set_value(
        "ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ 가나다라마바사아자차카타파하"
    )
    input_widget.focused = True

    [line] = input_widget.render(93)

    assert visible_width(line) <= 93


def test_input_render_does_not_overflow_wide_text_at_start_cursor_position() -> None:
    input_widget = Input()
    input_widget.set_value("これはテスト文章です。日本語のテキストが正しく表示されるかどうかを確認するためのサンプルテキストです。あいうえお")
    input_widget.focused = True

    [line] = input_widget.render(93)

    assert visible_width(line) <= 93


def test_input_render_does_not_overflow_wide_text_at_middle_cursor_position() -> None:
    input_widget = Input()
    input_widget.set_value("这是一段测试文本，用于验证中文字符在终端中的显示宽度是否被正确计算，如果不正确就会导致用户界面崩溃的问题")
    input_widget.focused = True
    for _ in range(10):
        input_widget.handle_input("\x1b[C")

    [line] = input_widget.render(93)

    assert visible_width(line) <= 93


def test_input_render_does_not_overflow_wide_text_at_end_cursor_position() -> None:
    input_widget = Input()
    input_widget.set_value("ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ０１２３４５６７８９ａｂｃｄｅｆｇｈｉｊｋｌｍ")
    input_widget.focused = True
    input_widget.handle_input("\x05")

    [line] = input_widget.render(93)

    assert visible_width(line) <= 93


def test_input_render_keeps_cursor_visible_while_horizontally_scrolling_wide_text() -> None:
    input_widget = Input()
    input_widget.set_value("가나다라마바사아자차카타파하")
    input_widget.focused = True
    input_widget.handle_input("\x01")
    for _ in range(10):
        input_widget.handle_input("\x1b[C")

    [line] = input_widget.render(20)

    assert CURSOR_MARKER in line
    assert visible_width(line) <= 20
    assert "자차" in line
    assert "가나" not in line
    assert "자차" in line
    assert "가나" not in line


def test_input_keeps_the_cursor_visible_when_horizontally_scrolling_wide_text_like_ts() -> None:
    """keeps the cursor visible when horizontally scrolling wide text"""
    input_widget = Input()
    input_widget.set_value("가나다라마바사아자차카타파하")
    input_widget.focused = True
    input_widget.handle_input("\x01")
    for _ in range(5):
        input_widget.handle_input("\x1b[C")

    [line] = input_widget.render(20)

    assert visible_width(line) <= 20
    assert CURSOR_MARKER in line


def test_input_moves_and_deletes_grapheme_clusters_as_single_units() -> None:
    input_widget = Input()
    input_widget.set_value("A👍🏽B")
    input_widget.handle_input("\x05")
    input_widget.handle_input("\x1b[D")
    input_widget.handle_input("x")
    assert input_widget.get_value() == "A👍🏽xB"

    input_widget.handle_input("\x7f")
    assert input_widget.get_value() == "A👍🏽B"

    input_widget.handle_input("\x1b[D")
    input_widget.handle_input("\x1b[3~")
    assert input_widget.get_value() == "AB"


def test_input_bracketed_paste_is_normalized_to_single_line_like_ts() -> None:
    input_widget = Input()
    input_widget.set_value("hello world")
    input_widget.handle_input("\x01")
    for _ in range(5):
        input_widget.handle_input("\x1b[C")

    input_widget.handle_input("\x1b[200~beep\r\nboop\tzip\n\x1b[201~")

    assert input_widget.get_value() == "hellobeepboop    zip world"


def test_input_alt_b_and_alt_f_follow_ts_word_motion_rules() -> None:
    input_widget = Input()
    input_widget.set_value("foo, bar baz")
    input_widget.handle_input("\x01")
    input_widget.handle_input("\x1bf")
    assert input_widget.cursor == 3
    input_widget.handle_input("\x1bf")
    assert input_widget.cursor == 4
    input_widget.handle_input("\x1bf")
    assert input_widget.cursor == 8
    input_widget.handle_input("\x1bb")
    assert input_widget.cursor == 5
    input_widget.handle_input("\x1bb")
    assert input_widget.cursor == 3
    input_widget.handle_input("\x1bb")
    assert input_widget.cursor == 0


def test_input_undo_covers_yank_paste_and_cursor_split_units() -> None:
    input_widget = Input()
    for ch in "hello ":
        input_widget.handle_input(ch)
    input_widget.handle_input("\x17")
    input_widget.handle_input("\x19")
    assert input_widget.get_value() == "hello "
    input_widget.handle_input("\x1b[45;5u")
    assert input_widget.get_value() == ""

    input_widget = Input()
    input_widget.handle_input("\x1b[200~hello world\x1b[201~")
    assert input_widget.get_value() == "hello world"
    input_widget.handle_input("\x1b[45;5u")
    assert input_widget.get_value() == ""

    input_widget = Input()
    input_widget.set_value("hello world")
    input_widget.handle_input("\x01")
    for _ in range(5):
        input_widget.handle_input("\x1b[C")
    input_widget.handle_input("\x1b[200~beep boop\x1b[201~")
    assert input_widget.get_value() == "hellobeep boop world"
    input_widget.handle_input("\x1b[45;5u")
    assert input_widget.get_value() == "hello world"


def test_input_undo_stack_and_delete_operations_match_ts_behavior() -> None:
    input_widget = Input()
    input_widget.handle_input("\x1b[45;5u")
    assert input_widget.get_value() == ""

    for ch in "hello world":
        input_widget.handle_input(ch)
    input_widget.handle_input("\x1b[45;5u")
    assert input_widget.get_value() == "hello"
    input_widget.handle_input("\x1b[45;5u")
    assert input_widget.get_value() == ""

    input_widget = Input()
    for ch in "hello  ":
        input_widget.handle_input(ch)
    input_widget.handle_input("\x1b[45;5u")
    assert input_widget.get_value() == "hello "
    input_widget.handle_input("\x1b[45;5u")
    assert input_widget.get_value() == "hello"
    input_widget.handle_input("\x1b[45;5u")
    assert input_widget.get_value() == ""

    input_widget = Input()
    for ch in "hello":
        input_widget.handle_input(ch)
    input_widget.handle_input("\x7f")
    assert input_widget.get_value() == "hell"
    input_widget.handle_input("\x1b[45;5u")
    assert input_widget.get_value() == "hello"

    input_widget.handle_input("\x01")
    input_widget.handle_input("\x1b[C")
    input_widget.handle_input("\x1b[3~")
    assert input_widget.get_value() == "hllo"
    input_widget.handle_input("\x1b[45;5u")
    assert input_widget.get_value() == "hello"


def test_input_does_nothing_when_undo_stack_is_empty_like_ts() -> None:
    input_widget = Input()

    input_widget.handle_input("\x1b[45;5u")

    assert input_widget.get_value() == ""


def test_input_coalesces_consecutive_word_characters_into_one_undo_unit_like_ts() -> None:
    input_widget = Input()
    for ch in "hello world":
        input_widget.handle_input(ch)

    assert input_widget.get_value() == "hello world"

    input_widget.handle_input("\x1b[45;5u")
    assert input_widget.get_value() == "hello"

    input_widget.handle_input("\x1b[45;5u")
    assert input_widget.get_value() == ""


def test_input_undoes_forward_delete_like_ts() -> None:
    input_widget = Input()
    for ch in "hello":
        input_widget.handle_input(ch)
    input_widget.handle_input("\x01")
    input_widget.handle_input("\x1b[C")
    input_widget.handle_input("\x1b[3~")

    assert input_widget.get_value() == "hllo"

    input_widget.handle_input("\x1b[45;5u")
    assert input_widget.get_value() == "hello"


def test_input_undoes_ctrl_w_ctrl_k_ctrl_u_and_alt_d() -> None:
    input_widget = Input()
    for ch in "hello world":
        input_widget.handle_input(ch)
    input_widget.handle_input("\x17")
    assert input_widget.get_value() == "hello "
    input_widget.handle_input("\x1b[45;5u")
    assert input_widget.get_value() == "hello world"

    input_widget.handle_input("\x01")
    for _ in range(6):
        input_widget.handle_input("\x1b[C")
    input_widget.handle_input("\x0b")
    assert input_widget.get_value() == "hello "
    input_widget.handle_input("\x1b[45;5u")
    assert input_widget.get_value() == "hello world"

    input_widget.handle_input("\x15")
    assert input_widget.get_value() == "world"
    input_widget.handle_input("\x1b[45;5u")
    assert input_widget.get_value() == "hello world"

    input_widget.set_value("hello world test")
    input_widget.handle_input("\x01")
    input_widget.handle_input("\x1bd")
    assert input_widget.get_value() == " world test"
    input_widget.handle_input("\x1b[45;5u")
    assert input_widget.get_value() == "hello world test"

    input_widget.set_value("hello world")
    input_widget.handle_input("\x01")
    input_widget.handle_input("\x1bd")
    assert input_widget.get_value() == " world"
    input_widget.handle_input("\x1b[45;5u")
    assert input_widget.get_value() == "hello world"


def test_input_undoes_ctrl_w_delete_word_backward_like_ts() -> None:
    input_widget = Input()
    for ch in "hello world":
        input_widget.handle_input(ch)

    input_widget.handle_input("\x17")
    assert input_widget.get_value() == "hello "

    input_widget.handle_input("\x1b[45;5u")
    assert input_widget.get_value() == "hello world"


def test_input_undoes_ctrl_k_delete_to_line_end_like_ts() -> None:
    input_widget = Input()
    for ch in "hello world":
        input_widget.handle_input(ch)
    input_widget.handle_input("\x01")
    for _ in range(6):
        input_widget.handle_input("\x1b[C")

    input_widget.handle_input("\x0b")
    assert input_widget.get_value() == "hello "

    input_widget.handle_input("\x1b[45;5u")
    assert input_widget.get_value() == "hello world"


def test_input_undoes_ctrl_u_delete_to_line_start_like_ts() -> None:
    input_widget = Input()
    for ch in "hello world":
        input_widget.handle_input(ch)
    input_widget.handle_input("\x01")
    for _ in range(6):
        input_widget.handle_input("\x1b[C")

    input_widget.handle_input("\x15")
    assert input_widget.get_value() == "world"

    input_widget.handle_input("\x1b[45;5u")
    assert input_widget.get_value() == "hello world"


def test_input_undoes_alt_d_delete_word_forward_like_ts() -> None:
    """undoes Alt+D (delete word forward)"""
    input_widget = Input()
    input_widget.set_value("hello world")
    input_widget.handle_input("\x01")

    input_widget.handle_input("\x1bd")
    assert input_widget.get_value() == " world"

    input_widget.handle_input("\x1b[45;5u")
    assert input_widget.get_value() == "hello world"


def test_input_cursor_movement_starts_new_undo_unit() -> None:
    input_widget = Input()
    for ch in "abc":
        input_widget.handle_input(ch)
    input_widget.handle_input("\x01")
    input_widget.handle_input("\x05")
    input_widget.handle_input("d")
    input_widget.handle_input("e")

    assert input_widget.get_value() == "abcde"
    input_widget.handle_input("\x1b[45;5u")
    assert input_widget.get_value() == "abc"
    input_widget.handle_input("\x1b[45;5u")
    assert input_widget.get_value() == ""

    input_widget.set_value("hello world")
    input_widget.handle_input("\x01")
    for _ in range(5):
        input_widget.handle_input("\x1b[C")
    input_widget.handle_input("\x1b[200~beep boop\x1b[201~")
    assert input_widget.get_value() == "hellobeep boop world"
    input_widget.handle_input("\x1b[45;5u")
    assert input_widget.get_value() == "hello world"


def test_input_undoes_paste_atomically_like_ts() -> None:
    """undoes paste atomically"""
    input_widget = Input()
    input_widget.set_value("hello world")
    input_widget.handle_input("\x01")
    for _ in range(5):
        input_widget.handle_input("\x1b[C")

    input_widget.handle_input("\x1b[200~beep boop\x1b[201~")
    assert input_widget.get_value() == "hellobeep boop world"

    input_widget.handle_input("\x1b[45;5u")
    assert input_widget.get_value() == "hello world"

    input_widget = Input()
    for ch in "abc":
        input_widget.handle_input(ch)
    input_widget.handle_input("\x01")
    input_widget.handle_input("\x05")
    input_widget.handle_input("d")
    input_widget.handle_input("e")
    assert input_widget.get_value() == "abcde"
    input_widget.handle_input("\x1b[45;5u")
    assert input_widget.get_value() == "abc"
    input_widget.handle_input("\x1b[45;5u")
    assert input_widget.get_value() == ""


def test_editor_renders_cursor_marker_and_text() -> None:
    editor = Editor(None, None)
    editor.focused = True
    editor.set_text("hello\nworld")
    editor.handle_input("\x1b[A")
    lines = editor.render(20)
    assert any(CURSOR_MARKER in line for line in lines)
    assert any("hello" in line or "world" in line for line in lines)


def test_editor_get_lines_returns_defensive_copy() -> None:
    editor = Editor(None, None)
    editor.set_text("a\nb")
    lines = editor.get_lines()
    assert lines == ["a", "b"]
    lines[0] = "mutated"
    assert editor.get_lines() == ["a", "b"]


def test_editor_returns_lines_as_a_defensive_copy_exact_title_like_ts() -> None:
    """returns lines as a defensive copy"""
    editor = Editor(None, None)
    editor.set_text("a\nb")
    lines = editor.get_lines()
    assert lines == ["a", "b"]
    lines[0] = "mutated"
    assert editor.get_lines() == ["a", "b"]


def test_editor_returns_lines_as_a_defensive_copy() -> None:
    editor = Editor(None, None)
    editor.set_text("a\nb")
    lines = editor.get_lines()
    assert lines == ["a", "b"]
    lines[0] = "mutated"
    assert editor.get_lines() == ["a", "b"]


def test_editor_get_cursor_reports_position_through_basic_edits() -> None:
    editor = Editor(None, None)

    assert editor.get_cursor() == {"line": 0, "col": 0}

    editor.handle_input("a")
    editor.handle_input("b")
    editor.handle_input("c")
    assert editor.get_cursor() == {"line": 0, "col": 3}

    editor.handle_input("\x1b[D")
    assert editor.get_cursor() == {"line": 0, "col": 2}


def test_editor_returns_cursor_position_exact_title_like_ts() -> None:
    """returns cursor position"""
    editor = Editor(None, None)

    assert editor.get_cursor() == {"line": 0, "col": 0}

    editor.handle_input("a")
    editor.handle_input("b")
    editor.handle_input("c")
    assert editor.get_cursor() == {"line": 0, "col": 3}

    editor.handle_input("\x1b[D")
    assert editor.get_cursor() == {"line": 0, "col": 2}


def test_editor_returns_cursor_position() -> None:
    editor = Editor(None, None)

    assert editor.get_cursor() == {"line": 0, "col": 0}

    editor.handle_input("a")
    editor.handle_input("b")
    editor.handle_input("c")
    assert editor.get_cursor() == {"line": 0, "col": 3}

    editor.handle_input("\x1b[D")
    assert editor.get_cursor() == {"line": 0, "col": 2}


def test_editor_up_arrow_does_nothing_when_history_is_empty() -> None:
    editor = Editor(None, None)

    editor.handle_input("\x1b[A")

    assert editor.get_text() == ""


def test_editor_does_nothing_on_up_arrow_when_history_is_empty_exact_title_like_ts() -> None:
    """does nothing on Up arrow when history is empty"""
    editor = Editor(None, None)

    editor.handle_input("\x1b[A")

    assert editor.get_text() == ""


def test_editor_up_arrow_shows_most_recent_history_entry_when_empty() -> None:
    editor = Editor(None, None)
    editor.add_to_history("first prompt")
    editor.add_to_history("second prompt")

    editor.handle_input("\x1b[A")

    assert editor.get_text() == "second prompt"


def test_editor_shows_most_recent_history_entry_on_up_arrow_when_editor_is_empty_exact_title_like_ts() -> None:
    """shows most recent history entry on Up arrow when editor is empty"""
    editor = Editor(None, None)
    editor.add_to_history("first prompt")
    editor.add_to_history("second prompt")

    editor.handle_input("\x1b[A")

    assert editor.get_text() == "second prompt"


def test_editor_repeated_up_arrow_cycles_through_history_entries() -> None:
    editor = Editor(None, None)
    editor.add_to_history("first")
    editor.add_to_history("second")
    editor.add_to_history("third")

    editor.handle_input("\x1b[A")
    assert editor.get_text() == "third"
    editor.handle_input("\x1b[A")
    assert editor.get_text() == "second"
    editor.handle_input("\x1b[A")
    assert editor.get_text() == "first"


def test_editor_cycles_through_history_entries_on_repeated_up_arrow_exact_title_like_ts() -> None:
    """cycles through history entries on repeated Up arrow"""
    editor = Editor(None, None)
    editor.add_to_history("first")
    editor.add_to_history("second")
    editor.add_to_history("third")

    editor.handle_input("\x1b[A")
    assert editor.get_text() == "third"
    editor.handle_input("\x1b[A")
    assert editor.get_text() == "second"
    editor.handle_input("\x1b[A")
    assert editor.get_text() == "first"


def test_editor_down_arrow_returns_to_empty_editor_after_history_browsing() -> None:
    editor = Editor(None, None)
    editor.add_to_history("first")
    editor.add_to_history("second")

    editor.handle_input("\x1b[A")
    assert editor.get_text() == "second"
    editor.handle_input("\x1b[A")
    assert editor.get_text() == "first"

    editor.handle_input("\x1b[B")
    assert editor.get_text() == "second"
    editor.handle_input("\x1b[B")
    assert editor.get_text() == ""


def test_editor_returns_to_empty_editor_on_down_arrow_after_browsing_history_exact_title_like_ts() -> None:
    """returns to empty editor on Down arrow after browsing history"""
    editor = Editor(None, None)
    editor.add_to_history("first")
    editor.add_to_history("second")

    editor.handle_input("\x1b[A")
    assert editor.get_text() == "second"
    editor.handle_input("\x1b[A")
    assert editor.get_text() == "first"

    editor.handle_input("\x1b[B")
    assert editor.get_text() == "second"
    editor.handle_input("\x1b[B")
    assert editor.get_text() == ""


def test_editor_down_arrow_navigates_forward_through_history() -> None:
    editor = Editor(None, None)
    editor.add_to_history("first")
    editor.add_to_history("second")
    editor.add_to_history("third")

    editor.handle_input("\x1b[A")
    editor.handle_input("\x1b[A")
    editor.handle_input("\x1b[A")

    editor.handle_input("\x1b[B")
    assert editor.get_text() == "second"
    editor.handle_input("\x1b[B")
    assert editor.get_text() == "third"
    editor.handle_input("\x1b[B")
    assert editor.get_text() == ""


def test_editor_navigates_forward_through_history_with_down_arrow_exact_title_like_ts() -> None:
    """navigates forward through history with Down arrow"""
    editor = Editor(None, None)
    editor.add_to_history("first")
    editor.add_to_history("second")
    editor.add_to_history("third")

    editor.handle_input("\x1b[A")
    editor.handle_input("\x1b[A")
    editor.handle_input("\x1b[A")

    editor.handle_input("\x1b[B")
    assert editor.get_text() == "second"
    editor.handle_input("\x1b[B")
    assert editor.get_text() == "third"
    editor.handle_input("\x1b[B")
    assert editor.get_text() == ""


def test_editor_down_arrow_returns_to_empty_after_single_history_entry() -> None:
    editor = Editor(None, None)
    editor.add_to_history("prompt")

    editor.handle_input("\x1b[A")
    assert editor.get_text() == "prompt"

    editor.handle_input("\x1b[B")
    assert editor.get_text() == ""


def test_editor_typing_character_exits_history_mode() -> None:
    editor = Editor(None, None)
    editor.add_to_history("first")
    editor.add_to_history("second")

    editor.handle_input("\x1b[A")
    assert editor.get_text() == "second"

    editor.handle_input("x")
    assert editor.get_text() == "secondx"


def test_editor_exits_history_mode_when_typing_a_character_exact_title_like_ts() -> None:
    """exits history mode when typing a character"""
    editor = Editor(None, None)
    editor.add_to_history("first")
    editor.add_to_history("second")

    editor.handle_input("\x1b[A")
    assert editor.get_text() == "second"

    editor.handle_input("x")
    assert editor.get_text() == "secondx"


def test_editor_set_text_exits_history_mode() -> None:
    editor = Editor(None, None)
    editor.add_to_history("first")
    editor.add_to_history("second")

    editor.handle_input("\x1b[A")
    assert editor.get_text() == "second"

    editor.set_text("replacement")
    assert editor.get_text() == "replacement"

    editor.handle_input("\x1b[A")
    assert editor.get_text() == "second"


def test_editor_exits_history_mode_on_settext_exact_title_like_ts() -> None:
    """exits history mode on setText"""
    editor = Editor(None, None)
    editor.add_to_history("first")
    editor.add_to_history("second")

    editor.handle_input("\x1b[A")
    assert editor.get_text() == "second"

    editor.set_text("replacement")
    assert editor.get_text() == "replacement"

    editor.handle_input("\x1b[A")
    assert editor.get_text() == "second"


def test_editor_exits_history_mode_on_set_text_exact_title_like_ts() -> None:
    """exits history mode on setText"""
    editor = Editor(None, None)
    editor.add_to_history("first")
    editor.add_to_history("second")

    editor.handle_input("\x1b[A")
    assert editor.get_text() == "second"

    editor.set_text("replacement")
    assert editor.get_text() == "replacement"

    editor.handle_input("\x1b[A")
    assert editor.get_text() == "second"


def test_editor_up_uses_cursor_movement_when_editor_has_content() -> None:
    editor = Editor(None, None)
    editor.add_to_history("history item")
    editor.set_text("line1\nline2")

    editor.handle_input("\x1b[A")
    editor.handle_input("X")

    assert editor.get_text() == "line1X\nline2"


def test_editor_uses_cursor_movement_instead_of_history_when_editor_has_content_exact_title_like_ts() -> None:
    """uses cursor movement instead of history when editor has content"""
    editor = Editor(None, None)
    editor.add_to_history("history item")
    editor.set_text("line1\nline2")

    editor.handle_input("\x1b[A")
    editor.handle_input("X")

    assert editor.get_text() == "line1X\nline2"


def test_editor_history_limits_entries_to_most_recent_100() -> None:
    editor = Editor(None, None)
    for i in range(105):
        editor.add_to_history(f"prompt {i}")

    for _ in range(100):
        editor.handle_input("\x1b[A")

    assert editor.get_text() == "prompt 5"
    editor.handle_input("\x1b[A")
    assert editor.get_text() == "prompt 5"


def test_editor_limits_history_to_100_entries_exact_title_like_ts() -> None:
    """limits history to 100 entries"""
    editor = Editor(None, None)
    for i in range(105):
        editor.add_to_history(f"prompt {i}")

    for _ in range(100):
        editor.handle_input("\x1b[A")

    assert editor.get_text() == "prompt 5"
    editor.handle_input("\x1b[A")
    assert editor.get_text() == "prompt 5"


def test_editor_down_exits_single_multiline_history_entry_when_on_last_line() -> None:
    editor = Editor(None, None)
    editor.add_to_history("line1\nline2\nline3")

    editor.handle_input("\x1b[A")
    assert editor.get_text() == "line1\nline2\nline3"

    editor.handle_input("\x1b[B")
    assert editor.get_text() == ""


def test_editor_down_moves_within_multiline_history_entry_before_exiting() -> None:
    editor = Editor(None, None)
    editor.add_to_history("line1\nline2\nline3")

    editor.handle_input("\x1b[A")
    editor.handle_input("\x1b[A")
    editor.handle_input("\x1b[A")

    editor.handle_input("\x1b[B")
    assert editor.get_text() == "line1\nline2\nline3"
    editor.handle_input("\x1b[B")
    assert editor.get_text() == "line1\nline2\nline3"
    editor.handle_input("\x1b[B")
    assert editor.get_text() == ""


def test_editor_allows_cursor_movement_within_multi_line_history_entry_with_down_exact_title_like_ts() -> None:
    """allows cursor movement within multi-line history entry with Down"""
    editor = Editor(None, None)
    editor.add_to_history("line1\nline2\nline3")

    editor.handle_input("\x1b[A")
    editor.handle_input("\x1b[A")
    editor.handle_input("\x1b[A")

    editor.handle_input("\x1b[B")
    assert editor.get_text() == "line1\nline2\nline3"
    editor.handle_input("\x1b[B")
    assert editor.get_text() == "line1\nline2\nline3"
    editor.handle_input("\x1b[B")
    assert editor.get_text() == ""


def test_editor_up_moves_within_multiline_history_entry_before_older_history() -> None:
    editor = Editor(None, None)
    editor.add_to_history("older entry")
    editor.add_to_history("line1\nline2\nline3")

    editor.handle_input("\x1b[A")
    assert editor.get_text() == "line1\nline2\nline3"

    editor.handle_input("\x1b[A")
    assert editor.get_text() == "line1\nline2\nline3"
    editor.handle_input("\x1b[A")
    assert editor.get_text() == "line1\nline2\nline3"

    editor.handle_input("\x1b[A")
    assert editor.get_text() == "older entry"


def test_editor_allows_cursor_movement_within_multi_line_history_entry_with_up_exact_title_like_ts() -> None:
    """allows cursor movement within multi-line history entry with Up"""
    editor = Editor(None, None)
    editor.add_to_history("older entry")
    editor.add_to_history("line1\nline2\nline3")

    editor.handle_input("\x1b[A")
    assert editor.get_text() == "line1\nline2\nline3"

    editor.handle_input("\x1b[A")
    assert editor.get_text() == "line1\nline2\nline3"
    editor.handle_input("\x1b[A")
    assert editor.get_text() == "line1\nline2\nline3"

    editor.handle_input("\x1b[A")
    assert editor.get_text() == "older entry"


def test_editor_navigates_from_multiline_entry_back_via_down_after_cursor_movement() -> None:
    editor = Editor(None, None)
    editor.add_to_history("line1\nline2\nline3")

    editor.handle_input("\x1b[A")
    editor.handle_input("\x1b[A")
    editor.handle_input("\x1b[A")

    editor.handle_input("\x1b[B")
    assert editor.get_text() == "line1\nline2\nline3"
    editor.handle_input("\x1b[B")
    assert editor.get_text() == "line1\nline2\nline3"
    editor.handle_input("\x1b[B")
    assert editor.get_text() == ""


def test_editor_navigates_from_multi_line_entry_back_to_newer_via_down_after_cursor_movement_exact_title_like_ts() -> None:
    """navigates from multi-line entry back to newer via Down after cursor movement"""
    editor = Editor(None, None)
    editor.add_to_history("line1\nline2\nline3")

    editor.handle_input("\x1b[A")
    editor.handle_input("\x1b[A")
    editor.handle_input("\x1b[A")

    editor.handle_input("\x1b[B")
    assert editor.get_text() == "line1\nline2\nline3"
    editor.handle_input("\x1b[B")
    assert editor.get_text() == "line1\nline2\nline3"
    editor.handle_input("\x1b[B")
    assert editor.get_text() == ""


def test_editor_ctrl_a_moves_to_line_start_before_inserting() -> None:
    """moves cursor to document start on Ctrl+A and inserts at the beginning"""
    editor = Editor(None, None)

    editor.handle_input("a")
    editor.handle_input("b")
    editor.handle_input("\x01")
    editor.handle_input("x")

    assert editor.get_text() == "xab"
    assert editor.get_cursor() == {"line": 0, "col": 1}


def test_editor_backslash_enter_turns_trailing_backslash_into_newline() -> None:
    editor = Editor(None, None)
    editor.handle_input("\\")
    editor.handle_input("\r")
    assert editor.get_text() == "\n"


def test_editor_converts_standalone_backslash_to_newline_on_enter_like_ts() -> None:
    """converts standalone backslash to newline on Enter"""
    editor = Editor(None, None)
    editor.handle_input("\\")
    editor.handle_input("\r")

    assert editor.get_text() == "\n"


def test_editor_converts_standalone_backslash_to_newline_on_enter() -> None:
    editor = Editor(None, None)
    editor.handle_input("\\")
    editor.handle_input("\r")

    assert editor.get_text() == "\n"


def test_editor_backslash_is_inserted_immediately_without_buffering() -> None:
    editor = Editor(None, None)
    editor.handle_input("\\")

    assert editor.get_text() == "\\"


def test_editor_inserts_backslash_immediately_no_buffering_exact_title_like_ts() -> None:
    """inserts backslash immediately (no buffering)"""
    editor = Editor(None, None)
    editor.handle_input("\\")

    assert editor.get_text() == "\\"


def test_editor_backslash_followed_by_other_character_stays_literal() -> None:
    editor = Editor(None, None)
    editor.handle_input("\\")
    editor.handle_input("x")

    assert editor.get_text() == "\\x"


def test_editor_inserts_backslash_normally_when_followed_by_other_characters_like_ts() -> None:
    """inserts backslash normally when followed by other characters"""
    editor = Editor(None, None)
    editor.handle_input("\\")
    editor.handle_input("x")

    assert editor.get_text() == "\\x"


def test_editor_inserts_backslash_normally_when_followed_by_other_characters() -> None:
    editor = Editor(None, None)
    editor.handle_input("\\")
    editor.handle_input("x")

    assert editor.get_text() == "\\x"


def test_editor_enter_submits_when_backslash_is_not_immediately_before_cursor() -> None:
    editor = Editor(None, None)
    submitted: list[str] = []
    editor.on_submit = submitted.append

    editor.handle_input("\\")
    editor.handle_input("x")
    editor.handle_input("\r")

    assert submitted == ["\\x"]
    assert editor.get_text() == ""


def test_editor_does_not_trigger_newline_when_backslash_is_not_immediately_before_cursor_exact_title_like_ts() -> None:
    """does not trigger newline when backslash is not immediately before cursor"""
    editor = Editor(None, None)
    submitted: list[str] = []
    editor.on_submit = submitted.append

    editor.handle_input("\\")
    editor.handle_input("x")
    editor.handle_input("\r")

    assert submitted == ["\\x"]
    assert editor.get_text() == ""


def test_editor_backslash_enter_only_removes_one_of_multiple_trailing_backslashes() -> None:
    editor = Editor(None, None)
    editor.handle_input("\\")
    editor.handle_input("\\")
    editor.handle_input("\\")

    assert editor.get_text() == "\\\\\\"

    editor.handle_input("\r")

    assert editor.get_text() == "\\\\\n"


def test_editor_only_removes_one_backslash_when_multiple_are_present_exact_title_like_ts() -> None:
    """only removes one backslash when multiple are present"""
    editor = Editor(None, None)
    editor.handle_input("\\")
    editor.handle_input("\\")
    editor.handle_input("\\")

    assert editor.get_text() == "\\\\\\"

    editor.handle_input("\r")

    assert editor.get_text() == "\\\\\n"


def test_editor_history_repeated_up_and_down_navigation() -> None:
    editor = Editor(None, None)
    editor.add_to_history("first")
    editor.add_to_history("second")
    editor.add_to_history("third")

    editor.handle_input("\x1b[A")
    assert editor.get_text() == "third"
    editor.handle_input("\x1b[A")
    assert editor.get_text() == "second"
    editor.handle_input("\x1b[A")
    assert editor.get_text() == "first"
    editor.handle_input("\x1b[B")
    assert editor.get_text() == "second"
    editor.handle_input("\x1b[B")
    assert editor.get_text() == "third"
    editor.handle_input("\x1b[B")
    assert editor.get_text() == ""


def test_editor_history_multiline_entry_uses_cursor_movement_before_older_history() -> None:
    editor = Editor(None, None)
    editor.add_to_history("older entry")
    editor.add_to_history("line1\nline2\nline3")

    editor.handle_input("\x1b[A")
    assert editor.get_text() == "line1\nline2\nline3"
    editor.handle_input("\x1b[A")
    assert editor.get_text() == "line1\nline2\nline3"
    assert editor.get_cursor() == {"line": 1, "col": 5}
    editor.handle_input("\x1b[A")
    assert editor.get_cursor() == {"line": 0, "col": 5}
    editor.handle_input("\x1b[A")
    assert editor.get_text() == "older entry"


def test_editor_moves_left_over_grapheme_cluster_as_single_unit() -> None:
    editor = Editor(None, None)
    editor.set_text("a👍🏽b")

    editor.handle_input("\x1b[D")
    assert editor.get_cursor() == {"line": 0, "col": 3}

    editor.handle_input("\x1b[D")
    assert editor.get_cursor() == {"line": 0, "col": 1}


def test_editor_backspace_deletes_grapheme_cluster_as_single_unit() -> None:
    editor = Editor(None, None)
    editor.set_text("a👍🏽")

    editor.handle_input("\x7f")

    assert editor.get_text() == "a"
    assert editor.get_cursor() == {"line": 0, "col": 1}


def test_editor_kitty_csi_u_ignores_unsupported_printable_modifiers() -> None:
    editor = Editor(None, None)
    editor.handle_input("\x1b[99;9u")
    assert editor.get_text() == ""


def test_editor_ignores_printable_csi_u_sequences_with_unsupported_modifiers() -> None:
    """ignores printable CSI-u sequences with unsupported modifiers"""
    editor = Editor(None, None)
    editor.handle_input("\x1b[99;9u")

    assert editor.get_text() == ""


def test_editor_inserts_mixed_ascii_umlauts_and_emoji_as_literal_text() -> None:
    """inserts mixed ASCII, umlauts, and emojis as literal text"""
    editor = Editor(None, None)
    for chunk in ["H", "e", "l", "l", "o", " ", "ä", "ö", "ü", " ", "😀"]:
        editor.handle_input(chunk)

    assert editor.get_text() == "Hello äöü 😀"


def test_editor_backspace_deletes_single_code_unit_unicode_characters() -> None:
    editor = Editor(None, None)
    editor.handle_input("ä")
    editor.handle_input("ö")
    editor.handle_input("ü")

    editor.handle_input("\x7f")

    assert editor.get_text() == "äö"


def test_editor_deletes_single_code_unit_unicode_characters_umlauts_with_backspace() -> None:
    """deletes single-code-unit unicode characters (umlauts) with Backspace"""
    editor = Editor(None, None)
    editor.handle_input("ä")
    editor.handle_input("ö")
    editor.handle_input("ü")

    editor.handle_input("\x7f")

    assert editor.get_text() == "äö"


def test_editor_backspace_deletes_multi_code_unit_emoji_with_single_keypress() -> None:
    """deletes multi-code-unit emojis with single Backspace"""
    editor = Editor(None, None)
    editor.handle_input("😀")
    editor.handle_input("👍")

    editor.handle_input("\x7f")

    assert editor.get_text() == "😀"


def test_editor_inserts_at_correct_position_after_moving_over_umlauts() -> None:
    """inserts characters at the correct position after cursor movement over umlauts"""
    editor = Editor(None, None)
    editor.handle_input("ä")
    editor.handle_input("ö")
    editor.handle_input("ü")

    editor.handle_input("\x1b[D")
    editor.handle_input("\x1b[D")
    editor.handle_input("x")

    assert editor.get_text() == "äxöü"


def test_editor_moves_across_multi_code_unit_emoji_with_single_arrow() -> None:
    """moves cursor across multi-code-unit emojis with single arrow key"""
    editor = Editor(None, None)
    editor.handle_input("😀")
    editor.handle_input("👍")
    editor.handle_input("🎉")

    editor.handle_input("\x1b[D")
    editor.handle_input("\x1b[D")
    editor.handle_input("x")

    assert editor.get_text() == "😀x👍🎉"


def test_editor_preserves_umlauts_across_line_breaks() -> None:
    """preserves umlauts across line breaks"""
    editor = Editor(None, None)
    for chunk in ["ä", "ö", "ü", "\n", "Ä", "Ö", "Ü"]:
        editor.handle_input(chunk)

    assert editor.get_text() == "äöü\nÄÖÜ"


def test_editor_set_text_replaces_document_with_unicode_text() -> None:
    editor = Editor(None, None)
    editor.set_text("Hällö Wörld! 😀 äöüÄÖÜß")

    assert editor.get_text() == "Hällö Wörld! 😀 äöüÄÖÜß"


def test_editor_replaces_the_entire_document_with_unicode_text_via_settext_paste_simulation() -> None:
    """replaces the entire document with unicode text via setText (paste simulation)"""
    editor = Editor(None, None)
    editor.set_text("Hällö Wörld! 😀 äöüÄÖÜß")

    assert editor.get_text() == "Hällö Wörld! 😀 äöüÄÖÜß"


def test_editor_kitty_csi_u_printables_insert_unicode_text() -> None:
    editor = Editor(None, None)
    for chunk in ["\x1b[97u", "\x1b[246u", "\x1b[128640u"]:
        editor.handle_input(chunk)

    assert editor.get_text() == "aö🚀"


def test_editor_word_navigation_handles_punctuation_runs_and_leading_whitespace() -> None:
    editor = Editor(None, None)
    editor.set_text("foo bar... baz")

    editor.handle_input("\x1b[1;5D")
    assert editor.get_cursor() == {"line": 0, "col": 11}
    editor.handle_input("\x1b[1;5D")
    assert editor.get_cursor() == {"line": 0, "col": 7}
    editor.handle_input("\x1b[1;5D")
    assert editor.get_cursor() == {"line": 0, "col": 4}

    editor.handle_input("\x1b[1;5C")
    assert editor.get_cursor() == {"line": 0, "col": 7}
    editor.handle_input("\x1b[1;5C")
    assert editor.get_cursor() == {"line": 0, "col": 10}
    editor.handle_input("\x1b[1;5C")
    assert editor.get_cursor() == {"line": 0, "col": 14}

    editor.set_text("   foo bar")
    editor.handle_input("\x01")
    editor.handle_input("\x1b[1;5C")
    assert editor.get_cursor() == {"line": 0, "col": 6}


def test_editor_navigates_words_correctly_with_ctrl_left_right() -> None:
    """navigates words correctly with Ctrl+Left/Right"""
    editor = Editor(None, None)
    editor.set_text("foo bar... baz")

    editor.handle_input("\x1b[1;5D")
    assert editor.get_cursor() == {"line": 0, "col": 11}
    editor.handle_input("\x1b[1;5D")
    assert editor.get_cursor() == {"line": 0, "col": 7}
    editor.handle_input("\x1b[1;5D")
    assert editor.get_cursor() == {"line": 0, "col": 4}

    editor.handle_input("\x1b[1;5C")
    assert editor.get_cursor() == {"line": 0, "col": 7}
    editor.handle_input("\x1b[1;5C")
    assert editor.get_cursor() == {"line": 0, "col": 10}
    editor.handle_input("\x1b[1;5C")
    assert editor.get_cursor() == {"line": 0, "col": 14}


def test_editor_wraps_wide_emoji_and_boundary_cases_without_overflow() -> None:
    """
    wraps lines correctly when text contains wide emojis
    wraps long text with emojis at correct positions
    wraps CJK characters correctly (each is 2 columns wide)
    does not exceed terminal width with emoji at wrap boundary
    """
    cases = [
        ("😀😀😀😀😀😀", 10, ["😀😀😀😀", "😀😀"]),
        ("0123456789✅", 10, ["012345678", "9✅"]),
        ("日本語テスト", 11, ["日本語テス", "ト"]),
    ]

    for text, width, expected_prefixes in cases:
        editor = Editor(None, None)
        editor.set_text(text)
        plain = [
            line.replace(CURSOR_MARKER, "").replace("\x1b[7m", "").replace("\x1b[27m", "").rstrip()
            for line in editor.render(width)
        ]

        assert len(plain) == len(expected_prefixes)
        for line, expected in zip(plain, expected_prefixes, strict=True):
            assert line.startswith(expected)
            assert visible_width(line) <= width


def test_editor_wraps_mixed_ascii_and_wide_characters_to_width() -> None:
    """
    handles mixed ASCII and wide characters in wrapping
    renders cursor correctly on wide characters
    """
    editor = Editor(None, None)
    editor.set_text("abc😀def😀ghi😀jkl")

    plain = [
        line.replace(CURSOR_MARKER, "").replace("\x1b[7m", "").replace("\x1b[27m", "")
        for line in editor.render(8)
    ]

    assert [visible_width(line) for line in plain] == [7, 7, 7]
    assert plain[0].startswith("abc😀de")
    assert plain[1].startswith("f😀ghi")
    assert plain[2].rstrip() == "😀jkl"


def test_editor_word_wrap_prefers_word_boundaries_and_avoids_leading_whitespace() -> None:
    """
    wraps at word boundaries instead of mid-word
    does not start lines with leading whitespace after word wrap
    """
    editor = Editor(None, None)
    editor.set_text("Hello world this is a test of word wrapping functionality")

    plain = [
        line.replace(CURSOR_MARKER, "").replace("\x1b[7m", "").replace("\x1b[27m", "")
        for line in editor.render(40)
    ]

    assert plain[0].startswith("Hello world this is a test of word ")
    assert plain[1].rstrip() == "wrapping functionality"

    editor.set_text("Word1 Word2 Word3 Word4 Word5 Word6")
    plain = [
        line.replace(CURSOR_MARKER, "").replace("\x1b[7m", "").replace("\x1b[27m", "")
        for line in editor.render(20)
    ]
    assert not plain[1].startswith(" ")


def test_editor_word_wrap_breaks_long_urls_at_character_level_and_preserves_multiple_spaces() -> None:
    """
    breaks long words (URLs) at character level
    preserves multiple spaces within words on same line
    """
    editor = Editor(None, None)
    editor.set_text("Check https://example.com/very/long/path/that/exceeds/width here")

    plain = [
        line.replace(CURSOR_MARKER, "").replace("\x1b[7m", "").replace("\x1b[27m", "")
        for line in editor.render(30)
    ]
    assert all(visible_width(line) == 29 for line in plain)
    assert plain[0].startswith("Check ")
    assert plain[1] == "https://example.com/very/long"
    assert plain[-1].rstrip() == "/path/that/exceeds/width here"

    editor.set_text("Word1   Word2    Word3")
    [line] = [
        line.replace(CURSOR_MARKER, "").replace("\x1b[7m", "").replace("\x1b[27m", "")
        for line in editor.render(50)
    ]
    assert "Word1   Word2    Word3" in line


def test_editor_render_preserves_multiple_spaces_within_words_on_same_line() -> None:
    editor = Editor(None, None)
    editor.set_text("Word1   Word2    Word3")

    [line] = [
        line.replace(CURSOR_MARKER, "").replace("\x1b[7m", "").replace("\x1b[27m", "")
        for line in editor.render(50)
    ]

    assert "Word1   Word2" in line


def test_editor_word_wrap_keeps_boundary_whitespace_and_moves_full_word_to_next_line() -> None:
    """
    wraps word to next line when it ends exactly at terminal width
    keeps whitespace at terminal width boundary on same line
    wraps word to next line when it fits width but not remaining space
    keeps word with multi-space and following word together when they fit
    keeps word with multi-space and following word when they fill width exactly
    splits when word plus multi-space plus word exceeds width
    """
    editor = Editor(None, None)

    chunks = editor._word_wrap_line("hello world test", 11)
    assert [chunk[0] for chunk in chunks] == ["hello ", "world test"]

    chunks = editor._word_wrap_line("hello world test", 12)
    assert [chunk[0] for chunk in chunks] == ["hello world ", "test"]

    chunks = editor._word_wrap_line("aaaaaaaaaaaa aaaa", 12)
    assert [chunk[0] for chunk in chunks] == ["aaaaaaaaaaaa", " aaaa"]

    chunks = editor._word_wrap_line("      aaaaaaaaaaaa", 12)
    assert [chunk[0] for chunk in chunks] == ["      ", "aaaaaaaaaaaa"]

    chunks = editor._word_wrap_line("Lorem ipsum dolor sit amet,    consectetur", 30)
    assert [chunk[0] for chunk in chunks] == [
        "Lorem ipsum dolor sit ",
        "amet,    consectetur",
    ]

    chunks = editor._word_wrap_line("Lorem ipsum dolor sit amet,              consectetur", 30)
    assert [chunk[0] for chunk in chunks] == [
        "Lorem ipsum dolor sit ",
        "amet,              consectetur",
    ]

    chunks = editor._word_wrap_line("Lorem ipsum dolor sit amet,               consectetur", 30)
    assert [chunk[0] for chunk in chunks] == [
        "Lorem ipsum dolor sit ",
        "amet,               ",
        "consectetur",
    ]


def test_editor_word_wrap_handles_unbreakable_word_filling_width_exactly_followed_by_space() -> None:
    """handles unbreakable word filling width exactly followed by space"""
    editor = Editor(None, None)

    chunks = editor._word_wrap_line("aaaaaaaaaaaa aaaa", 12)

    assert [chunk[0] for chunk in chunks] == ["aaaaaaaaaaaa", " aaaa"]


def test_editor_word_wrap_rechecks_overflow_after_backtracking_wrap_opportunity() -> None:
    """wordWrapLine re-checks overflow after backtracking to wrap opportunity"""
    editor = Editor(None, None)
    editor.handle_input(" ")
    for _ in range(35):
        editor.handle_input("b")

    big_content = "line\n" * 27
    editor.handle_input(f"\x1b[200~{big_content.rstrip()}\x1b[201~")
    for _ in range(4):
        editor.handle_input("b")

    lines = editor.render(54)

    assert all(visible_width(line) <= 54 for line in lines)


def test_editor_word_wrap_handles_long_whitespace_runs_like_ts() -> None:
    """
    breaks long whitespace at line boundary
    breaks long whitespace at line boundary 2
    """
    editor = Editor(None, None)

    chunks = editor._word_wrap_line("Lorem ipsum dolor sit amet,                         consectetur", 30)
    assert [chunk[0] for chunk in chunks] == [
        "Lorem ipsum dolor sit ",
        "amet,                         ",
        "consectetur",
    ]

    chunks = editor._word_wrap_line("Lorem ipsum dolor sit amet,                          consectetur", 30)
    assert [chunk[0] for chunk in chunks] == [
        "Lorem ipsum dolor sit ",
        "amet,                         ",
        " consectetur",
    ]

    chunks = editor._word_wrap_line("Lorem ipsum dolor sit amet,                                     consectetur", 30)
    assert [chunk[0] for chunk in chunks] == [
        "Lorem ipsum dolor sit ",
        "amet,                         ",
        "            consectetur",
    ]


def test_editor_word_wrap_breaks_whitespace_spanning_full_lines() -> None:
    """breaks whitespace spanning full lines"""
    editor = Editor(None, None)

    chunks = editor._word_wrap_line("Lorem ipsum dolor sit amet,                                     consectetur", 30)

    assert [chunk[0] for chunk in chunks] == [
        "Lorem ipsum dolor sit ",
        "amet,                         ",
        "            consectetur",
    ]


def test_editor_word_wrap_splits_oversized_atomic_segments_like_ts() -> None:
    """
    splits oversized atomic segment across multiple chunks
    splits oversized atomic segment at start of line
    splits oversized atomic segment at end of line
    splits consecutive oversized atomic segments
    """
    editor = Editor(None, None)
    marker = "[paste #1 +20 lines]"

    cases = [
        ("A" + marker + "B", [("A", 0, 1), (marker, 1, 1 + len(marker)), ("B", 1 + len(marker), 2 + len(marker))], "B"),
        (marker + "B", [(marker, 0, len(marker)), ("B", len(marker), len(marker) + 1)], "B"),
        ("A" + marker, [("A", 0, 1), (marker, 1, 1 + len(marker))], None),
    ]

    for line, segments, trailing in cases:
        chunks = editor._word_wrap_line(line, 10, segments)
        assert all(visible_width(chunk[0]) <= 10 for chunk in chunks)
        assert "".join(line[start:end] for _text, start, end in chunks) == line
        if trailing is not None:
            assert trailing in chunks[-1][0]
        else:
            assert chunks[0][0] == "A"

    m2 = "[paste #2 +30 lines]"
    line = marker + m2
    chunks = editor._word_wrap_line(line, 10, [(marker, 0, len(marker)), (m2, len(marker), len(line))])
    assert all(visible_width(chunk[0]) <= 10 for chunk in chunks)
    assert "".join(line[start:end] for _text, start, end in chunks) == line
    assert len(chunks) >= 4


def test_editor_word_wrap_resumes_normal_wrapping_after_oversized_atomic_segment() -> None:
    """wraps normally after oversized atomic segment"""
    editor = Editor(None, None)
    marker = "[paste #1 +20 lines]"
    line = f"{marker} hello world"
    segments = [(marker, 0, len(marker))]
    for idx, ch in enumerate(" hello world", start=len(marker)):
        segments.append((ch, idx, idx + 1))

    chunks = editor._word_wrap_line(line, 10, segments)

    assert all(visible_width(chunk[0]) <= 10 for chunk in chunks)
    assert chunks[-1][0] == "world"
    assert "".join(line[start:end] for _text, start, end in chunks) == line


def test_editor_moves_through_wrapped_visual_lines_without_getting_stuck() -> None:
    terminal = _TerminalStub(columns=15, rows=24)
    tui = TUI(terminal)
    editor = Editor(tui, None)

    editor.set_text("short\n123456789012345678901234567890")
    editor.render(15)

    assert editor.get_cursor() == {"line": 1, "col": 30}

    editor.handle_input("\x1b[A")
    assert editor.get_cursor()["line"] == 1
    editor.handle_input("\x1b[A")
    assert editor.get_cursor()["line"] == 1
    editor.handle_input("\x1b[A")
    assert editor.get_cursor()["line"] == 0


def test_editor_vertical_movement_preserves_sticky_column() -> None:
    editor = Editor(None, None)
    editor.set_text("1234567\n12\n123456789")

    editor.handle_input("\x1b[A")
    editor.handle_input("\x1b[A")
    assert editor.get_cursor() == {"line": 0, "col": 7}

    editor.handle_input("\x1b[B")
    assert editor.get_cursor() == {"line": 1, "col": 2}

    editor.handle_input("\x1b[B")
    assert editor.get_cursor() == {"line": 2, "col": 9}


def test_editor_word_movement_resets_sticky_column() -> None:
    editor = Editor(None, None)
    editor.set_text("hello world\n\nhello world")

    editor.handle_input("\x1b[A")
    editor.handle_input("\x1b[A")
    editor.handle_input("\x01")
    editor.handle_input("\x1b[B")
    editor.handle_input("\x1b[B")
    assert editor.get_cursor() == {"line": 2, "col": 0}

    editor.handle_input("\x1b[1;5C")
    assert editor.get_cursor() == {"line": 2, "col": 5}

    editor.handle_input("\x1b[A")
    editor.handle_input("\x1b[A")
    assert editor.get_cursor() == {"line": 0, "col": 5}


def test_editor_undo_resets_sticky_column_to_restored_cursor() -> None:
    """resets sticky column on undo"""
    editor = Editor(None, None)
    editor.set_text("1234567890\n\n1234567890")

    editor.handle_input("\x1b[A")
    editor.handle_input("\x1b[A")
    editor.handle_input("\x01")
    for _ in range(8):
        editor.handle_input("\x1b[C")
    assert editor.get_cursor() == {"line": 0, "col": 8}

    editor.handle_input("\x1b[B")
    editor.handle_input("\x1b[B")
    assert editor.get_cursor() == {"line": 2, "col": 8}

    editor.handle_input("X")
    assert editor.get_cursor() == {"line": 2, "col": 9}

    editor.handle_input("\x1b[A")
    editor.handle_input("\x1b[A")
    assert editor.get_cursor() == {"line": 0, "col": 9}

    editor.handle_input("\x1b[45;5u")
    assert editor.get_text() == "1234567890\n\n1234567890"
    assert editor.get_cursor() == {"line": 2, "col": 8}

    editor.handle_input("\x1b[A")
    editor.handle_input("\x1b[A")
    assert editor.get_cursor() == {"line": 0, "col": 8}


def test_editor_right_at_line_end_updates_preferred_visual_column() -> None:
    """sets preferredVisualCol when pressing right at end of prompt (last line)"""
    editor = Editor(None, None)
    editor.set_text("111111111x1111111111\n\n333333333_")

    editor.handle_input("\x1b[A")
    editor.handle_input("\x1b[A")
    editor.handle_input("\x05")
    assert editor.get_cursor() == {"line": 0, "col": 20}

    editor.handle_input("\x1b[B")
    editor.handle_input("\x1b[B")
    assert editor.get_cursor() == {"line": 2, "col": 10}

    editor.handle_input("\x1b[C")
    assert editor.get_cursor() == {"line": 2, "col": 10}

    editor.handle_input("\x1b[A")
    editor.handle_input("\x1b[A")
    assert editor.get_cursor() == {"line": 0, "col": 10}


def test_editor_typing_exits_history_browsing_mode() -> None:
    editor = Editor(None, None)
    editor.add_to_history("first")
    editor.add_to_history("second")

    editor.handle_input("\x1b[A")
    assert editor.get_text() == "second"

    editor.handle_input("x")

    assert editor.get_text() == "secondx"
    editor.handle_input("\x1b[B")
    assert editor.get_text() == "secondx"


def test_editor_set_text_exits_history_mode() -> None:
    editor = Editor(None, None)
    editor.add_to_history("first")
    editor.add_to_history("second")

    editor.handle_input("\x1b[A")
    editor.set_text("")
    editor.handle_input("\x1b[A")

    assert editor.get_text() == "second"


def test_editor_set_text_resets_sticky_column_like_ts() -> None:
    editor = Editor(None, None)
    editor.set_text("1234567890\n\n1234567890")

    editor.handle_input("\x01")
    for _ in range(8):
        editor.handle_input("\x1b[C")
    editor.handle_input("\x1b[A")

    editor.set_text("abcdefghij\n\nabcdefghij")
    assert editor.get_cursor() == {"line": 2, "col": 10}

    editor.handle_input("\x1b[A")
    editor.handle_input("\x1b[A")
    assert editor.get_cursor() == {"line": 0, "col": 10}


def test_editor_history_ignores_empty_and_whitespace_entries() -> None:
    editor = Editor(None, None)
    editor.add_to_history("")
    editor.add_to_history("   ")
    editor.add_to_history("valid")

    editor.handle_input("\x1b[A")
    assert editor.get_text() == "valid"

    editor.handle_input("\x1b[A")
    assert editor.get_text() == "valid"


def test_editor_history_does_not_add_empty_or_whitespace_entries() -> None:
    editor = Editor(None, None)
    editor.add_to_history("")
    editor.add_to_history("   ")
    editor.add_to_history("valid")

    editor.handle_input("\x1b[A")
    assert editor.get_text() == "valid"
    editor.handle_input("\x1b[A")
    assert editor.get_text() == "valid"


def test_editor_does_not_add_empty_strings_to_history_exact_title_like_ts() -> None:
    """does not add empty strings to history"""
    editor = Editor(None, None)
    editor.add_to_history("")
    editor.add_to_history("   ")
    editor.add_to_history("valid")

    editor.handle_input("\x1b[A")
    assert editor.get_text() == "valid"
    editor.handle_input("\x1b[A")
    assert editor.get_text() == "valid"


def test_editor_history_deduplicates_only_consecutive_entries() -> None:
    editor = Editor(None, None)
    editor.add_to_history("same")
    editor.add_to_history("same")
    editor.add_to_history("same")

    editor.handle_input("\x1b[A")
    assert editor.get_text() == "same"

    editor.handle_input("\x1b[A")
    assert editor.get_text() == "same"

    editor = Editor(None, None)
    editor.add_to_history("first")
    editor.add_to_history("second")
    editor.add_to_history("first")

    editor.handle_input("\x1b[A")
    assert editor.get_text() == "first"
    editor.handle_input("\x1b[A")
    assert editor.get_text() == "second"
    editor.handle_input("\x1b[A")
    assert editor.get_text() == "first"


def test_editor_history_allows_non_consecutive_duplicates() -> None:
    editor = Editor(None, None)
    editor.add_to_history("first")
    editor.add_to_history("second")
    editor.add_to_history("first")

    editor.handle_input("\x1b[A")
    assert editor.get_text() == "first"
    editor.handle_input("\x1b[A")
    assert editor.get_text() == "second"
    editor.handle_input("\x1b[A")
    assert editor.get_text() == "first"


def test_editor_allows_non_consecutive_duplicates_in_history_exact_title_like_ts() -> None:
    """allows non-consecutive duplicates in history"""
    editor = Editor(None, None)
    editor.add_to_history("first")
    editor.add_to_history("second")
    editor.add_to_history("first")

    editor.handle_input("\x1b[A")
    assert editor.get_text() == "first"
    editor.handle_input("\x1b[A")
    assert editor.get_text() == "second"
    editor.handle_input("\x1b[A")
    assert editor.get_text() == "first"


def test_editor_history_does_not_add_consecutive_duplicates() -> None:
    editor = Editor(None, None)
    editor.add_to_history("same")
    editor.add_to_history("same")
    editor.add_to_history("same")

    editor.handle_input("\x1b[A")
    assert editor.get_text() == "same"
    editor.handle_input("\x1b[A")
    assert editor.get_text() == "same"


def test_editor_does_not_add_consecutive_duplicates_to_history_exact_title_like_ts() -> None:
    """does not add consecutive duplicates to history"""
    editor = Editor(None, None)
    editor.add_to_history("same")
    editor.add_to_history("same")
    editor.add_to_history("same")

    editor.handle_input("\x1b[A")
    assert editor.get_text() == "same"
    editor.handle_input("\x1b[A")
    assert editor.get_text() == "same"


def test_editor_history_keeps_only_most_recent_100_entries() -> None:
    editor = Editor(None, None)
    for i in range(105):
        editor.add_to_history(f"prompt {i}")

    for _ in range(100):
        editor.handle_input("\x1b[A")

    assert editor.get_text() == "prompt 5"
    editor.handle_input("\x1b[A")
    assert editor.get_text() == "prompt 5"


def test_editor_up_moves_cursor_inside_existing_content_before_history() -> None:
    editor = Editor(None, None)
    editor.add_to_history("history item")
    editor.set_text("line1\nline2")

    editor.handle_input("\x1b[A")
    editor.handle_input("X")

    assert editor.get_text() == "line1X\nline2"


def test_editor_down_exits_single_multiline_history_entry() -> None:
    editor = Editor(None, None)
    editor.add_to_history("line1\nline2\nline3")

    editor.handle_input("\x1b[A")
    assert editor.get_text() == "line1\nline2\nline3"

    editor.handle_input("\x1b[B")
    assert editor.get_text() == ""


def test_editor_down_navigates_within_multiline_history_entry_before_exit() -> None:
    editor = Editor(None, None)
    editor.add_to_history("line1\nline2\nline3")

    editor.handle_input("\x1b[A")
    editor.handle_input("\x1b[A")
    editor.handle_input("\x1b[A")

    editor.handle_input("\x1b[B")
    assert editor.get_text() == "line1\nline2\nline3"
    assert editor.get_cursor() == {"line": 1, "col": 5}

    editor.handle_input("\x1b[B")
    assert editor.get_text() == "line1\nline2\nline3"
    assert editor.get_cursor() == {"line": 2, "col": 5}

    editor.handle_input("\x1b[B")
    assert editor.get_text() == ""


def test_editor_render_wraps_paste_marker_without_overflow_or_duplicate_tail() -> None:
    editor = Editor(None, None)
    editor.focused = True
    editor.set_text("[paste #1 +20 lines]B")

    plain = [
        line.replace(CURSOR_MARKER, "").replace("\x1b[7m", "").replace("\x1b[27m", "")
        for line in editor.render(10)
    ]

    assert all(visible_width(line) <= 10 for line in plain)
    assert "".join(segment.rstrip() for segment in plain).count("B") == 1


def test_editor_render_wraps_text_plus_paste_marker_without_duplicate_boundary_chars() -> None:
    editor = Editor(None, None)
    editor.focused = True
    editor.set_text("A[paste #1 +20 lines]B")

    plain = [
        line.replace(CURSOR_MARKER, "").replace("\x1b[7m", "").replace("\x1b[27m", "")
        for line in editor.render(10)
    ]

    assert all(visible_width(line) <= 10 for line in plain)
    joined = "".join(segment.rstrip() for segment in plain)
    assert joined.count("A") == 1
    assert joined.count("B") == 1


def test_editor_delete_word_backward_and_undo() -> None:
    """undoes Ctrl+W (delete word backward)"""
    editor = Editor(None, None)
    editor.set_text("hello world")

    editor.handle_input("\x17")
    assert editor.get_text() == "hello "

    editor.handle_input("\x1b[45;5u")
    assert editor.get_text() == "hello world"


def test_editor_ctrl_w_and_alt_backspace_delete_words_like_ts() -> None:
    """deletes words correctly with Ctrl+W and Alt+Backspace"""
    editor = Editor(None, None)

    editor.set_text("foo bar baz")
    editor.handle_input("\x17")
    assert editor.get_text() == "foo bar "

    editor.set_text("foo bar   ")
    editor.handle_input("\x17")
    assert editor.get_text() == "foo "

    editor.set_text("foo bar...")
    editor.handle_input("\x17")
    assert editor.get_text() == "foo bar"

    editor.set_text("line one\nline two")
    editor.handle_input("\x17")
    assert editor.get_text() == "line one\nline "

    editor.set_text("line one\n")
    editor.handle_input("\x17")
    assert editor.get_text() == "line one"

    editor.set_text("foo 😀😀 bar")
    editor.handle_input("\x17")
    assert editor.get_text() == "foo 😀😀 "
    editor.handle_input("\x17")
    assert editor.get_text() == "foo "

    editor.set_text("foo bar")
    editor.handle_input("\x1b\x7f")
    assert editor.get_text() == "foo "


def test_editor_delete_word_forward_and_yank() -> None:
    """Alt+D at end of line deletes newline"""
    editor = Editor(None, None)
    editor.set_text("line1\nline2")
    editor.handle_input("\x1b[A")
    editor.handle_input("\x05")

    editor.handle_input("\x1bd")
    assert editor.get_text() == "line1line2"

    editor.handle_input("\x19")
    assert editor.get_text() == "line1\nline2"


def test_editor_ctrl_u_saves_deleted_prefix_to_kill_ring_and_yanks_it() -> None:
    """
    Ctrl+U saves deleted text to kill ring
    undoes Ctrl+U (delete to line start)
    """
    editor = Editor(None, None)
    editor.set_text("hello world")
    editor.handle_input("\x01")
    for _ in range(6):
        editor.handle_input("\x1b[C")

    editor.handle_input("\x15")
    assert editor.get_text() == "world"

    editor.handle_input("\x19")
    assert editor.get_text() == "hello world"


def test_editor_ctrl_k_saves_deleted_suffix_to_kill_ring_and_yanks_it() -> None:
    """
    Ctrl+K saves deleted text to kill ring
    undoes Ctrl+K (delete to line end)
    """
    editor = Editor(None, None)
    editor.set_text("hello world")
    editor.handle_input("\x01")

    editor.handle_input("\x0b")
    assert editor.get_text() == ""

    editor.handle_input("\x19")
    assert editor.get_text() == "hello world"


def test_editor_consecutive_ctrl_w_accumulates_into_one_kill_ring_entry() -> None:
    """
    consecutive Ctrl+W accumulates into one kill ring entry
    Ctrl+W saves deleted text to kill ring and Ctrl+Y yanks it
    """
    editor = Editor(None, None)
    editor.set_text("one two three")

    editor.handle_input("\x17")
    editor.handle_input("\x17")
    editor.handle_input("\x17")

    assert editor.get_text() == ""
    editor.handle_input("\x19")
    assert editor.get_text() == "one two three"


def test_editor_ctrl_w_coalesces_consecutive_multiline_deletions_into_one_entry() -> None:
    """consecutive deletions across lines coalesce into one entry"""
    editor = Editor(None, None)
    editor.set_text("1\n2\n3")

    editor.handle_input("\x17")
    assert editor.get_text() == "1\n2\n"
    editor.handle_input("\x17")
    assert editor.get_text() == "1\n2"
    editor.handle_input("\x17")
    assert editor.get_text() == "1\n"
    editor.handle_input("\x17")
    assert editor.get_text() == "1"
    editor.handle_input("\x17")
    assert editor.get_text() == ""

    editor.handle_input("\x19")
    assert editor.get_text() == "1\n2\n3"


def test_editor_ctrl_k_at_line_end_deletes_newline_and_coalesces() -> None:
    """Ctrl+K at line end deletes newline and coalesces"""
    editor = Editor(None, None)
    editor.set_text("ab\ncd")
    editor.handle_input("\x1b[A")
    editor.handle_input("\x05")

    editor.handle_input("\x0b")
    assert editor.get_text() == "abcd"

    editor.handle_input("\x0b")
    assert editor.get_text() == "ab"

    editor.handle_input("\x19")
    assert editor.get_text() == "ab\ncd"


def test_editor_forward_deletions_append_during_kill_accumulation() -> None:
    """backward deletions prepend, forward deletions append during accumulation"""
    editor = Editor(None, None)
    editor.set_text("prefix|suffix")
    editor.handle_input("\x01")
    for _ in range(6):
        editor.handle_input("\x1b[C")

    editor.handle_input("\x0b")
    editor.handle_input("\x0b")

    assert editor.get_text() == "prefix"
    editor.handle_input("\x19")
    assert editor.get_text() == "prefix|suffix"


def test_editor_non_yank_actions_break_alt_y_chain_and_rotation_persists() -> None:
    """
    non-yank actions break Alt+Y chain
    kill ring rotation persists after cycling
    Alt+Y cycles through kill ring after Ctrl+Y
    """
    editor = Editor(None, None)
    editor.set_text("first")
    editor.handle_input("\x17")
    editor.set_text("second")
    editor.handle_input("\x17")
    editor.set_text("")

    editor.handle_input("\x19")
    assert editor.get_text() == "second"
    editor.handle_input("x")
    assert editor.get_text() == "secondx"
    editor.handle_input("\x1by")
    assert editor.get_text() == "secondx"

    editor = Editor(None, None)
    editor.set_text("first")
    editor.handle_input("\x17")
    editor.set_text("second")
    editor.handle_input("\x17")
    editor.set_text("third")
    editor.handle_input("\x17")
    editor.set_text("")

    editor.handle_input("\x19")
    editor.handle_input("\x1by")
    assert editor.get_text() == "second"
    editor.handle_input("x")
    editor.set_text("")
    editor.handle_input("\x19")
    assert editor.get_text() == "second"


def test_editor_ctrl_y_does_nothing_when_kill_ring_is_empty() -> None:
    """Ctrl+Y does nothing when kill ring is empty"""
    editor = Editor(None, None)
    editor.set_text("hello")

    editor.handle_input("\x19")

    assert editor.get_text() == "hello"


def test_editor_non_delete_actions_break_kill_accumulation() -> None:
    """non-delete actions break kill accumulation"""
    editor = Editor(None, None)
    editor.set_text("first")
    editor.handle_input("\x17")
    assert editor.get_text() == ""

    editor.handle_input("x")
    editor.set_text("second")
    editor.handle_input("\x17")
    assert editor.get_text() == ""

    editor.set_text("")
    editor.handle_input("\x19")
    assert editor.get_text() == "second"


def test_editor_ctrl_u_accumulates_multiline_deletes_including_newlines() -> None:
    """Ctrl+U accumulates multiline deletes including newlines"""
    editor = Editor(None, None)
    editor.set_text("line1\nline2\nline3")

    editor.handle_input("\x15")
    assert editor.get_text() == "line1\nline2\n"
    editor.handle_input("\x15")
    assert editor.get_text() == "line1\nline2"
    editor.handle_input("\x15")
    assert editor.get_text() == "line1\n"
    editor.handle_input("\x15")
    assert editor.get_text() == "line1"
    editor.handle_input("\x15")
    assert editor.get_text() == ""

    editor.handle_input("\x19")
    assert editor.get_text() == "line1\nline2\nline3"


def test_editor_alt_d_accumulates_deleted_text_into_kill_ring() -> None:
    """Alt+D deletes word forward and saves to kill ring"""
    editor = Editor(None, None)
    editor.set_text("hello world test")
    editor.handle_input("\x01")

    editor.handle_input("\x1bd")
    assert editor.get_text() == " world test"

    editor.handle_input("\x1bd")
    assert editor.get_text() == " test"

    editor.handle_input("\x19")
    assert editor.get_text() == "hello world test"


def test_editor_alt_d_at_line_end_deletes_newline_and_yanks_it() -> None:
    """Alt+D at end of line deletes newline"""
    editor = Editor(None, None)
    editor.set_text("line1\nline2")
    editor.handle_input("\x1b[A")
    editor.handle_input("\x05")

    editor.handle_input("\x1bd")
    assert editor.get_text() == "line1line2"

    editor.handle_input("\x19")
    assert editor.get_text() == "line1\nline2"


def test_editor_yank_pop_replaces_only_yanked_text_in_middle_of_line() -> None:
    """
    handles yank in middle of text
    handles yank-pop in middle of text
    Alt+Y does nothing if not preceded by yank
    Alt+Y does nothing if kill ring has ≤1 entry
    """
    editor = Editor(None, None)
    editor.set_text("FIRST")
    editor.handle_input("\x17")
    editor.set_text("SECOND")
    editor.handle_input("\x17")

    editor.set_text("hello world")
    editor.handle_input("\x01")
    for _ in range(6):
        editor.handle_input("\x1b[C")

    editor.handle_input("\x19")
    assert editor.get_text() == "hello SECONDworld"

    editor.handle_input("\x1by")
    assert editor.get_text() == "hello FIRSTworld"


def test_editor_multiline_yank_and_yank_pop_in_middle_of_text() -> None:
    """multiline yank and yank-pop in middle of text"""
    editor = Editor(None, None)
    editor.set_text("FIRST\nSECOND")
    for _ in range(4):
        editor.handle_input("\x15")
    assert editor.get_text() == ""

    editor.set_text("THIRD\nFOURTH")
    for _ in range(4):
        editor.handle_input("\x15")
    assert editor.get_text() == ""

    editor.set_text("hello world")
    editor.handle_input("\x01")
    for _ in range(6):
        editor.handle_input("\x1b[C")

    editor.handle_input("\x19")
    assert editor.get_text() == "hello THIRD\nFOURTHworld"

    editor.handle_input("\x1by")
    assert editor.get_text() == "hello FIRST\nSECONDworld"


def test_editor_insert_text_at_cursor_normalizes_crlf_and_is_atomic_for_undo() -> None:
    """
    undoes insertTextAtCursor atomically
    insertTextAtCursor normalizes CRLF and CR line endings
    undoes multi-line paste atomically
    """
    editor = Editor(None, None)
    editor.insert_text_at_cursor("a\r\nb\r\nc")
    assert editor.get_text() == "a\nb\nc"

    editor.handle_input("\x1b[45;5u")
    assert editor.get_text() == ""


def test_editor_insert_text_at_cursor_normalizes_cr_line_endings_like_ts() -> None:
    """insertTextAtCursor handles multiline text"""
    editor = Editor(None, None)
    editor.insert_text_at_cursor("x\ry\rz")

    assert editor.get_text() == "x\ny\nz"


def test_editor_insert_text_at_cursor_normalizes_crlf_and_cr_line_endings_like_ts() -> None:
    """insertTextAtCursor normalizes CRLF and CR line endings"""
    editor = Editor(None, None)
    editor.insert_text_at_cursor("a\r\nb\r\nc")
    assert editor.get_text() == "a\nb\nc"

    editor.handle_input("\x1b[45;5u")
    assert editor.get_text() == ""

    editor.insert_text_at_cursor("x\ry\rz")
    assert editor.get_text() == "x\ny\nz"


def test_editor_set_text_is_undoable() -> None:
    """undoes setText to empty string"""
    editor = Editor(None, None)
    editor.set_text("hello world")
    editor.set_text("")

    assert editor.get_text() == ""

    editor.handle_input("\x1b[45;5u")
    assert editor.get_text() == "hello world"


def test_editor_undoes_set_text_to_empty_string_like_ts() -> None:
    """undoes setText to empty string"""
    editor = Editor(None, None)
    editor.set_text("hello world")
    editor.set_text("")

    assert editor.get_text() == ""

    editor.handle_input("\x1b[45;5u")
    assert editor.get_text() == "hello world"


def test_editor_word_movement_navigates_words_and_punctuation() -> None:
    editor = Editor(None, None)
    editor.set_text("foo bar... baz")

    editor.handle_input("\x1b[1;5D")
    assert editor.get_cursor() == {"line": 0, "col": 11}

    editor.handle_input("\x1b[1;5D")
    assert editor.get_cursor() == {"line": 0, "col": 7}

    editor.handle_input("\x1b[1;5D")
    assert editor.get_cursor() == {"line": 0, "col": 4}

    editor.handle_input("\x1b[1;5C")
    assert editor.get_cursor() == {"line": 0, "col": 7}

    editor.handle_input("\x1b[1;5C")
    assert editor.get_cursor() == {"line": 0, "col": 10}

    editor.handle_input("\x1b[1;5C")
    assert editor.get_cursor() == {"line": 0, "col": 14}


def test_editor_word_movement_resets_sticky_column() -> None:
    editor = Editor(None, None)
    editor.set_text("hello world\n\nhello world")

    editor.handle_input("\x1b[A")
    editor.handle_input("\x1b[A")
    assert editor.get_cursor() == {"line": 0, "col": 11}

    editor.handle_input("\x1b[1;5D")
    assert editor.get_cursor() == {"line": 0, "col": 6}

    editor.handle_input("\x1b[B")
    editor.handle_input("\x1b[B")
    assert editor.get_cursor() == {"line": 2, "col": 6}


def test_editor_left_arrow_resets_sticky_column() -> None:
    """resets sticky column on horizontal movement (left arrow)"""
    editor = Editor(None, None)
    editor.set_text("1234567890\n\n1234567890")

    editor.handle_input("\x01")
    for _ in range(8):
        editor.handle_input("\x1b[C")
    editor.handle_input("\x1b[A")
    editor.handle_input("\x1b[A")
    assert editor.get_cursor() == {"line": 0, "col": 8}

    editor.handle_input("\x1b[D")
    assert editor.get_cursor() == {"line": 0, "col": 7}

    editor.handle_input("\x1b[B")
    editor.handle_input("\x1b[B")
    assert editor.get_cursor() == {"line": 2, "col": 7}


def test_editor_right_arrow_resets_sticky_column() -> None:
    """resets sticky column on horizontal movement (right arrow)"""
    editor = Editor(None, None)
    editor.set_text("1234567890\n\n1234567890")

    editor.handle_input("\x01")
    for _ in range(7):
        editor.handle_input("\x1b[C")
    editor.handle_input("\x1b[A")
    editor.handle_input("\x1b[A")
    assert editor.get_cursor() == {"line": 0, "col": 7}

    editor.handle_input("\x1b[C")
    assert editor.get_cursor() == {"line": 0, "col": 8}

    editor.handle_input("\x1b[B")
    editor.handle_input("\x1b[B")
    assert editor.get_cursor() == {"line": 2, "col": 8}


def test_editor_coalesces_word_typing_into_undo_units() -> None:
    """coalesces consecutive word characters into one undo unit"""
    editor = Editor(None, None)
    for ch in "hello world":
        editor.handle_input(ch)

    assert editor.get_text() == "hello world"

    editor.handle_input("\x1b[45;5u")
    assert editor.get_text() == "hello"

    editor.handle_input("\x1b[45;5u")
    assert editor.get_text() == ""


def test_editor_undo_does_nothing_when_stack_is_empty() -> None:
    """does nothing when undo stack is empty"""
    editor = Editor(None, None)
    editor.handle_input("\x1b[45;5u")
    assert editor.get_text() == ""


def test_editor_undo_coalesces_consecutive_word_characters_into_one_unit() -> None:
    """coalesces consecutive word characters into one undo unit"""
    editor = Editor(None, None)
    for char in "hello world":
        editor.handle_input(char)

    editor.handle_input("\x1b[45;5u")
    assert editor.get_text() == "hello"

    editor.handle_input("\x1b[45;5u")
    assert editor.get_text() == ""


def test_editor_undoes_spaces_one_at_a_time() -> None:
    """undoes spaces one at a time"""
    editor = Editor(None, None)
    for char in "hello  ":
        editor.handle_input(char)

    editor.handle_input("\x1b[45;5u")
    assert editor.get_text() == "hello "
    editor.handle_input("\x1b[45;5u")
    assert editor.get_text() == "hello"
    editor.handle_input("\x1b[45;5u")
    assert editor.get_text() == ""


def test_editor_undoes_newlines_and_subsequent_word_as_separate_units() -> None:
    """undoes newlines and signals next word to capture state"""
    editor = Editor(None, None)
    for char in "hello":
        editor.handle_input(char)
    editor.handle_input("\n")
    for char in "world":
        editor.handle_input(char)

    assert editor.get_text() == "hello\nworld"
    editor.handle_input("\x1b[45;5u")
    assert editor.get_text() == "hello\n"
    editor.handle_input("\x1b[45;5u")
    assert editor.get_text() == "hello"
    editor.handle_input("\x1b[45;5u")
    assert editor.get_text() == ""


def test_editor_undoes_backspace_and_forward_delete() -> None:
    """
    undoes backspace
    undoes forward delete
    """
    editor = Editor(None, None)
    for char in "hello":
        editor.handle_input(char)
    editor.handle_input("\x7f")
    assert editor.get_text() == "hell"
    editor.handle_input("\x1b[45;5u")
    assert editor.get_text() == "hello"

    editor.handle_input("\x01")
    editor.handle_input("\x1b[C")
    editor.handle_input("\x1b[3~")
    assert editor.get_text() == "hllo"
    editor.handle_input("\x1b[45;5u")
    assert editor.get_text() == "hello"


def test_editor_undoes_yank() -> None:
    """undoes yank"""
    editor = Editor(None, None)
    for char in "hello ":
        editor.handle_input(char)
    editor.handle_input("\x17")
    editor.handle_input("\x19")

    assert editor.get_text() == "hello "
    editor.handle_input("\x1b[45;5u")
    assert editor.get_text() == ""


def test_editor_submit_clears_text_and_undo_stack() -> None:
    """clears undo stack on submit"""
    editor = Editor(None, None)
    submitted: list[str] = []
    editor.on_submit = submitted.append

    for char in "hello":
        editor.handle_input(char)
    editor.handle_input("\r")

    assert submitted == ["hello"]
    assert editor.get_text() == ""
    editor.handle_input("\x1b[45;5u")
    assert editor.get_text() == ""


def test_editor_undo_exits_history_browsing_mode_to_pre_history_state() -> None:
    """exits history browsing mode on undo"""
    editor = Editor(None, None)
    editor.add_to_history("hello")
    for char in "world":
        editor.handle_input(char)

    editor.handle_input("\x17")
    assert editor.get_text() == ""
    editor.handle_input("\x1b[A")
    assert editor.get_text() == "hello"

    editor.handle_input("\x1b[45;5u")
    assert editor.get_text() == ""
    editor.handle_input("\x1b[45;5u")
    assert editor.get_text() == "world"


def test_editor_undo_restores_pre_history_state_after_multiple_navigations() -> None:
    """undo restores to pre-history state even after multiple history navigations"""
    editor = Editor(None, None)
    editor.add_to_history("first")
    editor.add_to_history("second")
    editor.add_to_history("third")
    for char in "current":
        editor.handle_input(char)

    editor.handle_input("\x17")
    assert editor.get_text() == ""
    editor.handle_input("\x1b[A")
    editor.handle_input("\x1b[A")
    editor.handle_input("\x1b[A")
    assert editor.get_text() == "first"

    editor.handle_input("\x1b[45;5u")
    assert editor.get_text() == ""
    editor.handle_input("\x1b[45;5u")
    assert editor.get_text() == "current"


def test_editor_undoes_spaces_one_at_a_time() -> None:
    editor = Editor(None, None)
    for ch in "hello  ":
        editor.handle_input(ch)

    editor.handle_input("\x1b[45;5u")
    assert editor.get_text() == "hello "

    editor.handle_input("\x1b[45;5u")
    assert editor.get_text() == "hello"

    editor.handle_input("\x1b[45;5u")
    assert editor.get_text() == ""


def test_editor_cursor_movement_starts_new_undo_unit() -> None:
    """cursor movement starts new undo unit"""
    editor = Editor(None, None)
    for ch in "hello world":
        editor.handle_input(ch)

    for _ in range(5):
        editor.handle_input("\x1b[D")

    for ch in "lol":
        editor.handle_input(ch)

    assert editor.get_text() == "hello lolworld"

    editor.handle_input("\x1b[45;5u")
    assert editor.get_text() == "hello world"


def test_editor_no_op_delete_operations_do_not_push_extra_undo_snapshots() -> None:
    """no-op delete operations do not push undo snapshots"""
    editor = Editor(None, None)
    for ch in "hello":
        editor.handle_input(ch)

    editor.handle_input("\x17")
    assert editor.get_text() == ""
    editor.handle_input("\x17")
    editor.handle_input("\x17")

    editor.handle_input("\x1b[45;5u")
    assert editor.get_text() == "hello"


def test_editor_undoes_autocomplete_acceptance_to_original_prefix() -> None:
    """undoes autocomplete"""
    class _Provider:
        def get_suggestions(self, lines: list[str], cursor_line: int, cursor_col: int) -> dict[str, object] | None:
            prefix = lines[cursor_line][:cursor_col]
            if prefix == "di":
                return {
                    "items": [AutocompleteItem(value="dist/", label="dist/")],
                    "prefix": "di",
                }
            return None

        def apply_completion(
            self,
            lines: list[str],
            cursor_line: int,
            cursor_col: int,
            item: AutocompleteItem,
            prefix: str,
        ) -> dict[str, object]:
            line = lines[cursor_line]
            before = line[: cursor_col - len(prefix)]
            after = line[cursor_col :]
            new_lines = list(lines)
            new_lines[cursor_line] = before + item.value + after
            return {
                "lines": new_lines,
                "cursor_line": cursor_line,
                "cursor_col": cursor_col - len(prefix) + len(item.value),
            }

    editor = Editor(None, None)
    editor.set_autocomplete_provider(_Provider())

    editor.handle_input("d")
    editor.handle_input("i")
    assert editor.get_text() == "di"

    editor.handle_input("\t")
    assert editor.is_showing_autocomplete() is True

    editor.handle_input("\t")
    assert editor.get_text() == "dist/"

    editor.handle_input("\x1b[45;5u")
    assert editor.get_text() == "di"


def test_editor_large_paste_uses_marker_and_expands_on_submit() -> None:
    """
    expands large pasted content literally in getExpandedText
    submits large pasted content literally
    """
    editor = Editor(None, None)
    submitted: list[str] = []
    pasted_text = "\n".join(f"line {i}" for i in range(1, 12))
    editor.on_submit = submitted.append

    editor.handle_input(f"\x1b[200~{pasted_text}\x1b[201~")

    assert "[paste #" in editor.get_text()
    assert editor.get_expanded_text() == pasted_text

    editor.handle_input("\r")
    assert submitted == [pasted_text]


def test_editor_creates_a_paste_marker_for_large_pastes_like_ts() -> None:
    """creates a paste marker for large pastes"""
    editor = Editor(None, None)
    pasted_text = "\n".join(f"line {i}" for i in range(1, 12))

    editor.handle_input(f"\x1b[200~{pasted_text}\x1b[201~")

    assert re.search(r"\[paste #\d+ \+\d+ lines\]", editor.get_text())


def test_editor_treats_paste_marker_as_atomic_for_cursor_and_delete() -> None:
    """
    treats paste marker as single unit for right arrow
    treats paste marker as single unit for backspace
    """
    editor = Editor(None, None)
    pasted_text = "\n".join(f"line {i}" for i in range(20))
    editor.handle_input("A")
    editor.handle_input(f"\x1b[200~{pasted_text}\x1b[201~")
    editor.handle_input("B")

    marker = editor.get_text()[1:-1]

    editor.handle_input("\x01")
    editor.handle_input("\x1b[C")
    assert editor.get_cursor() == {"line": 0, "col": 1}

    editor.handle_input("\x1b[C")
    assert editor.get_cursor() == {"line": 0, "col": 1 + len(marker)}

    editor.handle_input("\x7f")
    assert editor.get_text() == "AB"


def test_editor_left_arrow_skips_paste_marker_atomically() -> None:
    """treats paste marker as single unit for left arrow"""
    editor = Editor(None, None)
    pasted_text = "\n".join(f"line {i}" for i in range(20))
    editor.handle_input("A")
    editor.handle_input(f"\x1b[200~{pasted_text}\x1b[201~")
    editor.handle_input("B")

    marker = editor.get_text()[1:-1]

    editor.handle_input("\x1b[D")
    assert editor.get_cursor() == {"line": 0, "col": 1 + len(marker)}

    editor.handle_input("\x1b[D")
    assert editor.get_cursor() == {"line": 0, "col": 1}

    editor.handle_input("\x1b[D")
    assert editor.get_cursor() == {"line": 0, "col": 0}


def test_editor_forward_delete_removes_entire_paste_marker() -> None:
    """treats paste marker as single unit for forward delete"""
    editor = Editor(None, None)
    pasted_text = "\n".join(f"line {i}" for i in range(20))
    editor.handle_input("A")
    editor.handle_input(f"\x1b[200~{pasted_text}\x1b[201~")
    editor.handle_input("B")

    editor.handle_input("\x01")
    editor.handle_input("\x1b[C")
    editor.handle_input("\x1b[3~")

    assert editor.get_text() == "AB"
    assert editor.get_cursor() == {"line": 0, "col": 1}


def test_editor_undo_restores_paste_marker_after_backspace_deletion() -> None:
    """undo restores marker after backspace deletion"""
    editor = Editor(None, None)
    pasted_text = "\n".join(f"line {i}" for i in range(20))
    editor.handle_input("A")
    editor.handle_input(f"\x1b[200~{pasted_text}\x1b[201~")
    editor.handle_input("B")
    text_before = editor.get_text()

    editor.handle_input("\x01")
    editor.handle_input("\x1b[C")
    editor.handle_input("\x1b[C")
    editor.handle_input("\x7f")
    assert editor.get_text() == "AB"

    editor.handle_input("\x1b[45;5u")
    assert editor.get_text() == text_before


def test_editor_word_movement_treats_paste_marker_as_single_unit() -> None:
    """treats paste marker as single unit for word movement"""
    editor = Editor(None, None)
    pasted_text = "\n".join(f"line {i}" for i in range(20))
    editor.handle_input("X")
    editor.handle_input(" ")
    editor.handle_input(f"\x1b[200~{pasted_text}\x1b[201~")
    marker = editor.get_text().split(" ", 1)[1]
    editor.handle_input(" ")
    editor.handle_input("Y")

    editor.handle_input("\x01")
    editor.handle_input("\x1b[1;5C")
    assert editor.get_cursor() == {"line": 0, "col": 1}

    editor.handle_input("\x1b[1;5C")
    assert editor.get_cursor() == {"line": 0, "col": 2 + len(marker)}


def test_editor_handles_multiple_paste_markers_atomically() -> None:
    """handles multiple paste markers in same line"""
    editor = Editor(None, None)
    pasted_text = "\n".join(f"line {i}" for i in range(20))
    editor.handle_input(f"\x1b[200~{pasted_text}\x1b[201~")
    first_marker = editor.get_text()
    editor.handle_input(" ")
    editor.handle_input(f"\x1b[200~{pasted_text}\x1b[201~")
    second_marker = editor.get_text()[len(first_marker) + 1 :]

    editor.handle_input("\x01")
    editor.handle_input("\x1b[C")
    assert editor.get_cursor() == {"line": 0, "col": len(first_marker)}

    editor.handle_input("\x1b[C")
    assert editor.get_cursor() == {"line": 0, "col": len(first_marker) + 1}

    editor.handle_input("\x1b[C")
    assert editor.get_cursor() == {"line": 0, "col": len(first_marker) + 1 + len(second_marker)}


def test_editor_right_arrow_skips_multiple_paste_markers_in_same_line() -> None:
    """handles multiple paste markers in same line"""
    editor = Editor(None, None)
    pasted_text = "\n".join(f"line {i}" for i in range(20))
    editor.handle_input(f"\x1b[200~{pasted_text}\x1b[201~")
    first_marker = editor.get_text()
    editor.handle_input(" ")
    editor.handle_input(f"\x1b[200~{pasted_text}\x1b[201~")
    second_marker = editor.get_text()[len(first_marker) + 1 :]

    editor.handle_input("\x01")
    editor.handle_input("\x1b[C")
    assert editor.get_cursor() == {"line": 0, "col": len(first_marker)}

    editor.handle_input("\x1b[C")
    assert editor.get_cursor() == {"line": 0, "col": len(first_marker) + 1}

    editor.handle_input("\x1b[C")
    assert editor.get_cursor() == {"line": 0, "col": len(first_marker) + 1 + len(second_marker)}


def test_editor_manually_typed_marker_like_text_is_not_atomic() -> None:
    """does not treat manually typed marker-like text as atomic (no valid paste ID)"""
    editor = Editor(None, None)
    fake_marker = "[paste #99 +5 lines]"
    editor.set_text(fake_marker)

    editor.handle_input("\x01")
    editor.handle_input("\x1b[C")

    assert editor.get_cursor() == {"line": 0, "col": 1}


def test_editor_large_single_line_paste_uses_char_marker_and_expands_on_submit() -> None:
    editor = Editor(None, None)
    submitted: list[str] = []
    pasted_text = "x" * 1001
    editor.on_submit = submitted.append

    editor.handle_input(f"\x1b[200~{pasted_text}\x1b[201~")

    assert editor.get_text() == "[paste #1 1001 chars]"
    assert editor.get_expanded_text() == pasted_text

    editor.handle_input("\r")
    assert submitted == [pasted_text]


def test_editor_get_expanded_text_returns_large_paste_literally() -> None:
    """expands large pasted content literally in getExpandedText"""
    editor = Editor(None, None)
    pasted_text = "\n".join(
        [
            "line 1",
            "line 2",
            "line 3",
            "line 4",
            "line 5",
            "line 6",
            "line 7",
            "line 8",
            "line 9",
            "line 10",
            "tokens $1 $2 $& $$ $` $' end",
        ]
    )

    editor.handle_input(f"\x1b[200~{pasted_text}\x1b[201~")

    assert "[paste #" in editor.get_text()
    assert editor.get_expanded_text() == pasted_text


def test_editor_submit_returns_large_paste_literally() -> None:
    """submits large pasted content literally"""
    editor = Editor(None, None)
    pasted_text = "\n".join(
        [
            "line 1",
            "line 2",
            "line 3",
            "line 4",
            "line 5",
            "line 6",
            "line 7",
            "line 8",
            "line 9",
            "line 10",
            "tokens $1 $2 $& $$ $` $' end",
        ]
    )
    submitted: list[str] = []
    editor.on_submit = submitted.append

    editor.handle_input(f"\x1b[200~{pasted_text}\x1b[201~")
    editor.handle_input("\r")

    assert submitted == [pasted_text]


def test_editor_treats_char_paste_marker_as_atomic_for_cursor_and_delete() -> None:
    editor = Editor(None, None)
    pasted_text = "x" * 1001
    editor.handle_input("A")
    editor.handle_input(f"\x1b[200~{pasted_text}\x1b[201~")
    editor.handle_input("B")

    marker = editor.get_text()[1:-1]
    assert marker == "[paste #1 1001 chars]"

    editor.handle_input("\x01")
    editor.handle_input("\x1b[C")
    assert editor.get_cursor() == {"line": 0, "col": 1}

    editor.handle_input("\x1b[C")
    assert editor.get_cursor() == {"line": 0, "col": 1 + len(marker)}

    editor.handle_input("\x7f")
    assert editor.get_text() == "AB"


def test_editor_undoes_single_line_paste_atomically() -> None:
    """undoes single-line paste atomically"""
    editor = Editor(None, None)
    editor.set_text("hello world")
    editor.handle_input("\x01")
    for _ in range(5):
        editor.handle_input("\x1b[C")

    editor.handle_input("\x1b[200~beep boop\x1b[201~")
    assert editor.get_text() == "hellobeep boop world"

    editor.handle_input("\x1b[45;5u")
    assert editor.get_text() == "hello world"


def test_editor_undoes_multi_line_paste_atomically() -> None:
    """undoes multi-line paste atomically"""
    editor = Editor(None, None)
    editor.set_text("hello world")
    editor.handle_input("\x01")
    for _ in range(5):
        editor.handle_input("\x1b[C")

    editor.handle_input("\x1b[200~line1\nline2\nline3\x1b[201~")
    assert editor.get_text() == "helloline1\nline2\nline3 world"
    assert editor.get_cursor() == {"line": 2, "col": 5}

    editor.handle_input("\x1b[45;5u")
    assert editor.get_text() == "hello world"


def test_editor_insert_text_at_cursor_handles_multiline_text_atomically() -> None:
    editor = Editor(None, None)
    editor.set_text("hello world")
    editor.handle_input("\x01")
    for _ in range(5):
        editor.handle_input("\x1b[C")

    editor.insert_text_at_cursor("line1\nline2\nline3")
    assert editor.get_text() == "helloline1\nline2\nline3 world"
    assert editor.get_cursor() == {"line": 2, "col": 5}

    editor.handle_input("\x1b[45;5u")
    assert editor.get_text() == "hello world"


def test_editor_undoes_insert_text_at_cursor_atomically_like_ts() -> None:
    editor = Editor(None, None)
    editor.set_text("hello world")
    editor.handle_input("\x01")
    for _ in range(5):
        editor.handle_input("\x1b[C")

    editor.insert_text_at_cursor("/tmp/image.png")
    assert editor.get_text() == "hello/tmp/image.png world"

    editor.handle_input("\x1b[45;5u")
    assert editor.get_text() == "hello world"


def test_editor_insert_text_at_cursor_handles_multiline_text_like_ts() -> None:
    editor = Editor(None, None)
    editor.set_text("hello world")
    editor.handle_input("\x01")
    for _ in range(5):
        editor.handle_input("\x1b[C")

    editor.insert_text_at_cursor("line1\nline2\nline3")

    assert editor.get_text() == "helloline1\nline2\nline3 world"
    assert editor.get_cursor() == {"line": 2, "col": 5}


def test_editor_autocomplete_tab_show_accept_and_undo() -> None:
    class _Provider:
        def get_suggestions(self, lines: list[str], _cursor_line: int, cursor_col: int):
            prefix = lines[0][:cursor_col]
            if prefix == "di":
                return {
                    "items": [AutocompleteItem(value="dist/", label="dist/")],
                    "prefix": "di",
                }
            return None

        def apply_completion(
            self,
            lines: list[str],
            cursor_line: int,
            cursor_col: int,
            item: AutocompleteItem,
            prefix: str,
        ):
            line = lines[cursor_line]
            before = line[: cursor_col - len(prefix)]
            after = line[cursor_col:]
            new_line = before + item.value + after
            new_lines = list(lines)
            new_lines[cursor_line] = new_line
            return {
                "lines": new_lines,
                "cursor_line": cursor_line,
                "cursor_col": len(before) + len(item.value),
            }

    editor = Editor(None, None)
    editor.set_autocomplete_provider(_Provider())
    editor.handle_input("d")
    editor.handle_input("i")

    editor.handle_input("\t")
    assert editor.is_showing_autocomplete() is True

    editor.handle_input("\t")
    assert editor.get_text() == "dist/"
    assert editor.is_showing_autocomplete() is False

    editor.handle_input("\x1b[45;5u")
    assert editor.get_text() == "di"


def test_editor_hides_slash_autocomplete_when_backspacing_to_empty() -> None:
    """hides autocomplete when backspacing slash command to empty"""
    class _Provider:
        def get_suggestions(self, lines: list[str], _cursor_line: int, cursor_col: int):
            prefix = lines[0][:cursor_col]
            if prefix.startswith("/"):
                return {
                    "items": [
                        AutocompleteItem(value="/help", label="help"),
                        AutocompleteItem(value="/model", label="model"),
                    ],
                    "prefix": prefix,
                }
            return None

        def apply_completion(self, lines, cursor_line, cursor_col, item, prefix):  # noqa: ANN001
            line = lines[cursor_line]
            before = line[: cursor_col - len(prefix)]
            new_line = before + item.value + " "
            new_lines = list(lines)
            new_lines[cursor_line] = new_line
            return {"lines": new_lines, "cursor_line": cursor_line, "cursor_col": len(new_line)}

    editor = Editor(None, None)
    editor.set_autocomplete_provider(_Provider())

    editor.handle_input("/")
    assert editor.is_showing_autocomplete() is True

    editor.handle_input("\x7f")
    assert editor.get_text() == ""
    assert editor.is_showing_autocomplete() is False


def test_editor_force_mode_keeps_autocomplete_open_while_typing() -> None:
    """keeps suggestions open when typing in force mode (Tab-triggered)"""
    class _Provider:
        def get_suggestions(self, lines: list[str], _cursor_line: int, cursor_col: int):
            prefix = lines[0][:cursor_col]
            if prefix in {"", "r", "re"}:
                items = [
                    AutocompleteItem(value="readme.md", label="readme.md"),
                    AutocompleteItem(value="src/", label="src/"),
                ]
                filtered = [item for item in items if item.value.startswith(prefix)]
                return {"items": filtered, "prefix": prefix} if filtered else None
            return None

        def get_force_file_suggestions(self, lines: list[str], _cursor_line: int, cursor_col: int):
            prefix = lines[0][:cursor_col]
            items = [
                AutocompleteItem(value="readme.md", label="readme.md"),
                AutocompleteItem(value="src/", label="src/"),
            ]
            filtered = [item for item in items if item.value.startswith(prefix)]
            return {"items": filtered, "prefix": prefix} if filtered else None

        def apply_completion(self, lines, cursor_line, cursor_col, item, prefix):  # noqa: ANN001
            line = lines[cursor_line]
            before = line[: cursor_col - len(prefix)]
            after = line[cursor_col:]
            new_line = before + item.value + after
            new_lines = list(lines)
            new_lines[cursor_line] = new_line
            return {
                "lines": new_lines,
                "cursor_line": cursor_line,
                "cursor_col": len(before) + len(item.value),
            }

    editor = Editor(None, None)
    editor.set_autocomplete_provider(_Provider())

    editor.handle_input("\t")
    assert editor.is_showing_autocomplete() is True

    editor.handle_input("r")
    assert editor.get_text() == "r"
    assert editor.is_showing_autocomplete() is True

    editor.handle_input("e")
    assert editor.get_text() == "re"
    assert editor.is_showing_autocomplete() is True

    editor.handle_input("\t")
    assert editor.get_text() == "readme.md"
    assert editor.is_showing_autocomplete() is False


def test_editor_force_file_single_suggestion_applies_without_showing_menu() -> None:
    """auto-applies single force-file suggestion without showing menu"""
    class _Provider:
        def get_suggestions(self, lines: list[str], _cursor_line: int, cursor_col: int):  # noqa: ARG002
            return None

        def get_force_file_suggestions(self, lines: list[str], _cursor_line: int, cursor_col: int):
            prefix = lines[0][:cursor_col]
            if prefix == "Work":
                return {
                    "items": [AutocompleteItem(value="Workspace/", label="Workspace/")],
                    "prefix": "Work",
                }
            return None

        def apply_completion(self, lines, cursor_line, cursor_col, item, prefix):  # noqa: ANN001
            line = lines[cursor_line]
            before = line[: cursor_col - len(prefix)]
            after = line[cursor_col:]
            new_line = before + item.value + after
            new_lines = list(lines)
            new_lines[cursor_line] = new_line
            return {
                "lines": new_lines,
                "cursor_line": cursor_line,
                "cursor_col": len(before) + len(item.value),
            }

    editor = Editor(None, None)
    editor.set_autocomplete_provider(_Provider())
    for ch in "Work":
        editor.handle_input(ch)

    editor.handle_input("\t")

    assert editor.get_text() == "Workspace/"
    assert editor.is_showing_autocomplete() is False


def test_editor_force_file_multiple_suggestions_show_menu() -> None:
    """shows menu when force-file has multiple suggestions"""
    class _Provider:
        def get_suggestions(self, lines: list[str], _cursor_line: int, cursor_col: int):  # noqa: ARG002
            return None

        def get_force_file_suggestions(self, lines: list[str], _cursor_line: int, cursor_col: int):
            prefix = lines[0][:cursor_col]
            if prefix == "r":
                return {
                    "items": [
                        AutocompleteItem(value="readme.md", label="readme.md"),
                        AutocompleteItem(value="routes.py", label="routes.py"),
                    ],
                    "prefix": "r",
                }
            return None

        def apply_completion(self, lines, cursor_line, cursor_col, item, prefix):  # noqa: ANN001
            line = lines[cursor_line]
            before = line[: cursor_col - len(prefix)]
            after = line[cursor_col:]
            new_line = before + item.value + after
            new_lines = list(lines)
            new_lines[cursor_line] = new_line
            return {
                "lines": new_lines,
                "cursor_line": cursor_line,
                "cursor_col": len(before) + len(item.value),
            }

    editor = Editor(None, None)
    editor.set_autocomplete_provider(_Provider())
    editor.handle_input("r")
    editor.handle_input("\t")

    assert editor.get_text() == "r"
    assert editor.is_showing_autocomplete() is True


def test_editor_paste_file_path_inserts_leading_space_after_word_character() -> None:
    editor = Editor(None, None)
    editor.set_text("open")

    editor.handle_input("\x1b[200~/tmp/file.txt\x1b[201~")

    assert editor.get_text() == "open /tmp/file.txt"


def test_combined_autocomplete_provider_chains_command_and_argument_completion() -> None:
    """chains into argument completions after tab-completing slash command names"""
    from paw.pi_agent.tui import CombinedAutocompleteProvider, SlashCommand

    provider = CombinedAutocompleteProvider(
        [
            SlashCommand(value="model", label="model", description="Switch model", get_argument_completions=lambda prefix: [
                AutocompleteItem(value="claude-opus", label="claude-opus"),
                AutocompleteItem(value="claude-sonnet", label="claude-sonnet"),
            ] if "claude-opus".startswith(prefix) or "claude-sonnet".startswith(prefix) else []),
            SlashCommand(value="help", label="help", description="Show help"),
        ]
    )

    editor = Editor(None, None)
    editor.set_autocomplete_provider(provider)

    for ch in "/mod":
        editor.handle_input(ch)
    assert editor.is_showing_autocomplete() is True

    editor.handle_input("\t")
    assert editor.get_text() == "/model "
    assert editor.is_showing_autocomplete() is True

    editor.handle_input("\t")
    assert editor.get_text() == "/model claude-opus"
    assert editor.is_showing_autocomplete() is False


def test_editor_does_not_show_argument_completions_when_command_has_no_argument_completer() -> None:
    """does not show argument completions when command has no argument completer"""
    from paw.pi_agent.tui import CombinedAutocompleteProvider, SlashCommand

    provider = CombinedAutocompleteProvider(
        [
            SlashCommand(value="help", label="help", description="Show help"),
            SlashCommand(
                value="model",
                label="model",
                description="Switch model",
                get_argument_completions=lambda _prefix: [AutocompleteItem(value="claude-opus", label="claude-opus")],
            ),
        ]
    )

    editor = Editor(None, None)
    editor.set_autocomplete_provider(provider)

    for ch in "/he":
        editor.handle_input(ch)

    assert editor.is_showing_autocomplete() is True
    editor.handle_input("\t")
    assert editor.get_text() == "/help "
    assert editor.is_showing_autocomplete() is False


def test_combined_autocomplete_provider_applies_slash_command_with_trailing_space() -> None:
    from paw.pi_agent.tui import CombinedAutocompleteProvider, SlashCommand

    provider = CombinedAutocompleteProvider([SlashCommand(value="model", label="model", description="Switch model")])
    result = provider.get_suggestions(["/mo"], 0, 3)

    assert result is not None
    applied = provider.apply_completion(["/mo"], 0, 3, result["items"][0], result["prefix"])
    assert applied["lines"] == ["/model "]
    assert applied["cursor_col"] == len("/model ")


def test_combined_autocomplete_provider_delegates_force_file_suggestions() -> None:
    from paw.pi_agent.tui import CombinedAutocompleteProvider

    class _Provider:
        def get_suggestions(self, lines, cursor_line, cursor_col):  # noqa: ANN001
            return None

        def get_force_file_suggestions(self, lines, cursor_line, cursor_col):  # noqa: ANN001
            return {
                "items": [AutocompleteItem(value="./src/", label="./src/")],
                "prefix": "./s",
            }

        def apply_completion(self, lines, cursor_line, cursor_col, item, prefix):  # noqa: ANN001
            return {
                "lines": ["./src/"],
                "cursor_line": 0,
                "cursor_col": len("./src/"),
            }

    provider = CombinedAutocompleteProvider([_Provider()])
    result = provider.get_force_file_suggestions(["./s"], 0, 3)

    assert result is not None
    assert result["prefix"] == "./s"
    assert result["items"][0].value == "./src/"


def test_combined_autocomplete_provider_extracts_force_file_prefixes_like_ts() -> None:
    """
    extracts / from 'hey /' when forced
    extracts /A from '/A' when forced
    does not trigger for slash commands
    triggers for absolute paths after slash command argument
    """
    from paw.pi_agent.tui import CombinedAutocompleteProvider

    provider = CombinedAutocompleteProvider([], "/tmp")

    result = provider.get_force_file_suggestions(["hey /"], 0, 5)
    assert result is not None
    assert result["prefix"] == "/"

    result = provider.get_force_file_suggestions(["/model"], 0, 6)
    assert result is None

    result = provider.get_force_file_suggestions(["/command /"], 0, 10)
    assert result is not None
    assert result["prefix"] == "/"

    result = provider.get_force_file_suggestions(["/A"], 0, 2)
    if result is not None:
        assert result["prefix"] == "/A"


def test_combined_autocomplete_provider_preserves_dot_slash_and_quoted_path_completion() -> None:
    from paw.pi_agent.tui import CombinedAutocompleteProvider

    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "update.sh"), "w", encoding="utf-8") as handle:
            handle.write("#!/bin/bash\n")
        os.mkdir(os.path.join(tmpdir, "src"))
        with open(os.path.join(tmpdir, "src", "index.ts"), "w", encoding="utf-8") as handle:
            handle.write("export {};\n")
        os.mkdir(os.path.join(tmpdir, "my folder"))
        with open(os.path.join(tmpdir, "my folder", "test.txt"), "w", encoding="utf-8") as handle:
            handle.write("content\n")

        provider = CombinedAutocompleteProvider([], tmpdir)

        result = provider.get_force_file_suggestions(["./up"], 0, 4)
        assert result is not None
        assert any(item.value == "./update.sh" for item in result["items"])

        result = provider.get_force_file_suggestions(["./sr"], 0, 4)
        assert result is not None
        assert any(item.value == "./src/" for item in result["items"])

        result = provider.get_force_file_suggestions(['"my'], 0, 3)
        assert result is not None
        assert any(item.value == '"my folder/"' for item in result["items"])

        line = '"my folder/te"'
        cursor_col = len(line) - 1
        result = provider.get_force_file_suggestions([line], 0, cursor_col)
        assert result is not None
        item = next(entry for entry in result["items"] if entry.value == '"my folder/test.txt"')
        applied = provider.apply_completion([line], 0, cursor_col, item, result["prefix"])
        assert applied["lines"] == ['"my folder/test.txt"']


def test_combined_autocomplete_provider_supports_basic_at_file_suggestions() -> None:
    from paw.pi_agent.tui import CombinedAutocompleteProvider

    with tempfile.TemporaryDirectory() as tmpdir:
        os.mkdir(os.path.join(tmpdir, "src"))
        with open(os.path.join(tmpdir, "README.md"), "w", encoding="utf-8") as handle:
            handle.write("readme\n")
        with open(os.path.join(tmpdir, "src.txt"), "w", encoding="utf-8") as handle:
            handle.write("text\n")
        os.mkdir(os.path.join(tmpdir, "my folder"))
        os.mkdir(os.path.join(tmpdir, ".pi"))
        os.mkdir(os.path.join(tmpdir, ".github"))
        os.mkdir(os.path.join(tmpdir, ".git"))

        provider = CombinedAutocompleteProvider([], tmpdir)

        result = provider.get_suggestions(["@"], 0, 1)
        assert result is not None
        values = [item.value for item in result["items"]]
        assert "@README.md" in values
        assert "@src/" in values
        assert "@.pi/" in values
        assert "@.github/" in values
        assert not any(value == "@.git" or value.startswith("@.git/") for value in values)

        result = provider.get_suggestions(["@re"], 0, 3)
        assert result is not None
        assert [item.value for item in result["items"]] == ["@README.md"]

        result = provider.get_suggestions(["@file.txt"], 0, 9)
        assert result is None

        with open(os.path.join(tmpdir, "file.txt"), "w", encoding="utf-8") as handle:
            handle.write("content\n")
        result = provider.get_suggestions(["@file.txt"], 0, 9)
        assert result is not None
        assert "@file.txt" in [item.value for item in result["items"]]

        result = provider.get_suggestions(["@src"], 0, 4)
        assert result is not None
        values = [item.value for item in result["items"]]
        assert values[0] == "@src/"
        assert "@src.txt" in values

        result = provider.get_suggestions(["@my"], 0, 3)
        assert result is not None
        assert any(item.value == '@"my folder/"' for item in result["items"])


def test_combined_autocomplete_provider_supports_quoted_at_path_continuation() -> None:
    from paw.pi_agent.tui import CombinedAutocompleteProvider

    with tempfile.TemporaryDirectory() as tmpdir:
        os.mkdir(os.path.join(tmpdir, "my folder"))
        with open(os.path.join(tmpdir, "my folder", "test.txt"), "w", encoding="utf-8") as handle:
            handle.write("content\n")
        with open(os.path.join(tmpdir, "my folder", "other.txt"), "w", encoding="utf-8") as handle:
            handle.write("content\n")

        provider = CombinedAutocompleteProvider([], tmpdir)

        line = '@"my folder/"'
        result = provider.get_suggestions([line], 0, len(line) - 1)
        assert result is not None
        values = [item.value for item in result["items"]]
        assert '@"my folder/test.txt"' in values
        assert '@"my folder/other.txt"' in values

        line = '@"my folder/te"'
        cursor_col = len(line) - 1
        result = provider.get_suggestions([line], 0, cursor_col)
        assert result is not None
        item = next(entry for entry in result["items"] if entry.value == '@"my folder/test.txt"')
        applied = provider.apply_completion([line], 0, cursor_col, item, result["prefix"])
        assert applied["lines"] == ['@"my folder/test.txt" ']


def test_combined_autocomplete_provider_supports_direct_quoted_path_continuation() -> None:
    from paw.pi_agent.tui import CombinedAutocompleteProvider

    with tempfile.TemporaryDirectory() as tmpdir:
        os.mkdir(os.path.join(tmpdir, "my folder"))
        with open(os.path.join(tmpdir, "my folder", "test.txt"), "w", encoding="utf-8") as handle:
            handle.write("content\n")
        with open(os.path.join(tmpdir, "my folder", "other.txt"), "w", encoding="utf-8") as handle:
            handle.write("content\n")

        provider = CombinedAutocompleteProvider([], tmpdir)

        line = '"my folder/"'
        result = provider.get_force_file_suggestions([line], 0, len(line) - 1)
        assert result is not None
        values = [item.value for item in result["items"]]
        assert '"my folder/test.txt"' in values
        assert '"my folder/other.txt"' in values

        line = '"my folder/te"'
        cursor_col = len(line) - 1
        result = provider.get_force_file_suggestions([line], 0, cursor_col)
        assert result is not None
        item = next(entry for entry in result["items"] if entry.value == '"my folder/test.txt"')
        applied = provider.apply_completion([line], 0, cursor_col, item, result["prefix"])
        assert applied["lines"] == ['"my folder/test.txt"']


def test_combined_autocomplete_provider_supports_nested_and_scoped_at_search() -> None:
    from paw.pi_agent.tui import CombinedAutocompleteProvider

    with tempfile.TemporaryDirectory() as rootdir:
        base_dir = os.path.join(rootdir, "cwd")
        outside_dir = os.path.join(rootdir, "outside")
        os.mkdir(base_dir)
        os.mkdir(outside_dir)

        os.makedirs(os.path.join(base_dir, "src"), exist_ok=True)
        with open(os.path.join(base_dir, "src", "index.ts"), "w", encoding="utf-8") as handle:
            handle.write("export {};\n")

        os.makedirs(os.path.join(base_dir, "packages", "tui", "src"), exist_ok=True)
        os.makedirs(os.path.join(base_dir, "packages", "ai", "src"), exist_ok=True)
        with open(os.path.join(base_dir, "packages", "tui", "src", "autocomplete.ts"), "w", encoding="utf-8") as handle:
            handle.write("export {};\n")
        with open(os.path.join(base_dir, "packages", "ai", "src", "autocomplete.ts"), "w", encoding="utf-8") as handle:
            handle.write("export {};\n")

        os.makedirs(os.path.join(base_dir, "src", "components"), exist_ok=True)
        os.makedirs(os.path.join(base_dir, "src", "utils"), exist_ok=True)
        with open(os.path.join(base_dir, "src", "components", "Button.tsx"), "w", encoding="utf-8") as handle:
            handle.write("export {};\n")
        with open(os.path.join(base_dir, "src", "utils", "helpers.ts"), "w", encoding="utf-8") as handle:
            handle.write("export {};\n")

        os.makedirs(os.path.join(outside_dir, "nested", "deeper"), exist_ok=True)
        with open(os.path.join(outside_dir, "nested", "alpha.ts"), "w", encoding="utf-8") as handle:
            handle.write("export {};\n")
        with open(os.path.join(outside_dir, "nested", "deeper", "also-alpha.ts"), "w", encoding="utf-8") as handle:
            handle.write("export {};\n")
        with open(os.path.join(outside_dir, "nested", "deeper", "zzz.ts"), "w", encoding="utf-8") as handle:
            handle.write("export {};\n")

        provider = CombinedAutocompleteProvider([], base_dir)

        result = provider.get_suggestions(["@index"], 0, 6)
        assert result is not None
        assert any(item.value == "@src/index.ts" for item in result["items"])

        result = provider.get_suggestions(["@tui/src/auto"], 0, 13)
        assert result is not None
        values = [item.value for item in result["items"]]
        assert "@packages/tui/src/autocomplete.ts" in values
        assert "@packages/ai/src/autocomplete.ts" not in values

        result = provider.get_suggestions(["@components/"], 0, 12)
        assert result is not None
        values = [item.value for item in result["items"]]
        assert "@src/components/Button.tsx" in values
        assert "@src/utils/helpers.ts" not in values

        result = provider.get_suggestions(["@../outside/a"], 0, 13)
        assert result is not None
        values = [item.value for item in result["items"]]
        assert "@../outside/nested/alpha.ts" in values
        assert "@../outside/nested/deeper/also-alpha.ts" in values
        assert "@../outside/nested/deeper/zzz.ts" not in values


def test_editor_enter_keeps_exact_typed_autocomplete_value() -> None:
    class _Provider:
        def get_suggestions(self, lines: list[str], _cursor_line: int, cursor_col: int):
            before = lines[0][:cursor_col]
            match before:
                case "/argtest two":
                    return {
                        "items": [
                            AutocompleteItem(value="one", label="one"),
                            AutocompleteItem(value="two", label="two"),
                            AutocompleteItem(value="three", label="three"),
                        ],
                        "prefix": "two",
                    }
                case _:
                    return None

        def apply_completion(self, lines, cursor_line, cursor_col, item, prefix):  # noqa: ANN001
            line = lines[cursor_line]
            before = line[: cursor_col - len(prefix)]
            new_line = before + item.value
            new_lines = list(lines)
            new_lines[cursor_line] = new_line
            return {"lines": new_lines, "cursor_line": cursor_line, "cursor_col": len(new_line)}

    editor = Editor(None, None)
    editor.set_autocomplete_provider(_Provider())
    for ch in "/argtest two":
        editor.handle_input(ch)

    assert editor.is_showing_autocomplete() is True
    editor.handle_input("\r")
    assert editor.get_text() == "/argtest two"


def test_editor_exact_typed_autocomplete_value_stays_visible_before_enter() -> None:
    class _Provider:
        def get_suggestions(self, lines: list[str], _cursor_line: int, cursor_col: int):
            before = lines[0][:cursor_col]
            match before:
                case "/argtest two":
                    return {
                        "items": [
                            AutocompleteItem(value="one", label="one"),
                            AutocompleteItem(value="two", label="two"),
                            AutocompleteItem(value="three", label="three"),
                        ],
                        "prefix": "two",
                    }
                case _:
                    return None

        def apply_completion(self, lines, cursor_line, cursor_col, item, prefix):  # noqa: ANN001
            line = lines[cursor_line]
            before = line[: cursor_col - len(prefix)]
            new_line = before + item.value
            new_lines = list(lines)
            new_lines[cursor_line] = new_line
            return {"lines": new_lines, "cursor_line": cursor_line, "cursor_col": len(new_line)}

    editor = Editor(None, None)
    editor.set_autocomplete_provider(_Provider())
    for ch in "/argtest two":
        editor.handle_input(ch)

    assert editor.get_text() == "/argtest two"
    assert editor.is_showing_autocomplete() is True


def test_editor_applies_exact_typed_slash_argument_value_on_enter_even_when_first_item_is_highlighted_like_ts() -> None:
    """applies exact typed slash-argument value on Enter even when first item is highlighted"""
    class _Provider:
        def get_suggestions(self, lines: list[str], _cursor_line: int, cursor_col: int):
            before = lines[0][:cursor_col]
            if before == "/argtest two":
                return {
                    "items": [
                        AutocompleteItem(value="one", label="one"),
                        AutocompleteItem(value="two", label="two"),
                        AutocompleteItem(value="three", label="three"),
                    ],
                    "prefix": "two",
                }
            return None

        def apply_completion(self, lines, cursor_line, cursor_col, item, prefix):  # noqa: ANN001
            line = lines[cursor_line]
            before = line[: cursor_col - len(prefix)]
            new_line = before + item.value
            new_lines = list(lines)
            new_lines[cursor_line] = new_line
            return {"lines": new_lines, "cursor_line": cursor_line, "cursor_col": len(new_line)}

    editor = Editor(None, None)
    editor.set_autocomplete_provider(_Provider())
    for ch in "/argtest two":
        editor.handle_input(ch)

    assert editor.get_text() == "/argtest two"
    assert editor.is_showing_autocomplete() is True
    editor.handle_input("\r")
    assert editor.get_text() == "/argtest two"


def test_editor_enter_applies_first_prefix_match_for_autocomplete() -> None:
    class _Provider:
        def get_suggestions(self, lines: list[str], _cursor_line: int, cursor_col: int):
            before = lines[0][:cursor_col]
            if before == "/argtest t":
                return {
                    "items": [
                        AutocompleteItem(value="two", label="two"),
                        AutocompleteItem(value="three", label="three"),
                    ],
                    "prefix": "t",
                }
            return None

        def apply_completion(self, lines, cursor_line, cursor_col, item, prefix):  # noqa: ANN001
            line = lines[cursor_line]
            before = line[: cursor_col - len(prefix)]
            new_line = before + item.value
            new_lines = list(lines)
            new_lines[cursor_line] = new_line
            return {"lines": new_lines, "cursor_line": cursor_line, "cursor_col": len(new_line)}

    editor = Editor(None, None)
    editor.set_autocomplete_provider(_Provider())
    for ch in "/argtest t":
        editor.handle_input(ch)

    assert editor.is_showing_autocomplete() is True
    editor.handle_input("\r")
    assert editor.get_text() == "/argtest two"


def test_editor_enter_applies_unique_prefix_match_from_unfiltered_items() -> None:
    class _Provider:
        def get_suggestions(self, lines: list[str], _cursor_line: int, cursor_col: int):
            before = lines[0][:cursor_col]
            if before == "/argtest tw":
                return {
                    "items": [
                        AutocompleteItem(value="one", label="one"),
                        AutocompleteItem(value="two", label="two"),
                        AutocompleteItem(value="three", label="three"),
                    ],
                    "prefix": "tw",
                }
            return None

        def apply_completion(self, lines, cursor_line, cursor_col, item, prefix):  # noqa: ANN001
            line = lines[cursor_line]
            before = line[: cursor_col - len(prefix)]
            new_line = before + item.value
            new_lines = list(lines)
            new_lines[cursor_line] = new_line
            return {"lines": new_lines, "cursor_line": cursor_line, "cursor_col": len(new_line)}

    editor = Editor(None, None)
    editor.set_autocomplete_provider(_Provider())
    for ch in "/argtest tw":
        editor.handle_input(ch)

    assert editor.is_showing_autocomplete() is True
    editor.handle_input("\r")
    assert editor.get_text() == "/argtest two"


def test_editor_unique_prefix_match_stays_visible_while_typing_before_enter() -> None:
    class _Provider:
        def get_suggestions(self, lines: list[str], _cursor_line: int, cursor_col: int):
            before = lines[0][:cursor_col]
            if before == "/argtest tw":
                return {
                    "items": [
                        AutocompleteItem(value="one", label="one"),
                        AutocompleteItem(value="two", label="two"),
                        AutocompleteItem(value="three", label="three"),
                    ],
                    "prefix": "tw",
                }
            return None

        def apply_completion(self, lines, cursor_line, cursor_col, item, prefix):  # noqa: ANN001
            line = lines[cursor_line]
            before = line[: cursor_col - len(prefix)]
            new_line = before + item.value
            new_lines = list(lines)
            new_lines[cursor_line] = new_line
            return {"lines": new_lines, "cursor_line": cursor_line, "cursor_col": len(new_line)}

    editor = Editor(None, None)
    editor.set_autocomplete_provider(_Provider())
    for ch in "/argtest tw":
        editor.handle_input(ch)

    assert editor.get_text() == "/argtest tw"
    assert editor.is_showing_autocomplete() is True


def test_editor_enter_applies_first_prefix_match_when_multiple_unfiltered_items_match() -> None:
    class _Provider:
        def get_suggestions(self, lines: list[str], _cursor_line: int, cursor_col: int):
            before = lines[0][:cursor_col]
            if before == "/argtest t":
                return {
                    "items": [
                        AutocompleteItem(value="one", label="one"),
                        AutocompleteItem(value="two", label="two"),
                        AutocompleteItem(value="three", label="three"),
                    ],
                    "prefix": "t",
                }
            return None

        def apply_completion(self, lines, cursor_line, cursor_col, item, prefix):  # noqa: ANN001
            line = lines[cursor_line]
            before = line[: cursor_col - len(prefix)]
            new_line = before + item.value
            new_lines = list(lines)
            new_lines[cursor_line] = new_line
            return {"lines": new_lines, "cursor_line": cursor_line, "cursor_col": len(new_line)}

    editor = Editor(None, None)
    editor.set_autocomplete_provider(_Provider())
    for ch in "/argtest t":
        editor.handle_input(ch)

    assert editor.is_showing_autocomplete() is True
    editor.handle_input("\r")
    assert editor.get_text() == "/argtest two"


def test_editor_multiple_prefix_matches_stay_visible_while_typing_before_enter() -> None:
    class _Provider:
        def get_suggestions(self, lines: list[str], _cursor_line: int, cursor_col: int):
            before = lines[0][:cursor_col]
            if before == "/argtest t":
                return {
                    "items": [
                        AutocompleteItem(value="one", label="one"),
                        AutocompleteItem(value="two", label="two"),
                        AutocompleteItem(value="three", label="three"),
                    ],
                    "prefix": "t",
                }
            return None

        def apply_completion(self, lines, cursor_line, cursor_col, item, prefix):  # noqa: ANN001
            line = lines[cursor_line]
            before = line[: cursor_col - len(prefix)]
            new_line = before + item.value
            new_lines = list(lines)
            new_lines[cursor_line] = new_line
            return {"lines": new_lines, "cursor_line": cursor_line, "cursor_col": len(new_line)}

    editor = Editor(None, None)
    editor.set_autocomplete_provider(_Provider())
    for ch in "/argtest t":
        editor.handle_input(ch)

    assert editor.get_text() == "/argtest t"
    assert editor.is_showing_autocomplete() is True


def test_editor_selects_first_prefix_match_on_enter_when_typed_arg_is_not_exact_match_like_ts() -> None:
    """selects first prefix match on Enter when typed arg is not exact match"""
    class _Provider:
        def get_suggestions(self, lines: list[str], _cursor_line: int, cursor_col: int):
            before = lines[0][:cursor_col]
            if before == "/argtest t":
                return {
                    "items": [
                        AutocompleteItem(value="two", label="two"),
                        AutocompleteItem(value="three", label="three"),
                        AutocompleteItem(value="twelve", label="twelve"),
                    ],
                    "prefix": "t",
                }
            return None

        def apply_completion(self, lines, cursor_line, cursor_col, item, prefix):  # noqa: ANN001
            line = lines[cursor_line]
            before = line[: cursor_col - len(prefix)]
            new_line = before + item.value
            new_lines = list(lines)
            new_lines[cursor_line] = new_line
            return {"lines": new_lines, "cursor_line": cursor_line, "cursor_col": len(new_line)}

    editor = Editor(None, None)
    editor.set_autocomplete_provider(_Provider())
    for ch in "/argtest t":
        editor.handle_input(ch)

    assert editor.is_showing_autocomplete() is True
    editor.handle_input("\r")
    assert editor.get_text() == "/argtest two"


def test_editor_highlights_unique_prefix_match_as_user_types_before_full_exact_match_like_ts() -> None:
    """highlights unique prefix match as user types (before full exact match)"""
    class _Provider:
        def get_suggestions(self, lines: list[str], _cursor_line: int, cursor_col: int):
            before = lines[0][:cursor_col]
            if before == "/argtest tw":
                return {
                    "items": [
                        AutocompleteItem(value="one", label="one"),
                        AutocompleteItem(value="two", label="two"),
                        AutocompleteItem(value="three", label="three"),
                    ],
                    "prefix": "tw",
                }
            return None

        def apply_completion(self, lines, cursor_line, cursor_col, item, prefix):  # noqa: ANN001
            line = lines[cursor_line]
            before = line[: cursor_col - len(prefix)]
            new_line = before + item.value
            new_lines = list(lines)
            new_lines[cursor_line] = new_line
            return {"lines": new_lines, "cursor_line": cursor_line, "cursor_col": len(new_line)}

    editor = Editor(None, None)
    editor.set_autocomplete_provider(_Provider())
    for ch in "/argtest tw":
        editor.handle_input(ch)

    assert editor.get_text() == "/argtest tw"
    assert editor.is_showing_autocomplete() is True
    editor.handle_input("\r")
    assert editor.get_text() == "/argtest two"


def test_editor_selects_first_prefix_match_when_multiple_items_match_like_ts() -> None:
    """selects first prefix match when multiple items match"""
    class _Provider:
        def get_suggestions(self, lines: list[str], _cursor_line: int, cursor_col: int):
            before = lines[0][:cursor_col]
            if before == "/argtest t":
                return {
                    "items": [
                        AutocompleteItem(value="one", label="one"),
                        AutocompleteItem(value="two", label="two"),
                        AutocompleteItem(value="three", label="three"),
                    ],
                    "prefix": "t",
                }
            return None

        def apply_completion(self, lines, cursor_line, cursor_col, item, prefix):  # noqa: ANN001
            line = lines[cursor_line]
            before = line[: cursor_col - len(prefix)]
            new_line = before + item.value
            new_lines = list(lines)
            new_lines[cursor_line] = new_line
            return {"lines": new_lines, "cursor_line": cursor_line, "cursor_col": len(new_line)}

    editor = Editor(None, None)
    editor.set_autocomplete_provider(_Provider())
    for ch in "/argtest t":
        editor.handle_input(ch)

    assert editor.is_showing_autocomplete() is True
    editor.handle_input("\r")
    assert editor.get_text() == "/argtest two"


def test_editor_does_not_trigger_autocomplete_during_single_line_paste() -> None:
    """does not trigger autocomplete during single-line paste"""
    suggestion_calls = 0

    class _Provider:
        def get_suggestions(self, lines, _cursor_line, cursor_col):  # noqa: ANN001
            nonlocal suggestion_calls
            suggestion_calls += 1
            return None

        def apply_completion(self, lines, cursor_line, cursor_col, item, prefix):  # noqa: ANN001
            return {
                "lines": list(lines),
                "cursor_line": cursor_line,
                "cursor_col": cursor_col,
            }

    editor = Editor(None, None)
    editor.set_autocomplete_provider(_Provider())

    editor.handle_input("\x1b[200~look at @node_modules/react/index.js please\x1b[201~")

    assert editor.get_text() == "look at @node_modules/react/index.js please"
    assert suggestion_calls == 0
    assert editor.is_showing_autocomplete() is False


def test_editor_model_like_argument_completion_handles_hyphenated_values() -> None:
    class _Provider:
        def get_suggestions(self, lines: list[str], _cursor_line: int, cursor_col: int):
            before = lines[0][:cursor_col]
            import re

            match = re.match(r"^/model\s+(\S+)$", before)
            if not match:
                return None
            prefix = match.group(1)
            items = [
                AutocompleteItem(value="gpt-4o", label="gpt-4o"),
                AutocompleteItem(value="gpt-4o-mini", label="gpt-4o-mini"),
                AutocompleteItem(value="claude-sonnet", label="claude-sonnet"),
            ]
            filtered = [item for item in items if item.value.startswith(prefix)]
            return {"items": filtered, "prefix": prefix} if filtered else None

        def apply_completion(self, lines, cursor_line, cursor_col, item, prefix):  # noqa: ANN001
            line = lines[cursor_line]
            before = line[: cursor_col - len(prefix)]
            new_line = before + item.value
            new_lines = list(lines)
            new_lines[cursor_line] = new_line
            return {"lines": new_lines, "cursor_line": cursor_line, "cursor_col": len(new_line)}

    editor = Editor(None, None)
    editor.set_autocomplete_provider(_Provider())
    for ch in "/model gpt-4o-m":
        editor.handle_input(ch)

    assert editor.is_showing_autocomplete() is True
    editor.handle_input("\r")
    assert editor.get_text() == "/model gpt-4o-mini"


def test_editor_model_like_argument_completion_keeps_exact_typed_value_like_ts() -> None:
    class _Provider:
        def get_suggestions(self, lines: list[str], _cursor_line: int, cursor_col: int):
            before = lines[0][:cursor_col]
            import re

            match = re.match(r"^/model\s+(\S+)$", before)
            if not match:
                return None
            prefix = match.group(1)
            items = [
                AutocompleteItem(value="gpt-4o", label="gpt-4o"),
                AutocompleteItem(value="gpt-4o-mini", label="gpt-4o-mini"),
                AutocompleteItem(value="claude-sonnet", label="claude-sonnet"),
            ]
            filtered = [item for item in items if item.value.startswith(prefix)]
            return {"items": filtered, "prefix": prefix} if filtered else None

        def apply_completion(self, lines, cursor_line, cursor_col, item, prefix):  # noqa: ANN001
            line = lines[cursor_line]
            before = line[: cursor_col - len(prefix)]
            new_line = before + item.value
            new_lines = list(lines)
            new_lines[cursor_line] = new_line
            return {"lines": new_lines, "cursor_line": cursor_line, "cursor_col": len(new_line)}

    editor = Editor(None, None)
    editor.set_autocomplete_provider(_Provider())
    for ch in "/model gpt-4o-mini":
        editor.handle_input(ch)

    assert editor.get_text() == "/model gpt-4o-mini"
    assert editor.is_showing_autocomplete() is True
    editor.handle_input("\r")
    assert editor.get_text() == "/model gpt-4o-mini"


def test_editor_works_for_built_in_style_command_argument_completion_path_like_ts() -> None:
    """works for built-in-style command argument completion path (model-like)"""
    class _Provider:
        def get_suggestions(self, lines: list[str], _cursor_line: int, cursor_col: int):
            before = lines[0][:cursor_col]
            import re

            match = re.match(r"^/model\s+(\S+)$", before)
            if not match:
                return None
            prefix = match.group(1)
            items = [
                AutocompleteItem(value="gpt-4o", label="gpt-4o"),
                AutocompleteItem(value="gpt-4o-mini", label="gpt-4o-mini"),
                AutocompleteItem(value="claude-sonnet", label="claude-sonnet"),
            ]
            filtered = [item for item in items if item.value.startswith(prefix)]
            return {"items": filtered, "prefix": prefix} if filtered else None

        def apply_completion(self, lines, cursor_line, cursor_col, item, prefix):  # noqa: ANN001
            line = lines[cursor_line]
            before = line[: cursor_col - len(prefix)]
            new_line = before + item.value
            new_lines = list(lines)
            new_lines[cursor_line] = new_line
            return {"lines": new_lines, "cursor_line": cursor_line, "cursor_col": len(new_line)}

    editor = Editor(None, None)
    editor.set_autocomplete_provider(_Provider())
    for ch in "/model gpt-4o-mini":
        editor.handle_input(ch)

    assert editor.get_text() == "/model gpt-4o-mini"
    assert editor.is_showing_autocomplete() is True
    editor.handle_input("\r")
    assert editor.get_text() == "/model gpt-4o-mini"


def test_editor_character_jump_forward_and_backward() -> None:
    editor = Editor(None, None)
    editor.set_text("hello world")
    editor.handle_input("\x01")

    editor.handle_input("\x1d")
    editor.handle_input("o")
    assert editor.get_cursor() == {"line": 0, "col": 4}

    editor.handle_input("\x1d")
    editor.handle_input("o")
    assert editor.get_cursor() == {"line": 0, "col": 7}

    editor.handle_input("\x1b\x1d")
    editor.handle_input("o")
    assert editor.get_cursor() == {"line": 0, "col": 4}


def test_editor_character_jump_forward_to_first_occurrence_on_same_line() -> None:
    """jumps forward to first occurrence of character on same line"""
    editor = Editor(None, None)
    editor.set_text("hello world")
    editor.handle_input("\x01")

    editor.handle_input("\x1d")
    editor.handle_input("o")

    assert editor.get_cursor() == {"line": 0, "col": 4}


def test_editor_jumps_forward_to_first_occurrence_of_character_on_same_line_like_ts() -> None:
    editor = Editor(None, None)
    editor.set_text("hello world")
    editor.handle_input("\x01")

    editor.handle_input("\x1d")
    editor.handle_input("o")

    assert editor.get_cursor() == {"line": 0, "col": 4}


def test_editor_character_jump_forward_to_next_occurrence_after_cursor() -> None:
    editor = Editor(None, None)
    editor.set_text("hello world")
    editor.handle_input("\x01")
    for _ in range(4):
        editor.handle_input("\x1b[C")

    assert editor.get_cursor() == {"line": 0, "col": 4}

    editor.handle_input("\x1d")
    editor.handle_input("o")

    assert editor.get_cursor() == {"line": 0, "col": 7}


def test_editor_jumps_forward_to_next_occurrence_after_cursor_like_ts() -> None:
    """jumps forward to next occurrence after cursor"""
    editor = Editor(None, None)
    editor.set_text("hello world")
    editor.handle_input("\x01")
    for _ in range(4):
        editor.handle_input("\x1b[C")

    assert editor.get_cursor() == {"line": 0, "col": 4}

    editor.handle_input("\x1d")
    editor.handle_input("o")

    assert editor.get_cursor() == {"line": 0, "col": 7}


def test_editor_character_jump_forward_across_multiple_lines() -> None:
    editor = Editor(None, None)
    editor.set_text("abc\ndef\nghi")
    editor.handle_input("\x1b[A")
    editor.handle_input("\x1b[A")
    editor.handle_input("\x01")

    editor.handle_input("\x1d")
    editor.handle_input("g")

    assert editor.get_cursor() == {"line": 2, "col": 0}


def test_editor_jumps_forward_across_multiple_lines_like_ts() -> None:
    """jumps forward across multiple lines"""
    editor = Editor(None, None)
    editor.set_text("abc\ndef\nghi")
    editor.handle_input("\x1b[A")
    editor.handle_input("\x1b[A")
    editor.handle_input("\x01")

    editor.handle_input("\x1d")
    editor.handle_input("g")

    assert editor.get_cursor() == {"line": 2, "col": 0}


def test_editor_character_jump_backward_across_multiple_lines() -> None:
    editor = Editor(None, None)
    editor.set_text("abc\ndef\nghi")

    editor.handle_input("\x1b\x1d")
    editor.handle_input("a")

    assert editor.get_cursor() == {"line": 0, "col": 0}


def test_editor_character_jump_backward_to_first_occurrence_before_cursor_on_same_line() -> None:
    editor = Editor(None, None)
    editor.set_text("hello world")

    editor.handle_input("\x1b\x1d")
    editor.handle_input("o")

    assert editor.get_cursor() == {"line": 0, "col": 7}


def test_editor_jumps_backward_to_first_occurrence_before_cursor_on_same_line_like_ts() -> None:
    """jumps backward to first occurrence before cursor on same line"""
    editor = Editor(None, None)
    editor.set_text("hello world")

    editor.handle_input("\x1b\x1d")
    editor.handle_input("o")

    assert editor.get_cursor() == {"line": 0, "col": 7}


def test_editor_jumps_backward_across_multiple_lines_like_ts() -> None:
    """jumps backward across multiple lines"""
    editor = Editor(None, None)
    editor.set_text("abc\ndef\nghi")

    editor.handle_input("\x1b\x1d")
    editor.handle_input("a")

    assert editor.get_cursor() == {"line": 0, "col": 0}


def test_editor_forward_jump_does_nothing_when_character_is_not_found() -> None:
    """does nothing when character is not found (forward)"""
    editor = Editor(None, None)
    editor.set_text("hello world")
    editor.handle_input("\x01")

    editor.handle_input("\x1d")
    editor.handle_input("z")

    assert editor.get_cursor() == {"line": 0, "col": 0}


def test_editor_backward_jump_does_nothing_when_character_is_not_found() -> None:
    """does nothing when character is not found (backward)"""
    editor = Editor(None, None)
    editor.set_text("hello world")

    editor.handle_input("\x1b\x1d")
    editor.handle_input("z")

    assert editor.get_cursor() == {"line": 0, "col": 11}


def test_editor_character_jump_is_case_sensitive() -> None:
    """is case-sensitive"""
    editor = Editor(None, None)
    editor.set_text("Hello World")
    editor.handle_input("\x01")

    editor.handle_input("\x1d")
    editor.handle_input("h")
    assert editor.get_cursor() == {"line": 0, "col": 0}

    editor.handle_input("\x1d")
    editor.handle_input("W")
    assert editor.get_cursor() == {"line": 0, "col": 6}


def test_editor_ctrl_right_bracket_again_cancels_jump_mode() -> None:
    """cancels jump mode when Ctrl+] is pressed again"""
    editor = Editor(None, None)
    editor.set_text("hello world")
    editor.handle_input("\x01")

    editor.handle_input("\x1d")
    editor.handle_input("\x1d")
    editor.handle_input("o")

    assert editor.get_text() == "ohello world"


def test_editor_escape_cancels_jump_mode_and_processes_escape() -> None:
    """cancels jump mode on Escape and processes the Escape"""
    editor = Editor(None, None)
    editor.set_text("hello world")
    editor.handle_input("\x01")

    editor.handle_input("\x1d")
    editor.handle_input("\x1b")
    assert editor.get_cursor() == {"line": 0, "col": 0}

    editor.handle_input("o")
    assert editor.get_text() == "ohello world"


def test_editor_ctrl_alt_right_bracket_again_cancels_backward_jump_mode() -> None:
    """cancels backward jump mode when Ctrl+Alt+] is pressed again"""
    editor = Editor(None, None)
    editor.set_text("hello world")

    editor.handle_input("\x1b\x1d")
    editor.handle_input("\x1b\x1d")
    editor.handle_input("o")

    assert editor.get_text() == "hello worldo"


def test_editor_character_jump_searches_for_special_characters() -> None:
    """searches for special characters"""
    editor = Editor(None, None)
    editor.set_text("foo(bar) = baz;")
    editor.handle_input("\x01")

    editor.handle_input("\x1d")
    editor.handle_input("(")
    assert editor.get_cursor() == {"line": 0, "col": 3}

    editor.handle_input("\x1d")
    editor.handle_input("=")
    assert editor.get_cursor() == {"line": 0, "col": 9}


def test_editor_character_jump_handles_empty_text_gracefully() -> None:
    """handles empty text gracefully"""
    editor = Editor(None, None)
    editor.set_text("")

    assert editor.get_cursor() == {"line": 0, "col": 0}

    editor.handle_input("\x1d")
    editor.handle_input("x")

    assert editor.get_cursor() == {"line": 0, "col": 0}


def test_editor_jumping_resets_last_action_for_undo_grouping() -> None:
    editor = Editor(None, None)
    editor.set_text("hello world")
    editor.handle_input("\x01")

    editor.handle_input("x")
    assert editor.get_text() == "xhello world"

    editor.handle_input("\x1d")
    editor.handle_input("o")
    editor.handle_input("Y")
    assert editor.get_text() == "xhellYo world"

    editor.handle_input("\x1b[45;5u")
    assert editor.get_text() == "xhello world"


def test_editor_resets_last_action_when_jumping_like_ts() -> None:
    """resets lastAction when jumping"""
    editor = Editor(None, None)
    editor.set_text("hello world")
    editor.handle_input("\x01")

    editor.handle_input("x")
    assert editor.get_text() == "xhello world"

    editor.handle_input("\x1d")
    editor.handle_input("o")
    editor.handle_input("Y")
    assert editor.get_text() == "xhellYo world"

    editor.handle_input("\x1b[45;5u")
    assert editor.get_text() == "xhello world"


def test_editor_ctrl_a_resets_sticky_column_to_line_start() -> None:
    """resets sticky column on Ctrl+A (move to line start)"""
    editor = Editor(None, None)
    editor.set_text("1234567890\n\n1234567890")

    editor.handle_input("\x01")
    for _ in range(8):
        editor.handle_input("\x1b[C")

    editor.handle_input("\x1b[A")
    assert editor.get_cursor() == {"line": 1, "col": 0}

    editor.handle_input("\x01")
    assert editor.get_cursor() == {"line": 1, "col": 0}

    editor.handle_input("\x1b[A")
    assert editor.get_cursor() == {"line": 0, "col": 0}


def test_editor_ctrl_e_resets_sticky_column_to_line_end() -> None:
    """resets sticky column on Ctrl+E (move to line end)"""
    editor = Editor(None, None)
    editor.set_text("12345\n\n1234567890")

    editor.handle_input("\x01")
    for _ in range(3):
        editor.handle_input("\x1b[C")

    editor.handle_input("\x1b[A")
    editor.handle_input("\x1b[A")
    assert editor.get_cursor() == {"line": 0, "col": 3}

    editor.handle_input("\x05")
    assert editor.get_cursor() == {"line": 0, "col": 5}

    editor.handle_input("\x1b[B")
    editor.handle_input("\x1b[B")
    assert editor.get_cursor() == {"line": 2, "col": 5}


def test_editor_ctrl_left_resets_sticky_column() -> None:
    """resets sticky column on word movement (Ctrl+Left)"""
    editor = Editor(None, None)
    editor.set_text("hello world\n\nhello world")

    editor.handle_input("\x1b[A")
    editor.handle_input("\x1b[A")
    assert editor.get_cursor() == {"line": 0, "col": 11}

    editor.handle_input("\x1b[1;5D")
    assert editor.get_cursor() == {"line": 0, "col": 6}

    editor.handle_input("\x1b[B")
    editor.handle_input("\x1b[B")
    assert editor.get_cursor() == {"line": 2, "col": 6}


def test_editor_ctrl_right_resets_sticky_column() -> None:
    """resets sticky column on word movement (Ctrl+Right)"""
    editor = Editor(None, None)
    editor.set_text("hello world\n\nhello world")

    editor.handle_input("\x1b[A")
    editor.handle_input("\x1b[A")
    editor.handle_input("\x01")
    assert editor.get_cursor() == {"line": 0, "col": 0}

    editor.handle_input("\x1b[B")
    editor.handle_input("\x1b[B")
    assert editor.get_cursor() == {"line": 2, "col": 0}

    editor.handle_input("\x1b[1;5C")
    assert editor.get_cursor() == {"line": 2, "col": 5}

    editor.handle_input("\x1b[A")
    editor.handle_input("\x1b[A")
    assert editor.get_cursor() == {"line": 0, "col": 5}


def test_editor_typing_resets_sticky_column() -> None:
    """resets sticky column on typing"""
    editor = Editor(None, None)
    editor.set_text("1234567890\n\n1234567890")

    editor.handle_input("\x01")
    for _ in range(8):
        editor.handle_input("\x1b[C")

    editor.handle_input("\x1b[A")
    editor.handle_input("\x1b[A")
    assert editor.get_cursor() == {"line": 0, "col": 8}

    editor.handle_input("X")
    assert editor.get_cursor() == {"line": 0, "col": 9}

    editor.handle_input("\x1b[B")
    editor.handle_input("\x1b[B")
    assert editor.get_cursor() == {"line": 2, "col": 9}


def test_editor_backspace_resets_sticky_column() -> None:
    """resets sticky column on backspace"""
    editor = Editor(None, None)
    editor.set_text("1234567890\n\n1234567890")

    editor.handle_input("\x01")
    for _ in range(8):
        editor.handle_input("\x1b[C")

    editor.handle_input("\x1b[A")
    editor.handle_input("\x1b[A")
    assert editor.get_cursor() == {"line": 0, "col": 8}

    editor.handle_input("\x7f")
    assert editor.get_cursor() == {"line": 0, "col": 7}

    editor.handle_input("\x1b[B")
    editor.handle_input("\x1b[B")
    assert editor.get_cursor() == {"line": 2, "col": 7}


def test_editor_set_text_resets_sticky_column() -> None:
    """handles setText resetting sticky column"""
    editor = Editor(None, None)
    editor.set_text("1234567890\n\n1234567890")

    editor.handle_input("\x01")
    for _ in range(8):
        editor.handle_input("\x1b[C")
    editor.handle_input("\x1b[A")

    editor.set_text("abcdefghij\n\nabcdefghij")
    assert editor.get_cursor() == {"line": 2, "col": 10}

    editor.handle_input("\x1b[A")
    assert editor.get_cursor() == {"line": 1, "col": 0}

    editor.handle_input("\x1b[A")
    assert editor.get_cursor() == {"line": 0, "col": 10}


def test_editor_handles_multiple_consecutive_up_down_movements() -> None:
    """handles multiple consecutive up/down movements"""
    editor = Editor(None, None)
    editor.set_text("1234567890\nab\ncd\nef\n1234567890")

    editor.handle_input("\x01")
    for _ in range(7):
        editor.handle_input("\x1b[C")
    assert editor.get_cursor() == {"line": 4, "col": 7}

    editor.handle_input("\x1b[A")
    editor.handle_input("\x1b[A")
    editor.handle_input("\x1b[A")
    editor.handle_input("\x1b[A")
    assert editor.get_cursor() == {"line": 0, "col": 7}

    editor.handle_input("\x1b[B")
    editor.handle_input("\x1b[B")
    editor.handle_input("\x1b[B")
    editor.handle_input("\x1b[B")
    assert editor.get_cursor() == {"line": 4, "col": 7}


def test_editor_render_wraps_long_lines_to_width() -> None:
    """
    shows cursor at end of line before wrap, wraps on next char
    handles empty string
    handles single word that fits exactly
    force-breaks when wide char after word boundary wrap still overflows
    """
    editor = Editor(None, None)
    editor.set_text("0123456789ABCDEF")

    lines = editor.render(8)

    assert len(lines) == 3
    assert all(visible_width(line) == 7 for line in lines)


def test_editor_render_handles_large_paste_marker_narrow_width() -> None:
    """does not crash when paste marker is wider than terminal width"""
    editor = Editor(None, None)
    pasted_text = "\n".join(f"line {i}" for i in range(20))
    editor.handle_input(f"\x1b[200~{pasted_text}\x1b[201~")

    lines = editor.render(8)

    assert all(visible_width(line) <= 8 for line in lines)


def test_editor_render_handles_text_plus_paste_marker_with_cursor_on_marker() -> None:
    """does not crash when text + paste marker exceeds terminal width with cursor on marker"""
    editor = Editor(None, None)

    for _ in range(35):
        editor.handle_input("b")
    pasted_text = "\n".join("line" for _ in range(27))
    editor.handle_input(f"\x1b[200~{pasted_text}\x1b[201~")
    for _ in range(4):
        editor.handle_input("b")

    for _ in range(5):
        editor.handle_input("\x1b[D")

    lines = editor.render(54)

    assert all(visible_width(line) <= 54 for line in lines)


def test_editor_moves_through_wrapped_visual_lines() -> None:
    """moves correctly through wrapped visual lines without getting stuck"""
    terminal = _TerminalStub(columns=15, rows=24)
    tui = TUI(terminal)
    editor = Editor(tui, None)

    editor.set_text("short\n123456789012345678901234567890")
    editor.render(10)

    assert editor.get_cursor() == {"line": 1, "col": 30}

    editor.handle_input("\x1b[A")
    assert editor.get_cursor()["line"] == 1

    editor.handle_input("\x1b[A")
    assert editor.get_cursor()["line"] == 1

    editor.handle_input("\x1b[A")
    assert editor.get_cursor()["line"] == 1

    editor.handle_input("\x1b[A")
    assert editor.get_cursor()["line"] == 0


def test_editor_resize_clamps_and_restores_preferred_visual_column() -> None:
    """
    preserves target column when moving up through a shorter line
    preserves target column when moving down through a shorter line
    """
    terminal = _TerminalStub(columns=80, rows=24)
    tui = TUI(terminal)
    editor = Editor(tui, None)

    editor.set_text("short\n12345678901234567890")
    editor.handle_input("\x01")
    for _ in range(15):
        editor.handle_input("\x1b[C")
    assert editor.get_cursor() == {"line": 1, "col": 15}

    editor.handle_input("\x1b[A")
    assert editor.get_cursor() == {"line": 0, "col": 5}

    editor.render(10)
    editor.handle_input("\x1b[B")
    assert editor.get_cursor() == {"line": 1, "col": 8}

    editor.handle_input("\x1b[A")
    assert editor.get_cursor() == {"line": 0, "col": 5}

    editor.render(80)
    editor.handle_input("\x1b[B")
    assert editor.get_cursor() == {"line": 1, "col": 15}


def test_editor_resize_clamps_preferred_visual_column_on_same_line() -> None:
    terminal = _TerminalStub(columns=80, rows=24)
    tui = TUI(terminal)
    editor = Editor(tui, None)

    editor.set_text("12345678901234567890\n\n12345678901234567890")
    editor.handle_input("\x01")
    for _ in range(15):
        editor.handle_input("\x1b[C")

    editor.handle_input("\x1b[A")
    editor.handle_input("\x1b[A")
    assert editor.get_cursor() == {"line": 0, "col": 15}

    editor.render(12)
    editor.handle_input("\x1b[B")
    editor.handle_input("\x1b[B")

    assert editor.get_cursor() == {"line": 2, "col": 4}


def test_editor_handles_resizes_when_preferred_visual_col_is_on_same_line_like_ts() -> None:
    """handles editor resizes when preferredVisualCol is on the same line"""
    terminal = _TerminalStub(columns=80, rows=24)
    tui = TUI(terminal)
    editor = Editor(tui, None)

    editor.set_text("12345678901234567890\n\n12345678901234567890")
    editor.handle_input("\x01")
    for _ in range(15):
        editor.handle_input("\x1b[C")

    editor.handle_input("\x1b[A")
    editor.handle_input("\x1b[A")
    assert editor.get_cursor() == {"line": 0, "col": 15}

    editor.render(12)
    editor.handle_input("\x1b[B")
    editor.handle_input("\x1b[B")

    assert editor.get_cursor() == {"line": 2, "col": 4}


def test_editor_resize_preserves_preferred_visual_column_across_different_lines() -> None:
    terminal = _TerminalStub(columns=80, rows=24)
    tui = TUI(terminal)
    editor = Editor(tui, None)

    editor.set_text("short\n12345678901234567890")
    editor.handle_input("\x01")
    for _ in range(15):
        editor.handle_input("\x1b[C")
    assert editor.get_cursor() == {"line": 1, "col": 15}

    editor.handle_input("\x1b[A")
    assert editor.get_cursor() == {"line": 0, "col": 5}

    editor.render(10)
    editor.handle_input("\x1b[B")
    assert editor.get_cursor() == {"line": 1, "col": 8}

    editor.handle_input("\x1b[A")
    assert editor.get_cursor() == {"line": 0, "col": 5}

    editor.render(80)
    editor.handle_input("\x1b[B")
    assert editor.get_cursor() == {"line": 1, "col": 15}


def test_editor_handles_resizes_when_preferred_visual_col_is_on_different_line_like_ts() -> None:
    """handles editor resizes when preferredVisualCol is on a different line"""
    terminal = _TerminalStub(columns=80, rows=24)
    tui = TUI(terminal)
    editor = Editor(tui, None)

    editor.set_text("short\n12345678901234567890")
    editor.handle_input("\x01")
    for _ in range(15):
        editor.handle_input("\x1b[C")
    assert editor.get_cursor() == {"line": 1, "col": 15}

    editor.handle_input("\x1b[A")
    assert editor.get_cursor() == {"line": 0, "col": 5}

    editor.render(10)
    editor.handle_input("\x1b[B")
    assert editor.get_cursor() == {"line": 1, "col": 8}

    editor.handle_input("\x1b[A")
    assert editor.get_cursor() == {"line": 0, "col": 5}

    editor.render(80)
    editor.handle_input("\x1b[B")
    assert editor.get_cursor() == {"line": 1, "col": 15}


def test_editor_right_at_end_preserves_preferred_visual_column() -> None:
    terminal = _TerminalStub(columns=80, rows=24)
    tui = TUI(terminal)
    editor = Editor(tui, None)

    editor.set_text("111111111x1111111111\n\n333333333_")
    editor.handle_input("\x1b[A")
    editor.handle_input("\x1b[A")
    editor.handle_input("\x05")
    assert editor.get_cursor() == {"line": 0, "col": 20}

    editor.handle_input("\x1b[B")
    editor.handle_input("\x1b[B")
    assert editor.get_cursor() == {"line": 2, "col": 10}

    editor.handle_input("\x1b[C")
    assert editor.get_cursor() == {"line": 2, "col": 10}

    editor.handle_input("\x1b[A")
    editor.handle_input("\x1b[A")
    assert editor.get_cursor() == {"line": 0, "col": 10}


def test_editor_sets_preferred_visual_col_when_pressing_right_at_end_of_prompt_like_ts() -> None:
    terminal = _TerminalStub(columns=80, rows=24)
    tui = TUI(terminal)
    editor = Editor(tui, None)

    editor.set_text("111111111x1111111111\n\n333333333_")
    editor.handle_input("\x1b[A")
    editor.handle_input("\x1b[A")
    editor.handle_input("\x05")
    assert editor.get_cursor() == {"line": 0, "col": 20}

    editor.handle_input("\x1b[B")
    editor.handle_input("\x1b[B")
    assert editor.get_cursor() == {"line": 2, "col": 10}

    editor.handle_input("\x1b[C")
    assert editor.get_cursor() == {"line": 2, "col": 10}

    editor.handle_input("\x1b[A")
    editor.handle_input("\x1b[A")
    assert editor.get_cursor() == {"line": 0, "col": 10}


def test_input_yank_pop_replaces_last_yanked_segment_in_place() -> None:
    input_widget = Input()
    input_widget.kill_ring.push("FIRST", prepend=False)
    input_widget.kill_ring.push("SECOND", prepend=False)
    input_widget.set_value("a b")
    input_widget.handle_input("\x01")
    input_widget.handle_input("\x1b[C")
    input_widget.handle_input("\x1b[C")

    input_widget.handle_input("\x19")
    assert input_widget.get_value() == "a SECONDb"

    input_widget.handle_input("\x1by")
    assert input_widget.get_value() == "a FIRSTb"


def test_input_alt_y_handles_yank_pop_in_middle_of_text_like_ts() -> None:
    input_widget = Input()
    input_widget.kill_ring.push("FIRST", prepend=False)
    input_widget.kill_ring.push("SECOND", prepend=False)
    input_widget.set_value("a b")
    input_widget.handle_input("\x01")
    input_widget.handle_input("\x1b[C")
    input_widget.handle_input("\x1b[C")

    input_widget.handle_input("\x19")
    assert input_widget.get_value() == "a SECONDb"

    input_widget.handle_input("\x1by")
    assert input_widget.get_value() == "a FIRSTb"


def test_editor_yank_and_yank_pop_work_in_middle_of_text() -> None:
    editor = Editor(None, None)
    editor.set_text("word")
    editor.handle_input("\x17")
    editor.set_text("hello world")
    editor.handle_input("\x01")
    for _ in range(6):
        editor.handle_input("\x1b[C")

    editor.handle_input("\x19")
    assert editor.get_text() == "hello wordworld"

    editor = Editor(None, None)
    editor.set_text("FIRST")
    editor.handle_input("\x17")
    editor.set_text("SECOND")
    editor.handle_input("\x17")
    editor.set_text("hello world")
    editor.handle_input("\x01")
    for _ in range(6):
        editor.handle_input("\x1b[C")

    editor.handle_input("\x19")
    assert editor.get_text() == "hello SECONDworld"
    editor.handle_input("\x1by")
    assert editor.get_text() == "hello FIRSTworld"


def test_editor_multiline_yank_pop_works_in_middle_of_text() -> None:
    editor = Editor(None, None)
    editor.set_text("SINGLE")
    editor.handle_input("\x17")

    editor.set_text("A\nB")
    editor.handle_input("\x15")
    editor.handle_input("\x15")
    editor.handle_input("\x15")

    editor.set_text("hello world")
    editor.handle_input("\x01")
    for _ in range(6):
        editor.handle_input("\x1b[C")

    editor.handle_input("\x19")
    assert editor.get_text() == "hello A\nBworld"
    editor.handle_input("\x1by")
    assert editor.get_text() == "hello SINGLEworld"


def test_non_capturing_overlay_focus_round_trip() -> None:
    terminal = _TerminalStub(columns=40, rows=10)
    tui = TUI(terminal)
    editor = _FocusableComponent(["EDITOR"])
    overlay = _FocusableComponent(["OVERLAY"])

    tui.add_child(Text("base"))
    tui.set_focus(editor)
    tui.start()

    handle = tui.show_overlay(overlay, OverlayOptions(non_capturing=True))
    assert editor.focused is True
    assert overlay.focused is False

    handle.focus()
    assert editor.focused is False
    assert overlay.focused is True
    assert handle.is_focused() is True

    handle.unfocus()
    assert editor.focused is True
    assert overlay.focused is False
    assert handle.is_focused() is False


def test_overlay_visible_rule_skips_render_and_focus_until_terminal_is_large_enough() -> None:
    terminal = _TerminalStub(columns=20, rows=6)
    tui = TUI(terminal)
    editor = _FocusableComponent(["EDITOR"])
    overlay = _FocusableComponent(["OVERLAY"])

    tui.add_child(Text("base"))
    tui.set_focus(editor)
    tui.show_overlay(
        overlay,
        OverlayOptions(width=10, visible=lambda columns, _rows: columns >= 30),
    )
    tui.start()

    assert tui.has_overlay() is False
    assert editor.focused is True
    assert overlay.focused is False
    assert not any("OVERLAY" in line for line in tui.previous_lines)

    terminal.columns = 40
    tui.request_render()

    assert tui.has_overlay() is True
    assert any("OVERLAY" in line for line in tui.previous_lines)


def test_clear_on_shrink_removes_stale_rows() -> None:
    terminal = _TerminalStub(columns=40, rows=10)
    tui = TUI(terminal)
    body = _StaticComponent(["Line 0", "Line 1", "Line 2", "Line 3"])

    tui.set_clear_on_shrink(True)
    tui.add_child(body)
    tui.start()
    assert any("Line 3" in line for line in tui.previous_lines)
    assert tui.max_lines_rendered == 4

    tui.clear()
    tui.add_child(_StaticComponent(["Only line"]))
    tui.request_render()

    assert "Only line" in tui.previous_lines[0]
    assert len(tui.previous_lines) == 1
    assert tui.max_lines_rendered == 1


def test_tui_clears_empty_rows_when_content_shrinks_significantly_like_ts() -> None:
    """clears empty rows when content shrinks significantly"""
    terminal = _TerminalStub(columns=40, rows=10)
    tui = TUI(terminal)
    body = _StaticComponent(["Line 0", "Line 1", "Line 2", "Line 3"])

    tui.set_clear_on_shrink(True)
    tui.add_child(body)
    tui.start()
    tui.clear()
    tui.add_child(_StaticComponent(["Only line"]))
    tui.request_render()

    assert "Only line" in tui.previous_lines[0]
    assert len(tui.previous_lines) == 1


def test_tui_clear_on_shrink_handles_shrink_to_single_line() -> None:
    """handles shrink to single line"""
    terminal = _TerminalStub(columns=40, rows=10)
    tui = TUI(terminal)
    body = _StaticComponent(["Line 0", "Line 1", "Line 2", "Line 3"])

    tui.set_clear_on_shrink(True)
    tui.add_child(body)
    tui.start()

    body.lines = ["Only line"]
    tui.request_render()

    assert "Only line" in tui.previous_lines[0]
    assert len(tui.previous_lines) == 1


def test_tui_clear_on_shrink_single_line_clears_following_terminal_rows_like_ts() -> None:
    terminal = _TerminalStub(columns=40, rows=10)
    tui = TUI(terminal)
    body = _StaticComponent(["Line 0", "Line 1", "Line 2", "Line 3"])

    tui.set_clear_on_shrink(True)
    tui.add_child(body)
    tui.start()

    terminal.writes.clear()
    body.lines = ["Only line"]
    tui.request_render()

    assert tui.previous_lines == ["Only line\x1b[0m\x1b]8;;\x07"]
    assert any("\x1b[2J\x1b[H\x1b[3J" in write for write in terminal.writes)
    assert all("Line 1" not in write for write in terminal.writes)
    assert all("Line 2" not in write for write in terminal.writes)
    assert all("Line 3" not in write for write in terminal.writes)


def test_tui_resize_height_triggers_full_redraw() -> None:
    terminal = _TerminalStub(columns=40, rows=10)
    tui = TUI(terminal)
    body = _StaticComponent(["Line 0", "Line 1", "Line 2"])
    tui.add_child(body)
    tui.start()

    initial_redraws = tui.full_redraws
    terminal.rows = 15
    tui.request_render()

    assert tui.full_redraws > initial_redraws
    assert "Line 0" in tui.previous_lines[0]


def test_tui_triggers_full_rerender_when_terminal_height_changes_like_ts() -> None:
    """triggers full re-render when terminal height changes"""
    terminal = _TerminalStub(columns=40, rows=10)
    tui = TUI(terminal)
    body = _StaticComponent(["Line 0", "Line 1", "Line 2"])
    tui.add_child(body)
    tui.start()

    initial_redraws = tui.full_redraws
    terminal.rows = 15
    tui.request_render()

    assert tui.full_redraws > initial_redraws
    assert "Line 0" in tui.previous_lines[0]


def test_tui_triggers_full_rerender_when_terminal_height_changes_exact_title_like_ts() -> None:
    terminal = _TerminalStub(columns=40, rows=10)
    tui = TUI(terminal)
    body = _StaticComponent(["Line 0", "Line 1", "Line 2"])
    tui.add_child(body)
    tui.start()

    initial_redraws = tui.full_redraws
    terminal.rows = 15
    tui.request_render()

    assert tui.full_redraws > initial_redraws
    assert "Line 0" in tui.previous_lines[0]


def test_tui_triggers_full_re_render_when_terminal_height_changes_like_ts() -> None:
    terminal = _TerminalStub(columns=40, rows=10)
    tui = TUI(terminal)
    body = _StaticComponent(["Line 0", "Line 1", "Line 2"])
    tui.add_child(body)
    tui.start()

    initial_redraws = tui.full_redraws
    terminal.rows = 15
    tui.request_render()

    assert tui.full_redraws > initial_redraws
    assert "Line 0" in tui.previous_lines[0]


def test_tui_resize_width_triggers_full_redraw() -> None:
    terminal = _TerminalStub(columns=40, rows=10)
    tui = TUI(terminal)
    body = _StaticComponent(["Line 0", "Line 1", "Line 2"])
    tui.add_child(body)
    tui.start()

    initial_redraws = tui.full_redraws
    terminal.columns = 60
    tui.request_render()

    assert tui.full_redraws > initial_redraws


def test_tui_triggers_full_rerender_when_terminal_width_changes_like_ts() -> None:
    """triggers full re-render when terminal width changes"""
    terminal = _TerminalStub(columns=40, rows=10)
    tui = TUI(terminal)
    body = _StaticComponent(["Line 0", "Line 1", "Line 2"])
    tui.add_child(body)
    tui.start()

    initial_redraws = tui.full_redraws
    terminal.columns = 60
    tui.request_render()

    assert tui.full_redraws > initial_redraws


def test_tui_triggers_full_rerender_when_terminal_width_changes_exact_title_like_ts() -> None:
    terminal = _TerminalStub(columns=40, rows=10)
    tui = TUI(terminal)
    body = _StaticComponent(["Line 0", "Line 1", "Line 2"])
    tui.add_child(body)
    tui.start()

    initial_redraws = tui.full_redraws
    terminal.columns = 60
    tui.request_render()

    assert tui.full_redraws > initial_redraws


def test_tui_triggers_full_re_render_when_terminal_width_changes_like_ts() -> None:
    terminal = _TerminalStub(columns=40, rows=10)
    tui = TUI(terminal)
    body = _StaticComponent(["Line 0", "Line 1", "Line 2"])
    tui.add_child(body)
    tui.start()

    initial_redraws = tui.full_redraws
    terminal.columns = 60
    tui.request_render()

    assert tui.full_redraws > initial_redraws


def _test_select_theme() -> dict[str, object]:
    return {
        "selected_prefix": lambda text: text,
        "selected_text": lambda text: text,
        "description": lambda text: text,
        "scroll_info": lambda text: text,
        "no_match": lambda text: text,
    }


def _test_markdown_theme() -> dict[str, object]:
    return {
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


def _ansi(code: str, text: str) -> str:
    return f"\x1b[{code}m{text}\x1b[0m"


def _strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def _styled_markdown_theme() -> dict[str, object]:
    return {
        "heading": lambda text: _ansi("36;1", text),
        "link": lambda text: _ansi("34", text),
        "link_url": lambda text: _ansi("2", text),
        "code": lambda text: _ansi("33", text),
        "code_block": lambda text: _ansi("32", text),
        "code_block_border": lambda text: _ansi("2", text),
        "quote": lambda text: _ansi("3", text),
        "quote_border": lambda text: _ansi("2", text),
        "hr": lambda text: _ansi("2", text),
        "list_bullet": lambda text: _ansi("36", text),
        "bold": lambda text: _ansi("1", text),
        "italic": lambda text: _ansi("3", text),
        "strikethrough": lambda text: _ansi("9", text),
        "underline": lambda text: _ansi("4", text),
    }


def _test_settings_theme() -> dict[str, object]:
    return {
        "label": lambda text, selected: text,
        "value": lambda text, selected: text,
        "description": lambda text: text,
        "cursor": "→ ",
        "hint": lambda text: text,
    }


def _visible_index_of(line: str, text: str) -> int:
    index = line.index(text)
    assert index >= 0
    return visible_width(line[:index])


def test_select_list_normalizes_multiline_descriptions() -> None:
    """normalizes multiline descriptions to single line"""
    items = [
        SelectItem(
            value="test",
            label="test",
            description="Line one\nLine two\nLine three",
        )
    ]

    rendered = SelectList(items, 5, _test_select_theme()).render(100)

    assert rendered
    assert "\n" not in rendered[0]
    assert "Line one Line two Line three" in rendered[0]


def test_select_list_multiline_description_renders_as_single_line_text() -> None:
    items = [
        SelectItem(
            value="test",
            label="test",
            description="Line one\nLine two\nLine three",
        )
    ]

    [line] = SelectList(items, 5, _test_select_theme()).render(100)

    assert "\n" not in line
    assert "Line one Line two Line three" in line


def test_select_list_aligns_descriptions_when_primary_is_truncated() -> None:
    """keeps descriptions aligned when the primary text is truncated"""
    items = [
        SelectItem(value="short", label="short", description="short description"),
        SelectItem(
            value="very-long-command-name-that-needs-truncation",
            label="very-long-command-name-that-needs-truncation",
            description="long description",
        ),
    ]

    rendered = SelectList(items, 5, _test_select_theme()).render(80)

    assert _visible_index_of(rendered[0], "short description") == _visible_index_of(rendered[1], "long description")


def test_select_list_respects_primary_column_min_and_max_width() -> None:
    items = [
        SelectItem(
            value="very-long-command-name-that-needs-truncation",
            label="very-long-command-name-that-needs-truncation",
            description="first",
        ),
        SelectItem(value="short", label="short", description="second"),
    ]

    rendered = SelectList(
        items,
        5,
        _test_select_theme(),
        {"min_primary_column_width": 12, "max_primary_column_width": 20},
    ).render(80)

    assert _visible_index_of(rendered[0], "first") == 22
    assert _visible_index_of(rendered[1], "second") == 22


def test_select_list_respects_min_primary_column_width_independently() -> None:
    """uses the configured minimum primary column width"""
    items = [
        SelectItem(value="a", label="a", description="first"),
        SelectItem(value="bb", label="bb", description="second"),
    ]

    rendered = SelectList(
        items,
        5,
        _test_select_theme(),
        {"min_primary_column_width": 12, "max_primary_column_width": 20},
    ).render(80)

    assert rendered[0].index("first") == 14
    assert rendered[1].index("second") == 14


def test_select_list_respects_max_primary_column_width_independently() -> None:
    """uses the configured maximum primary column width"""
    items = [
        SelectItem(
            value="very-long-command-name-that-needs-truncation",
            label="very-long-command-name-that-needs-truncation",
            description="first",
        ),
        SelectItem(value="short", label="short", description="second"),
    ]

    rendered = SelectList(
        items,
        5,
        _test_select_theme(),
        {"min_primary_column_width": 12, "max_primary_column_width": 20},
    ).render(80)

    assert _visible_index_of(rendered[0], "first") == 22
    assert _visible_index_of(rendered[1], "second") == 22


def test_select_list_custom_primary_truncation_preserves_description_alignment() -> None:
    """allows overriding primary truncation while preserving description alignment"""
    items = [
        SelectItem(
            value="very-long-command-name-that-needs-truncation",
            label="very-long-command-name-that-needs-truncation",
            description="first",
        ),
        SelectItem(value="short", label="short", description="second"),
    ]

    rendered = SelectList(
        items,
        5,
        _test_select_theme(),
        {
            "min_primary_column_width": 12,
            "max_primary_column_width": 12,
            "truncate_primary": lambda ctx: ctx.text if len(ctx.text) <= ctx.max_width else f"{ctx.text[: max(0, ctx.max_width - 1)]}…",
        },
    ).render(80)

    assert "…" in rendered[0]
    assert _visible_index_of(rendered[0], "first") == _visible_index_of(rendered[1], "second")


def test_markdown_renders_nested_list_structure() -> None:
    markdown = Markdown(
        "- Item 1\n  - Nested 1.1\n  - Nested 1.2\n- Item 2",
        0,
        0,
        _test_markdown_theme(),
    )
    plain = markdown.render(80)
    assert any("- Item 1" in line for line in plain)
    assert any("  - Nested 1.1" in line for line in plain)
    assert any("  - Nested 1.2" in line for line in plain)
    assert any("- Item 2" in line for line in plain)


def test_markdown_renders_ordered_nested_list_structure() -> None:
    markdown = Markdown(
        "1. First\n   1. Nested first\n   2. Nested second\n2. Second",
        0,
        0,
        _test_markdown_theme(),
    )
    plain = markdown.render(80)
    assert any("1. First" in line for line in plain)
    assert any("  1. Nested first" in line for line in plain)
    assert any("  2. Nested second" in line for line in plain)
    assert any("2. Second" in line for line in plain)


def test_markdown_renders_deeply_nested_list_with_two_space_indentation_per_level() -> None:
    markdown = Markdown(
        "- Level 1\n  - Level 2\n    - Level 3\n      - Level 4",
        0,
        0,
        _styled_markdown_theme(),
    )
    plain = [_strip_ansi(line).rstrip() for line in markdown.render(80) if line.strip()]

    assert plain == [
        "- Level 1",
        "  - Level 2",
        "    - Level 3",
        "      - Level 4",
    ]


def test_markdown_renders_mixed_nested_lists_without_extra_indent() -> None:
    markdown = Markdown(
        "1. Ordered item\n   - Unordered nested\n   - Another nested\n2. Second ordered\n   - More nested",
        0,
        0,
        _styled_markdown_theme(),
    )
    plain = [_strip_ansi(line).rstrip() for line in markdown.render(80) if line.strip()]

    assert plain == [
        "1. Ordered item",
        "  - Unordered nested",
        "  - Another nested",
        "2. Second ordered",
        "  - More nested",
    ]


def test_markdown_renders_simple_nested_list_structure() -> None:
    markdown = Markdown(
        "- Item 1\n  - Nested 1.1\n  - Nested 1.2\n- Item 2",
        0,
        0,
        _test_markdown_theme(),
    )
    plain = [_strip_ansi(line).rstrip() for line in markdown.render(80) if line.strip()]

    assert any("- Item 1" in line for line in plain)
    assert any("  - Nested 1.1" in line for line in plain)
    assert any("  - Nested 1.2" in line for line in plain)
    assert any("- Item 2" in line for line in plain)


def test_markdown_renders_ordered_nested_list_structure() -> None:
    markdown = Markdown(
        "1. First\n   1. Nested first\n   2. Nested second\n2. Second",
        0,
        0,
        _test_markdown_theme(),
    )
    plain = [_strip_ansi(line).rstrip() for line in markdown.render(80) if line.strip()]

    assert any("1. First" in line for line in plain)
    assert any("  1. Nested first" in line for line in plain)
    assert any("  2. Nested second" in line for line in plain)
    assert any("2. Second" in line for line in plain)


def test_markdown_renders_deeply_nested_list_structure() -> None:
    markdown = Markdown(
        "- Level 1\n  - Level 2\n    - Level 3\n      - Level 4",
        0,
        0,
        _test_markdown_theme(),
    )
    plain = [_strip_ansi(line).rstrip() for line in markdown.render(80) if line.strip()]

    assert any("- Level 1" in line for line in plain)
    assert any("  - Level 2" in line for line in plain)
    assert any("    - Level 3" in line for line in plain)
    assert any("      - Level 4" in line for line in plain)


def test_markdown_renders_mixed_ordered_and_unordered_nested_lists() -> None:
    markdown = Markdown(
        "1. Ordered item\n   - Unordered nested\n   - Another nested\n2. Second ordered\n   - More nested",
        0,
        0,
        _test_markdown_theme(),
    )
    plain = [_strip_ansi(line).rstrip() for line in markdown.render(80) if line.strip()]

    assert any("1. Ordered item" in line for line in plain)
    assert any("  - Unordered nested" in line for line in plain)
    assert any("  - Another nested" in line for line in plain)
    assert any("2. Second ordered" in line for line in plain)
    assert any("  - More nested" in line for line in plain)


def test_markdown_should_render_simple_nested_list_exact_title_like_ts() -> None:
    markdown = Markdown(
        "- Item 1\n  - Nested 1.1\n  - Nested 1.2\n- Item 2",
        0,
        0,
        _test_markdown_theme(),
    )
    plain = [_strip_ansi(line).rstrip() for line in markdown.render(80) if line.strip()]

    assert any("- Item 1" in line for line in plain)
    assert any("  - Nested 1.1" in line for line in plain)
    assert any("  - Nested 1.2" in line for line in plain)
    assert any("- Item 2" in line for line in plain)


def test_markdown_should_render_ordered_nested_list_exact_title_like_ts() -> None:
    markdown = Markdown(
        "1. First\n   1. Nested first\n   2. Nested second\n2. Second",
        0,
        0,
        _test_markdown_theme(),
    )
    plain = [_strip_ansi(line).rstrip() for line in markdown.render(80) if line.strip()]

    assert any("1. First" in line for line in plain)
    assert any("  1. Nested first" in line for line in plain)
    assert any("  2. Nested second" in line for line in plain)
    assert any("2. Second" in line for line in plain)


def test_markdown_should_render_deeply_nested_list_exact_title_like_ts() -> None:
    markdown = Markdown(
        "- Level 1\n  - Level 2\n    - Level 3\n      - Level 4",
        0,
        0,
        _test_markdown_theme(),
    )
    plain = [_strip_ansi(line).rstrip() for line in markdown.render(80) if line.strip()]

    assert any("- Level 1" in line for line in plain)
    assert any("  - Level 2" in line for line in plain)
    assert any("    - Level 3" in line for line in plain)
    assert any("      - Level 4" in line for line in plain)


def test_markdown_should_render_mixed_ordered_and_unordered_nested_lists_exact_title_like_ts() -> None:
    markdown = Markdown(
        "1. Ordered item\n   - Unordered nested\n   - Another nested\n2. Second ordered\n   - More nested",
        0,
        0,
        _test_markdown_theme(),
    )
    plain = [_strip_ansi(line).rstrip() for line in markdown.render(80) if line.strip()]

    assert any("1. Ordered item" in line for line in plain)
    assert any("  - Unordered nested" in line for line in plain)
    assert any("  - Another nested" in line for line in plain)
    assert any("2. Second ordered" in line for line in plain)
    assert any("  - More nested" in line for line in plain)


def test_markdown_renders_simple_table() -> None:
    markdown = Markdown(
        "| Name | Age |\n| --- | --- |\n| Alice | 30 |\n| Bob | 25 |",
        0,
        0,
        _test_markdown_theme(),
    )
    plain = markdown.render(80)
    assert any("Name" in line for line in plain)
    assert any("Alice" in line for line in plain)
    assert any("│" in line for line in plain)
    assert any("─" in line for line in plain)


def test_markdown_should_render_simple_table_like_ts() -> None:
    markdown = Markdown(
        "| Name | Age |\n| --- | --- |\n| Alice | 30 |\n| Bob | 25 |",
        0,
        0,
        _test_markdown_theme(),
    )
    plain = markdown.render(80)

    assert any("Name" in line for line in plain)
    assert any("Age" in line for line in plain)
    assert any("Alice" in line for line in plain)
    assert any("Bob" in line for line in plain)
    assert any("│" in line for line in plain)
    assert any("─" in line for line in plain)


def test_markdown_table_does_not_end_with_trailing_blank_line() -> None:
    markdown = Markdown(
        "| Name |\n| --- |\n| Alice |",
        0,
        0,
        _test_markdown_theme(),
    )
    plain = [line.rstrip() for line in markdown.render(80)]
    assert plain[-1] != ""


def test_markdown_table_renders_row_dividers_between_data_rows() -> None:
    markdown = Markdown(
        "| Name | Age |\n| --- | --- |\n| Alice | 30 |\n| Bob | 25 |",
        0,
        0,
        _test_markdown_theme(),
    )
    plain = [line.rstrip() for line in markdown.render(80)]
    divider_lines = [line for line in plain if "┼" in line]
    assert len(divider_lines) == 2


def test_markdown_should_render_row_dividers_between_data_rows_like_ts() -> None:
    markdown = Markdown(
        "| Name | Age |\n| --- | --- |\n| Alice | 30 |\n| Bob | 25 |",
        0,
        0,
        _test_markdown_theme(),
    )
    plain = [line.rstrip() for line in markdown.render(80)]
    divider_lines = [line for line in plain if "┼" in line]

    assert len(divider_lines) == 2


def test_markdown_combines_heading_list_and_table() -> None:
    markdown = Markdown(
        "# Test Document\n\n- Item 1\n  - Nested item\n- Item 2\n\n| Col1 | Col2 |\n| --- | --- |\n| A | B |",
        0,
        0,
        _test_markdown_theme(),
    )
    plain = markdown.render(80)
    assert any("Test Document" in line for line in plain)
    assert any("- Item 1" in line for line in plain)
    assert any("Nested item" in line for line in plain)
    assert any("Col1" in line and "Col2" in line for line in plain)
    assert any("│" in line for line in plain)


def test_markdown_wraps_wide_tables_to_available_width() -> None:
    markdown = Markdown(
        "| Command | Description | Example |\n| --- | --- | --- |\n| npm install | Install all dependencies | npm install |\n| npm run build | Build the project | npm run build |",
        0,
        0,
        _test_markdown_theme(),
    )
    plain = [line.rstrip() for line in markdown.render(50)]
    assert all(len(line) <= 50 for line in plain)
    joined = " ".join(plain)
    assert "Command" in joined
    assert "Description" in joined
    assert "npm install" in joined
    assert "Install" in joined


def test_markdown_wraps_long_table_cell_content_to_multiple_rows() -> None:
    markdown = Markdown(
        "| Header |\n| --- |\n| This is a very long cell content that should wrap |",
        0,
        0,
        _test_markdown_theme(),
    )
    plain = [line.rstrip() for line in markdown.render(25)]
    data_rows = [line for line in plain if line.startswith("│") and "─" not in line]
    assert len(data_rows) > 2
    joined = " ".join(plain)
    assert "very long" in joined
    assert "cell content" in joined
    assert "should wrap" in joined


def test_markdown_should_wrap_long_cell_content_to_multiple_lines_like_ts() -> None:
    markdown = Markdown(
        "| Header |\n| --- |\n| This is a very long cell content that should wrap |",
        0,
        0,
        _test_markdown_theme(),
    )
    plain = [line.rstrip() for line in markdown.render(25)]
    data_rows = [line for line in plain if line.startswith("│") and "─" not in line]
    joined = " ".join(plain)

    assert len(data_rows) > 2
    assert "very long" in joined
    assert "cell content" in joined
    assert "should wrap" in joined


def test_markdown_wraps_long_unbroken_tokens_inside_table_cells() -> None:
    url = "https://example.com/this/is/a/very/long/url/that/should/wrap"
    markdown = Markdown(
        f"| Value |\n| --- |\n| prefix {url} |",
        0,
        0,
        _test_markdown_theme(),
    )
    width = 30
    plain = [line.rstrip() for line in markdown.render(width)]
    assert all(len(line) <= width for line in plain)
    table_lines = [line for line in plain if line.startswith("│")]
    assert table_lines
    assert all(line.count("│") == 2 for line in table_lines)
    extracted = "".join(plain).replace("│", "").replace("├", "").replace("┤", "").replace("─", "").replace(" ", "")
    assert "prefix" in extracted
    assert url in extracted


def test_markdown_should_wrap_long_unbroken_tokens_inside_table_cells_not_only_at_line_start_like_ts() -> None:
    url = "https://example.com/this/is/a/very/long/url/that/should/wrap"
    markdown = Markdown(
        f"| Value |\n| --- |\n| prefix {url} |",
        0,
        0,
        _test_markdown_theme(),
    )
    width = 30
    plain = [line.rstrip() for line in markdown.render(width)]
    table_lines = [line for line in plain if line.startswith("│")]
    extracted = "".join(plain).replace("│", "").replace("├", "").replace("┤", "").replace("─", "").replace(" ", "")

    assert all(len(line) <= width for line in plain)
    assert table_lines
    assert all(line.count("│") == 2 for line in table_lines)
    assert "prefix" in extracted
    assert url in extracted


def test_markdown_wraps_styled_inline_code_inside_table_cells_without_breaking_borders() -> None:
    markdown = Markdown(
        "| Code |\n| --- |\n| `averyveryveryverylongidentifier` |",
        0,
        0,
        _styled_markdown_theme(),
    )
    width = 20
    lines = markdown.render(width)
    assert "\x1b[33m" in "\n".join(lines)
    plain = [line.replace("\x1b[33m", "").replace("\x1b[0m", "").rstrip() for line in lines]
    assert all(len(line) <= width for line in plain)
    table_lines = [line for line in plain if line.startswith("│")]
    assert all(line.count("│") == 2 for line in table_lines)


def test_markdown_should_wrap_styled_inline_code_inside_table_cells_without_breaking_borders_like_ts() -> None:
    markdown = Markdown(
        "| Code |\n| --- |\n| `averyveryveryverylongidentifier` |",
        0,
        0,
        _styled_markdown_theme(),
    )
    width = 20
    lines = markdown.render(width)
    plain = [line.replace("\x1b[33m", "").replace("\x1b[0m", "").rstrip() for line in lines]
    table_lines = [line for line in plain if line.startswith("│")]

    assert "\x1b[33m" in "\n".join(lines)
    assert all(len(line) <= width for line in plain)
    assert all(line.count("│") == 2 for line in table_lines)


def test_markdown_handles_extremely_narrow_table_width_gracefully_like_ts() -> None:
    markdown = Markdown(
        "| A | B | C |\n| --- | --- | --- |\n| 1 | 2 | 3 |",
        0,
        0,
        _test_markdown_theme(),
    )
    plain = [line.rstrip() for line in markdown.render(15)]

    assert plain
    assert all(len(line) <= 15 for line in plain)


def test_markdown_should_handle_extremely_narrow_width_gracefully_like_ts() -> None:
    markdown = Markdown(
        "| A | B | C |\n| --- | --- | --- |\n| 1 | 2 | 3 |",
        0,
        0,
        _test_markdown_theme(),
    )
    plain = [line.rstrip() for line in markdown.render(15)]

    assert plain
    assert all(len(line) <= 15 for line in plain)


def test_markdown_table_fits_naturally_with_expected_structure() -> None:
    markdown = Markdown(
        "| A | B |\n| --- | --- |\n| 1 | 2 |",
        0,
        0,
        _test_markdown_theme(),
    )
    plain = [line.rstrip() for line in markdown.render(80)]

    header_line = next(line for line in plain if "A" in line and "B" in line)
    separator_line = next(line for line in plain if "├" in line and "┼" in line)
    data_line = next(line for line in plain if "1" in line and "2" in line)

    assert "│" in header_line
    assert separator_line
    assert data_line


def test_markdown_table_respects_padding_x_in_width_calculation() -> None:
    markdown = Markdown(
        "| Column One | Column Two |\n| --- | --- |\n| Data 1 | Data 2 |",
        2,
        0,
        _test_markdown_theme(),
    )
    plain = [line.rstrip() for line in markdown.render(40)]

    assert all(len(line) <= 40 for line in plain)
    table_row = next(line for line in plain if "│" in line)
    assert table_row.startswith("  ")


def test_markdown_respects_padding_x_when_calculating_table_width_like_ts() -> None:
    markdown = Markdown(
        "| Column One | Column Two |\n| --- | --- |\n| Data 1 | Data 2 |",
        2,
        0,
        _test_markdown_theme(),
    )
    plain = [line.rstrip() for line in markdown.render(40)]

    assert all(len(line) <= 40 for line in plain)
    table_row = next(line for line in plain if "│" in line)
    assert table_row.startswith("  ")


def test_markdown_table_width_calculation_respects_padding_x_like_ts() -> None:
    markdown = Markdown(
        "| Column One | Column Two |\n| --- | --- |\n| Data 1 | Data 2 |",
        2,
        0,
        _test_markdown_theme(),
    )
    plain = [line.rstrip() for line in markdown.render(40)]

    assert all(len(line) <= 40 for line in plain)
    table_row = next(line for line in plain if "│" in line)
    assert table_row.startswith("  ")


def test_markdown_should_respect_padding_x_when_calculating_table_width_like_ts() -> None:
    markdown = Markdown(
        "| Column One | Column Two |\n| --- | --- |\n| Data 1 | Data 2 |",
        2,
        0,
        _test_markdown_theme(),
    )
    plain = [line.rstrip() for line in markdown.render(40)]

    assert all(len(line) <= 40 for line in plain)
    table_row = next(line for line in plain if "│" in line)
    assert table_row.startswith("  ")


def test_markdown_should_respect_paddingx_when_calculating_table_width_like_ts() -> None:
    markdown = Markdown(
        "| Column One | Column Two |\n| --- | --- |\n| Data 1 | Data 2 |",
        2,
        0,
        _test_markdown_theme(),
    )
    plain = [line.rstrip() for line in markdown.render(40)]

    assert all(len(line) <= 40 for line in plain)
    table_row = next(line for line in plain if "│" in line)
    assert table_row.startswith("  ")


def test_markdown_table_last_block_has_no_trailing_blank_line_like_ts() -> None:
    markdown = Markdown(
        "| Name |\n| --- |\n| Alice |",
        0,
        0,
        _test_markdown_theme(),
    )
    plain = [line.rstrip() for line in markdown.render(80)]

    assert plain[-1] != ""


def test_markdown_should_not_add_a_trailing_blank_line_when_table_is_the_last_rendered_block_like_ts() -> None:
    markdown = Markdown(
        "| Name |\n| --- |\n| Alice |",
        0,
        0,
        _test_markdown_theme(),
    )
    plain = [line.rstrip() for line in markdown.render(80)]

    assert plain[-1] != ""


def test_markdown_renders_lists_and_tables_together_like_ts() -> None:
    markdown = Markdown(
        "# Test Document\n\n- Item 1\n  - Nested item\n- Item 2\n\n| Col1 | Col2 |\n| --- | --- |\n| A | B |",
        0,
        0,
        _test_markdown_theme(),
    )
    plain = [_strip_ansi(line) for line in markdown.render(80)]

    assert any("Test Document" in line for line in plain)
    assert any("- Item 1" in line for line in plain)
    assert any("  - Nested item" in line for line in plain)
    assert any("Col1" in line for line in plain)
    assert any("│" in line for line in plain)


def test_markdown_should_render_lists_and_tables_together_like_ts() -> None:
    markdown = Markdown(
        "# Test Document\n\n- Item 1\n  - Nested item\n- Item 2\n\n| Col1 | Col2 |\n| --- | --- |\n| A | B |",
        0,
        0,
        _test_markdown_theme(),
    )
    plain = [_strip_ansi(line) for line in markdown.render(80)]

    assert any("Test Document" in line for line in plain)
    assert any("- Item 1" in line for line in plain)
    assert any("  - Nested item" in line for line in plain)
    assert any("Col1" in line for line in plain)
    assert any("│" in line for line in plain)


def test_markdown_divider_last_block_has_no_trailing_blank_line_like_ts() -> None:
    markdown = Markdown("---", 0, 0, _test_markdown_theme())
    plain = [line.rstrip() for line in markdown.render(80)]

    assert plain[-1] != ""


def test_markdown_should_not_add_a_trailing_blank_line_when_divider_is_the_last_rendered_block_like_ts() -> None:
    markdown = Markdown("---", 0, 0, _test_markdown_theme())
    plain = [line.rstrip() for line in markdown.render(80)]

    assert plain[-1] != ""


def test_markdown_heading_last_block_has_no_trailing_blank_line_like_ts() -> None:
    markdown = Markdown("# Hello", 0, 0, _test_markdown_theme())
    plain = [line.rstrip() for line in markdown.render(80)]

    assert plain[-1] != ""


def test_markdown_should_not_add_a_trailing_blank_line_when_heading_is_the_last_rendered_block_like_ts() -> None:
    markdown = Markdown("# Hello", 0, 0, _test_markdown_theme())
    plain = [line.rstrip() for line in markdown.render(80)]

    assert plain[-1] != ""


def test_markdown_blockquote_last_block_has_no_trailing_blank_line_like_ts() -> None:
    markdown = Markdown("> This is a quote", 0, 0, _test_markdown_theme())
    plain = [line.rstrip() for line in markdown.render(80)]

    assert plain[-1] != ""


def test_markdown_should_not_add_a_trailing_blank_line_when_blockquote_is_the_last_rendered_block_like_ts() -> None:
    markdown = Markdown("> This is a quote", 0, 0, _test_markdown_theme())
    plain = [line.rstrip() for line in markdown.render(80)]

    assert plain[-1] != ""


def test_markdown_preserves_ordered_list_numbering_across_unindented_code_blocks() -> None:
    markdown = Markdown(
        "1. First item\n\n```typescript\n// code block\n```\n\n2. Second item\n\n```typescript\n// another code block\n```\n\n3. Third item",
        0,
        0,
        _test_markdown_theme(),
    )
    plain = [line.rstrip() for line in markdown.render(80)]
    numbered = [line.strip() for line in plain if line.strip() and line.strip()[0].isdigit()]

    assert len(numbered) == 3
    assert numbered[0].startswith("1.")
    assert numbered[1].startswith("2.")
    assert numbered[2].startswith("3.")


def test_markdown_maintains_numbering_when_code_blocks_are_not_indented_like_ts() -> None:
    markdown = Markdown(
        "1. First item\n\n```typescript\n// code block\n```\n\n2. Second item\n\n```typescript\n// another code block\n```\n\n3. Third item",
        0,
        0,
        _test_markdown_theme(),
    )
    plain = [line.rstrip() for line in markdown.render(80)]
    numbered = [line.strip() for line in plain if line.strip() and line.strip()[0].isdigit()]

    assert len(numbered) == 3
    assert numbered[0].startswith("1.")
    assert numbered[1].startswith("2.")
    assert numbered[2].startswith("3.")


def test_markdown_should_maintain_numbering_when_code_blocks_are_not_indented_llm_output_like_ts() -> None:
    markdown = Markdown(
        "1. First item\n\n```typescript\n// code block\n```\n\n2. Second item\n\n```typescript\n// another code block\n```\n\n3. Third item",
        0,
        0,
        _test_markdown_theme(),
    )
    plain = [line.rstrip() for line in markdown.render(80)]
    numbered = [line.strip() for line in plain if line.strip() and line.strip()[0].isdigit()]

    assert len(numbered) == 3
    assert numbered[0].startswith("1.")
    assert numbered[1].startswith("2.")
    assert numbered[2].startswith("3.")


def test_markdown_table_column_width_is_at_least_longest_word() -> None:
    longest_word = "superlongword"
    markdown = Markdown(
        f"| Column One | Column Two |\n| --- | --- |\n| {longest_word} short | otherword |\n| small | tiny |",
        0,
        0,
        _test_markdown_theme(),
    )

    plain = [line.rstrip() for line in markdown.render(32)]
    data_line = next(line for line in plain if longest_word in line)
    segments = data_line.split("│")[1:-1]
    first_segment = segments[0]
    first_column_width = len(first_segment) - 2

    assert first_column_width >= len(longest_word)


def test_markdown_should_keep_column_width_at_least_the_longest_word_like_ts() -> None:
    longest_word = "superlongword"
    markdown = Markdown(
        f"| Column One | Column Two |\n| --- | --- |\n| {longest_word} short | otherword |\n| small | tiny |",
        0,
        0,
        _test_markdown_theme(),
    )

    plain = [line.rstrip() for line in markdown.render(32)]
    data_line = next(line for line in plain if longest_word in line)
    segments = data_line.split("│")[1:-1]
    first_segment = segments[0]
    first_column_width = len(first_segment) - 2

    assert first_column_width >= len(longest_word)


def test_markdown_renders_table_with_alignment() -> None:
    markdown = Markdown(
        "| Left | Center | Right |\n| :--- | :---: | ---: |\n| A | B | C |\n| Long text | Middle | End |",
        0,
        0,
        _test_markdown_theme(),
    )
    plain = [line.rstrip() for line in markdown.render(80)]

    assert any("Left" in line and "Center" in line and "Right" in line for line in plain)
    assert any("Long text" in line and "Middle" in line and "End" in line for line in plain)


def test_markdown_should_render_table_with_alignment_like_ts() -> None:
    markdown = Markdown(
        "| Left | Center | Right |\n| :--- | :---: | ---: |\n| A | B | C |\n| Long text | Middle | End |",
        0,
        0,
        _test_markdown_theme(),
    )
    plain = [line.rstrip() for line in markdown.render(80)]

    assert any("Left" in line and "Center" in line and "Right" in line for line in plain)
    assert any("Long text" in line for line in plain)


def test_markdown_handles_tables_with_varying_column_widths() -> None:
    markdown = Markdown(
        "| Short | Very long column header |\n| --- | --- |\n| A | This is a much longer cell content |\n| B | Short |",
        0,
        0,
        _test_markdown_theme(),
    )
    plain = [line.rstrip() for line in markdown.render(80)]

    assert any("Very long column header" in line for line in plain)
    assert any("This is a much longer cell content" in line for line in plain)


def test_markdown_should_handle_tables_with_varying_column_widths_like_ts() -> None:
    markdown = Markdown(
        "| Short | Very long column header |\n| --- | --- |\n| A | This is a much longer cell content |\n| B | Short |",
        0,
        0,
        _test_markdown_theme(),
    )
    plain = [line.rstrip() for line in markdown.render(80)]

    assert plain
    assert any("Very long column header" in line for line in plain)
    assert any("This is a much longer cell content" in line for line in plain)


def test_markdown_wraps_table_when_it_exceeds_available_width() -> None:
    markdown = Markdown(
        "| Command | Description | Example |\n| --- | --- | --- |\n| npm install | Install all dependencies | npm install |\n| npm run build | Build the project | npm run build |",
        0,
        0,
        _test_markdown_theme(),
    )
    plain = [line.rstrip() for line in markdown.render(50)]

    assert all(len(line) <= 50 for line in plain)
    joined = " ".join(plain)
    assert "Command" in joined
    assert "Description" in joined
    assert "npm install" in joined
    assert "Install" in joined


def test_markdown_should_wrap_table_cells_when_table_exceeds_available_width_like_ts() -> None:
    markdown = Markdown(
        "| Command | Description | Example |\n| --- | --- | --- |\n| npm install | Install all dependencies | npm install |\n| npm run build | Build the project | npm run build |",
        0,
        0,
        _test_markdown_theme(),
    )
    plain = [line.rstrip() for line in markdown.render(50)]
    joined = " ".join(plain)

    assert all(len(line) <= 50 for line in plain)
    assert "Command" in joined
    assert "Description" in joined
    assert "npm install" in joined
    assert "Install" in joined


def test_markdown_renders_aligned_and_varying_width_tables() -> None:
    aligned = Markdown(
        "| Left | Center | Right |\n| :--- | :---: | ---: |\n| A | B | C |\n| Long text | Middle | End |",
        0,
        0,
        _test_markdown_theme(),
    )
    aligned_plain = [line.rstrip() for line in aligned.render(80)]

    assert any("Left" in line and "Center" in line and "Right" in line for line in aligned_plain)
    assert any("Long text" in line and "Middle" in line and "End" in line for line in aligned_plain)

    varying = Markdown(
        "| Short | Very long column header |\n| --- | --- |\n| A | This is a much longer cell content |\n| B | Short |",
        0,
        0,
        _test_markdown_theme(),
    )
    varying_plain = [line.rstrip() for line in varying.render(80)]

    assert any("Very long column header" in line for line in varying_plain)
    assert any("This is a much longer cell content" in line for line in varying_plain)


def test_markdown_respects_padding_x_when_rendering_tables() -> None:
    markdown = Markdown(
        "| Column One | Column Two |\n| --- | --- |\n| Data 1 | Data 2 |",
        2,
        0,
        _test_markdown_theme(),
    )
    plain = [line.rstrip() for line in markdown.render(40)]

    assert all(len(line) <= 40 for line in plain)
    table_row = next(line for line in plain if "│" in line)
    assert table_row.startswith("  ")


def test_markdown_handles_extremely_narrow_table_width_gracefully() -> None:
    markdown = Markdown(
        "| A | B | C |\n| --- | --- | --- |\n| 1 | 2 | 3 |",
        0,
        0,
        _test_markdown_theme(),
    )
    plain = [line.rstrip() for line in markdown.render(15)]
    assert plain
    assert all(len(line) <= 15 for line in plain)


def test_markdown_table_renders_correctly_when_it_fits_naturally() -> None:
    markdown = Markdown(
        "| A | B |\n| --- | --- |\n| 1 | 2 |",
        0,
        0,
        _test_markdown_theme(),
    )
    plain = [line.rstrip() for line in markdown.render(80)]

    header_line = next(line for line in plain if "A" in line and "B" in line)
    separator_line = next(line for line in plain if "├" in line and "┼" in line)
    data_line = next(line for line in plain if "1" in line and "2" in line)

    assert "│" in header_line
    assert separator_line
    assert data_line


def test_markdown_should_render_table_correctly_when_it_fits_naturally_like_ts() -> None:
    markdown = Markdown(
        "| A | B |\n| --- | --- |\n| 1 | 2 |",
        0,
        0,
        _test_markdown_theme(),
    )
    plain = [line.rstrip() for line in markdown.render(80)]

    header_line = next(line for line in plain if "A" in line and "B" in line)
    separator_line = next(line for line in plain if "├" in line and "┼" in line)
    data_line = next(line for line in plain if "1" in line and "2" in line)

    assert "│" in header_line
    assert separator_line
    assert data_line


def test_markdown_renders_fenced_code_block() -> None:
    markdown = Markdown(
        "```python\nprint('hello')\n```",
        0,
        0,
        _test_markdown_theme(),
    )
    plain = markdown.render(80)
    assert any("```python" in line for line in plain)
    assert any("print('hello')" in line for line in plain)
    assert any("```" == line.strip() for line in plain)


def test_markdown_wraps_blockquote_lines_with_border_on_each_line() -> None:
    markdown = Markdown(
        "> This is a very long blockquote line that should wrap to multiple lines when rendered",
        0,
        0,
        _test_markdown_theme(),
    )

    plain = [line.rstrip() for line in markdown.render(30) if line.strip()]

    assert len(plain) > 1
    assert all(line.startswith("│ ") for line in plain)


def test_markdown_wrapped_blockquote_preserves_content_like_ts() -> None:
    markdown = Markdown(
        "> This is a very long blockquote line that should wrap to multiple lines when rendered",
        0,
        0,
        _styled_markdown_theme(),
    )

    plain = [_strip_ansi(line).rstrip() for line in markdown.render(30) if line.strip()]
    joined = " ".join(plain)

    assert len(plain) > 1
    assert all(line.startswith("│ ") for line in plain)
    assert "very long" in joined
    assert "blockquote" in joined
    assert "multiple" in joined


def test_markdown_should_wrap_long_blockquote_lines_and_add_border_to_each_wrapped_line_like_ts() -> None:
    markdown = Markdown(
        "> This is a very long blockquote line that should wrap to multiple lines when rendered",
        0,
        0,
        _styled_markdown_theme(),
    )

    plain = [_strip_ansi(line).rstrip() for line in markdown.render(30) if line.strip()]
    joined = " ".join(plain)

    assert len(plain) > 1
    assert all(line.startswith("│ ") for line in plain)
    assert "very long" in joined
    assert "blockquote" in joined
    assert "multiple" in joined


def test_markdown_wrapped_blockquote_keeps_quote_border_on_every_nonempty_line_like_ts() -> None:
    markdown = Markdown(
        "> This is a very long blockquote line that should wrap to multiple lines when rendered",
        0,
        0,
        _styled_markdown_theme(),
    )

    plain = [_strip_ansi(line).rstrip() for line in markdown.render(30)]
    content_lines = [line for line in plain if line]

    assert len(content_lines) > 1
    assert all(line.startswith("│ ") for line in content_lines)


def test_markdown_lazy_continuation_blockquote_applies_consistent_styling_to_all_lines_like_ts() -> None:
    markdown = Markdown(
        ">Foo\nbar",
        0,
        0,
        _styled_markdown_theme(),
        {
            "color": lambda text: _ansi("35", text),
        },
    )

    lines = markdown.render(80)
    foo_line = next(line for line in lines if "Foo" in line)
    bar_line = next(line for line in lines if "bar" in line)
    plain = [_strip_ansi(line).rstrip() for line in lines]
    quoted = [line for line in plain if line.startswith("│ ")]

    assert len(quoted) == 2
    assert "\x1b[3m" in foo_line
    assert "\x1b[3m" in bar_line
    assert "\x1b[35m" not in foo_line
    assert "\x1b[35m" not in bar_line


def test_markdown_should_apply_consistent_styling_to_all_lines_in_lazy_continuation_blockquote_like_ts() -> None:
    markdown = Markdown(
        ">Foo\nbar",
        0,
        0,
        _styled_markdown_theme(),
        {
            "color": lambda text: _ansi("35", text),
        },
    )

    lines = markdown.render(80)
    foo_line = next(line for line in lines if "Foo" in line)
    bar_line = next(line for line in lines if "bar" in line)
    plain = [_strip_ansi(line).rstrip() for line in lines]
    quoted = [line for line in plain if line.startswith("│ ")]

    assert len(quoted) == 2
    assert "\x1b[3m" in foo_line
    assert "\x1b[3m" in bar_line
    assert "\x1b[35m" not in foo_line
    assert "\x1b[35m" not in bar_line


def test_markdown_lazy_continuation_blockquote_does_not_inherit_default_color() -> None:
    markdown = Markdown(
        ">Foo\nbar",
        0,
        0,
        _styled_markdown_theme(),
        {
            "color": lambda text: _ansi("35", text),
        },
    )

    lines = markdown.render(80)
    foo_line = next(line for line in lines if "Foo" in line)
    bar_line = next(line for line in lines if "bar" in line)
    plain = [_strip_ansi(line).rstrip() for line in lines]
    quoted = [line for line in plain if line.startswith("│ ")]

    assert len(quoted) == 2
    assert "\x1b[3m" in foo_line
    assert "\x1b[3m" in bar_line
    assert "\x1b[35m" not in foo_line
    assert "\x1b[35m" not in bar_line


def test_markdown_explicit_multiline_blockquote_does_not_inherit_default_color() -> None:
    markdown = Markdown(
        ">Foo\n>bar",
        0,
        0,
        _styled_markdown_theme(),
        {
            "color": lambda text: _ansi("36", text),
        },
    )

    lines = markdown.render(80)
    foo_line = next(line for line in lines if "Foo" in line)
    bar_line = next(line for line in lines if "bar" in line)
    plain = [_strip_ansi(line).rstrip() for line in lines]
    quoted = [line for line in plain if line.startswith("│ ")]

    assert len(quoted) == 2
    assert "\x1b[3m" in foo_line
    assert "\x1b[3m" in bar_line
    assert "\x1b[36m" not in foo_line
    assert "\x1b[36m" not in bar_line


def test_markdown_explicit_multiline_blockquote_applies_consistent_styling_to_all_lines_like_ts() -> None:
    markdown = Markdown(
        ">Foo\n>bar",
        0,
        0,
        _styled_markdown_theme(),
        {
            "color": lambda text: _ansi("36", text),
        },
    )

    lines = markdown.render(80)
    foo_line = next(line for line in lines if "Foo" in line)
    bar_line = next(line for line in lines if "bar" in line)
    plain = [_strip_ansi(line).rstrip() for line in lines]
    quoted = [line for line in plain if line.startswith("│ ")]

    assert len(quoted) == 2
    assert "\x1b[3m" in foo_line
    assert "\x1b[3m" in bar_line


def test_markdown_should_apply_consistent_styling_to_explicit_multiline_blockquote_like_ts() -> None:
    markdown = Markdown(
        ">Foo\n>bar",
        0,
        0,
        _styled_markdown_theme(),
        {
            "color": lambda text: _ansi("36", text),
        },
    )

    lines = markdown.render(80)
    foo_line = next(line for line in lines if "Foo" in line)
    bar_line = next(line for line in lines if "bar" in line)
    plain = [_strip_ansi(line).rstrip() for line in lines]
    quoted = [line for line in plain if line.startswith("│ ")]

    assert len(quoted) == 2
    assert "\x1b[3m" in foo_line
    assert "\x1b[3m" in bar_line
    assert "\x1b[36m" not in foo_line
    assert "\x1b[36m" not in bar_line
    assert "\x1b[36m" not in foo_line
    assert "\x1b[36m" not in bar_line


def test_markdown_renders_list_content_inside_blockquotes() -> None:
    markdown = Markdown(
        "> 1. bla bla\n> - nested bullet",
        0,
        0,
        _styled_markdown_theme(),
    )

    plain = [_strip_ansi(line).rstrip() for line in markdown.render(80)]
    quoted = [line for line in plain if line.startswith("│ ")]

    assert any("1. bla bla" in line for line in quoted)
    assert any("- nested bullet" in line for line in quoted)


def test_markdown_should_render_list_content_inside_blockquotes_like_ts() -> None:
    markdown = Markdown(
        "> 1. bla bla\n> - nested bullet",
        0,
        0,
        _styled_markdown_theme(),
    )

    plain = [_strip_ansi(line).rstrip() for line in markdown.render(80)]
    quoted = [line for line in plain if line.startswith("│ ")]

    assert any("1. bla bla" in line for line in quoted)
    assert any("- nested bullet" in line for line in quoted)


def test_markdown_wrapped_blockquotes_keep_quote_styling_without_default_color() -> None:
    markdown = Markdown(
        "> This is styled text that is long enough to wrap",
        0,
        0,
        _styled_markdown_theme(),
        {
            "color": lambda text: _ansi("33", text),
            "italic": True,
        },
    )

    lines = markdown.render(25)
    plain = [_strip_ansi(line).rstrip() for line in lines if line.strip()]
    joined = "\n".join(lines)

    assert all(line.startswith("│ ") for line in plain)
    assert "\x1b[3m" in joined
    assert "\x1b[33m" not in joined


def test_markdown_should_properly_indent_wrapped_blockquote_lines_with_styling_like_ts() -> None:
    markdown = Markdown(
        "> This is styled text that is long enough to wrap",
        0,
        0,
        _styled_markdown_theme(),
        {
            "color": lambda text: _ansi("33", text),
            "italic": True,
        },
    )

    lines = markdown.render(25)
    plain = [_strip_ansi(line).rstrip() for line in lines if line.strip()]
    joined = "\n".join(lines)

    assert all(line.startswith("│ ") for line in plain)
    assert "\x1b[3m" in joined
    assert "\x1b[33m" not in joined


def test_markdown_wrapped_blockquote_properly_indents_styled_lines_like_ts() -> None:
    markdown = Markdown(
        "> This is styled text that is long enough to wrap",
        0,
        0,
        _styled_markdown_theme(),
        {
            "color": lambda text: _ansi("33", text),
            "italic": True,
        },
    )

    lines = markdown.render(25)
    plain = [_strip_ansi(line).rstrip() for line in lines if line.strip()]
    joined = "\n".join(lines)

    assert all(line.startswith("│ ") for line in plain)
    assert "\x1b[3m" in joined
    assert "\x1b[33m" not in joined


def test_markdown_blockquotes_reapply_quote_style_after_bold_and_code() -> None:
    markdown = Markdown(
        "> Quote with **bold** and `code`",
        0,
        0,
        _styled_markdown_theme(),
    )

    lines = markdown.render(80)
    plain = [_strip_ansi(line) for line in lines]
    joined = "\n".join(lines)

    assert any(line.startswith("│ ") for line in plain)
    assert "Quote with bold and code" in " ".join(plain)
    assert "\x1b[1m" in joined
    assert "\x1b[33m" in joined
    assert "\x1b[3m" in joined


def test_markdown_should_render_inline_formatting_inside_blockquotes_and_reapply_quote_styling_after_like_ts() -> None:
    markdown = Markdown(
        "> Quote with **bold** and `code`",
        0,
        0,
        _styled_markdown_theme(),
    )

    lines = markdown.render(80)
    plain = [_strip_ansi(line) for line in lines]
    joined = "\n".join(lines)

    assert any(line.startswith("│ ") for line in plain)
    assert "Quote with bold and code" in " ".join(plain)
    assert "\x1b[1m" in joined
    assert "\x1b[33m" in joined
    assert "\x1b[3m" in joined


def test_markdown_code_block_has_single_blank_line_before_following_paragraph() -> None:
    markdown = Markdown(
        'hello world\n\n```js\nconst hello = "world";\n```\n\nagain, hello world',
        0,
        0,
        _test_markdown_theme(),
    )
    plain = [_strip_ansi(line).rstrip() for line in markdown.render(80)]

    closing_index = plain.index("```")
    after_closing = plain[closing_index + 1 :]
    assert next(index for index, line in enumerate(after_closing) if line != "") == 1


def test_markdown_normalizes_paragraph_and_code_block_spacing_like_ts() -> None:
    cases = [
        "hello this is text\n```\ncode block\n```\nmore text",
        "hello this is text\n\n```\ncode block\n```\n\nmore text",
    ]
    expected = ["hello this is text", "", "```", "  code block", "```", "", "more text"]

    for text in cases:
        markdown = Markdown(text, 0, 0, _test_markdown_theme())
        plain = [_strip_ansi(line).rstrip() for line in markdown.render(80)]
        assert plain == expected


def test_markdown_should_normalize_paragraph_and_code_block_spacing_to_one_blank_line_like_ts() -> None:
    cases = [
        "hello this is text\n```\ncode block\n```\nmore text",
        "hello this is text\n\n```\ncode block\n```\n\nmore text",
    ]
    expected = ["hello this is text", "", "```", "  code block", "```", "", "more text"]

    for text in cases:
        markdown = Markdown(text, 0, 0, _test_markdown_theme())
        plain = [_strip_ansi(line).rstrip() for line in markdown.render(80)]
        assert plain == expected


def test_markdown_code_block_last_block_has_no_trailing_blank_line_like_ts() -> None:
    cases = [
        "```js\nconst hello = 'world';\n```",
        "hello world\n\n```js\nconst hello = 'world';\n```",
    ]

    for text in cases:
        markdown = Markdown(text, 0, 0, _test_markdown_theme())
        plain = [_strip_ansi(line).rstrip() for line in markdown.render(80)]
        assert plain[-1] != ""


def test_markdown_should_not_add_a_trailing_blank_line_when_code_block_is_the_last_rendered_block_like_ts() -> None:
    cases = [
        "```js\nconst hello = 'world';\n```",
        "hello world\n\n```js\nconst hello = 'world';\n```",
    ]

    for text in cases:
        markdown = Markdown(text, 0, 0, _test_markdown_theme())
        plain = [_strip_ansi(line).rstrip() for line in markdown.render(80)]
        assert plain[-1] != ""


def test_markdown_renders_html_like_tags_as_text() -> None:
    markdown = Markdown(
        "This is text with <thinking>hidden content</thinking> that should be visible",
        0,
        0,
        _test_markdown_theme(),
    )

    joined = " ".join(markdown.render(80))
    assert "hidden content" in joined or "<thinking>" in joined


def test_markdown_should_render_content_with_html_like_tags_as_text_like_ts() -> None:
    markdown = Markdown(
        "This is text with <thinking>hidden content</thinking> that should be visible",
        0,
        0,
        _test_markdown_theme(),
    )

    joined = " ".join(markdown.render(80))

    assert "hidden content" in joined or "<thinking>" in joined


def test_markdown_html_like_tags_minimal_case_keeps_content_visible_like_ts() -> None:
    markdown = Markdown("<thinking>hidden content</thinking>", 0, 0, _test_markdown_theme())

    joined = " ".join(markdown.render(80))

    assert "hidden content" in joined or "<thinking>" in joined


def test_markdown_renders_html_tags_inside_code_blocks_literally() -> None:
    markdown = Markdown(
        "```html\n<div>Some HTML</div>\n```",
        0,
        0,
        _test_markdown_theme(),
    )

    joined = "\n".join(_strip_ansi(line) for line in markdown.render(80))
    assert "<div>" in joined
    assert "</div>" in joined


def test_markdown_should_render_html_tags_in_code_blocks_correctly_like_ts() -> None:
    markdown = Markdown(
        "```html\n<div>Some HTML</div>\n```",
        0,
        0,
        _test_markdown_theme(),
    )

    joined = "\n".join(_strip_ansi(line) for line in markdown.render(80))

    assert "<div>" in joined
    assert "</div>" in joined


def test_markdown_reapplies_default_text_style_after_inline_code_and_bold() -> None:
    markdown = Markdown(
        "This is thinking with `inline code` and **bold text** after",
        0,
        0,
        _styled_markdown_theme(),
        {
            "color": lambda text: _ansi("90", text),
            "italic": True,
        },
    )

    joined = "\n".join(markdown.render(80))

    assert joined.count("\x1b[90m") >= 2
    assert "\x1b[3m" in joined
    assert "\x1b[33m" in joined
    assert "\x1b[1m" in joined
    assert " after" in joined


def test_markdown_thinking_style_persists_after_inline_code() -> None:
    markdown = Markdown(
        "This is thinking with `inline code` and more text after",
        1,
        0,
        _styled_markdown_theme(),
        {
            "color": lambda text: _ansi("90", text),
            "italic": True,
        },
    )

    joined = "\n".join(markdown.render(80))

    assert "inline code" in joined
    assert "\x1b[90m" in joined
    assert "\x1b[3m" in joined
    assert "\x1b[33m" in joined


def test_markdown_preserves_gray_italic_styling_after_inline_code_like_ts() -> None:
    markdown = Markdown(
        "This is thinking with `inline code` and more text after",
        1,
        0,
        _styled_markdown_theme(),
        {
            "color": lambda text: _ansi("90", text),
            "italic": True,
        },
    )

    joined = "\n".join(markdown.render(80))

    assert "inline code" in joined
    assert "\x1b[90m" in joined
    assert "\x1b[3m" in joined
    assert "\x1b[33m" in joined


def test_markdown_should_preserve_gray_italic_styling_after_inline_code_like_ts() -> None:
    markdown = Markdown(
        "This is thinking with `inline code` and more text after",
        1,
        0,
        _styled_markdown_theme(),
        {
            "color": lambda text: _ansi("90", text),
            "italic": True,
        },
    )

    joined = "\n".join(markdown.render(80))

    assert "inline code" in joined
    assert "\x1b[90m" in joined
    assert "\x1b[3m" in joined
    assert "\x1b[33m" in joined


def test_markdown_thinking_style_persists_after_bold_text() -> None:
    markdown = Markdown(
        "This is thinking with **bold text** and more after",
        1,
        0,
        _styled_markdown_theme(),
        {
            "color": lambda text: _ansi("90", text),
            "italic": True,
        },
    )

    joined = "\n".join(markdown.render(80))

    assert "bold text" in joined
    assert "\x1b[90m" in joined
    assert "\x1b[3m" in joined
    assert "\x1b[1m" in joined


def test_markdown_preserves_gray_italic_styling_after_bold_text_like_ts() -> None:
    markdown = Markdown(
        "This is thinking with **bold text** and more after",
        1,
        0,
        _styled_markdown_theme(),
        {
            "color": lambda text: _ansi("90", text),
            "italic": True,
        },
    )

    joined = "\n".join(markdown.render(80))

    assert "bold text" in joined
    assert "\x1b[90m" in joined
    assert "\x1b[3m" in joined
    assert "\x1b[1m" in joined


def test_markdown_should_preserve_gray_italic_styling_after_bold_text_exact_title_like_ts() -> None:
    markdown = Markdown(
        "This is thinking with **bold text** and more after",
        1,
        0,
        _styled_markdown_theme(),
        {
            "color": lambda text: _ansi("90", text),
            "italic": True,
        },
    )

    joined = "\n".join(markdown.render(80))

    assert "bold text" in joined
    assert "\x1b[90m" in joined
    assert "\x1b[3m" in joined
    assert "\x1b[1m" in joined


def test_markdown_bare_url_is_not_duplicated() -> None:
    markdown = Markdown("Visit https://example.com for more", 0, 0, _test_markdown_theme())

    plain = " ".join(_strip_ansi(line) for line in markdown.render(80))

    assert plain.count("https://example.com") == 1


def test_markdown_bare_url_only_renders_once_in_minimal_case_like_ts() -> None:
    markdown = Markdown("https://example.com", 0, 0, _test_markdown_theme())

    plain = " ".join(_strip_ansi(line) for line in markdown.render(80))

    assert plain.count("https://example.com") == 1


def test_markdown_explicit_link_with_different_text_shows_url() -> None:
    markdown = Markdown("[click here](https://example.com)", 0, 0, _test_markdown_theme())

    plain = " ".join(_strip_ansi(line) for line in markdown.render(80))

    assert "click here" in plain
    assert "(https://example.com)" in plain


def test_markdown_autolinked_email_does_not_show_mailto_prefix() -> None:
    markdown = Markdown("Contact user@example.com for help", 0, 0, _test_markdown_theme())

    plain = " ".join(_strip_ansi(line) for line in markdown.render(80))

    assert "user@example.com" in plain
    assert "mailto:" not in plain


def test_markdown_autolinked_email_minimal_case_does_not_show_mailto_prefix_like_ts() -> None:
    markdown = Markdown("user@example.com", 0, 0, _test_markdown_theme())

    plain = " ".join(_strip_ansi(line) for line in markdown.render(80))

    assert "user@example.com" in plain
    assert "mailto:" not in plain


def test_markdown_explicit_mailto_link_with_different_text_shows_url() -> None:
    markdown = Markdown("[Email me](mailto:test@example.com)", 0, 0, _test_markdown_theme())

    plain = " ".join(_strip_ansi(line) for line in markdown.render(80))

    assert "Email me" in plain
    assert "(mailto:test@example.com)" in plain


def test_markdown_does_not_duplicate_autolinked_email_or_bare_url() -> None:
    email = Markdown("Contact user@example.com for help", 0, 0, _test_markdown_theme())
    bare_url = Markdown("Visit https://example.com for more", 0, 0, _test_markdown_theme())

    email_plain = " ".join(_strip_ansi(line) for line in email.render(80))
    bare_url_plain = " ".join(_strip_ansi(line) for line in bare_url.render(80))

    assert "user@example.com" in email_plain
    assert "mailto:" not in email_plain
    assert bare_url_plain.count("https://example.com") == 1


def test_markdown_should_not_duplicate_url_for_autolinked_emails_like_ts() -> None:
    markdown = Markdown("Contact user@example.com for help", 0, 0, _test_markdown_theme())

    plain = " ".join(_strip_ansi(line) for line in markdown.render(80))

    assert "user@example.com" in plain
    assert "mailto:" not in plain


def test_markdown_should_not_duplicate_url_for_bare_urls_like_ts() -> None:
    markdown = Markdown("Visit https://example.com for more", 0, 0, _test_markdown_theme())

    plain = " ".join(_strip_ansi(line) for line in markdown.render(80))

    assert plain.count("https://example.com") == 1


def test_markdown_explicit_links_show_url_only_when_text_differs() -> None:
    explicit = Markdown("[click here](https://example.com)", 0, 0, _test_markdown_theme())
    explicit_mailto = Markdown("[Email me](mailto:test@example.com)", 0, 0, _test_markdown_theme())
    matching_mailto = Markdown("[user@example.com](mailto:user@example.com)", 0, 0, _test_markdown_theme())

    explicit_plain = " ".join(_strip_ansi(line) for line in explicit.render(80))
    explicit_mailto_plain = " ".join(_strip_ansi(line) for line in explicit_mailto.render(80))
    matching_plain = " ".join(_strip_ansi(line) for line in matching_mailto.render(80))

    assert "click here" in explicit_plain
    assert "(https://example.com)" in explicit_plain
    assert "Email me" in explicit_mailto_plain
    assert "(mailto:test@example.com)" in explicit_mailto_plain
    assert "(mailto:" not in matching_plain


def test_markdown_should_show_url_for_explicit_markdown_links_with_different_text_like_ts() -> None:
    markdown = Markdown("[click here](https://example.com)", 0, 0, _test_markdown_theme())

    plain = " ".join(_strip_ansi(line) for line in markdown.render(80))

    assert "click here" in plain
    assert "(https://example.com)" in plain


def test_markdown_should_show_url_for_explicit_mailto_links_with_different_text_like_ts() -> None:
    markdown = Markdown("[Email me](mailto:test@example.com)", 0, 0, _test_markdown_theme())

    plain = " ".join(_strip_ansi(line) for line in markdown.render(80))

    assert "Email me" in plain
    assert "(mailto:test@example.com)" in plain


def test_markdown_explicit_url_with_different_text_shows_parenthesized_url_like_ts() -> None:
    markdown = Markdown("[click here](https://example.com)", 0, 0, _test_markdown_theme())

    plain = " ".join(_strip_ansi(line) for line in markdown.render(80))

    assert "click here" in plain
    assert "(https://example.com)" in plain


def test_markdown_explicit_mailto_with_different_text_shows_parenthesized_url_like_ts() -> None:
    markdown = Markdown("[Email me](mailto:test@example.com)", 0, 0, _test_markdown_theme())

    plain = " ".join(_strip_ansi(line) for line in markdown.render(80))

    assert "Email me" in plain
    assert "(mailto:test@example.com)" in plain


def test_markdown_matching_url_text_does_not_append_parenthesized_url_like_ts() -> None:
    markdown = Markdown("[https://example.com](https://example.com)", 0, 0, _test_markdown_theme())

    plain = " ".join(_strip_ansi(line) for line in markdown.render(80))

    assert "https://example.com" in plain
    assert plain.count("https://example.com") == 1
    assert "(https://example.com)" not in plain


def test_markdown_matching_mailto_text_does_not_append_parenthesized_url_like_ts() -> None:
    markdown = Markdown("[user@example.com](mailto:user@example.com)", 0, 0, _test_markdown_theme())

    plain = " ".join(_strip_ansi(line) for line in markdown.render(80))

    assert "user@example.com" in plain
    assert "(mailto:" not in plain


def test_markdown_normalizes_paragraph_and_code_block_spacing() -> None:
    cases = [
        "hello this is text\n```\ncode block\n```\nmore text",
        "hello this is text\n\n```\ncode block\n```\n\nmore text",
    ]
    expected = ["hello this is text", "", "```", "  code block", "```", "", "more text"]

    for text in cases:
        markdown = Markdown(text, 0, 0, _test_markdown_theme())
        plain = [line.rstrip() for line in markdown.render(80)]
        assert plain == expected


def test_markdown_code_block_last_block_has_no_trailing_blank_line() -> None:
    cases = [
        "```js\nconst hello = 'world';\n```",
        "hello world\n\n```js\nconst hello = 'world';\n```",
    ]

    for text in cases:
        markdown = Markdown(text, 0, 0, _test_markdown_theme())
        plain = [line.rstrip() for line in markdown.render(80)]
        assert plain[-1] != ""


def test_markdown_should_have_only_one_blank_line_between_code_block_and_following_paragraph_like_ts() -> None:
    markdown = Markdown(
        'hello world\n\n```js\nconst hello = "world";\n```\n\nagain, hello world',
        0,
        0,
        _test_markdown_theme(),
    )
    plain = [_strip_ansi(line).rstrip() for line in markdown.render(80)]
    closing_index = plain.index("```")
    after_closing = plain[closing_index + 1 :]

    assert next(index for index, line in enumerate(after_closing) if line != "") == 1


def test_markdown_divider_has_single_blank_line_before_following_paragraph() -> None:
    markdown = Markdown(
        "hello world\n\n---\n\nagain, hello world",
        0,
        0,
        _test_markdown_theme(),
    )
    plain = [line.rstrip() for line in markdown.render(80)]
    divider_index = next(index for index, line in enumerate(plain) if "─" in line)
    after_divider = plain[divider_index + 1 :]
    assert next(index for index, line in enumerate(after_divider) if line != "") == 1


def test_markdown_should_have_only_one_blank_line_between_divider_and_following_paragraph_like_ts() -> None:
    markdown = Markdown(
        "hello world\n\n---\n\nagain, hello world",
        0,
        0,
        _test_markdown_theme(),
    )
    plain = [line.rstrip() for line in markdown.render(80)]
    divider_index = next(index for index, line in enumerate(plain) if "─" in line)
    after_divider = plain[divider_index + 1 :]

    assert next(index for index, line in enumerate(after_divider) if line != "") == 1


def test_markdown_heading_has_single_blank_line_before_following_paragraph() -> None:
    markdown = Markdown(
        "# Hello\n\nThis is a paragraph",
        0,
        0,
        _test_markdown_theme(),
    )
    plain = [line.rstrip() for line in markdown.render(80)]
    heading_index = next(index for index, line in enumerate(plain) if "Hello" in line)
    after_heading = plain[heading_index + 1 :]
    assert next(index for index, line in enumerate(after_heading) if line != "") == 1


def test_markdown_should_have_only_one_blank_line_between_heading_and_following_paragraph_like_ts() -> None:
    markdown = Markdown(
        "# Hello\n\nThis is a paragraph",
        0,
        0,
        _test_markdown_theme(),
    )
    plain = [line.rstrip() for line in markdown.render(80)]
    heading_index = next(index for index, line in enumerate(plain) if "Hello" in line)
    after_heading = plain[heading_index + 1 :]

    assert next(index for index, line in enumerate(after_heading) if line != "") == 1


def test_markdown_blockquote_has_single_blank_line_before_following_paragraph() -> None:
    markdown = Markdown(
        "hello world\n\n> This is a quote\n\nagain, hello world",
        0,
        0,
        _test_markdown_theme(),
    )
    plain = [line.rstrip() for line in markdown.render(80)]
    quote_index = next(index for index, line in enumerate(plain) if "This is a quote" in line)
    after_quote = plain[quote_index + 1 :]
    assert next(index for index, line in enumerate(after_quote) if line != "") == 1


def test_markdown_should_have_only_one_blank_line_between_blockquote_and_following_paragraph_like_ts() -> None:
    markdown = Markdown(
        "hello world\n\n> This is a quote\n\nagain, hello world",
        0,
        0,
        _test_markdown_theme(),
    )
    plain = [line.rstrip() for line in markdown.render(80)]
    quote_index = next(index for index, line in enumerate(plain) if "This is a quote" in line)
    after_quote = plain[quote_index + 1 :]

    assert next(index for index, line in enumerate(after_quote) if line != "") == 1


def test_markdown_terminal_blocks_do_not_add_trailing_blank_line() -> None:
    cases = [
        ("---", _test_markdown_theme()),
        ("# Hello", _test_markdown_theme()),
        ("> This is a quote", _test_markdown_theme()),
    ]

    for text, theme in cases:
        markdown = Markdown(text, 0, 0, theme)
        plain = [line.rstrip() for line in markdown.render(80)]
        assert plain[-1] != ""


def test_markdown_prestyled_text_does_not_leak_styles_into_following_tui_line() -> None:
    terminal = _TerminalStub(columns=80, rows=6)
    tui = TUI(terminal)

    class _MarkdownWithInput:
        def __init__(self, markdown: Markdown) -> None:
            self.markdown = markdown
            self.markdown_line_count = 0

        def render(self, width: int) -> list[str]:
            lines = self.markdown.render(width)
            self.markdown_line_count = len(lines)
            return [*lines, "INPUT"]

        def invalidate(self) -> None:
            self.markdown.invalidate()

    markdown = Markdown(
        "This is thinking with `inline code`",
        1,
        0,
        _styled_markdown_theme(),
        {
            "color": lambda text: _ansi("90", text),
            "italic": True,
        },
    )

    component = _MarkdownWithInput(markdown)
    tui.add_child(component)
    tui.start()

    input_row = component.markdown_line_count
    assert input_row > 0
    assert "INPUT" in _strip_ansi(tui.previous_lines[input_row])
    assert "\x1b[3m" not in tui.previous_lines[input_row]
    assert "\x1b[90m" not in tui.previous_lines[input_row]


def test_markdown_rendered_in_tui_does_not_leak_italic_style_into_following_line() -> None:
    terminal = _TerminalStub(columns=80, rows=6)
    tui = TUI(terminal)

    class _MarkdownWithInput:
        def __init__(self, markdown: Markdown) -> None:
            self.markdown = markdown
            self.markdown_line_count = 0

        def render(self, width: int) -> list[str]:
            lines = self.markdown.render(width)
            self.markdown_line_count = len(lines)
            return [*lines, "INPUT"]

        def invalidate(self) -> None:
            self.markdown.invalidate()

    markdown = Markdown(
        "This is thinking with `inline code`",
        1,
        0,
        _styled_markdown_theme(),
        {
            "color": lambda text: _ansi("90", text),
            "italic": True,
        },
    )

    component = _MarkdownWithInput(markdown)
    tui.add_child(component)
    tui.start()

    input_row = component.markdown_line_count
    assert input_row > 0
    assert "\x1b[3m" not in tui.previous_lines[input_row]


def test_markdown_should_not_leak_styles_into_following_lines_when_rendered_in_tui_like_ts() -> None:
    terminal = _TerminalStub(columns=80, rows=6)
    tui = TUI(terminal)

    class _MarkdownWithInput:
        def __init__(self, markdown: Markdown) -> None:
            self.markdown = markdown
            self.markdown_line_count = 0

        def render(self, width: int) -> list[str]:
            lines = self.markdown.render(width)
            self.markdown_line_count = len(lines)
            return [*lines, "INPUT"]

        def invalidate(self) -> None:
            self.markdown.invalidate()

    markdown = Markdown(
        "This is thinking with `inline code`",
        1,
        0,
        _styled_markdown_theme(),
        {
            "color": lambda text: _ansi("90", text),
            "italic": True,
        },
    )

    component = _MarkdownWithInput(markdown)
    tui.add_child(component)
    tui.start()

    input_row = component.markdown_line_count
    assert input_row > 0
    assert "INPUT" in _strip_ansi(tui.previous_lines[input_row])
    assert "\x1b[3m" not in tui.previous_lines[input_row]
    assert "\x1b[90m" not in tui.previous_lines[input_row]


def test_image_renders_fallback_when_terminal_lacks_image_support(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.delenv("TERM_PROGRAM", raising=False)
    monkeypatch.delenv("KITTY_WINDOW_ID", raising=False)
    monkeypatch.delenv("ITERM_SESSION_ID", raising=False)
    monkeypatch.delenv("WEZTERM_PANE", raising=False)
    reset_capabilities_cache()
    image = Image(
        "Zm9v",
        "image/png",
        ImageTheme(fallback_color=lambda text: text),
        ImageOptions(filename="cat.png"),
        ImageDimensions(width_px=10, height_px=20),
    )
    [line] = image.render(80)
    reset_capabilities_cache()
    assert line == "[Image: cat.png [image/png] 10x20]"


def test_image_renders_kitty_sequence_when_supported(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setenv("TERM_PROGRAM", "kitty")
    reset_capabilities_cache()
    image = Image(
        "Zm9v",
        "image/png",
        ImageTheme(fallback_color=lambda text: text),
        ImageOptions(max_width_cells=10),
        ImageDimensions(width_px=10, height_px=20),
    )
    lines = image.render(80)
    reset_capabilities_cache()
    assert lines
    assert "\x1b_G" in lines[-1]


def test_settings_list_renders_items_description_and_hint() -> None:
    settings = SettingsList(
        [
            SettingItem(id="theme", label="Theme", current_value="light", description="UI theme", values=["light", "dark"]),
            SettingItem(id="model", label="Model", current_value="gpt-5"),
        ],
        5,
        _test_settings_theme(),
        lambda _id, _value: None,
        lambda: None,
    )
    lines = settings.render(60)
    assert any("Theme" in line and "light" in line for line in lines)
    assert any("UI theme" in line for line in lines)
    assert any("Enter/Space to change" in line for line in lines)


def test_settings_list_cycles_values_and_calls_on_change() -> None:
    changes: list[tuple[str, str]] = []
    settings = SettingsList(
        [SettingItem(id="theme", label="Theme", current_value="light", values=["light", "dark"])],
        5,
        _test_settings_theme(),
        lambda setting_id, value: changes.append((setting_id, value)),
        lambda: None,
    )
    settings.handle_input("\r")
    assert settings.items[0].current_value == "dark"
    assert changes == [("theme", "dark")]


def test_settings_list_search_filters_by_label() -> None:
    settings = SettingsList(
        [
            SettingItem(id="theme", label="Theme", current_value="light"),
            SettingItem(id="model", label="Model", current_value="gpt-5"),
        ],
        5,
        _test_settings_theme(),
        lambda _id, _value: None,
        lambda: None,
        {"enable_search": True},
    )
    settings.handle_input("mod")
    lines = settings.render(60)
    assert any("Model" in line for line in lines)
    assert not any("Theme" in line for line in lines if "No matching settings" not in line)


class _SubmenuComponent:
    def __init__(self, initial_value: str, done) -> None:  # noqa: ANN001
        self.initial_value = initial_value
        self.done = done
        self.inputs: list[str] = []
        self.invalidated = False

    def render(self, width: int) -> list[str]:  # noqa: ARG002
        return [f"submenu:{self.initial_value}"]

    def handle_input(self, data: str) -> None:
        self.inputs.append(data)
        if data == "\r":
            self.done("updated")
        elif data == "\x1b":
            self.done()

    def invalidate(self) -> None:
        self.invalidated = True


def test_settings_list_opens_submenu_and_applies_selected_value() -> None:
    changes: list[tuple[str, str]] = []
    submenu_instances: list[_SubmenuComponent] = []

    def open_submenu(current_value: str, done):  # noqa: ANN001
        submenu = _SubmenuComponent(current_value, done)
        submenu_instances.append(submenu)
        return submenu

    settings = SettingsList(
        [SettingItem(id="theme", label="Theme", current_value="light", submenu=open_submenu)],
        5,
        _test_settings_theme(),
        lambda setting_id, value: changes.append((setting_id, value)),
        lambda: None,
    )

    settings.handle_input("\r")
    assert submenu_instances
    assert settings.render(60) == ["submenu:light"]

    settings.handle_input("\r")

    assert settings.items[0].current_value == "updated"
    assert changes == [("theme", "updated")]
    assert any("Theme" in line and "updated" in line for line in settings.render(60))


def test_settings_list_submenu_cancel_restores_selection_without_change() -> None:
    changes: list[tuple[str, str]] = []

    settings = SettingsList(
        [
            SettingItem(id="theme", label="Theme", current_value="light"),
            SettingItem(
                id="model",
                label="Model",
                current_value="gpt-5",
                submenu=lambda current_value, done: _SubmenuComponent(current_value, done),
            ),
        ],
        5,
        _test_settings_theme(),
        lambda setting_id, value: changes.append((setting_id, value)),
        lambda: None,
    )

    settings.handle_input("\x1b[B")
    settings.handle_input("\r")
    assert settings.render(60) == ["submenu:gpt-5"]

    settings.handle_input("\x1b")

    lines = settings.render(60)
    assert changes == []
    assert any("→ Model" in line for line in lines)
    assert any("gpt-5" in line for line in lines)


def test_settings_list_invalidate_delegates_to_active_submenu() -> None:
    submenu = _SubmenuComponent("light", lambda selected_value=None: None)
    settings = SettingsList(
        [
            SettingItem(
                id="theme",
                label="Theme",
                current_value="light",
                submenu=lambda current_value, done: submenu,
            )
        ],
        5,
        _test_settings_theme(),
        lambda _id, _value: None,
        lambda: None,
    )

    settings.handle_input("\r")
    settings.invalidate()

    assert submenu.invalidated is True


@dataclass
class _TerminalStub:
    columns: int = 80
    rows: int = 24
    writes: list[str] = field(default_factory=list)

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
        self.writes.append(f"move:{lines}")

    def hide_cursor(self) -> None:
        self.writes.append("hide")

    def show_cursor(self) -> None:
        self.writes.append("show")

    def clear_line(self) -> None:
        return None

    def clear_from_cursor(self) -> None:
        return None

    def clear_screen(self) -> None:
        return None

    def set_title(self, title: str) -> None:
        self.writes.append(f"title:{title}")


class _StaticComponent:
    def __init__(self, lines: list[str]) -> None:
        self.lines = lines

    def render(self, width: int) -> list[str]:  # noqa: ARG002
        return self.lines

    def handle_input(self, data: str) -> None:  # noqa: ARG002
        return None

    def invalidate(self) -> None:
        return None


class _CountingComponent(_StaticComponent):
    def __init__(self, lines: list[str]) -> None:
        super().__init__(lines)
        self.render_calls = 0

    def render(self, width: int) -> list[str]:  # noqa: ARG002
        self.render_calls += 1
        return self.lines


class _FocusableComponent(_StaticComponent):
    def __init__(self, lines: list[str]) -> None:
        super().__init__(lines)
        self.focused = False
        self.inputs: list[str] = []

    def handle_input(self, data: str) -> None:
        self.inputs.append(data)


class _InvalidateTrackingComponent(_FocusableComponent):
    def __init__(self, lines: list[str]) -> None:
        super().__init__(lines)
        self.invalidated = 0

    def invalidate(self) -> None:
        self.invalidated += 1


class _KeyReleaseFocusableComponent(_FocusableComponent):
    wantsKeyRelease = True


def test_overlay_top_left_is_composited_into_rendered_lines() -> None:
    terminal = _TerminalStub()
    tui = TUI(terminal)
    tui.add_child(_StaticComponent(["base"]))
    tui.show_overlay(_StaticComponent(["TOP"]), OverlayOptions(anchor="top-left", width=10))
    lines = tui._composite_overlays(tui.render(terminal.columns), terminal.columns, terminal.rows)  # type: ignore[attr-defined]
    assert "TOP" in lines[0]


def test_overlay_top_left_anchor_starts_at_first_column() -> None:
    terminal = _TerminalStub(columns=80, rows=24)
    tui = TUI(terminal)
    tui.add_child(_StaticComponent([]))
    tui.show_overlay(_StaticComponent(["TOP-LEFT"]), OverlayOptions(anchor="top-left", width=10))
    tui.start()

    first_row = tui.previous_lines[0]
    assert "TOP-LEFT" in first_row
    assert _visible_index_of(first_row, "TOP-LEFT") == 0


def test_overlay_should_position_overlay_at_top_left_like_ts() -> None:
    """should position overlay at top-left"""
    terminal = _TerminalStub(columns=80, rows=24)
    tui = TUI(terminal)
    tui.add_child(_StaticComponent([]))
    tui.show_overlay(_StaticComponent(["TOP-LEFT"]), OverlayOptions(anchor="top-left", width=10))
    tui.start()

    first_row = _strip_ansi(tui.previous_lines[0])
    assert "TOP-LEFT" in first_row
    assert _visible_index_of(first_row, "TOP-LEFT") == 0


def test_center_overlay_renders_when_content_is_shorter_than_terminal_height() -> None:
    """should render overlay when content is shorter than terminal height"""
    terminal = _TerminalStub(columns=40, rows=24)
    tui = TUI(terminal)
    tui.add_child(_StaticComponent(["Line 1", "Line 2", "Line 3"]))
    tui.show_overlay(_StaticComponent(["OVERLAY_TOP", "OVERLAY_MID", "OVERLAY_BOT"]))

    tui.start()

    assert any("OVERLAY" in line for line in tui.previous_lines)


def test_overlay_bottom_right_is_composited_into_last_row_end() -> None:
    terminal = _TerminalStub(columns=20, rows=6)
    tui = TUI(terminal)
    tui.add_child(_StaticComponent(["base"]))
    tui.show_overlay(_StaticComponent(["BTM-RIGHT"]), OverlayOptions(anchor="bottom-right", width=10))
    lines = tui._composite_overlays(tui.render(terminal.columns), terminal.columns, terminal.rows)  # type: ignore[attr-defined]
    last_row = _strip_ansi(lines[-1])

    assert "BTM-RIGHT" in last_row
    assert _visible_index_of(last_row, "BTM-RIGHT") == 10


def test_overlay_bottom_right_anchor_hugs_last_row_and_terminal_end() -> None:
    terminal = _TerminalStub(columns=80, rows=24)
    tui = TUI(terminal)
    tui.add_child(_StaticComponent([]))
    tui.show_overlay(_StaticComponent(["BTM-RIGHT"]), OverlayOptions(anchor="bottom-right", width=10))
    tui.start()

    last_row = tui.previous_lines[23]
    assert "BTM-RIGHT" in last_row
    assert _visible_index_of(last_row, "BTM-RIGHT") == 70


def test_overlay_should_position_overlay_at_bottom_right_like_ts() -> None:
    """should position overlay at bottom-right"""
    terminal = _TerminalStub(columns=80, rows=24)
    tui = TUI(terminal)
    tui.add_child(_StaticComponent([]))
    tui.show_overlay(_StaticComponent(["BTM-RIGHT"]), OverlayOptions(anchor="bottom-right", width=10))
    tui.start()

    last_row = _strip_ansi(tui.previous_lines[23])
    assert "BTM-RIGHT" in last_row
    assert _visible_index_of(last_row, "BTM-RIGHT") == 70


def test_overlays_at_different_positions_render_without_interference() -> None:
    terminal = _TerminalStub(columns=80, rows=24)
    tui = TUI(terminal)
    tui.add_child(_StaticComponent([]))
    tui.show_overlay(_StaticComponent(["TOP-LEFT"]), OverlayOptions(anchor="top-left", width=15))
    tui.show_overlay(_StaticComponent(["BTM-RIGHT"]), OverlayOptions(anchor="bottom-right", width=15))
    tui.start()

    assert "TOP-LEFT" in tui.previous_lines[0]
    assert "BTM-RIGHT" in tui.previous_lines[23]


def test_overlay_margin_number_offsets_top_left_anchor() -> None:
    terminal = _TerminalStub(columns=20, rows=10)
    tui = TUI(terminal)
    tui.add_child(_StaticComponent(["base"]))
    tui.show_overlay(_StaticComponent(["MARGIN"]), OverlayOptions(anchor="top-left", width=10, margin=5))
    lines = tui._composite_overlays(tui.render(terminal.columns), terminal.columns, terminal.rows)  # type: ignore[attr-defined]
    row = _strip_ansi(lines[5])

    assert "MARGIN" in row
    assert _visible_index_of(row, "MARGIN") == 5


def test_overlay_should_respect_margin_as_number_like_ts() -> None:
    """should respect margin as number"""
    terminal = _TerminalStub(columns=20, rows=10)
    tui = TUI(terminal)
    tui.add_child(_StaticComponent(["base"]))
    tui.show_overlay(_StaticComponent(["MARGIN"]), OverlayOptions(anchor="top-left", width=10, margin=5))
    lines = tui._composite_overlays(tui.render(terminal.columns), terminal.columns, terminal.rows)  # type: ignore[attr-defined]

    assert "MARGIN" not in _strip_ansi(lines[0])
    assert "MARGIN" not in _strip_ansi(lines[4])
    assert "MARGIN" in _strip_ansi(lines[5])
    assert _visible_index_of(_strip_ansi(lines[5]), "MARGIN") == 5


def test_overlay_margin_object_offsets_top_left_anchor() -> None:
    terminal = _TerminalStub(columns=80, rows=24)
    tui = TUI(terminal)
    tui.add_child(_StaticComponent([]))
    tui.show_overlay(
        _StaticComponent(["MARGIN"]),
        OverlayOptions(anchor="top-left", width=10, margin=OverlayMargin(top=2, left=3, right=0, bottom=0)),
    )
    tui.start()

    row = _strip_ansi(tui.previous_lines[2])
    assert "MARGIN" in row
    assert _visible_index_of(row, "MARGIN") == 3


def test_overlay_should_respect_margin_object_like_ts() -> None:
    """should respect margin object"""
    terminal = _TerminalStub(columns=80, rows=24)
    tui = TUI(terminal)
    tui.add_child(_StaticComponent([]))
    tui.show_overlay(
        _StaticComponent(["MARGIN"]),
        OverlayOptions(anchor="top-left", width=10, margin=OverlayMargin(top=2, left=3, right=0, bottom=0)),
    )
    tui.start()

    row = _strip_ansi(tui.previous_lines[2])
    assert "MARGIN" in row
    assert _visible_index_of(row, "MARGIN") == 3


def test_overlay_offset_applies_from_anchor_position() -> None:
    terminal = _TerminalStub(columns=20, rows=10)
    tui = TUI(terminal)
    tui.add_child(_StaticComponent(["base"]))
    tui.show_overlay(_StaticComponent(["OFFSET"]), OverlayOptions(anchor="top-left", width=10, offset_x=10, offset_y=5))
    lines = tui._composite_overlays(tui.render(terminal.columns), terminal.columns, terminal.rows)  # type: ignore[attr-defined]
    row = _strip_ansi(lines[5])

    assert "OFFSET" in row
    assert _visible_index_of(row, "OFFSET") == 10


def test_overlay_should_apply_offset_x_and_offset_y_from_anchor_position_like_ts() -> None:
    """should apply offsetX and offsetY from anchor position"""
    terminal = _TerminalStub(columns=20, rows=10)
    tui = TUI(terminal)
    tui.add_child(_StaticComponent(["base"]))
    tui.show_overlay(_StaticComponent(["OFFSET"]), OverlayOptions(anchor="top-left", width=10, offset_x=10, offset_y=5))
    lines = tui._composite_overlays(tui.render(terminal.columns), terminal.columns, terminal.rows)  # type: ignore[attr-defined]
    row = _strip_ansi(lines[5])

    assert "OFFSET" in row
    assert _visible_index_of(row, "OFFSET") == 10


def test_overlay_should_apply_offset_x_and_offset_y_from_anchor_position_exact_title_like_ts() -> None:
    terminal = _TerminalStub(columns=20, rows=10)
    tui = TUI(terminal)
    tui.add_child(_StaticComponent(["base"]))
    tui.show_overlay(_StaticComponent(["OFFSET"]), OverlayOptions(anchor="top-left", width=10, offset_x=10, offset_y=5))
    lines = tui._composite_overlays(tui.render(terminal.columns), terminal.columns, terminal.rows)  # type: ignore[attr-defined]
    row = _strip_ansi(lines[5])

    assert "OFFSET" in row
    assert _visible_index_of(row, "OFFSET") == 10


def test_overlay_should_apply_offset_x_and_offset_y_from_anchor_position_with_offset_x_offset_y_title_like_ts() -> None:
    terminal = _TerminalStub(columns=20, rows=10)
    tui = TUI(terminal)
    tui.add_child(_StaticComponent(["base"]))
    tui.show_overlay(_StaticComponent(["OFFSET"]), OverlayOptions(anchor="top-left", width=10, offset_x=10, offset_y=5))
    lines = tui._composite_overlays(tui.render(terminal.columns), terminal.columns, terminal.rows)  # type: ignore[attr-defined]
    row = _strip_ansi(lines[5])

    assert "OFFSET" in row
    assert _visible_index_of(row, "OFFSET") == 10


def test_overlay_should_apply_offsetx_and_offsety_from_anchor_position_exact_title_like_ts() -> None:
    terminal = _TerminalStub(columns=20, rows=10)
    tui = TUI(terminal)
    tui.add_child(_StaticComponent(["base"]))
    tui.show_overlay(_StaticComponent(["OFFSET"]), OverlayOptions(anchor="top-left", width=10, offset_x=10, offset_y=5))
    lines = tui._composite_overlays(tui.render(terminal.columns), terminal.columns, terminal.rows)  # type: ignore[attr-defined]
    row = _strip_ansi(lines[5])

    assert "OFFSET" in row
    assert _visible_index_of(row, "OFFSET") == 10


def test_overlay_percentage_positioning_centers_overlay() -> None:
    terminal = _TerminalStub(columns=20, rows=10)
    tui = TUI(terminal)
    tui.add_child(_StaticComponent(["base"]))
    tui.show_overlay(_StaticComponent(["PCT"]), OverlayOptions(width=10, row="50%", col="50%"))
    lines = tui._composite_overlays(tui.render(terminal.columns), terminal.columns, terminal.rows)  # type: ignore[attr-defined]
    row = _strip_ansi(lines[4])

    assert "PCT" in row
    assert _visible_index_of(row, "PCT") == 5


def test_overlay_should_position_with_row_percent_and_col_percent_like_ts() -> None:
    """should position with rowPercent and colPercent"""
    terminal = _TerminalStub(columns=80, rows=24)
    tui = TUI(terminal)
    tui.add_child(_StaticComponent([]))
    tui.show_overlay(_StaticComponent(["PCT"]), OverlayOptions(width=10, row="50%", col="50%"))
    tui.start()

    found_row = next((i for i, line in enumerate(tui.previous_lines) if "PCT" in line), -1)
    assert 10 <= found_row <= 13


def test_overlay_should_position_with_row_percent_and_col_percent_exact_title_like_ts() -> None:
    terminal = _TerminalStub(columns=80, rows=24)
    tui = TUI(terminal)
    tui.add_child(_StaticComponent([]))
    tui.show_overlay(_StaticComponent(["PCT"]), OverlayOptions(width=10, row="50%", col="50%"))
    tui.start()

    found_row = next((i for i, line in enumerate(tui.previous_lines) if "PCT" in line), -1)
    assert 10 <= found_row <= 13


def test_overlay_should_position_with_row_percent_and_col_percent_with_row_percent_col_percent_title_like_ts() -> None:
    terminal = _TerminalStub(columns=80, rows=24)
    tui = TUI(terminal)
    tui.add_child(_StaticComponent([]))
    tui.show_overlay(_StaticComponent(["PCT"]), OverlayOptions(width=10, row="50%", col="50%"))
    tui.start()

    found_row = next((i for i, line in enumerate(tui.previous_lines) if "PCT" in line), -1)
    assert 10 <= found_row <= 13


def test_overlay_should_position_with_rowpercent_and_colpercent_exact_title_like_ts() -> None:
    terminal = _TerminalStub(columns=80, rows=24)
    tui = TUI(terminal)
    tui.add_child(_StaticComponent([]))
    tui.show_overlay(_StaticComponent(["PCT"]), OverlayOptions(width=10, row="50%", col="50%"))
    tui.start()

    found_row = next((i for i, line in enumerate(tui.previous_lines) if "PCT" in line), -1)
    assert 10 <= found_row <= 13


def test_overlay_row_percent_zero_positions_overlay_at_top() -> None:
    terminal = _TerminalStub(columns=80, rows=24)
    tui = TUI(terminal)
    tui.add_child(_StaticComponent([]))
    tui.show_overlay(_StaticComponent(["TOP"]), OverlayOptions(width=10, row="0%"))
    tui.start()

    assert "TOP" in tui.previous_lines[0]


def test_overlay_row_percent_zero_should_position_at_top_like_ts() -> None:
    terminal = _TerminalStub(columns=80, rows=24)
    tui = TUI(terminal)
    tui.add_child(_StaticComponent([]))
    tui.show_overlay(_StaticComponent(["TOP"]), OverlayOptions(width=10, row="0%"))
    tui.start()

    assert "TOP" in tui.previous_lines[0]


def test_overlay_row_percent_0_should_position_at_top_like_ts() -> None:
    """rowPercent 0 should position at top"""
    terminal = _TerminalStub(columns=80, rows=24)
    tui = TUI(terminal)
    tui.add_child(_StaticComponent([]))
    tui.show_overlay(_StaticComponent(["TOP"]), OverlayOptions(width=10, row="0%"))
    tui.start()

    assert "TOP" in tui.previous_lines[0]


def test_overlay_row_percent_0_should_position_at_top_with_row_percent_title_like_ts() -> None:
    terminal = _TerminalStub(columns=80, rows=24)
    tui = TUI(terminal)
    tui.add_child(_StaticComponent([]))
    tui.show_overlay(_StaticComponent(["TOP"]), OverlayOptions(width=10, row="0%"))
    tui.start()

    assert "TOP" in tui.previous_lines[0]


def test_overlay_rowpercent_0_should_position_at_top_exact_title_like_ts() -> None:
    terminal = _TerminalStub(columns=80, rows=24)
    tui = TUI(terminal)
    tui.add_child(_StaticComponent([]))
    tui.show_overlay(_StaticComponent(["TOP"]), OverlayOptions(width=10, row="0%"))
    tui.start()

    assert "TOP" in tui.previous_lines[0]


def test_overlay_row_percent_hundred_positions_overlay_at_bottom() -> None:
    terminal = _TerminalStub(columns=80, rows=24)
    tui = TUI(terminal)
    tui.add_child(_StaticComponent([]))
    tui.show_overlay(_StaticComponent(["BOTTOM"]), OverlayOptions(width=10, row="100%"))
    tui.start()

    assert "BOTTOM" in tui.previous_lines[23]


def test_overlay_row_percent_hundred_should_position_at_bottom_like_ts() -> None:
    terminal = _TerminalStub(columns=80, rows=24)
    tui = TUI(terminal)
    tui.add_child(_StaticComponent([]))
    tui.show_overlay(_StaticComponent(["BOTTOM"]), OverlayOptions(width=10, row="100%"))
    tui.start()

    assert "BOTTOM" in tui.previous_lines[23]


def test_overlay_row_percent_100_should_position_at_bottom_like_ts() -> None:
    """rowPercent 100 should position at bottom"""
    terminal = _TerminalStub(columns=80, rows=24)
    tui = TUI(terminal)
    tui.add_child(_StaticComponent([]))
    tui.show_overlay(_StaticComponent(["BOTTOM"]), OverlayOptions(width=10, row="100%"))
    tui.start()

    assert "BOTTOM" in tui.previous_lines[23]


def test_overlay_row_percent_100_should_position_at_bottom_with_row_percent_title_like_ts() -> None:
    terminal = _TerminalStub(columns=80, rows=24)
    tui = TUI(terminal)
    tui.add_child(_StaticComponent([]))
    tui.show_overlay(_StaticComponent(["BOTTOM"]), OverlayOptions(width=10, row="100%"))
    tui.start()

    assert "BOTTOM" in tui.previous_lines[23]


def test_overlay_rowpercent_100_should_position_at_bottom_exact_title_like_ts() -> None:
    terminal = _TerminalStub(columns=80, rows=24)
    tui = TUI(terminal)
    tui.add_child(_StaticComponent([]))
    tui.show_overlay(_StaticComponent(["BOTTOM"]), OverlayOptions(width=10, row="100%"))
    tui.start()

    assert "BOTTOM" in tui.previous_lines[23]


def test_overlay_max_height_truncates_rendered_lines() -> None:
    terminal = _TerminalStub(columns=20, rows=10)
    tui = TUI(terminal)
    tui.add_child(_StaticComponent(["base"]))
    tui.show_overlay(
        _StaticComponent(["L1", "L2", "L3", "L4", "L5", "L6"]),
        OverlayOptions(max_height="50%"),
    )
    lines = tui._composite_overlays(tui.render(terminal.columns), terminal.columns, terminal.rows)  # type: ignore[attr-defined]
    joined = "\n".join(lines)

    assert "L1" in joined
    assert "L5" in joined
    assert "L6" not in joined


def test_overlay_should_truncate_overlay_to_max_height_percent_like_ts() -> None:
    """should truncate overlay to maxHeightPercent"""
    terminal = _TerminalStub(columns=20, rows=10)
    tui = TUI(terminal)
    tui.add_child(_StaticComponent(["base"]))
    tui.show_overlay(
        _StaticComponent(["L1", "L2", "L3", "L4", "L5", "L6"]),
        OverlayOptions(max_height="50%"),
    )
    lines = tui._composite_overlays(tui.render(terminal.columns), terminal.columns, terminal.rows)  # type: ignore[attr-defined]
    joined = "\n".join(lines)

    assert "L1" in joined
    assert "L5" in joined
    assert "L6" not in joined


def test_overlay_should_truncate_overlay_to_max_height_percent_exact_title_like_ts() -> None:
    terminal = _TerminalStub(columns=20, rows=10)
    tui = TUI(terminal)
    tui.add_child(_StaticComponent(["base"]))
    tui.show_overlay(
        _StaticComponent(["L1", "L2", "L3", "L4", "L5", "L6"]),
        OverlayOptions(max_height="50%"),
    )
    lines = tui._composite_overlays(tui.render(terminal.columns), terminal.columns, terminal.rows)  # type: ignore[attr-defined]
    joined = "\n".join(lines)

    assert "L1" in joined
    assert "L5" in joined
    assert "L6" not in joined


def test_overlay_should_truncate_overlay_to_max_height_percent_with_max_height_percent_title_like_ts() -> None:
    terminal = _TerminalStub(columns=20, rows=10)
    tui = TUI(terminal)
    tui.add_child(_StaticComponent(["base"]))
    tui.show_overlay(
        _StaticComponent(["L1", "L2", "L3", "L4", "L5", "L6"]),
        OverlayOptions(max_height="50%"),
    )
    lines = tui._composite_overlays(tui.render(terminal.columns), terminal.columns, terminal.rows)  # type: ignore[attr-defined]
    joined = "\n".join(lines)

    assert "L1" in joined
    assert "L5" in joined
    assert "L6" not in joined


def test_overlay_should_truncate_overlay_to_maxheightpercent_exact_title_like_ts() -> None:
    terminal = _TerminalStub(columns=20, rows=10)
    tui = TUI(terminal)
    tui.add_child(_StaticComponent(["base"]))
    tui.show_overlay(
        _StaticComponent(["L1", "L2", "L3", "L4", "L5", "L6"]),
        OverlayOptions(max_height="50%"),
    )
    lines = tui._composite_overlays(tui.render(terminal.columns), terminal.columns, terminal.rows)  # type: ignore[attr-defined]
    joined = "\n".join(lines)

    assert "L1" in joined
    assert "L5" in joined
    assert "L6" not in joined


def test_overlay_numeric_max_height_keeps_only_first_three_lines() -> None:
    terminal = _TerminalStub(columns=80, rows=24)
    tui = TUI(terminal)
    tui.add_child(_StaticComponent([]))
    tui.show_overlay(
        _StaticComponent(["Line 1", "Line 2", "Line 3", "Line 4", "Line 5"]),
        OverlayOptions(max_height=3),
    )
    tui.start()

    joined = "\n".join(tui.previous_lines)
    assert "Line 1" in joined
    assert "Line 2" in joined
    assert "Line 3" in joined
    assert "Line 4" not in joined
    assert "Line 5" not in joined


def test_overlay_should_truncate_overlay_to_max_height_like_ts() -> None:
    """should truncate overlay to maxHeight"""
    terminal = _TerminalStub(columns=80, rows=24)
    tui = TUI(terminal)
    tui.add_child(_StaticComponent([]))
    tui.show_overlay(
        _StaticComponent(["Line 1", "Line 2", "Line 3", "Line 4", "Line 5"]),
        OverlayOptions(max_height=3),
    )
    tui.start()

    joined = "\n".join(tui.previous_lines)
    assert "Line 1" in joined
    assert "Line 2" in joined
    assert "Line 3" in joined
    assert "Line 4" not in joined
    assert "Line 5" not in joined


def test_overlay_should_truncate_overlay_to_max_height_exact_title_like_ts() -> None:
    terminal = _TerminalStub(columns=80, rows=24)
    tui = TUI(terminal)
    tui.add_child(_StaticComponent([]))
    tui.show_overlay(
        _StaticComponent(["Line 1", "Line 2", "Line 3", "Line 4", "Line 5"]),
        OverlayOptions(max_height=3),
    )
    tui.start()

    joined = "\n".join(tui.previous_lines)
    assert "Line 1" in joined
    assert "Line 2" in joined
    assert "Line 3" in joined
    assert "Line 4" not in joined
    assert "Line 5" not in joined


def test_overlay_should_truncate_overlay_to_max_height_with_max_height_title_like_ts() -> None:
    terminal = _TerminalStub(columns=80, rows=24)
    tui = TUI(terminal)
    tui.add_child(_StaticComponent([]))
    tui.show_overlay(
        _StaticComponent(["Line 1", "Line 2", "Line 3", "Line 4", "Line 5"]),
        OverlayOptions(max_height=3),
    )
    tui.start()

    joined = "\n".join(tui.previous_lines)
    assert "Line 1" in joined
    assert "Line 2" in joined
    assert "Line 3" in joined
    assert "Line 4" not in joined
    assert "Line 5" not in joined


def test_overlay_should_truncate_overlay_to_maxheight_exact_title_like_ts() -> None:
    terminal = _TerminalStub(columns=80, rows=24)
    tui = TUI(terminal)
    tui.add_child(_StaticComponent([]))
    tui.show_overlay(
        _StaticComponent(["Line 1", "Line 2", "Line 3", "Line 4", "Line 5"]),
        OverlayOptions(max_height=3),
    )
    tui.start()

    joined = "\n".join(tui.previous_lines)
    assert "Line 1" in joined
    assert "Line 2" in joined
    assert "Line 3" in joined
    assert "Line 4" not in joined
    assert "Line 5" not in joined


def test_overlay_negative_margins_are_clamped_to_zero() -> None:
    terminal = _TerminalStub(columns=20, rows=10)
    tui = TUI(terminal)
    tui.add_child(_StaticComponent(["base"]))
    tui.show_overlay(
        _StaticComponent(["NEG-MARGIN"]),
        OverlayOptions(
            anchor="top-left",
            width=12,
            margin=OverlayMargin(top=-5, left=-10, right=0, bottom=0),
        ),
    )
    lines = tui._composite_overlays(tui.render(terminal.columns), terminal.columns, terminal.rows)  # type: ignore[attr-defined]
    row = _strip_ansi(lines[0]).replace("\x1b]8;;\x07", "")

    assert "NEG-MARGIN" in row
    assert row.startswith("NEG-MARGIN")


def test_overlay_should_clamp_negative_margins_to_zero_like_ts() -> None:
    """should clamp negative margins to zero"""
    terminal = _TerminalStub(columns=20, rows=10)
    tui = TUI(terminal)
    tui.add_child(_StaticComponent(["base"]))
    tui.show_overlay(
        _StaticComponent(["NEG-MARGIN"]),
        OverlayOptions(
            anchor="top-left",
            width=12,
            margin=OverlayMargin(top=-5, left=-10, right=0, bottom=0),
        ),
    )
    lines = tui._composite_overlays(tui.render(terminal.columns), terminal.columns, terminal.rows)  # type: ignore[attr-defined]
    row = _strip_ansi(lines[0]).replace("\x1b]8;;\x07", "")

    assert "NEG-MARGIN" in row
    assert row.startswith("NEG-MARGIN")


def test_overlay_absolute_row_and_col_override_anchor() -> None:
    terminal = _TerminalStub(columns=20, rows=10)
    tui = TUI(terminal)
    tui.add_child(_StaticComponent(["base"]))
    tui.show_overlay(
        _StaticComponent(["ABSOLUTE"]),
        OverlayOptions(anchor="bottom-right", row=3, col=5, width=10),
    )
    lines = tui._composite_overlays(tui.render(terminal.columns), terminal.columns, terminal.rows)  # type: ignore[attr-defined]
    row = _strip_ansi(lines[3])

    assert "ABSOLUTE" in row
    assert _visible_index_of(row, "ABSOLUTE") == 5


def test_overlay_row_and_col_should_override_anchor_like_ts() -> None:
    """row and col should override anchor"""
    terminal = _TerminalStub(columns=20, rows=10)
    tui = TUI(terminal)
    tui.add_child(_StaticComponent(["base"]))
    tui.show_overlay(
        _StaticComponent(["ABSOLUTE"]),
        OverlayOptions(anchor="bottom-right", row=3, col=5, width=10),
    )
    lines = tui._composite_overlays(tui.render(terminal.columns), terminal.columns, terminal.rows)  # type: ignore[attr-defined]
    row = _strip_ansi(lines[3])

    assert "ABSOLUTE" in row
    assert _visible_index_of(row, "ABSOLUTE") == 5


class _WidthRecordingOverlay(_StaticComponent):
    def __init__(self, lines: list[str]) -> None:
        super().__init__(lines)
        self.requested_width: int | None = None

    def render(self, width: int) -> list[str]:
        self.requested_width = width
        return super().render(width)


def test_overlay_width_percent_uses_terminal_width() -> None:
    terminal = _TerminalStub(columns=100, rows=24)
    tui = TUI(terminal)
    overlay = _WidthRecordingOverlay(["test"])
    tui.add_child(_StaticComponent(["base"]))

    tui.show_overlay(overlay, OverlayOptions(width="50%"))
    tui.start()

    assert overlay.requested_width == 50


def test_overlay_should_render_overlay_at_percentage_of_terminal_width_like_ts() -> None:
    """should render overlay at percentage of terminal width"""
    terminal = _TerminalStub(columns=100, rows=24)
    tui = TUI(terminal)
    overlay = _WidthRecordingOverlay(["test"])
    tui.add_child(_StaticComponent(["base"]))

    tui.show_overlay(overlay, OverlayOptions(width="50%"))
    tui.start()

    assert overlay.requested_width == 50


def test_overlay_min_width_overrides_small_percentage_width() -> None:
    terminal = _TerminalStub(columns=100, rows=24)
    tui = TUI(terminal)
    overlay = _WidthRecordingOverlay(["test"])
    tui.add_child(_StaticComponent(["base"]))

    tui.show_overlay(overlay, OverlayOptions(width="10%", min_width=30))
    tui.start()

    assert overlay.requested_width == 30


def test_overlay_should_respect_min_width_when_width_percent_results_in_smaller_width_like_ts() -> None:
    """should respect minWidth when widthPercent results in smaller width"""
    terminal = _TerminalStub(columns=100, rows=24)
    tui = TUI(terminal)
    overlay = _WidthRecordingOverlay(["test"])
    tui.add_child(_StaticComponent(["base"]))

    tui.show_overlay(overlay, OverlayOptions(width="10%", min_width=30))
    tui.start()

    assert overlay.requested_width == 30


def test_overlay_should_respect_min_width_when_width_percent_results_in_smaller_width_exact_title_like_ts() -> None:
    terminal = _TerminalStub(columns=100, rows=24)
    tui = TUI(terminal)
    overlay = _WidthRecordingOverlay(["test"])
    tui.add_child(_StaticComponent(["base"]))

    tui.show_overlay(overlay, OverlayOptions(width="10%", min_width=30))
    tui.start()

    assert overlay.requested_width == 30


def test_overlay_should_respect_minwidth_when_widthpercent_results_in_smaller_width_exact_title_like_ts() -> None:
    terminal = _TerminalStub(columns=100, rows=24)
    tui = TUI(terminal)
    overlay = _WidthRecordingOverlay(["test"])
    tui.add_child(_StaticComponent(["base"]))

    tui.show_overlay(overlay, OverlayOptions(width="10%", min_width=30))
    tui.start()

    assert overlay.requested_width == 30


def test_overlay_on_hyperlink_ansi_base_content_does_not_crash() -> None:
    terminal = _TerminalStub(columns=40, rows=10)
    tui = TUI(terminal)
    hyperlink = "\x1b]8;;file:///path/to/file.ts\x07file.ts\x1b]8;;\x07"
    tui.add_child(_StaticComponent([f"See {hyperlink} for details", f"See {hyperlink} for details"]))

    tui.show_overlay(_StaticComponent(["OVERLAY"]), OverlayOptions(anchor="center", width=12))
    tui.start()

    assert any("OVERLAY" in line for line in tui.previous_lines)


def test_overlay_should_handle_overlay_on_base_content_with_osc_sequences_like_ts() -> None:
    """should handle overlay on base content with OSC sequences"""
    terminal = _TerminalStub(columns=40, rows=10)
    tui = TUI(terminal)
    hyperlink = "\x1b]8;;file:///path/to/file.ts\x07file.ts\x1b]8;;\x07"
    tui.add_child(_StaticComponent([f"See {hyperlink} for details", f"See {hyperlink} for details"]))

    tui.show_overlay(_StaticComponent(["OVERLAY-TEXT"]), OverlayOptions(anchor="center", width=20))
    tui.start()

    assert len(tui.previous_lines) > 0


def test_overlay_should_position_overlay_at_top_center_like_ts() -> None:
    """should position overlay at top-center"""
    terminal = _TerminalStub(columns=80, rows=24)
    tui = TUI(terminal)
    tui.add_child(_StaticComponent([]))
    tui.show_overlay(_StaticComponent(["CENTERED"]), OverlayOptions(anchor="top-center", width=10))
    tui.start()

    first_row = tui.previous_lines[0]
    col_index = _visible_index_of(first_row, "CENTERED")
    assert "CENTERED" in first_row
    assert 30 <= col_index <= 40


def test_overlay_remains_visible_when_content_is_shorter_than_terminal_height() -> None:
    terminal = _TerminalStub(columns=80, rows=24)
    tui = TUI(terminal)
    tui.add_child(_StaticComponent(["Line 1", "Line 2", "Line 3"]))

    tui.show_overlay(
        _StaticComponent(["OVERLAY_TOP", "OVERLAY_MID", "OVERLAY_BOT"]),
        OverlayOptions(anchor="center", width=20),
    )
    tui.start()

    assert any("OVERLAY_" in line for line in tui.previous_lines)


def test_overlay_handles_complex_ansi_content_without_exceeding_terminal_width() -> None:
    terminal = _TerminalStub(columns=80, rows=24)
    tui = TUI(terminal)
    complex_line = (
        "\x1b[48;2;40;50;40m \x1b[38;2;128;128;128mSome styled content\x1b[39m\x1b[49m"
        "\x1b]8;;http://example.com\x07link\x1b]8;;\x07"
        + " more content " * 10
    )

    tui.add_child(_StaticComponent([]))
    tui.show_overlay(_StaticComponent([complex_line, complex_line, complex_line]), OverlayOptions(width=60))
    tui.start()

    assert len(tui.previous_lines) > 0
    assert all(visible_width(line) <= terminal.columns for line in tui.previous_lines)


def test_overlay_should_handle_overlay_with_complex_ansi_sequences_without_crashing_like_ts() -> None:
    """should handle overlay with complex ANSI sequences without crashing"""
    terminal = _TerminalStub(columns=80, rows=24)
    tui = TUI(terminal)
    complex_line = (
        "\x1b[48;2;40;50;40m \x1b[38;2;128;128;128mSome styled content\x1b[39m\x1b[49m"
        "\x1b]8;;http://example.com\x07link\x1b]8;;\x07"
        + " more content " * 10
    )

    tui.add_child(_StaticComponent([]))
    tui.show_overlay(_StaticComponent([complex_line, complex_line, complex_line]), OverlayOptions(width=60))
    tui.start()

    assert len(tui.previous_lines) > 0


def test_overlay_handles_wide_characters_at_declared_width_boundary() -> None:
    terminal = _TerminalStub(columns=80, rows=24)
    tui = TUI(terminal)
    tui.add_child(_StaticComponent([]))
    tui.show_overlay(_StaticComponent(["中文日本語한글テスト漢字"]), OverlayOptions(width=15))
    tui.start()

    assert len(tui.previous_lines) > 0
    assert all(visible_width(line) <= terminal.columns for line in tui.previous_lines)


def test_overlay_should_handle_wide_characters_at_overlay_boundary_like_ts() -> None:
    """should handle wide characters at overlay boundary"""
    terminal = _TerminalStub(columns=80, rows=24)
    tui = TUI(terminal)
    tui.add_child(_StaticComponent([]))
    tui.show_overlay(_StaticComponent(["中文日本語한글テスト漢字"]), OverlayOptions(width=15))
    tui.start()

    assert len(tui.previous_lines) > 0


def test_overlay_on_styled_base_content_remains_visible() -> None:
    terminal = _TerminalStub(columns=80, rows=24)
    tui = TUI(terminal)

    class _StyledContent:
        def render(self, width: int) -> list[str]:
            styled_line = f"\x1b[1m\x1b[38;2;255;0;0m{'X' * width}\x1b[0m"
            return [styled_line, styled_line, styled_line]

        def invalidate(self) -> None:
            return None

    tui.add_child(_StyledContent())
    tui.show_overlay(_StaticComponent(["OVERLAY"]), OverlayOptions(anchor="center", width=20))
    tui.start()

    assert any("OVERLAY" in line for line in tui.previous_lines)


def test_overlay_should_handle_overlay_composited_on_styled_base_content_like_ts() -> None:
    """should handle overlay composited on styled base content"""
    terminal = _TerminalStub(columns=80, rows=24)
    tui = TUI(terminal)

    class _StyledContent:
        def render(self, width: int) -> list[str]:
            styled_line = f"\x1b[1m\x1b[38;2;255;0;0m{'X' * width}\x1b[0m"
            return [styled_line, styled_line, styled_line]

        def invalidate(self) -> None:
            return None

    tui.add_child(_StyledContent())
    tui.show_overlay(_StaticComponent(["OVERLAY"]), OverlayOptions(anchor="center", width=20))
    tui.start()

    assert any("OVERLAY" in line for line in tui.previous_lines)


def test_overlay_declared_width_caps_overwide_overlay_content_without_crashing() -> None:
    terminal = _TerminalStub(columns=80, rows=24)
    tui = TUI(terminal)
    tui.add_child(_StaticComponent([]))
    tui.show_overlay(_StaticComponent(["X" * 100]), OverlayOptions(width=20))
    tui.start()

    assert len(tui.previous_lines) > 0
    assert all(visible_width(line) <= terminal.columns for line in tui.previous_lines)


def test_overlay_should_truncate_overlay_lines_that_exceed_declared_width_like_ts() -> None:
    """should truncate overlay lines that exceed declared width"""
    terminal = _TerminalStub(columns=80, rows=24)
    tui = TUI(terminal)
    tui.add_child(_StaticComponent([]))
    tui.show_overlay(_StaticComponent(["X" * 100]), OverlayOptions(width=20))
    tui.start()

    assert len(tui.previous_lines) > 0
    assert all(visible_width(line) <= terminal.columns for line in tui.previous_lines)


def test_overlay_positioned_at_terminal_edge_does_not_crash() -> None:
    terminal = _TerminalStub(columns=80, rows=24)
    tui = TUI(terminal)
    tui.add_child(_StaticComponent([]))
    tui.show_overlay(_StaticComponent(["X" * 50]), OverlayOptions(col=60, width=20))
    tui.start()

    assert len(tui.previous_lines) > 0
    assert all(visible_width(line) <= terminal.columns for line in tui.previous_lines)


def test_overlay_should_handle_overlay_positioned_at_terminal_edge_like_ts() -> None:
    """should handle overlay positioned at terminal edge"""
    terminal = _TerminalStub(columns=80, rows=24)
    tui = TUI(terminal)
    tui.add_child(_StaticComponent([]))
    tui.show_overlay(_StaticComponent(["X" * 50]), OverlayOptions(col=60, width=20))
    tui.start()

    assert len(tui.previous_lines) > 0


def test_overlay_top_center_anchor_positions_overlay_near_horizontal_center() -> None:
    terminal = _TerminalStub(columns=80, rows=24)
    tui = TUI(terminal)
    tui.add_child(_StaticComponent([]))
    tui.show_overlay(_StaticComponent(["CENTERED"]), OverlayOptions(anchor="top-center", width=10))
    tui.start()

    first_row = _strip_ansi(tui.previous_lines[0])
    assert "CENTERED" in first_row
    col_index = _visible_index_of(first_row, "CENTERED")
    assert 30 <= col_index <= 40


def test_overlay_compositing_does_not_leak_styles_to_following_line() -> None:
    """should not leak styles when overlay slicing drops trailing SGR resets"""
    terminal = _TerminalStub(columns=20, rows=6)
    tui = TUI(terminal)
    tui.add_child(_StaticComponent(["\x1b[3m" + "X" * 20 + "\x1b[23m", "INPUT"]))

    tui.show_overlay(_StaticComponent(["OVR"]), OverlayOptions(row=0, col=5, width=3, non_capturing=True))
    tui.start()

    assert tui.previous_lines[1] == "INPUT\x1b[0m\x1b]8;;\x07"


def test_tui_base_line_trailing_reset_beyond_visible_width_does_not_leak_style() -> None:
    """should not leak styles when a trailing reset sits beyond the last visible column (no overlay)"""
    terminal = _TerminalStub(columns=20, rows=6)
    tui = TUI(terminal)
    tui.add_child(_StaticComponent(["\x1b[3m" + "X" * 20 + "\x1b[23m", "INPUT"]))
    tui.start()

    assert tui.previous_lines[1] == "INPUT\x1b[0m\x1b]8;;\x07"


def test_non_capturing_overlay_preserves_focus_and_can_restore_it() -> None:
    terminal = _TerminalStub()
    tui = TUI(terminal)
    editor = _FocusableComponent(["EDITOR"])
    overlay = _FocusableComponent(["OVERLAY"])
    tui.set_focus(editor)
    handle = tui.show_overlay(overlay, OverlayOptions(non_capturing=True))

    assert editor.focused is True
    assert overlay.focused is False

    handle.focus()
    assert editor.focused is False
    assert overlay.focused is True

    handle.unfocus()
    assert editor.focused is True
    assert overlay.focused is False


def test_non_capturing_overlay_preserves_focus_on_creation_like_ts() -> None:
    terminal = _TerminalStub()
    tui = TUI(terminal)
    editor = _FocusableComponent(["EDITOR"])
    overlay = _FocusableComponent(["OVERLAY"])
    tui.set_focus(editor)
    tui.show_overlay(overlay, OverlayOptions(non_capturing=True))

    assert editor.focused is True
    assert overlay.focused is False


def test_non_capturing_overlay_focus_transfers_focus_to_the_overlay_like_ts() -> None:
    terminal = _TerminalStub()
    tui = TUI(terminal)
    editor = _FocusableComponent(["EDITOR"])
    overlay = _FocusableComponent(["OVERLAY"])
    tui.set_focus(editor)
    handle = tui.show_overlay(overlay, OverlayOptions(non_capturing=True))

    handle.focus()

    assert editor.focused is False
    assert overlay.focused is True


def test_non_capturing_overlay_unfocus_restores_previous_focus_like_ts() -> None:
    terminal = _TerminalStub()
    tui = TUI(terminal)
    editor = _FocusableComponent(["EDITOR"])
    overlay = _FocusableComponent(["OVERLAY"])
    tui.set_focus(editor)
    handle = tui.show_overlay(overlay, OverlayOptions(non_capturing=True))
    handle.focus()

    handle.unfocus()

    assert editor.focused is True
    assert overlay.focused is False


def test_non_capturing_overlay_unhide_does_not_auto_focus() -> None:
    terminal = _TerminalStub()
    tui = TUI(terminal)
    editor = _FocusableComponent(["EDITOR"])
    overlay = _FocusableComponent(["OVERLAY"])
    tui.set_focus(editor)

    handle = tui.show_overlay(overlay, OverlayOptions(non_capturing=True))
    handle.set_hidden(True)
    handle.set_hidden(False)

    assert editor.focused is True
    assert overlay.focused is False


def test_non_capturing_overlay_set_hidden_false_does_not_auto_focus_like_ts() -> None:
    terminal = _TerminalStub()
    tui = TUI(terminal)
    editor = _FocusableComponent(["EDITOR"])
    overlay = _FocusableComponent(["OVERLAY"])
    tui.set_focus(editor)

    handle = tui.show_overlay(overlay, OverlayOptions(non_capturing=True))
    handle.set_hidden(True)
    handle.set_hidden(False)

    assert editor.focused is True
    assert overlay.focused is False


def test_non_capturing_overlay_set_hidden_false_on_non_capturing_overlay_does_not_auto_focus_exact_title_like_ts() -> None:
    terminal = _TerminalStub()
    tui = TUI(terminal)
    editor = _FocusableComponent(["EDITOR"])
    overlay = _FocusableComponent(["OVERLAY"])
    tui.set_focus(editor)

    handle = tui.show_overlay(overlay, OverlayOptions(non_capturing=True))
    handle.set_hidden(True)
    handle.set_hidden(False)

    assert editor.focused is True
    assert overlay.focused is False


def test_non_capturing_overlay_set_hidden_false_on_non_capturing_overlay_does_not_auto_focus_with_set_hidden_title_like_ts() -> None:
    terminal = _TerminalStub()
    tui = TUI(terminal)
    editor = _FocusableComponent(["EDITOR"])
    overlay = _FocusableComponent(["OVERLAY"])
    tui.set_focus(editor)

    handle = tui.show_overlay(overlay, OverlayOptions(non_capturing=True))
    handle.set_hidden(True)
    handle.set_hidden(False)

    assert editor.focused is True
    assert overlay.focused is False


def test_non_capturing_overlay_sethidden_false_on_non_capturing_overlay_does_not_auto_focus_exact_title_like_ts() -> None:
    terminal = _TerminalStub()
    tui = TUI(terminal)
    editor = _FocusableComponent(["EDITOR"])
    overlay = _FocusableComponent(["OVERLAY"])
    tui.set_focus(editor)

    handle = tui.show_overlay(overlay, OverlayOptions(non_capturing=True))
    handle.set_hidden(True)
    handle.set_hidden(False)

    assert editor.focused is True
    assert overlay.focused is False


def test_hide_focused_capturing_overlay_restores_focus_to_editor_even_with_non_capturing_below() -> None:
    terminal = _TerminalStub()
    tui = TUI(terminal)
    editor = _FocusableComponent(["EDITOR"])
    non_capturing = _FocusableComponent(["NC"])
    capturing = _FocusableComponent(["CAP"])
    tui.set_focus(editor)

    tui.show_overlay(non_capturing, OverlayOptions(non_capturing=True))
    handle = tui.show_overlay(capturing)
    assert capturing.focused is True

    handle.hide()

    assert editor.focused is True
    assert non_capturing.focused is False
    assert capturing.focused is False


def test_non_capturing_overlay_hide_when_focused_restores_focus_correctly_like_ts() -> None:
    terminal = _TerminalStub()
    tui = TUI(terminal)
    editor = _FocusableComponent(["EDITOR"])
    non_capturing = _FocusableComponent(["NC"])
    capturing = _FocusableComponent(["CAP"])
    tui.set_focus(editor)

    tui.show_overlay(non_capturing, OverlayOptions(non_capturing=True))
    handle = tui.show_overlay(capturing)
    assert capturing.focused is True

    handle.hide()

    assert editor.focused is True
    assert non_capturing.focused is False
    assert capturing.focused is False


def test_non_capturing_overlay_hide_when_overlay_is_not_focused_does_not_change_focus_exact_title_like_ts() -> None:
    terminal = _TerminalStub()
    tui = TUI(terminal)
    editor = _FocusableComponent(["EDITOR"])
    overlay = _FocusableComponent(["OVERLAY"])
    tui.set_focus(editor)

    handle = tui.show_overlay(overlay, OverlayOptions(non_capturing=True))
    handle.hide()

    assert editor.focused is True


def test_non_capturing_overlay_capturing_overlay_removed_with_non_capturing_below_restores_focus_to_editor_exact_title_like_ts() -> None:
    terminal = _TerminalStub()
    tui = TUI(terminal)
    editor = _FocusableComponent(["EDITOR"])
    non_capturing = _FocusableComponent(["NC"])
    capturing = _FocusableComponent(["CAP"])
    tui.set_focus(editor)

    tui.show_overlay(non_capturing, OverlayOptions(non_capturing=True))
    handle = tui.show_overlay(capturing)
    handle.hide()

    assert editor.focused is True
    assert non_capturing.focused is False


def test_sub_overlay_cleanup_then_hide_overlay_restores_focus_and_input_to_editor() -> None:
    terminal = _TerminalStub()
    tui = TUI(terminal)
    editor = _FocusableComponent(["EDITOR"])
    timer = _FocusableComponent(["TIMER"])
    controller = _FocusableComponent(["CTRL"])

    tui.add_child(_StaticComponent([]))
    tui.set_focus(editor)
    tui.start()

    timer_handle = tui.show_overlay(timer, OverlayOptions(non_capturing=True))
    tui.show_overlay(controller)

    assert controller.focused is True
    assert editor.focused is False

    timer_handle.hide()
    tui.hide_overlay()

    assert editor.focused is True
    assert controller.focused is False
    assert timer.focused is False

    terminal.on_input("x")

    assert editor.inputs == ["x"]
    assert controller.inputs == []
    assert timer.inputs == []


def test_non_capturing_overlay_sub_overlay_cleanup_then_hide_overlay_restores_focus_and_input_to_editor_like_ts() -> None:
    terminal = _TerminalStub()
    tui = TUI(terminal)
    editor = _FocusableComponent(["EDITOR"])
    timer = _FocusableComponent(["TIMER"])
    controller = _FocusableComponent(["CTRL"])

    tui.add_child(_StaticComponent([]))
    tui.set_focus(editor)
    tui.start()

    timer_handle = tui.show_overlay(timer, OverlayOptions(non_capturing=True))
    tui.show_overlay(controller)
    timer_handle.hide()
    tui.hide_overlay()

    assert editor.focused is True
    assert controller.focused is False
    assert timer.focused is False

    terminal.on_input("x")
    assert editor.inputs == ["x"]


def test_non_capturing_overlay_sub_overlay_cleanup_then_hide_overlay_restores_focus_and_input_to_editor_with_hide_overlay_title_like_ts() -> None:
    terminal = _TerminalStub()
    tui = TUI(terminal)
    editor = _FocusableComponent(["EDITOR"])
    timer = _FocusableComponent(["TIMER"])
    controller = _FocusableComponent(["CTRL"])

    tui.add_child(_StaticComponent([]))
    tui.set_focus(editor)
    tui.start()

    timer_handle = tui.show_overlay(timer, OverlayOptions(non_capturing=True))
    tui.show_overlay(controller)
    timer_handle.hide()
    tui.hide_overlay()
    terminal.on_input("x")

    assert editor.focused is True
    assert controller.focused is False
    assert timer.focused is False
    assert editor.inputs == ["x"]


def test_non_capturing_overlay_sub_overlay_cleanup_then_hideoverlay_restores_focus_and_input_to_editor_exact_title_like_ts() -> None:
    terminal = _TerminalStub()
    tui = TUI(terminal)
    editor = _FocusableComponent(["EDITOR"])
    timer = _FocusableComponent(["TIMER"])
    controller = _FocusableComponent(["CTRL"])

    tui.add_child(_StaticComponent([]))
    tui.set_focus(editor)
    tui.start()

    timer_handle = tui.show_overlay(timer, OverlayOptions(non_capturing=True))
    tui.show_overlay(controller)
    timer_handle.hide()
    tui.hide_overlay()
    terminal.on_input("x")

    assert editor.focused is True
    assert controller.focused is False
    assert timer.focused is False
    assert editor.inputs == ["x"]


@pytest.mark.anyio
async def test_microtask_deferred_sub_overlay_pattern_restores_focus() -> None:
    terminal = _TerminalStub()
    tui = TUI(terminal)
    editor = _FocusableComponent(["EDITOR"])
    timer = _FocusableComponent(["TIMER"])
    controller = _FocusableComponent(["CTRL"])

    tui.add_child(_StaticComponent([]))
    tui.set_focus(editor)
    tui.start()

    timer_handle = tui.show_overlay(timer, OverlayOptions(non_capturing=True))

    async def _show_controller() -> None:
        await asyncio.sleep(0)
        tui.show_overlay(controller)

    task = asyncio.create_task(_show_controller())
    await asyncio.sleep(0)
    await task

    assert controller.focused is True
    assert editor.focused is False

    timer_handle.hide()
    tui.hide_overlay()
    await asyncio.sleep(0)

    assert editor.focused is True
    assert controller.focused is False
    assert timer.focused is False


@pytest.mark.anyio
async def test_non_capturing_overlay_microtask_deferred_sub_overlay_pattern_restores_focus_like_ts() -> None:
    terminal = _TerminalStub()
    tui = TUI(terminal)
    editor = _FocusableComponent(["EDITOR"])
    timer = _FocusableComponent(["TIMER"])
    controller = _FocusableComponent(["CTRL"])

    tui.add_child(_StaticComponent([]))
    tui.set_focus(editor)
    tui.start()

    timer_handle = tui.show_overlay(timer, OverlayOptions(non_capturing=True))

    async def _show_controller() -> None:
        await asyncio.sleep(0)
        tui.show_overlay(controller)

    task = asyncio.create_task(_show_controller())
    await asyncio.sleep(0)
    await task

    timer_handle.hide()
    tui.hide_overlay()
    await asyncio.sleep(0)

    assert editor.focused is True
    assert controller.focused is False
    assert timer.focused is False


@pytest.mark.anyio
async def test_non_capturing_overlay_microtask_deferred_sub_overlay_pattern_show_extension_custom_simulation_restores_focus_like_ts() -> None:
    terminal = _TerminalStub()
    tui = TUI(terminal)
    editor = _FocusableComponent(["EDITOR"])
    timer = _FocusableComponent(["TIMER"])
    controller = _FocusableComponent(["CTRL"])

    tui.add_child(_StaticComponent([]))
    tui.set_focus(editor)
    tui.start()

    timer_handle = tui.show_overlay(timer, OverlayOptions(non_capturing=True))

    async def _show_controller() -> None:
        await asyncio.sleep(0)
        tui.show_overlay(controller)

    task = asyncio.create_task(_show_controller())
    await asyncio.sleep(0)
    await task

    timer_handle.hide()
    tui.hide_overlay()
    await asyncio.sleep(0)

    assert editor.focused is True
    assert controller.focused is False
    assert timer.focused is False


@pytest.mark.anyio
async def test_non_capturing_overlay_microtask_deferred_sub_overlay_pattern_showextensioncustom_simulation_restores_focus_exact_title_like_ts() -> None:
    terminal = _TerminalStub()
    tui = TUI(terminal)
    editor = _FocusableComponent(["EDITOR"])
    timer = _FocusableComponent(["TIMER"])
    controller = _FocusableComponent(["CTRL"])

    tui.add_child(_StaticComponent([]))
    tui.set_focus(editor)
    tui.start()

    timer_handle = tui.show_overlay(timer, OverlayOptions(non_capturing=True))

    async def _show_controller() -> None:
        await asyncio.sleep(0)
        tui.show_overlay(controller)

    task = asyncio.create_task(_show_controller())
    await asyncio.sleep(0)
    await task

    timer_handle.hide()
    tui.hide_overlay()
    await asyncio.sleep(0)

    assert editor.focused is True
    assert controller.focused is False
    assert timer.focused is False


def test_later_stacked_overlay_renders_on_top() -> None:
    terminal = _TerminalStub(columns=80, rows=24)
    tui = TUI(terminal)
    tui.add_child(_StaticComponent([]))
    tui.show_overlay(_StaticComponent(["FIRST-OVERLAY"]), OverlayOptions(anchor="top-left", width=20))
    tui.show_overlay(_StaticComponent(["SECOND"]), OverlayOptions(anchor="top-left", width=10))
    tui.start()

    assert "SECOND" in tui.previous_lines[0]


def test_overlay_should_render_multiple_overlays_with_later_ones_on_top_like_ts() -> None:
    """should render multiple overlays with later ones on top"""
    terminal = _TerminalStub(columns=80, rows=24)
    tui = TUI(terminal)
    tui.add_child(_StaticComponent([]))
    tui.show_overlay(_StaticComponent(["FIRST-OVERLAY"]), OverlayOptions(anchor="top-left", width=20))
    tui.show_overlay(_StaticComponent(["SECOND"]), OverlayOptions(anchor="top-left", width=10))
    tui.start()

    assert "SECOND" in tui.previous_lines[0]


def test_hide_overlay_reveals_previous_overlay_in_stack_order() -> None:
    terminal = _TerminalStub(columns=80, rows=24)
    tui = TUI(terminal)
    tui.add_child(_StaticComponent([]))
    tui.show_overlay(_StaticComponent(["FIRST"]), OverlayOptions(anchor="top-left", width=10))
    tui.show_overlay(_StaticComponent(["SECOND"]), OverlayOptions(anchor="top-left", width=10))
    tui.start()

    assert "SECOND" in tui.previous_lines[0]

    tui.hide_overlay()

    assert "FIRST" in tui.previous_lines[0]


def test_overlay_should_handle_overlays_at_different_positions_without_interference_like_ts() -> None:
    """should handle overlays at different positions without interference"""
    terminal = _TerminalStub(columns=80, rows=24)
    tui = TUI(terminal)
    tui.add_child(_StaticComponent([]))
    tui.show_overlay(_StaticComponent(["TOP-LEFT"]), OverlayOptions(anchor="top-left", width=15))
    tui.show_overlay(_StaticComponent(["BTM-RIGHT"]), OverlayOptions(anchor="bottom-right", width=15))
    tui.start()

    assert "TOP-LEFT" in tui.previous_lines[0]
    assert "BTM-RIGHT" in tui.previous_lines[23]


def test_overlay_should_properly_hide_overlays_in_stack_order_like_ts() -> None:
    """should properly hide overlays in stack order"""
    terminal = _TerminalStub(columns=80, rows=24)
    tui = TUI(terminal)
    tui.add_child(_StaticComponent([]))
    tui.show_overlay(_StaticComponent(["FIRST"]), OverlayOptions(anchor="top-left", width=10))
    tui.show_overlay(_StaticComponent(["SECOND"]), OverlayOptions(anchor="top-left", width=10))
    tui.start()

    assert "SECOND" in tui.previous_lines[0]

    tui.hide_overlay()

    assert "FIRST" in tui.previous_lines[0]


def test_invisible_focused_overlay_redirects_input_to_topmost_visible_capturing_overlay() -> None:
    terminal = _TerminalStub()
    tui = TUI(terminal)
    editor = _FocusableComponent(["EDITOR"])
    fallback = _FocusableComponent(["FALLBACK"])
    non_capturing = _FocusableComponent(["NC"])
    primary = _FocusableComponent(["PRIMARY"])
    is_visible = True

    tui.set_focus(editor)
    tui.start()
    tui.show_overlay(fallback)
    tui.show_overlay(non_capturing, OverlayOptions(non_capturing=True))
    tui.show_overlay(primary, OverlayOptions(visible=lambda _columns, _rows: is_visible))
    assert primary.focused is True

    is_visible = False
    terminal.on_input("x")

    assert primary.inputs == []
    assert non_capturing.inputs == []
    assert fallback.inputs == ["x"]
    assert fallback.focused is True


def test_non_capturing_overlay_handle_input_redirection_skips_non_capturing_overlays_when_focused_overlay_becomes_invisible_like_ts() -> None:
    terminal = _TerminalStub()
    tui = TUI(terminal)
    editor = _FocusableComponent(["EDITOR"])
    fallback = _FocusableComponent(["FALLBACK"])
    non_capturing = _FocusableComponent(["NC"])
    primary = _FocusableComponent(["PRIMARY"])
    is_visible = True

    tui.set_focus(editor)
    tui.start()
    tui.show_overlay(fallback)
    tui.show_overlay(non_capturing, OverlayOptions(non_capturing=True))
    tui.show_overlay(primary, OverlayOptions(visible=lambda _columns, _rows: is_visible))

    is_visible = False
    terminal.on_input("x")

    assert primary.inputs == []
    assert non_capturing.inputs == []
    assert fallback.inputs == ["x"]


def test_non_capturing_overlay_handle_input_redirection_skips_non_capturing_overlays_when_focused_overlay_becomes_invisible_exact_title_like_ts() -> None:
    terminal = _TerminalStub()
    tui = TUI(terminal)
    editor = _FocusableComponent(["EDITOR"])
    fallback = _FocusableComponent(["FALLBACK"])
    non_capturing = _FocusableComponent(["NC"])
    primary = _FocusableComponent(["PRIMARY"])
    is_visible = True

    tui.set_focus(editor)
    tui.start()
    tui.show_overlay(fallback)
    tui.show_overlay(non_capturing, OverlayOptions(non_capturing=True))
    tui.show_overlay(primary, OverlayOptions(visible=lambda _columns, _rows: is_visible))

    is_visible = False
    terminal.on_input("x")

    assert primary.inputs == []
    assert non_capturing.inputs == []
    assert fallback.inputs == ["x"]


def test_non_capturing_overlay_handle_input_redirection_skips_non_capturing_overlays_when_focused_overlay_becomes_invisible_with_handle_input_title_like_ts() -> None:
    terminal = _TerminalStub()
    tui = TUI(terminal)
    editor = _FocusableComponent(["EDITOR"])
    fallback = _FocusableComponent(["FALLBACK"])
    non_capturing = _FocusableComponent(["NC"])
    primary = _FocusableComponent(["PRIMARY"])
    is_visible = True

    tui.set_focus(editor)
    tui.start()
    tui.show_overlay(fallback)
    tui.show_overlay(non_capturing, OverlayOptions(non_capturing=True))
    tui.show_overlay(primary, OverlayOptions(visible=lambda _columns, _rows: is_visible))
    is_visible = False
    terminal.on_input("x")

    assert primary.inputs == []
    assert non_capturing.inputs == []
    assert fallback.inputs == ["x"]


def test_non_capturing_overlay_handleinput_redirection_skips_non_capturing_overlays_when_focused_overlay_becomes_invisible_exact_title_like_ts() -> None:
    terminal = _TerminalStub()
    tui = TUI(terminal)
    editor = _FocusableComponent(["EDITOR"])
    fallback = _FocusableComponent(["FALLBACK"])
    non_capturing = _FocusableComponent(["NC"])
    primary = _FocusableComponent(["PRIMARY"])
    is_visible = True

    tui.set_focus(editor)
    tui.start()
    tui.show_overlay(fallback)
    tui.show_overlay(non_capturing, OverlayOptions(non_capturing=True))
    tui.show_overlay(primary, OverlayOptions(visible=lambda _columns, _rows: is_visible))
    is_visible = False
    terminal.on_input("x")

    assert primary.inputs == []
    assert non_capturing.inputs == []
    assert fallback.inputs == ["x"]


def test_hide_overlay_ignores_topmost_non_capturing_overlay_for_focus_reassignment() -> None:
    terminal = _TerminalStub()
    tui = TUI(terminal)
    editor = _FocusableComponent(["EDITOR"])
    capturing = _FocusableComponent(["CAP"])
    non_capturing = _FocusableComponent(["NC"])
    tui.set_focus(editor)

    tui.show_overlay(capturing)
    tui.show_overlay(non_capturing, OverlayOptions(non_capturing=True))
    assert capturing.focused is True

    tui.hide_overlay()

    assert capturing.focused is True
    assert non_capturing.focused is False


def test_non_capturing_overlay_hide_overlay_does_not_reassign_focus_when_topmost_overlay_is_non_capturing_like_ts() -> None:
    terminal = _TerminalStub()
    tui = TUI(terminal)
    editor = _FocusableComponent(["EDITOR"])
    capturing = _FocusableComponent(["CAP"])
    non_capturing = _FocusableComponent(["NC"])
    tui.set_focus(editor)

    tui.show_overlay(capturing)
    tui.show_overlay(non_capturing, OverlayOptions(non_capturing=True))
    tui.hide_overlay()

    assert capturing.focused is True
    assert non_capturing.focused is False


def test_non_capturing_overlay_hide_overlay_does_not_reassign_focus_when_topmost_overlay_is_non_capturing_exact_title_like_ts() -> None:
    terminal = _TerminalStub()
    tui = TUI(terminal)
    editor = _FocusableComponent(["EDITOR"])
    capturing = _FocusableComponent(["CAP"])
    non_capturing = _FocusableComponent(["NC"])
    tui.set_focus(editor)

    tui.show_overlay(capturing)
    tui.show_overlay(non_capturing, OverlayOptions(non_capturing=True))
    tui.hide_overlay()

    assert capturing.focused is True
    assert non_capturing.focused is False


def test_non_capturing_overlay_hide_overlay_does_not_reassign_focus_when_topmost_overlay_is_non_capturing_with_hide_overlay_title_like_ts() -> None:
    terminal = _TerminalStub()
    tui = TUI(terminal)
    editor = _FocusableComponent(["EDITOR"])
    capturing = _FocusableComponent(["CAP"])
    non_capturing = _FocusableComponent(["NC"])
    tui.set_focus(editor)

    tui.show_overlay(capturing)
    tui.show_overlay(non_capturing, OverlayOptions(non_capturing=True))
    tui.hide_overlay()

    assert capturing.focused is True
    assert non_capturing.focused is False


def test_non_capturing_overlay_hideoverlay_does_not_reassign_focus_when_topmost_overlay_is_non_capturing_exact_title_like_ts() -> None:
    terminal = _TerminalStub()
    tui = TUI(terminal)
    editor = _FocusableComponent(["EDITOR"])
    capturing = _FocusableComponent(["CAP"])
    non_capturing = _FocusableComponent(["NC"])
    tui.set_focus(editor)

    tui.show_overlay(capturing)
    tui.show_overlay(non_capturing, OverlayOptions(non_capturing=True))
    tui.hide_overlay()

    assert capturing.focused is True
    assert non_capturing.focused is False


def test_non_capturing_overlay_capturing_overlay_unfocus_on_topmost_capturing_overlay_falls_back_to_prefocus_exact_title_like_ts() -> None:
    terminal = _TerminalStub()
    tui = TUI(terminal)
    editor = _FocusableComponent(["EDITOR"])
    capturing = _FocusableComponent(["CAP"])
    tui.set_focus(editor)

    handle = tui.show_overlay(capturing)
    handle.unfocus()

    assert editor.focused is True
    assert capturing.focused is False


def test_unfocus_with_null_prefocus_clears_focus_and_does_not_route_input_back() -> None:
    terminal = _TerminalStub()
    tui = TUI(terminal)
    overlay = _FocusableComponent(["OVERLAY"])

    tui.start()
    handle = tui.show_overlay(overlay)
    assert overlay.focused is True

    handle.unfocus()
    terminal.on_input("x")

    assert overlay.focused is False
    assert overlay.inputs == []
    assert handle.is_focused() is False


def test_non_capturing_overlay_unfocus_with_null_prefocus_clears_focus_and_does_not_route_input_back_like_ts() -> None:
    terminal = _TerminalStub()
    tui = TUI(terminal)
    overlay = _FocusableComponent(["OVERLAY"])

    tui.start()
    handle = tui.show_overlay(overlay)
    handle.unfocus()
    terminal.on_input("x")

    assert overlay.focused is False
    assert overlay.inputs == []
    assert handle.is_focused() is False


def test_non_capturing_overlay_unfocus_with_null_prefocus_clears_focus_and_does_not_route_input_back_exact_title_like_ts() -> None:
    terminal = _TerminalStub()
    tui = TUI(terminal)
    overlay = _FocusableComponent(["OVERLAY"])

    tui.start()
    handle = tui.show_overlay(overlay)
    handle.unfocus()
    terminal.on_input("x")

    assert overlay.focused is False
    assert overlay.inputs == []
    assert handle.is_focused() is False


def test_non_capturing_overlay_unfocus_with_null_pre_focus_clears_focus_and_does_not_route_input_back_like_ts() -> None:
    terminal = _TerminalStub()
    tui = TUI(terminal)
    overlay = _FocusableComponent(["OVERLAY"])

    tui.start()
    handle = tui.show_overlay(overlay)
    handle.unfocus()
    terminal.on_input("x")

    assert overlay.focused is False
    assert overlay.inputs == []
    assert handle.is_focused() is False


def test_non_capturing_overlay_unfocus_with_null_prefocus_clears_focus_and_does_not_route_input_back_to_overlay_exact_title_like_ts() -> None:
    terminal = _TerminalStub()
    tui = TUI(terminal)
    overlay = _FocusableComponent(["OVERLAY"])

    tui.start()
    handle = tui.show_overlay(overlay)
    handle.unfocus()
    terminal.on_input("x")

    assert overlay.focused is False
    assert overlay.inputs == []
    assert handle.is_focused() is False


def test_toggle_focus_between_non_capturing_overlays_then_unfocus_returns_to_editor() -> None:
    terminal = _TerminalStub()
    tui = TUI(terminal)
    editor = _FocusableComponent(["EDITOR"])
    overlay_a = _FocusableComponent(["A"])
    overlay_b = _FocusableComponent(["B"])
    tui.set_focus(editor)

    handle_a = tui.show_overlay(overlay_a, OverlayOptions(non_capturing=True))
    handle_b = tui.show_overlay(overlay_b, OverlayOptions(non_capturing=True))
    handle_a.focus()
    handle_b.focus()
    handle_a.focus()
    handle_a.unfocus()

    assert editor.focused is True
    assert overlay_a.focused is False
    assert overlay_b.focused is False


def test_non_capturing_overlay_multiple_capturing_and_non_capturing_overlays_restore_focus_through_removals_like_ts() -> None:
    terminal = _TerminalStub()
    tui = TUI(terminal)
    editor = _FocusableComponent(["EDITOR"])
    overlay_a = _FocusableComponent(["A"])
    overlay_b = _FocusableComponent(["B"])
    tui.set_focus(editor)

    handle_a = tui.show_overlay(overlay_a, OverlayOptions(non_capturing=True))
    handle_b = tui.show_overlay(overlay_b, OverlayOptions(non_capturing=True))
    handle_a.focus()
    handle_b.focus()
    handle_a.focus()
    handle_a.unfocus()

    assert editor.focused is True
    assert overlay_a.focused is False
    assert overlay_b.focused is False


def test_focus_on_hidden_non_capturing_overlay_is_noop() -> None:
    terminal = _TerminalStub()
    tui = TUI(terminal)
    editor = _FocusableComponent(["EDITOR"])
    overlay = _FocusableComponent(["OVERLAY"])
    tui.set_focus(editor)

    handle = tui.show_overlay(overlay, OverlayOptions(non_capturing=True))
    handle.set_hidden(True)
    handle.focus()

    assert editor.focused is True
    assert handle.is_focused() is False


def test_non_capturing_overlay_focus_on_hidden_overlay_is_a_noop_like_ts() -> None:
    terminal = _TerminalStub()
    tui = TUI(terminal)
    editor = _FocusableComponent(["EDITOR"])
    overlay = _FocusableComponent(["OVERLAY"])
    tui.set_focus(editor)

    handle = tui.show_overlay(overlay, OverlayOptions(non_capturing=True))
    handle.set_hidden(True)
    handle.focus()

    assert editor.focused is True
    assert handle.is_focused() is False


def test_non_capturing_overlay_focus_on_hidden_overlay_is_a_noop_exact_title_like_ts() -> None:
    terminal = _TerminalStub()
    tui = TUI(terminal)
    editor = _FocusableComponent(["EDITOR"])
    overlay = _FocusableComponent(["OVERLAY"])
    tui.set_focus(editor)

    handle = tui.show_overlay(overlay, OverlayOptions(non_capturing=True))
    handle.set_hidden(True)
    handle.focus()

    assert editor.focused is True
    assert handle.is_focused() is False


def test_non_capturing_overlay_focus_on_hidden_overlay_with_focus_hidden_title_like_ts() -> None:
    terminal = _TerminalStub()
    tui = TUI(terminal)
    editor = _FocusableComponent(["EDITOR"])
    overlay = _FocusableComponent(["OVERLAY"])
    tui.set_focus(editor)

    handle = tui.show_overlay(overlay, OverlayOptions(non_capturing=True))
    handle.set_hidden(True)
    handle.focus()

    assert editor.focused is True
    assert handle.is_focused() is False


def test_non_capturing_overlay_focus_on_hidden_overlay_is_no_op_exact_title_like_ts() -> None:
    terminal = _TerminalStub()
    tui = TUI(terminal)
    editor = _FocusableComponent(["EDITOR"])
    overlay = _FocusableComponent(["OVERLAY"])
    tui.set_focus(editor)

    handle = tui.show_overlay(overlay, OverlayOptions(non_capturing=True))
    handle.set_hidden(True)
    handle.focus()

    assert editor.focused is True
    assert handle.is_focused() is False


def test_non_capturing_overlay_focus_on_hidden_overlay_is_a_no_op_exact_title_with_article_like_ts() -> None:
    terminal = _TerminalStub()
    tui = TUI(terminal)
    editor = _FocusableComponent(["EDITOR"])
    overlay = _FocusableComponent(["OVERLAY"])
    tui.set_focus(editor)

    handle = tui.show_overlay(overlay, OverlayOptions(non_capturing=True))
    handle.set_hidden(True)
    handle.focus()

    assert editor.focused is True
    assert handle.is_focused() is False


def test_focus_after_hide_is_noop() -> None:
    terminal = _TerminalStub()
    tui = TUI(terminal)
    editor = _FocusableComponent(["EDITOR"])
    overlay = _FocusableComponent(["OVERLAY"])
    tui.set_focus(editor)

    handle = tui.show_overlay(overlay, OverlayOptions(non_capturing=True))
    handle.hide()
    handle.focus()

    assert editor.focused is True
    assert handle.is_focused() is False


def test_non_capturing_overlay_focus_after_hide_is_a_noop_like_ts() -> None:
    terminal = _TerminalStub()
    tui = TUI(terminal)
    editor = _FocusableComponent(["EDITOR"])
    overlay = _FocusableComponent(["OVERLAY"])
    tui.set_focus(editor)

    handle = tui.show_overlay(overlay, OverlayOptions(non_capturing=True))
    handle.hide()
    handle.focus()

    assert editor.focused is True
    assert handle.is_focused() is False


def test_non_capturing_overlay_focus_after_hide_is_a_noop_exact_title_like_ts() -> None:
    terminal = _TerminalStub()
    tui = TUI(terminal)
    editor = _FocusableComponent(["EDITOR"])
    overlay = _FocusableComponent(["OVERLAY"])
    tui.set_focus(editor)

    handle = tui.show_overlay(overlay, OverlayOptions(non_capturing=True))
    handle.hide()
    handle.focus()

    assert editor.focused is True
    assert handle.is_focused() is False


def test_non_capturing_overlay_focus_after_hide_with_focus_after_hide_title_like_ts() -> None:
    terminal = _TerminalStub()
    tui = TUI(terminal)
    editor = _FocusableComponent(["EDITOR"])
    overlay = _FocusableComponent(["OVERLAY"])
    tui.set_focus(editor)

    handle = tui.show_overlay(overlay, OverlayOptions(non_capturing=True))
    handle.hide()
    handle.focus()

    assert editor.focused is True
    assert handle.is_focused() is False


def test_non_capturing_overlay_focus_after_hide_is_no_op_exact_title_like_ts() -> None:
    terminal = _TerminalStub()
    tui = TUI(terminal)
    editor = _FocusableComponent(["EDITOR"])
    overlay = _FocusableComponent(["OVERLAY"])
    tui.set_focus(editor)

    handle = tui.show_overlay(overlay, OverlayOptions(non_capturing=True))
    handle.hide()
    handle.focus()

    assert editor.focused is True
    assert handle.is_focused() is False


def test_non_capturing_overlay_focus_after_hide_is_a_no_op_exact_title_with_article_like_ts() -> None:
    terminal = _TerminalStub()
    tui = TUI(terminal)
    editor = _FocusableComponent(["EDITOR"])
    overlay = _FocusableComponent(["OVERLAY"])
    tui.set_focus(editor)

    handle = tui.show_overlay(overlay, OverlayOptions(non_capturing=True))
    handle.hide()
    handle.focus()

    assert editor.focused is True
    assert handle.is_focused() is False


def test_unfocus_when_overlay_does_not_have_focus_is_noop() -> None:
    terminal = _TerminalStub()
    tui = TUI(terminal)
    editor = _FocusableComponent(["EDITOR"])
    overlay = _FocusableComponent(["OVERLAY"])
    tui.set_focus(editor)

    handle = tui.show_overlay(overlay, OverlayOptions(non_capturing=True))
    handle.unfocus()

    assert editor.focused is True
    assert overlay.focused is False


def test_non_capturing_overlay_unfocus_when_overlay_does_not_have_focus_is_a_noop_like_ts() -> None:
    terminal = _TerminalStub()
    tui = TUI(terminal)
    editor = _FocusableComponent(["EDITOR"])
    overlay = _FocusableComponent(["OVERLAY"])
    tui.set_focus(editor)

    handle = tui.show_overlay(overlay, OverlayOptions(non_capturing=True))
    handle.unfocus()

    assert editor.focused is True
    assert overlay.focused is False


def test_non_capturing_overlay_unfocus_when_overlay_does_not_have_focus_is_a_noop_exact_title_like_ts() -> None:
    terminal = _TerminalStub()
    tui = TUI(terminal)
    editor = _FocusableComponent(["EDITOR"])
    overlay = _FocusableComponent(["OVERLAY"])
    tui.set_focus(editor)

    handle = tui.show_overlay(overlay, OverlayOptions(non_capturing=True))
    handle.unfocus()

    assert editor.focused is True
    assert overlay.focused is False


def test_non_capturing_overlay_unfocus_when_overlay_does_not_have_focus_with_unfocus_title_like_ts() -> None:
    terminal = _TerminalStub()
    tui = TUI(terminal)
    editor = _FocusableComponent(["EDITOR"])
    overlay = _FocusableComponent(["OVERLAY"])
    tui.set_focus(editor)

    handle = tui.show_overlay(overlay, OverlayOptions(non_capturing=True))
    handle.unfocus()

    assert editor.focused is True
    assert overlay.focused is False


def test_non_capturing_overlay_unfocus_when_overlay_does_not_have_focus_is_no_op_exact_title_like_ts() -> None:
    terminal = _TerminalStub()
    tui = TUI(terminal)
    editor = _FocusableComponent(["EDITOR"])
    overlay = _FocusableComponent(["OVERLAY"])
    tui.set_focus(editor)

    handle = tui.show_overlay(overlay, OverlayOptions(non_capturing=True))
    handle.unfocus()

    assert editor.focused is True
    assert overlay.focused is False


def test_non_capturing_overlay_unfocus_when_overlay_does_not_have_focus_is_a_no_op_exact_title_with_article_like_ts() -> None:
    terminal = _TerminalStub()
    tui = TUI(terminal)
    editor = _FocusableComponent(["EDITOR"])
    overlay = _FocusableComponent(["OVERLAY"])
    tui.set_focus(editor)

    handle = tui.show_overlay(overlay, OverlayOptions(non_capturing=True))
    handle.unfocus()

    assert editor.focused is True
    assert overlay.focused is False


def test_default_overlapping_overlay_render_order_follows_creation_order() -> None:
    terminal = _TerminalStub(columns=20, rows=6)
    tui = TUI(terminal)
    tui.start()

    tui.show_overlay(_StaticComponent(["A"]), OverlayOptions(row=0, col=0, width=1, non_capturing=True))
    tui.show_overlay(_StaticComponent(["B"]), OverlayOptions(row=0, col=0, width=1, non_capturing=True))

    assert "B" in tui.previous_lines[0]


def test_non_capturing_overlay_default_rendering_order_for_overlapping_overlays_follows_creation_order_like_ts() -> None:
    terminal = _TerminalStub(columns=20, rows=6)
    tui = TUI(terminal)
    tui.start()

    tui.show_overlay(_StaticComponent(["A"]), OverlayOptions(row=0, col=0, width=1, non_capturing=True))
    tui.show_overlay(_StaticComponent(["B"]), OverlayOptions(row=0, col=0, width=1, non_capturing=True))

    assert "B" in tui.previous_lines[0]


def test_focusing_lower_overlay_brings_it_to_top_visually() -> None:
    terminal = _TerminalStub(columns=20, rows=6)
    tui = TUI(terminal)
    tui.start()

    lower = tui.show_overlay(_StaticComponent(["A"]), OverlayOptions(row=0, col=0, width=1, non_capturing=True))
    tui.show_overlay(_StaticComponent(["B"]), OverlayOptions(row=0, col=0, width=1, non_capturing=True))
    assert "B" in tui.previous_lines[0]

    lower.focus()

    assert "A" in tui.previous_lines[0]


def test_non_capturing_overlay_focus_on_lower_overlay_renders_it_on_top_like_ts() -> None:
    terminal = _TerminalStub(columns=20, rows=6)
    tui = TUI(terminal)
    tui.start()

    lower = tui.show_overlay(_StaticComponent(["A"]), OverlayOptions(row=0, col=0, width=1, non_capturing=True))
    tui.show_overlay(_StaticComponent(["B"]), OverlayOptions(row=0, col=0, width=1, non_capturing=True))
    lower.focus()

    assert "A" in tui.previous_lines[0]


def test_unfocus_does_not_change_visual_order_until_another_overlay_is_focused() -> None:
    terminal = _TerminalStub(columns=20, rows=6)
    tui = TUI(terminal)
    editor = _FocusableComponent(["EDITOR"])
    tui.set_focus(editor)
    tui.start()

    overlay_a = tui.show_overlay(_StaticComponent(["A"]), OverlayOptions(row=0, col=0, width=1, non_capturing=True))
    overlay_b = tui.show_overlay(_StaticComponent(["B"]), OverlayOptions(row=0, col=0, width=1, non_capturing=True))
    assert "B" in tui.previous_lines[0]

    overlay_a.focus()
    assert "A" in tui.previous_lines[0]

    overlay_a.unfocus()
    assert "A" in tui.previous_lines[0]

    overlay_b.focus()
    assert "B" in tui.previous_lines[0]


def test_focusing_already_focused_overlay_bumps_visual_order() -> None:
    terminal = _TerminalStub(columns=20, rows=6)
    tui = TUI(terminal)
    editor = _FocusableComponent(["EDITOR"])
    tui.set_focus(editor)
    tui.start()

    overlay_a = tui.show_overlay(_StaticComponent(["A"]), OverlayOptions(row=0, col=0, width=1, non_capturing=True))
    tui.show_overlay(_StaticComponent(["B"]), OverlayOptions(row=0, col=0, width=1, non_capturing=True))
    overlay_a.focus()
    tui.show_overlay(_StaticComponent(["C"]), OverlayOptions(row=0, col=0, width=1, non_capturing=True))

    assert "C" in tui.previous_lines[0]

    overlay_a.focus()

    assert "A" in tui.previous_lines[0]
    assert overlay_a.is_focused() is True


def test_non_capturing_overlay_focus_on_already_focused_overlay_bumps_visual_order_like_ts() -> None:
    terminal = _TerminalStub(columns=20, rows=6)
    tui = TUI(terminal)
    editor = _FocusableComponent(["EDITOR"])
    tui.set_focus(editor)
    tui.start()

    overlay_a = tui.show_overlay(_StaticComponent(["A"]), OverlayOptions(row=0, col=0, width=1, non_capturing=True))
    tui.show_overlay(_StaticComponent(["B"]), OverlayOptions(row=0, col=0, width=1, non_capturing=True))
    overlay_a.focus()
    tui.show_overlay(_StaticComponent(["C"]), OverlayOptions(row=0, col=0, width=1, non_capturing=True))
    overlay_a.focus()

    assert "A" in tui.previous_lines[0]


def test_focusing_middle_overlay_places_it_on_top_while_preserving_other_order() -> None:
    terminal = _TerminalStub(columns=20, rows=6)
    tui = TUI(terminal)
    tui.start()

    tui.show_overlay(_StaticComponent(["A"]), OverlayOptions(row=0, col=0, width=1, non_capturing=True))
    middle = tui.show_overlay(_StaticComponent(["B"]), OverlayOptions(row=0, col=0, width=1, non_capturing=True))
    top = tui.show_overlay(_StaticComponent(["C"]), OverlayOptions(row=0, col=0, width=1, non_capturing=True))
    assert "C" in tui.previous_lines[0]

    middle.focus()
    assert "B" in tui.previous_lines[0]

    middle.hide()
    assert "C" in tui.previous_lines[0]

    top.hide()
    assert "A" in tui.previous_lines[0]


def test_non_capturing_overlay_focusing_middle_overlay_places_it_on_top_while_preserving_others_relative_order_like_ts() -> None:
    terminal = _TerminalStub(columns=20, rows=6)
    tui = TUI(terminal)
    tui.start()

    tui.show_overlay(_StaticComponent(["A"]), OverlayOptions(row=0, col=0, width=1, non_capturing=True))
    middle = tui.show_overlay(_StaticComponent(["B"]), OverlayOptions(row=0, col=0, width=1, non_capturing=True))
    top = tui.show_overlay(_StaticComponent(["C"]), OverlayOptions(row=0, col=0, width=1, non_capturing=True))
    middle.focus()
    assert "B" in tui.previous_lines[0]

    middle.hide()
    assert "C" in tui.previous_lines[0]
    top.hide()
    assert "A" in tui.previous_lines[0]


def test_capturing_overlay_unhidden_renders_on_top_again() -> None:
    terminal = _TerminalStub(columns=20, rows=6)
    tui = TUI(terminal)
    tui.start()

    tui.show_overlay(_StaticComponent(["A"]), OverlayOptions(row=0, col=0, width=1, non_capturing=True))
    capturing = tui.show_overlay(_StaticComponent(["B"]), OverlayOptions(row=0, col=0, width=1))
    assert "B" in tui.previous_lines[0]

    capturing.set_hidden(True)
    tui.show_overlay(_StaticComponent(["C"]), OverlayOptions(row=0, col=0, width=1, non_capturing=True))
    assert "C" in tui.previous_lines[0]

    capturing.set_hidden(False)

    assert "B" in tui.previous_lines[0]


def test_non_capturing_overlay_capturing_overlay_hidden_and_shown_again_renders_on_top_after_unhide_like_ts() -> None:
    terminal = _TerminalStub(columns=20, rows=6)
    tui = TUI(terminal)
    tui.start()

    tui.show_overlay(_StaticComponent(["A"]), OverlayOptions(row=0, col=0, width=1, non_capturing=True))
    capturing = tui.show_overlay(_StaticComponent(["B"]), OverlayOptions(row=0, col=0, width=1))
    capturing.set_hidden(True)
    tui.show_overlay(_StaticComponent(["C"]), OverlayOptions(row=0, col=0, width=1, non_capturing=True))
    capturing.set_hidden(False)

    assert "B" in tui.previous_lines[0]


def test_tui_appends_line_reset_after_each_rendered_line() -> None:
    terminal = _TerminalStub(columns=20, rows=6)
    tui = TUI(terminal)
    tui.add_child(_StaticComponent(["\x1b[3mItalic", "Plain"]))

    tui.start()

    buffer = next(write for write in terminal.writes if "Italic" in write)
    assert "\x1b[3mItalic\x1b[0m\x1b]8;;\x07\r\nPlain\x1b[0m\x1b]8;;\x07" in buffer


def test_tui_resets_styles_after_each_rendered_line_like_ts() -> None:
    """resets styles after each rendered line"""
    terminal = _TerminalStub(columns=20, rows=6)
    tui = TUI(terminal)
    tui.add_child(_StaticComponent(["\x1b[3mItalic", "Plain"]))

    tui.start()

    buffer = next(write for write in terminal.writes if "Italic" in write)
    assert "\x1b[3mItalic\x1b[0m\x1b]8;;\x07\r\nPlain\x1b[0m\x1b]8;;\x07" in buffer


def test_tui_renders_correctly_when_multiple_non_adjacent_lines_change() -> None:
    """renders correctly when multiple non-adjacent lines change"""
    terminal = _TerminalStub(columns=40, rows=10)
    tui = TUI(terminal)
    body = _StaticComponent(["Line 0", "Line 1", "Line 2", "Line 3", "Line 4"])

    tui.add_child(body)
    tui.start()

    body.lines = ["Line 0", "CHANGED 1", "Line 2", "CHANGED 3", "Line 4"]
    tui.request_render()

    assert "Line 0" in tui.previous_lines[0]
    assert "CHANGED 1" in tui.previous_lines[1]
    assert "Line 2" in tui.previous_lines[2]
    assert "CHANGED 3" in tui.previous_lines[3]
    assert "Line 4" in tui.previous_lines[4]


def test_tui_tracks_cursor_after_shrink_with_unchanged_prefix_lines() -> None:
    terminal = _TerminalStub(columns=40, rows=10)
    tui = TUI(terminal)
    body = _StaticComponent(["Line 0", "Line 1", "Line 2", "Line 3", "Line 4"])

    tui.add_child(body)
    tui.start()

    body.lines = ["Line 0", "Line 1", "Line 2"]
    tui.request_render()
    body.lines = ["Line 0", "CHANGED", "Line 2"]
    tui.request_render()

    assert "Line 0" in tui.previous_lines[0]
    assert "CHANGED" in tui.previous_lines[1]
    assert "Line 2" in tui.previous_lines[2]


def test_tui_tracks_cursor_correctly_when_content_shrinks_with_unchanged_remaining_lines_like_ts() -> None:
    """tracks cursor correctly when content shrinks with unchanged remaining lines"""
    terminal = _TerminalStub(columns=40, rows=10)
    tui = TUI(terminal)
    body = _StaticComponent(["Line 0", "Line 1", "Line 2", "Line 3", "Line 4"])

    tui.add_child(body)
    tui.start()

    body.lines = ["Line 0", "Line 1", "Line 2"]
    tui.request_render()
    body.lines = ["Line 0", "CHANGED", "Line 2"]
    tui.request_render()

    assert "Line 0" in tui.previous_lines[0]
    assert "CHANGED" in tui.previous_lines[1]
    assert "Line 2" in tui.previous_lines[2]


def test_tui_renders_correctly_when_only_middle_line_changes() -> None:
    terminal = _TerminalStub(columns=40, rows=10)
    tui = TUI(terminal)
    body = _StaticComponent(["Header", "Working...", "Footer"])

    tui.add_child(body)
    tui.start()

    for frame in ["|", "/", "-", "\\"]:
        body.lines = ["Header", f"Working {frame}", "Footer"]
        tui.request_render()
        assert "Header" in tui.previous_lines[0]
        assert f"Working {frame}" in tui.previous_lines[1]
        assert "Footer" in tui.previous_lines[2]


def test_tui_renders_correctly_when_only_a_middle_line_changes_spinner_case_like_ts() -> None:
    """renders correctly when only a middle line changes (spinner case)"""
    terminal = _TerminalStub(columns=40, rows=10)
    tui = TUI(terminal)
    body = _StaticComponent(["Header", "Working...", "Footer"])

    tui.add_child(body)
    tui.start()

    for frame in ["|", "/", "-", "\\"]:
        body.lines = ["Header", f"Working {frame}", "Footer"]
        tui.request_render()
        assert "Header" in tui.previous_lines[0]
        assert f"Working {frame}" in tui.previous_lines[1]
        assert "Footer" in tui.previous_lines[2]


def test_tui_middle_line_diff_render_does_not_trigger_full_redraw() -> None:
    terminal = _TerminalStub(columns=40, rows=10)
    tui = TUI(terminal)
    body = _StaticComponent(["Header", "Working...", "Footer"])

    tui.add_child(body)
    tui.start()
    initial_redraws = tui.full_redraws

    body.lines = ["Header", "Working /", "Footer"]
    tui.request_render()

    assert tui.full_redraws == initial_redraws
    assert all("\x1b[2J\x1b[H\x1b[3J" not in write for write in terminal.writes[1:])


@pytest.mark.anyio
async def test_tui_request_render_coalesces_within_one_event_loop_tick() -> None:
    terminal = _TerminalStub(columns=40, rows=10)
    tui = TUI(terminal)
    body = _CountingComponent(["Line 0"])

    tui.add_child(body)
    tui.start()
    await asyncio.sleep(0)

    initial_calls = body.render_calls
    terminal.writes.clear()
    body.lines = ["Line 1"]

    tui.request_render()
    tui.request_render()
    tui.request_render()

    assert tui.render_requested is True
    assert body.render_calls == initial_calls
    assert terminal.writes == []

    await asyncio.sleep(0)

    assert tui.render_requested is False
    assert body.render_calls == initial_calls + 1
    assert "Line 1" in tui.previous_lines[0]


@pytest.mark.anyio
async def test_tui_force_request_render_still_coalesces_to_one_scheduled_redraw() -> None:
    terminal = _TerminalStub(columns=40, rows=10)
    tui = TUI(terminal)
    body = _CountingComponent(["Line 0", "Line 1"])

    tui.add_child(body)
    tui.start()
    await asyncio.sleep(0)

    initial_calls = body.render_calls
    initial_redraws = tui.full_redraws
    terminal.writes.clear()
    body.lines = ["Changed"]

    tui.request_render(force=True)
    tui.request_render()

    assert tui.render_requested is True
    assert body.render_calls == initial_calls

    await asyncio.sleep(0)

    assert body.render_calls == initial_calls + 1
    assert tui.full_redraws > initial_redraws
    assert any("\x1b[2J\x1b[H\x1b[3J" in write for write in terminal.writes)
    assert tui.previous_lines == ["Changed\x1b[0m\x1b]8;;\x07"]


def test_tui_clear_on_shrink_to_empty_clears_all_lines() -> None:
    terminal = _TerminalStub(columns=40, rows=10)
    tui = TUI(terminal)
    body = _StaticComponent(["Line 0", "Line 1", "Line 2"])

    tui.set_clear_on_shrink(True)
    tui.add_child(body)
    tui.start()

    body.lines = []
    tui.request_render()

    assert tui.previous_lines == []
    assert any("\x1b[2J\x1b[H\x1b[3J" in write for write in terminal.writes)


def test_tui_handles_shrink_to_empty_like_ts() -> None:
    """handles shrink to empty"""
    terminal = _TerminalStub(columns=40, rows=10)
    tui = TUI(terminal)
    body = _StaticComponent(["Line 0", "Line 1", "Line 2"])

    tui.set_clear_on_shrink(True)
    tui.add_child(body)
    tui.start()

    body.lines = []
    tui.request_render()

    assert tui.previous_lines == []


def test_tui_handles_transition_from_content_to_empty_and_back_to_content() -> None:
    """handles transition from content to empty and back to content"""
    terminal = _TerminalStub(columns=40, rows=10)
    tui = TUI(terminal)
    body = _StaticComponent(["Line 0", "Line 1", "Line 2"])

    tui.add_child(body)
    tui.start()

    assert "Line 0" in tui.previous_lines[0]
    assert "Line 1" in tui.previous_lines[1]

    body.lines = []
    tui.request_render()

    assert tui.previous_lines == []

    body.lines = ["New Line 0", "New Line 1"]
    tui.request_render()

    assert tui.previous_lines == [
        "New Line 0\x1b[0m\x1b]8;;\x07",
        "New Line 1\x1b[0m\x1b]8;;\x07",
    ]


def test_tui_renders_correctly_when_first_line_changes() -> None:
    terminal = _TerminalStub(columns=40, rows=10)
    tui = TUI(terminal)
    body = _StaticComponent(["Line 0", "Line 1", "Line 2", "Line 3"])

    tui.add_child(body)
    tui.start()
    body.lines = ["CHANGED", "Line 1", "Line 2", "Line 3"]
    tui.request_render()

    assert "CHANGED" in tui.previous_lines[0]
    assert "Line 1" in tui.previous_lines[1]
    assert "Line 2" in tui.previous_lines[2]
    assert "Line 3" in tui.previous_lines[3]


def test_tui_renders_correctly_when_first_line_changes_but_rest_stays_same_like_ts() -> None:
    """renders correctly when first line changes but rest stays same"""
    terminal = _TerminalStub(columns=40, rows=10)
    tui = TUI(terminal)
    body = _StaticComponent(["Line 0", "Line 1", "Line 2", "Line 3"])

    tui.add_child(body)
    tui.start()
    body.lines = ["CHANGED", "Line 1", "Line 2", "Line 3"]
    tui.request_render()

    assert "CHANGED" in tui.previous_lines[0]
    assert "Line 1" in tui.previous_lines[1]
    assert "Line 2" in tui.previous_lines[2]
    assert "Line 3" in tui.previous_lines[3]


def test_tui_renders_correctly_when_last_line_changes() -> None:
    terminal = _TerminalStub(columns=40, rows=10)
    tui = TUI(terminal)
    body = _StaticComponent(["Line 0", "Line 1", "Line 2", "Line 3"])

    tui.add_child(body)
    tui.start()
    body.lines = ["Line 0", "Line 1", "Line 2", "CHANGED"]
    tui.request_render()

    assert "Line 0" in tui.previous_lines[0]
    assert "Line 1" in tui.previous_lines[1]
    assert "Line 2" in tui.previous_lines[2]
    assert "CHANGED" in tui.previous_lines[3]


def test_tui_renders_correctly_when_last_line_changes_but_rest_stays_same_like_ts() -> None:
    """renders correctly when last line changes but rest stays same"""
    terminal = _TerminalStub(columns=40, rows=10)
    tui = TUI(terminal)
    body = _StaticComponent(["Line 0", "Line 1", "Line 2", "Line 3"])

    tui.add_child(body)
    tui.start()
    body.lines = ["Line 0", "Line 1", "Line 2", "CHANGED"]
    tui.request_render()

    assert "Line 0" in tui.previous_lines[0]
    assert "Line 1" in tui.previous_lines[1]
    assert "Line 2" in tui.previous_lines[2]
    assert "CHANGED" in tui.previous_lines[3]


def test_tui_resize_still_triggers_full_redraw_and_clear_sequence() -> None:
    terminal = _TerminalStub(columns=40, rows=10)
    tui = TUI(terminal)
    body = _StaticComponent(["Line 0", "Line 1", "Line 2"])

    tui.add_child(body)
    tui.start()
    initial_redraws = tui.full_redraws

    terminal.columns = 60
    tui.request_render()

    assert tui.full_redraws > initial_redraws
    assert any("\x1b[2J\x1b[H\x1b[3J" in write for write in terminal.writes)


def test_tui_appending_beyond_viewport_preserves_preexisting_lines() -> None:
    terminal = _TerminalStub(columns=20, rows=8)
    tui = TUI(terminal)
    body = _StaticComponent(["HEAD", "=== PRE ==="])
    tui.add_child(body)
    tui.start()

    for i in range(11):
        body.lines.append(f"PRE {i:02d}")
        tui.request_render()

    body.lines.extend(["", "=== POST ==="])
    tui.request_render()
    for i in range(5):
        body.lines.append(f"POST {i:02d}")
        tui.request_render()

    viewport = [line.replace("\x1b[0m\x1b]8;;\x07", "").rstrip() for line in tui.previous_lines[-terminal.rows :]]
    assert "PRE 10" in viewport[0]
    assert "=== POST ===" in viewport[2]
    assert viewport[-1] == "POST 04"


def test_tui_parses_cell_size_response_updates_dimensions_and_invalidates(monkeypatch) -> None:  # noqa: ANN001
    terminal = _TerminalStub(columns=40, rows=10)
    tui = TUI(terminal)
    body = _InvalidateTrackingComponent(["body"])

    monkeypatch.setattr("paw.pi_agent.tui.tui.get_capabilities", lambda: type("Caps", (), {"images": "kitty"})())
    previous_dims = get_cell_dimensions()
    set_cell_dimensions(CellDimensions(width_px=9, height_px=18))

    try:
        tui.add_child(body)
        tui.start()

        assert "\x1b[16t" in terminal.writes
        assert tui.cell_size_query_pending is True

        terminal.on_input("\x1b[6;42;17t")

        assert get_cell_dimensions() == CellDimensions(width_px=17, height_px=42)
        assert body.invalidated >= 1
        assert tui.cell_size_query_pending is False
    finally:
        set_cell_dimensions(previous_dims)


def test_tui_filters_cell_size_response_and_forwards_remaining_input(monkeypatch) -> None:  # noqa: ANN001
    terminal = _TerminalStub(columns=40, rows=10)
    tui = TUI(terminal)
    body = _InvalidateTrackingComponent(["body"])

    monkeypatch.setattr("paw.pi_agent.tui.tui.get_capabilities", lambda: type("Caps", (), {"images": "kitty"})())
    previous_dims = get_cell_dimensions()
    set_cell_dimensions(CellDimensions(width_px=9, height_px=18))

    try:
        tui.add_child(body)
        tui.set_focus(body)
        tui.start()

        terminal.on_input("\x1b[6;30;15ta")

        assert body.inputs == ["a"]
        assert get_cell_dimensions() == CellDimensions(width_px=15, height_px=30)
        assert tui.input_buffer == ""
    finally:
        set_cell_dimensions(previous_dims)


def test_tui_input_listener_can_consume_via_ts_style_result_object() -> None:
    terminal = _TerminalStub(columns=40, rows=10)
    tui = TUI(terminal)
    body = _FocusableComponent(["body"])
    tui.add_child(body)
    tui.set_focus(body)
    tui.start()

    tui.add_input_listener(lambda _data: {"consume": True})
    terminal.on_input("x")

    assert body.inputs == []


def test_tui_input_listener_can_replace_data_via_ts_style_result_object() -> None:
    terminal = _TerminalStub(columns=40, rows=10)
    tui = TUI(terminal)
    body = _FocusableComponent(["body"])
    tui.add_child(body)
    tui.set_focus(body)
    tui.start()

    tui.add_input_listener(lambda _data: {"data": "y"})
    terminal.on_input("x")

    assert body.inputs == ["y"]


def test_tui_input_listener_empty_replacement_short_circuits_routing_like_ts() -> None:
    terminal = _TerminalStub(columns=40, rows=10)
    tui = TUI(terminal)
    body = _FocusableComponent(["body"])
    tui.add_child(body)
    tui.set_focus(body)
    tui.start()

    tui.add_input_listener(lambda _data: {"data": ""})
    terminal.on_input("x")

    assert body.inputs == []


def test_tui_filters_key_release_events_by_default() -> None:
    terminal = _TerminalStub(columns=40, rows=10)
    tui = TUI(terminal)
    body = _FocusableComponent(["body"])
    tui.add_child(body)
    tui.set_focus(body)
    tui.start()

    terminal.on_input("\x1b[1089::99;5:3u")

    assert body.inputs == []


def test_tui_allows_key_release_events_for_opted_in_components() -> None:
    terminal = _TerminalStub(columns=40, rows=10)
    tui = TUI(terminal)
    body = _KeyReleaseFocusableComponent(["body"])
    tui.add_child(body)
    tui.set_focus(body)
    tui.start()

    terminal.on_input("\x1b[1089::99;5:3u")

    assert body.inputs == ["\x1b[1089::99;5:3u"]
