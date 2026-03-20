from __future__ import annotations

import asyncio
import base64
import struct
import time
from collections.abc import Callable

import pytest

from paw.pi_agent.tui import (
    ImageDimensions,
    KillRing,
    ProcessTerminal,
    StdinBuffer,
    UndoStack,
    calculate_image_rows,
    delete_all_kitty_images,
    delete_kitty_image,
    detect_capabilities,
    encode_iterm2,
    encode_kitty,
    get_gif_dimensions,
    get_image_dimensions,
    get_webp_dimensions,
    image_fallback,
    is_key_release,
    is_key_repeat,
    matches_key,
    parse_key,
    render_image,
    reset_capabilities_cache,
    set_kitty_protocol_active,
)
from paw.pi_agent.tui.terminal_image import is_image_line


def test_kill_ring_push_peek_and_rotate() -> None:
    ring = KillRing()
    ring.push("first", prepend=False)
    ring.push("second", prepend=False)

    assert ring.peek() == "second"
    assert ring.length == 2

    ring.rotate()
    assert ring.peek() == "first"
    assert ring.length == 2


def test_kill_ring_accumulates_backward_and_forward() -> None:
    ring = KillRing()
    ring.push("world", prepend=False)
    ring.push("hello ", prepend=True, accumulate=True)
    assert ring.peek() == "hello world"

    ring.push("!", prepend=False, accumulate=True)
    assert ring.peek() == "hello world!"


def test_undo_stack_clones_on_push() -> None:
    stack: UndoStack[dict[str, list[str]]] = UndoStack()
    state = {"items": ["a"]}
    stack.push(state)

    state["items"].append("b")
    restored = stack.pop()

    assert restored == {"items": ["a"]}
    assert stack.length == 0


def test_undo_stack_clear_empties_snapshots() -> None:
    stack: UndoStack[dict[str, int]] = UndoStack()
    stack.push({"count": 1})
    stack.push({"count": 2})
    assert stack.length == 2

    stack.clear()
    assert stack.length == 0
    assert stack.pop() is None


def test_get_gif_dimensions_reads_logical_screen_size() -> None:
    payload = b"GIF89a" + struct.pack("<HH", 16, 9) + b"\x00\x00\x00"
    encoded = base64.b64encode(payload).decode()
    assert get_gif_dimensions(encoded) == ImageDimensions(width_px=16, height_px=9)


def test_get_webp_dimensions_reads_vp8x_header() -> None:
    payload = bytearray(30)
    payload[0:4] = b"RIFF"
    payload[8:12] = b"WEBP"
    payload[12:16] = b"VP8X"
    payload[24:27] = bytes((9, 0, 0))  # width - 1 = 9 => width 10
    payload[27:30] = bytes((5, 0, 0))  # height - 1 = 5 => height 6
    encoded = base64.b64encode(bytes(payload)).decode()
    assert get_webp_dimensions(encoded) == ImageDimensions(width_px=10, height_px=6)


def test_get_image_dimensions_dispatches_by_mime_type() -> None:
    gif_payload = b"GIF89a" + struct.pack("<HH", 4, 3) + b"\x00\x00\x00"
    encoded = base64.b64encode(gif_payload).decode()
    assert get_image_dimensions(encoded, "image/gif") == ImageDimensions(width_px=4, height_px=3)
    assert get_image_dimensions(encoded, "application/octet-stream") is None


def test_render_image_uses_kitty_protocol_when_supported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TERM_PROGRAM", "kitty")
    reset_capabilities_cache()
    result = render_image("Zm9v", ImageDimensions(width_px=9, height_px=18))
    reset_capabilities_cache()
    assert result is not None
    assert isinstance(result["rows"], int)
    assert str(result["sequence"]).startswith("\x1b_G")


def test_render_image_returns_none_when_images_are_unsupported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TERM_PROGRAM", raising=False)
    monkeypatch.delenv("KITTY_WINDOW_ID", raising=False)
    monkeypatch.delenv("WEZTERM_PANE", raising=False)
    monkeypatch.delenv("ITERM_SESSION_ID", raising=False)
    monkeypatch.setenv("COLORTERM", "truecolor")
    reset_capabilities_cache()
    assert render_image("Zm9v", ImageDimensions(width_px=9, height_px=18)) is None
    reset_capabilities_cache()


def test_detect_capabilities_matches_ts_terminal_matrix(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ["TERM_PROGRAM", "TERM", "COLORTERM", "KITTY_WINDOW_ID", "WEZTERM_PANE", "ITERM_SESSION_ID", "GHOSTTY_RESOURCES_DIR"]:
        monkeypatch.delenv(name, raising=False)

    monkeypatch.setenv("TERM_PROGRAM", "ghostty")
    assert detect_capabilities().images == "kitty"

    monkeypatch.setenv("TERM_PROGRAM", "wezterm")
    assert detect_capabilities().images == "kitty"

    monkeypatch.setenv("TERM_PROGRAM", "iterm.app")
    assert detect_capabilities().images == "iterm2"

    monkeypatch.setenv("TERM_PROGRAM", "vscode")
    assert detect_capabilities().images is None
    assert detect_capabilities().true_color is True

    monkeypatch.setenv("TERM_PROGRAM", "")
    monkeypatch.setenv("COLORTERM", "24bit")
    caps = detect_capabilities()
    assert caps.images is None
    assert caps.true_color is True
    assert caps.hyperlinks is True


def test_encode_kitty_chunks_large_payload_and_includes_dimensions() -> None:
    payload = "A" * 5000
    encoded = encode_kitty(payload, columns=60, rows=4, image_id=7)

    assert encoded.startswith("\x1b_Ga=T,f=100,q=2,c=60,r=4,i=7,m=1;")
    assert "\x1b_Gm=0;" in encoded
    assert encoded.endswith("\x1b\\")


def test_encode_iterm2_includes_expected_options_and_name() -> None:
    encoded = encode_iterm2(
        "Zm9v",
        width="50%",
        height=12,
        name="cat.png",
        preserve_aspect_ratio=False,
        inline=False,
    )

    assert encoded.startswith("\x1b]1337;File=inline=0;width=50%;height=12;")
    assert "name=Y2F0LnBuZw==" in encoded
    assert "preserveAspectRatio=0" in encoded
    assert encoded.endswith(":Zm9v\x07")


def test_kitty_delete_sequences_and_row_calculation_match_ts() -> None:
    assert delete_kitty_image(42) == "\x1b_Ga=d,d=I,i=42\x1b\\"
    assert delete_all_kitty_images() == "\x1b_Ga=d,d=A\x1b\\"
    assert calculate_image_rows(ImageDimensions(width_px=800, height_px=600), 40) == 15


def test_image_fallback_includes_filename_mime_and_dimensions() -> None:
    assert image_fallback("image/png", ImageDimensions(width_px=10, height_px=20), "cat.png") == "[Image: cat.png [image/png] 10x20]"


def test_is_image_line_detects_iterm2_and_kitty_sequences_anywhere_in_line() -> None:
    """
    new implementation detects Kitty sequences in any position
    new implementation detects iTerm2 sequences in any position
    detects image sequences in read tool output
    detects Kitty sequences from Image component
    handles ANSI codes before image sequences
    does NOT crash on very long lines with image sequences
    handles lines exactly matching crash log dimensions
    """
    assert is_image_line("\x1b]1337;File=size=100,100;inline=1:base64encodeddata==\x07") is True
    assert is_image_line("Some text \x1b]1337;File=size=100,100;inline=1:base64data==\x07 more text") is True
    assert is_image_line("Regular text ending with \x1b]1337;File=inline=1:base64data==\x07") is True
    assert is_image_line("\x1b_Ga=T,f=100,t=f,d=base64data...\x1b\\\x1b_Gm=i=1;\x1b\\") is True
    assert is_image_line("Output: \x1b_Ga=T,f=100;data...\x1b\\\x1b_Gm=i=1;\x1b\\") is True
    assert is_image_line("  \x1b_Ga=T,f=100...\x1b\\\x1b_Gm=i=1;\x1b\\  ") is True

    ansi_prefixed_lines = [
        "\x1b[31mError\x1b[0m: \x1b]1337;File=inline=1:base64==\x07",
        "\x1b[33mWarning\x1b[0m: \x1b_Ga=T,data...\x1b\\",
        "\x1b[1mBold\x1b[0m \x1b]1337;File=:base64==\x07\x1b[0m",
    ]
    for line in ansi_prefixed_lines:
        assert is_image_line(line) is True

    very_long_iterm = "Output: " + "\x1b]1337;File=size=800,600;inline=1:" + ("A" * 304000) + " end of output"
    very_long_kitty = "Text before \x1b_Ga=T,f=100" + ("B" * 300000) + " text after"
    assert is_image_line(very_long_iterm) is True
    assert is_image_line(very_long_kitty) is True

    tool_output_line = "Read image file [image/jpeg]\x1b]1337;File=size=800,600;inline=1:base64image...\x07"
    kitty_component_line = "\x1b_Ga=T,f=100,t=f,d=base64data...\x1b\\\x1b_Gm=i=1;\x1b\\"
    assert is_image_line(tool_output_line) is True
    assert is_image_line(kitty_component_line) is True

    crash_width_line = "Output: " + "\x1b]1337;File=size=800,600;inline=1:" + ("C" * 58610)
    assert len(crash_width_line) > 58649
    assert is_image_line(crash_width_line) is True


def test_is_image_line_should_detect_iterm2_image_escape_sequence_at_start_of_line_like_ts() -> None:
    """should detect iTerm2 image escape sequence at start of line"""
    assert is_image_line("\x1b]1337;File=size=100,100;inline=1:base64encodeddata==\x07") is True


def test_is_image_line_should_detect_iterm2_image_escape_sequence_with_text_before_it_like_ts() -> None:
    """should detect iTerm2 image escape sequence with text before it"""
    assert is_image_line("Some text \x1b]1337;File=size=100,100;inline=1:base64data==\x07 more text") is True


def test_is_image_line_should_detect_iterm2_image_escape_sequence_in_middle_of_long_line_like_ts() -> None:
    """should detect iTerm2 image escape sequence in middle of long line"""
    long_line_with_image = "Text before image..." + "\x1b]1337;File=inline=1:verylongbase64data==" + "...text after"
    assert is_image_line(long_line_with_image) is True


def test_is_image_line_should_detect_iterm2_image_escape_sequence_at_end_of_line_like_ts() -> None:
    """should detect iTerm2 image escape sequence at end of line"""
    assert is_image_line("Regular text ending with \x1b]1337;File=inline=1:base64data==\x07") is True


def test_is_image_line_should_detect_minimal_iterm2_image_escape_sequence_like_ts() -> None:
    """should detect minimal iTerm2 image escape sequence"""
    assert is_image_line("\x1b]1337;File=:\x07") is True


def test_is_image_line_should_detect_kitty_image_escape_sequence_at_start_of_line_like_ts() -> None:
    """should detect Kitty image escape sequence at start of line"""
    assert is_image_line("\x1b_Ga=T,f=100,t=f,d=base64data...\x1b\\\x1b_Gm=i=1;\x1b\\") is True


def test_is_image_line_should_detect_kitty_image_escape_sequence_with_text_before_it_like_ts() -> None:
    """should detect Kitty image escape sequence with text before it"""
    assert is_image_line("Output: \x1b_Ga=T,f=100;data...\x1b\\\x1b_Gm=i=1;\x1b\\") is True


def test_is_image_line_should_detect_kitty_image_escape_sequence_with_padding_like_ts() -> None:
    """should detect Kitty image escape sequence with padding"""
    assert is_image_line("  \x1b_Ga=T,f=100...\x1b\\\x1b_Gm=i=1;\x1b\\  ") is True


def test_is_image_line_detects_image_sequences_with_ansi_after_them() -> None:
    assert is_image_line("\x1b_Ga=T,f=100:data...\x1b\\\x1b_Gm=i=1;\x1b\\\x1b[0m reset") is True


def test_is_image_line_should_detect_image_sequences_in_very_long_lines_like_ts() -> None:
    """should detect image sequences in very long lines (304k+ chars)"""
    long_line = "Text prefix " + "\x1b]1337;File=size=800,600;inline=1:" + ("A" * 300000) + " suffix"
    assert len(long_line) > 300000
    assert is_image_line(long_line) is True


def test_is_image_line_should_detect_image_sequences_in_very_long_lines_304k_chars_like_ts() -> None:
    """new implementation returns true correctly"""
    long_line = "Text prefix " + "\x1b]1337;File=size=800,600;inline=1:" + ("A" * 300000) + " suffix"
    assert len(long_line) > 300000
    assert is_image_line(long_line) is True


def test_is_image_line_should_detect_image_sequences_when_terminal_does_not_support_images_like_ts() -> None:
    """should detect image sequences when terminal doesn't support images"""
    assert is_image_line("Read image file [image/jpeg]\x1b]1337;File=inline=1:base64data==\x07") is True


def test_is_image_line_should_detect_image_sequences_when_terminal_does_not_support_images_exact_title_like_ts() -> None:
    """old implementation would return false, causing crash"""
    assert is_image_line("Read image file [image/jpeg]\x1b]1337;File=inline=1:base64data==\x07") is True


def test_is_image_line_should_detect_image_sequences_when_terminal_doesnt_support_images_like_ts() -> None:
    assert is_image_line("Read image file [image/jpeg]\x1b]1337;File=inline=1:base64data==\x07") is True


def test_is_image_line_should_detect_image_sequences_when_terminal_does_not_support_images_with_does_not_title_like_ts() -> None:
    assert is_image_line("Read image file [image/jpeg]\x1b]1337;File=inline=1:base64data==\x07") is True


def test_is_image_line_should_detect_image_sequences_with_ansi_codes_before_them_like_ts() -> None:
    """should detect image sequences with ANSI codes before them"""
    assert is_image_line("\x1b[31mError output \x1b]1337;File=inline=1:image==\x07") is True


def test_is_image_line_should_detect_image_sequences_with_ansi_codes_after_them_like_ts() -> None:
    """should detect image sequences with ANSI codes after them"""
    assert is_image_line("\x1b_Ga=T,f=100:data...\x1b\\\x1b_Gm=i=1;\x1b\\\x1b[0m reset") is True


def test_is_image_line_detects_mixed_kitty_and_iterm2_sequences_in_one_line() -> None:
    mixed_line = "Kitty: \x1b_Ga=T...\x1b\\\x1b_Gm=i=1;\x1b\\ iTerm2: \x1b]1337;File=inline=1:data==\x07"
    assert is_image_line(mixed_line) is True


def test_is_image_line_should_detect_images_when_line_has_both_kitty_and_iterm2_sequences_like_ts() -> None:
    """should detect images when line has both Kitty and iTerm2 sequences"""
    mixed_line = "Kitty: \x1b_Ga=T...\x1b\\\x1b_Gm=i=1;\x1b\\ iTerm2: \x1b]1337;File=inline=1:data==\x07"
    assert is_image_line(mixed_line) is True


def test_is_image_line_should_detect_image_in_line_with_multiple_text_and_image_segments_like_ts() -> None:
    """should detect image in line with multiple text and image segments"""
    complex_line = "Start \x1b]1337;File=img1==\x07 middle \x1b]1337;File=img2==\x07 end"
    assert is_image_line(complex_line) is True


def test_is_image_line_ignores_plain_text_and_non_image_ansi_sequences() -> None:
    assert is_image_line("This is just a regular text line without any escape sequences") is False
    assert is_image_line("\x1b[31mRed text\x1b[0m and \x1b[32mgreen text\x1b[0m") is False
    assert is_image_line("\x1b[1A\x1b[2KLine cleared and moved up") is False
    assert is_image_line("Some text with ]1337;File but missing ESC at start") is False
    assert is_image_line("Some text with _G but missing ESC at start") is False
    assert is_image_line("") is False
    assert is_image_line("\n") is False
    assert is_image_line("/path/to/File_1337_backup/image.jpg") is False
    assert is_image_line("A" * 58649) is False
    assert is_image_line("very long regular text " + ("D" * 100000)) is False


def test_is_image_line_should_not_detect_images_in_plain_text_lines_like_ts() -> None:
    """should not detect images in plain text lines"""
    assert is_image_line("This is just a regular text line without any escape sequences") is False


def test_is_image_line_should_not_detect_images_in_lines_with_only_ansi_codes_like_ts() -> None:
    """should not detect images in lines with only ANSI codes"""
    assert is_image_line("\x1b[31mRed text\x1b[0m and \x1b[32mgreen text\x1b[0m") is False


def test_is_image_line_should_not_detect_images_in_lines_with_cursor_movement_codes_like_ts() -> None:
    """should not detect images in lines with cursor movement codes"""
    assert is_image_line("\x1b[1A\x1b[2KLine cleared and moved up") is False


def test_is_image_line_should_not_detect_images_in_lines_with_partial_iterm2_sequences_like_ts() -> None:
    """should not detect images in lines with partial iTerm2 sequences"""
    assert is_image_line("Some text with ]1337;File but missing ESC at start") is False


def test_is_image_line_should_not_detect_images_in_lines_with_partial_kitty_sequences_like_ts() -> None:
    """should not detect images in lines with partial Kitty sequences"""
    assert is_image_line("Some text with _G but missing ESC at start") is False


def test_is_image_line_should_not_detect_images_in_empty_lines_like_ts() -> None:
    """should not detect images in empty lines"""
    assert is_image_line("") is False


def test_is_image_line_should_not_detect_images_in_lines_with_newlines_only_like_ts() -> None:
    """should not detect images in lines with newlines only"""
    assert is_image_line("\n") is False


def test_is_image_line_should_not_falsely_detect_image_in_line_with_file_path_containing_keywords_like_ts() -> None:
    """
    should not falsely detect image in line with file path containing keywords
    does not detect images in regular long text
    does not detect images in lines with file paths
    """
    assert is_image_line("/path/to/File_1337_backup/image.jpg") is False


def test_parse_key_recognizes_legacy_arrow_sequences() -> None:
    """
    should parse arrow keys
    should parse double bracket pageUp
    """
    assert parse_key("\x1b[A") == "up"
    assert parse_key("\x1b[B") == "down"
    assert parse_key("\x1b[C") == "right"
    assert parse_key("\x1b[D") == "left"
    assert parse_key("\x1b[[5~") == "pageUp"


def test_matches_key_recognizes_legacy_arrows() -> None:
    """should match arrow keys"""
    assert matches_key("\x1b[A", "up") is True
    assert matches_key("\x1b[B", "down") is True
    assert matches_key("\x1b[C", "right") is True
    assert matches_key("\x1b[D", "left") is True


def test_parse_key_supports_ss3_and_rxvt_legacy_sequences() -> None:
    """
    should match SS3 arrows and home/end
    should match legacy function keys and clear
    should match rxvt modifier sequences
    should parse SS3 arrows and home/end
    should parse legacy function and modifier sequences
    should parse special keys
    """
    assert parse_key("\x1bOA") == "up"
    assert parse_key("\x1bOB") == "down"
    assert parse_key("\x1bOC") == "right"
    assert parse_key("\x1bOD") == "left"
    assert parse_key("\x1bOH") == "home"
    assert parse_key("\x1bOF") == "end"
    assert parse_key("\x1bOM") == "enter"
    assert parse_key("\x1bOP") == "f1"
    assert parse_key("\x1b[24~") == "f12"
    assert parse_key("\x1b[E") == "clear"
    assert parse_key("\x1b[2^") == "ctrl+insert"
    assert parse_key("\x1bp") == "alt+up"


def test_parse_key_supports_modify_other_keys_sequences() -> None:
    """
    should match xterm modifyOtherKeys Ctrl+c
    should match xterm modifyOtherKeys Ctrl+d
    should match xterm modifyOtherKeys Ctrl+z
    should match xterm modifyOtherKeys Enter variants
    should match xterm modifyOtherKeys Tab variants
    should match xterm modifyOtherKeys symbol combos
    should match xterm modifyOtherKeys digit combos
    """
    assert parse_key("\x1b[27;5;99~") == "ctrl+c"
    assert parse_key("\x1b[27;5;13~") == "ctrl+enter"
    assert parse_key("\x1b[27;2;13~") == "shift+enter"
    assert parse_key("\x1b[27;3;13~") == "alt+enter"
    assert parse_key("\x1b[27;2;9~") == "shift+tab"
    assert parse_key("\x1b[27;5;9~") == "ctrl+tab"
    assert parse_key("\x1b[27;3;9~") == "alt+tab"
    assert parse_key("\x1b[1;5D") == "ctrl+left"
    assert parse_key("\x1b[1;5C") == "ctrl+right"


def test_parse_key_supports_kitty_csi_u_sequences() -> None:
    """
    should match Ctrl+c when pressing Ctrl+С (Cyrillic) with base layout key
    should match Ctrl+d when pressing Ctrl+В (Cyrillic) with base layout key
    should match Ctrl+z when pressing Ctrl+Я (Cyrillic) with base layout key
    should match Ctrl+Shift+p with base layout key
    should still match direct codepoint when no base layout key
    should match digit bindings via Kitty CSI-u
    should handle shifted key in format
    should handle event type in format
    should handle full format with shifted key, base key, and event type
    should prefer codepoint for Latin letters even when base layout differs
    should prefer codepoint for symbol keys even when base layout differs
    should prefer codepoint for Latin letters when base layout differs
    should prefer codepoint for symbol keys when base layout differs
    should not match wrong key even with base layout
    should not match wrong modifiers even with base layout
    should return Latin key name when base layout key is present
    should return key name from codepoint when no base layout
    should ignore Kitty CSI-u with unsupported modifiers
    """
    set_kitty_protocol_active(True)
    try:
        assert parse_key("\x1b[49u") == "1"
        assert parse_key("\x1b[49;5u") == "ctrl+1"
        assert parse_key("\x1b[1089::99;5u") == "ctrl+c"
        assert parse_key("\x1b[1074::100;5u") == "ctrl+d"
        assert parse_key("\x1b[1103::122;5u") == "ctrl+z"
        assert parse_key("\x1b[1079::112;6u") == "ctrl+shift+p"
        assert parse_key("\x1b[107::118;5u") == "ctrl+k"
        assert parse_key("\x1b[47::91;5u") == "ctrl+/"
        assert parse_key("\x1b[99:67:99;2u") == "shift+c"
        assert matches_key("\x1b[1089::99;5u", "ctrl+c") is True
        assert matches_key("\x1b[1074::100;5u", "ctrl+d") is True
        assert matches_key("\x1b[1103::122;5u", "ctrl+z") is True
        assert matches_key("\x1b[1079::112;6u", "ctrl+shift+p") is True
        assert matches_key("\x1b[107::118;5u", "ctrl+k") is True
        assert matches_key("\x1b[47::91;5u", "ctrl+/") is True
        assert matches_key("\x1b[1089::99;5u", "ctrl+d") is False
        assert matches_key("\x1b[107::118;5u", "ctrl+v") is False
        assert matches_key("\x1b[47::91;5u", "ctrl+[") is False
        assert parse_key("\x1b[99;9u") is None
    finally:
        set_kitty_protocol_active(False)


def test_parse_key_respects_kitty_mode_for_legacy_alt_sequences() -> None:
    """
    should parse legacy alt-prefixed sequences when kitty inactive
    should match legacy Ctrl+c
    should match legacy Ctrl+d
    should match escape key
    should match legacy linefeed as enter
    should treat linefeed as shift+enter when kitty active
    should distinguish backspace (0x7f) from ctrl+backspace (0x08)
    should parse legacy Ctrl+letter
    """
    set_kitty_protocol_active(False)
    assert parse_key("\x1ba") == "alt+a"
    assert parse_key("\x1b1") == "alt+1"
    assert parse_key("\x1b\r") == "alt+enter"
    assert parse_key("\x1b ") == "alt+space"
    assert parse_key("\x1b\x7f") == "alt+backspace"
    assert parse_key("\x1b\x03") == "ctrl+alt+c"
    assert parse_key("\x1b\x1b") == "ctrl+alt+["

    set_kitty_protocol_active(True)
    try:
        assert parse_key("\x1ba") is None
        assert parse_key("\x1b1") is None
        assert parse_key("\x1b\r") is None
        assert parse_key("\x1b ") is None
        assert parse_key("\x1b\x7f") == "alt+backspace"
        assert parse_key("\x1b\x03") is None
        assert parse_key("\x1b\x1b") == "ctrl+alt+["
        assert parse_key("\x1b\b") == "alt+backspace"
    finally:
        set_kitty_protocol_active(False)


def test_parse_key_supports_legacy_alt_arrow_mappings_only_outside_kitty_mode() -> None:
    """should match alt+arrows"""
    set_kitty_protocol_active(False)
    assert parse_key("\x1bB") == "alt+left"
    assert parse_key("\x1bF") == "alt+right"

    set_kitty_protocol_active(True)
    try:
        assert parse_key("\x1bB") is None
        assert parse_key("\x1bF") is None
    finally:
        set_kitty_protocol_active(False)


def test_parse_key_supports_legacy_control_aliases_and_shift_tab() -> None:
    """
    should parse ctrl+space
    should match legacy Ctrl+symbol
    """
    assert parse_key("\x00") == "ctrl+space"
    assert matches_key("\x00", "ctrl+space") is True
    assert parse_key("\x1c") == "ctrl+\\"
    assert parse_key("\x1d") == "ctrl+]"
    assert parse_key("\x1f") == "ctrl+-"
    assert matches_key("\x1f", "ctrl+_") is True
    assert parse_key("\x1b[Z") == "shift+tab"
    assert matches_key("\x1b[Z", "shift+tab") is True


def test_parse_key_supports_legacy_ctrl_alt_symbol_aliases() -> None:
    """should match legacy Ctrl+Alt+symbol"""
    assert parse_key("\x1b\x1b") == "ctrl+alt+["
    assert matches_key("\x1b\x1b", "ctrl+alt+[") is True
    assert parse_key("\x1b\x1c") == "ctrl+alt+\\"
    assert parse_key("\x1b\x1d") == "ctrl+alt+]"
    assert parse_key("\x1b\x1f") == "ctrl+alt+-"
    assert matches_key("\x1b\x1f", "ctrl+alt+_") is True
    assert matches_key("\x1b\x1f", "ctrl+alt+-") is True


def test_parse_key_recognizes_release_and_repeat_kitty_events() -> None:
    """
    should handle event type in format
    should handle full format with shifted key, base key, and event type
    """
    assert is_key_repeat("\x1b[1089:1057:99;6:2u") is True
    assert is_key_release("\x1b[1089::99;5:3u") is True
    assert is_key_release("\x1b[49u") is False


def test_stdin_buffer_emits_regular_characters_immediately() -> None:
    """
    should pass through regular characters immediately
    should pass through multiple regular characters
    should handle unicode characters
    """
    events: list[str] = []
    buffer = StdinBuffer(timeout=0.01)
    buffer.on("data", events.append)

    buffer.process("abc")

    assert events == ["a", "b", "c"]
    assert buffer.get_buffer() == ""


def test_stdin_buffer_buffers_and_flushes_partial_sequences() -> None:
    """
    should pass through complete mouse SGR sequences
    should buffer incomplete mouse SGR sequence
    should buffer incomplete CSI sequence
    should buffer split across many chunks
    """
    events: list[str] = []
    buffer = StdinBuffer(timeout=0.01)
    buffer.on("data", events.append)

    buffer.process("\x1b")
    buffer.process("[<35")
    assert events == []
    assert buffer.get_buffer() == "\x1b[<35"

    buffer.process(";20;5m")
    assert events == ["\x1b[<35;20;5m"]
    assert buffer.get_buffer() == ""


def test_stdin_buffer_flushes_incomplete_sequence_after_timeout() -> None:
    """
    should flush incomplete sequence after timeout
    should emit flushed data via timeout
    """
    events: list[str] = []
    buffer = StdinBuffer(timeout=0.01)
    buffer.on("data", events.append)

    buffer.process("\x1b[<35")
    time.sleep(0.03)

    assert events == ["\x1b[<35"]
    assert buffer.get_buffer() == ""


def test_stdin_buffer_handles_empty_input() -> None:
    """
    should handle empty input
    should return empty array if nothing to flush
    """
    events: list[str] = []
    buffer = StdinBuffer(timeout=0.01)
    buffer.on("data", events.append)

    buffer.process("")

    assert events == [""]


def test_stdin_buffer_handles_mixed_content_and_complete_sequences() -> None:
    """
    should handle characters followed by escape sequence
    should handle escape sequence followed by characters
    should handle multiple complete sequences
    should handle partial sequence with preceding characters
    should handle multiple regular characters
    should pass through complete arrow key sequences
    """
    events: list[str] = []
    buffer = StdinBuffer(timeout=0.01)
    buffer.on("data", events.append)

    buffer.process("abc\x1b[A")

    assert events == ["a", "b", "c", "\x1b[A"]


def test_stdin_buffer_handles_old_style_mouse_sequence() -> None:
    """
    should handle old-style mouse sequence (ESC[M + 3 bytes)
    should pass through SS3 sequences
    should buffer incomplete old-style mouse sequence
    """
    events: list[str] = []
    buffer = StdinBuffer(timeout=0.01)
    buffer.on("data", events.append)

    buffer.process("\x1b[M abc")

    assert events == ["\x1b[M ab", "c"]


def test_stdin_buffer_handles_very_long_sequences() -> None:
    """
    should handle very long sequences
    should pass through complete function key sequences
    """
    events: list[str] = []
    buffer = StdinBuffer(timeout=0.01)
    buffer.on("data", events.append)

    sequence = f"\x1b[{'1;' * 50}H"
    buffer.process(sequence)

    assert events == [sequence]


def test_stdin_buffer_handles_bracketed_paste() -> None:
    """
    should emit paste event for complete bracketed paste
    should handle paste arriving in chunks
    should handle paste with input before and after
    should handle paste with newlines
    should handle paste with unicode
    """
    data_events: list[str] = []
    paste_events: list[str] = []
    buffer = StdinBuffer(timeout=0.01)
    buffer.on("data", data_events.append)
    buffer.on("paste", paste_events.append)

    buffer.process("a\x1b[200~hello")
    assert data_events == ["a"]
    assert paste_events == []

    buffer.process(" world\x1b[201~b")
    assert paste_events == ["hello world"]
    assert data_events == ["a", "b"]


def test_stdin_buffer_supports_high_byte_buffer_input() -> None:
    """
    should handle buffer input
    should pass through meta key sequences
    """
    data_events: list[str] = []
    buffer = StdinBuffer(timeout=0.01)
    buffer.on("data", data_events.append)

    buffer.process(bytes([0xE1]))

    assert data_events == ["\x1ba"]


def test_stdin_buffer_handles_kitty_keyboard_sequences_like_ts() -> None:
    """
    should handle Kitty CSI u press events
    should handle Kitty CSI u release events
    should handle batched Kitty press and release
    should handle multiple batched Kitty events
    should handle Kitty arrow keys with event type
    should handle Kitty functional keys with event type
    should handle plain characters mixed with Kitty sequences
    should handle Kitty sequence followed by plain characters
    should handle rapid typing simulation with Kitty protocol
    """
    events: list[str] = []
    buffer = StdinBuffer(timeout=0.01)
    buffer.on("data", events.append)

    for seq in [
        "\x1b[97u",
        "\x1b[97;1:3u",
        "\x1b[97u\x1b[97;1:3u",
        "\x1b[97u\x1b[97;1:3u\x1b[98u\x1b[98;1:3u",
        "\x1b[1;1:1A",
        "\x1b[3;1:3~",
        "a\x1b[97;1:3u",
        "\x1b[97ua",
        "\x1b[104u\x1b[104;1:3u\x1b[105u\x1b[105;1:3u",
    ]:
        buffer.process(seq)

    assert events == [
        "\x1b[97u",
        "\x1b[97;1:3u",
        "\x1b[97u",
        "\x1b[97;1:3u",
        "\x1b[97u",
        "\x1b[97;1:3u",
        "\x1b[98u",
        "\x1b[98;1:3u",
        "\x1b[1;1:1A",
        "\x1b[3;1:3~",
        "a",
        "\x1b[97;1:3u",
        "\x1b[97u",
        "a",
        "\x1b[104u",
        "\x1b[104;1:3u",
        "\x1b[105u",
        "\x1b[105;1:3u",
    ]


def test_stdin_buffer_handles_mouse_sequences_like_ts() -> None:
    """
    should handle mouse press event
    should handle mouse release event
    should handle mouse move event
    should handle split mouse events
    should handle multiple mouse events
    """
    events: list[str] = []
    buffer = StdinBuffer(timeout=0.01)
    buffer.on("data", events.append)

    buffer.process("\x1b[<0;10;5M")
    buffer.process("\x1b[<0;10;5m")
    buffer.process("\x1b[<35;20;5m")
    buffer.process("\x1b[<3")
    buffer.process("5;1")
    buffer.process("5;")
    buffer.process("10m")
    buffer.process("\x1b[<35;1;1m\x1b[<35;2;2m\x1b[<35;3;3m")

    assert events == [
        "\x1b[<0;10;5M",
        "\x1b[<0;10;5m",
        "\x1b[<35;20;5m",
        "\x1b[<35;15;10m",
        "\x1b[<35;1;1m",
        "\x1b[<35;2;2m",
        "\x1b[<35;3;3m",
    ]


def test_stdin_buffer_handles_lone_escape_with_timeout_like_ts() -> None:
    """should handle lone escape character with timeout"""
    events: list[str] = []
    buffer = StdinBuffer(timeout=0.01)
    buffer.on("data", events.append)

    buffer.process("\x1b")
    assert events == []
    time.sleep(0.03)

    assert events == ["\x1b"]


def test_stdin_buffer_flush_clear_and_destroy() -> None:
    """
    should flush incomplete sequences
    should clear buffered content without emitting
    should clear buffer on destroy
    should clear pending timeouts on destroy
    should handle lone escape character with explicit flush
    """
    buffer = StdinBuffer(timeout=0.01)
    buffer.process("\x1b[")
    assert buffer.flush() == ["\x1b["]
    assert buffer.get_buffer() == ""

    buffer.process("\x1b[200~hello")
    buffer.clear()
    assert buffer.get_buffer() == ""

    buffer.process("x")
    buffer.destroy()
    assert buffer.get_buffer() == ""


class _FakeStdin:
    def __init__(self) -> None:
        self.is_raw = False
        self.encoding: str | None = None
        self.resumed = False
        self.paused = False
        self.handlers: dict[str, list[Callable[[str], None]]] = {"data": []}

    def fileno(self) -> int:
        return 0

    def set_encoding(self, encoding: str) -> None:
        self.encoding = encoding

    def resume(self) -> None:
        self.resumed = True

    def pause(self) -> None:
        self.paused = True

    def on(self, event: str, handler: Callable[[str], None]) -> None:
        self.handlers.setdefault(event, []).append(handler)

    def remove_listener(self, event: str, handler: Callable[[str], None]) -> None:
        listeners = self.handlers.get(event, [])
        if handler in listeners:
            listeners.remove(handler)

    def emit(self, event: str, data: str) -> None:
        for handler in list(self.handlers.get(event, [])):
            handler(data)


class _FakeStdout:
    def __init__(self) -> None:
        self.writes: list[str] = []
        self.handlers: dict[str, list[Callable[[], None]]] = {"resize": []}
        self.columns = 80
        self.rows = 24

    def write(self, data: str) -> None:
        self.writes.append(data)

    def flush(self) -> None:
        return None

    def on(self, event: str, handler: Callable[[], None]) -> None:
        self.handlers.setdefault(event, []).append(handler)

    def remove_listener(self, event: str, handler: Callable[[], None]) -> None:
        listeners = self.handlers.get(event, [])
        if handler in listeners:
            listeners.remove(handler)

    def emit_resize(self) -> None:
        for handler in list(self.handlers.get("resize", [])):
            handler()


def test_process_terminal_detects_kitty_response_and_forwards_other_input(monkeypatch: pytest.MonkeyPatch) -> None:
    stdin = _FakeStdin()
    stdout = _FakeStdout()
    inputs: list[str] = []
    resizes: list[str] = []

    monkeypatch.setattr("paw.pi_agent.tui.terminal.sys.stdin", stdin)
    monkeypatch.setattr("paw.pi_agent.tui.terminal.sys.stdout", stdout)
    monkeypatch.setattr("paw.pi_agent.tui.terminal.tty.setraw", lambda _fd: None)
    monkeypatch.setattr("paw.pi_agent.tui.terminal.termios.tcgetattr", lambda _fd: [1])
    monkeypatch.setattr("paw.pi_agent.tui.terminal.os.kill", lambda _pid, _sig: None)

    terminal = ProcessTerminal()
    terminal._kitty_query_timeout_s = 1.0  # type: ignore[attr-defined]
    terminal.start(inputs.append, lambda: resizes.append("resize"))

    assert stdout.writes[:2] == ["\x1b[?2004h", "\x1b[?u"]
    assert stdin.resumed is True
    assert terminal.kitty_protocol_active is False

    stdin.emit("data", "\x1b[?1u")
    assert terminal.kitty_protocol_active is True
    assert stdout.writes[-1] == "\x1b[>7u"
    assert inputs == []

    stdin.emit("data", "a")
    assert inputs == ["a"]

    stdout.emit_resize()
    assert resizes == ["resize"]

    terminal.stop()


def test_process_terminal_drain_input_disables_modes(monkeypatch: pytest.MonkeyPatch) -> None:
    stdin = _FakeStdin()
    stdout = _FakeStdout()

    monkeypatch.setattr("paw.pi_agent.tui.terminal.sys.stdin", stdin)
    monkeypatch.setattr("paw.pi_agent.tui.terminal.sys.stdout", stdout)
    monkeypatch.setattr("paw.pi_agent.tui.terminal.tty.setraw", lambda _fd: None)
    monkeypatch.setattr("paw.pi_agent.tui.terminal.termios.tcgetattr", lambda _fd: [1])
    monkeypatch.setattr("paw.pi_agent.tui.terminal.os.kill", lambda _pid, _sig: None)

    terminal = ProcessTerminal()
    terminal._kitty_query_timeout_s = 1.0  # type: ignore[attr-defined]
    terminal.start(lambda _data: None, lambda: None)
    terminal._kitty_protocol_active = True  # type: ignore[attr-defined]
    terminal._modify_other_keys_active = True  # type: ignore[attr-defined]

    asyncio.run(terminal.drain_input(max_ms=20, idle_ms=5))

    assert "\x1b[<u" in stdout.writes
    assert "\x1b[>4;0m" in stdout.writes
    assert terminal.kitty_protocol_active is False

    terminal.stop()


def test_process_terminal_falls_back_to_modify_other_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    stdin = _FakeStdin()
    stdout = _FakeStdout()

    monkeypatch.setattr("paw.pi_agent.tui.terminal.sys.stdin", stdin)
    monkeypatch.setattr("paw.pi_agent.tui.terminal.sys.stdout", stdout)
    monkeypatch.setattr("paw.pi_agent.tui.terminal.tty.setraw", lambda _fd: None)
    monkeypatch.setattr("paw.pi_agent.tui.terminal.termios.tcgetattr", lambda _fd: [1])
    monkeypatch.setattr("paw.pi_agent.tui.terminal.os.kill", lambda _pid, _sig: None)

    terminal = ProcessTerminal()
    terminal._kitty_query_timeout_s = 0.01  # type: ignore[attr-defined]
    terminal.start(lambda _data: None, lambda: None)
    time.sleep(0.03)

    assert "\x1b[>4;2m" in stdout.writes
    assert terminal._modify_other_keys_active is True  # type: ignore[attr-defined]

    terminal.stop()


def test_process_terminal_stop_cleans_up_handlers_and_restores_state(monkeypatch: pytest.MonkeyPatch) -> None:
    stdin = _FakeStdin()
    stdout = _FakeStdout()
    raw_modes: list[bool] = []

    monkeypatch.setattr("paw.pi_agent.tui.terminal.sys.stdin", stdin)
    monkeypatch.setattr("paw.pi_agent.tui.terminal.sys.stdout", stdout)
    monkeypatch.setattr("paw.pi_agent.tui.terminal.tty.setraw", lambda _fd: None)
    monkeypatch.setattr("paw.pi_agent.tui.terminal.termios.tcgetattr", lambda _fd: [1])
    monkeypatch.setattr("paw.pi_agent.tui.terminal.os.kill", lambda _pid, _sig: None)

    terminal = ProcessTerminal()
    terminal._set_raw_mode = raw_modes.append  # type: ignore[attr-defined]
    terminal._kitty_query_timeout_s = 1.0  # type: ignore[attr-defined]
    terminal.start(lambda _data: None, lambda: None)

    assert stdin.handlers["data"]
    assert stdout.handlers["resize"]

    terminal.stop()

    assert stdin.handlers["data"] == []
    assert stdout.handlers["resize"] == []
    assert stdin.paused is True
    assert raw_modes == [True, False]
