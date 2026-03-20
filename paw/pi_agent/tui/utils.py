from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

import regex

_SEGMENTER = regex.compile(r"\X")
_ANSI_CSI = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z~]")
_ANSI_OSC = re.compile(r"\x1b\].*?(?:\x07|\x1b\\)")
_ANSI_APC = re.compile(r"\x1b_.*?(?:\x07|\x1b\\)")


def get_segmenter() -> regex.Pattern[str]:
    return _SEGMENTER


def _could_be_emoji(segment: str) -> bool:
    if not segment:
        return False
    cp = ord(segment[0])
    return (
        0x1F000 <= cp <= 0x1FBFF
        or 0x2300 <= cp <= 0x23FF
        or 0x2600 <= cp <= 0x27BF
        or 0x2B50 <= cp <= 0x2B55
        or "\ufe0f" in segment
        or len(segment) > 2
    )


def extract_ansi_code(text: str, pos: int) -> tuple[str, int] | None:
    if pos < 0 or pos >= len(text) or text[pos] != "\x1b":
        return None

    for pattern in (_ANSI_CSI, _ANSI_OSC, _ANSI_APC):
        match = pattern.match(text, pos)
        if match:
            code = match.group(0)
            return code, len(code)
    return None


def _grapheme_width(segment: str) -> int:
    if not segment:
        return 0
    if all(unicodedata.category(ch).startswith(("C", "M")) for ch in segment):
        return 0
    if _could_be_emoji(segment):
        return 2
    base = segment.lstrip("".join(ch for ch in segment if unicodedata.category(ch) in {"Cf", "Cc", "Cs", "Mn", "Mc"}))
    cp = ord(base[0]) if base else ord(segment[0])
    if 0x1F1E6 <= cp <= 0x1F1FF:
        return 2
    if unicodedata.east_asian_width(chr(cp)) in {"F", "W"}:
        return 2
    return 1


def _iter_graphemes(text: str) -> list[str]:
    return [m.group(0) for m in _SEGMENTER.finditer(text)]


def visible_width(text: str) -> int:
    if not text:
        return 0
    if all(0x20 <= ord(ch) <= 0x7E for ch in text):
        return len(text)

    clean = text.replace("\t", "   ")
    while True:
        match = extract_ansi_code(clean, clean.find("\x1b"))
        if not match:
            break
        code, length = match
        idx = clean.find(code)
        clean = clean[:idx] + clean[idx + length :]
    width = 0
    for segment in _iter_graphemes(clean):
        width += _grapheme_width(segment)
    return width


@dataclass(slots=True)
class _AnsiCodeTracker:
    bold: bool = False
    dim: bool = False
    italic: bool = False
    underline: bool = False
    blink: bool = False
    inverse: bool = False
    hidden: bool = False
    strikethrough: bool = False
    fg_color: str | None = None
    bg_color: str | None = None

    def reset(self) -> None:
        self.bold = False
        self.dim = False
        self.italic = False
        self.underline = False
        self.blink = False
        self.inverse = False
        self.hidden = False
        self.strikethrough = False
        self.fg_color = None
        self.bg_color = None

    def clear(self) -> None:
        self.reset()

    def process(self, ansi_code: str) -> None:
        if not ansi_code.endswith("m"):
            return
        match = re.match(r"\x1b\[([\d;]*)m", ansi_code)
        if not match:
            return
        params = match.group(1)
        if params in {"", "0"}:
            self.reset()
            return
        parts = params.split(";")
        i = 0
        while i < len(parts):
            code = int(parts[i] or "0")
            if code in {38, 48} and i + 2 < len(parts):
                if parts[i + 1] == "5" and i + 2 < len(parts):
                    color_code = f"{parts[i]};{parts[i + 1]};{parts[i + 2]}"
                    if code == 38:
                        self.fg_color = color_code
                    else:
                        self.bg_color = color_code
                    i += 3
                    continue
                if parts[i + 1] == "2" and i + 4 < len(parts):
                    color_code = f"{parts[i]};{parts[i + 1]};{parts[i + 2]};{parts[i + 3]};{parts[i + 4]}"
                    if code == 38:
                        self.fg_color = color_code
                    else:
                        self.bg_color = color_code
                    i += 5
                    continue
            if code == 0:
                self.reset()
            elif code == 1:
                self.bold = True
            elif code == 2:
                self.dim = True
            elif code == 3:
                self.italic = True
            elif code == 4:
                self.underline = True
            elif code == 5:
                self.blink = True
            elif code == 7:
                self.inverse = True
            elif code == 8:
                self.hidden = True
            elif code == 9:
                self.strikethrough = True
            elif code in {21, 22}:
                self.bold = False
                if code == 22:
                    self.dim = False
            elif code == 23:
                self.italic = False
            elif code == 24:
                self.underline = False
            elif code == 25:
                self.blink = False
            elif code == 27:
                self.inverse = False
            elif code == 28:
                self.hidden = False
            elif code == 29:
                self.strikethrough = False
            elif code == 39:
                self.fg_color = None
            elif code == 49:
                self.bg_color = None
            elif 30 <= code <= 37 or 90 <= code <= 97:
                self.fg_color = str(code)
            elif 40 <= code <= 47 or 100 <= code <= 107:
                self.bg_color = str(code)
            i += 1

    def get_active_codes(self) -> str:
        codes: list[str] = []
        if self.bold:
            codes.append("1")
        if self.dim:
            codes.append("2")
        if self.italic:
            codes.append("3")
        if self.underline:
            codes.append("4")
        if self.blink:
            codes.append("5")
        if self.inverse:
            codes.append("7")
        if self.hidden:
            codes.append("8")
        if self.strikethrough:
            codes.append("9")
        if self.fg_color:
            codes.append(self.fg_color)
        if self.bg_color:
            codes.append(self.bg_color)
        return f"\x1b[{';'.join(codes)}m" if codes else ""

    def has_active_codes(self) -> bool:
        return any(
            [
                self.bold,
                self.dim,
                self.italic,
                self.underline,
                self.blink,
                self.inverse,
                self.hidden,
                self.strikethrough,
                self.fg_color is not None,
                self.bg_color is not None,
            ]
        )

    def get_line_end_reset(self) -> str:
        return "\x1b[24m" if self.underline else ""


def _update_tracker_from_text(text: str, tracker: _AnsiCodeTracker) -> None:
    i = 0
    while i < len(text):
        ansi = extract_ansi_code(text, i)
        if ansi:
            code, length = ansi
            tracker.process(code)
            i += length
        else:
            i += 1


def _split_into_tokens_with_ansi(text: str) -> list[str]:
    tokens: list[str] = []
    current = ""
    pending_ansi = ""
    in_whitespace = False
    i = 0

    while i < len(text):
        ansi = extract_ansi_code(text, i)
        if ansi:
            code, length = ansi
            pending_ansi += code
            i += length
            continue

        char = text[i]
        char_is_space = char == " "
        if char_is_space != in_whitespace and current:
            tokens.append(current)
            current = ""
        if pending_ansi:
            current += pending_ansi
            pending_ansi = ""
        in_whitespace = char_is_space
        current += char
        i += 1

    if pending_ansi:
        current += pending_ansi
    if current:
        tokens.append(current)
    return tokens


def wrap_text_with_ansi(text: str, width: int) -> list[str]:
    if not text:
        return [""]

    input_lines = text.split("\n")
    result: list[str] = []
    tracker = _AnsiCodeTracker()

    for input_line in input_lines:
        prefix = tracker.get_active_codes() if result else ""
        result.extend(_wrap_single_line(prefix + input_line, width))
        _update_tracker_from_text(input_line, tracker)
    return result or [""]


def _break_long_word(word: str, width: int, tracker: _AnsiCodeTracker) -> list[str]:
    lines: list[str] = []
    current_line = tracker.get_active_codes()
    current_width = 0
    segments: list[tuple[str, str]] = []
    i = 0
    while i < len(word):
        ansi = extract_ansi_code(word, i)
        if ansi:
            code, length = ansi
            segments.append(("ansi", code))
            i += length
            continue
        end = i
        while end < len(word) and not extract_ansi_code(word, end):
            end += 1
        for grapheme in _iter_graphemes(word[i:end]):
            segments.append(("grapheme", grapheme))
        i = end

    for kind, value in segments:
        if kind == "ansi":
            current_line += value
            tracker.process(value)
            continue
        if not value:
            continue
        grapheme_width = visible_width(value)
        if current_width + grapheme_width > width:
            line_end_reset = tracker.get_line_end_reset()
            if line_end_reset:
                current_line += line_end_reset
            lines.append(current_line)
            current_line = tracker.get_active_codes()
            current_width = 0
        current_line += value
        current_width += grapheme_width
    if current_line:
        lines.append(current_line)
    return lines or [""]


def _wrap_single_line(line: str, width: int) -> list[str]:
    if not line:
        return [""]
    if visible_width(line) <= width:
        return [line]

    wrapped: list[str] = []
    tracker = _AnsiCodeTracker()
    tokens = _split_into_tokens_with_ansi(line)
    current_line = ""
    current_visible_length = 0

    for token in tokens:
        token_visible_length = visible_width(token)
        is_whitespace = token.strip() == ""
        if token_visible_length > width and not is_whitespace:
            if current_line:
                line_end_reset = tracker.get_line_end_reset()
                if line_end_reset:
                    current_line += line_end_reset
                wrapped.append(current_line)
                current_line = ""
                current_visible_length = 0
            broken = _break_long_word(token, width, tracker)
            wrapped.extend(broken[:-1])
            current_line = broken[-1]
            current_visible_length = visible_width(current_line)
            continue

        total_needed = current_visible_length + token_visible_length
        if total_needed > width and current_visible_length > 0:
            line_to_wrap = current_line.rstrip()
            line_end_reset = tracker.get_line_end_reset()
            if line_end_reset:
                line_to_wrap += line_end_reset
            wrapped.append(line_to_wrap)
            if is_whitespace:
                current_line = tracker.get_active_codes()
                current_visible_length = 0
            else:
                current_line = tracker.get_active_codes() + token
                current_visible_length = token_visible_length
        else:
            current_line += token
            current_visible_length += token_visible_length
        _update_tracker_from_text(token, tracker)

    if current_line:
        wrapped.append(current_line)
    return [line.rstrip() for line in wrapped] or [""]


def is_whitespace_char(char: str) -> bool:
    return bool(re.match(r"\s", char))


def is_punctuation_char(char: str) -> bool:
    return bool(re.match(r"""[(){}\[\]<>.,;:'"!?+\-=*/\\|&%^$#@~`]""", char))


def apply_background_to_line(line: str, width: int, bg_fn) -> str:
    visible_len = visible_width(line)
    padding = " " * max(0, width - visible_len)
    return bg_fn(line + padding)


def truncate_to_width(text: str, max_width: int, ellipsis: str = "...", pad: bool = False) -> str:
    text_visible_width = visible_width(text)
    if text_visible_width <= max_width:
        return text + " " * (max_width - text_visible_width) if pad else text

    ellipsis_width = visible_width(ellipsis)
    target_width = max_width - ellipsis_width
    if target_width <= 0:
        return ellipsis[:max_width]

    segments: list[tuple[str, str]] = []
    i = 0
    while i < len(text):
        ansi = extract_ansi_code(text, i)
        if ansi:
            code, length = ansi
            segments.append(("ansi", code))
            i += length
        else:
            end = i
            while end < len(text) and not extract_ansi_code(text, end):
                end += 1
            for grapheme in _iter_graphemes(text[i:end]):
                segments.append(("grapheme", grapheme))
            i = end

    result = ""
    current_width = 0
    for kind, value in segments:
        if kind == "ansi":
            result += value
            continue
        if not value:
            continue
        grapheme_width = visible_width(value)
        if current_width + grapheme_width > target_width:
            break
        result += value
        current_width += grapheme_width
    truncated = f"{result}\x1b[0m{ellipsis}"
    if pad:
        return truncated + " " * max(0, max_width - visible_width(truncated))
    return truncated


def slice_with_width(line: str, start_col: int, length: int, strict: bool = False) -> dict[str, int | str]:
    if length <= 0:
        return {"text": "", "width": 0}
    end_col = start_col + length
    result = ""
    result_width = 0
    current_col = 0
    i = 0
    pending_ansi = ""

    while i < len(line):
        ansi = extract_ansi_code(line, i)
        if ansi:
            code, length_ = ansi
            if current_col >= start_col and current_col < end_col:
                result += code
            elif current_col < start_col:
                pending_ansi += code
            i += length_
            continue

        text_end = i
        while text_end < len(line) and not extract_ansi_code(line, text_end):
            text_end += 1
        for segment in _iter_graphemes(line[i:text_end]):
            w = _grapheme_width(segment)
            in_range = current_col >= start_col and current_col < end_col
            fits = (not strict) or current_col + w <= end_col
            if in_range and fits:
                if pending_ansi:
                    result += pending_ansi
                    pending_ansi = ""
                result += segment
                result_width += w
            current_col += w
            if current_col >= end_col:
                break
        i = text_end
        if current_col >= end_col:
            break

    return {"text": result, "width": result_width}


def slice_by_column(line: str, start_col: int, length: int, strict: bool = False) -> str:
    return str(slice_with_width(line, start_col, length, strict)["text"])


def extract_segments(
    line: str,
    before_end: int,
    after_start: int,
    after_len: int,
    strict_after: bool = False,
) -> dict[str, int | str]:
    before = ""
    before_width = 0
    after = ""
    after_width = 0
    current_col = 0
    i = 0
    pending_ansi_before = ""
    after_started = False
    after_end = after_start + after_len
    tracker = _AnsiCodeTracker()

    while i < len(line):
        ansi = extract_ansi_code(line, i)
        if ansi:
            code, length = ansi
            tracker.process(code)
            if current_col < before_end:
                pending_ansi_before += code
            elif current_col >= after_start and current_col < after_end and after_started:
                after += code
            i += length
            continue

        text_end = i
        while text_end < len(line) and not extract_ansi_code(line, text_end):
            text_end += 1
        for segment in _iter_graphemes(line[i:text_end]):
            w = _grapheme_width(segment)
            if current_col < before_end:
                if pending_ansi_before:
                    before += pending_ansi_before
                    pending_ansi_before = ""
                before += segment
                before_width += w
            elif current_col >= after_start and current_col < after_end:
                fits = (not strict_after) or current_col + w <= after_end
                if fits:
                    if not after_started:
                        after += tracker.get_active_codes()
                        after_started = True
                    after += segment
                    after_width += w
            current_col += w
            if (after_len <= 0 and current_col >= before_end) or (after_len > 0 and current_col >= after_end):
                break
        i = text_end
        if (after_len <= 0 and current_col >= before_end) or (after_len > 0 and current_col >= after_end):
            break

    return {"before": before, "beforeWidth": before_width, "after": after, "afterWidth": after_width}
