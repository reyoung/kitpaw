from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from markdown_it import MarkdownIt
from markdown_it.token import Token

from ..terminal_image import is_image_line
from ..utils import visible_width, wrap_text_with_ansi


@dataclass(slots=True)
class DefaultTextStyle:
    color: object | None = None
    bg_color: object | None = None
    bold: bool = False
    italic: bool = False
    strikethrough: bool = False
    underline: bool = False


def _theme_call(theme: Any, key: str, text: str) -> str:
    fn = getattr(theme, key, None)
    if fn is None and isinstance(theme, dict):
        fn = theme.get(key)
    if callable(fn):
        return str(fn(text))
    return text


@dataclass(slots=True)
class _InlineStyleContext:
    apply_text: Any
    style_prefix: str


class Markdown:
    def __init__(self, text: str, padding_x: int, padding_y: int, theme, default_text_style=None) -> None:
        self.text = text
        self.padding_x = padding_x
        self.padding_y = padding_y
        self.theme = theme
        self.default_text_style = default_text_style
        self._cache: tuple[str, int, list[str]] | None = None
        self._md = MarkdownIt("commonmark").enable("table")
        self._default_style_prefix: str | None = None

    def set_text(self, text: str) -> None:
        self.text = text
        self.invalidate()

    def invalidate(self) -> None:
        self._cache = None
        self._default_style_prefix = None

    def render(self, width: int) -> list[str]:
        if self._cache and self._cache[0] == self.text and self._cache[1] == width:
            return self._cache[2]
        if not self.text.strip():
            self._cache = (self.text, width, [])
            return []

        content_width = max(1, width - self.padding_x * 2)
        tokens = self._md.parse(self.text.replace("\t", "   "))
        rendered = self._render_tokens(tokens, content_width)

        wrapped_lines: list[str] = []
        for line in rendered:
            if is_image_line(line):
                wrapped_lines.append(line)
            else:
                wrapped_lines.extend(wrap_text_with_ansi(line, content_width))

        empty_line = " " * width
        result: list[str] = [empty_line] * self.padding_y
        for line in wrapped_lines:
            if is_image_line(line):
                result.append(line)
                continue
            padded = (" " * self.padding_x) + line + (" " * self.padding_x)
            result.append(padded + " " * max(0, width - visible_width(padded)))
        result.extend([empty_line] * self.padding_y)
        self._cache = (self.text, width, result)
        return result

    def _render_tokens(self, tokens: list[Token], width: int) -> list[str]:
        lines: list[str] = []
        i = 0
        while i < len(tokens):
            token = tokens[i]
            if token.type == "heading_open":
                inline = tokens[i + 1]
                level = int(token.tag[1:]) if token.tag.startswith("h") else 1
                prefix = "" if level <= 2 else f"{'#' * level} "
                text = self._render_inline(inline.children or [])
                if level == 1:
                    lines.append(_theme_call(self.theme, "heading", _theme_call(self.theme, "underline", _theme_call(self.theme, "bold", text))))
                elif level == 2:
                    lines.append(_theme_call(self.theme, "heading", _theme_call(self.theme, "bold", text)))
                else:
                    lines.append(_theme_call(self.theme, "heading", _theme_call(self.theme, "bold", prefix + text)))
                i += 3
                if i < len(tokens) and tokens[i].type != "space":
                    lines.append("")
                continue

            if token.type == "paragraph_open":
                inline = tokens[i + 1]
                lines.append(self._render_inline(inline.children or []))
                i += 3
                if i < len(tokens) and tokens[i].type not in {"space", "bullet_list_open", "ordered_list_open"}:
                    lines.append("")
                continue

            if token.type == "fence":
                fence_info = token.info.strip()
                lines.append(_theme_call(self.theme, "code_block_border", f"```{fence_info}"))
                for code_line in token.content.rstrip("\n").split("\n"):
                    lines.append(f"  {_theme_call(self.theme, 'code_block', code_line)}")
                lines.append(_theme_call(self.theme, "code_block_border", "```"))
                i += 1
                if i < len(tokens) and tokens[i].type != "space":
                    lines.append("")
                continue

            if token.type in {"bullet_list_open", "ordered_list_open"}:
                list_lines, next_index = self._render_list(tokens, i, 0)
                lines.extend(list_lines)
                i = next_index
                continue

            if token.type == "blockquote_open":
                quote_lines, next_index = self._render_blockquote(tokens, i, width)
                lines.extend(quote_lines)
                i = next_index
                if i < len(tokens) and tokens[i].type != "space":
                    lines.append("")
                continue

            if token.type == "table_open":
                table_lines, next_index = self._render_table(tokens, i, width)
                lines.extend(table_lines)
                i = next_index
                if i < len(tokens) and tokens[i].type != "space":
                    lines.append("")
                continue

            if token.type == "hr":
                lines.append(_theme_call(self.theme, "hr", "─" * min(width, 80)))
                i += 1
                if i < len(tokens) and tokens[i].type != "space":
                    lines.append("")
                continue

            if token.type == "space":
                lines.append("")
            elif token.type == "html_block":
                content = token.content.strip()
                if content:
                    lines.append(self._apply_default_style(content))
            elif token.type == "inline":
                text = self._render_inline(token.children or [])
                if text:
                    lines.append(text)
            i += 1

        while lines and lines[-1] == "":
            lines.pop()
        return lines

    def _apply_default_style(self, text: str) -> str:
        if not self.default_text_style:
            return text

        styled = text
        color_fn = getattr(self.default_text_style, "color", None)
        if color_fn is None and isinstance(self.default_text_style, dict):
            color_fn = self.default_text_style.get("color")
        if callable(color_fn):
            styled = str(color_fn(styled))

        for style_key, theme_key in (
            ("bold", "bold"),
            ("italic", "italic"),
            ("strikethrough", "strikethrough"),
            ("underline", "underline"),
        ):
            enabled = getattr(self.default_text_style, style_key, None)
            if enabled is None and isinstance(self.default_text_style, dict):
                enabled = self.default_text_style.get(style_key, False)
            if enabled:
                styled = _theme_call(self.theme, theme_key, styled)

        return styled

    def _style_prefix(self, style_fn: Any) -> str:
        sentinel = "\0"
        styled = str(style_fn(sentinel))
        sentinel_index = styled.find(sentinel)
        if sentinel_index < 0:
            return ""
        return styled[:sentinel_index]

    def _default_inline_style_context(self) -> _InlineStyleContext:
        if self._default_style_prefix is None:
            self._default_style_prefix = self._style_prefix(self._apply_default_style)
        return _InlineStyleContext(
            apply_text=self._apply_default_style,
            style_prefix=self._default_style_prefix,
        )

    def _reapply_style_prefix(self, text: str, style_prefix: str) -> str:
        if not style_prefix:
            return text
        return text.replace("\x1b[0m", f"\x1b[0m{style_prefix}")

    def _plain_inline_text(self, tokens: list[Token]) -> str:
        parts: list[str] = []
        idx = 0
        while idx < len(tokens):
            token = tokens[idx]
            if token.type == "text":
                parts.append(token.content)
            elif token.type in {"softbreak", "hardbreak"}:
                parts.append("\n")
            elif token.type == "code_inline":
                parts.append(token.content)
            elif token.type == "html_inline":
                parts.append(token.content)
            elif token.type in {"strong_open", "em_open", "s_open", "link_open"}:
                closing_type = {
                    "strong_open": "strong_close",
                    "em_open": "em_close",
                    "s_open": "s_close",
                    "link_open": "link_close",
                }[token.type]
                inner, consumed = self._collect_inline(tokens, idx + 1, closing_type)
                parts.append(self._plain_inline_text(inner))
                idx += consumed + 2
                continue
            idx += 1
        return "".join(parts)

    def _render_inline(self, tokens: list[Token], style_context: _InlineStyleContext | None = None) -> str:
        context = style_context or self._default_inline_style_context()
        apply_text = context.apply_text
        style_prefix = context.style_prefix

        def apply_text_with_newlines(text: str) -> str:
            return "\n".join(str(apply_text(segment)) for segment in text.split("\n"))

        result = ""
        idx = 0
        while idx < len(tokens):
            token = tokens[idx]
            if token.type == "text":
                result += apply_text_with_newlines(token.content)
            elif token.type in {"softbreak", "hardbreak"}:
                result += "\n"
            elif token.type == "code_inline":
                result += _theme_call(self.theme, "code", token.content) + style_prefix
            elif token.type == "strong_open":
                inner, consumed = self._collect_inline(tokens, idx + 1, "strong_close")
                result += _theme_call(self.theme, "bold", self._render_inline(inner, context)) + style_prefix
                idx += consumed + 2
                continue
            elif token.type == "em_open":
                inner, consumed = self._collect_inline(tokens, idx + 1, "em_close")
                result += _theme_call(self.theme, "italic", self._render_inline(inner, context)) + style_prefix
                idx += consumed + 2
                continue
            elif token.type == "s_open":
                inner, consumed = self._collect_inline(tokens, idx + 1, "s_close")
                result += _theme_call(self.theme, "strikethrough", self._render_inline(inner, context)) + style_prefix
                idx += consumed + 2
                continue
            elif token.type == "link_open":
                href = token.attrGet("href") or ""
                inner, consumed = self._collect_inline(tokens, idx + 1, "link_close")
                link_text = self._render_inline(inner, context)
                raw_link_text = self._plain_inline_text(inner)
                href_for_comparison = href.removeprefix("mailto:")
                result += _theme_call(self.theme, "link", _theme_call(self.theme, "underline", link_text))
                if href and raw_link_text not in {href, href_for_comparison}:
                    result += _theme_call(self.theme, "link_url", f" ({href})")
                result += style_prefix
                idx += consumed + 2
                continue
            elif token.type == "html_inline":
                result += apply_text_with_newlines(token.content)
            idx += 1
        return result

    def _collect_inline(self, tokens: list[Token], start: int, closing_type: str) -> tuple[list[Token], int]:
        collected: list[Token] = []
        depth = 0
        idx = start
        while idx < len(tokens):
            token = tokens[idx]
            if token.type == closing_type and depth == 0:
                break
            if token.type.endswith("_open") and token.type.replace("_open", "_close") == closing_type:
                depth += 1
            elif token.type == closing_type:
                depth -= 1
            collected.append(token)
            idx += 1
        return collected, idx - start

    def _render_list(self, tokens: list[Token], start_index: int, depth: int) -> tuple[list[str], int]:
        lines: list[str] = []
        ordered = tokens[start_index].type == "ordered_list_open"
        start_number = int(tokens[start_index].attrGet("start") or "1")
        idx = start_index + 1
        item_index = 0
        while idx < len(tokens) and tokens[idx].type != ("ordered_list_close" if ordered else "bullet_list_close"):
            if tokens[idx].type != "list_item_open":
                idx += 1
                continue
            item_lines, idx = self._render_list_item(tokens, idx + 1, depth + 1)
            bullet = f"{start_number + item_index}. " if ordered else "- "
            indent = "  " * depth
            if item_lines:
                first_line = item_lines[0]
                if self._is_nested_list_line(first_line):
                    lines.append(first_line)
                else:
                    lines.append(indent + _theme_call(self.theme, "list_bullet", bullet) + first_line)
                for line in item_lines[1:]:
                    if self._is_nested_list_line(line):
                        lines.append(line)
                    else:
                        lines.append(f"{indent}  {line}")
            else:
                lines.append(indent + _theme_call(self.theme, "list_bullet", bullet))
            item_index += 1
        return lines, idx + 1

    def _render_list_item(self, tokens: list[Token], start_index: int, depth: int) -> tuple[list[str], int]:
        lines: list[str] = []
        idx = start_index
        while idx < len(tokens) and tokens[idx].type != "list_item_close":
            token = tokens[idx]
            if token.type == "paragraph_open":
                inline = tokens[idx + 1]
                lines.append(self._render_inline(inline.children or []))
                idx += 3
                continue
            if token.type in {"bullet_list_open", "ordered_list_open"}:
                nested_lines, idx = self._render_list(tokens, idx, depth)
                lines.extend(nested_lines)
                continue
            if token.type == "fence":
                info = token.info.strip()
                lines.append(_theme_call(self.theme, "code_block_border", f"```{info}"))
                for code_line in token.content.rstrip("\n").split("\n"):
                    lines.append(f"  {_theme_call(self.theme, 'code_block', code_line)}")
                lines.append(_theme_call(self.theme, "code_block_border", "```"))
            idx += 1
        return lines, idx + 1

    def _is_nested_list_line(self, line: str) -> bool:
        return bool(re.match(r"^\s+\x1b\[[0-9;]*m[-\d]", line))

    def _render_blockquote(self, tokens: list[Token], start_index: int, width: int) -> tuple[list[str], int]:
        inner: list[Token] = []
        idx = start_index + 1
        depth = 1
        while idx < len(tokens):
            token = tokens[idx]
            if token.type == "blockquote_open":
                depth += 1
            elif token.type == "blockquote_close":
                depth -= 1
                if depth == 0:
                    break
            inner.append(token)
            idx += 1

        # Blockquotes should use quote styling only and not inherit the
        # default text color/style used for surrounding assistant content.
        previous_default_text_style = self.default_text_style
        self.default_text_style = None
        try:
            quote_lines = self._render_tokens(inner, max(1, width - 2))
        finally:
            self.default_text_style = previous_default_text_style
        while quote_lines and quote_lines[-1] == "":
            quote_lines.pop()

        def quote_style(text: str) -> str:
            return _theme_call(self.theme, "quote", _theme_call(self.theme, "italic", text))

        quote_style_prefix = self._style_prefix(quote_style)
        rendered: list[str] = []
        for line in quote_lines:
            styled_line = quote_style(self._reapply_style_prefix(line, quote_style_prefix))
            for wrapped_line in wrap_text_with_ansi(styled_line, max(1, width - 2)):
                rendered.append(_theme_call(self.theme, "quote_border", "│ ") + wrapped_line)
        return rendered, idx + 1

    def _render_table(self, tokens: list[Token], start_index: int, available_width: int) -> tuple[list[str], int]:
        idx = start_index + 1
        headers: list[str] = []
        rows: list[list[str]] = []
        current_row: list[str] | None = None
        current_cell: list[Token] = []
        in_header = True

        while idx < len(tokens):
            token = tokens[idx]
            if token.type == "table_close":
                break
            if token.type == "thead_close":
                in_header = False
            elif token.type == "tr_open":
                current_row = []
            elif token.type in {"th_close", "td_close"}:
                assert current_row is not None
                current_row.append(self._render_inline(current_cell))
                current_cell = []
            elif token.type == "tr_close":
                if current_row is not None:
                    if in_header:
                        headers = current_row
                    else:
                        rows.append(current_row)
                current_row = None
            elif token.type == "inline":
                current_cell = token.children or []
            idx += 1

        if not headers:
            return [], idx + 1

        num_cols = len(headers)
        border_overhead = 3 * num_cols + 1
        available_for_cells = available_width - border_overhead
        if available_for_cells < num_cols:
            return wrap_text_with_ansi(self.text, available_width), idx + 1

        widths = [max(visible_width(headers[i]), *(visible_width(row[i]) for row in rows if i < len(row))) for i in range(num_cols)]
        total = sum(widths)
        if total > available_for_cells:
            scale = available_for_cells / total
            widths = [max(1, int(width * scale)) for width in widths]
            while sum(widths) < available_for_cells:
                for i in range(num_cols):
                    widths[i] += 1
                    if sum(widths) >= available_for_cells:
                        break

        def render_row(cells: list[str]) -> list[str]:
            wrapped = [wrap_text_with_ansi(cells[i] if i < len(cells) else "", max(1, widths[i])) for i in range(num_cols)]
            height = max(len(cell_lines) for cell_lines in wrapped)
            output: list[str] = []
            for line_index in range(height):
                parts = []
                for col in range(num_cols):
                    text = wrapped[col][line_index] if line_index < len(wrapped[col]) else ""
                    parts.append(text + " " * max(0, widths[col] - visible_width(text)))
                output.append(f"│ {' │ '.join(parts)} │")
            return output

        lines = [f"┌─{'─┬─'.join('─' * width for width in widths)}─┐"]
        lines.extend(render_row(headers))
        lines.append(f"├─{'─┼─'.join('─' * width for width in widths)}─┤")
        for row_index, row in enumerate(rows):
            lines.extend(render_row(row))
            if row_index < len(rows) - 1:
                lines.append(f"├─{'─┼─'.join('─' * width for width in widths)}─┤")
        lines.append(f"└─{'─┴─'.join('─' * width for width in widths)}─┘")
        return lines, idx + 1
