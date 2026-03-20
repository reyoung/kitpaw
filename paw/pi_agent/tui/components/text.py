from __future__ import annotations

from ..utils import apply_background_to_line, visible_width, wrap_text_with_ansi


class Text:
    def __init__(self, text: str = "", padding_x: int = 1, padding_y: int = 1, custom_bg_fn=None) -> None:
        self.text = text
        self.padding_x = padding_x
        self.padding_y = padding_y
        self.custom_bg_fn = custom_bg_fn
        self.cached_text: str | None = None
        self.cached_width: int | None = None
        self.cached_lines: list[str] | None = None

    def set_text(self, text: str) -> None:
        self.text = text
        self.invalidate()

    def set_custom_bg_fn(self, custom_bg_fn=None) -> None:
        self.custom_bg_fn = custom_bg_fn
        self.invalidate()

    def invalidate(self) -> None:
        self.cached_text = None
        self.cached_width = None
        self.cached_lines = None

    def render(self, width: int) -> list[str]:
        if self.cached_lines is not None and self.cached_text == self.text and self.cached_width == width:
            return self.cached_lines
        if not self.text or self.text.strip() == "":
            self.cached_text = self.text
            self.cached_width = width
            self.cached_lines = []
            return []
        normalized_text = self.text.replace("\t", "   ")
        content_width = max(1, width - self.padding_x * 2)
        wrapped_lines = wrap_text_with_ansi(normalized_text, content_width)
        left_margin = " " * self.padding_x
        right_margin = " " * self.padding_x
        content_lines: list[str] = []
        for line in wrapped_lines:
            line_with_margins = left_margin + line + right_margin
            if self.custom_bg_fn:
                content_lines.append(apply_background_to_line(line_with_margins, width, self.custom_bg_fn))
            else:
                content_lines.append(line_with_margins + " " * max(0, width - visible_width(line_with_margins)))
        empty_line = " " * width
        empty_lines = [apply_background_to_line(empty_line, width, self.custom_bg_fn) if self.custom_bg_fn else empty_line for _ in range(self.padding_y)]
        result = [*empty_lines, *content_lines, *empty_lines]
        self.cached_text = self.text
        self.cached_width = width
        self.cached_lines = result
        return result or [""]
