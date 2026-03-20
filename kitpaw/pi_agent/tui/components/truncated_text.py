from __future__ import annotations

from ..utils import truncate_to_width, visible_width


class TruncatedText:
    def __init__(self, text: str, padding_x: int = 0, padding_y: int = 0) -> None:
        self.text = text
        self.padding_x = padding_x
        self.padding_y = padding_y

    def invalidate(self) -> None:
        return None

    def render(self, width: int) -> list[str]:
        result = [" " * width for _ in range(self.padding_y)]
        available_width = max(1, width - self.padding_x * 2)
        single_line_text = self.text.split("\n", 1)[0]
        display_text = truncate_to_width(single_line_text, available_width)
        line_with_padding = (" " * self.padding_x) + display_text + (" " * self.padding_x)
        result.append(line_with_padding + " " * max(0, width - visible_width(line_with_padding)))
        result.extend([" " * width for _ in range(self.padding_y)])
        return result
