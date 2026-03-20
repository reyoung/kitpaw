from __future__ import annotations

from ..utils import apply_background_to_line, visible_width


class Box:
    def __init__(self, padding_x: int = 1, padding_y: int = 1, bg_fn=None) -> None:
        self.children: list[object] = []
        self.padding_x = padding_x
        self.padding_y = padding_y
        self.bg_fn = bg_fn
        self._cache: tuple[int, str | None, list[str], list[str]] | None = None

    def add_child(self, component) -> None:
        self.children.append(component)
        self._cache = None

    def remove_child(self, component) -> None:
        if component in self.children:
            self.children.remove(component)
            self._cache = None

    def clear(self) -> None:
        self.children = []
        self._cache = None

    def set_bg_fn(self, bg_fn=None) -> None:
        self.bg_fn = bg_fn

    def invalidate(self) -> None:
        self._cache = None
        for child in self.children:
            if hasattr(child, "invalidate"):
                child.invalidate()

    def render(self, width: int) -> list[str]:
        if not self.children:
            return []
        content_width = max(1, width - self.padding_x * 2)
        left_pad = " " * self.padding_x
        child_lines: list[str] = []
        for child in self.children:
            for line in child.render(content_width):
                child_lines.append(left_pad + line)
        bg_sample = self.bg_fn("test") if self.bg_fn else None
        if self._cache and self._cache[0] == width and self._cache[1] == bg_sample and self._cache[2] == child_lines:
            return self._cache[3]
        result = []
        for _ in range(self.padding_y):
            result.append(self._apply_bg("", width))
        for line in child_lines:
            result.append(self._apply_bg(line, width))
        for _ in range(self.padding_y):
            result.append(self._apply_bg("", width))
        self._cache = (width, bg_sample, child_lines, result)
        return result

    def _apply_bg(self, line: str, width: int) -> str:
        vis_len = visible_width(line)
        padded = line + " " * max(0, width - vis_len)
        return apply_background_to_line(padded, width, self.bg_fn) if self.bg_fn else padded
