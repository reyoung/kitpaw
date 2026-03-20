from __future__ import annotations


class Spacer:
    def __init__(self, lines: int = 1) -> None:
        self.lines = lines

    def set_lines(self, lines: int) -> None:
        self.lines = lines

    def invalidate(self) -> None:
        return None

    def render(self, _width: int) -> list[str]:
        return [""] * self.lines
