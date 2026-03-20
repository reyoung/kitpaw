from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class KillRing:
    """Ring buffer for Emacs-style kill/yank operations."""

    _ring: list[str] = field(default_factory=list)

    def push(self, text: str, *, prepend: bool, accumulate: bool = False) -> None:
        if not text:
            return

        if accumulate and self._ring:
            last = self._ring.pop()
            self._ring.append(text + last if prepend else last + text)
            return

        self._ring.append(text)

    def peek(self) -> str | None:
        if not self._ring:
            return None
        return self._ring[-1]

    def rotate(self) -> None:
        if len(self._ring) <= 1:
            return
        last = self._ring.pop()
        self._ring.insert(0, last)

    @property
    def length(self) -> int:
        return len(self._ring)
