from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Generic, TypeVar

StateT = TypeVar("StateT")


@dataclass(slots=True)
class UndoStack(Generic[StateT]):
    """Generic undo stack with clone-on-push semantics."""

    _stack: list[StateT] = field(default_factory=list)

    def push(self, state: StateT) -> None:
        self._stack.append(deepcopy(state))

    def pop(self) -> StateT | None:
        if not self._stack:
            return None
        return self._stack.pop()

    def clear(self) -> None:
        self._stack.clear()

    @property
    def length(self) -> int:
        return len(self._stack)
