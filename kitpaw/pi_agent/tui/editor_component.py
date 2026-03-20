from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class EditorComponent(Protocol):
    """Protocol for custom editor implementations."""

    on_submit: object | None
    on_change: object | None

    def render(self, width: int) -> list[str]:
        raise NotImplementedError

    def invalidate(self) -> None:
        raise NotImplementedError

    def get_text(self) -> str:
        raise NotImplementedError

    def set_text(self, text: str) -> None:
        raise NotImplementedError

    def handle_input(self, data: str) -> None:
        raise NotImplementedError

    def add_to_history(self, text: str) -> None:
        raise NotImplementedError

    def insert_text_at_cursor(self, text: str) -> None:
        raise NotImplementedError

    def get_expanded_text(self) -> str:
        raise NotImplementedError

    def set_autocomplete_provider(self, provider: object) -> None:
        raise NotImplementedError

    def set_padding_x(self, padding: int) -> None:
        raise NotImplementedError

    def set_autocomplete_max_visible(self, max_visible: int) -> None:
        raise NotImplementedError
