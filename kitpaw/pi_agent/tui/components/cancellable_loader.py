from __future__ import annotations

from .loader import Loader


class CancellableLoader(Loader):
    def __init__(self, ui, spinner_color_fn, message_color_fn, message: str = "Loading...") -> None:
        super().__init__(ui, spinner_color_fn, message_color_fn, message)
        self.cancelled = False

    def cancel(self) -> None:
        self.cancelled = True
        self.set_message("Cancelled")
