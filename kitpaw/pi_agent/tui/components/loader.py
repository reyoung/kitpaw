from __future__ import annotations

from .text import Text


class Loader(Text):
    def __init__(self, ui, spinner_color_fn, message_color_fn, message: str = "Loading...") -> None:
        super().__init__("", 1, 0)
        self.frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self.current_frame = 0
        self.ui = ui
        self.spinner_color_fn = spinner_color_fn
        self.message_color_fn = message_color_fn
        self.message = message
        self.update_display()

    def render(self, width: int) -> list[str]:
        return [""] + super().render(width)

    def set_message(self, message: str) -> None:
        self.message = message
        self.update_display()

    def update_display(self) -> None:
        frame = self.frames[self.current_frame]
        self.set_text(f"{self.spinner_color_fn(frame)} {self.message_color_fn(self.message)}")
        if self.ui:
            self.ui.request_render()
