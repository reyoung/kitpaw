from __future__ import annotations

from dataclasses import dataclass

from ..terminal_image import (
    ImageDimensions,
    ImageRenderOptions,
    get_capabilities,
    get_image_dimensions,
    image_fallback,
    render_image,
)


@dataclass(slots=True)
class ImageTheme:
    fallback_color: callable | None = None


@dataclass(slots=True)
class ImageOptions:
    max_width_cells: int | None = None
    max_height_cells: int | None = None
    filename: str | None = None
    image_id: int | None = None


class Image:
    def __init__(
        self,
        base64_data: str,
        mime_type: str,
        theme: ImageTheme | None = None,
        options: ImageOptions | None = None,
        dimensions: ImageDimensions | None = None,
    ) -> None:
        self.base64_data = base64_data
        self.mime_type = mime_type
        self.theme = theme or ImageTheme()
        self.options = options or ImageOptions()
        self.dimensions = dimensions or get_image_dimensions(base64_data, mime_type) or ImageDimensions(width_px=800, height_px=600)
        self.image_id = self.options.image_id
        self.cached_lines: list[str] | None = None
        self.cached_width: int | None = None

    def get_image_id(self) -> int | None:
        return self.image_id

    def invalidate(self) -> None:
        self.cached_lines = None
        self.cached_width = None

    def render(self, width: int) -> list[str]:
        if self.cached_lines is not None and self.cached_width == width:
            return self.cached_lines

        max_width = min(width - 2, self.options.max_width_cells or 60)
        caps = get_capabilities()
        if caps.images:
            result = render_image(
                self.base64_data,
                self.dimensions,
                ImageRenderOptions(
                    max_width_cells=max_width,
                    max_height_cells=self.options.max_height_cells,
                    image_id=self.image_id,
                ),
            )
            if result:
                image_id = result.get("image_id")
                if isinstance(image_id, int):
                    self.image_id = image_id
                rows = int(result["rows"])
                sequence = str(result["sequence"])
                lines = [""] * max(0, rows - 1)
                move_up = f"\x1b[{rows - 1}A" if rows > 1 else ""
                lines.append(move_up + sequence)
                self.cached_lines = lines
                self.cached_width = width
                return lines

        fallback = image_fallback(self.mime_type, self.dimensions, self.options.filename)
        if callable(self.theme.fallback_color):
            fallback = str(self.theme.fallback_color(fallback))
        self.cached_lines = [fallback]
        self.cached_width = width
        return self.cached_lines
