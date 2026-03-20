from __future__ import annotations

import base64
import os
import struct
from dataclasses import dataclass

ImageProtocol = str | None


@dataclass(slots=True)
class TerminalCapabilities:
    images: ImageProtocol
    true_color: bool
    hyperlinks: bool


@dataclass(slots=True)
class CellDimensions:
    width_px: int
    height_px: int


@dataclass(slots=True)
class ImageDimensions:
    width_px: int
    height_px: int


@dataclass(slots=True)
class ImageRenderOptions:
    max_width_cells: int | None = None
    max_height_cells: int | None = None
    preserve_aspect_ratio: bool = True
    image_id: int | None = None


_cached_capabilities: TerminalCapabilities | None = None
_cell_dimensions = CellDimensions(width_px=9, height_px=18)


def get_cell_dimensions() -> CellDimensions:
    return _cell_dimensions


def set_cell_dimensions(dims: CellDimensions) -> None:
    global _cell_dimensions
    _cell_dimensions = dims


def detect_capabilities() -> TerminalCapabilities:
    term_program = os.getenv("TERM_PROGRAM", "").lower()
    term = os.getenv("TERM", "").lower()
    color_term = os.getenv("COLORTERM", "").lower()
    if os.getenv("KITTY_WINDOW_ID") or term_program == "kitty":
        return TerminalCapabilities(images="kitty", true_color=True, hyperlinks=True)
    if term_program == "ghostty" or "ghostty" in term or os.getenv("GHOSTTY_RESOURCES_DIR"):
        return TerminalCapabilities(images="kitty", true_color=True, hyperlinks=True)
    if os.getenv("WEZTERM_PANE") or term_program == "wezterm":
        return TerminalCapabilities(images="kitty", true_color=True, hyperlinks=True)
    if os.getenv("ITERM_SESSION_ID") or term_program == "iterm.app":
        return TerminalCapabilities(images="iterm2", true_color=True, hyperlinks=True)
    if term_program in {"vscode", "alacritty"}:
        return TerminalCapabilities(images=None, true_color=True, hyperlinks=True)
    return TerminalCapabilities(images=None, true_color=color_term in {"truecolor", "24bit"}, hyperlinks=True)


def get_capabilities() -> TerminalCapabilities:
    global _cached_capabilities
    if _cached_capabilities is None:
        _cached_capabilities = detect_capabilities()
    return _cached_capabilities


def reset_capabilities_cache() -> None:
    global _cached_capabilities
    _cached_capabilities = None


KITTY_PREFIX = "\x1b_G"
ITERM2_PREFIX = "\x1b]1337;File="


def is_image_line(line: str) -> bool:
    return line.startswith(KITTY_PREFIX) or line.startswith(ITERM2_PREFIX) or KITTY_PREFIX in line or ITERM2_PREFIX in line


def allocate_image_id() -> int:
    return 1 + int.from_bytes(os.urandom(4), "big") % 0xFFFFFFFE


def encode_kitty(base64_data: str, *, columns: int | None = None, rows: int | None = None, image_id: int | None = None) -> str:
    params = ["a=T", "f=100", "q=2"]
    if columns:
        params.append(f"c={columns}")
    if rows:
        params.append(f"r={rows}")
    if image_id:
        params.append(f"i={image_id}")
    if len(base64_data) <= 4096:
        return f"\x1b_G{','.join(params)};{base64_data}\x1b\\"
    chunks: list[str] = []
    offset = 0
    first = True
    while offset < len(base64_data):
        chunk = base64_data[offset : offset + 4096]
        last = offset + 4096 >= len(base64_data)
        if first:
            chunks.append(f"\x1b_G{','.join(params)},m=1;{chunk}\x1b\\")
            first = False
        elif last:
            chunks.append(f"\x1b_Gm=0;{chunk}\x1b\\")
        else:
            chunks.append(f"\x1b_Gm=1;{chunk}\x1b\\")
        offset += 4096
    return "".join(chunks)


def delete_kitty_image(image_id: int) -> str:
    return f"\x1b_Ga=d,d=I,i={image_id}\x1b\\"


def delete_all_kitty_images() -> str:
    return "\x1b_Ga=d,d=A\x1b\\"


def encode_iterm2(base64_data: str, *, width: int | str | None = None, height: int | str | None = None, name: str | None = None, preserve_aspect_ratio: bool = True, inline: bool = True) -> str:
    params = [f"inline={1 if inline else 0}"]
    if width is not None:
        params.append(f"width={width}")
    if height is not None:
        params.append(f"height={height}")
    if name:
        params.append(f"name={base64.b64encode(name.encode()).decode()}")
    if not preserve_aspect_ratio:
        params.append("preserveAspectRatio=0")
    return f"\x1b]1337;File={';'.join(params)}:{base64_data}\x07"


def calculate_image_rows(image_dimensions: ImageDimensions, target_width_cells: int, cell_dimensions: CellDimensions | None = None) -> int:
    dims = cell_dimensions or _cell_dimensions
    target_width_px = target_width_cells * dims.width_px
    scale = target_width_px / image_dimensions.width_px
    scaled_height_px = image_dimensions.height_px * scale
    rows = int((scaled_height_px + dims.height_px - 1) // dims.height_px)
    return max(1, rows)


def get_png_dimensions(base64_data: str) -> ImageDimensions | None:
    try:
        buffer = base64.b64decode(base64_data)
        if len(buffer) < 24 or buffer[:4] != b"\x89PNG":
            return None
        width, height = struct.unpack(">II", buffer[16:24])
        return ImageDimensions(width_px=width, height_px=height)
    except Exception:
        return None


def get_jpeg_dimensions(base64_data: str) -> ImageDimensions | None:
    try:
        buffer = base64.b64decode(base64_data)
        if len(buffer) < 2 or buffer[:2] != b"\xff\xd8":
            return None
        offset = 2
        while offset < len(buffer) - 9:
            if buffer[offset] != 0xFF:
                offset += 1
                continue
            marker = buffer[offset + 1]
            if marker in {0xC0, 0xC1, 0xC2}:
                height, width = struct.unpack(">HH", buffer[offset + 5 : offset + 9])
                return ImageDimensions(width_px=width, height_px=height)
            if offset + 3 >= len(buffer):
                return None
            length = struct.unpack(">H", buffer[offset + 2 : offset + 4])[0]
            if length < 2:
                return None
            offset += 2 + length
        return None
    except Exception:
        return None


def get_gif_dimensions(base64_data: str) -> ImageDimensions | None:
    try:
        buffer = base64.b64decode(base64_data)
        if len(buffer) < 10:
            return None
        signature = buffer[:6].decode("ascii", errors="ignore")
        if signature not in {"GIF87a", "GIF89a"}:
            return None
        width, height = struct.unpack("<HH", buffer[6:10])
        return ImageDimensions(width_px=width, height_px=height)
    except Exception:
        return None


def get_webp_dimensions(base64_data: str) -> ImageDimensions | None:
    try:
        buffer = base64.b64decode(base64_data)
        if len(buffer) < 30:
            return None
        if buffer[:4] != b"RIFF" or buffer[8:12] != b"WEBP":
            return None
        chunk = buffer[12:16]
        if chunk == b"VP8 ":
            width, height = struct.unpack("<HH", buffer[26:30])
            return ImageDimensions(width_px=width & 0x3FFF, height_px=height & 0x3FFF)
        if chunk == b"VP8L":
            if len(buffer) < 25:
                return None
            bits = struct.unpack("<I", buffer[21:25])[0]
            width = (bits & 0x3FFF) + 1
            height = ((bits >> 14) & 0x3FFF) + 1
            return ImageDimensions(width_px=width, height_px=height)
        if chunk == b"VP8X":
            width = buffer[24] | (buffer[25] << 8) | (buffer[26] << 16)
            height = buffer[27] | (buffer[28] << 8) | (buffer[29] << 16)
            return ImageDimensions(width_px=width + 1, height_px=height + 1)
        return None
    except Exception:
        return None


def get_image_dimensions(base64_data: str, mime_type: str) -> ImageDimensions | None:
    if mime_type == "image/png":
        return get_png_dimensions(base64_data)
    if mime_type == "image/jpeg":
        return get_jpeg_dimensions(base64_data)
    if mime_type == "image/gif":
        return get_gif_dimensions(base64_data)
    if mime_type == "image/webp":
        return get_webp_dimensions(base64_data)
    return None


def render_image(
    base64_data: str,
    image_dimensions: ImageDimensions,
    options: ImageRenderOptions | None = None,
) -> dict[str, str | int | None] | None:
    caps = get_capabilities()
    if not caps.images:
        return None

    resolved = options or ImageRenderOptions()
    max_width = resolved.max_width_cells or 80
    rows = calculate_image_rows(image_dimensions, max_width, get_cell_dimensions())
    if resolved.max_height_cells is not None:
        rows = min(rows, resolved.max_height_cells)
        rows = max(1, rows)

    if caps.images == "kitty":
        sequence = encode_kitty(
            base64_data,
            columns=max_width,
            rows=rows,
            image_id=resolved.image_id,
        )
        return {"sequence": sequence, "rows": rows, "image_id": resolved.image_id}

    if caps.images == "iterm2":
        sequence = encode_iterm2(
            base64_data,
            width=max_width,
            height="auto",
            preserve_aspect_ratio=resolved.preserve_aspect_ratio,
        )
        return {"sequence": sequence, "rows": rows, "image_id": None}

    return None


def image_fallback(mime_type: str, dimensions: ImageDimensions | None = None, filename: str | None = None) -> str:
    parts: list[str] = []
    if filename:
        parts.append(filename)
    parts.append(f"[{mime_type}]")
    if dimensions:
        parts.append(f"{dimensions.width_px}x{dimensions.height_px}")
    return f"[Image: {' '.join(parts)}]"
