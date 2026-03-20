from __future__ import annotations

import base64
from pathlib import Path

from ...agent.types import AgentTool, AgentToolResult
from ...ai.types import ImageContent, TextContent
from .path_utils import resolve_read_path
from .truncate import DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, format_size, truncate_head


def _detect_image_mime(data: bytes) -> str | None:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"GIF87a") or data.startswith(b"GIF89a"):
        return "image/gif"
    if data.startswith(b"RIFF") and b"WEBP" in data[:16]:
        return "image/webp"
    return None


def create_read_tool(cwd: str) -> AgentTool[dict[str, object], dict[str, object] | None]:
    async def execute(_tool_call_id: str, args: dict[str, object], *_args) -> AgentToolResult[dict[str, object] | None]:
        path = str(args["path"])
        offset = int(args["offset"]) if args.get("offset") is not None else None
        limit = int(args["limit"]) if args.get("limit") is not None else None
        absolute_path = Path(resolve_read_path(path, cwd))
        data = absolute_path.read_bytes()
        mime_type = _detect_image_mime(data)
        if mime_type:
            return AgentToolResult(
                content=[
                    TextContent(text=f"Read image file [{mime_type}]"),
                    ImageContent(data=base64.b64encode(data).decode("ascii"), mime_type=mime_type),
                ],
                details=None,
            )

        text = data.decode("utf-8")
        all_lines = text.split("\n")
        start = max(0, (offset or 1) - 1)
        if start >= len(all_lines):
            raise ValueError(f"Offset {offset} is beyond end of file ({len(all_lines)} lines total)")
        selected = all_lines[start : start + limit] if limit is not None else all_lines[start:]
        truncation = truncate_head("\n".join(selected))
        output = truncation.content
        details: dict[str, object] | None = None
        if truncation.first_line_exceeds_limit:
            line_display = start + 1
            output = (
                f"[Line {line_display} is {format_size(len(all_lines[start].encode('utf-8')))}, exceeds "
                f"{format_size(DEFAULT_MAX_BYTES)} limit. Use bash: sed -n '{line_display}p' {path} | head -c {DEFAULT_MAX_BYTES}]"
            )
            details = {"truncation": truncation.__dict__}
        elif truncation.truncated:
            end_display = start + truncation.output_lines
            output += (
                f"\n\n[Showing lines {start + 1}-{end_display} of {len(all_lines)}. "
                f"Use offset={end_display + 1} to continue.]"
            )
            details = {"truncation": truncation.__dict__}
        elif limit is not None and start + len(selected) < len(all_lines):
            output += (
                f"\n\n[{len(all_lines) - (start + len(selected))} more lines in file. "
                f"Use offset={start + len(selected) + 1} to continue.]"
            )
        return AgentToolResult(content=[TextContent(text=output)], details=details)

    return AgentTool(
        name="read",
        label="read",
        description=(
            f"Read the contents of a file. Output is truncated to {DEFAULT_MAX_LINES} lines or "
            f"{DEFAULT_MAX_BYTES // 1024}KB. Use offset/limit for large files."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to read"},
                "offset": {"type": "number", "description": "1-indexed line offset"},
                "limit": {"type": "number", "description": "Maximum number of lines"},
            },
            "required": ["path"],
        },
        execute=execute,
    )
