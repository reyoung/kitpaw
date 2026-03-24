from __future__ import annotations

from pathlib import Path

import aiofiles

from ....agent.types import AgentTool, AgentToolResult
from ....ai.types import TextContent
from ._path_utils import resolve_safe_path


def create_read_file_tool(cwd: str) -> AgentTool:
    """Read the contents of a file on disk."""

    async def execute(tool_call_id, args, cancel_event=None, on_update=None):
        path = args.get("path", "")
        start_line = args.get("start_line")
        end_line = args.get("end_line")

        try:
            resolved = resolve_safe_path(cwd, path)
        except ValueError as e:
            return AgentToolResult(content=[TextContent(text=f"Error: {e}")], details=None)
        try:
            async with aiofiles.open(resolved, encoding="utf-8", errors="replace") as f:
                text = await f.read()
        except FileNotFoundError:
            return AgentToolResult(
                content=[TextContent(text=f"Error: File not found: {resolved}")],
                details=None,
            )
        except IsADirectoryError:
            return AgentToolResult(
                content=[TextContent(text=f"Error: Path is a directory: {resolved}")],
                details=None,
            )
        except Exception as e:
            return AgentToolResult(
                content=[TextContent(text=f"Error reading file: {e}")],
                details=None,
            )

        lines = text.splitlines(keepends=True)
        total = len(lines)

        if start_line is not None or end_line is not None:
            s = max((start_line or 1) - 1, 0)
            e = end_line if end_line is not None else total
            selected = lines[s:e]
            numbered = "".join(
                f"{i + s + 1:>6}\t{line}" for i, line in enumerate(selected)
            )
            result = f"({total} lines total)\n{numbered}"
        else:
            numbered = "".join(
                f"{i + 1:>6}\t{line}" for i, line in enumerate(lines)
            )
            result = numbered

        return AgentToolResult(content=[TextContent(text=result)], details=None)

    return AgentTool(
        name="read_file",
        label="Read File",
        description=(
            "Read the contents of a file on disk. "
            "Returns the file content with line numbers. "
            "Use start_line and end_line to read a specific range of lines."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The path of the file to read, relative to the project root.",
                },
                "start_line": {
                    "type": "integer",
                    "description": "The 1-based line number to start reading from (inclusive).",
                },
                "end_line": {
                    "type": "integer",
                    "description": "The 1-based line number to stop reading at (inclusive).",
                },
            },
            "required": ["path"],
        },
        execute=execute,
    )
