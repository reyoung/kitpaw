from __future__ import annotations

import asyncio
from pathlib import Path

from ....agent.types import AgentTool, AgentToolResult
from ....ai.types import TextContent
from ._path_utils import resolve_safe_path


def create_list_directory_tool(cwd: str) -> AgentTool:
    """List the contents of a directory."""

    async def execute(tool_call_id, args, cancel_event=None, on_update=None):
        path = args.get("path", ".")
        try:
            resolved = resolve_safe_path(cwd, path)
        except ValueError as e:
            return AgentToolResult(content=[TextContent(text=f"Error: {e}")], details=None)

        loop = asyncio.get_running_loop()

        exists = await loop.run_in_executor(None, resolved.exists)
        if not exists:
            return AgentToolResult(
                content=[TextContent(text=f"Error: Path does not exist: {resolved}")],
                details=None,
            )
        is_dir = await loop.run_in_executor(None, resolved.is_dir)
        if not is_dir:
            return AgentToolResult(
                content=[TextContent(text=f"Error: Path is not a directory: {resolved}")],
                details=None,
            )

        try:
            def _list_dir():
                entries = sorted(resolved.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
                lines: list[str] = []
                for entry in entries:
                    suffix = "/" if entry.is_dir() else ""
                    lines.append(f"{entry.name}{suffix}")
                return lines

            lines = await loop.run_in_executor(None, _list_dir)

            if not lines:
                return AgentToolResult(
                    content=[TextContent(text="Directory is empty.")],
                    details=None,
                )

            return AgentToolResult(
                content=[TextContent(text="\n".join(lines))],
                details=None,
            )
        except PermissionError:
            return AgentToolResult(
                content=[TextContent(text=f"Error: Permission denied: {resolved}")],
                details=None,
            )
        except Exception as e:
            return AgentToolResult(
                content=[TextContent(text=f"Error listing directory: {e}")],
                details=None,
            )

    return AgentTool(
        name="list_directory",
        label="List Directory",
        description=(
            "List the contents of a directory. "
            "Returns directory entries sorted with directories first, then files, both in alphabetical order. "
            "Directories are marked with a trailing '/'."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The path of the directory to list, relative to the project root.",
                },
            },
            "required": ["path"],
        },
        execute=execute,
    )
