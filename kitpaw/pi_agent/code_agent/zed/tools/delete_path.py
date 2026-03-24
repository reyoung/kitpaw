from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

from ....agent.types import AgentTool, AgentToolResult
from ....ai.types import TextContent
from ._path_utils import resolve_safe_path


def create_delete_path_tool(cwd: str) -> AgentTool:
    """Delete a file or directory."""

    async def execute(tool_call_id, args, cancel_event=None, on_update=None):
        path = args.get("path", "")
        try:
            resolved = resolve_safe_path(cwd, path)
        except ValueError as e:
            return AgentToolResult(content=[TextContent(text=f"Error: {e}")], details=None)

        loop = asyncio.get_running_loop()

        exists = await loop.run_in_executor(None, resolved.exists)
        if not exists:
            return AgentToolResult(
                content=[TextContent(text=f"Error: Path does not exist: {path}")],
                details=None,
            )

        try:
            def _delete():
                if resolved.is_dir():
                    shutil.rmtree(str(resolved))
                else:
                    resolved.unlink()

            await loop.run_in_executor(None, _delete)

            return AgentToolResult(
                content=[TextContent(text=f"Deleted: {path}")],
                details=None,
            )
        except Exception as e:
            return AgentToolResult(
                content=[TextContent(text=f"Error deleting path: {e}")],
                details=None,
            )

    return AgentTool(
        name="delete_path",
        label="Delete Path",
        description=(
            "Delete a file or directory. "
            "Directories are deleted recursively including all contents."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The path to delete, relative to the project root.",
                },
            },
            "required": ["path"],
        },
        execute=execute,
    )
