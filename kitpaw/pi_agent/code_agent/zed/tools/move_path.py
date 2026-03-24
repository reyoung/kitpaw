from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

from ....agent.types import AgentTool, AgentToolResult
from ....ai.types import TextContent
from ._path_utils import resolve_safe_path


def create_move_path_tool(cwd: str) -> AgentTool:
    """Move or rename a file or directory."""

    async def execute(tool_call_id, args, cancel_event=None, on_update=None):
        source_path = args.get("source_path", "")
        destination_path = args.get("destination_path", "")

        try:
            src = resolve_safe_path(cwd, source_path)
            dst = resolve_safe_path(cwd, destination_path)
        except ValueError as e:
            return AgentToolResult(content=[TextContent(text=f"Error: {e}")], details=None)

        loop = asyncio.get_running_loop()

        exists = await loop.run_in_executor(None, src.exists)
        if not exists:
            return AgentToolResult(
                content=[TextContent(text=f"Error: Source path does not exist: {source_path}")],
                details=None,
            )

        try:
            def _move():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dst))

            await loop.run_in_executor(None, _move)
            return AgentToolResult(
                content=[TextContent(text=f"Moved {source_path} to {destination_path}")],
                details=None,
            )
        except Exception as e:
            return AgentToolResult(
                content=[TextContent(text=f"Error moving path: {e}")],
                details=None,
            )

    return AgentTool(
        name="move_path",
        label="Move Path",
        description=(
            "Move or rename a file or directory. "
            "Parent directories of the destination are created automatically."
        ),
        parameters={
            "type": "object",
            "properties": {
                "source_path": {
                    "type": "string",
                    "description": "The source path to move from, relative to the project root.",
                },
                "destination_path": {
                    "type": "string",
                    "description": "The destination path to move to, relative to the project root.",
                },
            },
            "required": ["source_path", "destination_path"],
        },
        execute=execute,
    )
