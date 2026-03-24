from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

from ....agent.types import AgentTool, AgentToolResult
from ....ai.types import TextContent
from ._path_utils import resolve_safe_path


def create_copy_path_tool(cwd: str) -> AgentTool:
    """Copy a file or directory to a new location."""

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
            def _copy():
                dst.parent.mkdir(parents=True, exist_ok=True)
                if src.is_dir():
                    shutil.copytree(str(src), str(dst))
                else:
                    shutil.copy2(str(src), str(dst))

            await loop.run_in_executor(None, _copy)

            return AgentToolResult(
                content=[TextContent(text=f"Copied {source_path} to {destination_path}")],
                details=None,
            )
        except Exception as e:
            return AgentToolResult(
                content=[TextContent(text=f"Error copying path: {e}")],
                details=None,
            )

    return AgentTool(
        name="copy_path",
        label="Copy Path",
        description=(
            "Copy a file or directory to a new location. "
            "Directories are copied recursively. "
            "Parent directories of the destination are created automatically."
        ),
        parameters={
            "type": "object",
            "properties": {
                "source_path": {
                    "type": "string",
                    "description": "The source path to copy from, relative to the project root.",
                },
                "destination_path": {
                    "type": "string",
                    "description": "The destination path to copy to, relative to the project root.",
                },
            },
            "required": ["source_path", "destination_path"],
        },
        execute=execute,
    )
