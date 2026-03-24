from __future__ import annotations

import asyncio
from pathlib import Path

from ....agent.types import AgentTool, AgentToolResult
from ....ai.types import TextContent
from ._path_utils import resolve_safe_path


def create_create_directory_tool(cwd: str) -> AgentTool:
    """Create a new directory."""

    async def execute(tool_call_id, args, cancel_event=None, on_update=None):
        path = args.get("path", "")
        try:
            resolved = resolve_safe_path(cwd, path)
        except ValueError as e:
            return AgentToolResult(content=[TextContent(text=f"Error: {e}")], details=None)

        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, lambda: resolved.mkdir(parents=True, exist_ok=True))
            return AgentToolResult(
                content=[TextContent(text=f"Created directory: {path}")],
                details=None,
            )
        except Exception as e:
            return AgentToolResult(
                content=[TextContent(text=f"Error creating directory: {e}")],
                details=None,
            )

    return AgentTool(
        name="create_directory",
        label="Create Directory",
        description=(
            "Create a new directory at the specified path. "
            "Parent directories are created automatically if they don't exist."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The path of the directory to create, relative to the project root.",
                },
            },
            "required": ["path"],
        },
        execute=execute,
    )
