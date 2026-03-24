from __future__ import annotations

import asyncio
import platform
from pathlib import Path

from ....agent.types import AgentTool, AgentToolResult
from ....ai.types import TextContent
from ._path_utils import resolve_safe_path


def create_open_tool(cwd: str) -> AgentTool:
    """Open a file or URL with the default application."""

    async def execute(tool_call_id, args, cancel_event=None, on_update=None):
        path_or_url = args.get("path_or_url", "")

        if not path_or_url:
            return AgentToolResult(
                content=[TextContent(text="Error: No path or URL provided.")],
                details=None,
            )

        # Determine the open command based on the platform
        system = platform.system()
        if system == "Darwin":
            cmd = ["open", path_or_url]
        elif system == "Linux":
            cmd = ["xdg-open", path_or_url]
        elif system == "Windows":
            cmd = ["start", "", path_or_url]
        else:
            return AgentToolResult(
                content=[TextContent(text=f"Error: Unsupported platform: {system}")],
                details=None,
            )

        # If it looks like a relative path (not a URL), resolve it
        if not path_or_url.startswith(("http://", "https://", "file://")):
            try:
                resolved = resolve_safe_path(cwd, path_or_url)
            except ValueError as e:
                return AgentToolResult(content=[TextContent(text=f"Error: {e}")], details=None)
            cmd[-1] = str(resolved)

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await asyncio.wait_for(proc.wait(), timeout=10)
            return AgentToolResult(
                content=[TextContent(text=f"Opened: {path_or_url}")],
                details=None,
            )
        except Exception as e:
            return AgentToolResult(
                content=[TextContent(text=f"Error opening {path_or_url}: {e}")],
                details=None,
            )

    return AgentTool(
        name="open",
        label="Open",
        description=(
            "Open a file or URL with the default application. "
            "On macOS uses 'open', on Linux uses 'xdg-open'."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path_or_url": {
                    "type": "string",
                    "description": "The file path (relative to project root) or URL to open.",
                },
            },
            "required": ["path_or_url"],
        },
        execute=execute,
    )
