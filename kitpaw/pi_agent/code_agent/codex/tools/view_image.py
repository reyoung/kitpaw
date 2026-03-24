from __future__ import annotations

import json
import os

from ....agent.types import AgentTool, AgentToolResult
from ....ai.types import TextContent


def create_view_image_tool(cwd: str) -> AgentTool:
    """View an image file. Returns the image file path and metadata."""

    async def execute(tool_call_id, args, cancel_event=None, on_update=None):
        path: str = args.get("path", "")

        if not path:
            return AgentToolResult(
                content=[TextContent(text=json.dumps({"error": "No path provided."}))],
                details=None,
            )

        # Resolve relative paths against cwd
        if not os.path.isabs(path):
            path = os.path.join(cwd, path)

        if not os.path.isfile(path):
            return AgentToolResult(
                content=[TextContent(text=json.dumps({"error": f"File not found: {path}"}))],
                details=None,
            )

        size = os.path.getsize(path)
        result = {
            "path": path,
            "size_bytes": size,
        }

        return AgentToolResult(
            content=[TextContent(text=json.dumps(result))],
            details=None,
        )

    return AgentTool(
        name="view_image",
        label="View Image",
        description=(
            "View an image file. Returns the image file path and metadata."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the image file.",
                },
            },
            "required": ["path"],
        },
        execute=execute,
    )
