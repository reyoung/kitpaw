from __future__ import annotations

from ....agent.types import AgentTool, AgentToolResult
from ....ai.types import TextContent


def create_restore_file_from_disk_tool(cwd: str) -> AgentTool:
    """Restore file contents from disk."""

    async def execute(tool_call_id, args, cancel_event=None, on_update=None):
        paths = args.get("paths", [])
        # In CLI context, we always read from disk directly.
        # This is a no-op provided for compatibility with the Zed editor tool set.
        path_list = ", ".join(paths) if paths else "(none)"
        return AgentToolResult(
            content=[TextContent(text=f"Files restored from disk: {path_list}")],
            details=None,
        )

    return AgentTool(
        name="restore_file_from_disk",
        label="Restore File from Disk",
        description=(
            "Restores the file contents from disk, discarding any unsaved in-editor changes. "
            "In CLI mode, files are always read directly from disk so this is a no-op confirmation."
        ),
        parameters={
            "type": "object",
            "properties": {
                "paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "The list of file paths to restore, relative to the project root.",
                },
            },
            "required": ["paths"],
        },
        execute=execute,
    )
