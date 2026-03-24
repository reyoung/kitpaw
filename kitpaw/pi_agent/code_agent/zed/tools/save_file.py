from __future__ import annotations

from ....agent.types import AgentTool, AgentToolResult
from ....ai.types import TextContent


def create_save_file_tool(cwd: str) -> AgentTool:
    """Save files to disk."""

    async def execute(tool_call_id, args, cancel_event=None, on_update=None):
        paths = args.get("paths", [])
        # In CLI context, files are already saved to disk.
        # This is a no-op provided for compatibility with the Zed editor tool set.
        path_list = ", ".join(paths) if paths else "(none)"
        return AgentToolResult(
            content=[TextContent(text=f"Files are already saved to disk: {path_list}")],
            details=None,
        )

    return AgentTool(
        name="save_file",
        label="Save File",
        description=(
            "Saves the specified files to disk. "
            "In CLI mode, files are written directly to disk so this is a no-op confirmation."
        ),
        parameters={
            "type": "object",
            "properties": {
                "paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "The list of file paths to save, relative to the project root.",
                },
            },
            "required": ["paths"],
        },
        execute=execute,
    )
