from __future__ import annotations

from datetime import datetime, timezone

from ....agent.types import AgentTool, AgentToolResult
from ....ai.types import TextContent


def create_now_tool(cwd: str) -> AgentTool:
    """Get the current date and time."""

    async def execute(tool_call_id, args, cancel_event=None, on_update=None):
        tz = args.get("timezone", "utc")

        if tz == "utc":
            dt = datetime.now(timezone.utc)
            result = dt.strftime("%Y-%m-%d %H:%M:%S UTC")
        else:
            dt = datetime.now()
            result = dt.strftime("%Y-%m-%d %H:%M:%S (local)")

        return AgentToolResult(
            content=[TextContent(text=result)],
            details=None,
        )

    return AgentTool(
        name="now",
        label="Now",
        description=(
            "Returns the current date and time in the specified timezone. "
            "Useful for time-sensitive tasks or when the user asks about the current time."
        ),
        parameters={
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "enum": ["utc", "local"],
                    "description": "The timezone to return the current time in: 'utc' or 'local'.",
                },
            },
            "required": ["timezone"],
        },
        execute=execute,
    )
