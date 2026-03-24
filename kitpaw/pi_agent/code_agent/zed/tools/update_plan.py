from __future__ import annotations

import json

from ....agent.types import AgentTool, AgentToolResult
from ....ai.types import TextContent

_STATUS_ICONS = {
    "pending": "[ ]",
    "in_progress": "[~]",
    "completed": "[x]",
}

_PRIORITY_LABELS = {
    "high": "HIGH",
    "medium": "MED",
    "low": "LOW",
}


def create_update_plan_tool(cwd: str) -> AgentTool:
    """Update and display the current task plan."""

    async def execute(tool_call_id, args, cancel_event=None, on_update=None):
        plan = args.get("plan", [])

        if not plan:
            return AgentToolResult(
                content=[TextContent(text="No plan steps provided.")],
                details=None,
            )

        lines = ["Plan:"]
        for i, item in enumerate(plan, 1):
            step = item.get("step", "")
            status = item.get("status", "pending")
            priority = item.get("priority", "medium")
            icon = _STATUS_ICONS.get(status, "[ ]")
            prio = _PRIORITY_LABELS.get(priority, "MED")
            lines.append(f"  {icon} [{prio}] {i}. {step}")

        output = "\n".join(lines)

        return AgentToolResult(
            content=[TextContent(text=output)],
            details=None,
        )

    return AgentTool(
        name="update_plan",
        label="Update Plan",
        description=(
            "Update and display the current task plan. "
            "Each step has a status (pending, in_progress, completed) and priority (high, medium, low)."
        ),
        parameters={
            "type": "object",
            "properties": {
                "plan": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "step": {
                                "type": "string",
                                "description": "Description of the plan step.",
                            },
                            "status": {
                                "type": "string",
                                "enum": ["pending", "in_progress", "completed"],
                                "description": "The current status of this step.",
                            },
                            "priority": {
                                "type": "string",
                                "enum": ["high", "medium", "low"],
                                "description": "The priority level of this step.",
                            },
                        },
                        "required": ["step", "status", "priority"],
                    },
                    "description": "The list of plan steps with their status and priority.",
                },
            },
            "required": ["plan"],
        },
        execute=execute,
    )
