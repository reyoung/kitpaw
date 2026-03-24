from __future__ import annotations

from ....agent.types import AgentTool, AgentToolResult
from ....ai.types import TextContent

_STATUS_ICONS = {
    "pending": "[ ]",
    "in_progress": "[~]",
    "completed": "[x]",
}


def create_update_plan_tool(cwd: str) -> AgentTool:
    """Update and display the current task plan."""

    async def execute(tool_call_id, args, cancel_event=None, on_update=None):
        plan: list[dict] = args.get("plan", [])
        explanation: str = args.get("explanation", "")

        if not plan:
            return AgentToolResult(
                content=[TextContent(text="No plan steps provided.")],
                details=None,
            )

        lines = ["Plan:"]
        for i, item in enumerate(plan, 1):
            step = item.get("step", "")
            status = item.get("status", "pending")
            icon = _STATUS_ICONS.get(status, "[ ]")
            lines.append(f"  {icon} {i}. {step}")

        if explanation:
            lines.append("")
            lines.append(f"Explanation: {explanation}")

        output = "\n".join(lines)

        return AgentToolResult(
            content=[TextContent(text=output)],
            details=None,
        )

    return AgentTool(
        name="update_plan",
        label="Update Plan",
        description=(
            "Update and display the current task plan. Each step has a title and a "
            "status (pending, in_progress, completed). Optionally provide an "
            "explanation of why the plan changed."
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
                        },
                        "required": ["step", "status"],
                    },
                    "description": "The list of plan steps with their status.",
                },
                "explanation": {
                    "type": "string",
                    "description": "Optional explanation of why the plan changed.",
                },
            },
            "required": ["plan"],
        },
        execute=execute,
    )
