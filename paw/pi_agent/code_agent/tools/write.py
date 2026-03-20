from __future__ import annotations

from pathlib import Path

from ...agent.types import AgentTool, AgentToolResult
from ...ai.types import TextContent
from .path_utils import resolve_to_cwd


def create_write_tool(cwd: str) -> AgentTool[dict[str, str], None]:
    async def execute(_tool_call_id: str, args: dict[str, str], *_args) -> AgentToolResult[None]:
        path = str(args["path"])
        content = str(args["content"])
        absolute_path = Path(resolve_to_cwd(path, cwd))
        absolute_path.parent.mkdir(parents=True, exist_ok=True)
        absolute_path.write_text(content, encoding="utf-8")
        return AgentToolResult(
            content=[TextContent(text=f"Successfully wrote {len(content)} bytes to {path}")],
            details=None,
        )

    return AgentTool(
        name="write",
        label="write",
        description="Write content to a file, creating parent directories as needed.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
        execute=execute,
    )
