from __future__ import annotations

from pathlib import Path

from ...agent.types import AgentTool, AgentToolResult
from ...ai.types import TextContent
from .path_utils import resolve_to_cwd
from .truncate import DEFAULT_MAX_BYTES, format_size, truncate_head

DEFAULT_LIMIT = 500


def create_ls_tool(cwd: str) -> AgentTool[dict[str, object], dict[str, object] | None]:
    async def execute(_tool_call_id: str, args: dict[str, object], *_args) -> AgentToolResult[dict[str, object] | None]:
        directory = Path(resolve_to_cwd(str(args.get("path", ".")), cwd))
        limit = int(args.get("limit", DEFAULT_LIMIT))
        if not directory.exists():
            raise ValueError(f"Path not found: {directory}")
        if not directory.is_dir():
            raise ValueError(f"Not a directory: {directory}")
        entries = sorted(directory.iterdir(), key=lambda p: p.name.lower())
        formatted = [(entry.name + ("/" if entry.is_dir() else "")) for entry in entries[:limit]]
        if not formatted:
            return AgentToolResult(content=[TextContent(text="(empty directory)")], details=None)
        raw = "\n".join(formatted)
        truncation = truncate_head(raw, max_lines=10**9)
        output = truncation.content
        notices: list[str] = []
        details: dict[str, object] | None = None
        if len(entries) > limit:
            notices.append(f"{limit} entries limit reached. Use limit={limit * 2} for more")
        if truncation.truncated:
            notices.append(f"{format_size(DEFAULT_MAX_BYTES)} limit reached")
        if notices:
            output += f"\n\n[{'. '.join(notices)}]"
            details = {"truncation": truncation.__dict__}
        return AgentToolResult(content=[TextContent(text=output)], details=details)

    return AgentTool(
        name="ls",
        label="ls",
        description="List directory contents.",
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string"}, "limit": {"type": "number"}},
        },
        execute=execute,
    )
