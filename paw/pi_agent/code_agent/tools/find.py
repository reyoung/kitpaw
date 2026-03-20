from __future__ import annotations

import fnmatch
from pathlib import Path

from ...agent.types import AgentTool, AgentToolResult
from ...ai.types import TextContent
from .path_utils import resolve_to_cwd
from .truncate import DEFAULT_MAX_BYTES, format_size, truncate_head

DEFAULT_LIMIT = 1000


def create_find_tool(cwd: str) -> AgentTool[dict[str, object], dict[str, object] | None]:
    async def execute(_tool_call_id: str, args: dict[str, object], *_args) -> AgentToolResult[dict[str, object] | None]:
        pattern = str(args["pattern"])
        root = Path(resolve_to_cwd(str(args.get("path", ".")), cwd))
        limit = int(args.get("limit", DEFAULT_LIMIT))
        if not root.exists():
            raise ValueError(f"Path not found: {root}")
        results: list[str] = []
        for path in root.rglob("*"):
            rel = path.relative_to(root).as_posix()
            if ".git/" in rel or "node_modules/" in rel:
                continue
            if fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(path.name, pattern):
                results.append(rel + ("/" if path.is_dir() else ""))
                if len(results) >= limit:
                    break
        if not results:
            return AgentToolResult(content=[TextContent(text="No files found matching pattern")], details=None)
        raw = "\n".join(results)
        truncation = truncate_head(raw, max_lines=10**9)
        output = truncation.content
        details: dict[str, object] | None = None
        notices: list[str] = []
        if len(results) >= limit:
            notices.append(f"{limit} results limit reached. Use limit={limit * 2} for more, or refine pattern")
        if truncation.truncated:
            notices.append(f"{format_size(DEFAULT_MAX_BYTES)} limit reached")
        if notices:
            output += f"\n\n[{'. '.join(notices)}]"
            details = {"truncation": truncation.__dict__}
        return AgentToolResult(content=[TextContent(text=output)], details=details)

    return AgentTool(
        name="find",
        label="find",
        description="Search for files by glob pattern.",
        parameters={
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "path": {"type": "string"},
                "limit": {"type": "number"},
            },
            "required": ["pattern"],
        },
        execute=execute,
    )
