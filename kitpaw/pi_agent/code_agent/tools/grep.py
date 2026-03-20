from __future__ import annotations

import re
from pathlib import Path

from ...agent.types import AgentTool, AgentToolResult
from ...ai.types import TextContent
from .path_utils import resolve_to_cwd
from .truncate import (
    DEFAULT_MAX_BYTES,
    GREP_MAX_LINE_LENGTH,
    format_size,
    truncate_head,
    truncate_line,
)

DEFAULT_LIMIT = 100


def create_grep_tool(cwd: str) -> AgentTool[dict[str, object], dict[str, object] | None]:
    async def execute(_tool_call_id: str, args: dict[str, object], *_args) -> AgentToolResult[dict[str, object] | None]:
        pattern = str(args["pattern"])
        search_path = Path(resolve_to_cwd(str(args.get("path", ".")), cwd))
        glob = str(args["glob"]) if args.get("glob") else None
        ignore_case = bool(args.get("ignoreCase", False))
        literal = bool(args.get("literal", False))
        context = int(args.get("context", 0))
        limit = int(args.get("limit", DEFAULT_LIMIT))
        flags = re.IGNORECASE if ignore_case else 0
        regex = re.compile(re.escape(pattern) if literal else pattern, flags)
        files: list[Path]
        if search_path.is_dir():
            files = [p for p in search_path.rglob("*") if p.is_file()]
        elif search_path.exists():
            files = [search_path]
        else:
            raise ValueError(f"Path not found: {search_path}")

        output_lines: list[str] = []
        matches = 0
        lines_truncated = False
        for file_path in files:
            rel = file_path.relative_to(search_path).as_posix() if search_path.is_dir() else file_path.name
            if ".git/" in rel or "node_modules/" in rel:
                continue
            if glob and not file_path.match(glob):
                continue
            try:
                lines = file_path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n").split("\n")
            except UnicodeDecodeError:
                continue
            for idx, line in enumerate(lines, start=1):
                if not regex.search(line):
                    continue
                start = max(1, idx - context)
                end = min(len(lines), idx + context)
                for current in range(start, end + 1):
                    text, was_truncated = truncate_line(lines[current - 1], GREP_MAX_LINE_LENGTH)
                    lines_truncated = lines_truncated or was_truncated
                    sep = ":" if current == idx else "-"
                    output_lines.append(f"{rel}{sep}{current}{sep} {text}")
                matches += 1
                if matches >= limit:
                    break
            if matches >= limit:
                break
        if not output_lines:
            return AgentToolResult(content=[TextContent(text="No matches found")], details=None)
        raw = "\n".join(output_lines)
        truncation = truncate_head(raw, max_lines=10**9)
        output = truncation.content
        notices: list[str] = []
        details: dict[str, object] | None = None
        if matches >= limit:
            notices.append(f"{limit} matches limit reached. Use limit={limit * 2} for more, or refine pattern")
        if truncation.truncated:
            notices.append(f"{format_size(DEFAULT_MAX_BYTES)} limit reached")
        if lines_truncated:
            notices.append(f"Some lines truncated to {GREP_MAX_LINE_LENGTH} chars. Use read tool to see full lines")
        if notices:
            output += f"\n\n[{'. '.join(notices)}]"
            details = {"truncation": truncation.__dict__, "linesTruncated": lines_truncated}
        return AgentToolResult(content=[TextContent(text=output)], details=details)

    return AgentTool(
        name="grep",
        label="grep",
        description="Search file contents for a pattern.",
        parameters={
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "path": {"type": "string"},
                "glob": {"type": "string"},
                "ignoreCase": {"type": "boolean"},
                "literal": {"type": "boolean"},
                "context": {"type": "number"},
                "limit": {"type": "number"},
            },
            "required": ["pattern"],
        },
        execute=execute,
    )
