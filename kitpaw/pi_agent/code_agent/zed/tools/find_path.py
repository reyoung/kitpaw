from __future__ import annotations

import asyncio
from pathlib import Path

from ....agent.types import AgentTool, AgentToolResult
from ....ai.types import TextContent

_PAGE_SIZE = 50


def create_find_path_tool(cwd: str) -> AgentTool:
    """Find files and directories matching a glob pattern."""

    async def execute(tool_call_id, args, cancel_event=None, on_update=None):
        glob_pattern = args.get("glob", "")
        offset = args.get("offset", 0)

        root = Path(cwd)
        try:
            loop = asyncio.get_running_loop()

            def _glob():
                return sorted(
                    str(p.relative_to(root)) for p in root.glob(glob_pattern)
                    if not any(part.startswith(".") for part in p.relative_to(root).parts)
                )

            matches = await loop.run_in_executor(None, _glob)
        except Exception as e:
            return AgentToolResult(
                content=[TextContent(text=f"Error searching for paths: {e}")],
                details=None,
            )

        total = len(matches)
        if total == 0:
            return AgentToolResult(
                content=[TextContent(text=f"No files found matching glob pattern: {glob_pattern}")],
                details=None,
            )

        page = matches[offset : offset + _PAGE_SIZE]
        end_idx = min(offset + _PAGE_SIZE, total)

        result_parts = [f"Found {total} matching paths (showing {offset + 1}-{end_idx}):\n"]
        result_parts.extend(page)

        if end_idx < total:
            result_parts.append(f"\n... {total - end_idx} more results. Use offset={end_idx} to see more.")

        return AgentToolResult(
            content=[TextContent(text="\n".join(result_parts))],
            details=None,
        )

    return AgentTool(
        name="find_path",
        label="Find Path",
        description=(
            "Search for files and directories matching a glob pattern in the project. "
            "Results are paginated with 50 entries per page. "
            "Hidden files (starting with '.') are excluded by default."
        ),
        parameters={
            "type": "object",
            "properties": {
                "glob": {
                    "type": "string",
                    "description": "The glob pattern to match file paths against (e.g., '**/*.py', 'src/**/test_*').",
                },
                "offset": {
                    "type": "integer",
                    "description": "The 0-based offset to start showing results from. Defaults to 0.",
                },
            },
            "required": ["glob"],
        },
        execute=execute,
    )
