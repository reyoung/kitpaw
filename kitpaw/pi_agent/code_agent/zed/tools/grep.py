from __future__ import annotations

import asyncio
from pathlib import Path

from ....agent.types import AgentTool, AgentToolResult
from ....ai.types import TextContent

_PAGE_SIZE = 50


def create_grep_tool(cwd: str) -> AgentTool:
    """Search for text matching a regex pattern in files."""

    async def execute(tool_call_id, args, cancel_event=None, on_update=None):
        regex = args.get("regex", "")
        include_pattern = args.get("include_pattern")
        offset = args.get("offset", 0)
        case_sensitive = args.get("case_sensitive", False)

        rg_args = ["rg", "--line-number", "--no-heading", "--color=never"]

        if not case_sensitive:
            rg_args.append("-i")

        if include_pattern:
            rg_args.extend(["-g", include_pattern])

        rg_args.extend(["--", regex, "."])

        try:
            proc = await asyncio.create_subprocess_exec(
                *rg_args,
                cwd=str(Path(cwd)),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
        except FileNotFoundError:
            return AgentToolResult(
                content=[TextContent(text="Error: ripgrep (rg) is not installed. Please install it first.")],
                details=None,
            )
        except asyncio.TimeoutError:
            return AgentToolResult(
                content=[TextContent(text="Error: grep search timed out after 30 seconds.")],
                details=None,
            )
        except Exception as e:
            return AgentToolResult(
                content=[TextContent(text=f"Error running grep: {e}")],
                details=None,
            )

        output = stdout.decode("utf-8", errors="replace")
        stderr_text = stderr.decode("utf-8", errors="replace").strip()

        if proc.returncode not in (0, 1):
            # rg exit code 1 = no matches, 2+ = error
            return AgentToolResult(
                content=[TextContent(text=f"Error: ripgrep failed (exit {proc.returncode}): {stderr_text or output}")],
                details=None,
            )

        if not output.strip():
            return AgentToolResult(
                content=[TextContent(text="No matches found.")],
                details=None,
            )

        lines = output.splitlines()
        total = len(lines)
        page = lines[offset : offset + _PAGE_SIZE]
        end_idx = min(offset + _PAGE_SIZE, total)

        result_parts = [f"Showing matches {offset + 1}-{end_idx} of {total} total matches:\n"]
        result_parts.extend(page)

        if end_idx < total:
            result_parts.append(f"\n... {total - end_idx} more matches. Use offset={end_idx} to see more.")

        return AgentToolResult(
            content=[TextContent(text="\n".join(result_parts))],
            details=None,
        )

    return AgentTool(
        name="grep",
        label="Grep",
        description=(
            "Search for text matching a regex pattern across files in the project. "
            "Uses ripgrep for fast searching. Results are paginated with 50 matches per page."
        ),
        parameters={
            "type": "object",
            "properties": {
                "regex": {
                    "type": "string",
                    "description": "The regular expression pattern to search for.",
                },
                "include_pattern": {
                    "type": "string",
                    "description": "Optional glob pattern to filter which files to search (e.g., '*.py', 'src/**/*.ts').",
                },
                "offset": {
                    "type": "integer",
                    "description": "The 0-based offset to start showing results from. Defaults to 0.",
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Whether the search should be case-sensitive. Defaults to false.",
                },
            },
            "required": ["regex"],
        },
        execute=execute,
    )
