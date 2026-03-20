from __future__ import annotations

import difflib
from pathlib import Path

from ...agent.types import AgentTool, AgentToolResult
from ...ai.types import TextContent
from .path_utils import resolve_to_cwd


def create_edit_tool(cwd: str) -> AgentTool[dict[str, str], dict[str, object]]:
    async def execute(_tool_call_id: str, args: dict[str, str], *_args) -> AgentToolResult[dict[str, object]]:
        path = str(args["path"])
        old_text = str(args["oldText"])
        new_text = str(args["newText"])
        absolute_path = Path(resolve_to_cwd(path, cwd))
        if not absolute_path.exists():
            raise ValueError(f"File not found: {path}")
        original = absolute_path.read_text(encoding="utf-8")
        occurrences = original.count(old_text)
        if occurrences == 0:
            raise ValueError(
                f"Could not find the exact text in {path}. The old text must match exactly including all whitespace and newlines."
            )
        if occurrences > 1:
            raise ValueError(
                f"Found {occurrences} occurrences of the text in {path}. The text must be unique. Please provide more context to make it unique."
            )
        updated = original.replace(old_text, new_text, 1)
        if updated == original:
            raise ValueError(f"No changes made to {path}. The replacement produced identical content.")
        absolute_path.write_text(updated, encoding="utf-8")
        diff = "\n".join(
            difflib.unified_diff(
                original.splitlines(),
                updated.splitlines(),
                fromfile=path,
                tofile=path,
                lineterm="",
            )
        )
        return AgentToolResult(
            content=[TextContent(text=f"Successfully replaced text in {path}.")],
            details={"diff": diff},
        )

    return AgentTool(
        name="edit",
        label="edit",
        description="Edit a file by replacing exact text. The oldText must be unique in the file.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "oldText": {"type": "string"},
                "newText": {"type": "string"},
            },
            "required": ["path", "oldText", "newText"],
        },
        execute=execute,
    )
