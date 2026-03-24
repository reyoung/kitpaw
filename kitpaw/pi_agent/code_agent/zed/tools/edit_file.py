from __future__ import annotations

import asyncio

import aiofiles

from ....agent.types import AgentTool, AgentToolResult
from ....ai.types import TextContent
from ._path_utils import resolve_safe_path


def create_edit_file_tool(cwd: str) -> AgentTool:
    """Write or edit file contents.

    In Zed, when the model supports streaming tools the
    ``StreamingEditFileTool`` is exposed under the name ``edit_file``.
    This implementation follows that convention: the tool is always called
    ``edit_file`` and supports both ``write`` and ``edit`` modes.
    """

    async def execute(tool_call_id, args, cancel_event=None, on_update=None):
        display_description = args.get("display_description", "")
        path = args.get("path", "")
        mode = args.get("mode", "write")
        content = args.get("content")
        edits = args.get("edits")

        try:
            resolved = resolve_safe_path(cwd, path)
        except ValueError as e:
            return AgentToolResult(content=[TextContent(text=f"Error: {e}")], details=None)

        if mode == "write":
            if content is None:
                return AgentToolResult(
                    content=[TextContent(text="Error: 'content' is required for 'write' mode.")],
                    details=None,
                )
            try:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, lambda: resolved.parent.mkdir(parents=True, exist_ok=True))
                async with aiofiles.open(resolved, "w", encoding="utf-8") as f:
                    await f.write(content)
                return AgentToolResult(
                    content=[TextContent(text=f"Wrote to {path}: {display_description}")],
                    details=None,
                )
            except Exception as e:
                return AgentToolResult(
                    content=[TextContent(text=f"Error writing file: {e}")],
                    details=None,
                )

        elif mode == "edit":
            if not edits:
                return AgentToolResult(
                    content=[TextContent(text="Error: 'edits' is required for 'edit' mode.")],
                    details=None,
                )

            loop = asyncio.get_running_loop()
            exists = await loop.run_in_executor(None, resolved.exists)
            if not exists:
                return AgentToolResult(
                    content=[TextContent(text=f"Error: File does not exist: {path}")],
                    details=None,
                )

            try:
                async with aiofiles.open(resolved, encoding="utf-8") as f:
                    text = await f.read()

                applied = 0
                failed: list[str] = []
                for edit in edits:
                    old_text = edit.get("old_text", "")
                    new_text = edit.get("new_text", "")

                    if old_text in text:
                        text = text.replace(old_text, new_text, 1)
                        applied += 1
                    else:
                        snippet = old_text[:60].replace("\n", "\\n")
                        failed.append(f"  - Not found: '{snippet}...'")

                async with aiofiles.open(resolved, "w", encoding="utf-8") as f:
                    await f.write(text)

                parts = [f"Edited {path}: {display_description}"]
                parts.append(f"Applied {applied}/{len(edits)} edit(s).")
                if failed:
                    parts.append("Failed edits:")
                    parts.extend(failed)

                return AgentToolResult(
                    content=[TextContent(text="\n".join(parts))],
                    details=None,
                )
            except Exception as e:
                return AgentToolResult(
                    content=[TextContent(text=f"Error editing file: {e}")],
                    details=None,
                )
        else:
            return AgentToolResult(
                content=[TextContent(text=f"Error: Unknown mode '{mode}'. Use 'write' or 'edit'.")],
                details=None,
            )

    return AgentTool(
        name="edit_file",
        label="Edit File",
        description=(
            "Write or edit file contents. "
            "In 'write' mode, provide the full file content to write. If the file doesn't exist, it will be created. "
            "In 'edit' mode, provide a list of {old_text, new_text} replacements to apply sequentially. "
            "Each edit replaces the first occurrence of old_text with new_text."
        ),
        parameters={
            "type": "object",
            "properties": {
                "display_description": {
                    "type": "string",
                    "description": (
                        "A one-line, user-friendly markdown description of the edit. "
                        "Be terse, but descriptive. NEVER mention the file path in this description."
                    ),
                },
                "path": {
                    "type": "string",
                    "description": "The path of the file to write or edit, relative to the project root.",
                },
                "mode": {
                    "type": "string",
                    "enum": ["write", "edit"],
                    "description": "The mode: 'write' to replace entire file content, 'edit' to apply text replacements.",
                },
                "content": {
                    "type": "string",
                    "description": "The full file content to write (required for 'write' mode).",
                },
                "edits": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "old_text": {
                                "type": "string",
                                "description": "The exact text to find in the file.",
                            },
                            "new_text": {
                                "type": "string",
                                "description": "The text to replace it with.",
                            },
                        },
                        "required": ["old_text", "new_text"],
                    },
                    "description": "List of text replacements to apply (required for 'edit' mode).",
                },
            },
            "required": ["display_description", "path", "mode"],
        },
        execute=execute,
    )
