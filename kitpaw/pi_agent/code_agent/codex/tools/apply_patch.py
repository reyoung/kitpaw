from __future__ import annotations

import asyncio

from ....agent.types import AgentTool, AgentToolResult
from ....ai.types import TextContent
from ...zed.tools._path_utils import resolve_safe_path


def create_apply_patch_tool(cwd: str) -> AgentTool:
    """Apply a unified-diff patch to the project."""

    async def execute(tool_call_id, args, cancel_event=None, on_update=None):
        patch: str = args.get("patch", "")

        if not patch.strip():
            return AgentToolResult(
                content=[TextContent(text="Error: Empty patch provided.")],
                details=None,
            )

        # Validate that referenced paths stay inside the workspace.
        for line in patch.splitlines():
            for prefix in ("+++ b/", "+++ ", "--- a/", "--- "):
                if line.startswith(prefix):
                    rel = line[len(prefix):].strip()
                    if rel and rel != "/dev/null":
                        try:
                            resolve_safe_path(cwd, rel)
                        except ValueError as e:
                            return AgentToolResult(
                                content=[TextContent(text=f"Error: {e}")],
                                details=None,
                            )
                    break  # only match the first matching prefix per line

        try:
            proc = await asyncio.create_subprocess_exec(
                "patch", "-p1", "--no-backup-if-mismatch",
                cwd=cwd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            stdout, _ = await asyncio.wait_for(
                proc.communicate(input=patch.encode()), timeout=30.0
            )
        except FileNotFoundError:
            return AgentToolResult(
                content=[TextContent(text="Error: 'patch' command not found on this system.")],
                details=None,
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return AgentToolResult(
                content=[TextContent(text="Error: patch command timed out after 30s.")],
                details=None,
            )
        except Exception as e:
            return AgentToolResult(
                content=[TextContent(text=f"Error running patch: {e}")],
                details=None,
            )

        output = stdout.decode("utf-8", errors="replace").strip()

        if proc.returncode == 0:
            summary = output if output else "Patch applied successfully."
            return AgentToolResult(
                content=[TextContent(text=summary)],
                details=None,
            )
        else:
            return AgentToolResult(
                content=[TextContent(text=f"Patch failed (exit {proc.returncode}):\n{output}")],
                details=None,
            )

    return AgentTool(
        name="apply_patch",
        label="Apply Patch",
        description=(
            "Apply a unified diff patch to files in the project. Supports creating, "
            "modifying, and deleting files.\n"
            "- To create a file: use '--- /dev/null' and '+++ b/path'\n"
            "- To modify a file: use '--- a/path' and '+++ b/path' with standard unified diff hunks\n"
            "- To delete a file: use '--- a/path' and '+++ /dev/null'"
        ),
        parameters={
            "type": "object",
            "properties": {
                "patch": {
                    "type": "string",
                    "description": (
                        "The patch to apply. Supports unified diff format or simple file operations:\n"
                        "- To create a file: start with '--- /dev/null' and '+++ b/path'\n"
                        "- To modify a file: unified diff with '--- a/path' and '+++ b/path'\n"
                        "- To delete a file: start with '--- a/path' and '+++ /dev/null'"
                    ),
                },
            },
            "required": ["patch"],
        },
        execute=execute,
    )
