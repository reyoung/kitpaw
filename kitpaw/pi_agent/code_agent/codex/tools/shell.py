from __future__ import annotations

import asyncio
import json
import time

from ....agent.types import AgentTool, AgentToolResult
from ....ai.types import TextContent

_OUTPUT_LIMIT = 256 * 1024  # 256KB


def create_shell_tool(cwd: str) -> AgentTool:
    """Execute a command as a subprocess and return its output."""

    async def execute(tool_call_id, args, cancel_event=None, on_update=None):
        command: list[str] = args.get("command", [])
        timeout_ms: int = args.get("timeout_ms", 120_000)
        timeout_s = timeout_ms / 1000
        workdir: str = args.get("workdir") or cwd
        # Accepted but ignored in CLI mode
        _sandbox_permissions: str = args.get("sandbox_permissions", "use_default")
        _justification: str = args.get("justification", "")

        if not command:
            return AgentToolResult(
                content=[TextContent(text=json.dumps({"exit_code": 1, "output": "Error: No command provided.", "duration_seconds": 0.0}))],
                details=None,
            )

        start = time.monotonic()

        try:
            proc = await asyncio.create_subprocess_exec(
                *command,
                cwd=workdir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            try:
                stdout, _ = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout_s
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                duration = time.monotonic() - start
                result = {
                    "exit_code": -1,
                    "output": f"Command timed out after {timeout_s}s",
                    "duration_seconds": round(duration, 1),
                }
                return AgentToolResult(
                    content=[TextContent(text=json.dumps(result))],
                    details=None,
                )
        except Exception as e:
            duration = time.monotonic() - start
            result = {
                "exit_code": -1,
                "output": f"Error executing command: {e}",
                "duration_seconds": round(duration, 1),
            }
            return AgentToolResult(
                content=[TextContent(text=json.dumps(result))],
                details=None,
            )

        duration = time.monotonic() - start
        output = stdout.decode("utf-8", errors="replace")

        if len(output) > _OUTPUT_LIMIT:
            output = output[:_OUTPUT_LIMIT] + "\n... (output truncated at 256KB)"

        result = {
            "exit_code": proc.returncode,
            "output": output,
            "duration_seconds": round(duration, 1),
        }

        return AgentToolResult(
            content=[TextContent(text=json.dumps(result))],
            details=None,
        )

    return AgentTool(
        name="shell",
        label="Shell",
        description=(
            "Runs a command in a subprocess and returns its exit code, combined "
            "stdout/stderr output, and wall-clock duration. Use this for any "
            "file-system inspection, builds, tests, git operations, or other CLI tasks. "
            "Output is truncated at 256KB."
        ),
        parameters={
            "type": "object",
            "properties": {
                "command": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "The command and arguments to execute.",
                },
                "timeout_ms": {
                    "type": "integer",
                    "description": "Timeout in milliseconds. Defaults to 120000.",
                },
                "workdir": {
                    "type": "string",
                    "description": "Working directory for the command. Defaults to the project root.",
                },
                "sandbox_permissions": {
                    "type": "string",
                    "enum": ["require_escalated", "use_default"],
                    "description": "Sandbox permission mode.",
                },
                "justification": {
                    "type": "string",
                    "description": "Justification for requiring escalated permissions.",
                },
            },
            "required": ["command"],
        },
        execute=execute,
    )
