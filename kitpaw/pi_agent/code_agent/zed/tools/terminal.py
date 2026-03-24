from __future__ import annotations

import asyncio
from pathlib import Path

from ....agent.types import AgentTool, AgentToolResult
from ....ai.types import TextContent
from ._path_utils import resolve_safe_path

_OUTPUT_LIMIT = 16 * 1024  # 16KB


def create_terminal_tool(cwd: str) -> AgentTool:
    """Execute a shell command in the terminal."""

    async def execute(tool_call_id, args, cancel_event=None, on_update=None):
        command = args.get("command", "")
        cd = args.get("cd", ".")
        timeout_ms = args.get("timeout_ms")

        try:
            work_dir = resolve_safe_path(cwd, cd)
        except ValueError as e:
            return AgentToolResult(content=[TextContent(text=f"Error: {e}")], details=None)
        if not work_dir.is_dir():
            return AgentToolResult(
                content=[TextContent(text=f"Error: Directory does not exist: {work_dir}")],
                details=None,
            )

        timeout_sec = timeout_ms / 1000.0 if timeout_ms else 120.0

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                cwd=str(work_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout_sec
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return AgentToolResult(
                    content=[TextContent(text=f"Error: Command timed out after {timeout_sec:.0f}s")],
                    details=None,
                )
        except Exception as e:
            return AgentToolResult(
                content=[TextContent(text=f"Error executing command: {e}")],
                details=None,
            )

        out = stdout.decode("utf-8", errors="replace")
        err = stderr.decode("utf-8", errors="replace")

        parts: list[str] = []
        if out:
            if len(out) > _OUTPUT_LIMIT:
                out = out[:_OUTPUT_LIMIT] + "\n... (output truncated)"
            parts.append(out)
        if err:
            if len(err) > _OUTPUT_LIMIT:
                err = err[:_OUTPUT_LIMIT] + "\n... (stderr truncated)"
            parts.append(f"stderr:\n{err}")

        exit_code = proc.returncode
        parts.append(f"\n[exit code: {exit_code}]")

        return AgentToolResult(
            content=[TextContent(text="\n".join(parts))],
            details=None,
        )

    return AgentTool(
        name="terminal",
        label="Terminal",
        description=(
            "Executes a shell command on the user's machine in the specified directory "
            "and returns the output. The command will run in a non-interactive shell. "
            "NOTE: Do not use shell substitutions like $(...) or backticks. "
            "Output is limited to 16KB."
        ),
        parameters={
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute.",
                },
                "cd": {
                    "type": "string",
                    "description": "The working directory in which to execute the command, relative to the project root.",
                },
                "timeout_ms": {
                    "type": "integer",
                    "description": "Optional timeout in milliseconds. If not set, defaults to 120 seconds.",
                },
            },
            "required": ["command", "cd"],
        },
        execute=execute,
    )
