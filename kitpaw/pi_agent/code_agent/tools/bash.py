from __future__ import annotations

import asyncio
import os

from ...agent.types import AgentTool, AgentToolResult
from ...ai.types import TextContent
from .truncate import DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, format_size, truncate_tail


def create_bash_tool(cwd: str, command_prefix: str | None = None) -> AgentTool[dict[str, object], dict[str, object] | None]:
    async def execute(
        _tool_call_id: str,
        args: dict[str, object],
        cancel_event: asyncio.Event | None = None,
        on_update=None,
    ) -> AgentToolResult[dict[str, object] | None]:
        command = str(args["command"])
        timeout = float(args["timeout"]) if args.get("timeout") is not None else None
        if command_prefix:
            command = f"{command_prefix}\n{command}"
        proc = await asyncio.create_subprocess_shell(
            command,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=os.environ.copy(),
        )
        try:
            if cancel_event is not None:
                waiter = asyncio.create_task(cancel_event.wait())
            else:
                waiter = None
            chunks: list[bytes] = []
            while True:
                read_task = asyncio.create_task(proc.stdout.read(4096))  # type: ignore[union-attr]
                tasks = {read_task}
                if waiter is not None:
                    tasks.add(waiter)
                done, _pending = await asyncio.wait(tasks, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
                if waiter is not None and waiter in done and waiter.result():
                    proc.kill()
                    raise ValueError("Command aborted")
                if read_task not in done:
                    proc.kill()
                    raise ValueError(f"Command timed out after {timeout:g} seconds")
                chunk = read_task.result()
                if not chunk:
                    break
                chunks.append(chunk)
                if on_update is not None:
                    rolling = b"".join(chunks).decode("utf-8", errors="replace")
                    truncated = truncate_tail(rolling)
                    on_update(AgentToolResult(content=[TextContent(text=truncated.content or "")], details=None))
            exit_code = await proc.wait()
        finally:
            if cancel_event is not None:
                waiter.cancel()
        full_output = b"".join(chunks).decode("utf-8", errors="replace")
        truncation = truncate_tail(full_output)
        output = truncation.content or "(no output)"
        details = {"truncation": truncation.__dict__} if truncation.truncated else None
        if truncation.truncated:
            start_line = truncation.total_lines - truncation.output_lines + 1
            if truncation.last_line_partial:
                output += (
                    f"\n\n[Showing last {format_size(truncation.output_bytes)} of line {truncation.total_lines} "
                    f"({format_size(DEFAULT_MAX_BYTES)} limit)]"
                )
            elif truncation.truncated_by == "lines":
                output += (
                    f"\n\n[Showing lines {start_line}-{truncation.total_lines} of {truncation.total_lines}.]"
                )
            else:
                output += (
                    f"\n\n[Showing lines {start_line}-{truncation.total_lines} of {truncation.total_lines} "
                    f"({format_size(DEFAULT_MAX_BYTES)} limit).]"
                )
        if exit_code != 0:
            raise ValueError(f"{output}\n\nCommand exited with code {exit_code}")
        return AgentToolResult(content=[TextContent(text=output)], details=details)

    return AgentTool(
        name="bash",
        label="bash",
        description=(
            f"Execute a bash command. Output is truncated to the last {DEFAULT_MAX_LINES} lines or "
            f"{DEFAULT_MAX_BYTES // 1024}KB."
        ),
        parameters={
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "timeout": {"type": "number"},
            },
            "required": ["command"],
        },
        execute=execute,
    )
