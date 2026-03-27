from __future__ import annotations

import asyncio
import os

from ...agent.types import AgentTool, AgentToolResult
from ...ai.types import TextContent
from .truncate import DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, format_size, truncate_tail


def create_uv_tool(cwd: str) -> AgentTool[dict[str, object], dict[str, object] | None]:
    """Create a tool for running uv (Python package / project manager) commands.

    This tool provides a safe, dedicated interface for uv operations such as:
    - ``uv pip install / uninstall / list``
    - ``uv add / remove``
    - ``uv sync / lock``
    - ``uv run``
    - ``uv venv``
    - ``uv tool install / run``
    - ``uv python install / list``

    Only commands starting with ``uv`` are accepted.
    """

    _ALLOWED_SUBCOMMANDS = frozenset({
        "pip",
        "add",
        "remove",
        "sync",
        "lock",
        "run",
        "venv",
        "tool",
        "python",
        "init",
        "tree",
        "export",
        "cache",
        "self",
        "version",
        "--version",
        "--help",
        "help",
    })

    async def execute(
        _tool_call_id: str,
        args: dict[str, object],
        cancel_event: asyncio.Event | None = None,
        on_update=None,
    ) -> AgentToolResult[dict[str, object] | None]:
        command = str(args.get("command", "")).strip()
        if not command:
            raise ValueError("command is required")

        # Validate: must start with "uv"
        parts = command.split()
        if not parts or parts[0] != "uv":
            raise ValueError(f"Only uv commands are allowed. Got: {parts[0] if parts else '(empty)'}")

        # Validate subcommand
        if len(parts) > 1 and parts[1] not in _ALLOWED_SUBCOMMANDS:
            raise ValueError(
                f"Unknown uv subcommand: {parts[1]}. "
                f"Allowed: {', '.join(sorted(_ALLOWED_SUBCOMMANDS))}"
            )

        timeout = float(args["timeout"]) if args.get("timeout") is not None else 120.0

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
                done, _pending = await asyncio.wait(
                    tasks, timeout=timeout, return_when=asyncio.FIRST_COMPLETED
                )
                if waiter is not None and waiter in done and waiter.result():
                    proc.kill()
                    raise ValueError("uv command aborted")
                if read_task not in done:
                    proc.kill()
                    raise ValueError(f"uv command timed out after {timeout:g} seconds")
                chunk = read_task.result()
                if not chunk:
                    break
                chunks.append(chunk)
                if on_update is not None:
                    rolling = b"".join(chunks).decode("utf-8", errors="replace")
                    truncated = truncate_tail(rolling)
                    on_update(
                        AgentToolResult(
                            content=[TextContent(text=truncated.content or "")],
                            details=None,
                        )
                    )
            exit_code = await proc.wait()
        finally:
            if cancel_event is not None and waiter is not None:
                waiter.cancel()

        full_output = b"".join(chunks).decode("utf-8", errors="replace")
        truncation = truncate_tail(full_output)
        output = truncation.content or "(no output)"
        details = {"truncation": truncation.to_dict()} if truncation.truncated else None

        if truncation.truncated:
            start_line = truncation.total_lines - truncation.output_lines + 1
            if truncation.last_line_partial:
                output += (
                    f"\n\n[Showing last {format_size(truncation.output_bytes)} of line {truncation.total_lines} "
                    f"({format_size(DEFAULT_MAX_BYTES)} limit)]"
                )
            elif truncation.truncated_by == "lines":
                output += f"\n\n[Showing lines {start_line}-{truncation.total_lines} of {truncation.total_lines}.]"
            else:
                output += (
                    f"\n\n[Showing lines {start_line}-{truncation.total_lines} of {truncation.total_lines} "
                    f"({format_size(DEFAULT_MAX_BYTES)} limit).]"
                )

        if exit_code != 0:
            raise ValueError(f"{output}\n\nuv command exited with code {exit_code}")
        return AgentToolResult(content=[TextContent(text=output)], details=details)

    return AgentTool(
        name="uv",
        label="uv",
        description=(
            "Run uv (Python package & project manager) commands. "
            "Supports: pip, add, remove, sync, lock, run, venv, tool, python, init, tree, export, cache. "
            f"Output is truncated to the last {DEFAULT_MAX_LINES} lines or {DEFAULT_MAX_BYTES // 1024}KB."
        ),
        parameters={
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": (
                        "The full uv command to execute, e.g. 'uv pip install requests', "
                        "'uv add flask', 'uv sync', 'uv run python script.py'."
                    ),
                },
                "timeout": {
                    "type": "number",
                    "description": "Timeout in seconds (default: 120).",
                },
            },
            "required": ["command"],
        },
        execute=execute,
    )
