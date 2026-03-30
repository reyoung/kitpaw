from __future__ import annotations

import asyncio
import os
import signal
import uuid
from typing import Any

from ..pi_agent.agent.types import AgentTool
from ..pi_agent.code_agent.tools.path_utils import PathTraversalError, resolve_to_cwd
from .context import OpenClawToolContext
from .registry import ProcessHandle, get_process_registry
from .result_utils import error_result, json_result

DEFAULT_LOG_TAIL_LINES = 200


def _derive_process_name(command: str) -> str:
    return command.strip()[:120]


def _validate_env(raw_env: Any) -> dict[str, str]:
    if raw_env is None:
        return {}
    if not isinstance(raw_env, dict):
        raise ValueError("env must be an object of string keys and values.")
    env: dict[str, str] = {}
    for key, value in raw_env.items():
        if not isinstance(key, str) or not key:
            raise ValueError("env keys must be non-empty strings.")
        env[key] = str(value)
    return env


def _validate_timeout(raw_timeout: Any) -> float | None:
    if raw_timeout is None:
        return None
    if not isinstance(raw_timeout, (int, float)) or raw_timeout <= 0:
        raise ValueError("timeout must be a positive number of seconds.")
    return float(raw_timeout)


def _validate_yield_ms(raw_yield_ms: Any) -> int | None:
    if raw_yield_ms is None:
        return None
    if not isinstance(raw_yield_ms, (int, float)) or raw_yield_ms < 0:
        raise ValueError("yield_ms must be a non-negative number.")
    return int(raw_yield_ms)


def _validate_process_timeout_ms(raw_timeout: Any) -> int:
    if raw_timeout is None:
        return 0
    if not isinstance(raw_timeout, (int, float)) or raw_timeout < 0:
        raise ValueError("timeout_ms must be a non-negative number.")
    return int(raw_timeout)


def _resolve_workdir(context: OpenClawToolContext, requested: Any) -> str:
    if requested is None:
        return context.workspace_dir
    if not isinstance(requested, str) or not requested.strip():
        raise ValueError("workdir must be a non-empty string.")
    try:
        return resolve_to_cwd(requested, context.workspace_dir)
    except PathTraversalError as exc:
        raise ValueError(str(exc)) from exc


async def _terminate_process(handle: ProcessHandle, reason: str) -> None:
    process = handle.process
    if process is None or process.returncode is not None:
        handle.mark_finished(
            status="killed" if reason == "killed" else "failed",
            exit_code=handle.exit_code,
            stop_reason=reason,
            error_message=handle.error_message,
        )
        return

    handle.termination_reason = reason
    try:
        if os.name != "nt" and process.pid:
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
    except ProcessLookupError:
        pass

    try:
        await asyncio.wait_for(process.wait(), timeout=1.0)
    except asyncio.TimeoutError:
        try:
            if os.name != "nt" and process.pid:
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        except ProcessLookupError:
            pass
        await process.wait()


async def _stream_process_output(handle: ProcessHandle) -> None:
    process = handle.process
    if process is None or process.stdout is None:
        return
    while True:
        chunk = await process.stdout.read(4096)
        if not chunk:
            break
        handle.append_output(chunk.decode("utf-8", errors="replace"))


async def _wait_for_process(handle: ProcessHandle) -> None:
    process = handle.process
    if process is None:
        handle.mark_finished(
            status="failed",
            exit_code=None,
            stop_reason="spawn_error",
            error_message="Process was not started.",
        )
        return

    try:
        if handle.timeout_seconds is not None:
            exit_code = await asyncio.wait_for(process.wait(), timeout=handle.timeout_seconds)
        else:
            exit_code = await process.wait()
    except asyncio.TimeoutError:
        await _terminate_process(handle, "timeout")
        await process.wait()
        exit_code = process.returncode
        handle.mark_finished(
            status="failed",
            exit_code=exit_code,
            stop_reason="timeout",
            error_message=f"Command timed out after {handle.timeout_seconds:g} seconds.",
        )
    except Exception as exc:  # noqa: BLE001
        handle.mark_finished(
            status="failed",
            exit_code=process.returncode,
            stop_reason="error",
            error_message=str(exc),
        )
    else:
        if handle.termination_reason == "killed":
            handle.mark_finished(
                status="killed",
                exit_code=exit_code,
                stop_reason="killed",
                error_message="Process was killed.",
            )
        elif exit_code == 0:
            handle.mark_finished(
                status="completed",
                exit_code=exit_code,
                stop_reason="completed",
                error_message=None,
            )
        else:
            handle.mark_finished(
                status="failed",
                exit_code=exit_code,
                stop_reason="failed",
                error_message=f"Command exited with code {exit_code}.",
            )
    finally:
        if handle.reader_task is not None:
            await handle.reader_task


async def _spawn_process(
    context: OpenClawToolContext,
    *,
    command: str,
    workdir: str,
    env: dict[str, str],
    timeout_seconds: float | None,
) -> ProcessHandle:
    process = await asyncio.create_subprocess_shell(
        command,
        cwd=workdir,
        env={**os.environ.copy(), **env},
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        start_new_session=os.name != "nt",
    )
    handle = ProcessHandle(
        owner_session_id=context.session_id,
        session_id=uuid.uuid4().hex,
        command=command,
        cwd=workdir,
        name=_derive_process_name(command),
        process=process,
        pid=process.pid,
        timeout_seconds=timeout_seconds,
    )
    handle.reader_task = asyncio.create_task(_stream_process_output(handle))
    handle.wait_task = asyncio.create_task(_wait_for_process(handle))
    return handle


def _slice_output_text(output: str, offset: int | None = None, limit: int | None = None) -> dict[str, Any]:
    lines = output.splitlines()
    using_default_tail = offset is None and limit is None
    effective_offset = max(0, offset or 0)
    effective_limit = limit

    if using_default_tail and len(lines) > DEFAULT_LOG_TAIL_LINES:
        effective_offset = len(lines) - DEFAULT_LOG_TAIL_LINES
        effective_limit = DEFAULT_LOG_TAIL_LINES

    if effective_limit is None:
        selected = lines[effective_offset:]
    else:
        if effective_limit < 0:
            raise ValueError("limit must be non-negative.")
        selected = lines[effective_offset : effective_offset + effective_limit]

    note = None
    if using_default_tail and len(lines) > DEFAULT_LOG_TAIL_LINES:
        note = f"showing last {DEFAULT_LOG_TAIL_LINES} of {len(lines)} lines"

    return {
        "text": "\n".join(selected),
        "offset": effective_offset,
        "limit": effective_limit,
        "total_lines": len(lines),
        "note": note,
    }


def _process_payload(handle: ProcessHandle, *, include_output: bool = False) -> dict[str, Any]:
    payload = handle.to_summary()
    if include_output:
        payload.update(_slice_output_text(handle.output))
    return payload


def _foreground_payload(handle: ProcessHandle) -> dict[str, Any]:
    payload = {
        "status": handle.status,
        "exit_code": handle.exit_code,
        "stop_reason": handle.stop_reason,
        "error_message": handle.error_message,
        "cwd": handle.cwd,
    }
    payload.update(_slice_output_text(handle.output))
    return payload


async def _wait_for_handle(handle: ProcessHandle, cancel_event: asyncio.Event | None) -> None:
    if handle.wait_task is None:
        return
    if cancel_event is None:
        await handle.wait_task
        return

    cancel_task = asyncio.create_task(cancel_event.wait())
    try:
        done, _pending = await asyncio.wait(
            {handle.wait_task, cancel_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if cancel_task in done and cancel_task.result():
            await _terminate_process(handle, "aborted")
            await handle.wait_task
            handle.mark_finished(
                status="failed",
                exit_code=handle.exit_code,
                stop_reason="aborted",
                error_message="Command aborted.",
            )
        else:
            await handle.wait_task
    finally:
        cancel_task.cancel()


def create_exec_tool(context: OpenClawToolContext) -> AgentTool:
    async def execute(tool_call_id, args, cancel_event=None, on_update=None):
        del tool_call_id, on_update
        command = str(args.get("command", "")).strip()
        if not command:
            return error_result("Missing required field: command.")

        if args.get("pty") is True:
            return error_result("pty is not supported in this Python claw build.")

        unsupported = [name for name in ("host", "security", "ask", "node", "elevated") if args.get(name) is not None]
        if unsupported:
            joined = ", ".join(sorted(unsupported))
            return error_result(f"Unsupported exec parameters: {joined}.")

        try:
            workdir = _resolve_workdir(context, args.get("workdir"))
            env = _validate_env(args.get("env"))
            timeout_seconds = _validate_timeout(args.get("timeout"))
            yield_ms = _validate_yield_ms(args.get("yield_ms"))
        except ValueError as exc:
            return error_result(str(exc))

        background = bool(args.get("background", False))

        try:
            handle = await _spawn_process(
                context,
                command=command,
                workdir=workdir,
                env=env,
                timeout_seconds=timeout_seconds,
            )
        except Exception as exc:  # noqa: BLE001
            return error_result(f"Failed to start command: {exc}")

        registry = get_process_registry()

        if background:
            registry.register(handle)
            await asyncio.sleep(0)
            return json_result(_process_payload(handle, include_output=True))

        if yield_ms is not None:
            try:
                await asyncio.wait_for(asyncio.shield(handle.wait_task), timeout=yield_ms / 1000)
            except asyncio.TimeoutError:
                registry.register(handle)
                return json_result(_process_payload(handle, include_output=True))
            await _wait_for_handle(handle, cancel_event)
            return json_result(_foreground_payload(handle))

        await _wait_for_handle(handle, cancel_event)
        return json_result(_foreground_payload(handle))

    return AgentTool(
        name="exec",
        label="exec",
        description=(
            "Execute a local shell command. Use background=true or yield_ms to keep the process "
            "running and manage it later with the process tool."
        ),
        parameters={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to execute."},
                "workdir": {"type": "string", "description": "Optional working directory inside the workspace."},
                "env": {"type": "object", "description": "Optional environment variable overrides."},
                "yield_ms": {
                    "type": "number",
                    "description": "Wait this many milliseconds before backgrounding a still-running command.",
                },
                "background": {"type": "boolean", "description": "Start the command in the background immediately."},
                "timeout": {"type": "number", "description": "Optional timeout in seconds."},
                "pty": {"type": "boolean", "description": "Unsupported in this build."},
            },
            "required": ["command"],
        },
        execute=execute,
    )


def create_process_tool(context: OpenClawToolContext) -> AgentTool:
    async def execute(tool_call_id, args, cancel_event=None, on_update=None):
        del tool_call_id, cancel_event, on_update
        action = str(args.get("action", "")).strip()
        if not action:
            return error_result("Missing required field: action.")

        registry = get_process_registry()

        if action == "list":
            return json_result(
                {
                    "status": "ok",
                    "items": [handle.to_summary() for handle in registry.list_for_owner(context.session_id)],
                }
            )

        session_id = str(args.get("session_id", "")).strip()
        if not session_id:
            return error_result("Missing required field: session_id.")

        handle = registry.resolve(context.session_id, session_id)
        if handle is None:
            return error_result("Process session is not visible from this claw session.", session_id=session_id)

        if action == "poll":
            try:
                timeout_ms = _validate_process_timeout_ms(args.get("timeout_ms", args.get("timeout")))
            except ValueError as exc:
                return error_result(str(exc), session_id=session_id)
            await handle.wait_for_activity(timeout_ms)
            return json_result(_process_payload(handle, include_output=True))

        if action == "log":
            offset = args.get("offset")
            limit = args.get("limit")
            if offset is not None and (not isinstance(offset, int) or offset < 0):
                return error_result("offset must be a non-negative integer.", session_id=session_id)
            if limit is not None and (not isinstance(limit, int) or limit < 0):
                return error_result("limit must be a non-negative integer.", session_id=session_id)
            payload = _process_payload(handle)
            payload.update(_slice_output_text(handle.output, offset=offset, limit=limit))
            return json_result(payload)

        if action == "write":
            data = args.get("data")
            if not isinstance(data, str) or not data:
                return error_result("Missing required field: data.", session_id=session_id)
            if handle.status == "running":
                process = handle.process
                stdin = process.stdin if process is not None else None
                if stdin is None or stdin.is_closing():
                    return error_result("Process stdin is not available.", session_id=session_id)
                stdin.write(data.encode("utf-8"))
                await stdin.drain()
                handle.mark_activity()
                return json_result(_process_payload(handle, include_output=True))
            return error_result("Process is not running.", session_id=session_id)

        if action == "kill":
            if handle.status == "running":
                await _terminate_process(handle, "killed")
                if handle.wait_task is not None:
                    await handle.wait_task
            return json_result(_process_payload(handle, include_output=True))

        if action == "remove":
            if handle.status == "running":
                return error_result("Cannot remove a running process session.", session_id=session_id)
            registry.remove(handle.session_id)
            return json_result({"status": "ok", "session_id": handle.session_id, "removed": True})

        return error_result("Unsupported action.", action=action, session_id=session_id)

    return AgentTool(
        name="process",
        label="process",
        description="Inspect and manage background exec sessions started by the current claw session.",
        parameters={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["list", "poll", "log", "write", "kill", "remove"],
                },
                "session_id": {"type": "string", "description": "Background process session id."},
                "data": {"type": "string", "description": "Data to write to stdin for write."},
                "offset": {"type": "integer", "description": "0-based log line offset for log."},
                "limit": {"type": "integer", "description": "Maximum log lines to return for log."},
                "timeout_ms": {
                    "type": "integer",
                    "description": "Optional poll wait timeout in milliseconds.",
                },
            },
            "required": ["action"],
        },
        execute=execute,
    )
