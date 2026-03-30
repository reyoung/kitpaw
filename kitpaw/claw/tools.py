from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable
from pathlib import Path
from typing import Any

from ..pi_agent.agent.types import AgentTool, AgentToolResult
from ..pi_agent.ai.models import get_model
from ..pi_agent.ai.types import TextContent
from ..pi_agent.code_agent.sdk import CreateAgentSessionOptions, create_agent_session
from ..pi_agent.code_agent.session_manager import SessionInfo, SessionManager
from ..pi_agent.code_agent.tools import create_coding_tools
from .context import OpenClawToolContext
from .registry import SubagentHandle, get_subagent_registry, now_iso


def _json_result(payload: dict[str, Any]) -> AgentToolResult[dict[str, Any]]:
    return AgentToolResult(
        content=[TextContent(text=json.dumps(payload))],
        details=payload,
    )


def _error_result(message: str, **extra: Any) -> AgentToolResult[dict[str, Any]]:
    return _json_result({"status": "error", "error": message, **extra})


def _extract_message_text(message: dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return ""


def _list_session_infos(context: OpenClawToolContext) -> list[SessionInfo]:
    return SessionManager.list_session_infos(context.workspace_dir)


def _resolve_session_info(
    context: OpenClawToolContext,
    session_id: str | None = None,
) -> SessionInfo | None:
    target = (session_id or context.session_id).strip()
    if not target:
        return None
    for info in _list_session_infos(context):
        if info.id == target or info.path == target or (info.name and info.name == target):
            return info
    return None


def _session_info_payload(info: SessionInfo, context: OpenClawToolContext) -> dict[str, Any]:
    return {
        "id": info.id,
        "session_id": info.id,
        "path": info.path,
        "name": info.name,
        "cwd": info.cwd,
        "created": info.created,
        "modified": info.modified,
        "message_count": info.message_count,
        "first_message": info.first_message,
        "is_current": info.id == context.session_id,
    }


def _serialize_history_messages(manager: SessionManager, limit: int | None = None) -> list[dict[str, Any]]:
    messages = manager.build_runtime_context()["messages"]
    items: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role not in {"user", "assistant"}:
            continue
        items.append(
            {
                "role": role,
                "text": _extract_message_text(message),
                "timestamp": message.get("timestamp"),
            }
        )
    if limit is not None and limit > 0:
        return items[-limit:]
    return items


def _build_child_context(
    parent_context: OpenClawToolContext,
    *,
    child_session_id: str,
    model_id: str,
    thinking_level: str,
    workspace_dir: str,
) -> OpenClawToolContext:
    return OpenClawToolContext(
        cwd=workspace_dir,
        workspace_dir=workspace_dir,
        spawn_workspace_dir=workspace_dir,
        agent_id=parent_context.agent_id,
        session_id=child_session_id,
        controller_session_id=child_session_id,
        model_provider=parent_context.model_provider,
        model_id=model_id,
        thinking_level=thinking_level,
        sandboxed=parent_context.sandboxed,
        system_prompt=parent_context.system_prompt,
        on_yield=parent_context.on_yield,
    )


async def _create_child_handle(
    parent_context: OpenClawToolContext,
    *,
    label: str | None,
    mode: str,
    cleanup: str,
    workspace_dir: str,
    model_id: str,
    thinking_level: str,
) -> SubagentHandle:
    child_manager = SessionManager.create(workspace_dir)
    child_context = _build_child_context(
        parent_context,
        child_session_id=child_manager.get_session_id(),
        model_id=model_id,
        thinking_level=thinking_level,
        workspace_dir=workspace_dir,
    )
    child_tools = create_openclaw_coding_tools(workspace_dir, child_context)
    model = get_model(parent_context.model_provider, model_id)
    result = await create_agent_session(
        CreateAgentSessionOptions(
            cwd=workspace_dir,
            model=model,
            thinking_level=thinking_level,
            tools=child_tools,
            session_manager=child_manager,
        )
    )
    if parent_context.system_prompt:
        result.session.agent.set_system_prompt(parent_context.system_prompt)
    if label:
        result.session.set_session_name(label)
    return SubagentHandle(
        controller_session_id=parent_context.controller_session_id,
        session_id=child_manager.get_session_id(),
        label=label or None,
        session=result.session,
        mode=mode,
        cleanup=cleanup,
    )


async def _run_subagent(handle: SubagentHandle, message: str) -> dict[str, Any]:
    if handle.current_task is not None and not handle.current_task.done():
        return {
            "status": "error",
            "error": "Subagent is already running.",
            "session_id": handle.session_id,
        }

    async def runner() -> dict[str, Any]:
        output_parts: list[str] = []

        def listener(event: Any) -> None:
            if getattr(event, "type", None) != "message_update":
                return
            assistant_event = getattr(event, "assistant_message_event", None)
            if getattr(assistant_event, "type", None) == "text_delta":
                output_parts.append(assistant_event.delta)

        unsubscribe = handle.session.subscribe(listener)
        try:
            await handle.session.prompt(message)
        finally:
            unsubscribe()

        output = "".join(output_parts).strip()
        if not output:
            output = handle.session.session_manager.get_last_assistant_text() or ""
        last_message = handle.session.messages[-1] if handle.session.messages else None
        stop_reason = getattr(last_message, "stop_reason", None)
        error_message = getattr(last_message, "error_message", None)
        status = "completed"
        if stop_reason == "aborted":
            status = "aborted"
        elif stop_reason == "error" or error_message:
            status = "error"
        return {
            "status": status,
            "output": output,
            "stop_reason": stop_reason,
            "error_message": error_message,
        }

    handle.status = "running"
    handle.last_task = message
    handle.updated_at = now_iso()
    task = asyncio.create_task(runner())
    handle.current_task = task
    try:
        result = await task
    except Exception as exc:  # noqa: BLE001
        result = {
            "status": "error",
            "output": "",
            "stop_reason": "error",
            "error_message": str(exc),
        }
    finally:
        if handle.current_task is task:
            handle.current_task = None

    handle.status = str(result["status"])
    handle.last_output = str(result.get("output", ""))
    handle.stop_reason = result.get("stop_reason")
    handle.error_message = result.get("error_message")
    handle.updated_at = now_iso()
    return result


async def _run_subagent_with_timeout(
    handle: SubagentHandle,
    message: str,
    timeout_seconds: int | None,
) -> dict[str, Any]:
    if timeout_seconds is None or timeout_seconds <= 0:
        return await _run_subagent(handle, message)
    try:
        return await asyncio.wait_for(_run_subagent(handle, message), timeout=timeout_seconds)
    except asyncio.TimeoutError:
        await handle.session.abort()
        handle.status = "error"
        handle.stop_reason = "timeout"
        handle.error_message = f"Subagent run timed out after {timeout_seconds} seconds."
        handle.updated_at = now_iso()
        return {
            "status": "error",
            "output": "",
            "stop_reason": "timeout",
            "error_message": handle.error_message,
        }


def create_sessions_list_tool(context: OpenClawToolContext) -> AgentTool:
    async def execute(tool_call_id, args, cancel_event=None, on_update=None):
        infos = _list_session_infos(context)
        return _json_result(
            {
                "status": "ok",
                "current_session_id": context.session_id,
                "items": [_session_info_payload(info, context) for info in infos],
            }
        )

    return AgentTool(
        name="sessions_list",
        label="Sessions List",
        description="List saved sessions for the current workspace.",
        parameters={"type": "object", "properties": {}, "required": []},
        execute=execute,
    )


def create_sessions_history_tool(context: OpenClawToolContext) -> AgentTool:
    async def execute(tool_call_id, args, cancel_event=None, on_update=None):
        session_id = args.get("session_id")
        limit_raw = args.get("limit")
        limit = int(limit_raw) if isinstance(limit_raw, int) and limit_raw > 0 else None
        info = _resolve_session_info(context, session_id)
        if info is None:
            return _error_result("Session not found.", session_id=session_id or context.session_id)
        manager = SessionManager.open(info.path)
        return _json_result(
            {
                "status": "ok",
                "session": _session_info_payload(info, context),
                "messages": _serialize_history_messages(manager, limit=limit),
            }
        )

    return AgentTool(
        name="sessions_history",
        label="Sessions History",
        description="Show user and assistant message history for a saved session.",
        parameters={
            "type": "object",
            "properties": {
                "session_id": {"type": "string", "description": "Session id, name, or path."},
                "limit": {"type": "integer", "description": "Optional message limit."},
            },
            "required": [],
        },
        execute=execute,
    )


def create_session_status_tool(context: OpenClawToolContext) -> AgentTool:
    async def execute(tool_call_id, args, cancel_event=None, on_update=None):
        session_id = args.get("session_id")
        info = _resolve_session_info(context, session_id)
        if info is None:
            return _error_result("Session not found.", session_id=session_id or context.session_id)
        manager = SessionManager.open(info.path)
        runtime = manager.build_runtime_context()
        return _json_result(
            {
                "status": "ok",
                "session": _session_info_payload(info, context),
                "stats": manager.get_stats(),
                "runtime": {
                    "model": runtime.get("model"),
                    "thinking_level": runtime.get("thinkingLevel"),
                },
            }
        )

    return AgentTool(
        name="session_status",
        label="Session Status",
        description="Show session metadata, message counts, and runtime state.",
        parameters={
            "type": "object",
            "properties": {
                "session_id": {"type": "string", "description": "Session id, name, or path."},
            },
            "required": [],
        },
        execute=execute,
    )


def create_sessions_spawn_tool(context: OpenClawToolContext) -> AgentTool:
    async def execute(tool_call_id, args, cancel_event=None, on_update=None):
        runtime = args.get("runtime", "subagent")
        if runtime != "subagent":
            return _error_result("Only runtime='subagent' is supported in Python OpenClaw.")
        task = args.get("task", "").strip()
        if not task:
            return _error_result("Missing required field: task.")
        mode = args.get("mode", "run")
        if mode not in {"run", "session"}:
            return _error_result("Unsupported mode.", mode=mode)
        cleanup = args.get("cleanup", "keep")
        if cleanup not in {"keep", "delete"}:
            return _error_result("Unsupported cleanup mode.", cleanup=cleanup)
        workspace_dir = str(Path(args.get("cwd") or context.spawn_workspace_dir or context.workspace_dir).resolve())
        model_id = str(args.get("model") or context.model_id)
        thinking_level = str(args.get("thinking") or context.thinking_level)
        timeout_seconds = args.get("run_timeout_seconds")
        if timeout_seconds is not None and not isinstance(timeout_seconds, int):
            return _error_result("run_timeout_seconds must be an integer.")

        handle = await _create_child_handle(
            context,
            label=str(args.get("label")).strip() if isinstance(args.get("label"), str) and args.get("label").strip() else None,
            mode=mode,
            cleanup=cleanup,
            workspace_dir=workspace_dir,
            model_id=model_id,
            thinking_level=thinking_level,
        )
        registry = get_subagent_registry()
        registry.register(handle)
        run_result = await _run_subagent_with_timeout(handle, task, timeout_seconds)
        if cleanup == "delete":
            registry.remove(handle.session_id)
        status = run_result.get("status", "ok")
        payload = {
            "status": "ok" if status != "error" else "error",
            "session_id": handle.session_id,
            "session_file": handle.session.session_file,
            "label": handle.label,
            "mode": mode,
            "cleanup": cleanup,
            "output": run_result.get("output", ""),
            "stop_reason": run_result.get("stop_reason"),
            "error_message": run_result.get("error_message"),
        }
        if status == "error":
            payload["error"] = run_result.get("error_message") or "Subagent run failed."
        return _json_result(payload)

    return AgentTool(
        name="sessions_spawn",
        label="Sessions Spawn",
        description="Spawn a local child session and optionally keep it for follow-up work.",
        parameters={
            "type": "object",
            "properties": {
                "task": {"type": "string", "description": "Initial task for the child session."},
                "label": {"type": "string", "description": "Optional child session label."},
                "runtime": {"type": "string", "enum": ["subagent"]},
                "model": {"type": "string", "description": "Optional model override."},
                "thinking": {"type": "string", "description": "Optional thinking level override."},
                "cwd": {"type": "string", "description": "Optional workspace override."},
                "mode": {"type": "string", "enum": ["run", "session"]},
                "cleanup": {"type": "string", "enum": ["keep", "delete"]},
                "run_timeout_seconds": {"type": "integer", "description": "Optional run timeout."},
            },
            "required": ["task"],
        },
        execute=execute,
    )


def create_sessions_send_tool(context: OpenClawToolContext) -> AgentTool:
    async def execute(tool_call_id, args, cancel_event=None, on_update=None):
        session_id = str(args.get("session_id", "")).strip()
        message = str(args.get("message", "")).strip()
        if not session_id:
            return _error_result("Missing required field: session_id.")
        if not message:
            return _error_result("Missing required field: message.")
        registry = get_subagent_registry()
        handle = registry.resolve(context.controller_session_id, session_id)
        if handle is None:
            return _error_result("Session is not visible from this controller.", session_id=session_id)
        run_result = await _run_subagent(handle, message)
        if handle.cleanup == "delete":
            registry.remove(handle.session_id)
        payload = {
            "status": "ok" if run_result.get("status") != "error" else "error",
            "session_id": handle.session_id,
            "output": run_result.get("output", ""),
            "stop_reason": run_result.get("stop_reason"),
            "error_message": run_result.get("error_message"),
        }
        if run_result.get("status") == "error":
            payload["error"] = run_result.get("error_message") or "Subagent run failed."
        return _json_result(payload)

    return AgentTool(
        name="sessions_send",
        label="Sessions Send",
        description="Send a follow-up message to a visible child session.",
        parameters={
            "type": "object",
            "properties": {
                "session_id": {"type": "string", "description": "Child session id."},
                "message": {"type": "string", "description": "Message to send."},
            },
            "required": ["session_id", "message"],
        },
        execute=execute,
    )


def create_sessions_yield_tool(context: OpenClawToolContext) -> AgentTool:
    async def execute(tool_call_id, args, cancel_event=None, on_update=None):
        message = str(args.get("message", "")).strip()
        if not message:
            return _error_result("Missing required field: message.")
        if context.on_yield is None:
            return _json_result(
                {
                    "status": "ok",
                    "yielded": False,
                    "aborted": False,
                    "message": message,
                }
            )
        callback_result = context.on_yield(message)
        if isinstance(callback_result, Awaitable):
            await callback_result
        raise asyncio.CancelledError("sessions_yield")

    return AgentTool(
        name="sessions_yield",
        label="Sessions Yield",
        description="Notify the outer runtime and abort the current run when a yield handler is bound.",
        parameters={
            "type": "object",
            "properties": {
                "message": {"type": "string", "description": "Yield message."},
            },
            "required": ["message"],
        },
        execute=execute,
    )


def create_subagents_tool(context: OpenClawToolContext) -> AgentTool:
    async def execute(tool_call_id, args, cancel_event=None, on_update=None):
        action = str(args.get("action") or "list")
        registry = get_subagent_registry()
        if action == "list":
            return _json_result(
                {
                    "status": "ok",
                    "items": [
                        handle.to_summary()
                        for handle in registry.list_for_controller(context.controller_session_id)
                    ],
                }
            )

        target = str(args.get("target", "")).strip()
        if not target:
            return _error_result("Missing required field: target.")
        handle = registry.resolve(context.controller_session_id, target)
        if handle is None:
            return _error_result("Unknown subagent target.", target=target)

        if action == "steer":
            message = str(args.get("message", "")).strip()
            if not message:
                return _error_result("Missing required field: message.", target=target)
            run_result = await _run_subagent(handle, message)
            if handle.cleanup == "delete":
                registry.remove(handle.session_id)
            payload = {
                "status": "ok" if run_result.get("status") != "error" else "error",
                "session_id": handle.session_id,
                "label": handle.label,
                "output": run_result.get("output", ""),
                "stop_reason": run_result.get("stop_reason"),
                "error_message": run_result.get("error_message"),
            }
            if run_result.get("status") == "error":
                payload["error"] = run_result.get("error_message") or "Subagent run failed."
            return _json_result(payload)

        if action == "kill":
            if handle.current_task is not None and not handle.current_task.done():
                await handle.session.abort()
            registry.remove(handle.session_id)
            return _json_result(
                {
                    "status": "ok",
                    "session_id": handle.session_id,
                    "label": handle.label,
                }
            )

        return _error_result("Unsupported action.", action=action)

    return AgentTool(
        name="subagents",
        label="Subagents",
        description="List, steer, or kill child sessions owned by the current controller session.",
        parameters={
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["list", "steer", "kill"]},
                "target": {"type": "string", "description": "Session id or label."},
                "message": {"type": "string", "description": "Follow-up message for steer."},
            },
            "required": [],
        },
        execute=execute,
    )


def create_openclaw_coding_tools(cwd: str, context: OpenClawToolContext) -> list[AgentTool]:
    return [
        *create_coding_tools(cwd),
        create_sessions_list_tool(context),
        create_sessions_history_tool(context),
        create_session_status_tool(context),
        create_sessions_spawn_tool(context),
        create_sessions_send_tool(context),
        create_sessions_yield_tool(context),
        create_subagents_tool(context),
    ]
