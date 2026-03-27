from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import asdict, is_dataclass
from typing import Any

from ..agent_session import AgentSession
from ..tool_error_limit import ToolErrorLimitExceededError, consume_tool_error_limit_exception
from ..types import PromptOptions


def _encode(value: Any) -> Any:
    if is_dataclass(value):
        return {k: _encode(v) for k, v in asdict(value).items()}
    if isinstance(value, list):
        return [_encode(v) for v in value]
    if isinstance(value, dict):
        return {k: _encode(v) for k, v in value.items()}
    return value


async def run_rpc_mode(session: AgentSession) -> int:
    def emit(event: Any) -> None:
        sys.stdout.write(json.dumps(_encode(event), default=str) + "\n")
        sys.stdout.flush()

    unsubscribe = session.subscribe(emit)
    loop = asyncio.get_event_loop()
    try:
        while True:
            line = await loop.run_in_executor(None, sys.stdin.readline)
            if not line:
                break
            if not line.strip():
                continue
            try:
                command = json.loads(line)
            except json.JSONDecodeError as exc:
                sys.stdout.write(
                    json.dumps({"type": "response", "id": None, "command": None, "success": False, "error": f"Invalid JSON: {exc}"})
                    + "\n"
                )
                sys.stdout.flush()
                continue
            command_type = command.get("type")
            command_id = command.get("id")
            try:
                if command_type == "prompt":
                    options = None
                    if command.get("streamingBehavior") is not None:
                        options = PromptOptions(streaming_behavior=command.get("streamingBehavior"))
                    await session.prompt(
                        command["message"],
                        options,
                    )
                    if (limit_error := consume_tool_error_limit_exception(session)) is not None:
                        raise limit_error
                    payload: Any = {}
                elif command_type == "steer":
                    await session.steer(command["message"])
                    if (limit_error := consume_tool_error_limit_exception(session)) is not None:
                        raise limit_error
                    payload = {}
                elif command_type == "follow_up":
                    await session.follow_up(command["message"])
                    if (limit_error := consume_tool_error_limit_exception(session)) is not None:
                        raise limit_error
                    payload = {}
                elif command_type == "get_state":
                    payload = _encode(session.get_state())
                elif command_type == "get_messages":
                    payload = {"messages": _encode(session.messages)}
                elif command_type == "get_session_stats":
                    payload = session.get_session_stats()
                elif command_type == "get_settings":
                    payload = session.get_settings_snapshot()
                elif command_type == "get_settings_schema":
                    payload = session.get_settings_schema()
                elif command_type == "update_settings":
                    payload = session.update_settings(command.get("patch", {}))
                elif command_type == "get_compaction_state":
                    payload = session.get_compaction_state()
                elif command_type == "get_compaction_schema":
                    payload = session.get_compaction_schema()
                elif command_type == "get_themes":
                    payload = session.get_themes()
                elif command_type == "get_theme_selector_schema":
                    payload = session.get_theme_selector_schema()
                elif command_type == "get_available_models":
                    payload = {"models": _encode(session.get_available_models())}
                elif command_type == "get_model_selector_schema":
                    payload = session.get_model_selector_schema()
                elif command_type == "get_thinking_selector_schema":
                    payload = session.get_thinking_selector_schema()
                elif command_type == "get_steering_selector_schema":
                    payload = session.get_steering_selector_schema()
                elif command_type == "get_follow_up_selector_schema":
                    payload = session.get_follow_up_selector_schema()
                elif command_type == "get_package_selector_schema":
                    payload = session.get_package_selector_schema()
                elif command_type == "get_resource_schema":
                    payload = session.get_resource_schema()
                elif command_type == "get_resource_item":
                    payload = session.get_resource_item(command["kind"], command["itemId"])
                elif command_type == "list_packages":
                    payload = {"packages": session.list_packages()}
                elif command_type == "install_package":
                    payload = session.install_package(command["source"], bool(command.get("local")))
                elif command_type == "remove_package":
                    payload = session.remove_package(command["source"], bool(command.get("local")))
                elif command_type == "update_packages":
                    payload = session.update_packages(command.get("source"))
                elif command_type == "get_package_selector_item":
                    payload = session.get_package_selector_item(command["source"])
                elif command_type == "get_command_schema":
                    payload = session.get_command_schema()
                elif command_type == "get_selector_registry":
                    payload = session.get_selector_registry()
                elif command_type == "get_selector":
                    payload = session.get_selector(command["selectorId"])
                elif command_type == "get_selector_item":
                    payload = session.get_selector_item(command["selectorId"], command["itemId"])
                elif command_type == "get_commands":
                    payload = {
                        "commands": [
                            {"name": item["name"], "source": "prompt", "description": item["description"]}
                            for item in session.get_command_schema()["commands"]
                        ]
                    }
                elif command_type == "get_fork_messages":
                    payload = {"messages": session.get_fork_messages()}
                elif command_type == "get_tree":
                    payload = {"tree": session.get_tree()}
                elif command_type == "get_tree_schema":
                    payload = session.get_tree_schema()
                elif command_type == "get_session_selector_schema":
                    payload = session.get_session_selector_schema()
                elif command_type == "get_last_assistant_text":
                    payload = {"text": session.get_last_assistant_text()}
                elif command_type == "list_sessions":
                    payload = {"sessions": session.list_sessions()}
                elif command_type == "list_session_infos":
                    payload = {"sessions": _encode(session.list_session_infos())}
                elif command_type == "resolve_session":
                    payload = {"path": session.resolve_session(command["query"])}
                elif command_type == "abort":
                    await session.abort()
                    payload = {}
                elif command_type == "set_model":
                    await session.set_model(command["provider"], command["modelId"])
                    payload = _encode(session.model)
                elif command_type == "set_theme":
                    payload = session.set_theme(command["theme"])
                elif command_type == "cycle_model":
                    payload = _encode(await session.cycle_model())
                elif command_type == "switch_session":
                    if command.get("sessionPath"):
                        payload = {"cancelled": not (await session.switch_session(command["sessionPath"]))}
                    else:
                        resolved = await session.resolve_and_switch_session(command["query"])
                        payload = {"cancelled": False, "path": resolved}
                elif command_type == "set_session_name":
                    name = command["name"].strip()
                    if not name:
                        raise ValueError("Session name cannot be empty")
                    session.set_session_name(name)
                    payload = {}
                elif command_type == "set_thinking_level":
                    session.set_thinking_level(command.get("thinkingLevel", command.get("level")))
                    payload = {"thinkingLevel": session.thinking_level}
                elif command_type == "cycle_thinking_level":
                    payload = {"thinkingLevel": session.cycle_thinking_level()}
                elif command_type == "set_steering_mode":
                    payload = {"steeringMode": session.set_steering_mode(command["mode"])}
                elif command_type == "set_follow_up_mode":
                    payload = {"followUpMode": session.set_follow_up_mode(command["mode"])}
                elif command_type == "set_compaction_enabled":
                    payload = session.set_compaction_enabled(command["enabled"])
                elif command_type == "set_compaction_reserve_tokens":
                    payload = session.set_compaction_reserve_tokens(int(command["reserveTokens"]))
                elif command_type == "set_compaction_keep_recent_tokens":
                    payload = session.set_compaction_keep_recent_tokens(int(command["keepRecentTokens"]))
                elif command_type == "bash":
                    payload = await session.bash(command["command"])
                elif command_type == "export_html":
                    payload = {"path": await session.export_to_html(command.get("outputPath"))}
                elif command_type == "fork":
                    result = await session.fork(command["entryId"])
                    payload = {"text": result["selectedText"], "cancelled": result["cancelled"]}
                elif command_type == "branch":
                    payload = session.branch(command.get("entryId"))
                elif command_type == "branch_with_summary":
                    payload = session.branch_with_summary(command.get("entryId"), command["summary"])
                elif command_type == "auto_branch_with_summary":
                    payload = await session.auto_branch_with_summary(command.get("entryId"), command.get("instructions"))
                elif command_type == "compact":
                    payload = session.compact(command["firstKeptEntryId"], command["summary"], int(command["tokensBefore"]))
                elif command_type == "auto_compact":
                    payload = await session.auto_compact(command["firstKeptEntryId"], command.get("instructions"))
                elif command_type == "generate_summary":
                    payload = await session.generate_summary(command.get("instructions"))
                elif command_type == "new_session":
                    await session.new_session()
                    payload = {"cancelled": False}
                else:
                    raise ValueError(f"Unknown RPC command: {command_type}")
                sys.stdout.write(
                    json.dumps({"type": "response", "id": command_id, "command": command_type, "success": True, "data": payload}, default=str)
                    + "\n"
                )
                sys.stdout.flush()
            except Exception as exc:  # noqa: BLE001
                sys.stdout.write(
                    json.dumps(
                        {"type": "response", "id": command_id, "command": command_type, "success": False, "error": str(exc)},
                        default=str,
                    )
                    + "\n"
                )
                sys.stdout.flush()
                if isinstance(exc, ToolErrorLimitExceededError):
                    return 1
    finally:
        unsubscribe()
    return 0
