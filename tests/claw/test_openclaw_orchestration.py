from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from kitpaw.claw import OpenClawToolContext, create_openclaw_coding_tools
from kitpaw.pi_agent.ai import UserMessage, get_model
from kitpaw.pi_agent.code_agent.sdk import CreateAgentSessionOptions, create_agent_session
from kitpaw.pi_agent.code_agent.session_manager import SessionManager
from tests.test_mock_e2e import make_chunk, run_mock_openai_server


def _make_context(
    tmp_path: Path,
    *,
    session_id: str,
    controller_session_id: str | None = None,
    on_yield=None,
) -> OpenClawToolContext:
    return OpenClawToolContext(
        cwd=str(tmp_path),
        workspace_dir=str(tmp_path),
        spawn_workspace_dir=str(tmp_path),
        agent_id="main",
        session_id=session_id,
        controller_session_id=controller_session_id or session_id,
        model_provider="openai",
        model_id="gpt-4o-mini",
        thinking_level="medium",
        sandboxed=False,
        system_prompt="You are a helpful orchestration test agent.",
        on_yield=on_yield,
    )


def _tool_by_name(tools: list[Any], name: str):
    return next(tool for tool in tools if tool.name == name)


def _json_payload(result) -> dict[str, Any]:
    return json.loads(result.content[0].text)


def _append_assistant(manager: SessionManager, text: str) -> None:
    manager.append_message(
        {
            "role": "assistant",
            "content": [{"type": "text", "text": text}],
            "api": "openai-completions",
            "provider": "openai",
            "model": "gpt-4o-mini",
            "usage": {},
            "stopReason": "stop",
            "timestamp": 1,
        }
    )


def test_openclaw_tool_inventory_and_names(tmp_path: Path) -> None:
    manager = SessionManager.in_memory(str(tmp_path))
    context = _make_context(tmp_path, session_id=manager.get_session_id())

    tools = create_openclaw_coding_tools(str(tmp_path), context)

    names = {tool.name for tool in tools}
    assert names == {
        "read",
        "bash",
        "edit",
        "write",
        "uv",
        "sessions_list",
        "sessions_history",
        "session_status",
        "sessions_spawn",
        "sessions_send",
        "sessions_yield",
        "subagents",
    }


@pytest.mark.anyio
async def test_sessions_list_history_and_status(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(tmp_path / "agent"))

    session_root = tmp_path / "agent" / "sessions"
    current = SessionManager.create(str(tmp_path), session_root)
    current.set_session_name("current")
    current.append_message(UserMessage(content="current prompt"))
    _append_assistant(current, "current reply")

    other = SessionManager.create(str(tmp_path), session_root)
    other.set_session_name("other")
    other.append_message(UserMessage(content="other prompt"))
    _append_assistant(other, "other reply")

    context = _make_context(tmp_path, session_id=current.get_session_id())
    tools = create_openclaw_coding_tools(str(tmp_path), context)

    list_payload = _json_payload(
        await _tool_by_name(tools, "sessions_list").execute("list", {}, None, None)
    )
    assert {item["session_id"] for item in list_payload["items"]} == {
        current.get_session_id(),
        other.get_session_id(),
    }
    current_item = next(item for item in list_payload["items"] if item["session_id"] == current.get_session_id())
    assert current_item["name"] == "current"

    history_payload = _json_payload(
        await _tool_by_name(tools, "sessions_history").execute(
            "history",
            {"session_id": other.get_session_id()},
            None,
            None,
        )
    )
    assert [item["text"] for item in history_payload["messages"]] == [
        "other prompt",
        "other reply",
    ]

    status_payload = _json_payload(
        await _tool_by_name(tools, "session_status").execute("status", {}, None, None)
    )
    assert status_payload["session"]["id"] == current.get_session_id()
    assert status_payload["session"]["name"] == "current"
    assert status_payload["stats"]["messageCount"] == 2


@pytest.mark.anyio
async def test_sessions_spawn_session_send_subagents_and_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(tmp_path / "agent"))
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")

    with run_mock_openai_server(
        [
            make_chunk(delta={"content": "child ok"}),
            make_chunk(delta={}, finish_reason="stop", usage={"prompt_tokens": 4, "completion_tokens": 2}),
        ]
    ) as (base_url, _state):
        monkeypatch.setenv("OPENAI_BASE_URL", base_url)

        manager = SessionManager.in_memory(str(tmp_path))
        context = _make_context(tmp_path, session_id=manager.get_session_id())
        tools = create_openclaw_coding_tools(str(tmp_path), context)

        spawn_payload = _json_payload(
            await _tool_by_name(tools, "sessions_spawn").execute(
                "spawn",
                {"task": "first task", "mode": "session", "label": "worker"},
                None,
                None,
            )
        )
        assert spawn_payload["status"] == "ok"
        child_session_id = spawn_payload["session_id"]

        list_payload = _json_payload(
            await _tool_by_name(tools, "subagents").execute(
                "subagents-list",
                {"action": "list"},
                None,
                None,
            )
        )
        assert any(item["session_id"] == child_session_id for item in list_payload["items"])

        send_payload = _json_payload(
            await _tool_by_name(tools, "sessions_send").execute(
                "send",
                {"session_id": child_session_id, "message": "second task"},
                None,
                None,
            )
        )
        assert send_payload["status"] == "ok"

        steer_payload = _json_payload(
            await _tool_by_name(tools, "subagents").execute(
                "subagents-steer",
                {"action": "steer", "target": child_session_id, "message": "third task"},
                None,
                None,
            )
        )
        assert steer_payload["status"] == "ok"

        history_payload = _json_payload(
            await _tool_by_name(tools, "sessions_history").execute(
                "history-child",
                {"session_id": child_session_id},
                None,
                None,
            )
        )
        assert [item["text"] for item in history_payload["messages"] if item["role"] == "user"] == [
            "first task",
            "second task",
            "third task",
        ]

        kill_payload = _json_payload(
            await _tool_by_name(tools, "subagents").execute(
                "subagents-kill",
                {"action": "kill", "target": child_session_id},
                None,
                None,
            )
        )
        assert kill_payload["status"] == "ok"

        list_after_payload = _json_payload(
            await _tool_by_name(tools, "subagents").execute(
                "subagents-list-after",
                {"action": "list"},
                None,
                None,
            )
        )
        assert not any(item["session_id"] == child_session_id for item in list_after_payload["items"])


@pytest.mark.anyio
async def test_sessions_spawn_run_cleanup_delete_removes_registry_handle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(tmp_path / "agent"))
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")

    with run_mock_openai_server(
        [
            make_chunk(delta={"content": "one shot"}),
            make_chunk(delta={}, finish_reason="stop", usage={"prompt_tokens": 4, "completion_tokens": 2}),
        ]
    ) as (base_url, _state):
        monkeypatch.setenv("OPENAI_BASE_URL", base_url)

        manager = SessionManager.in_memory(str(tmp_path))
        context = _make_context(tmp_path, session_id=manager.get_session_id())
        tools = create_openclaw_coding_tools(str(tmp_path), context)

        spawn_payload = _json_payload(
            await _tool_by_name(tools, "sessions_spawn").execute(
                "spawn-run",
                {"task": "one shot task", "mode": "run", "cleanup": "delete"},
                None,
                None,
            )
        )
        assert spawn_payload["status"] == "ok"
        assert spawn_payload["output"] == "one shot"

        list_payload = _json_payload(
            await _tool_by_name(tools, "subagents").execute(
                "subagents-list",
                {"action": "list"},
                None,
                None,
            )
        )
        assert list_payload["items"] == []


@pytest.mark.anyio
async def test_sessions_yield_noop_and_abort_with_handler(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = SessionManager.in_memory(str(tmp_path))

    noop_context = _make_context(tmp_path, session_id=manager.get_session_id())
    noop_tools = create_openclaw_coding_tools(str(tmp_path), noop_context)
    noop_payload = _json_payload(
        await _tool_by_name(noop_tools, "sessions_yield").execute(
            "yield-noop",
            {"message": "pause"},
            None,
            None,
        )
    )
    assert noop_payload["status"] == "ok"
    assert noop_payload["yielded"] is False

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")

    yielded: list[str] = []
    context = _make_context(
        tmp_path,
        session_id=manager.get_session_id(),
        on_yield=yielded.append,
    )
    tools = create_openclaw_coding_tools(str(tmp_path), context)

    with run_mock_openai_server(
        [
            make_chunk(
                delta={
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "sessions_yield",
                                "arguments": '{"message":"pause"}',
                            },
                        }
                    ]
                },
                finish_reason="tool_calls",
                usage={"prompt_tokens": 4, "completion_tokens": 2},
            ),
        ]
    ) as (base_url, _state):
        model = replace(get_model("openai", "gpt-4o-mini"), base_url=base_url)
        result = await create_agent_session(
            CreateAgentSessionOptions(
                cwd=str(tmp_path),
                model=model,
                tools=tools,
                session_manager=manager,
            )
        )

        await result.session.prompt("yield now")

    assert yielded == ["pause"]
    last_message = result.session.messages[-1]
    assert getattr(last_message, "stop_reason", None) == "aborted"
    assert getattr(last_message, "error_message", None) == "sessions_yield"
