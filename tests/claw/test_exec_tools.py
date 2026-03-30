from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from kitpaw.claw import OpenClawToolContext, create_openclaw_coding_tools
from kitpaw.claw.registry import get_process_registry
from kitpaw.pi_agent.code_agent.session_manager import SessionManager


def _make_context(tmp_path: Path, *, session_id: str) -> OpenClawToolContext:
    return OpenClawToolContext(
        cwd=str(tmp_path),
        workspace_dir=str(tmp_path),
        spawn_workspace_dir=str(tmp_path),
        agent_id="claw",
        session_id=session_id,
        controller_session_id=session_id,
        model_provider="openai",
        model_id="gpt-4o-mini",
        thinking_level="medium",
        sandboxed=False,
        system_prompt="You are Claw.",
    )


def _tool_by_name(tools: list[Any], name: str):
    return next(tool for tool in tools if tool.name == name)


def _json_payload(result) -> dict[str, Any]:
    return json.loads(result.content[0].text)


@pytest.fixture(autouse=True)
def _clear_process_registry() -> None:
    get_process_registry().clear()


@pytest.mark.anyio
async def test_exec_foreground_success_failure_and_timeout(tmp_path: Path) -> None:
    manager = SessionManager.in_memory(str(tmp_path))
    context = _make_context(tmp_path, session_id=manager.get_session_id())
    tools = create_openclaw_coding_tools(str(tmp_path), context)
    exec_tool = _tool_by_name(tools, "exec")

    success = _json_payload(
        await exec_tool.execute(
            "exec-success",
            {"command": f'{sys.executable} -c "print(\\"hi\\")"'},
            None,
            None,
        )
    )
    assert success["status"] == "completed"
    assert success["exit_code"] == 0
    assert success["text"] == "hi"
    assert "session_id" not in success

    failure = _json_payload(
        await exec_tool.execute(
            "exec-failure",
            {"command": f'{sys.executable} -c "import sys; print(\\"bad\\"); sys.exit(3)"'},
            None,
            None,
        )
    )
    assert failure["status"] == "failed"
    assert failure["exit_code"] == 3
    assert "bad" in failure["text"]
    assert "session_id" not in failure

    timeout = _json_payload(
        await exec_tool.execute(
            "exec-timeout",
            {"command": f'{sys.executable} -c "import time; time.sleep(1)"', "timeout": 0.05},
            None,
            None,
        )
    )
    assert timeout["status"] == "failed"
    assert timeout["stop_reason"] == "timeout"
    assert "session_id" not in timeout


@pytest.mark.anyio
async def test_exec_background_and_process_lifecycle(tmp_path: Path) -> None:
    manager = SessionManager.in_memory(str(tmp_path))
    context = _make_context(tmp_path, session_id=manager.get_session_id())
    tools = create_openclaw_coding_tools(str(tmp_path), context)
    exec_tool = _tool_by_name(tools, "exec")
    process_tool = _tool_by_name(tools, "process")
    command = (
        f'{sys.executable} -u -c "import sys,time; '
        'print(\\"ready\\"); '
        'sys.stdout.flush(); '
        'line=sys.stdin.readline().strip(); '
        'print(f\\"echo:{line}\\"); '
        'sys.stdout.flush(); '
        'time.sleep(0.05)"'
    )

    started = _json_payload(
        await exec_tool.execute(
            "exec-background",
            {"command": command, "background": True},
            None,
            None,
        )
    )
    assert started["status"] == "running"
    session_id = started["session_id"]

    listed = _json_payload(
        await process_tool.execute("process-list", {"action": "list"}, None, None)
    )
    assert any(item["session_id"] == session_id for item in listed["items"])

    await process_tool.execute(
        "process-write",
        {"action": "write", "session_id": session_id, "data": "hello\n"},
        None,
        None,
    )

    polled = None
    for _ in range(5):
        polled = _json_payload(
            await process_tool.execute(
                "process-poll",
                {"action": "poll", "session_id": session_id, "timeout_ms": 1000},
                None,
                None,
            )
        )
        if polled["status"] == "completed":
            break
    assert polled is not None
    assert polled["status"] == "completed"
    assert "ready" in polled["text"]
    assert "echo:hello" in polled["text"]

    logged = _json_payload(
        await process_tool.execute(
            "process-log",
            {"action": "log", "session_id": session_id},
            None,
            None,
        )
    )
    assert "ready" in logged["text"]
    assert "echo:hello" in logged["text"]

    removed = _json_payload(
        await process_tool.execute(
            "process-remove",
            {"action": "remove", "session_id": session_id},
            None,
            None,
        )
    )
    assert removed["removed"] is True


@pytest.mark.anyio
async def test_exec_yield_ms_foreground_completion_and_background_timeout(tmp_path: Path) -> None:
    manager = SessionManager.in_memory(str(tmp_path))
    context = _make_context(tmp_path, session_id=manager.get_session_id())
    tools = create_openclaw_coding_tools(str(tmp_path), context)
    exec_tool = _tool_by_name(tools, "exec")
    process_tool = _tool_by_name(tools, "process")

    completed = _json_payload(
        await exec_tool.execute(
            "exec-yield-complete",
            {
                "command": f'{sys.executable} -u -c "import time; time.sleep(0.02); print(\\"done\\")"',
                "yield_ms": 100,
            },
            None,
            None,
        )
    )
    assert completed["status"] == "completed"
    assert completed["text"] == "done"
    assert "session_id" not in completed

    running = _json_payload(
        await exec_tool.execute(
            "exec-yield-running",
            {
                "command": f'{sys.executable} -u -c "import time; time.sleep(0.2); print(\\"done\\")"',
                "yield_ms": 20,
            },
            None,
            None,
        )
    )
    assert running["status"] == "running"
    session_id = running["session_id"]

    final = None
    for _ in range(5):
        final = _json_payload(
            await process_tool.execute(
                "process-poll-yield",
                {"action": "poll", "session_id": session_id, "timeout_ms": 1000},
                None,
                None,
            )
        )
        if final["status"] == "completed":
            break
    assert final is not None
    assert final["status"] == "completed"
    assert "done" in final["text"]


@pytest.mark.anyio
async def test_exec_rejects_unsupported_params_and_workdir_escape(tmp_path: Path) -> None:
    manager = SessionManager.in_memory(str(tmp_path))
    context = _make_context(tmp_path, session_id=manager.get_session_id())
    tools = create_openclaw_coding_tools(str(tmp_path), context)
    exec_tool = _tool_by_name(tools, "exec")

    unsupported = _json_payload(
        await exec_tool.execute(
            "exec-pty",
            {"command": "echo hi", "pty": True},
            None,
            None,
        )
    )
    assert unsupported["status"] == "error"
    assert "pty" in unsupported["error"]

    escaped = _json_payload(
        await exec_tool.execute(
            "exec-workdir",
            {"command": "echo hi", "workdir": "../outside"},
            None,
            None,
        )
    )
    assert escaped["status"] == "error"
    assert "outside the working directory" in escaped["error"]
