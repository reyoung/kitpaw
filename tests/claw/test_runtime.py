from __future__ import annotations

from pathlib import Path

import pytest

from kitpaw.claw import ClawResourceLoader, CreateClawSessionOptions, create_claw_session
from kitpaw.pi_agent.ai import UserMessage
from kitpaw.pi_agent.code_agent.session_manager import SessionManager


def test_claw_resource_loader_builds_claw_prompt(tmp_path: Path) -> None:
    loader = ClawResourceLoader(str(tmp_path), str(tmp_path / "agent"), None)

    prompt = loader.build_system_prompt_with_tools(
        ["read", "write", "sessions_spawn", "subagents"],
    )

    assert "You are Claw" in prompt
    assert "## Tooling" in prompt
    assert "- sessions_spawn: Spawn a local subagent session" in prompt
    assert "Tool names are case-sensitive." in prompt
    assert "underlying code agent" in prompt


@pytest.mark.anyio
async def test_create_claw_session_binds_claw_tools_and_prompt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(tmp_path / "agent"))
    (tmp_path / "AGENTS.md").write_text("Follow the local project rules.", encoding="utf-8")

    result = await create_claw_session(
        CreateClawSessionOptions(
            cwd=str(tmp_path),
        )
    )
    session = result.session

    tool_names = {tool.name for tool in session.agent.state.tools}
    assert {
        "read",
        "edit",
        "write",
        "exec",
        "process",
        "apply_patch",
        "sessions_list",
        "sessions_history",
        "session_status",
        "sessions_spawn",
        "sessions_send",
        "sessions_yield",
        "subagents",
    } <= tool_names
    assert "You are Claw" in session.system_prompt
    assert "## Tooling" in session.system_prompt
    assert "## Safety" in session.system_prompt
    assert "# Project Context" in session.system_prompt
    assert "Follow the local project rules." in session.system_prompt
    assert isinstance(result.resource_loader, ClawResourceLoader)


@pytest.mark.anyio
async def test_create_claw_session_rebinds_resumed_sessions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(tmp_path / "agent"))

    session_root = tmp_path / "agent" / "sessions"
    manager = SessionManager.create(str(tmp_path), session_root)
    manager.set_session_name("saved")
    manager.append_message(UserMessage(content="resume me"))
    opened = SessionManager.open(manager.get_session_file())

    result = await create_claw_session(
        CreateClawSessionOptions(
            cwd=str(tmp_path),
            session_manager=opened,
        )
    )
    session = result.session

    assert session.session_id == manager.get_session_id()
    assert [message.role for message in session.messages] == ["user"]
    assert "You are Claw" in session.system_prompt
    assert "sessions_send" in {tool.name for tool in session.agent.state.tools}


@pytest.mark.anyio
async def test_create_claw_session_preserves_explicit_system_prompt_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(tmp_path / "agent"))

    result = await create_claw_session(
        CreateClawSessionOptions(
            cwd=str(tmp_path),
            system_prompt="OVERRIDE PROMPT",
        )
    )

    assert result.session.system_prompt == "OVERRIDE PROMPT"
