from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from kitpaw.pi_agent.code_agent.codex.resource_loader import CodexResourceLoader
from kitpaw.pi_agent.code_agent.codex.system_prompt import build_codex_system_prompt
from kitpaw.pi_agent.code_agent.codex.tools import create_codex_tools
from kitpaw.pi_agent.code_agent.sdk import CreateAgentSessionOptions, create_agent_session


# ---------------------------------------------------------------------------
# System prompt tests
# ---------------------------------------------------------------------------

def test_codex_system_prompt_includes_core_sections() -> None:
    prompt = build_codex_system_prompt(
        available_tools=["shell", "apply_patch", "update_plan"],
        cwd="/tmp/myproject",
    )
    # Uses original prompt.md verbatim
    assert "Personality" in prompt
    assert "AGENTS.md" in prompt
    assert "Preamble" in prompt
    assert "Planning" in prompt
    assert "Task execution" in prompt
    assert "update_plan" in prompt


def test_codex_system_prompt_ignores_project_rules() -> None:
    """Project rules are NOT in the system prompt (sent as separate messages)."""
    prompt = build_codex_system_prompt(
        available_tools=["shell"],
        cwd="/tmp",
        project_rules=[("AGENTS.md", "Always use type hints.")],
    )
    assert "Always use type hints." not in prompt
    assert "Personality" in prompt


# ---------------------------------------------------------------------------
# Tool tests
# ---------------------------------------------------------------------------

def test_codex_tools_count_and_names() -> None:
    tools = create_codex_tools("/tmp")
    names = {t.name for t in tools}
    assert len(tools) == 5
    assert names == {"shell", "apply_patch", "update_plan", "view_image", "request_user_input"}


def test_codex_tool_schemas_have_required_fields() -> None:
    tools = create_codex_tools("/tmp")
    for tool in tools:
        assert tool.name
        assert tool.label
        assert tool.description
        assert isinstance(tool.parameters, dict)
        assert tool.parameters.get("type") == "object"


@pytest.mark.anyio
async def test_shell_tool(tmp_path: Path) -> None:
    tools = create_codex_tools(str(tmp_path))
    shell_tool = next(t for t in tools if t.name == "shell")
    result = await shell_tool.execute("c1", {"command": ["echo", "hello"]}, None, None)
    text = result.content[0].text
    assert "hello" in text


@pytest.mark.anyio
async def test_shell_tool_timeout(tmp_path: Path) -> None:
    tools = create_codex_tools(str(tmp_path))
    shell_tool = next(t for t in tools if t.name == "shell")
    result = await shell_tool.execute("c1", {"command": ["sleep", "10"], "timeout_ms": 1000}, None, None)
    text = result.content[0].text
    assert "timeout" in text.lower() or "timed out" in text.lower()


@pytest.mark.anyio
async def test_apply_patch_create_file(tmp_path: Path) -> None:
    tools = create_codex_tools(str(tmp_path))
    patch_tool = next(t for t in tools if t.name == "apply_patch")

    patch = (
        "--- /dev/null\n"
        "+++ b/new_file.txt\n"
        "@@ -0,0 +1,2 @@\n"
        "+hello\n"
        "+world\n"
    )
    result = await patch_tool.execute("c1", {"patch": patch}, None, None)
    text = result.content[0].text
    # Either the patch command succeeded or our fallback created the file
    assert "new_file" in text or (tmp_path / "new_file.txt").exists()


@pytest.mark.anyio
async def test_update_plan_tool() -> None:
    tools = create_codex_tools("/tmp")
    plan_tool = next(t for t in tools if t.name == "update_plan")
    result = await plan_tool.execute("c1", {
        "plan": [
            {"step": "Read code", "status": "completed"},
            {"step": "Write tests", "status": "in_progress"},
            {"step": "Refactor", "status": "pending"},
        ],
        "explanation": "Starting tests now",
    }, None, None)
    text = result.content[0].text
    assert "Read code" in text
    assert "Write tests" in text
    assert "Refactor" in text
    assert "Starting tests now" in text


@pytest.mark.anyio
async def test_view_image_tool(tmp_path: Path) -> None:
    tools = create_codex_tools(str(tmp_path))
    view_tool = next(t for t in tools if t.name == "view_image")

    # Create a fake image file
    img_file = tmp_path / "test.png"
    img_file.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

    result = await view_tool.execute("c1", {"path": str(img_file)}, None, None)
    text = result.content[0].text
    assert str(img_file) in text
    assert "size_bytes" in text


@pytest.mark.anyio
async def test_view_image_tool_missing_file(tmp_path: Path) -> None:
    tools = create_codex_tools(str(tmp_path))
    view_tool = next(t for t in tools if t.name == "view_image")
    result = await view_tool.execute("c1", {"path": str(tmp_path / "nope.png")}, None, None)
    text = result.content[0].text
    assert "error" in text.lower() or "not found" in text.lower()


@pytest.mark.anyio
async def test_request_user_input_tool(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import io
    monkeypatch.setattr("sys.stdin", io.StringIO("yes\n"))

    tools = create_codex_tools(str(tmp_path))
    input_tool = next(t for t in tools if t.name == "request_user_input")
    result = await input_tool.execute("c1", {
        "questions": [{"question": "Continue?", "options": ["yes", "no"]}],
    }, None, None)
    text = result.content[0].text
    assert "yes" in text
    assert "Continue?" in text


# ---------------------------------------------------------------------------
# Resource loader tests
# ---------------------------------------------------------------------------

def test_codex_resource_loader_build_system_prompt() -> None:
    loader = CodexResourceLoader("/tmp", "/tmp/agent", None)
    prompt = loader.build_system_prompt(None, [])
    assert "Personality" in prompt


def test_codex_resource_loader_agents_md_as_project_rules(tmp_path: Path) -> None:
    import asyncio

    (tmp_path / "AGENTS.md").write_text("Use ruff for linting.", encoding="utf-8")
    loader = CodexResourceLoader(str(tmp_path), str(tmp_path / "agent"), None)
    asyncio.run(loader.reload())
    # System prompt should NOT contain AGENTS.md (it's a separate message in Codex mode)
    prompt = loader.build_system_prompt(None, [])
    assert "Personality" in prompt
    assert "Use ruff for linting." not in prompt
    # AGENTS.md should be in the separate messages
    agents_msgs = loader.get_agents_md_messages()
    assert len(agents_msgs) == 1
    assert "Use ruff for linting." in agents_msgs[0]["content"]


# ---------------------------------------------------------------------------
# Session creation test
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_codex_agent_session_creation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")

    codex_loader = CodexResourceLoader(str(tmp_path), str(tmp_path / "agent"), None)
    codex_tools = create_codex_tools(str(tmp_path))
    codex_loader.set_tool_names([t.name for t in codex_tools])

    result = await create_agent_session(
        CreateAgentSessionOptions(
            cwd=str(tmp_path),
            resource_loader=codex_loader,
            tools=codex_tools,
        )
    )
    session = result.session

    assert "Personality" in session.agent.state.system_prompt
    assert len(session.agent.state.tools) == 5
    tool_names = {t.name for t in session.agent.state.tools}
    assert tool_names == {"shell", "apply_patch", "update_plan", "view_image", "request_user_input"}


# ---------------------------------------------------------------------------
# Compaction test
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_codex_compaction_enabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")

    from kitpaw.pi_agent.code_agent.codex.compaction import configure_codex_compaction

    result = await create_agent_session(
        CreateAgentSessionOptions(
            cwd=str(tmp_path),
            resource_loader=CodexResourceLoader(str(tmp_path), str(tmp_path / "agent"), None),
            tools=create_codex_tools(str(tmp_path)),
        )
    )
    configure_codex_compaction(result.session)
    state = result.session.get_compaction_state()
    # Codex keeps auto-compact enabled (unlike Zed)
    assert state["enabled"] is True


# ---------------------------------------------------------------------------
# CLI integration test
# ---------------------------------------------------------------------------

def test_codex_agent_cli_flag(tmp_path: Path) -> None:
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH")
    root = str(Path(__file__).resolve().parent.parent)
    env["PYTHONPATH"] = root if not pythonpath else f"{root}:{pythonpath}"
    env["PI_CODING_AGENT_DIR"] = str(tmp_path / "agent")
    env["OPENAI_API_KEY"] = "test-key"
    env["OPENAI_MODEL"] = "gpt-4o-mini"

    process = subprocess.run(
        [sys.executable, "-m", "kitpaw.pi_agent.code_agent", "--agent", "codex", "--no-session"],
        cwd=root,
        env=env,
        input="/quit\n",
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert process.returncode == 0, process.stderr or process.stdout
