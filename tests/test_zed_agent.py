from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from kitpaw.pi_agent.code_agent.sdk import CreateAgentSessionOptions, create_agent_session
from kitpaw.pi_agent.code_agent.tool_error_limit import (
    configure_tool_error_limit,
    consume_tool_error_limit_exception,
)
from kitpaw.pi_agent.code_agent.zed.resource_loader import ZedResourceLoader
from kitpaw.pi_agent.code_agent.zed.system_prompt import build_zed_system_prompt
from kitpaw.pi_agent.code_agent.zed.tools import create_zed_tools

# ---------------------------------------------------------------------------
# Mock OpenAI helpers (copied from test_mock_e2e for isolation)
# ---------------------------------------------------------------------------

def _make_chunk(*, delta: dict, finish_reason: str | None = None, usage: dict | None = None) -> dict:
    chunk: dict = {
        "id": "chatcmpl-mock",
        "object": "chat.completion.chunk",
        "created": 1_710_000_000,
        "model": "mock-model",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    if usage is not None:
        chunk["usage"] = usage
    return chunk


@contextmanager
def _run_mock_openai_server(chunks: list[dict]):
    state: dict[str, object] = {"request": None}

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(length).decode("utf-8")
            state["request"] = json.loads(raw_body)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            for chunk in chunks:
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode("utf-8"))
                self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

        def log_message(self, format: str, *args) -> None:  # noqa: A003
            return None

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/v1", state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@contextmanager
def _run_mock_openai_server_sequences(responses: list[list[dict]]):
    state: dict[str, object] = {"requests": []}
    remaining = [list(chunks) for chunks in responses]
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(length).decode("utf-8")
            request = json.loads(raw_body)
            with lock:
                state["requests"].append(request)
                chunks = remaining.pop(0) if remaining else []

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            for chunk in chunks:
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode("utf-8"))
                self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

        def log_message(self, format: str, *args) -> None:  # noqa: A003
            return None

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/v1", state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


# ---------------------------------------------------------------------------
# System prompt tests
# ---------------------------------------------------------------------------

def test_zed_system_prompt_includes_core_sections() -> None:
    prompt = build_zed_system_prompt(
        available_tools=["grep", "terminal", "read_file", "update_plan", "spawn_agent"],
        worktrees=["/tmp/myproject"],
    )
    assert "highly skilled software engineer" in prompt
    assert "## Communication" in prompt
    assert "## Tool Use" in prompt
    assert "## Planning" in prompt
    assert "## Searching and Reading" in prompt
    assert "## Code Block Formatting" in prompt
    assert "## Multi-agent delegation" in prompt


def test_zed_system_prompt_conditional_sections() -> None:
    # No tools → no Tool Use section
    prompt_no_tools = build_zed_system_prompt(available_tools=[], worktrees=["/tmp"])
    assert "## Tool Use" not in prompt_no_tools
    assert "no ability to use tools" in prompt_no_tools

    # No update_plan → no Planning section
    prompt_no_plan = build_zed_system_prompt(available_tools=["grep"], worktrees=["/tmp"])
    assert "## Planning" not in prompt_no_plan

    # No spawn_agent → no Multi-agent section
    assert "## Multi-agent" not in prompt_no_plan


# ---------------------------------------------------------------------------
# Tool list tests
# ---------------------------------------------------------------------------

def test_zed_tools_count_and_names() -> None:
    tools = create_zed_tools("/tmp")
    names = {t.name for t in tools}
    # Default: all CLI-supported tools including web_search, diagnostics, spawn_agent
    assert len(tools) == 17
    expected = {
        "read_file", "edit_file", "terminal", "grep", "find_path",
        "list_directory", "copy_path", "move_path", "delete_path",
        "create_directory", "fetch", "open", "now", "update_plan",
        "web_search", "diagnostics", "spawn_agent",
    }
    assert names == expected


def test_zed_tools_all_enabled() -> None:
    """Pass enabled=all to get the full set including editor-only stubs."""
    from kitpaw.pi_agent.code_agent.zed.tools import ALL_TOOL_NAMES

    tools = create_zed_tools("/tmp", enabled=ALL_TOOL_NAMES)
    names = {t.name for t in tools}
    assert len(tools) == 19
    assert "save_file" in names
    assert "restore_file_from_disk" in names


def test_zed_tool_schemas_have_required_fields() -> None:
    tools = create_zed_tools("/tmp")
    for tool in tools:
        assert tool.name, "Tool missing name"
        assert tool.label, f"Tool {tool.name} missing label"
        assert tool.description, f"Tool {tool.name} missing description"
        assert isinstance(tool.parameters, dict), f"Tool {tool.name} parameters not a dict"
        assert tool.parameters.get("type") == "object", f"Tool {tool.name} parameters type not 'object'"


# ---------------------------------------------------------------------------
# Tool execution tests
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_read_file_tool(tmp_path: Path) -> None:
    (tmp_path / "hello.txt").write_text("line1\nline2\nline3\n", encoding="utf-8")
    tools = create_zed_tools(str(tmp_path))
    read_tool = next(t for t in tools if t.name == "read_file")
    result = await read_tool.execute("call1", {"path": "hello.txt"}, None, None)
    text = result.content[0].text
    assert "line1" in text
    assert "line2" in text


@pytest.mark.anyio
async def test_read_file_tool_with_worktree_prefix(tmp_path: Path) -> None:
    """Model may prefix paths with the worktree root name (e.g. 'mydir/hello.txt')."""
    (tmp_path / "hello.txt").write_text("content\n", encoding="utf-8")
    tools = create_zed_tools(str(tmp_path))
    read_tool = next(t for t in tools if t.name == "read_file")
    # Use the worktree root name as prefix — should still resolve correctly
    worktree_name = tmp_path.name
    result = await read_tool.execute("call1", {"path": f"{worktree_name}/hello.txt"}, None, None)
    assert "content" in result.content[0].text


@pytest.mark.anyio
async def test_path_traversal_rejected(tmp_path: Path) -> None:
    tools = create_zed_tools(str(tmp_path))
    read_tool = next(t for t in tools if t.name == "read_file")
    result = await read_tool.execute("call1", {"path": "../../../etc/passwd"}, None, None)
    assert "Error" in result.content[0].text
    tools = create_zed_tools(str(tmp_path))
    terminal_tool = next(t for t in tools if t.name == "terminal")
    result = await terminal_tool.execute("call1", {"command": "echo hello", "cd": str(tmp_path)}, None, None)
    assert "hello" in result.content[0].text


@pytest.mark.anyio
async def test_edit_file_write_mode(tmp_path: Path) -> None:
    tools = create_zed_tools(str(tmp_path))
    edit_tool = next(t for t in tools if t.name == "edit_file")
    result = await edit_tool.execute("call1", {
        "display_description": "Create test file",
        "path": "test.txt",
        "mode": "write",
        "content": "hello world\n",
    }, None, None)
    assert (tmp_path / "test.txt").read_text() == "hello world\n"
    assert "test.txt" in result.content[0].text or "Wrote" in result.content[0].text


@pytest.mark.anyio
async def test_edit_file_edit_mode(tmp_path: Path) -> None:
    (tmp_path / "test.txt").write_text("hello world\n", encoding="utf-8")
    tools = create_zed_tools(str(tmp_path))
    edit_tool = next(t for t in tools if t.name == "edit_file")
    await edit_tool.execute("call1", {
        "display_description": "Replace hello with goodbye",
        "path": "test.txt",
        "mode": "edit",
        "edits": [{"old_text": "hello", "new_text": "goodbye"}],
    }, None, None)
    assert (tmp_path / "test.txt").read_text() == "goodbye world\n"


@pytest.mark.anyio
async def test_grep_tool(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("def hello():\n    pass\n", encoding="utf-8")
    tools = create_zed_tools(str(tmp_path))
    grep_tool = next(t for t in tools if t.name == "grep")
    result = await grep_tool.execute("call1", {"regex": "def hello"}, None, None)
    assert "a.py" in result.content[0].text


@pytest.mark.anyio
async def test_list_directory_tool(tmp_path: Path) -> None:
    (tmp_path / "file.txt").touch()
    (tmp_path / "subdir").mkdir()
    tools = create_zed_tools(str(tmp_path))
    ls_tool = next(t for t in tools if t.name == "list_directory")
    result = await ls_tool.execute("call1", {"path": "."}, None, None)
    text = result.content[0].text
    assert "file.txt" in text
    assert "subdir" in text


@pytest.mark.anyio
async def test_copy_move_delete_tools(tmp_path: Path) -> None:
    (tmp_path / "src.txt").write_text("content", encoding="utf-8")
    tools = create_zed_tools(str(tmp_path))

    copy_tool = next(t for t in tools if t.name == "copy_path")
    await copy_tool.execute("c1", {"source_path": "src.txt", "destination_path": "copy.txt"}, None, None)
    assert (tmp_path / "copy.txt").read_text() == "content"

    move_tool = next(t for t in tools if t.name == "move_path")
    await move_tool.execute("c2", {"source_path": "copy.txt", "destination_path": "moved.txt"}, None, None)
    assert (tmp_path / "moved.txt").exists()
    assert not (tmp_path / "copy.txt").exists()

    delete_tool = next(t for t in tools if t.name == "delete_path")
    await delete_tool.execute("c3", {"path": "moved.txt"}, None, None)
    assert not (tmp_path / "moved.txt").exists()


@pytest.mark.anyio
async def test_create_directory_tool(tmp_path: Path) -> None:
    tools = create_zed_tools(str(tmp_path))
    mkdir_tool = next(t for t in tools if t.name == "create_directory")
    await mkdir_tool.execute("c1", {"path": "a/b/c"}, None, None)
    assert (tmp_path / "a" / "b" / "c").is_dir()


@pytest.mark.anyio
async def test_now_tool() -> None:
    tools = create_zed_tools("/tmp")
    now_tool = next(t for t in tools if t.name == "now")
    result = await now_tool.execute("c1", {"timezone": "utc"}, None, None)
    assert "UTC" in result.content[0].text or "20" in result.content[0].text


@pytest.mark.anyio
async def test_diagnostics_tool_python_project(tmp_path: Path) -> None:
    """diagnostics should detect pyproject.toml and attempt to run a linter."""
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test'\n", encoding="utf-8")
    (tmp_path / "bad.py").write_text("def f(\n", encoding="utf-8")  # syntax error
    tools = create_zed_tools(str(tmp_path))
    diag_tool = next(t for t in tools if t.name == "diagnostics")
    result = await diag_tool.execute("c1", {}, None, None)
    text = result.content[0].text
    # Should either show ruff output or a fallback message (ruff may not be installed)
    assert text  # non-empty response


@pytest.mark.anyio
async def test_diagnostics_tool_no_project(tmp_path: Path) -> None:
    """diagnostics in a directory with no project files gives a helpful message."""
    tools = create_zed_tools(str(tmp_path))
    diag_tool = next(t for t in tools if t.name == "diagnostics")
    result = await diag_tool.execute("c1", {}, None, None)
    text = result.content[0].text
    assert "terminal" in text.lower() or "no" in text.lower() or "not" in text.lower()


@pytest.mark.anyio
async def test_spawn_agent_tool_no_parent() -> None:
    """spawn_agent without parent_agent should return an error."""
    tools = create_zed_tools("/tmp")
    spawn_tool = next(t for t in tools if t.name == "spawn_agent")
    result = await spawn_tool.execute("c1", {"label": "test", "message": "hello"}, None, None)
    text = result.content[0].text
    # Without parent_agent, should error gracefully
    assert "error" in text.lower() or "session_id" in text.lower()
# ---------------------------------------------------------------------------

def test_zed_resource_loader_build_system_prompt() -> None:
    loader = ZedResourceLoader("/tmp", "/tmp/agent", None)
    prompt = loader.build_system_prompt(None, [])
    assert "highly skilled software engineer" in prompt


def test_zed_resource_loader_agents_md_as_project_rules(tmp_path: Path) -> None:
    """AGENTS.md should be injected as project rules, not replace the system prompt."""
    (tmp_path / "AGENTS.md").write_text("Always use type hints.", encoding="utf-8")
    loader = ZedResourceLoader(str(tmp_path), str(tmp_path / "agent"), None)
    import asyncio
    asyncio.run(loader.reload())
    prompt = loader.build_system_prompt(None, [])
    # Main Zed prompt is still present
    assert "highly skilled software engineer" in prompt
    # AGENTS.md content is injected under User's Custom Instructions
    assert "## User's Custom Instructions" in prompt
    assert "Always use type hints." in prompt


# ---------------------------------------------------------------------------
# Session integration test with mock server
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_zed_agent_session_creation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")

    zed_loader = ZedResourceLoader(str(tmp_path), str(tmp_path / "agent"), None)
    zed_tools = create_zed_tools(str(tmp_path))

    result = await create_agent_session(
        CreateAgentSessionOptions(
            cwd=str(tmp_path),
            resource_loader=zed_loader,
            tools=zed_tools,
        )
    )
    session = result.session

    # System prompt should be Zed-style
    assert "highly skilled software engineer" in session.agent.state.system_prompt

    # Should have 17 CLI-supported tools
    assert len(session.agent.state.tools) == 17

    # Tool names should match Zed CLI set
    tool_names = {t.name for t in session.agent.state.tools}
    assert "terminal" in tool_names
    assert "read_file" in tool_names
    assert "edit_file" in tool_names
    assert "spawn_agent" in tool_names
    assert "web_search" in tool_names
    assert "diagnostics" in tool_names


@pytest.mark.anyio
async def test_zed_agent_compaction_disabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")

    from kitpaw.pi_agent.code_agent.zed.compaction import configure_zed_compaction

    result = await create_agent_session(
        CreateAgentSessionOptions(
            cwd=str(tmp_path),
            resource_loader=ZedResourceLoader(str(tmp_path), str(tmp_path / "agent"), None),
            tools=create_zed_tools(str(tmp_path)),
        )
    )
    configure_zed_compaction(result.session)
    state = result.session.get_compaction_state()
    assert state["enabled"] is False


@pytest.mark.anyio
async def test_zed_spawn_agent_child_tool_errors_count_toward_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")

    parent_chunks = [
        _make_chunk(
            delta={
                "tool_calls": [
                    {
                        "id": "call_spawn",
                        "type": "function",
                        "function": {
                            "name": "spawn_agent",
                            "arguments": '{"label":"worker","message":"trigger child tool error"}',
                        },
                    }
                ]
            },
            finish_reason="tool_calls",
            usage={"prompt_tokens": 4, "completion_tokens": 2},
        )
    ]
    child_chunks = [
        _make_chunk(
            delta={
                "tool_calls": [
                    {
                        "id": "call_run",
                        "type": "function",
                        "function": {"name": "run", "arguments": "{}"},
                    }
                ]
            },
            finish_reason="tool_calls",
            usage={"prompt_tokens": 4, "completion_tokens": 2},
        )
    ]

    with _run_mock_openai_server_sequences([parent_chunks, child_chunks]) as (base_url, _state):
        monkeypatch.setenv("OPENAI_BASE_URL", base_url)
        loader = ZedResourceLoader(str(tmp_path), str(tmp_path / "agent"), None)
        tools = create_zed_tools(str(tmp_path))
        result = await create_agent_session(
            CreateAgentSessionOptions(
                cwd=str(tmp_path),
                resource_loader=loader,
                tools=tools,
            )
        )
        session = result.session
        session.agent.set_tools(create_zed_tools(str(tmp_path), parent_agent=session.agent))
        configure_tool_error_limit(session, 1)

        await session.prompt("spawn a helper")

    limit_error = consume_tool_error_limit_exception(session)
    assert limit_error is not None
    assert limit_error.failures == 1
    assert limit_error.tool_name == "run"
    assert len(_state["requests"]) == 2


# ---------------------------------------------------------------------------
# CLI integration test
# ---------------------------------------------------------------------------

def test_zed_agent_cli_flag(tmp_path: Path) -> None:
    """Verify --agent zed starts without error and uses Zed tools."""
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH")
    root = str(Path(__file__).resolve().parent.parent)
    env["PYTHONPATH"] = root if not pythonpath else f"{root}:{pythonpath}"
    env["PI_CODING_AGENT_DIR"] = str(tmp_path / "agent")
    env["OPENAI_API_KEY"] = "test-key"
    env["OPENAI_MODEL"] = "gpt-4o-mini"

    # Use /quit immediately — just verify the process starts and exits cleanly.
    process = subprocess.run(
        [sys.executable, "-m", "kitpaw.pi_agent.code_agent", "--agent", "zed", "--no-session"],
        cwd=root,
        env=env,
        input="/quit\n",
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert process.returncode == 0, process.stderr or process.stdout
