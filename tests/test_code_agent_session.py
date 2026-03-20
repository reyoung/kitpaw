from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from paw.pi_agent.ai import UserMessage, get_model
from paw.pi_agent.code_agent.package_manager import PackageManager
from paw.pi_agent.code_agent.sdk import CreateAgentSessionOptions, create_agent_session
from paw.pi_agent.code_agent.session_manager import SessionManager
from tests.test_mock_e2e import make_chunk, run_mock_openai_server


@pytest.mark.anyio
async def test_create_agent_session_persists_messages(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")

    with run_mock_openai_server(
        [
            make_chunk(delta={"content": "pong"}),
            make_chunk(delta={}, finish_reason="stop", usage={"prompt_tokens": 4, "completion_tokens": 2}),
        ]
    ) as (base_url, _state):
        model = replace(get_model("openai", "gpt-4o-mini"), base_url=base_url)
        session_manager = SessionManager.create(str(tmp_path))
        result = await create_agent_session(
            CreateAgentSessionOptions(cwd=str(tmp_path), model=model, session_manager=session_manager)
        )

        await result.session.prompt("Reply with pong")

    session_file = result.session.session_file
    assert session_file is not None
    text = Path(session_file).read_text(encoding="utf-8")
    assert '"type": "session"' in text
    assert '"role": "user"' in text
    assert '"role": "assistant"' in text


@pytest.mark.anyio
async def test_print_mode_outputs_response(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")

    with run_mock_openai_server(
        [
            make_chunk(delta={"content": "hello"}),
            make_chunk(delta={}, finish_reason="stop", usage={"prompt_tokens": 4, "completion_tokens": 2}),
        ]
    ) as (base_url, _state):
        model = replace(get_model("openai", "gpt-4o-mini"), base_url=base_url)
        result = await create_agent_session(
            CreateAgentSessionOptions(cwd=str(tmp_path), model=model, session_manager=SessionManager.in_memory(str(tmp_path)))
        )
        from paw.pi_agent.code_agent.modes.print_mode import run_print_mode

        exit_code = await run_print_mode(result.session, "Say hello")

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out.strip() == "hello"


def test_cli_continue_uses_latest_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    session_root = tmp_path / "agent" / "sessions"
    older = SessionManager.create(str(tmp_path), session_root)
    older.append_message(UserMessage(content="first"))
    newer = SessionManager.create(str(tmp_path), session_root)
    newer.append_message(UserMessage(content="second"))

    found = SessionManager.find_most_recent_session(str(tmp_path), session_root)
    assert found is not None

    reopened = SessionManager.open(found)
    assert reopened.entries[-1]["message"]["content"] in {"first", "second"}


def test_session_manager_fork_and_assistant_text(tmp_path: Path) -> None:
    manager = SessionManager.create(str(tmp_path))
    first = manager.append_message(UserMessage(content="first"))
    manager.append_message(
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "reply one"}],
            "api": "openai-completions",
            "provider": "openai",
            "model": "gpt-4o-mini",
            "usage": {},
            "stopReason": "stop",
            "timestamp": 1,
        }
    )
    manager.append_message(UserMessage(content="second"))
    assert manager.get_last_assistant_text() == "reply one"
    forked, selected = manager.fork_to_new_manager(first["id"])
    assert selected == "first"
    assert len(forked.entries) == 0


def test_session_manager_tree_and_branch(tmp_path: Path) -> None:
    manager = SessionManager.create(str(tmp_path))
    first = manager.append_message(UserMessage(content="first"))
    manager.append_message(UserMessage(content="second"))
    manager.branch(first["id"])
    branch_user = manager.append_message(UserMessage(content="branch"))

    tree = manager.get_tree()
    assert tree[0]["entry"]["id"] == first["id"]
    assert len(tree[0]["children"]) == 2
    assert manager.get_leaf_id() == branch_user["id"]
    assert [message["content"] for message in manager.get_branch_messages()] == ["first", "branch"]


@pytest.mark.anyio
async def test_agent_session_tree_schema(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")

    manager = SessionManager.create(str(tmp_path))
    first = manager.append_message(UserMessage(content="first"))
    manager.append_message(UserMessage(content="second"))
    manager.branch(first["id"])
    branch_leaf = manager.append_message(UserMessage(content="branch"))

    result = await create_agent_session(CreateAgentSessionOptions(cwd=str(tmp_path), session_manager=manager))
    schema = result.session.get_tree_schema()

    assert schema["currentLeafId"] == branch_leaf["id"]
    assert schema["itemOrder"][0] == first["id"]
    first_item = next(item for item in schema["items"] if item["id"] == first["id"])
    branch_item = next(item for item in schema["items"] if item["id"] == branch_leaf["id"])
    assert first_item["depth"] == 0
    assert first_item["childCount"] == 2
    assert first_item["isOnCurrentBranch"] is True
    assert branch_item["depth"] == 1
    assert branch_item["isLeaf"] is True


@pytest.mark.anyio
async def test_agent_session_session_selector_schema(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(tmp_path / "agent"))

    session_root = tmp_path / "agent" / "sessions"
    current = SessionManager.create(str(tmp_path), session_root)
    current.append_message(UserMessage(content="current prompt"))
    current.set_session_name("current")
    other = SessionManager.create(str(tmp_path), session_root)
    other.append_message(UserMessage(content="other prompt"))
    other.set_session_name("other")

    result = await create_agent_session(CreateAgentSessionOptions(cwd=str(tmp_path), session_manager=current))
    schema = result.session.get_session_selector_schema()

    assert schema["currentSessionFile"] == current.get_session_file()
    assert Path(other.get_session_file()) in {Path(path) for path in schema["itemOrder"]}
    current_item = next(item for item in schema["items"] if item["path"] == current.get_session_file())
    assert current_item["isCurrent"] is True
    assert current_item["label"] == "current"
    assert current_item["description"] == "current prompt"


@pytest.mark.anyio
async def test_agent_session_model_and_theme_selector_schema(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(tmp_path / "agent"))

    themes_dir = tmp_path / "agent" / "themes"
    themes_dir.mkdir(parents=True)
    (themes_dir / "dark.json").write_text('{"name":"dark"}', encoding="utf-8")
    (themes_dir / "light.json").write_text('{"name":"light"}', encoding="utf-8")

    result = await create_agent_session(CreateAgentSessionOptions(cwd=str(tmp_path), session_manager=SessionManager.in_memory(str(tmp_path))))
    theme_schema = result.session.get_theme_selector_schema()
    model_schema = result.session.get_model_selector_schema()

    assert theme_schema["currentTheme"] == "dark"
    assert any(item["id"] == "light" for item in theme_schema["items"])
    assert any(item["isCurrent"] for item in theme_schema["items"])
    assert model_schema["currentModel"]["id"] == "gpt-4o-mini"
    assert any(item["id"] == "openai/gpt-4o-mini" for item in model_schema["items"])
    assert any(item["isCurrent"] for item in model_schema["items"])


@pytest.mark.anyio
async def test_agent_session_command_schema(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")

    result = await create_agent_session(
        CreateAgentSessionOptions(cwd=str(tmp_path), session_manager=SessionManager.in_memory(str(tmp_path)))
    )
    schema = result.session.get_command_schema()

    assert schema["groups"][0]["id"] == "general"
    assert schema["itemOrder"][0] == "help"
    selector_item = next(item for item in schema["commands"] if item["name"] == "selector-item")
    assert selector_item["group"] == "resources"
    assert selector_item["usage"] == "/selector-item SELECTOR_ID ITEM_ID"
    theme_schema = next(item for item in schema["commands"] if item["name"] == "theme-schema")
    assert theme_schema["group"] == "appearance"
    assert theme_schema["usage"] == "/theme schema"
    resources_schema = next(item for item in schema["commands"] if item["name"] == "resources-schema")
    assert resources_schema["group"] == "resources"
    assert resources_schema["usage"] == "/resources schema"


def test_session_manager_resolve_session_matches_entire_message_text(tmp_path: Path) -> None:
    session_root = tmp_path / "agent" / "sessions"
    manager = SessionManager.create(str(tmp_path), session_root)
    manager.append_message(UserMessage(content="first prompt"))
    manager.append_message(
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "tail match text"}],
            "api": "openai-completions",
            "provider": "openai",
            "model": "gpt-4o-mini",
            "usage": {},
            "stopReason": "stop",
            "timestamp": 1,
        }
    )

    resolved = SessionManager.resolve_session(str(tmp_path), "tail match", session_root)

    assert resolved == Path(manager.get_session_file()).resolve()


def test_session_manager_search_session_infos_matches_name_and_message_text(tmp_path: Path) -> None:
    session_root = tmp_path / "agent" / "sessions"
    first = SessionManager.create(str(tmp_path), session_root)
    first.append_message(UserMessage(content="alpha prompt"))
    first.set_session_name("beta-one")
    second = SessionManager.create(str(tmp_path), session_root)
    second.append_message(UserMessage(content="beta tail match"))
    second.set_session_name("beta-two")

    matches = SessionManager.search_session_infos(str(tmp_path), "beta", session_root)

    assert {Path(info.path) for info in matches} == {
        Path(first.get_session_file()).resolve(),
        Path(second.get_session_file()).resolve(),
    }


def test_session_manager_search_session_infos_supports_regex_and_phrases(tmp_path: Path) -> None:
    session_root = tmp_path / "agent" / "sessions"
    regex_session = SessionManager.create(str(tmp_path), session_root)
    regex_session.append_message(UserMessage(content="alpha one"))
    regex_session.append_message(UserMessage(content="beta two"))
    phrase_session = SessionManager.create(str(tmp_path), session_root)
    phrase_session.append_message(UserMessage(content="node cve now"))

    regex_matches = SessionManager.search_session_infos(str(tmp_path), "re:beta\\s+two", session_root)
    phrase_matches = SessionManager.search_session_infos(str(tmp_path), '"node cve"', session_root)

    assert [Path(info.path) for info in regex_matches] == [Path(regex_session.get_session_file()).resolve()]
    assert [Path(info.path) for info in phrase_matches] == [Path(phrase_session.get_session_file()).resolve()]


def test_session_manager_list_all_session_infos_collects_all_directories(tmp_path: Path) -> None:
    current_root = tmp_path / "agent" / "sessions" / "---cwd-a--"
    other_root = tmp_path / "agent" / "sessions" / "---cwd-b--"
    current_root.mkdir(parents=True)
    other_root.mkdir(parents=True)
    first = SessionManager.create("cwd-a", tmp_path / "agent" / "sessions")
    first.append_message(UserMessage(content="alpha"))
    second = SessionManager.create("cwd-b", tmp_path / "agent" / "sessions")
    second.append_message(UserMessage(content="beta"))

    infos = SessionManager.list_all_session_infos(tmp_path / "agent" / "sessions")

    assert {Path(info.path) for info in infos} == {
        Path(first.get_session_file()).resolve(),
        Path(second.get_session_file()).resolve(),
    }


@pytest.mark.anyio
async def test_agent_session_thinking_and_compaction_schema(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")

    result = await create_agent_session(
        CreateAgentSessionOptions(cwd=str(tmp_path), session_manager=SessionManager.in_memory(str(tmp_path)))
    )
    thinking_schema = result.session.get_thinking_selector_schema()
    compaction_schema = result.session.get_compaction_schema()

    assert thinking_schema["currentThinkingLevel"] == "medium"
    assert any(item["id"] == "medium" and item["isCurrent"] for item in thinking_schema["items"])
    assert compaction_schema["fieldOrder"][0] == "enabled"
    reserve_field = next(field for field in compaction_schema["fields"] if field["id"] == "reserveTokens")
    assert reserve_field["command"] == "/compaction reserve <n>"


@pytest.mark.anyio
async def test_agent_session_queue_mode_selector_schema(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")

    result = await create_agent_session(
        CreateAgentSessionOptions(cwd=str(tmp_path), session_manager=SessionManager.in_memory(str(tmp_path)))
    )
    result.session.set_steering_mode("all")
    result.session.set_follow_up_mode("all")

    steering_schema = result.session.get_steering_selector_schema()
    follow_up_schema = result.session.get_follow_up_selector_schema()

    assert steering_schema["currentSteeringMode"] == "all"
    assert any(item["id"] == "all" and item["isCurrent"] for item in steering_schema["items"])
    assert follow_up_schema["currentFollowUpMode"] == "all"
    assert any(item["id"] == "all" and item["isCurrent"] for item in follow_up_schema["items"])


@pytest.mark.anyio
async def test_agent_session_package_and_resource_schema(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(tmp_path / "agent"))

    source = tmp_path / "pkgsrc"
    source.mkdir()
    (source / "note.txt").write_text("hello", encoding="utf-8")
    skills_dir = tmp_path / "agent" / "skills" / "demo"
    skills_dir.mkdir(parents=True)
    (skills_dir / "SKILL.md").write_text("---\nname: demo\ndescription: demo skill\n---\nbody\n", encoding="utf-8")
    prompts_dir = tmp_path / "agent" / "prompts"
    prompts_dir.mkdir(parents=True)
    (prompts_dir / "starter.md").write_text("prompt", encoding="utf-8")
    themes_dir = tmp_path / "agent" / "themes"
    themes_dir.mkdir(parents=True)
    (themes_dir / "light.json").write_text('{"name":"light"}', encoding="utf-8")

    result = await create_agent_session(
        CreateAgentSessionOptions(cwd=str(tmp_path), session_manager=SessionManager.in_memory(str(tmp_path)))
    )
    PackageManager(str(tmp_path), str(tmp_path / "agent"), None).install(str(source))
    await result.session.resource_loader.reload()

    package_schema = result.session.get_package_selector_schema()
    resource_schema = result.session.get_resource_schema()

    assert any(item["source"] == str(source) for item in package_schema["items"])
    assert resource_schema["counts"]["skills"] == 1
    assert resource_schema["counts"]["prompts"] == 1
    assert resource_schema["counts"]["themes"] == 1
    resource_item = result.session.get_resource_item("skills", "demo")
    assert resource_item["kind"] == "skills"
    assert resource_item["requestedItemId"] == "demo"
    assert resource_item["resolvedItemId"] == "demo"
    assert resource_item["item"]["id"] == "demo"
    package_item = result.session.get_package_selector_item(str(source))
    assert package_item["selector"]["id"] == "packages"
    assert package_item["requestedItemId"] == str(source)
    assert package_item["resolvedItemId"] == str(source)
    assert package_item["item"]["id"] == str(source)
    resource_selector_item = result.session.get_selector_item("resources", "skills:demo")
    assert resource_selector_item["selector"]["id"] == "resources"
    assert resource_selector_item["requestedItemId"] == "skills:demo"
    assert resource_selector_item["resolvedItemId"] == "demo"
    assert resource_selector_item["item"]["id"] == "demo"

    extra = tmp_path / "pkgsrc2"
    extra.mkdir()
    (extra / "note.txt").write_text("world", encoding="utf-8")
    installed = result.session.install_package(str(extra))
    assert installed["source"] == str(extra)
    assert installed["scope"] == "user"
    assert any(item["source"] == str(extra) for item in result.session.list_packages())
    updated = result.session.update_packages(str(extra))
    assert str(extra) in updated["updated"]
    removed = result.session.remove_package(str(extra))
    assert removed["removed"] is True
    assert not any(item["source"] == str(extra) for item in result.session.list_packages())


@pytest.mark.anyio
async def test_agent_session_selector_registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")

    result = await create_agent_session(
        CreateAgentSessionOptions(cwd=str(tmp_path), session_manager=SessionManager.in_memory(str(tmp_path)))
    )
    registry = result.session.get_selector_registry()

    assert registry["groups"][0]["id"] == "general"
    assert "settings" in registry["itemOrder"]
    assert "theme" in registry["itemOrder"]
    selectors = {item["id"]: item for item in registry["selectors"]}
    assert selectors["commands"]["getter"] == "get_command_schema"
    assert selectors["theme"]["currentKey"] == "currentTheme"
    assert selectors["packages"]["getter"] == "get_package_selector_schema"
    selector = result.session.get_selector("theme")
    assert selector["selector"]["id"] == "theme"
    assert selector["preview"] == "dark"
    assert "itemOrder" in selector["data"]
    item = result.session.get_selector_item("model", "openai/gpt-4o-mini")
    assert item["item"]["id"] == "openai/gpt-4o-mini"


def test_session_manager_build_runtime_context_with_branch_summary_and_compaction(tmp_path: Path) -> None:
    manager = SessionManager.create(str(tmp_path))
    first = manager.append_message(UserMessage(content="first"))
    manager.append_message(UserMessage(content="second"))
    manager.branch_with_summary(first["id"], "branch summary")
    branch_leaf = manager.append_message(UserMessage(content="after branch"))
    manager.append_compaction("compact summary", first["id"], 123)
    manager.append_message(UserMessage(content="after compact"))

    branch_context = manager.build_runtime_context(branch_leaf["id"])
    assert branch_context["messages"][1]["role"] == "branchSummary"
    assert branch_context["messages"][1]["summary"] == "branch summary"

    compact_context = manager.build_runtime_context()
    assert compact_context["messages"][0]["role"] == "compactionSummary"
    assert compact_context["messages"][0]["summary"] == "compact summary"
    assert compact_context["messages"][-1]["content"] == "after compact"


def test_session_manager_persists_session_name_and_counts(tmp_path: Path) -> None:
    manager = SessionManager.create(str(tmp_path))
    first = manager.append_message(UserMessage(content="first"))
    manager.branch_with_summary(first["id"], "branch summary")
    manager.append_compaction("compact summary", first["id"], 12)
    manager.set_session_name("demo")

    reopened = SessionManager.open(manager.get_session_file())
    stats = reopened.get_stats()
    assert reopened.get_session_name() == "demo"
    assert stats["branchSummaries"] == 1
    assert stats["compactions"] == 1


def test_session_manager_lists_session_infos(tmp_path: Path) -> None:
    root = str(tmp_path)
    manager = SessionManager.create(root, tmp_path / "agent" / "sessions")
    manager.append_message(UserMessage(content="first prompt"))
    manager.set_session_name("demo")

    infos = SessionManager.list_session_infos(root, tmp_path / "agent" / "sessions")
    assert len(infos) == 1
    assert infos[0].name == "demo"
    assert infos[0].message_count == 1
    assert infos[0].first_message == "first prompt"


def test_session_manager_resolves_session_queries(tmp_path: Path) -> None:
    root = str(tmp_path)
    session_root = tmp_path / "agent" / "sessions"
    first = SessionManager.create(root, session_root)
    first.append_message(UserMessage(content="alpha prompt"))
    first.set_session_name("alpha")
    second = SessionManager.create(root, session_root)
    second.append_message(UserMessage(content="beta prompt"))
    second.set_session_name("beta")

    resolved_by_name = SessionManager.resolve_session(root, "beta", session_root)
    resolved_by_id = SessionManager.resolve_session(root, second.get_session_id()[:6], session_root)
    resolved_by_message = SessionManager.resolve_session(root, "beta prompt", session_root)

    assert resolved_by_name == Path(second.get_session_file())
    assert resolved_by_id == Path(second.get_session_file())
    assert resolved_by_message == Path(second.get_session_file())


@pytest.mark.anyio
async def test_agent_session_generate_summary_and_auto_compact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")

    manager = SessionManager.create(str(tmp_path))
    first = manager.append_message(UserMessage(content="first"))
    manager.append_message(UserMessage(content="second"))

    with run_mock_openai_server(
        [
            make_chunk(delta={"content": "summarized context"}),
            make_chunk(delta={}, finish_reason="stop", usage={"prompt_tokens": 10, "completion_tokens": 4}),
        ]
    ) as (base_url, _state):
        model = replace(get_model("openai", "gpt-4o-mini"), base_url=base_url)
        result = await create_agent_session(CreateAgentSessionOptions(cwd=str(tmp_path), model=model, session_manager=manager))
        summary = await result.session.generate_summary()
        compacted = await result.session.auto_compact(first["id"])

    assert summary["summary"] == "summarized context"
    assert compacted["compactionEntryId"]
    assert result.session.messages[0]["role"] == "compactionSummary"


@pytest.mark.anyio
async def test_agent_session_auto_compacts_after_prompt_when_threshold_exceeded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")

    settings_dir = tmp_path / "agent"
    settings_dir.mkdir(parents=True, exist_ok=True)
    (tmp_path / ".pi").mkdir(parents=True, exist_ok=True)
    (tmp_path / ".pi" / "settings.json").write_text(
        '{"compaction":{"enabled":true,"reserveTokens":10,"keepRecentTokens":8}}',
        encoding="utf-8",
    )

    with run_mock_openai_server(
        [
            make_chunk(delta={"content": "assistant reply that is long enough to trigger compaction"}),
            make_chunk(delta={}, finish_reason="stop", usage={"prompt_tokens": 20, "completion_tokens": 12}),
            make_chunk(delta={"content": "auto summary"}),
            make_chunk(delta={}, finish_reason="stop", usage={"prompt_tokens": 10, "completion_tokens": 4}),
        ]
    ) as (base_url, _state):
        session_manager = SessionManager.create(str(tmp_path))
        session_manager.append_message(UserMessage(content="existing conversation history that should be compacted"))
        session_manager.append_message(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "existing assistant history"}],
                "api": "openai-completions",
                "provider": "openai",
                "model": "gpt-4o-mini",
                "usage": {},
                "stopReason": "stop",
                "timestamp": 1,
            }
        )
        model = replace(get_model("openai", "gpt-4o-mini"), base_url=base_url, context_window=20)
        result = await create_agent_session(
            CreateAgentSessionOptions(cwd=str(tmp_path), agent_dir=str(settings_dir), model=model, session_manager=session_manager)
        )
        await result.session.prompt("user prompt that is also long enough to push the context over the edge")

    assert isinstance(result.session.messages[0], dict)
    assert result.session.messages[0]["role"] == "compactionSummary"
    assert result.session.get_session_stats()["compactions"] == 1


@pytest.mark.anyio
async def test_agent_session_compaction_state_and_updates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")

    result = await create_agent_session(
        CreateAgentSessionOptions(cwd=str(tmp_path), agent_dir=str(tmp_path / "agent"), session_manager=SessionManager.in_memory(str(tmp_path)))
    )

    initial = result.session.get_compaction_state()
    assert initial["enabled"] is True

    disabled = result.session.set_compaction_enabled(False)
    reserved = result.session.set_compaction_reserve_tokens(123)
    kept = result.session.set_compaction_keep_recent_tokens(45)

    assert disabled["enabled"] is False
    assert reserved["reserveTokens"] == 123
    assert kept["keepRecentTokens"] == 45


@pytest.mark.anyio
async def test_agent_session_theme_state_and_updates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(tmp_path / "agent"))
    themes_dir = tmp_path / "agent" / "themes"
    themes_dir.mkdir(parents=True)
    (themes_dir / "light.json").write_text('{"name":"light"}', encoding="utf-8")
    (themes_dir / "dark.json").write_text('{"name":"dark"}', encoding="utf-8")

    result = await create_agent_session(
        CreateAgentSessionOptions(cwd=str(tmp_path), agent_dir=str(tmp_path / "agent"), session_manager=SessionManager.in_memory(str(tmp_path)))
    )
    await result.session.resource_loader.reload()

    initial = result.session.get_themes()
    updated = result.session.set_theme("light")

    assert initial["currentTheme"] == "dark"
    assert {item["name"] for item in initial["themes"]} == {"dark", "light"}
    assert updated["currentTheme"] == "light"
    assert '"theme": "light"' in (tmp_path / ".pi" / "settings.json").read_text(encoding="utf-8")


@pytest.mark.anyio
async def test_agent_session_settings_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")

    result = await create_agent_session(
        CreateAgentSessionOptions(cwd=str(tmp_path), agent_dir=str(tmp_path / "agent"), session_manager=SessionManager.in_memory(str(tmp_path)))
    )
    result.session.set_session_name("demo")
    result.session.set_steering_mode("all")
    result.session.set_follow_up_mode("all")
    snapshot = result.session.get_settings_snapshot()

    assert snapshot["theme"]
    assert snapshot["model"]["id"] == "gpt-4o-mini"
    assert snapshot["thinkingLevel"] == "medium"
    assert snapshot["steeringMode"] == "all"
    assert snapshot["followUpMode"] == "all"
    assert snapshot["quietStartup"] is False
    assert snapshot["blockImages"] is False
    assert snapshot["showImages"] is True
    assert snapshot["enableSkillCommands"] is True
    assert snapshot["transport"] == "sse"
    assert snapshot["retry"]["enabled"] is True
    assert snapshot["sessionName"] == "demo"
    assert "compaction" in snapshot


@pytest.mark.anyio
async def test_agent_session_settings_schema(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(tmp_path / "agent"))
    themes_dir = tmp_path / "agent" / "themes"
    themes_dir.mkdir(parents=True)
    (themes_dir / "light.json").write_text('{"name":"light"}', encoding="utf-8")

    result = await create_agent_session(
        CreateAgentSessionOptions(cwd=str(tmp_path), agent_dir=str(tmp_path / "agent"), session_manager=SessionManager.in_memory(str(tmp_path)))
    )
    await result.session.resource_loader.reload()
    schema = result.session.get_settings_schema()

    assert schema["fieldOrder"][0] == "name"
    assert schema["groups"][0]["id"] == "session"
    assert any(group["id"] == "network" for group in schema["groups"])
    fields = {field["id"]: field for field in schema["fields"]}
    assert fields["model"]["type"] == "select"
    assert any(option["id"] == "gpt-4o-mini" for option in fields["model"]["options"])
    assert fields["model"]["label"] == "Model"
    assert fields["model"]["group"] == "model"
    assert fields["model"]["order"] == 20
    assert fields["model"]["updatePath"] == "model"
    assert fields["theme"]["type"] == "select"
    assert any(option["value"] == "light" for option in fields["theme"]["options"])
    assert fields["theme"]["description"] == "Color theme for the interface"
    assert fields["quiet"]["type"] == "boolean"
    assert fields["retry"]["type"] == "object"
    assert any(child["id"] == "maxRetries" for child in fields["retry"]["fields"])
    assert any(child["updatePath"] == "retry.maxRetries" for child in fields["retry"]["fields"])
    assert fields["compaction"]["type"] == "object"
    assert any(child["id"] == "shouldCompact" and child["readonly"] is True for child in fields["compaction"]["fields"])


@pytest.mark.anyio
async def test_agent_session_update_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(tmp_path / "agent"))
    themes_dir = tmp_path / "agent" / "themes"
    themes_dir.mkdir(parents=True)
    (themes_dir / "light.json").write_text('{"name":"light"}', encoding="utf-8")

    result = await create_agent_session(
        CreateAgentSessionOptions(cwd=str(tmp_path), agent_dir=str(tmp_path / "agent"), session_manager=SessionManager.in_memory(str(tmp_path)))
    )
    await result.session.resource_loader.reload()

    updated = result.session.update_settings(
        {
            "model": "gpt-4o",
            "theme": "light",
            "sessionName": "renamed",
            "quietStartup": True,
            "blockImages": True,
            "showImages": False,
            "enableSkillCommands": False,
            "transport": "websocket",
            "retry": {"enabled": False, "maxRetries": 7, "baseDelayMs": 1234, "maxDelayMs": 9876},
            "thinkingLevel": "high",
            "steeringMode": "all",
            "followUpMode": "all",
            "compaction": {"enabled": False, "reserveTokens": 111, "keepRecentTokens": 22},
        }
    )

    assert updated["model"]["id"] == "gpt-4o"
    assert updated["theme"] == "light"
    assert updated["sessionName"] == "renamed"
    assert updated["quietStartup"] is True
    assert updated["blockImages"] is True
    assert updated["showImages"] is False
    assert updated["enableSkillCommands"] is False
    assert updated["transport"] == "websocket"
    assert updated["retry"]["enabled"] is False
    assert updated["retry"]["maxRetries"] == 7
    assert updated["retry"]["baseDelayMs"] == 1234
    assert updated["retry"]["maxDelayMs"] == 9876
    assert updated["thinkingLevel"] == "high"
    assert updated["steeringMode"] == "all"
    assert updated["followUpMode"] == "all"
    assert updated["compaction"]["enabled"] is False
    assert updated["compaction"]["reserveTokens"] == 111
    assert updated["compaction"]["keepRecentTokens"] == 22


@pytest.mark.anyio
async def test_create_agent_session_restores_compaction_and_branch_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")

    manager = SessionManager.create(str(tmp_path))
    first = manager.append_message(UserMessage(content="first"))
    manager.branch_with_summary(first["id"], "branch summary")
    manager.append_message(UserMessage(content="after branch"))
    manager.append_compaction("compact summary", first["id"], 99)
    manager.append_message(UserMessage(content="after compact"))

    result = await create_agent_session(CreateAgentSessionOptions(cwd=str(tmp_path), session_manager=manager))

    restored_roles = [message.get("role") if isinstance(message, dict) else getattr(message, "role", None) for message in result.session.messages]
    assert restored_roles == ["compactionSummary", "user", "branchSummary", "user", "user"]


@pytest.mark.anyio
async def test_agent_session_cycles_and_queue_modes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("OPENAI_FALLBACK_MODEL", "gpt-4o")

    result = await create_agent_session(
        CreateAgentSessionOptions(cwd=str(tmp_path), agent_dir=str(tmp_path / "agent"), session_manager=SessionManager.in_memory(str(tmp_path)))
    )

    assert result.session.get_state().steering_mode == "one-at-a-time"
    assert result.session.get_state().follow_up_mode == "one-at-a-time"

    await result.session.cycle_model()
    assert result.session.model.id == "gpt-4o"

    result.session.set_thinking_level("off")
    assert result.session.cycle_thinking_level() == "minimal"
    assert result.session.set_steering_mode("all") == "all"
    assert result.session.set_follow_up_mode("all") == "all"
    state = result.session.get_state()
    assert state.steering_mode == "all"
    assert state.follow_up_mode == "all"

    settings_path = tmp_path / ".pi" / "settings.json"
    assert settings_path.exists()
    settings_text = settings_path.read_text(encoding="utf-8")
    assert '"steeringMode": "all"' in settings_text
    assert '"followUpMode": "all"' in settings_text
