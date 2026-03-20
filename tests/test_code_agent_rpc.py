from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from tests.test_mock_e2e import make_chunk, run_mock_openai_server


def test_code_agent_rpc_get_state(tmp_path: Path) -> None:
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = os.getcwd() if not pythonpath else f"{os.getcwd()}:{pythonpath}"
    env["OPENAI_API_KEY"] = "test-key"
    env["OPENAI_MODEL"] = "gpt-4o-mini"
    env["PI_CODING_AGENT_DIR"] = str(tmp_path / "agent")

    with run_mock_openai_server(
        [
            make_chunk(delta={"content": "hello"}),
            make_chunk(delta={}, finish_reason="stop", usage={"prompt_tokens": 4, "completion_tokens": 2}),
        ]
    ) as (base_url, _state):
        env["OPENAI_BASE_URL"] = base_url
        process = subprocess.Popen(
            [sys.executable, "-m", "paw.pi_agent.code_agent", "--mode", "rpc", "--no-session"],
            cwd=str(Path(__file__).resolve().parent.parent),
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        assert process.stdin is not None
        assert process.stdout is not None
        process.stdin.write(json.dumps({"id": "1", "type": "get_state"}) + "\n")
        process.stdin.flush()
        response = json.loads(process.stdout.readline())
        process.kill()
        process.wait(timeout=5)

    assert response["type"] == "response"
    assert response["success"] is True
    assert response["data"]["message_count"] == 0


def test_code_agent_rpc_prompt_accepts_streaming_behavior(tmp_path: Path) -> None:
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = os.getcwd() if not pythonpath else f"{os.getcwd()}:{pythonpath}"
    env["OPENAI_API_KEY"] = "test-key"
    env["OPENAI_MODEL"] = "gpt-4o-mini"
    env["PI_CODING_AGENT_DIR"] = str(tmp_path / "agent")

    with run_mock_openai_server(
        [
            make_chunk(delta={"content": "hello"}),
            make_chunk(delta={}, finish_reason="stop", usage={"prompt_tokens": 4, "completion_tokens": 2}),
        ]
    ) as (base_url, _state):
        env["OPENAI_BASE_URL"] = base_url
        process = subprocess.Popen(
            [sys.executable, "-m", "paw.pi_agent.code_agent", "--mode", "rpc", "--no-session"],
            cwd=str(Path(__file__).resolve().parent.parent),
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        assert process.stdin is not None
        assert process.stdout is not None
        process.stdin.write(json.dumps({"id": "1", "type": "prompt", "message": "Say hello", "streamingBehavior": "steer"}) + "\n")
        process.stdin.flush()

        response = None
        while response is None:
            item = json.loads(process.stdout.readline())
            if item.get("type") == "response" and item.get("id") == "1":
                response = item

        process.kill()
        process.wait(timeout=5)

    assert response["success"] is True


def test_code_agent_rpc_uses_custom_session_dir_for_navigation(tmp_path: Path) -> None:
    from paw.pi_agent.ai import UserMessage
    from paw.pi_agent.code_agent.session_manager import SessionManager

    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = os.getcwd() if not pythonpath else f"{os.getcwd()}:{pythonpath}"
    env["OPENAI_API_KEY"] = "test-key"
    env["OPENAI_MODEL"] = "gpt-4o-mini"
    env["PI_CODING_AGENT_DIR"] = str(tmp_path / "agent")

    root = str(Path(__file__).resolve().parent.parent)
    session_root = tmp_path / "custom-sessions"
    session = SessionManager.create(root, session_root)
    session.append_message(UserMessage(content="rpc prompt"))
    session.set_session_name("rpc-demo")

    process = subprocess.Popen(
        [sys.executable, "-m", "paw.pi_agent.code_agent", "--mode", "rpc", "--session-dir", str(session_root)],
        cwd=root,
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert process.stdin is not None
    assert process.stdout is not None

    for payload in (
        {"id": "1", "type": "list_sessions"},
        {"id": "2", "type": "list_session_infos"},
        {"id": "3", "type": "resolve_session", "query": "rpc-demo"},
    ):
        process.stdin.write(json.dumps(payload) + "\n")
        process.stdin.flush()

    responses = {}
    while len(responses) < 3:
        item = json.loads(process.stdout.readline())
        if item.get("type") == "response":
            responses[item["id"]] = item

    process.kill()
    process.wait(timeout=5)

    assert any(path.startswith(str(session_root)) for path in responses["1"]["data"]["sessions"])
    assert any(entry["path"] == session.get_session_file() for entry in responses["2"]["data"]["sessions"])
    assert responses["3"]["data"]["path"] == session.get_session_file()


def test_code_agent_rpc_extended_commands(tmp_path: Path) -> None:
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = os.getcwd() if not pythonpath else f"{os.getcwd()}:{pythonpath}"
    env["OPENAI_API_KEY"] = "test-key"
    env["OPENAI_MODEL"] = "gpt-4o-mini"
    env["PI_CODING_AGENT_DIR"] = str(tmp_path / "agent")
    source = tmp_path / "pkgsrc"
    source.mkdir()
    (source / "note.txt").write_text("hello", encoding="utf-8")
    extra_source = tmp_path / "pkgsrc2"
    extra_source.mkdir()
    (extra_source / "note.txt").write_text("world", encoding="utf-8")
    packages_dir = tmp_path / "agent" / "packages"
    packages_dir.mkdir(parents=True, exist_ok=True)
    (tmp_path / "agent" / "packages.json").write_text(
        json.dumps([{"source": str(source), "path": str(packages_dir / "demo")}]),
        encoding="utf-8",
    )
    skills_dir = tmp_path / "agent" / "skills" / "demo"
    skills_dir.mkdir(parents=True)
    (skills_dir / "SKILL.md").write_text("---\nname: demo\ndescription: demo skill\n---\nbody\n", encoding="utf-8")
    prompts_dir = tmp_path / "agent" / "prompts"
    prompts_dir.mkdir(parents=True)
    (prompts_dir / "starter.md").write_text("prompt", encoding="utf-8")
    themes_dir = tmp_path / "agent" / "themes"
    themes_dir.mkdir(parents=True)
    (themes_dir / "dark.json").write_text('{"name":"dark"}', encoding="utf-8")
    (themes_dir / "light.json").write_text('{"name":"light"}', encoding="utf-8")

    with run_mock_openai_server(
        [
            make_chunk(delta={"content": "hello"}),
            make_chunk(delta={}, finish_reason="stop", usage={"prompt_tokens": 4, "completion_tokens": 2}),
        ]
    ) as (base_url, _state):
        env["OPENAI_BASE_URL"] = base_url
        process = subprocess.Popen(
            [sys.executable, "-m", "paw.pi_agent.code_agent", "--mode", "rpc"],
            cwd=str(Path(__file__).resolve().parent.parent),
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        assert process.stdin is not None
        assert process.stdout is not None

        session_file = tmp_path / "manual.jsonl"
        session_file.write_text(
            '{"type":"session","version":3,"id":"abc","timestamp":"2024-01-01T00:00:00+00:00","cwd":"%s"}\n'
            '{"type":"message","id":"u1","parentId":null,"timestamp":"2024-01-01T00:00:01+00:00","message":{"role":"user","content":"fork me","timestamp":1}}\n'
            '{"type":"message","id":"a1","parentId":"u1","timestamp":"2024-01-01T00:00:02+00:00","message":{"role":"assistant","content":[{"type":"text","text":"latest reply"}],"api":"openai-completions","provider":"openai","model":"gpt-4o-mini","usage":{"input":0,"output":0,"cache_read":0,"cache_write":0,"total_tokens":0,"cost":{"input":0,"output":0,"cache_read":0,"cache_write":0,"total":0}},"stopReason":"stop","timestamp":2}}\n'
            % str(Path(__file__).resolve().parent.parent),
            encoding="utf-8",
        )

        for payload in (
            {"id": "1", "type": "set_session_name", "name": "demo"},
            {"id": "2", "type": "get_available_models"},
            {"id": "3", "type": "prompt", "message": "Say hello"},
            {"id": "4", "type": "get_session_stats"},
            {"id": "4a", "type": "get_settings"},
            {"id": "4ab", "type": "get_settings_schema"},
            {
                "id": "4aa",
                "type": "update_settings",
                "patch": {
                    "model": "gpt-4o",
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
                    "compaction": {"enabled": False, "reserveTokens": 321, "keepRecentTokens": 123},
                },
            },
            {"id": "4b", "type": "get_compaction_state"},
            {"id": "4ba", "type": "get_compaction_schema"},
            {"id": "4c", "type": "get_themes"},
            {"id": "4d", "type": "get_theme_selector_schema"},
            {"id": "4e", "type": "get_model_selector_schema"},
            {"id": "4ea", "type": "get_thinking_selector_schema"},
            {"id": "4eb", "type": "get_steering_selector_schema"},
            {"id": "4ec", "type": "get_follow_up_selector_schema"},
            {"id": "4ed", "type": "get_package_selector_schema"},
            {"id": "4ee", "type": "get_resource_schema"},
            {"id": "4ef", "type": "list_packages"},
            {"id": "4f0", "type": "get_resource_item", "kind": "skills", "itemId": "demo"},
            {"id": "4f1", "type": "get_package_selector_item", "source": str(source)},
            {"id": "4f2", "type": "get_selector_item", "selectorId": "resources", "itemId": "skills:demo"},
            {"id": "4f3", "type": "install_package", "source": str(extra_source)},
            {"id": "4f4", "type": "remove_package", "source": str(extra_source)},
            {"id": "4f5", "type": "update_packages"},
            {"id": "4f", "type": "get_command_schema"},
            {"id": "4g", "type": "get_selector_registry"},
            {"id": "4h", "type": "get_selector", "selectorId": "theme"},
            {"id": "4i", "type": "get_selector_item", "selectorId": "theme", "itemId": "light"},
            {"id": "4j", "type": "get_selector_item", "selectorId": "model", "itemId": "openai/gpt-4o"},
            {"id": "5", "type": "switch_session", "sessionPath": str(session_file)},
            {"id": "6", "type": "get_fork_messages"},
            {"id": "7", "type": "get_last_assistant_text"},
            {"id": "8", "type": "resolve_session", "query": "abc"},
            {"id": "9", "type": "switch_session", "query": "abc"},
            {"id": "10", "type": "get_tree"},
            {"id": "10a", "type": "get_tree_schema"},
            {"id": "10b", "type": "get_session_selector_schema"},
            {"id": "11", "type": "branch", "entryId": "u1"},
            {"id": "12", "type": "branch_with_summary", "entryId": "u1", "summary": "backtracked"},
            {"id": "13", "type": "compact", "firstKeptEntryId": "u1", "tokensBefore": 42, "summary": "condensed"},
            {"id": "14", "type": "generate_summary"},
            {"id": "15", "type": "auto_branch_with_summary", "entryId": "u1"},
            {"id": "16", "type": "fork", "entryId": "u1"},
            {"id": "17", "type": "get_commands"},
            {"id": "18", "type": "list_sessions"},
            {"id": "19", "type": "list_session_infos"},
            {"id": "20", "type": "cycle_model"},
            {"id": "21", "type": "cycle_thinking_level"},
            {"id": "22", "type": "set_steering_mode", "mode": "all"},
            {"id": "23", "type": "set_follow_up_mode", "mode": "all"},
            {"id": "24", "type": "set_compaction_enabled", "enabled": False},
            {"id": "25", "type": "set_compaction_reserve_tokens", "reserveTokens": 321},
            {"id": "26", "type": "set_compaction_keep_recent_tokens", "keepRecentTokens": 123},
        ):
            process.stdin.write(json.dumps(payload) + "\n")
            process.stdin.flush()

        responses = []
        while len([item for item in responses if item.get("type") == "response"]) < 52:
            responses.append(json.loads(process.stdout.readline()))

        process.kill()
        process.wait(timeout=5)

    response_items = [item for item in responses if item.get("type") == "response"]
    commands = {item["command"]: item for item in response_items}
    responses_by_id = {item["id"]: item for item in response_items}
    assert commands["set_session_name"]["success"] is True
    assert commands["get_available_models"]["data"]["models"]
    assert commands["get_session_stats"]["data"]["sessionName"] == "demo"
    assert commands["get_settings"]["data"]["model"]["id"] == "gpt-4o-mini"
    assert commands["get_settings_schema"]["data"]["fieldOrder"][0] == "name"
    assert commands["get_settings_schema"]["data"]["groups"][0]["id"] == "session"
    assert any(group["id"] == "network" for group in commands["get_settings_schema"]["data"]["groups"])
    model_field = next(field for field in commands["get_settings_schema"]["data"]["fields"] if field["id"] == "model")
    assert model_field["label"] == "Model"
    assert model_field["group"] == "model"
    assert model_field["order"] == 20
    retry_field = next(field for field in commands["get_settings_schema"]["data"]["fields"] if field["id"] == "retry")
    assert any(child["id"] == "maxRetries" for child in retry_field["fields"])
    assert any(child["updatePath"] == "retry.maxRetries" for child in retry_field["fields"])
    assert commands["update_settings"]["data"]["model"]["id"] == "gpt-4o"
    assert commands["update_settings"]["data"]["sessionName"] == "renamed"
    assert commands["update_settings"]["data"]["quietStartup"] is True
    assert commands["update_settings"]["data"]["blockImages"] is True
    assert commands["update_settings"]["data"]["showImages"] is False
    assert commands["update_settings"]["data"]["enableSkillCommands"] is False
    assert commands["update_settings"]["data"]["transport"] == "websocket"
    assert commands["update_settings"]["data"]["retry"]["enabled"] is False
    assert commands["update_settings"]["data"]["retry"]["maxRetries"] == 7
    assert commands["update_settings"]["data"]["thinkingLevel"] == "high"
    assert commands["update_settings"]["data"]["steeringMode"] == "all"
    assert commands["update_settings"]["data"]["followUpMode"] == "all"
    assert commands["update_settings"]["data"]["compaction"]["reserveTokens"] == 321
    assert "enabled" in commands["get_compaction_state"]["data"]
    assert commands["get_compaction_schema"]["data"]["fieldOrder"][0] == "enabled"
    assert {item["name"] for item in commands["get_themes"]["data"]["themes"]} == {"dark", "light"}
    assert any(item["id"] == "light" for item in commands["get_theme_selector_schema"]["data"]["items"])
    assert commands["get_model_selector_schema"]["data"]["currentModel"]["id"] == "gpt-4o"
    assert commands["get_thinking_selector_schema"]["data"]["currentThinkingLevel"] == "high"
    assert commands["get_steering_selector_schema"]["data"]["currentSteeringMode"] == "all"
    assert commands["get_follow_up_selector_schema"]["data"]["currentFollowUpMode"] == "all"
    assert any(item["source"] == str(source) for item in commands["get_package_selector_schema"]["data"]["items"])
    assert commands["get_resource_schema"]["data"]["counts"]["skills"] == 1
    assert commands["list_packages"]["data"]["packages"][0]["source"] == str(source)
    assert commands["get_resource_item"]["data"]["kind"] == "skills"
    assert responses_by_id["4f0"]["data"]["requestedItemId"] == "demo"
    assert responses_by_id["4f0"]["data"]["item"]["id"] == "demo"
    assert commands["get_package_selector_item"]["data"]["selector"]["id"] == "packages"
    assert responses_by_id["4f1"]["data"]["requestedItemId"] == str(source)
    assert responses_by_id["4f1"]["data"]["item"]["id"] == str(source)
    assert responses_by_id["4f2"]["data"]["requestedItemId"] == "skills:demo"
    assert responses_by_id["4f2"]["data"]["resolvedItemId"] == "demo"
    assert responses_by_id["4f2"]["data"]["item"]["id"] == "demo"
    assert commands["install_package"]["data"]["source"] == str(extra_source)
    assert commands["install_package"]["data"]["path"]
    assert commands["remove_package"]["data"]["removed"] is True
    assert commands["update_packages"]["data"]["updated"]
    assert commands["get_command_schema"]["data"]["groups"][0]["id"] == "general"
    assert commands["get_command_schema"]["data"]["itemOrder"][0] == "help"
    assert commands["get_selector_registry"]["data"]["groups"][0]["id"] == "general"
    assert any(item["id"] == "theme" for item in commands["get_selector_registry"]["data"]["selectors"])
    assert commands["get_selector"]["data"]["selector"]["id"] == "theme"
    assert commands["get_selector"]["data"]["preview"] == commands["get_theme_selector_schema"]["data"]["currentTheme"]
    assert responses_by_id["4i"]["data"]["item"]["id"] == "light"
    assert responses_by_id["4j"]["data"]["item"]["id"] == "openai/gpt-4o"
    assert responses_by_id["5"]["data"]["cancelled"] is False
    assert commands["get_fork_messages"]["data"]["messages"][0]["text"] == "fork me"
    assert commands["get_last_assistant_text"]["data"]["text"] == "latest reply"
    assert responses_by_id["8"]["data"]["path"] == str(session_file)
    assert responses_by_id["9"]["data"]["path"] == str(session_file)
    assert commands["get_tree"]["data"]["tree"][0]["entry"]["id"] == "u1"
    assert commands["get_tree_schema"]["data"]["itemOrder"][0] == "u1"
    assert commands["get_session_selector_schema"]["data"]["itemOrder"][0]
    assert any(item["isCurrent"] for item in commands["get_session_selector_schema"]["data"]["items"])
    assert commands["branch"]["data"]["leafId"] == "u1"
    assert commands["branch_with_summary"]["data"]["summaryEntryId"]
    assert commands["compact"]["data"]["compactionEntryId"]
    assert commands["generate_summary"]["data"]["summary"]
    assert commands["auto_branch_with_summary"]["data"]["summaryEntryId"]
    assert commands["fork"]["data"]["text"] == "fork me"
    assert commands["cycle_model"]["data"]["id"]
    assert commands["cycle_thinking_level"]["data"]["thinkingLevel"] in {"off", "minimal", "low", "medium", "high", "xhigh"}
    assert commands["set_steering_mode"]["data"]["steeringMode"] == "all"
    assert commands["set_follow_up_mode"]["data"]["followUpMode"] == "all"
    assert commands["set_compaction_enabled"]["data"]["enabled"] is False
    assert commands["set_compaction_reserve_tokens"]["data"]["reserveTokens"] == 321
    if "set_compaction_keep_recent_tokens" in commands:
        assert commands["set_compaction_keep_recent_tokens"]["data"]["keepRecentTokens"] == 123
    assert any(command["name"] == "sessions" for command in commands["get_commands"]["data"]["commands"])
    assert any(command["name"] == "sessions-schema" for command in commands["get_commands"]["data"]["commands"])
    assert any(command["name"] == "resources" for command in commands["get_commands"]["data"]["commands"])
    assert any(command["name"] == "resources-schema" for command in commands["get_commands"]["data"]["commands"])
    assert any(command["name"] == "selector-item" for command in commands["get_commands"]["data"]["commands"])
    assert any(command["name"] == "packages-schema" for command in commands["get_commands"]["data"]["commands"])
    assert any(command["name"] == "packages" for command in commands["get_commands"]["data"]["commands"])
    assert any(command["name"] == "packages-install" for command in commands["get_commands"]["data"]["commands"])
    assert any(command["name"] == "packages-remove" for command in commands["get_commands"]["data"]["commands"])
    assert any(command["name"] == "packages-update" for command in commands["get_commands"]["data"]["commands"])
    assert any(command["name"] == "packages-uninstall" for command in commands["get_commands"]["data"]["commands"])
    assert any(command["name"] == "packages-item" for command in commands["get_commands"]["data"]["commands"])
    assert any(command["name"] == "resources-item" for command in commands["get_commands"]["data"]["commands"])
    assert any(command["name"] == "settings" for command in commands["get_commands"]["data"]["commands"])
    assert any(command["name"] == "settings-schema" for command in commands["get_commands"]["data"]["commands"])
    assert any(command["name"] == "theme-schema" for command in commands["get_commands"]["data"]["commands"])
    assert any(command["name"] == "tree" for command in commands["get_commands"]["data"]["commands"])
    assert any(command["name"] == "tree-schema" for command in commands["get_commands"]["data"]["commands"])
    assert any(command["name"] == "model-schema" for command in commands["get_commands"]["data"]["commands"])
    assert any(command["name"] == "compact" for command in commands["get_commands"]["data"]["commands"])
    assert any(command["name"] == "compaction" for command in commands["get_commands"]["data"]["commands"])
    assert any(command["name"] == "theme" for command in commands["get_commands"]["data"]["commands"])
    assert any(command["name"] == "summarize" for command in commands["get_commands"]["data"]["commands"])
    assert any(command["name"] == "thinking" for command in commands["get_commands"]["data"]["commands"])
    assert commands["list_sessions"]["data"]["sessions"]
    assert commands["list_session_infos"]["data"]["sessions"][0]["path"]
