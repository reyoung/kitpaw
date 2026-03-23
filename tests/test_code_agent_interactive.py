from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_code_agent_interactive_session_commands(tmp_path: Path) -> None:
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH")
    root = str(Path(__file__).resolve().parent.parent)
    env["PYTHONPATH"] = root if not pythonpath else f"{root}:{pythonpath}"
    env["PI_CODING_AGENT_DIR"] = str(tmp_path / "agent")
    env["OPENAI_API_KEY"] = "test-key"
    env["OPENAI_MODEL"] = "gpt-4o-mini"
    env["OPENAI_FALLBACK_MODEL"] = "gpt-4o"

    process = subprocess.run(
        [sys.executable, "-m", "kitpaw.pi_agent.code_agent", "--no-session"],
        cwd=root,
        env=env,
        input="/help\n/help schema\n/selectors\n/selector theme\n/selector-item model openai/gpt-4o-mini\n/name demo\n/session\n/settings\n/settings schema\n/model schema\n/model\n/cycle-model\n/thinking schema\n/thinking\n/thinking cycle\n/steering schema\n/steering all\n/followup schema\n/followup all\n/new\n/quit\n",
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode == 0, process.stderr or process.stdout
    # Human-friendly /help output
    assert "  General" in process.stdout
    assert "  Session" in process.stdout
    assert "/help" in process.stdout
    assert "Show interactive help" in process.stdout
    # Raw /help schema output
    assert "group general: label=General order=10" in process.stdout
    assert "theme-schema: group=appearance order=70 usage=/theme schema" in process.stdout
    assert "resources-schema: group=resources order=180 usage=/resources schema" in process.stdout
    assert "packages-schema: group=resources order=190 usage=/packages schema" in process.stdout
    assert "packages: group=resources order=195 usage=/packages" in process.stdout
    assert "packages-install: group=resources order=196 usage=/packages install SOURCE [local]" in process.stdout
    assert "packages-remove: group=resources order=197 usage=/packages remove SOURCE [local]" in process.stdout
    assert "packages-update: group=resources order=198 usage=/packages update [SOURCE]" in process.stdout
    assert "packages-uninstall: group=resources order=199 usage=/packages uninstall SOURCE [local]" in process.stdout
    assert "packages-item: group=resources order=200 usage=/packages item SOURCE" in process.stdout
    assert "resources-item: group=resources order=201 usage=/resources item KIND ID" in process.stdout
    assert "selector-item: group=resources order=25 usage=/selector-item SELECTOR_ID ITEM_ID" in process.stdout
    assert "commands: group=general kind=list getter=get_command_schema currentKey=None preview=" in process.stdout
    assert "theme: group=appearance kind=list getter=get_theme_selector_schema currentKey=currentTheme preview=" in process.stdout
    assert "'currentTheme':" in process.stdout
    assert "'itemOrder':" in process.stdout
    assert "model: itemId=openai/gpt-4o-mini" in process.stdout
    assert "session name: demo" in process.stdout
    assert "sessionName: demo" in process.stdout
    assert "steeringMode:" in process.stdout
    assert "followUpMode:" in process.stdout
    assert "group network: label=Network order=60" in process.stdout
    assert "model: order=20 label=Model group=model type=select" in process.stdout
    assert "retry: order=120 label=Retry policy group=network type=object" in process.stdout
    assert "  maxRetries: label=Max retries type=number updatePath=retry.maxRetries" in process.stdout
    assert "currentModel: openai/gpt-4o-mini" in process.stdout
    assert "openai/gpt-4o-mini: position=" in process.stdout
    assert "currentThinkingLevel: medium" in process.stdout
    assert "medium: position=" in process.stdout
    assert "currentSteeringMode:" in process.stdout
    assert "all: position=1" in process.stdout
    assert "description=Queue every steering message" in process.stdout
    assert "currentFollowUpMode:" in process.stdout
    assert "description=Queue every follow-up message" in process.stdout
    assert "openai/gpt-4o-mini" in process.stdout
    assert "model: openai/gpt-4o" in process.stdout
    assert "thinking: medium" in process.stdout
    assert "thinking: high" in process.stdout
    assert "steering: all" in process.stdout
    assert "follow-up: all" in process.stdout
    assert "new session" in process.stdout
    assert "model: itemId=openai/gpt-4o-mini resolvedItemId=openai/gpt-4o-mini" in process.stdout


def test_code_agent_interactive_compaction_commands(tmp_path: Path) -> None:
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH")
    root = str(Path(__file__).resolve().parent.parent)
    env["PYTHONPATH"] = root if not pythonpath else f"{root}:{pythonpath}"
    env["PI_CODING_AGENT_DIR"] = str(tmp_path / "agent")
    env["OPENAI_API_KEY"] = "test-key"
    env["OPENAI_MODEL"] = "gpt-4o-mini"

    process = subprocess.run(
        [sys.executable, "-m", "kitpaw.pi_agent.code_agent", "--no-session"],
        cwd=root,
        env=env,
        input="/compaction schema\n/compaction\n/compaction enabled false\n/compaction reserve 123\n/compaction keep 45\n/quit\n",
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode == 0, process.stderr or process.stdout
    assert "reserveTokens: order=20 label=Reserve tokens type=number current=" in process.stdout
    assert "enabled:" in process.stdout
    assert "thresholdTokens:" in process.stdout
    assert "compaction: enabled=False" in process.stdout
    assert "reserve=123" in process.stdout
    assert "keep=45" in process.stdout


def test_code_agent_interactive_theme_commands(tmp_path: Path) -> None:
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH")
    root = str(Path(__file__).resolve().parent.parent)
    env["PYTHONPATH"] = root if not pythonpath else f"{root}:{pythonpath}"
    env["PI_CODING_AGENT_DIR"] = str(tmp_path / "agent")
    env["OPENAI_API_KEY"] = "test-key"
    env["OPENAI_MODEL"] = "gpt-4o-mini"
    themes_dir = tmp_path / "agent" / "themes"
    themes_dir.mkdir(parents=True)
    (themes_dir / "dark.json").write_text('{"name":"dark"}', encoding="utf-8")
    (themes_dir / "light.json").write_text('{"name":"light"}', encoding="utf-8")

    process = subprocess.run(
        [sys.executable, "-m", "kitpaw.pi_agent.code_agent", "--no-session"],
        cwd=root,
        env=env,
        input="/theme schema\n/theme\n/theme light\n/quit\n",
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode == 0, process.stderr or process.stdout
    assert "currentTheme:" in process.stdout
    assert "dark: position=" in process.stdout
    assert "light: position=" in process.stdout
    assert "theme:" in process.stdout
    assert " dark " in process.stdout
    assert " light " in process.stdout
    assert "theme: light" in process.stdout


def test_code_agent_interactive_settings_update_commands(tmp_path: Path) -> None:
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH")
    root = str(Path(__file__).resolve().parent.parent)
    env["PYTHONPATH"] = root if not pythonpath else f"{root}:{pythonpath}"
    env["PI_CODING_AGENT_DIR"] = str(tmp_path / "agent")
    env["OPENAI_API_KEY"] = "test-key"
    env["OPENAI_MODEL"] = "gpt-4o-mini"
    themes_dir = tmp_path / "agent" / "themes"
    themes_dir.mkdir(parents=True)
    (themes_dir / "light.json").write_text('{"name":"light"}', encoding="utf-8")

    process = subprocess.run(
        [sys.executable, "-m", "kitpaw.pi_agent.code_agent", "--no-session"],
        cwd=root,
        env=env,
        input="/settings name renamed\n/settings model gpt-4o\n/settings theme light\n/settings quiet true\n/settings block-images true\n/settings show-images false\n/settings skill-commands false\n/settings transport websocket\n/settings retry enabled false\n/settings retry max-retries 7\n/settings retry base-delay-ms 1234\n/settings retry max-delay-ms 9876\n/settings thinking high\n/settings steering all\n/settings followup all\n/settings compaction enabled false\n/quit\n",
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode == 0, process.stderr or process.stdout
    assert "settings:" in process.stdout
    assert "'sessionName': 'renamed'" in process.stdout
    assert "'model': {'provider': 'openai', 'id': 'gpt-4o'}" in process.stdout
    assert "'theme': 'light'" in process.stdout
    assert "'quietStartup': True" in process.stdout
    assert "'blockImages': True" in process.stdout
    assert "'showImages': False" in process.stdout
    assert "'enableSkillCommands': False" in process.stdout
    assert "'transport': 'websocket'" in process.stdout
    assert "'retry': {'enabled': False, 'maxRetries': 7, 'baseDelayMs': 1234, 'maxDelayMs': 9876}" in process.stdout
    assert "'thinkingLevel': 'high'" in process.stdout
    assert "'steeringMode': 'all'" in process.stdout
    assert "'followUpMode': 'all'" in process.stdout
    assert "'enabled': False" in process.stdout


def test_code_agent_interactive_resource_and_package_schema(tmp_path: Path) -> None:
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH")
    root = str(Path(__file__).resolve().parent.parent)
    env["PYTHONPATH"] = root if not pythonpath else f"{root}:{pythonpath}"
    env["PI_CODING_AGENT_DIR"] = str(tmp_path / "agent")
    env["OPENAI_API_KEY"] = "test-key"
    env["OPENAI_MODEL"] = "gpt-4o-mini"

    source = tmp_path / "pkgsrc"
    source.mkdir()
    (source / "note.txt").write_text("hello", encoding="utf-8")
    extra = tmp_path / "pkgsrc2"
    extra.mkdir()
    (extra / "note.txt").write_text("world", encoding="utf-8")
    package_path = tmp_path / "agent" / "packages" / "demo"
    package_path.mkdir(parents=True)
    (tmp_path / "agent" / "packages.json").write_text(
        f'[{{"source": "{source}", "path": "{package_path}"}}]',
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
    (themes_dir / "light.json").write_text('{"name":"light"}', encoding="utf-8")

    process = subprocess.run(
        [sys.executable, "-m", "kitpaw.pi_agent.code_agent", "--no-session"],
        cwd=root,
        env=env,
        input=f"/resources\n/resources schema\n/resources item skills demo\n/selector-item resources skills:demo\n/packages schema\n/packages item {source}\n/packages\n/packages install {extra}\n/packages\n/packages remove {extra}\n/packages update\n/quit\n",
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode == 0, process.stderr or process.stdout
    assert "skills: 1" in process.stdout
    assert "prompts: 1" in process.stdout
    assert "themes: 1" in process.stdout
    assert "skill demo: source=global" in process.stdout or "skill demo: source=project" in process.stdout
    assert "skills: itemId=demo resolvedItemId=demo" in process.stdout
    assert "resources: itemId=skills:demo resolvedItemId=demo" in process.stdout
    assert f"{source}: scope=user path={package_path}" in process.stdout
    assert f"packages: itemId={source} resolvedItemId={source}" in process.stdout
    assert f"'source': '{source}'" in process.stdout or f'"source": "{source}"' in process.stdout
    assert f"user: {extra}" in process.stdout
    assert f"installed: {extra} ->" in process.stdout
    assert f"removed: {extra}" in process.stdout


def test_code_agent_interactive_resume_and_fork_listing(tmp_path: Path) -> None:
    from kitpaw.pi_agent.ai import UserMessage
    from kitpaw.pi_agent.code_agent.session_manager import SessionManager

    root = str(Path(__file__).resolve().parent.parent)
    session = SessionManager.create(root, tmp_path / "agent" / "sessions")
    session.append_message(UserMessage(content="fork me"))
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = root if not pythonpath else f"{root}:{pythonpath}"
    env["PI_CODING_AGENT_DIR"] = str(tmp_path / "agent")
    env["OPENAI_API_KEY"] = "test-key"
    env["OPENAI_MODEL"] = "gpt-4o-mini"

    process = subprocess.run(
        [sys.executable, "-m", "kitpaw.pi_agent.code_agent", "--no-session"],
        cwd=root,
        env=env,
        input="/resume\n/fork\n/quit\n",
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode == 0, process.stderr or process.stdout
    assert "resumed:" in process.stdout
    assert "fork me" in process.stdout


def test_code_agent_interactive_resume_and_switch_by_query(tmp_path: Path) -> None:
    from kitpaw.pi_agent.ai import UserMessage
    from kitpaw.pi_agent.code_agent.session_manager import SessionManager

    root = str(Path(__file__).resolve().parent.parent)
    first = SessionManager.create(root, tmp_path / "agent" / "sessions")
    first.append_message(UserMessage(content="alpha prompt"))
    first.set_session_name("alpha")
    second = SessionManager.create(root, tmp_path / "agent" / "sessions")
    second.append_message(UserMessage(content="beta prompt"))
    second.set_session_name("beta")

    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = root if not pythonpath else f"{root}:{pythonpath}"
    env["PI_CODING_AGENT_DIR"] = str(tmp_path / "agent")
    env["OPENAI_API_KEY"] = "test-key"
    env["OPENAI_MODEL"] = "gpt-4o-mini"

    process = subprocess.run(
        [sys.executable, "-m", "kitpaw.pi_agent.code_agent", "--no-session"],
        cwd=root,
        env=env,
        input="/resume beta\n/switch alpha\n/quit\n",
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode == 0, process.stderr or process.stdout
    assert f"resumed: {second.get_session_file()}" in process.stdout
    assert f"switched: {first.get_session_file()}" in process.stdout


def test_code_agent_interactive_fork_by_id_and_last(tmp_path: Path) -> None:
    from kitpaw.pi_agent.code_agent.session_manager import SessionManager

    root = str(Path(__file__).resolve().parent.parent)
    session = SessionManager.create(root, tmp_path / "agent" / "sessions")
    user_entry = session.append_message({"role": "user", "content": "fork target", "timestamp": 1})
    session.append_message(
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "assistant tail"}],
            "api": "openai-completions",
            "provider": "openai",
            "model": "gpt-4o-mini",
            "usage": {},
            "stopReason": "stop",
            "timestamp": 2,
        }
    )

    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = root if not pythonpath else f"{root}:{pythonpath}"
    env["PI_CODING_AGENT_DIR"] = str(tmp_path / "agent")
    env["OPENAI_API_KEY"] = "test-key"
    env["OPENAI_MODEL"] = "gpt-4o-mini"

    process = subprocess.run(
        [sys.executable, "-m", "kitpaw.pi_agent.code_agent"],
        cwd=root,
        env=env,
        input=f"/resume\n/last\n/fork {user_entry['id']}\n/quit\n",
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode == 0, process.stderr or process.stdout
    assert "assistant tail" in process.stdout
    assert "forked: fork target" in process.stdout


def test_code_agent_interactive_tree_and_branch(tmp_path: Path) -> None:
    from kitpaw.pi_agent.ai import UserMessage
    from kitpaw.pi_agent.code_agent.session_manager import SessionManager

    root = str(Path(__file__).resolve().parent.parent)
    session = SessionManager.create(root, tmp_path / "agent" / "sessions")
    first = session.append_message(UserMessage(content="first"))
    session.append_message(UserMessage(content="second"))

    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = root if not pythonpath else f"{root}:{pythonpath}"
    env["PI_CODING_AGENT_DIR"] = str(tmp_path / "agent")
    env["OPENAI_API_KEY"] = "test-key"
    env["OPENAI_MODEL"] = "gpt-4o-mini"

    process = subprocess.run(
        [sys.executable, "-m", "kitpaw.pi_agent.code_agent"],
        cwd=root,
        env=env,
        input=f"/resume\n/tree schema\n/tree\n/tree {first['id']}\n/tree\n/quit\n",
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode == 0, process.stderr or process.stdout
    assert "currentLeafId:" in process.stdout
    assert f"{first['id']}: parentId=None depth=0" in process.stdout
    assert f"{first['id']} user" in process.stdout
    assert f"branched: {first['id']}" in process.stdout


def test_code_agent_interactive_branch_summary_and_compact(tmp_path: Path) -> None:
    from kitpaw.pi_agent.ai import UserMessage
    from kitpaw.pi_agent.code_agent.session_manager import SessionManager

    root = str(Path(__file__).resolve().parent.parent)
    session = SessionManager.create(root, tmp_path / "agent" / "sessions")
    first = session.append_message(UserMessage(content="first"))
    session.append_message(UserMessage(content="second"))

    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = root if not pythonpath else f"{root}:{pythonpath}"
    env["PI_CODING_AGENT_DIR"] = str(tmp_path / "agent")
    env["OPENAI_API_KEY"] = "test-key"
    env["OPENAI_MODEL"] = "gpt-4o-mini"

    process = subprocess.run(
        [sys.executable, "-m", "kitpaw.pi_agent.code_agent"],
        cwd=root,
        env=env,
        input=f"/resume\n/branch-summary {first['id']} branch summary\n/compact {first['id']} 77 compact summary\n/quit\n",
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode == 0, process.stderr or process.stdout
    assert "branch summary:" in process.stdout
    assert "compaction:" in process.stdout


def test_code_agent_interactive_sessions_and_switch(tmp_path: Path) -> None:
    from kitpaw.pi_agent.ai import UserMessage
    from kitpaw.pi_agent.code_agent.session_manager import SessionManager

    root = str(Path(__file__).resolve().parent.parent)
    first = SessionManager.create(root, tmp_path / "agent" / "sessions")
    second = SessionManager.create(root, tmp_path / "agent" / "sessions")
    first.append_message(UserMessage(content="one"))
    second.append_message(UserMessage(content="two"))
    session_path = second.get_session_file()

    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = root if not pythonpath else f"{root}:{pythonpath}"
    env["PI_CODING_AGENT_DIR"] = str(tmp_path / "agent")
    env["OPENAI_API_KEY"] = "test-key"
    env["OPENAI_MODEL"] = "gpt-4o-mini"

    process = subprocess.run(
        [sys.executable, "-m", "kitpaw.pi_agent.code_agent", "--no-session"],
        cwd=root,
        env=env,
        input=f"/sessions schema\n/sessions\n/switch {session_path}\n/quit\n",
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode == 0, process.stderr or process.stdout
    assert "currentSessionFile:" in process.stdout
    assert "isCurrent=" in process.stdout
    assert first.get_session_file() in process.stdout
    assert second.get_session_file() in process.stdout
    assert "messages=" in process.stdout
    assert f"switched: {session_path}" in process.stdout
