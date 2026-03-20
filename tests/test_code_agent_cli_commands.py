from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _base_env(tmp_path: Path) -> dict[str, str]:
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH")
    root = str(Path(__file__).resolve().parent.parent)
    env["PYTHONPATH"] = root if not pythonpath else f"{root}:{pythonpath}"
    env["PI_CODING_AGENT_DIR"] = str(tmp_path / "agent")
    return env


def test_code_agent_package_install_list_remove(tmp_path: Path) -> None:
    source = tmp_path / "pkgsrc"
    source.mkdir()
    (source / "note.txt").write_text("hello", encoding="utf-8")
    env = _base_env(tmp_path)
    root = str(Path(__file__).resolve().parent.parent)

    install = subprocess.run(
        [sys.executable, "-m", "paw.pi_agent.code_agent", "install", str(source)],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert install.returncode == 0, install.stderr or install.stdout
    assert "Installed" in install.stdout

    listed = subprocess.run(
        [sys.executable, "-m", "paw.pi_agent.code_agent", "list"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert listed.returncode == 0
    assert str(source) in listed.stdout

    removed = subprocess.run(
        [sys.executable, "-m", "paw.pi_agent.code_agent", "remove", str(source)],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert removed.returncode == 0, removed.stderr or removed.stdout
    assert "Removed" in removed.stdout


def test_code_agent_list_shows_empty_when_no_packages(tmp_path: Path) -> None:
    env = _base_env(tmp_path)
    root = str(Path(__file__).resolve().parent.parent)
    result = subprocess.run(
        [sys.executable, "-m", "paw.pi_agent.code_agent", "list"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "No packages installed." in result.stdout


def test_code_agent_export_html_from_session_file(tmp_path: Path) -> None:
    env = _base_env(tmp_path)
    root = str(Path(__file__).resolve().parent.parent)
    session = tmp_path / "session.jsonl"
    session.write_text(
        '{"type":"session","version":3,"id":"abc","timestamp":"2024-01-01T00:00:00+00:00","cwd":"%s"}\n'
        '{"type":"message","id":"m1","parentId":null,"timestamp":"2024-01-01T00:00:01+00:00","message":{"role":"user","content":"hello","timestamp":1}}\n'
        % root,
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, "-m", "paw.pi_agent.code_agent", "--export", str(session)],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    html_path = Path(result.stdout.strip())
    assert html_path.exists()
    assert "<h3>user</h3>" in html_path.read_text(encoding="utf-8")


def test_code_agent_export_html_includes_session_info_and_summaries(tmp_path: Path) -> None:
    env = _base_env(tmp_path)
    root = str(Path(__file__).resolve().parent.parent)
    session = tmp_path / "session.jsonl"
    session.write_text(
        '{"type":"session","version":3,"id":"abc","timestamp":"2024-01-01T00:00:00+00:00","cwd":"%s"}\n'
        '{"type":"session_info","id":"s1","parentId":null,"timestamp":"2024-01-01T00:00:00+00:00","name":"demo"}\n'
        '{"type":"message","id":"m1","parentId":"s1","timestamp":"2024-01-01T00:00:01+00:00","message":{"role":"user","content":"hello","timestamp":1}}\n'
        '{"type":"branch_summary","id":"b1","parentId":"m1","timestamp":"2024-01-01T00:00:02+00:00","fromId":"m1","summary":"branch text"}\n'
        '{"type":"compaction","id":"c1","parentId":"b1","timestamp":"2024-01-01T00:00:03+00:00","summary":"compact text","firstKeptEntryId":"m1","tokensBefore":123}\n'
        % root,
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, "-m", "paw.pi_agent.code_agent", "--export", str(session)],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    html_text = Path(result.stdout.strip()).read_text(encoding="utf-8")
    assert "<title>demo</title>" in html_text
    assert "<h3>session_info</h3>" in html_text
    assert "<h3>branch_summary</h3>" in html_text
    assert "<h3>compaction</h3>" in html_text


def test_code_agent_session_flag_accepts_query(tmp_path: Path) -> None:
    from paw.pi_agent.ai import UserMessage
    from paw.pi_agent.code_agent.session_manager import SessionManager

    env = _base_env(tmp_path)
    root = str(Path(__file__).resolve().parent.parent)
    manager = SessionManager.create(root, tmp_path / "agent" / "sessions")
    manager.append_message(UserMessage(content="alpha prompt"))
    manager.set_session_name("alpha")

    result = subprocess.run(
        [sys.executable, "-m", "paw.pi_agent.code_agent", "--session", "alpha"],
        cwd=root,
        env={**env, "OPENAI_API_KEY": "test-key", "OPENAI_MODEL": "gpt-4o-mini"},
        input="/session\n/quit\n",
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    assert "sessionName: alpha" in result.stdout


def test_code_agent_resume_flag_accepts_query(tmp_path: Path) -> None:
    from paw.pi_agent.ai import UserMessage
    from paw.pi_agent.code_agent.session_manager import SessionManager

    env = _base_env(tmp_path)
    root = str(Path(__file__).resolve().parent.parent)
    first = SessionManager.create(root, tmp_path / "agent" / "sessions")
    first.append_message(UserMessage(content="alpha prompt"))
    first.set_session_name("alpha")
    second = SessionManager.create(root, tmp_path / "agent" / "sessions")
    second.append_message(UserMessage(content="beta prompt"))
    second.set_session_name("beta")

    result = subprocess.run(
        [sys.executable, "-m", "paw.pi_agent.code_agent", "--resume", "beta"],
        cwd=root,
        env={**env, "OPENAI_API_KEY": "test-key", "OPENAI_MODEL": "gpt-4o-mini"},
        input="/session\n/quit\n",
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    assert "sessionName: beta" in result.stdout


def test_code_agent_resume_flag_opens_session_picker(tmp_path: Path) -> None:
    from paw.pi_agent.ai import UserMessage
    from paw.pi_agent.code_agent.session_manager import SessionManager

    env = _base_env(tmp_path)
    root = str(Path(__file__).resolve().parent.parent)
    first = SessionManager.create(root, tmp_path / "agent" / "sessions")
    first.append_message(UserMessage(content="alpha prompt"))
    first.set_session_name("alpha")
    second = SessionManager.create(root, tmp_path / "agent" / "sessions")
    second.append_message(UserMessage(content="beta prompt"))
    second.set_session_name("beta")

    result = subprocess.run(
        [sys.executable, "-m", "paw.pi_agent.code_agent", "--resume"],
        cwd=root,
        env={**env, "OPENAI_API_KEY": "test-key", "OPENAI_MODEL": "gpt-4o-mini"},
        input="beta\n/session\n/quit\n",
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    assert "Resume Session (Current Folder)" in result.stdout
    assert "1. " in result.stdout
    assert "2. " in result.stdout
    assert "sessionName: beta" in result.stdout


def test_code_agent_resume_flag_searches_multiple_matches(tmp_path: Path) -> None:
    from paw.pi_agent.ai import UserMessage
    from paw.pi_agent.code_agent.session_manager import SessionManager

    env = _base_env(tmp_path)
    root = str(Path(__file__).resolve().parent.parent)
    older = SessionManager.create(root, tmp_path / "agent" / "sessions")
    older.append_message(UserMessage(content="older prompt"))
    older.set_session_name("beta-old")
    newer = SessionManager.create(root, tmp_path / "agent" / "sessions")
    newer.append_message(UserMessage(content="newer prompt"))
    newer.set_session_name("beta-new")

    result = subprocess.run(
        [sys.executable, "-m", "paw.pi_agent.code_agent", "--resume", "beta"],
        cwd=root,
        env={**env, "OPENAI_API_KEY": "test-key", "OPENAI_MODEL": "gpt-4o-mini"},
        input="1\n/session\n/quit\n",
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    assert "Resume Session (Current Folder)" in result.stdout
    assert "beta-new" in result.stdout
    assert "beta-old" in result.stdout
    assert "sessionName: beta-old" in result.stdout or "sessionName: beta-new" in result.stdout


def test_code_agent_resume_flag_shortens_home_paths(tmp_path: Path) -> None:
    from paw.pi_agent.ai import UserMessage
    from paw.pi_agent.code_agent.session_manager import SessionManager

    root = str(Path(__file__).resolve().parent.parent)
    home = tmp_path / "home"
    session_root = home / ".pi" / "agent" / "sessions"
    session_root.mkdir(parents=True)
    session = SessionManager.create(root, session_root)
    session.append_message(UserMessage(content="home prompt"))
    session.set_session_name("home-session")

    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = root if not pythonpath else f"{root}:{pythonpath}"
    env["HOME"] = str(home)
    env.pop("PI_CODING_AGENT_DIR", None)
    env["OPENAI_API_KEY"] = "test-key"
    env["OPENAI_MODEL"] = "gpt-4o-mini"

    result = subprocess.run(
        [sys.executable, "-m", "paw.pi_agent.code_agent", "--resume"],
        cwd=root,
        env=env,
        input="home-session\n/quit\n",
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    assert "~/.pi/agent/sessions" in result.stdout


def test_code_agent_resume_flag_supports_all_scope(tmp_path: Path) -> None:
    from paw.pi_agent.ai import UserMessage
    from paw.pi_agent.code_agent.session_manager import SessionManager

    env = _base_env(tmp_path)
    root = str(Path(__file__).resolve().parent.parent)
    session_root = tmp_path / "agent" / "sessions"
    first = SessionManager.create("cwd-a", session_root)
    first.append_message(UserMessage(content="alpha prompt"))
    first.set_session_name("alpha")
    second = SessionManager.create("cwd-b", session_root)
    second.append_message(UserMessage(content="beta prompt"))
    second.set_session_name("beta")

    result = subprocess.run(
        [sys.executable, "-m", "paw.pi_agent.code_agent", "--resume", "all:beta"],
        cwd=root,
        env={**env, "OPENAI_API_KEY": "test-key", "OPENAI_MODEL": "gpt-4o-mini"},
        input="/session\n/quit\n",
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    assert "sessionName: beta" in result.stdout


def test_code_agent_resume_flag_supports_all_scope_fuzzy_query(tmp_path: Path) -> None:
    from paw.pi_agent.ai import UserMessage
    from paw.pi_agent.code_agent.session_manager import SessionManager

    env = _base_env(tmp_path)
    root = str(Path(__file__).resolve().parent.parent)
    session_root = tmp_path / "agent" / "sessions"
    first = SessionManager.create("cwd-a", session_root)
    first.append_message(UserMessage(content="alpha prompt"))
    first.set_session_name("alpha")
    second = SessionManager.create("cwd-b", session_root)
    second.append_message(UserMessage(content="beta tail match"))
    second.set_session_name("beta")

    result = subprocess.run(
        [sys.executable, "-m", "paw.pi_agent.code_agent", "--resume", r"all:re:beta\s+tail"],
        cwd=root,
        env={**env, "OPENAI_API_KEY": "test-key", "OPENAI_MODEL": "gpt-4o-mini"},
        input="/session\n/quit\n",
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    assert "sessionName: beta" in result.stdout


def test_code_agent_resume_flag_shows_scope_title(tmp_path: Path) -> None:
    from paw.pi_agent.ai import UserMessage
    from paw.pi_agent.code_agent.session_manager import SessionManager

    env = _base_env(tmp_path)
    root = str(Path(__file__).resolve().parent.parent)
    session_root = tmp_path / "agent" / "sessions"
    current = SessionManager.create(root, session_root)
    current.append_message(UserMessage(content="alpha prompt"))
    current.set_session_name("alpha")
    other = SessionManager.create("other-cwd", session_root)
    other.append_message(UserMessage(content="alpha prompt two"))
    other.set_session_name("alpha")

    result = subprocess.run(
        [sys.executable, "-m", "paw.pi_agent.code_agent", "--resume"],
        cwd=root,
        env={**env, "OPENAI_API_KEY": "test-key", "OPENAI_MODEL": "gpt-4o-mini"},
        input="alpha\n/session\n/quit\n",
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    assert "Resume Session (Current Folder)" in result.stdout

    result_all = subprocess.run(
        [sys.executable, "-m", "paw.pi_agent.code_agent", "--resume", "all:alpha"],
        cwd=root,
        env={**env, "OPENAI_API_KEY": "test-key", "OPENAI_MODEL": "gpt-4o-mini"},
        input="1\n/session\n/quit\n",
        capture_output=True,
        text=True,
        check=False,
    )
    assert result_all.returncode == 0, result_all.stderr or result_all.stdout
    assert "Resume Session (All)" in result_all.stdout
    assert "alpha prompt" in result_all.stdout


def test_code_agent_session_dir_flag_controls_new_session_path(tmp_path: Path) -> None:
    env = _base_env(tmp_path)
    root = str(Path(__file__).resolve().parent.parent)
    session_root = tmp_path / "custom-sessions"

    result = subprocess.run(
        [sys.executable, "-m", "paw.pi_agent.code_agent", "--session-dir", str(session_root)],
        cwd=root,
        env={**env, "OPENAI_API_KEY": "test-key", "OPENAI_MODEL": "gpt-4o-mini"},
        input="/session\n/quit\n",
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    assert f"sessionFile: {session_root}" in result.stdout


def test_code_agent_theme_flag_sets_theme(tmp_path: Path) -> None:
    env = _base_env(tmp_path)
    workdir = tmp_path / "work"
    workdir.mkdir()
    themes_dir = tmp_path / "agent" / "themes"
    themes_dir.mkdir(parents=True)
    (themes_dir / "dark.json").write_text('{"name":"dark"}', encoding="utf-8")
    (themes_dir / "light.json").write_text('{"name":"light"}', encoding="utf-8")
    settings_path = workdir / ".pi" / "settings.json"
    original = settings_path.read_text(encoding="utf-8") if settings_path.exists() else None
    try:
        result = subprocess.run(
            [sys.executable, "-m", "paw.pi_agent.code_agent", "--theme", "light"],
            cwd=workdir,
            env={**env, "OPENAI_API_KEY": "test-key", "OPENAI_MODEL": "gpt-4o-mini"},
            input="/theme\n/quit\n",
            capture_output=True,
            text=True,
            check=False,
        )
        saved = settings_path.read_text(encoding="utf-8")
    finally:
        if original is None:
            settings_path.unlink(missing_ok=True)
        else:
            settings_path.write_text(original, encoding="utf-8")

    assert result.returncode == 0, result.stderr or result.stdout
    assert "theme: light" in result.stdout
    assert '"theme": "light"' in saved


def test_code_agent_handles_empty_project_settings_file(tmp_path: Path) -> None:
    env = _base_env(tmp_path)
    root = str(Path(__file__).resolve().parent.parent)
    settings_dir = Path(root) / ".pi"
    settings_dir.mkdir(exist_ok=True)
    settings_path = settings_dir / "settings.json"
    original = settings_path.read_text(encoding="utf-8") if settings_path.exists() else None
    settings_path.write_text("", encoding="utf-8")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "paw.pi_agent.code_agent", "--no-session"],
            cwd=root,
            env={**env, "OPENAI_API_KEY": "test-key", "OPENAI_MODEL": "gpt-4o-mini"},
            input="/quit\n",
            capture_output=True,
            text=True,
            check=False,
        )
    finally:
        if original is None:
            settings_path.unlink(missing_ok=True)
        else:
            settings_path.write_text(original, encoding="utf-8")

    assert result.returncode == 0, result.stderr or result.stdout
