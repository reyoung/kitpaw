from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_claw_module_cli_starts(tmp_path: Path) -> None:
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH")
    root = str(Path(__file__).resolve().parent.parent.parent)
    env["PYTHONPATH"] = root if not pythonpath else f"{root}:{pythonpath}"
    env["PI_CODING_AGENT_DIR"] = str(tmp_path / "agent")
    env["OPENAI_API_KEY"] = "test-key"
    env["OPENAI_MODEL"] = "gpt-4o-mini"

    process = subprocess.run(
        [sys.executable, "-m", "kitpaw.claw", "--no-session"],
        cwd=root,
        env=env,
        input="/quit\n",
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert process.returncode == 0, process.stderr or process.stdout
    assert "claw interactive mode" in process.stdout


def test_claw_cli_rejects_agent_flag(tmp_path: Path) -> None:
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH")
    root = str(Path(__file__).resolve().parent.parent.parent)
    env["PYTHONPATH"] = root if not pythonpath else f"{root}:{pythonpath}"
    env["PI_CODING_AGENT_DIR"] = str(tmp_path / "agent")

    process = subprocess.run(
        [sys.executable, "-m", "kitpaw.claw", "--agent", "codex", "--no-session"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert process.returncode != 0
    assert "unrecognized arguments: --agent" in process.stderr
