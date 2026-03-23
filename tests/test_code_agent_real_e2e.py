from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from kitpaw.pi_agent.ai.local_env import load_local_env

pytestmark = [pytest.mark.e2e, pytest.mark.real_e2e]


def test_code_agent_print_real_openai_compatible_smoke() -> None:
    load_local_env()
    if os.getenv("PAW_RUN_REAL_E2E") != "1":
        pytest.skip("Set PAW_RUN_REAL_E2E=1 to run the real upstream smoke test.")

    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY is not configured.")

    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH")
    root = str(Path(__file__).resolve().parent.parent)
    env["PYTHONPATH"] = root if not pythonpath else f"{root}:{pythonpath}"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "kitpaw.pi_agent.code_agent",
            "-p",
            "Reply with exactly the word pong.",
        ],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    assert result.stdout.strip()


def test_code_agent_interactive_streaming_real_e2e(tmp_path: Path) -> None:
    """Verify that interactive mode streams text via sys.stdout.write()."""
    load_local_env()
    if os.getenv("PAW_RUN_REAL_E2E") != "1":
        pytest.skip("Set PAW_RUN_REAL_E2E=1 to run the real upstream smoke test.")

    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY is not configured.")

    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH")
    root = str(Path(__file__).resolve().parent.parent)
    env["PYTHONPATH"] = root if not pythonpath else f"{root}:{pythonpath}"
    env["PI_CODING_AGENT_DIR"] = str(tmp_path / "agent")

    result = subprocess.run(
        [sys.executable, "-m", "kitpaw.pi_agent.code_agent", "--no-session"],
        cwd=root,
        env=env,
        input="Reply with exactly the word pong.\n/quit\n",
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    # The banner and prompt are printed, then the streamed response appears.
    # Verify that the model's response is present in stdout.
    assert "pong" in result.stdout.lower(), f"Expected 'pong' in stdout:\n{result.stdout}"
