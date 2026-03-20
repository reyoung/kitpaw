from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from paw.pi_agent.ai.local_env import load_local_env

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
            "paw.pi_agent.code_agent",
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
