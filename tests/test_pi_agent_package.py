from __future__ import annotations

import os
import subprocess
import sys


def test_import_paw_pi_agent_does_not_eagerly_import_tui() -> None:
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = os.getcwd() if not pythonpath else f"{os.getcwd()}:{pythonpath}"

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import paw.pi_agent; "
                "raise SystemExit(0 if 'paw.pi_agent.tui' not in sys.modules else 1)"
            ),
        ],
        check=False,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr or result.stdout
