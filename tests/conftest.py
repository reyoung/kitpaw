from __future__ import annotations

from paw.pi_agent.ai.local_env import load_local_env


def pytest_configure() -> None:
    load_local_env()
