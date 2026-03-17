from __future__ import annotations

import os

import pytest

from paw.pi_agent.ai import Context, TextContent, UserMessage, complete, get_model
from paw.pi_agent.ai.local_env import load_local_env

pytestmark = [pytest.mark.e2e, pytest.mark.real_e2e]


def test_real_openai_compatible_smoke() -> None:
    load_local_env()
    if os.getenv("PAW_RUN_REAL_E2E") != "1":
        pytest.skip("Set PAW_RUN_REAL_E2E=1 to run the real upstream smoke test.")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY is not configured.")

    models = []
    for name in (os.getenv("OPENAI_MODEL"), os.getenv("OPENAI_FALLBACK_MODEL")):
        if name and name not in models:
            models.append(name)

    if not models:
        models.append("gpt-4o-mini")

    errors: list[str] = []
    for model_name in models:
        result = complete(
            get_model("openai", model_name),
            Context(messages=[UserMessage(content="Reply with exactly the word pong.")]),
            {"api_key": api_key, "max_tokens": 256},
        )
        if result.stop_reason == "error":
            errors.append(f"{model_name}: {result.error_message}")
            continue

        text = "".join(block.text for block in result.content if isinstance(block, TextContent)).strip()
        assert text
        assert result.stop_reason in {"stop", "length"}
        return

    pytest.fail("Real upstream smoke failed for all configured models:\n" + "\n".join(errors))
