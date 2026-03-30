from __future__ import annotations

import json
from typing import Any

from ..pi_agent.agent.types import AgentToolResult
from ..pi_agent.ai.types import TextContent


def json_result(payload: dict[str, Any]) -> AgentToolResult[dict[str, Any]]:
    return AgentToolResult(
        content=[TextContent(text=json.dumps(payload, ensure_ascii=False))],
        details=payload,
    )


def error_result(message: str, **extra: Any) -> AgentToolResult[dict[str, Any]]:
    return json_result({"status": "error", "error": message, **extra})
