from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from typing import Any

from ..agent_session import AgentSession
from ..tool_error_limit import peek_tool_error_limit_exception


def _encode(value: Any) -> Any:
    if is_dataclass(value):
        return {k: _encode(v) for k, v in asdict(value).items()}
    if isinstance(value, list):
        return [_encode(v) for v in value]
    if isinstance(value, dict):
        return {k: _encode(v) for k, v in value.items()}
    return value


async def run_json_mode(session: AgentSession, message: str) -> int:
    print(json.dumps({"type": "session", "id": session.session_id, "cwd": session.cwd}))
    unsubscribe = session.subscribe(lambda event: print(json.dumps(_encode(event), default=str)))
    try:
        await session.prompt(message)
    finally:
        unsubscribe()
    if peek_tool_error_limit_exception(session) is not None:
        return 1
    return 0
