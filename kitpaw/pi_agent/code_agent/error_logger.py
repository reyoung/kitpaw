"""Lightweight JSONL error logger for agent sessions.

Subscribes to session events and writes one JSON line per error
(e.g. tool call failures) to the specified file.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from .agent_session import AgentSession


def _extract_error_text(result: Any) -> str:
    """Best-effort extraction of a human-readable error string from a tool result."""
    if isinstance(result, str):
        return result
    # AgentToolResult has a .content list of TextContent/ImageContent
    content = getattr(result, "content", None)
    if content and isinstance(content, list):
        parts: list[str] = []
        for item in content:
            text = getattr(item, "text", None)
            if text:
                parts.append(text)
        if parts:
            return "\n".join(parts)
    return str(result)


def setup_error_logger(session: AgentSession, path: str) -> Callable[[], None]:
    """Subscribe to *session* events and log errors to a JSONL file at *path*.

    Returns an unsubscribe callable that also closes the file.
    """
    fh = open(path, "a", encoding="utf-8")  # noqa: SIM115

    def _on_event(event: Any) -> None:
        if getattr(event, "type", None) != "tool_execution_end":
            return
        if not getattr(event, "is_error", False):
            return

        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": "tool_error",
            "tool_name": getattr(event, "tool_name", ""),
            "tool_call_id": getattr(event, "tool_call_id", ""),
            "error": _extract_error_text(getattr(event, "result", "")),
        }
        fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        fh.flush()

    unsubscribe = session.subscribe(_on_event)

    def cleanup() -> None:
        unsubscribe()
        fh.close()

    return cleanup
