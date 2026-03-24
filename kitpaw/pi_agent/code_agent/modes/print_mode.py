from __future__ import annotations

import sys

from ..agent_session import AgentSession
from .tool_display import make_tool_listener


async def run_print_mode(session: AgentSession, message: str) -> int:
    streaming_started = False

    def text_listener(event) -> None:
        nonlocal streaming_started
        if getattr(event, "type", None) == "message_update":
            assistant_event = getattr(event, "assistant_message_event", None)
            if getattr(assistant_event, "type", None) == "text_delta":
                streaming_started = True
                sys.stdout.write(assistant_event.delta)
                sys.stdout.flush()

    tool_listener = make_tool_listener()

    unsub_text = session.subscribe(text_listener)
    unsub_tool = session.subscribe(tool_listener)
    try:
        await session.prompt(message)
    finally:
        unsub_text()
        unsub_tool()
    if streaming_started:
        sys.stdout.write("\n")
        sys.stdout.flush()
    return 0
