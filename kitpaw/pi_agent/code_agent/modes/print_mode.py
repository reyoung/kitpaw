from __future__ import annotations

import sys

from ..agent_session import AgentSession


async def run_print_mode(session: AgentSession, message: str) -> int:
    streaming_started = False

    def listener(event) -> None:
        nonlocal streaming_started
        if getattr(event, "type", None) == "message_update":
            assistant_event = getattr(event, "assistant_message_event", None)
            if getattr(assistant_event, "type", None) == "text_delta":
                streaming_started = True
                sys.stdout.write(assistant_event.delta)
                sys.stdout.flush()

    unsubscribe = session.subscribe(listener)
    try:
        await session.prompt(message)
    finally:
        unsubscribe()
    if streaming_started:
        sys.stdout.write("\n")
        sys.stdout.flush()
    return 0
