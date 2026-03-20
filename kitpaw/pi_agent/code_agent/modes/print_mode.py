from __future__ import annotations

from ..agent_session import AgentSession


async def run_print_mode(session: AgentSession, message: str) -> int:
    chunks: list[str] = []

    def listener(event) -> None:
        if getattr(event, "type", None) == "message_update":
            assistant_event = getattr(event, "assistant_message_event", None)
            if getattr(assistant_event, "type", None) == "text_delta":
                chunks.append(assistant_event.delta)

    unsubscribe = session.subscribe(listener)
    try:
        await session.prompt(message)
    finally:
        unsubscribe()
    print("".join(chunks).strip())
    return 0
