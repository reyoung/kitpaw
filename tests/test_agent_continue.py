from __future__ import annotations

import asyncio
import inspect

from kitpaw.pi_agent.agent import Agent
from kitpaw.pi_agent.agent.types import MessageEndEvent
from kitpaw.pi_agent.ai.types import AssistantMessage, TextContent, Usage, UserMessage, now_ms


def _assistant_message(text: str) -> AssistantMessage:
    return AssistantMessage(
        api="openai-responses",
        provider="openai",
        model="gpt-4o-mini",
        content=[TextContent(text=text)],
        usage=Usage(),
        stop_reason="stop",
        timestamp=now_ms(),
    )


def _user_message(text: str) -> UserMessage:
    return UserMessage(content=[TextContent(text=text)], timestamp=now_ms())


def test_continue_skips_trailing_assistant_messages_and_reuses_last_non_assistant_context(
    monkeypatch,
) -> None:
    async def scenario() -> None:
        seen_context_roles: list[str] = []

        async def fake_run_agent_loop(
            prompts,  # noqa: ANN001
            context,  # noqa: ANN001
            config,  # noqa: ANN001
            emit,  # noqa: ANN001
            cancel_event,  # noqa: ANN001
            stream_fn,  # noqa: ANN001
        ) -> None:
            seen_context_roles.extend(getattr(message, "role", "") for message in context.messages)
            result = emit(MessageEndEvent(message=_assistant_message("continued")))
            if inspect.isawaitable(result):
                await result

        monkeypatch.setattr("kitpaw.pi_agent.agent.agent._run_agent_loop", fake_run_agent_loop)

        agent = Agent()
        agent.replace_messages(
            [
                _user_message("Initial"),
                _assistant_message("Partial answer"),
                _assistant_message("Another assistant tail"),
            ]
        )

        await agent.continue_()

        assert seen_context_roles == ["user"]
        assert [getattr(message, "role", None) for message in agent.state.messages] == [
            "user",
            "assistant",
            "assistant",
            "assistant",
        ]

    asyncio.run(scenario())


def test_continue_still_rejects_when_context_only_has_assistant_messages(monkeypatch) -> None:
    async def scenario() -> None:
        agent = Agent()
        agent.replace_messages([_assistant_message("hello")])

        try:
            await agent.continue_()
        except ValueError as exc:
            assert str(exc) == "Cannot continue from message role: assistant"
        else:
            raise AssertionError("continue_() should reject assistant-only context")

    asyncio.run(scenario())
