from __future__ import annotations

import asyncio

from kitpaw.pi_agent.agent import Agent


def test_agent_abort_only_signals_current_run_and_does_not_cancel_caller(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        started = asyncio.Event()
        released = asyncio.Event()
        observed_cancel_event: asyncio.Event | None = None

        async def fake_run_agent_loop(
            prompts,  # noqa: ANN001
            context,  # noqa: ANN001
            config,  # noqa: ANN001
            emit,  # noqa: ANN001
            cancel_event,  # noqa: ANN001
            stream_fn,  # noqa: ANN001
        ) -> None:
            nonlocal observed_cancel_event
            observed_cancel_event = cancel_event
            started.set()
            await cancel_event.wait()
            await released.wait()

        monkeypatch.setattr("kitpaw.pi_agent.agent.agent._run_agent_loop", fake_run_agent_loop)

        agent = Agent()
        prompt_task = asyncio.create_task(agent.prompt("hello"))

        await started.wait()
        assert agent.state.is_streaming is True

        agent.abort()

        assert observed_cancel_event is not None
        assert observed_cancel_event.is_set() is True
        assert prompt_task.cancelled() is False
        assert prompt_task.done() is False

        wait_task = asyncio.create_task(agent.wait_for_idle())
        await asyncio.sleep(0)
        assert wait_task.done() is False

        released.set()
        await wait_task
        await prompt_task

        assert agent.state.is_streaming is False

    asyncio.run(scenario())


def test_agent_abort_is_noop_when_nothing_is_running() -> None:
    async def scenario() -> None:
        agent = Agent()
        agent.abort()
        await agent.wait_for_idle()

    asyncio.run(scenario())
