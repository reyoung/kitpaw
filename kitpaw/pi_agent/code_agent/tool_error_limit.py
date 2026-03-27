from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from ..agent.types import AfterToolCallContext, AfterToolCallResult
from .agent_session import AgentSession


class ToolErrorLimitExceededError(RuntimeError):
    def __init__(self, limit: int, failures: int, tool_name: str) -> None:
        super().__init__(
            f"Tool error limit reached after {failures} failures (limit: {limit}). Last failing tool: {tool_name}."
        )
        self.limit = limit
        self.failures = failures
        self.tool_name = tool_name


_TOOL_ERROR_LIMIT_EXCEPTION_ATTR = "_tool_error_limit_exception"


def configure_tool_error_limit(session: AgentSession, limit: int) -> None:
    if limit < 1:
        raise ValueError("max_tool_errors must be at least 1")

    previous_hook = session.agent._after_tool_call  # type: ignore[attr-defined]
    failure_count = 0
    setattr(session, _TOOL_ERROR_LIMIT_EXCEPTION_ATTR, None)

    async def limited_after_tool_call(
        context: AfterToolCallContext,
        cancel_event: asyncio.Event | None,
    ) -> AfterToolCallResult | None:
        nonlocal failure_count

        previous_result: AfterToolCallResult | None = None
        if previous_hook is not None:
            previous_result = previous_hook(context, cancel_event)
            if isinstance(previous_result, Awaitable):
                previous_result = await previous_result

        effective_is_error = context.is_error
        if previous_result is not None and previous_result.is_error is not None:
            effective_is_error = previous_result.is_error

        if effective_is_error:
            failure_count += 1
            if failure_count >= limit:
                error = ToolErrorLimitExceededError(limit=limit, failures=failure_count, tool_name=context.tool_call.name)
                setattr(session, _TOOL_ERROR_LIMIT_EXCEPTION_ATTR, error)
                raise error

        return previous_result

    session.agent.set_after_tool_call(limited_after_tool_call)


def consume_tool_error_limit_exception(session: AgentSession) -> ToolErrorLimitExceededError | None:
    error = getattr(session, _TOOL_ERROR_LIMIT_EXCEPTION_ATTR, None)
    setattr(session, _TOOL_ERROR_LIMIT_EXCEPTION_ATTR, None)
    return error


def peek_tool_error_limit_exception(session: AgentSession) -> ToolErrorLimitExceededError | None:
    return getattr(session, _TOOL_ERROR_LIMIT_EXCEPTION_ATTR, None)
