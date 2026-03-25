from __future__ import annotations

import asyncio
import contextlib
import inspect
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import asdict, dataclass, is_dataclass, replace
from typing import Any

from jsonschema import ValidationError
from jsonschema.validators import validator_for

from ..ai.event_stream import EventStream
from ..ai.stream import stream as default_stream
from ..ai.types import (
    AssistantMessage,
    Context,
    Message,
    Model,
    SimpleStreamOptions,
    TextContent,
    ToolResultMessage,
    Usage,
    UserMessage,
    now_ms,
)
from .types import (
    AfterToolCallContext,
    AfterToolCallResult,
    AgentContext,
    AgentEndEvent,
    AgentEvent,
    AgentLoopConfig,
    AgentMessage,
    AgentStartEvent,
    AgentTool,
    AgentToolCall,
    AgentToolResult,
    AgentToolUpdateCallback,
    BeforeToolCallContext,
    BeforeToolCallResult,
    MessageEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolExecutionUpdateEvent,
    TurnEndEvent,
    TurnStartEvent,
)

AgentEventSink = Callable[[AgentEvent], Awaitable[None] | None]


class AgentEventStream(EventStream[AgentEvent, list[AgentMessage]]):
    def __init__(self, producer: Callable[[], AsyncIterator[AgentEvent]]) -> None:
        super().__init__(
            producer=producer,
            is_complete=lambda event: isinstance(event, AgentEndEvent),
            extract_result=lambda event: event.messages,
        )


def default_convert_to_llm(messages: list[AgentMessage]) -> list[Message]:
    """Pass through only built-in ai message dataclasses.

    This default is intentionally conservative. Custom AgentMessageLike values
    remain part of agent state and events, but they are filtered out here
    unless the caller overrides convert_to_llm.
    """

    return [
        message
        for message in messages
        if isinstance(message, (UserMessage, AssistantMessage, ToolResultMessage))
    ]


def normalize_simple_stream_options(
    options: SimpleStreamOptions | Mapping[str, Any] | None,
) -> SimpleStreamOptions:
    if options is None:
        return SimpleStreamOptions()
    if isinstance(options, SimpleStreamOptions):
        return replace(options)
    if is_dataclass(options):
        return SimpleStreamOptions(**asdict(options))
    if isinstance(options, Mapping):
        return SimpleStreamOptions(**dict(options))
    raise TypeError(f"Unsupported stream options type: {type(options)!r}")


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _clone_agent_message(message: AgentMessage) -> AgentMessage:
    if is_dataclass(message):
        try:
            return replace(message)
        except TypeError:
            return message
    return message


def _agent_message_role(message: AgentMessage) -> str | None:
    return getattr(message, "role", None)


def _assistant_stop_reason(message: AgentMessage) -> str | None:
    return getattr(message, "stop_reason", None)


def _assistant_error_message(message: AgentMessage) -> str | None:
    return getattr(message, "error_message", None)


def _create_empty_usage() -> Usage:
    return Usage()


def create_error_assistant_message(model: Model, reason: str, error_message: str) -> AssistantMessage:
    return AssistantMessage(
        api=model.api,
        provider=model.provider,
        model=model.id,
        content=[TextContent(text="")],
        usage=_create_empty_usage(),
        stop_reason=reason,  # type: ignore[arg-type]
        error_message=error_message,
        timestamp=now_ms(),
    )


def _create_error_tool_result(message: str) -> AgentToolResult[Any]:
    return AgentToolResult(content=[TextContent(text=message)], details={})


def _normalize_before_result(
    result: BeforeToolCallResult | Mapping[str, Any] | None,
) -> BeforeToolCallResult | None:
    if result is None:
        return None
    if isinstance(result, BeforeToolCallResult):
        return result
    if isinstance(result, Mapping):
        return BeforeToolCallResult(block=bool(result.get("block", False)), reason=result.get("reason"))
    if is_dataclass(result):
        return BeforeToolCallResult(**asdict(result))
    raise TypeError(f"Unsupported beforeToolCall result type: {type(result)!r}")


def _normalize_after_result(
    result: AfterToolCallResult | Mapping[str, Any] | None,
) -> AfterToolCallResult | None:
    if result is None:
        return None
    if isinstance(result, AfterToolCallResult):
        return result
    if isinstance(result, Mapping):
        return AfterToolCallResult(
            content=result.get("content"),
            details=result.get("details"),
            is_error=result.get("is_error"),
        )
    if is_dataclass(result):
        return AfterToolCallResult(**asdict(result))
    raise TypeError(f"Unsupported afterToolCall result type: {type(result)!r}")


def _validate_tool_arguments(tool: AgentTool[Any, Any], tool_call: AgentToolCall) -> Any:
    schema = tool.parameters
    try:
        validator_cls = validator_for(schema)
        validator_cls.check_schema(schema)
        validator = validator_cls(schema)
        validator.validate(tool_call.arguments)
        return tool_call.arguments
    except ValidationError as exc:
        raise ValueError(f"Tool {tool.name} arguments are invalid: {exc.message}") from exc
    except Exception as exc:
        raise ValueError(f"Tool {tool.name} arguments could not be validated: {exc}") from exc


def _invoke_tool_execute(
    tool: AgentTool[Any, Any],
    tool_call_id: str,
    args: Any,
    cancel_event: asyncio.Event | None,
    on_update: AgentToolUpdateCallback | None,
) -> Any:
    execute = tool.execute
    try:
        signature = inspect.signature(execute)
    except (TypeError, ValueError):
        return execute(tool_call_id, args, cancel_event, on_update)

    parameters = list(signature.parameters.values())
    positional = [
        parameter
        for parameter in parameters
        if parameter.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    has_var_positional = any(parameter.kind == inspect.Parameter.VAR_POSITIONAL for parameter in parameters)

    if has_var_positional or len(positional) >= 4:
        return execute(tool_call_id, args, cancel_event, on_update)
    if len(positional) == 3:
        return execute(tool_call_id, args, cancel_event)

    keyword_args: dict[str, Any] = {}
    if "cancel_event" in signature.parameters:
        keyword_args["cancel_event"] = cancel_event
    if "on_update" in signature.parameters:
        keyword_args["on_update"] = on_update
    if keyword_args:
        return execute(tool_call_id, args, **keyword_args)
    return execute(tool_call_id, args)


async def _emit_event(emit: AgentEventSink, event: AgentEvent) -> None:
    result = emit(event)
    if inspect.isawaitable(result):
        await result


def agent_loop(
    prompts: list[AgentMessage],
    context: AgentContext,
    config: AgentLoopConfig,
    cancel_event: asyncio.Event | None = None,
    stream_fn: Callable[[Model, Context, SimpleStreamOptions | Mapping[str, Any] | None], Any] | None = None,
) -> AgentEventStream:
    async def producer() -> AsyncIterator[AgentEvent]:
        queue: asyncio.Queue[AgentEvent | object] = asyncio.Queue()
        done = object()

        async def emit(event: AgentEvent) -> None:
            await queue.put(event)

        async def runner() -> None:
            try:
                await _run_agent_loop(prompts, context, config, emit, cancel_event, stream_fn)
            finally:
                await queue.put(done)

        task = asyncio.create_task(runner())
        try:
            while True:
                item = await queue.get()
                if item is done:
                    break
                yield item  # type: ignore[misc]
        finally:
            if not task.done():
                task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    return AgentEventStream(producer)


def agent_loop_continue(
    context: AgentContext,
    config: AgentLoopConfig,
    cancel_event: asyncio.Event | None = None,
    stream_fn: Callable[[Model, Context, SimpleStreamOptions | Mapping[str, Any] | None], Any] | None = None,
) -> AgentEventStream:
    if not context.messages:
        raise ValueError("Cannot continue: no messages in context")

    if _agent_message_role(context.messages[-1]) == "assistant":
        raise ValueError("Cannot continue from message role: assistant")

    return agent_loop([], context, config, cancel_event, stream_fn)


async def _run_agent_loop(
    prompts: list[AgentMessage],
    context: AgentContext,
    config: AgentLoopConfig,
    emit: AgentEventSink,
    cancel_event: asyncio.Event | None = None,
    stream_fn: Callable[[Model, Context, SimpleStreamOptions | Mapping[str, Any] | None], Any] | None = None,
) -> list[AgentMessage]:
    new_messages: list[AgentMessage] = list(prompts)
    current_context = AgentContext(
        system_prompt=context.system_prompt,
        messages=[*context.messages, *prompts],
        tools=context.tools,
    )

    try:
        await _emit_event(emit, AgentStartEvent())
        await _emit_event(emit, TurnStartEvent())
        for prompt in prompts:
            await _emit_event(emit, MessageStartEvent(message=_clone_agent_message(prompt)))
            await _emit_event(emit, MessageEndEvent(message=_clone_agent_message(prompt)))

        await _run_loop(current_context, new_messages, config, emit, cancel_event, stream_fn)
        return new_messages
    except asyncio.CancelledError as exc:
        error_message = create_error_assistant_message(
            config.model,
            "aborted",
            str(exc) if str(exc) else "Request aborted by user",
        )
        new_messages.append(error_message)
        await _emit_event(emit, MessageStartEvent(message=error_message))
        await _emit_event(emit, MessageEndEvent(message=error_message))
        await _emit_event(emit, AgentEndEvent(messages=new_messages))
        return new_messages
    except Exception as exc:
        error_message = create_error_assistant_message(config.model, "error", str(exc))
        new_messages.append(error_message)
        await _emit_event(emit, MessageStartEvent(message=error_message))
        await _emit_event(emit, MessageEndEvent(message=error_message))
        await _emit_event(emit, AgentEndEvent(messages=new_messages))
        return new_messages


async def _run_loop(
    current_context: AgentContext,
    new_messages: list[AgentMessage],
    config: AgentLoopConfig,
    emit: AgentEventSink,
    cancel_event: asyncio.Event | None = None,
    stream_fn: Callable[[Model, Context, SimpleStreamOptions | Mapping[str, Any] | None], Any] | None = None,
) -> None:
    first_turn = True
    pending_messages = list(
        ((await _maybe_await(config.get_steering_messages())) or []) if config.get_steering_messages else []
    )

    while True:
        has_more_tool_calls = True

        while has_more_tool_calls or pending_messages:
            if not first_turn:
                await _emit_event(emit, TurnStartEvent())
            else:
                first_turn = False

            if pending_messages:
                for message in pending_messages:
                    await _emit_event(emit, MessageStartEvent(message=_clone_agent_message(message)))
                    await _emit_event(emit, MessageEndEvent(message=_clone_agent_message(message)))
                    current_context.messages.append(message)
                    new_messages.append(message)
                pending_messages = []

            assistant_message = await _stream_assistant_response(
                current_context,
                config,
                emit,
                cancel_event,
                stream_fn,
            )
            new_messages.append(assistant_message)

            if _assistant_stop_reason(assistant_message) in {"error", "aborted"}:
                await _emit_event(emit, TurnEndEvent(message=assistant_message, tool_results=[]))
                await _emit_event(emit, AgentEndEvent(messages=new_messages))
                return

            tool_calls = [block for block in assistant_message.content if getattr(block, "type", None) == "toolCall"]
            has_more_tool_calls = bool(tool_calls)

            tool_results: list[ToolResultMessage] = []
            if has_more_tool_calls:
                tool_results.extend(
                    await _execute_tool_calls(
                        current_context,
                        assistant_message,
                        config,
                        emit,
                        cancel_event,
                    )
                )
                for result in tool_results:
                    current_context.messages.append(result)
                    new_messages.append(result)

            await _emit_event(emit, TurnEndEvent(message=assistant_message, tool_results=tool_results))
            pending_messages = list(
                ((await _maybe_await(config.get_steering_messages())) or []) if config.get_steering_messages else []
            )

        follow_up_messages = list(
            ((await _maybe_await(config.get_follow_up_messages())) or []) if config.get_follow_up_messages else []
        )
        if follow_up_messages:
            pending_messages = follow_up_messages
            continue

        break

    await _emit_event(emit, AgentEndEvent(messages=new_messages))


async def _stream_assistant_response(
    context: AgentContext,
    config: AgentLoopConfig,
    emit: AgentEventSink,
    cancel_event: asyncio.Event | None = None,
    stream_fn: Callable[[Model, Context, SimpleStreamOptions | Mapping[str, Any] | None], Any] | None = None,
) -> AssistantMessage:
    messages = context.messages
    if config.transform_context:
        transformed = await _maybe_await(config.transform_context(messages, cancel_event))
        if transformed is not None:
            messages = transformed

    llm_messages = await _maybe_await(config.convert_to_llm(messages))
    llm_context = Context(
        system_prompt=context.system_prompt,
        messages=llm_messages,
        tools=context.tools,
    )

    options = normalize_simple_stream_options(config.stream_options)
    if config.get_api_key:
        resolved_api_key = await _maybe_await(config.get_api_key(config.model.provider))
        if resolved_api_key:
            options.api_key = resolved_api_key

    stream_function = stream_fn or default_stream
    response = await _maybe_await(stream_function(config.model, llm_context, options))

    partial_message: AssistantMessage | None = None
    added_partial = False

    async for event in response:
        match event.type:
            case "start":
                partial_message = event.partial
                context.messages.append(partial_message)
                added_partial = True
                await _emit_event(emit, MessageStartEvent(message=_clone_agent_message(partial_message)))
            case (
                "text_start"
                | "text_delta"
                | "text_end"
                | "thinking_start"
                | "thinking_delta"
                | "thinking_end"
                | "toolcall_start"
                | "toolcall_delta"
                | "toolcall_end"
            ):
                if partial_message is not None:
                    partial_message = event.partial
                    context.messages[-1] = partial_message
                    await _emit_event(
                        emit,
                        MessageUpdateEvent(
                            message=_clone_agent_message(partial_message),
                            assistant_message_event=event,
                        ),
                    )
            case "done" | "error":
                final_message = await response.result()
                if added_partial:
                    context.messages[-1] = final_message
                else:
                    context.messages.append(final_message)
                    await _emit_event(emit, MessageStartEvent(message=_clone_agent_message(final_message)))
                await _emit_event(emit, MessageEndEvent(message=final_message))
                return final_message

    final_message = await response.result()
    if added_partial:
        context.messages[-1] = final_message
    else:
        context.messages.append(final_message)
        await _emit_event(emit, MessageStartEvent(message=_clone_agent_message(final_message)))
    await _emit_event(emit, MessageEndEvent(message=final_message))
    return final_message


async def _execute_tool_calls(
    current_context: AgentContext,
    assistant_message: AssistantMessage,
    config: AgentLoopConfig,
    emit: AgentEventSink,
    cancel_event: asyncio.Event | None = None,
) -> list[ToolResultMessage]:
    tool_calls = [block for block in assistant_message.content if getattr(block, "type", None) == "toolCall"]
    if config.tool_execution == "sequential":
        return await _execute_tool_calls_sequential(
            current_context,
            assistant_message,
            tool_calls,
            config,
            emit,
            cancel_event,
        )
    return await _execute_tool_calls_parallel(
        current_context,
        assistant_message,
        tool_calls,
        config,
        emit,
        cancel_event,
    )


async def _execute_tool_calls_sequential(
    current_context: AgentContext,
    assistant_message: AssistantMessage,
    tool_calls: list[AgentToolCall],
    config: AgentLoopConfig,
    emit: AgentEventSink,
    cancel_event: asyncio.Event | None = None,
) -> list[ToolResultMessage]:
    results: list[ToolResultMessage] = []

    for tool_call in tool_calls:
        await _emit_event(
            emit,
            ToolExecutionStartEvent(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                args=tool_call.arguments,
            ),
        )

        preparation = await _prepare_tool_call(current_context, assistant_message, tool_call, config, cancel_event)
        if isinstance(preparation, ImmediateToolCallOutcome):
            results.append(await _emit_tool_call_outcome(tool_call, preparation.result, preparation.is_error, emit))
        else:
            executed = await _execute_prepared_tool_call(preparation, emit, cancel_event)
            results.append(
                await _finalize_executed_tool_call(
                    current_context,
                    assistant_message,
                    preparation,
                    executed,
                    config,
                    emit,
                    cancel_event,
                )
            )

    return results


async def _execute_tool_calls_parallel(
    current_context: AgentContext,
    assistant_message: AssistantMessage,
    tool_calls: list[AgentToolCall],
    config: AgentLoopConfig,
    emit: AgentEventSink,
    cancel_event: asyncio.Event | None = None,
) -> list[ToolResultMessage]:
    results: list[ToolResultMessage] = []
    runnable_calls: list[PreparedToolCall] = []

    for tool_call in tool_calls:
        await _emit_event(
            emit,
            ToolExecutionStartEvent(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                args=tool_call.arguments,
            ),
        )

        preparation = await _prepare_tool_call(current_context, assistant_message, tool_call, config, cancel_event)
        if isinstance(preparation, ImmediateToolCallOutcome):
            results.append(await _emit_tool_call_outcome(tool_call, preparation.result, preparation.is_error, emit))
        else:
            runnable_calls.append(preparation)

    running_calls = [
        (prepared, asyncio.create_task(_execute_prepared_tool_call(prepared, emit, cancel_event)))
        for prepared in runnable_calls
    ]

    for prepared, task in running_calls:
        executed = await task
        results.append(
            await _finalize_executed_tool_call(
                current_context,
                assistant_message,
                prepared,
                executed,
                config,
                emit,
                cancel_event,
            )
        )

    return results


@dataclass(slots=True)
class PreparedToolCall:
    tool_call: AgentToolCall
    tool: AgentTool[Any, Any]
    args: Any


@dataclass(slots=True)
class ImmediateToolCallOutcome:
    result: AgentToolResult[Any]
    is_error: bool


@dataclass(slots=True)
class ExecutedToolCallOutcome:
    result: AgentToolResult[Any]
    is_error: bool


async def _prepare_tool_call(
    current_context: AgentContext,
    assistant_message: AssistantMessage,
    tool_call: AgentToolCall,
    config: AgentLoopConfig,
    cancel_event: asyncio.Event | None = None,
) -> PreparedToolCall | ImmediateToolCallOutcome:
    tool = next((candidate for candidate in (current_context.tools or []) if candidate.name == tool_call.name), None)
    if tool is None:
        msg = config.format_tool_not_found(tool_call.name) if config.format_tool_not_found else f"Tool {tool_call.name} not found"
        return ImmediateToolCallOutcome(
            result=_create_error_tool_result(msg),
            is_error=True,
        )

    try:
        validated_args = _validate_tool_arguments(tool, tool_call)
        if config.before_tool_call:
            before_result = await _maybe_await(
                config.before_tool_call(
                    BeforeToolCallContext(
                        assistant_message=assistant_message,
                        tool_call=tool_call,
                        args=validated_args,
                        context=current_context,
                    ),
                    cancel_event,
                )
            )
            normalized_before = _normalize_before_result(before_result)
            if normalized_before and normalized_before.block:
                return ImmediateToolCallOutcome(
                    result=_create_error_tool_result(normalized_before.reason or "Tool execution was blocked"),
                    is_error=True,
                )
        return PreparedToolCall(tool_call=tool_call, tool=tool, args=validated_args)
    except Exception as exc:
        return ImmediateToolCallOutcome(
            result=_create_error_tool_result(str(exc)),
            is_error=True,
        )


async def _execute_prepared_tool_call(
    prepared: PreparedToolCall,
    emit: AgentEventSink,
    cancel_event: asyncio.Event | None = None,
) -> ExecutedToolCallOutcome:
    update_tasks: list[asyncio.Task[None]] = []

    def on_update(partial_result: AgentToolResult[Any]) -> None:
        task_or_none = emit(
            ToolExecutionUpdateEvent(
                tool_call_id=prepared.tool_call.id,
                tool_name=prepared.tool_call.name,
                args=prepared.tool_call.arguments,
                partial_result=partial_result,
            )
        )
        if inspect.isawaitable(task_or_none):
            update_tasks.append(asyncio.create_task(task_or_none))

    try:
        result = await _maybe_await(
            _invoke_tool_execute(prepared.tool, prepared.tool_call.id, prepared.args, cancel_event, on_update)
        )
        if update_tasks:
            await asyncio.gather(*update_tasks)
        return ExecutedToolCallOutcome(result=result, is_error=False)
    except Exception as exc:
        if update_tasks:
            await asyncio.gather(*update_tasks, return_exceptions=True)
        return ExecutedToolCallOutcome(result=_create_error_tool_result(str(exc)), is_error=True)


async def _finalize_executed_tool_call(
    current_context: AgentContext,
    assistant_message: AssistantMessage,
    prepared: PreparedToolCall,
    executed: ExecutedToolCallOutcome,
    config: AgentLoopConfig,
    emit: AgentEventSink,
    cancel_event: asyncio.Event | None = None,
) -> ToolResultMessage:
    result = executed.result
    is_error = executed.is_error

    if config.after_tool_call:
        after_result = await _maybe_await(
            config.after_tool_call(
                AfterToolCallContext(
                    assistant_message=assistant_message,
                    tool_call=prepared.tool_call,
                    args=prepared.args,
                    result=result,
                    is_error=is_error,
                    context=current_context,
                ),
                cancel_event,
            )
        )
        normalized_after = _normalize_after_result(after_result)
        if normalized_after:
            result = AgentToolResult(
                content=normalized_after.content if normalized_after.content is not None else result.content,
                details=normalized_after.details if normalized_after.details is not None else result.details,
            )
            is_error = normalized_after.is_error if normalized_after.is_error is not None else is_error

    return await _emit_tool_call_outcome(prepared.tool_call, result, is_error, emit)


async def _emit_tool_call_outcome(
    tool_call: AgentToolCall,
    result: AgentToolResult[Any],
    is_error: bool,
    emit: AgentEventSink,
) -> ToolResultMessage:
    await _emit_event(
        emit,
        ToolExecutionEndEvent(
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
            result=result,
            is_error=is_error,
        ),
    )

    tool_result_message = ToolResultMessage(
        tool_call_id=tool_call.id,
        tool_name=tool_call.name,
        content=result.content,
        details=result.details,
        is_error=is_error,
        timestamp=now_ms(),
    )

    await _emit_event(emit, MessageStartEvent(message=tool_result_message))
    await _emit_event(emit, MessageEndEvent(message=tool_result_message))
    return tool_result_message
