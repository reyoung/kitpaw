from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from typing import Any, Callable, Literal

from ..ai.stream import stream as default_stream
from ..ai.types import ImageContent, Model, SimpleStreamOptions, TextContent, UserMessage
from .agent_loop import (
    AgentEndEvent,
    AgentLoopConfig,
    AgentMessage,
    MessageEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    TurnEndEvent,
    _run_agent_loop,
    default_convert_to_llm,
    normalize_simple_stream_options,
)
from .types import AgentContext, AgentOptions, AgentState, AgentTool, ThinkingLevel


def _normalize_agent_options(options: AgentOptions | Mapping[str, Any] | None) -> AgentOptions:
    if options is None:
        return AgentOptions()
    if isinstance(options, AgentOptions):
        return options
    if is_dataclass(options):
        return AgentOptions(**asdict(options))
    if isinstance(options, Mapping):
        return AgentOptions(**dict(options))
    raise TypeError(f"Unsupported agent options type: {type(options)!r}")


def _build_state(initial_state: Mapping[str, Any] | None) -> AgentState:
    state = AgentState()
    if not initial_state:
        return state

    for key, value in initial_state.items():
        if hasattr(state, key):
            setattr(state, key, value)
    return state


class Agent:
    def __init__(self, options: AgentOptions | Mapping[str, Any] | None = None) -> None:
        opts = _normalize_agent_options(options)
        self._state = _build_state(opts.initial_state)
        self.convert_to_llm = opts.convert_to_llm or default_convert_to_llm
        self.transform_context = opts.transform_context
        self.steering_mode = opts.steering_mode
        self.follow_up_mode = opts.follow_up_mode
        self.stream_fn = opts.stream_fn or default_stream
        self.get_api_key = opts.get_api_key
        self.stream_options = opts.stream_options
        self.tool_execution = opts.tool_execution
        self._before_tool_call = opts.before_tool_call
        self._after_tool_call = opts.after_tool_call
        self._listeners: set[Callable[[Any], None]] = set()
        self._running_future: asyncio.Future[None] | None = None
        self._current_task: asyncio.Task[Any] | None = None
        self._cancel_event: asyncio.Event | None = None
        self._steering_queue: list[AgentMessage] = []
        self._follow_up_queue: list[AgentMessage] = []

    @property
    def state(self) -> AgentState:
        return self._state

    def subscribe(self, fn: Callable[[Any], None]) -> Callable[[], None]:
        self._listeners.add(fn)
        return lambda: self._listeners.discard(fn)

    def set_system_prompt(self, value: str) -> None:
        self._state.system_prompt = value

    def set_model(self, model: Model) -> None:
        self._state.model = model

    def set_thinking_level(self, level: ThinkingLevel) -> None:
        self._state.thinking_level = level

    def set_steering_mode(self, mode: Literal["all", "one-at-a-time"]) -> None:
        self.steering_mode = mode

    def get_steering_mode(self) -> Literal["all", "one-at-a-time"]:
        return self.steering_mode

    def set_follow_up_mode(self, mode: Literal["all", "one-at-a-time"]) -> None:
        self.follow_up_mode = mode

    def get_follow_up_mode(self) -> Literal["all", "one-at-a-time"]:
        return self.follow_up_mode

    def set_tools(self, tools: list[AgentTool[Any, Any]]) -> None:
        self._state.tools = tools

    def replace_messages(self, messages: list[AgentMessage]) -> None:
        self._state.messages = list(messages)

    def append_message(self, message: AgentMessage) -> None:
        self._state.messages = [*self._state.messages, message]

    def clear_messages(self) -> None:
        self._state.messages = []

    def steer(self, message: AgentMessage) -> None:
        self._steering_queue.append(message)

    def follow_up(self, message: AgentMessage) -> None:
        self._follow_up_queue.append(message)

    def clear_steering_queue(self) -> None:
        self._steering_queue = []

    def clear_follow_up_queue(self) -> None:
        self._follow_up_queue = []

    def clear_all_queues(self) -> None:
        self._steering_queue = []
        self._follow_up_queue = []

    def has_queued_messages(self) -> bool:
        return bool(self._steering_queue or self._follow_up_queue)

    def abort(self) -> None:
        if self._cancel_event is not None:
            self._cancel_event.set()

    async def wait_for_idle(self) -> None:
        if self._running_future is not None:
            await self._running_future

    def reset(self) -> None:
        self._state.messages = []
        self._state.is_streaming = False
        self._state.stream_message = None
        self._state.pending_tool_calls = set()
        self._state.error = None
        self.clear_all_queues()

    async def prompt(
        self,
        input: str | AgentMessage | Sequence[AgentMessage],
        images: list[ImageContent] | None = None,
    ) -> None:
        """Start a new turn from text or agent messages.

        Custom AgentMessageLike inputs are accepted into agent state as-is. They
        only reach the model if convert_to_llm maps them to standard ai.Message
        objects before the request is sent.
        """

        if self._state.is_streaming:
            raise ValueError(
                "Agent is already processing a prompt. Use steer() or follow_up() to queue messages, or wait for completion.",
            )

        messages = self._normalize_prompt_input(input, images)
        await self._run_with_messages(messages)

    async def continue_(self) -> None:
        if self._state.is_streaming:
            raise ValueError("Agent is already processing. Wait for completion before continuing.")

        if not self._state.messages:
            raise ValueError("No messages to continue from")

        if _agent_message_role(self._state.messages[-1]) == "assistant":
            queued_steering = self._dequeue_steering_messages()
            if queued_steering:
                await self._run_with_messages(queued_steering, skip_initial_steering_poll=True)
                return

            queued_follow_up = self._dequeue_follow_up_messages()
            if queued_follow_up:
                await self._run_with_messages(queued_follow_up)
                return

            context_messages = _strip_trailing_assistant_messages(self._state.messages)
            if not context_messages:
                raise ValueError("Cannot continue from message role: assistant")
            await self._run_with_messages(None, context_messages=context_messages)
            return

        await self._run_with_messages(None, context_messages=list(self._state.messages))

    def _process_loop_event(self, event: Any) -> None:
        if isinstance(event, MessageStartEvent):
            self._state.stream_message = event.message
        elif isinstance(event, MessageUpdateEvent):
            self._state.stream_message = event.message
        elif isinstance(event, MessageEndEvent):
            self._state.stream_message = None
            self.append_message(event.message)
            if _agent_message_role(event.message) == "assistant" and _assistant_error_message(event.message):
                self._state.error = _assistant_error_message(event.message)
        elif isinstance(event, ToolExecutionStartEvent):
            self._state.pending_tool_calls = {*self._state.pending_tool_calls, event.tool_call_id}
        elif isinstance(event, ToolExecutionEndEvent):
            self._state.pending_tool_calls = {
                tool_call_id for tool_call_id in self._state.pending_tool_calls if tool_call_id != event.tool_call_id
            }
        elif isinstance(event, TurnEndEvent):
            if _agent_message_role(event.message) == "assistant" and _assistant_error_message(event.message):
                self._state.error = _assistant_error_message(event.message)
        elif isinstance(event, AgentEndEvent):
            self._state.is_streaming = False
            self._state.stream_message = None

        for listener in list(self._listeners):
            listener(event)

    def _normalize_prompt_input(
        self,
        input: str | AgentMessage | Sequence[AgentMessage],
        images: list[ImageContent] | None = None,
    ) -> list[AgentMessage]:
        # String inputs are normalized here. Custom app messages are left
        # untouched so convert_to_llm can handle them in full context.
        if isinstance(input, str):
            content: list[TextContent | ImageContent] = [TextContent(text=input)]
            if images:
                content.extend(images)
            return [
                UserMessage(
                    content=input if not images else content,
                )
            ]

        if isinstance(input, Sequence) and not hasattr(input, "role"):
            return list(input)

        return [input]

    def _dequeue_steering_messages(self) -> list[AgentMessage]:
        if self.steering_mode == "one-at-a-time":
            if not self._steering_queue:
                return []
            first = self._steering_queue[0]
            self._steering_queue = self._steering_queue[1:]
            return [first]

        steering = list(self._steering_queue)
        self._steering_queue = []
        return steering

    def _dequeue_follow_up_messages(self) -> list[AgentMessage]:
        if self.follow_up_mode == "one-at-a-time":
            if not self._follow_up_queue:
                return []
            first = self._follow_up_queue[0]
            self._follow_up_queue = self._follow_up_queue[1:]
            return [first]

        follow_up = list(self._follow_up_queue)
        self._follow_up_queue = []
        return follow_up

    def _build_stream_options(self) -> SimpleStreamOptions | dict[str, Any] | None:
        options = normalize_simple_stream_options(self.stream_options)
        if self._state.thinking_level != "off":
            options.reasoning = self._state.thinking_level
        return options

    def _build_loop_config(self, skip_initial_steering_poll: bool = False) -> AgentLoopConfig:
        options = self._build_stream_options()

        async def get_steering_messages() -> list[AgentMessage]:
            nonlocal skip_initial_steering_poll
            if skip_initial_steering_poll:
                skip_initial_steering_poll = False
                return []
            return self._dequeue_steering_messages()

        async def get_follow_up_messages() -> list[AgentMessage]:
            return self._dequeue_follow_up_messages()

        return AgentLoopConfig(
            model=self._state.model,
            convert_to_llm=self.convert_to_llm,
            transform_context=self.transform_context,
            get_api_key=self.get_api_key,
            get_steering_messages=get_steering_messages,
            get_follow_up_messages=get_follow_up_messages,
            tool_execution=self.tool_execution,
            before_tool_call=self._before_tool_call,
            after_tool_call=self._after_tool_call,
            stream_options=options,
        )

    async def _run_with_messages(
        self,
        messages: list[AgentMessage] | None,
        *,
        skip_initial_steering_poll: bool = False,
        context_messages: list[AgentMessage] | None = None,
    ) -> None:
        self._running_future = asyncio.get_running_loop().create_future()
        self._current_task = asyncio.current_task()
        self._cancel_event = asyncio.Event()
        self._state.is_streaming = True
        self._state.stream_message = None
        self._state.error = None

        loop_config = self._build_loop_config(skip_initial_steering_poll=skip_initial_steering_poll)
        context = AgentContext(
            system_prompt=self._state.system_prompt,
            messages=list(self._state.messages if context_messages is None else context_messages),
            tools=self._state.tools,
        )

        try:
            await _run_agent_loop(
                messages or [],
                context,
                loop_config,
                self._process_loop_event,
                self._cancel_event,
                self.stream_fn,
            )
        finally:
            self._state.is_streaming = False
            self._state.stream_message = None
            self._state.pending_tool_calls = set()
            self._current_task = None
            self._cancel_event = None
            if self._running_future is not None and not self._running_future.done():
                self._running_future.set_result(None)
            self._running_future = None


def _agent_message_role(message: AgentMessage) -> str | None:
    return getattr(message, "role", None)


def _assistant_error_message(message: AgentMessage) -> str | None:
    return getattr(message, "error_message", None)


def _strip_trailing_assistant_messages(messages: Sequence[AgentMessage]) -> list[AgentMessage]:
    trimmed = list(messages)
    while trimmed and _agent_message_role(trimmed[-1]) == "assistant":
        trimmed.pop()
    return trimmed
