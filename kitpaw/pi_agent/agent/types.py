from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Generic, Literal, Protocol, TypeAlias, TypeVar

from ..ai.event_stream import AssistantMessageEventStream
from ..ai.types import (
    AssistantMessage,
    AssistantMessageEvent,
    Context,
    ImageContent,
    Message,
    Model,
    SimpleStreamOptions,
    TextContent,
    ToolCall,
    ToolResultMessage,
)

ThinkingLevel: TypeAlias = Literal["off", "minimal", "low", "medium", "high", "xhigh"]
ToolExecutionMode: TypeAlias = Literal["sequential", "parallel"]


class AgentMessageLike(Protocol):
    """Structural app-level message protocol.

    Any object with a ``role`` can be stored in agent state and emitted via
    agent events. That does not make it LLM-compatible by itself; callers must
    provide ``convert_to_llm`` when custom messages should reach the model.
    """

    role: str


# AgentMessage is intentionally broader than ai.Message so higher layers can
# keep custom app messages in state. Only convert_to_llm decides which of these
# messages are filtered out, passed through, or expanded into ai.Message values.
AgentMessage: TypeAlias = Message | AgentMessageLike
AgentToolCall: TypeAlias = ToolCall

TDetails = TypeVar("TDetails")
TParameters = TypeVar("TParameters")


def _default_model() -> Model:
    from ..ai.local_env import load_local_env
    from ..ai.models import get_model

    load_local_env()
    return get_model("openai", "gpt-4o-mini")


@dataclass(slots=True)
class AgentToolResult(Generic[TDetails]):
    content: list[TextContent | ImageContent]
    details: TDetails


AgentToolUpdateCallback: TypeAlias = Callable[[AgentToolResult[Any]], None]


@dataclass(slots=True)
class AgentTool(Generic[TParameters, TDetails]):
    name: str
    label: str
    description: str
    parameters: dict[str, Any]
    execute: Callable[
        [str, TParameters, asyncio.Event | None, AgentToolUpdateCallback | None],
        AgentToolResult[TDetails] | Awaitable[AgentToolResult[TDetails]],
    ]


@dataclass(slots=True)
class AgentContext:
    system_prompt: str = ""
    messages: list[AgentMessage] = field(default_factory=list)
    tools: list[AgentTool[Any, Any]] | None = None


@dataclass(slots=True)
class BeforeToolCallResult:
    block: bool = False
    reason: str | None = None


@dataclass(slots=True)
class AfterToolCallResult:
    content: list[TextContent | ImageContent] | None = None
    details: Any | None = None
    is_error: bool | None = None


@dataclass(slots=True)
class BeforeToolCallContext:
    assistant_message: AssistantMessage
    tool_call: AgentToolCall
    args: Any
    context: AgentContext


@dataclass(slots=True)
class AfterToolCallContext:
    assistant_message: AssistantMessage
    tool_call: AgentToolCall
    args: Any
    result: AgentToolResult[Any]
    is_error: bool
    context: AgentContext


@dataclass(slots=True)
class AgentState:
    system_prompt: str = ""
    model: Model = field(default_factory=_default_model)
    thinking_level: ThinkingLevel = "off"
    tools: list[AgentTool[Any, Any]] = field(default_factory=list)
    messages: list[AgentMessage] = field(default_factory=list)
    is_streaming: bool = False
    stream_message: AgentMessage | None = None
    pending_tool_calls: set[str] = field(default_factory=set)
    error: str | None = None


@dataclass(slots=True, kw_only=True)
class AgentLoopConfig:
    """Low-level agent loop configuration.

    ``convert_to_llm`` receives the full AgentMessage list before each model
    call. It is a batch transform on purpose: one app message may map to zero,
    one, or many ai.Message values depending on the caller's needs.
    """

    model: Model
    # Custom AgentMessageLike values stay in state unless this function maps
    # them to standard ai.Message objects. The default converter is conservative
    # and only passes through built-in ai message dataclasses.
    convert_to_llm: Callable[[list[AgentMessage]], list[Message] | Awaitable[list[Message]]]
    transform_context: Callable[
        [list[AgentMessage], asyncio.Event | None],
        Awaitable[list[AgentMessage]] | list[AgentMessage],
    ] | None = None
    get_api_key: Callable[[str], str | Awaitable[str | None] | None] | None = None
    get_steering_messages: Callable[[], list[AgentMessage] | Awaitable[list[AgentMessage]]] | None = None
    get_follow_up_messages: Callable[[], list[AgentMessage] | Awaitable[list[AgentMessage]]] | None = None
    tool_execution: ToolExecutionMode = "parallel"
    before_tool_call: Callable[
        [BeforeToolCallContext, asyncio.Event | None],
        BeforeToolCallResult | Awaitable[BeforeToolCallResult | None] | None,
    ] | None = None
    after_tool_call: Callable[
        [AfterToolCallContext, asyncio.Event | None],
        AfterToolCallResult | Awaitable[AfterToolCallResult | None] | None,
    ] | None = None
    stream_options: SimpleStreamOptions | dict[str, Any] | None = None


@dataclass(slots=True, kw_only=True)
class AgentOptions:
    initial_state: dict[str, Any] | None = None
    # Required when callers want custom AgentMessageLike values to affect model
    # input. This is a batch transform, not a per-message encoder.
    convert_to_llm: Callable[[list[AgentMessage]], list[Message] | Awaitable[list[Message]]] | None = None
    transform_context: Callable[
        [list[AgentMessage], asyncio.Event | None],
        Awaitable[list[AgentMessage]] | list[AgentMessage],
    ] | None = None
    steering_mode: Literal["all", "one-at-a-time"] = "one-at-a-time"
    follow_up_mode: Literal["all", "one-at-a-time"] = "one-at-a-time"
    stream_fn: Callable[
        [Model, Context, SimpleStreamOptions | dict[str, Any] | None],
        AssistantMessageEventStream | Awaitable[AssistantMessageEventStream],
    ] | None = None
    get_api_key: Callable[[str], str | Awaitable[str | None] | None] | None = None
    stream_options: SimpleStreamOptions | dict[str, Any] | None = None
    tool_execution: ToolExecutionMode = "parallel"
    before_tool_call: Callable[
        [BeforeToolCallContext, asyncio.Event | None],
        BeforeToolCallResult | Awaitable[BeforeToolCallResult | None] | None,
    ] | None = None
    after_tool_call: Callable[
        [AfterToolCallContext, asyncio.Event | None],
        AfterToolCallResult | Awaitable[AfterToolCallResult | None] | None,
    ] | None = None


@dataclass(slots=True)
class AgentStartEvent:
    type: Literal["agent_start"] = "agent_start"


@dataclass(slots=True)
class AgentEndEvent:
    messages: list[AgentMessage]
    type: Literal["agent_end"] = "agent_end"


@dataclass(slots=True)
class TurnStartEvent:
    type: Literal["turn_start"] = "turn_start"


@dataclass(slots=True)
class TurnEndEvent:
    message: AgentMessage
    tool_results: list[ToolResultMessage]
    type: Literal["turn_end"] = "turn_end"


@dataclass(slots=True)
class MessageStartEvent:
    message: AgentMessage
    type: Literal["message_start"] = "message_start"


@dataclass(slots=True)
class MessageUpdateEvent:
    message: AgentMessage
    assistant_message_event: AssistantMessageEvent
    type: Literal["message_update"] = "message_update"


@dataclass(slots=True)
class MessageEndEvent:
    message: AgentMessage
    type: Literal["message_end"] = "message_end"


@dataclass(slots=True)
class ToolExecutionStartEvent:
    tool_call_id: str
    tool_name: str
    args: Any
    type: Literal["tool_execution_start"] = "tool_execution_start"


@dataclass(slots=True)
class ToolExecutionUpdateEvent:
    tool_call_id: str
    tool_name: str
    args: Any
    partial_result: Any
    type: Literal["tool_execution_update"] = "tool_execution_update"


@dataclass(slots=True)
class ToolExecutionEndEvent:
    tool_call_id: str
    tool_name: str
    result: Any
    is_error: bool
    type: Literal["tool_execution_end"] = "tool_execution_end"


AgentEvent: TypeAlias = (
    AgentStartEvent
    | AgentEndEvent
    | TurnStartEvent
    | TurnEndEvent
    | MessageStartEvent
    | MessageUpdateEvent
    | MessageEndEvent
    | ToolExecutionStartEvent
    | ToolExecutionUpdateEvent
    | ToolExecutionEndEvent
)
