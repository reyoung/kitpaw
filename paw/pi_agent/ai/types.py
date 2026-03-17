from __future__ import annotations

from dataclasses import dataclass, field
from time import time
from typing import Any, Awaitable, Callable, Literal, TypeAlias

Api: TypeAlias = str
Provider: TypeAlias = str
ThinkingLevel: TypeAlias = Literal["minimal", "low", "medium", "high", "xhigh"]
StopReason: TypeAlias = Literal["stop", "length", "toolUse", "error", "aborted"]
ToolChoice: TypeAlias = Literal["auto", "none", "required"] | dict[str, Any]
PayloadOverride: TypeAlias = dict[str, Any] | None | Awaitable[dict[str, Any] | None]
PayloadTransform: TypeAlias = Callable[[dict[str, Any], "Model"], PayloadOverride]


def now_ms() -> int:
    return int(time() * 1000)


@dataclass(slots=True)
class Cost:
    input: float = 0.0
    output: float = 0.0
    cache_read: float = 0.0
    cache_write: float = 0.0
    total: float = 0.0


@dataclass(slots=True)
class Usage:
    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_write: int = 0
    total_tokens: int = 0
    cost: Cost = field(default_factory=Cost)


@dataclass(slots=True)
class TextContent:
    text: str
    type: Literal["text"] = "text"
    text_signature: str | None = None


@dataclass(slots=True)
class ThinkingContent:
    thinking: str
    type: Literal["thinking"] = "thinking"
    thinking_signature: str | None = None
    redacted: bool = False


@dataclass(slots=True)
class ImageContent:
    data: str
    mime_type: str
    type: Literal["image"] = "image"


@dataclass(slots=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    type: Literal["toolCall"] = "toolCall"
    thought_signature: str | None = None


UserContentBlock: TypeAlias = TextContent | ImageContent
AssistantContentBlock: TypeAlias = TextContent | ThinkingContent | ToolCall
ToolResultContentBlock: TypeAlias = TextContent | ImageContent


@dataclass(slots=True)
class UserMessage:
    content: str | list[UserContentBlock]
    timestamp: int = field(default_factory=now_ms)
    role: Literal["user"] = "user"


@dataclass(slots=True)
class AssistantMessage:
    api: Api
    provider: Provider
    model: str
    content: list[AssistantContentBlock] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    stop_reason: StopReason = "stop"
    error_message: str | None = None
    timestamp: int = field(default_factory=now_ms)
    role: Literal["assistant"] = "assistant"


@dataclass(slots=True)
class ToolResultMessage:
    tool_call_id: str
    tool_name: str
    content: list[ToolResultContentBlock]
    is_error: bool = False
    details: Any | None = None
    timestamp: int = field(default_factory=now_ms)
    role: Literal["toolResult"] = "toolResult"


Message: TypeAlias = UserMessage | AssistantMessage | ToolResultMessage


@dataclass(slots=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]


@dataclass(slots=True)
class Context:
    messages: list[Message]
    system_prompt: str | None = None
    tools: list[Tool] | None = None


@dataclass(slots=True)
class OpenAICompletionsCompat:
    supports_store: bool = True
    supports_developer_role: bool = True
    supports_reasoning_effort: bool = True
    reasoning_effort_map: dict[ThinkingLevel, str] = field(default_factory=dict)
    supports_usage_in_streaming: bool = True
    max_tokens_field: Literal["max_completion_tokens", "max_tokens"] = "max_completion_tokens"
    requires_tool_result_name: bool = False
    requires_assistant_after_tool_result: bool = False
    requires_thinking_as_text: bool = False
    thinking_format: Literal["openai", "zai", "qwen", "qwen-chat-template"] = "openai"
    supports_strict_mode: bool = True


@dataclass(slots=True)
class ModelCost:
    input: float = 0.0
    output: float = 0.0
    cache_read: float = 0.0
    cache_write: float = 0.0


@dataclass(slots=True)
class Model:
    id: str
    name: str
    api: Api
    provider: Provider
    base_url: str
    reasoning: bool
    input: list[Literal["text", "image"]]
    cost: ModelCost
    context_window: int
    max_tokens: int
    headers: dict[str, str] = field(default_factory=dict)
    compat: OpenAICompletionsCompat | None = None


@dataclass(slots=True)
class StreamOptions:
    temperature: float | None = None
    max_tokens: int | None = None
    api_key: str | None = None
    headers: dict[str, str] | None = None
    on_payload: PayloadTransform | None = None


@dataclass(slots=True)
class SimpleStreamOptions(StreamOptions):
    reasoning: ThinkingLevel | None = None


@dataclass(slots=True)
class OpenAICompletionsOptions(StreamOptions):
    tool_choice: ToolChoice | None = None
    reasoning_effort: ThinkingLevel | None = None


@dataclass(slots=True)
class StartEvent:
    partial: AssistantMessage
    type: Literal["start"] = "start"


@dataclass(slots=True)
class TextStartEvent:
    content_index: int
    partial: AssistantMessage
    type: Literal["text_start"] = "text_start"


@dataclass(slots=True)
class TextDeltaEvent:
    content_index: int
    delta: str
    partial: AssistantMessage
    type: Literal["text_delta"] = "text_delta"


@dataclass(slots=True)
class TextEndEvent:
    content_index: int
    content: str
    partial: AssistantMessage
    type: Literal["text_end"] = "text_end"


@dataclass(slots=True)
class ThinkingStartEvent:
    content_index: int
    partial: AssistantMessage
    type: Literal["thinking_start"] = "thinking_start"


@dataclass(slots=True)
class ThinkingDeltaEvent:
    content_index: int
    delta: str
    partial: AssistantMessage
    type: Literal["thinking_delta"] = "thinking_delta"


@dataclass(slots=True)
class ThinkingEndEvent:
    content_index: int
    content: str
    partial: AssistantMessage
    type: Literal["thinking_end"] = "thinking_end"


@dataclass(slots=True)
class ToolCallStartEvent:
    content_index: int
    partial: AssistantMessage
    type: Literal["toolcall_start"] = "toolcall_start"


@dataclass(slots=True)
class ToolCallDeltaEvent:
    content_index: int
    delta: str
    partial: AssistantMessage
    type: Literal["toolcall_delta"] = "toolcall_delta"


@dataclass(slots=True)
class ToolCallEndEvent:
    content_index: int
    tool_call: ToolCall
    partial: AssistantMessage
    type: Literal["toolcall_end"] = "toolcall_end"


@dataclass(slots=True)
class DoneEvent:
    reason: Literal["stop", "length", "toolUse"]
    message: AssistantMessage
    type: Literal["done"] = "done"


@dataclass(slots=True)
class ErrorEvent:
    reason: Literal["error", "aborted"]
    error: AssistantMessage
    type: Literal["error"] = "error"


AssistantMessageEvent: TypeAlias = (
    StartEvent
    | TextStartEvent
    | TextDeltaEvent
    | TextEndEvent
    | ThinkingStartEvent
    | ThinkingDeltaEvent
    | ThinkingEndEvent
    | ToolCallStartEvent
    | ToolCallDeltaEvent
    | ToolCallEndEvent
    | DoneEvent
    | ErrorEvent
)
