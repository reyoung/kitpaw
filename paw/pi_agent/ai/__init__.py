from __future__ import annotations

from .env_api_keys import get_env_api_key
from .event_stream import AssistantMessageEventStream
from .local_env import load_local_env
from .models import get_model
from .stream import complete, complete_simple, stream, stream_simple
from .types import (
    AssistantMessage,
    AssistantMessageEvent,
    Context,
    ImageContent,
    Model,
    OpenAICompletionsCompat,
    OpenAICompletionsOptions,
    SimpleStreamOptions,
    StreamOptions,
    TextContent,
    ThinkingContent,
    Tool,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)

load_local_env()

__all__ = [
    "AssistantMessage",
    "AssistantMessageEvent",
    "AssistantMessageEventStream",
    "Context",
    "ImageContent",
    "Model",
    "OpenAICompletionsCompat",
    "OpenAICompletionsOptions",
    "SimpleStreamOptions",
    "StreamOptions",
    "TextContent",
    "ThinkingContent",
    "Tool",
    "ToolCall",
    "ToolResultMessage",
    "UserMessage",
    "complete",
    "complete_simple",
    "get_env_api_key",
    "get_model",
    "load_local_env",
    "stream",
    "stream_simple",
]
