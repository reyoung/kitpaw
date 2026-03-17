from __future__ import annotations

from typing import Any, Mapping

from .event_stream import AssistantMessageEventStream
from .types import AssistantMessage, Context, Model, OpenAICompletionsOptions, SimpleStreamOptions, StreamOptions
from .providers.openai_completions import stream_openai_completions, stream_simple_openai_completions


def stream(
    model: Model,
    context: Context,
    options: OpenAICompletionsOptions | StreamOptions | Mapping[str, Any] | None = None,
) -> AssistantMessageEventStream:
    if model.api != "openai-completions":
        raise ValueError(f"Unsupported api: {model.api}")
    return stream_openai_completions(model, context, options)


def complete(
    model: Model,
    context: Context,
    options: OpenAICompletionsOptions | StreamOptions | Mapping[str, Any] | None = None,
) -> AssistantMessage:
    return stream(model, context, options).result()


def stream_simple(
    model: Model,
    context: Context,
    options: SimpleStreamOptions | Mapping[str, Any] | None = None,
) -> AssistantMessageEventStream:
    if model.api != "openai-completions":
        raise ValueError(f"Unsupported api: {model.api}")
    return stream_simple_openai_completions(model, context, options)


def complete_simple(
    model: Model,
    context: Context,
    options: SimpleStreamOptions | Mapping[str, Any] | None = None,
) -> AssistantMessage:
    return stream_simple(model, context, options).result()
