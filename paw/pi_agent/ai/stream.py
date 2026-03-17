from __future__ import annotations

from typing import Any, Mapping

from .event_stream import AssistantMessageEventStream
from .types import AssistantMessage, Context, Model, OpenAICompletionsOptions, SimpleStreamOptions, StreamOptions
from .providers.openai_completions import astream_openai_completions, astream_simple_openai_completions


def astream(
    model: Model,
    context: Context,
    options: OpenAICompletionsOptions | StreamOptions | Mapping[str, Any] | None = None,
) -> AssistantMessageEventStream:
    if model.api != "openai-completions":
        raise ValueError(f"Unsupported api: {model.api}")
    return astream_openai_completions(model, context, options)


async def acomplete(
    model: Model,
    context: Context,
    options: OpenAICompletionsOptions | StreamOptions | Mapping[str, Any] | None = None,
) -> AssistantMessage:
    return await astream(model, context, options).result()


def astream_simple(
    model: Model,
    context: Context,
    options: SimpleStreamOptions | Mapping[str, Any] | None = None,
) -> AssistantMessageEventStream:
    if model.api != "openai-completions":
        raise ValueError(f"Unsupported api: {model.api}")
    return astream_simple_openai_completions(model, context, options)


async def acomplete_simple(
    model: Model,
    context: Context,
    options: SimpleStreamOptions | Mapping[str, Any] | None = None,
) -> AssistantMessage:
    return await astream_simple(model, context, options).result()
