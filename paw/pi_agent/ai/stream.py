from __future__ import annotations

from typing import Any, Mapping

from .event_stream import AssistantMessageEventStream
from .types import AssistantMessage, Context, Model, SimpleStreamOptions
from .providers.openai_completions import stream_openai_completions


def stream(
    model: Model,
    context: Context,
    options: SimpleStreamOptions | Mapping[str, Any] | None = None,
) -> AssistantMessageEventStream:
    if model.api != "openai-completions":
        raise ValueError(f"Unsupported api: {model.api}")
    return stream_openai_completions(model, context, options)


async def complete(
    model: Model,
    context: Context,
    options: SimpleStreamOptions | Mapping[str, Any] | None = None,
) -> AssistantMessage:
    return await stream(model, context, options).result()
