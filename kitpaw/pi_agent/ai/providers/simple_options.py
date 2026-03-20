from __future__ import annotations

from typing import Literal

from ..types import Model, SimpleStreamOptions, StreamOptions, ThinkingLevel


def build_base_options(
    model: Model, options: SimpleStreamOptions | None = None, api_key: str | None = None
) -> StreamOptions:
    return StreamOptions(
        temperature=options.temperature if options else None,
        max_tokens=(
            options.max_tokens if options and options.max_tokens else min(model.max_tokens, 32_000)
        ),
        api_key=api_key or (options.api_key if options else None),
        headers=options.headers if options else None,
        on_payload=options.on_payload if options else None,
    )


def clamp_reasoning(
    effort: ThinkingLevel | None,
) -> Literal["minimal", "low", "medium", "high"] | None:
    if effort == "xhigh":
        return "high"
    return effort
