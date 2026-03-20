from __future__ import annotations

from dataclasses import replace

from .env_api_keys import get_env_base_url
from .types import Model, ModelCost, Usage


def _build_openai_model(
    model_id: str,
    *,
    name: str | None = None,
    reasoning: bool = True,
    input_modalities: list[str] | None = None,
    context_window: int = 128_000,
    max_tokens: int = 16_384,
) -> Model:
    return Model(
        id=model_id,
        name=name or model_id,
        api="openai-completions",
        provider="openai",
        base_url=get_env_base_url("openai"),
        reasoning=reasoning,
        input=list(input_modalities or ["text", "image"]),
        cost=ModelCost(),
        context_window=context_window,
        max_tokens=max_tokens,
    )


_KNOWN_OPENAI_MODELS: dict[str, Model] = {
    "gpt-4o-mini": _build_openai_model("gpt-4o-mini", name="GPT-4o mini"),
    "gpt-4o": _build_openai_model("gpt-4o", name="GPT-4o"),
}


def get_model(provider: str, model_id: str) -> Model:
    if provider != "openai":
        raise ValueError(f"Unsupported provider: {provider}")

    known = _KNOWN_OPENAI_MODELS.get(model_id)
    if known is not None:
        return replace(known, base_url=get_env_base_url("openai"))

    return _build_openai_model(model_id)


def calculate_cost(model: Model, usage: Usage) -> Usage:
    usage.cost.input = (model.cost.input / 1_000_000) * usage.input
    usage.cost.output = (model.cost.output / 1_000_000) * usage.output
    usage.cost.cache_read = (model.cost.cache_read / 1_000_000) * usage.cache_read
    usage.cost.cache_write = (model.cost.cache_write / 1_000_000) * usage.cache_write
    usage.cost.total = (
        usage.cost.input + usage.cost.output + usage.cost.cache_read + usage.cost.cache_write
    )
    return usage


def supports_xhigh(model: Model) -> bool:
    return "gpt-5.2" in model.id or "gpt-5.3" in model.id or "gpt-5.4" in model.id
