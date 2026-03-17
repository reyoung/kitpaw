from __future__ import annotations

import os

from .local_env import load_local_env

DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"


def get_env_api_key(provider: str = "openai") -> str | None:
    load_local_env()
    if provider != "openai":
        return None
    return os.getenv("OPENAI_API_KEY")


def get_env_base_url(provider: str = "openai") -> str:
    load_local_env()
    if provider != "openai":
        raise ValueError(f"Unsupported provider: {provider}")
    return os.getenv("OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL).rstrip("/")


def get_env_default_model() -> str | None:
    load_local_env()
    return os.getenv("OPENAI_MODEL")


def get_env_fallback_model() -> str | None:
    load_local_env()
    return os.getenv("OPENAI_FALLBACK_MODEL")
