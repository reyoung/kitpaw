from __future__ import annotations

from dataclasses import replace

from ..ai.env_api_keys import get_env_default_model, get_env_fallback_model
from ..ai.models import get_model
from ..ai.types import Model
from .auth_storage import AuthStorage


class ModelRegistry:
    def __init__(self, auth_storage: AuthStorage) -> None:
        self.auth_storage = auth_storage

    def find(self, provider: str, model_id: str) -> Model:
        return get_model(provider, model_id)

    def list_models(self) -> list[Model]:
        names: list[str] = []
        for model_name in (get_env_default_model(), get_env_fallback_model(), "gpt-4o-mini", "gpt-4o"):
            if model_name and model_name not in names:
                names.append(model_name)
        return [replace(get_model("openai", name)) for name in names]

    async def get_api_key_for_provider(self, provider: str) -> str | None:
        return self.auth_storage.get_api_key(provider)
