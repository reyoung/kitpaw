from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..ai.env_api_keys import get_env_api_key


@dataclass(slots=True)
class ApiKeyCredential:
    type: str
    key: str


class AuthStorage:
    def __init__(self, path: Path) -> None:
        self.path = path

    @classmethod
    def create(cls, path: str | Path) -> "AuthStorage":
        return cls(Path(path))

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def get_api_key(self, provider: str) -> str | None:
        data = self._read()
        credential = data.get(provider)
        if isinstance(credential, dict) and credential.get("type") == "api_key":
            return credential.get("key")
        return get_env_api_key(provider)

    def set_api_key(self, provider: str, key: str) -> None:
        data = self._read()
        data[provider] = {"type": "api_key", "key": key}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")
