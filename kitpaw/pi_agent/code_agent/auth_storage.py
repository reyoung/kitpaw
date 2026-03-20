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
        self._runtime_keys: dict[str, str] = {}

    @classmethod
    def create(cls, path: str | Path) -> "AuthStorage":
        return cls(Path(path))

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}

    def get_api_key(self, provider: str) -> str | None:
        # Check in-memory runtime keys first (set via CLI --api-key).
        runtime_key = self._runtime_keys.get(provider)
        if runtime_key is not None:
            return runtime_key
        data = self._read()
        credential = data.get(provider)
        if isinstance(credential, dict) and credential.get("type") == "api_key":
            return credential.get("key")
        return get_env_api_key(provider)

    def set_runtime_api_key(self, provider: str, key: str) -> None:
        """Store an API key in memory only. Not persisted to disk and not
        leaked to child processes via environment variables."""
        self._runtime_keys[provider] = key

    def set_api_key(self, provider: str, key: str) -> None:
        data = self._read()
        data[provider] = {"type": "api_key", "key": key}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        import os
        import stat
        self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        # Restrict file permissions to owner-only (0o600).
        try:
            os.chmod(self.path, stat.S_IRUSR | stat.S_IWUSR)
        except OSError:
            pass
