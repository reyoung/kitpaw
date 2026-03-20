from __future__ import annotations

import os
from pathlib import Path

_LOADED = False


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _parse_env_line(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or "=" not in stripped:
        return None

    key, raw_value = stripped.split("=", 1)
    key = key.strip()
    value = raw_value.strip()
    if not key:
        return None

    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]

    if value.startswith("${") and value.endswith("}"):
        value = os.getenv(value[2:-1], "")

    return key, value


def load_local_env(force: bool = False) -> None:
    global _LOADED

    if _LOADED and not force:
        return

    for filename in (".env.local", ".env"):
        path = repo_root() / filename
        if not path.exists():
            continue

        for line in path.read_text(encoding="utf-8").splitlines():
            parsed = _parse_env_line(line)
            if not parsed:
                continue
            key, value = parsed
            os.environ.setdefault(key, value)

    _LOADED = True
