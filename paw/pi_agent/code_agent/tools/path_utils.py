from __future__ import annotations

from pathlib import Path


def resolve_to_cwd(path: str, cwd: str) -> str:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = Path(cwd) / candidate
    return str(candidate.resolve())


def resolve_read_path(path: str, cwd: str) -> str:
    return resolve_to_cwd(path, cwd)
