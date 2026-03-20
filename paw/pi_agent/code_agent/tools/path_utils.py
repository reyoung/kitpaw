from __future__ import annotations

from pathlib import Path


class PathTraversalError(ValueError):
    """Raised when a resolved path escapes the working directory."""


def resolve_to_cwd(path: str, cwd: str) -> str:
    cwd_resolved = Path(cwd).resolve()
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = cwd_resolved / candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(cwd_resolved)
    except ValueError:
        raise PathTraversalError(
            f"Path '{path}' resolves to '{resolved}' which is outside the working directory '{cwd_resolved}'"
        )
    return str(resolved)


def resolve_read_path(path: str, cwd: str) -> str:
    return resolve_to_cwd(path, cwd)
