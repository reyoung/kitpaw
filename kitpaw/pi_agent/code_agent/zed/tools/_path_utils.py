from __future__ import annotations

from pathlib import Path


def resolve_safe_path(cwd: str, path: str) -> Path:
    """Resolve *path* relative to *cwd* and reject traversal outside the workspace.

    If *path* starts with the basename of *cwd* (the worktree root name that
    appears in the Zed system prompt), that prefix is stripped automatically so
    that both ``myproject/src/foo.py`` and ``src/foo.py`` resolve correctly.

    Raises :class:`ValueError` when the resolved path escapes *cwd*.
    """
    root = Path(cwd).resolve()
    root_name = root.name

    # Strip leading worktree root name if present (e.g. "kitpaw/src/foo.py" → "src/foo.py")
    parts = Path(path).parts
    if parts and parts[0] == root_name:
        path = str(Path(*parts[1:])) if len(parts) > 1 else "."

    resolved = (root / path).resolve()
    if not (resolved == root or str(resolved).startswith(str(root) + "/")):
        raise ValueError(
            f"Path '{path}' resolves to '{resolved}' which is outside the project root '{root}'"
        )
    return resolved
