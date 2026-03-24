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


def _load_file(path: Path) -> None:
    """Load KEY=value pairs from *path* into ``os.environ`` (setdefault)."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        parsed = _parse_env_line(line)
        if not parsed:
            continue
        key, value = parsed
        os.environ.setdefault(key, value)


def kitpaw_env_files() -> list[Path]:
    """Return the ordered list of ``.kitpaw`` files that will be loaded.

    Order: repo-root first, then CWD (CWD values take precedence because
    ``os.environ.setdefault`` is used — the *first* file to set a key wins,
    so CWD is loaded **after** repo-root only when we want CWD to override.
    To let CWD win we load it *first*).
    """
    cwd = Path.cwd().resolve()
    root = repo_root()
    paths: list[Path] = []
    cwd_file = cwd / ".kitpaw"
    root_file = root / ".kitpaw"
    # CWD first so its values take precedence (setdefault keeps first writer).
    if cwd_file.exists():
        paths.append(cwd_file)
    if root_file.exists() and root_file != cwd_file:
        paths.append(root_file)
    return paths


def load_local_env(force: bool = False) -> None:
    global _LOADED

    if _LOADED and not force:
        return

    # Load CWD/.kitpaw first so it takes precedence over repo-root/.kitpaw
    # (os.environ.setdefault keeps whichever value is set first).
    for path in kitpaw_env_files():
        _load_file(path)

    _LOADED = True
