from __future__ import annotations

import re
import subprocess
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version as pkg_version
from pathlib import Path

try:
    from ._generated_version import version as _generated_version
except ImportError:
    _generated_version = None

_EXACT_TAG_RE = re.compile(r"^v(?P<base>\d+\.\d+\.\d+)$")
_EXACT_TAG_DIRTY_RE = re.compile(r"^v(?P<base>\d+\.\d+\.\d+)-dirty$")
_DESCRIBE_RE = re.compile(r"^v(?P<base>\d+\.\d+\.\d+)-(?P<count>\d+)-g(?P<sha>[0-9a-f]+)(?P<dirty>-dirty)?$")
_SHA_RE = re.compile(r"^(?P<sha>[0-9a-f]+)(?P<dirty>-dirty)?$")


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _run_git(args: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=_repo_root(),
            capture_output=True,
            text=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def normalize_git_describe(describe: str, *, short_sha: str | None = None) -> str:
    if match := _EXACT_TAG_RE.fullmatch(describe):
        return match.group("base")
    if match := _EXACT_TAG_DIRTY_RE.fullmatch(describe):
        sha = short_sha or _run_git(["rev-parse", "--short", "HEAD"]) or "unknown"
        return f"{match.group('base')}+g{sha}.dirty"
    if match := _DESCRIBE_RE.fullmatch(describe):
        suffix = f"+{match.group('count')}.g{match.group('sha')}"
        if match.group("dirty"):
            suffix += ".dirty"
        return f"{match.group('base')}{suffix}"
    if match := _SHA_RE.fullmatch(describe):
        suffix = f"0+g{match.group('sha')}"
        if match.group("dirty"):
            suffix += ".dirty"
        return suffix
    raise ValueError(f"Unsupported git describe output: {describe}")


@lru_cache(maxsize=1)
def get_version() -> str:
    describe = _run_git(["describe", "--tags", "--dirty", "--always", "--match", "v[0-9]*"])
    if describe is not None:
        short_sha = _run_git(["rev-parse", "--short", "HEAD"])
        return normalize_git_describe(describe, short_sha=short_sha)
    if _generated_version:
        return str(_generated_version)
    try:
        return pkg_version("kitpaw")
    except PackageNotFoundError:
        return "0+unknown"


__all__ = ["get_version", "normalize_git_describe"]
