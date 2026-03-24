from __future__ import annotations

from pathlib import Path

# Load Codex prompt.md verbatim from the ref/ tree.
_PROMPT_MD = Path(__file__).resolve().parent.parent.parent.parent.parent / "ref" / "codex" / "codex-rs" / "core" / "prompt.md"

_CACHED_PROMPT: str | None = None


def _load_prompt() -> str:
    global _CACHED_PROMPT
    if _CACHED_PROMPT is not None:
        return _CACHED_PROMPT
    if _PROMPT_MD.exists():
        _CACHED_PROMPT = _PROMPT_MD.read_text(encoding="utf-8")
        return _CACHED_PROMPT
    # Fallback if ref/ is absent (installed package without ref tree).
    _CACHED_PROMPT = (
        "You are a coding agent running in the Codex CLI, a terminal-based coding assistant. "
        "Codex CLI is an open source project led by OpenAI. You are expected to be precise, safe, and helpful."
    )
    return _CACHED_PROMPT


def build_codex_system_prompt(
    available_tools: list[str],
    cwd: str,
    project_rules: list[tuple[str, str]] | None = None,
) -> str:
    """Return the Codex system prompt verbatim from ``prompt.md``.

    *project_rules* are intentionally ignored — in Codex mode AGENTS.md
    content is sent as a separate user message, not merged into the
    system prompt.
    """
    return _load_prompt()
