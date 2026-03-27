from __future__ import annotations

from typing import Any

from ...ai import UserMessage
from ...ai.types import TextContent
from ..summarizer import generate_summary
from ..types import CompactionPreparation, CompactionResult

CODEX_COMPACTION_PROMPT = (
    "You are performing a CONTEXT CHECKPOINT COMPACTION. Create a handoff summary for another LLM "
    "that will resume the task.\n\n"
    "Include:\n"
    "- Current progress and key decisions made\n"
    "- Important context, constraints, or user preferences\n"
    "- What remains to be done (clear next steps)\n"
    "- Any critical data, examples, or references needed to continue\n\n"
    "Be concise, structured, and focused on helping the next LLM seamlessly continue the work."
)

CODEX_SUMMARY_PREFIX = (
    "Another language model started to solve this problem and produced a summary of its thinking "
    "process. You also have access to the state of the tools that were used by that language model. "
    "Use this to build on the work that has already been done and avoid duplicating work. Here is "
    "the summary produced by the other language model, use the information in this summary to "
    "assist with your own analysis:"
)

COMPACT_USER_MESSAGE_MAX_TOKENS = 20_000


def _approx_token_count(text: str) -> int:
    return max(len(text) // 4, 1) if text else 0


def is_summary_message(text: str) -> bool:
    """Return True if *text* is a compaction summary produced by a previous run."""
    return text.startswith(f"{CODEX_SUMMARY_PREFIX}\n")


def _extract_message_text(message: Any) -> str | None:
    """Best-effort extraction of text content from various message types."""
    content = getattr(message, "content", None)
    if content is None:
        return None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            text = getattr(block, "text", None)
            if text:
                parts.append(text)
        return "\n".join(parts) if parts else None
    return None


def collect_user_messages(messages: list[Any]) -> list[str]:
    """Collect user message texts, filtering out previous compaction summaries."""
    result: list[str] = []
    for message in messages:
        role = getattr(message, "role", None)
        if role != "user":
            continue
        text = _extract_message_text(message)
        if text is None:
            continue
        if is_summary_message(text):
            continue
        result.append(text)
    return result


def build_compacted_history(
    user_messages: list[str],
    summary_text: str,
    max_user_tokens: int = COMPACT_USER_MESSAGE_MAX_TOKENS,
) -> list[UserMessage]:
    """Build replacement history matching Codex Rust's ``build_compacted_history``.

    Selects the most recent user messages (up to *max_user_tokens*) and appends
    a summary message.  Returns a list of ``UserMessage`` objects suitable for
    direct assignment to the agent's message list.
    """
    selected: list[str] = []
    if max_user_tokens > 0:
        remaining = max_user_tokens
        for msg in reversed(user_messages):
            if remaining <= 0:
                break
            tokens = _approx_token_count(msg)
            if tokens <= remaining:
                selected.append(msg)
                remaining -= tokens
            else:
                # Truncate the message to fit the remaining budget.
                char_limit = remaining * 4
                selected.append(msg[:char_limit])
                break
        selected.reverse()

    history: list[UserMessage] = []
    for msg in selected:
        history.append(UserMessage(content=[TextContent(text=msg)]))

    # Append the compaction summary as the final message.
    effective_summary = summary_text if summary_text else "(no summary available)"
    history.append(UserMessage(content=[TextContent(text=effective_summary)]))
    return history


async def codex_compaction_hook(
    preparation: CompactionPreparation,
    *,
    model=None,
) -> CompactionResult | None:
    """Compaction hook that uses the Codex compaction prompt.

    Generates a summary using the Codex-specific compaction instructions,
    then builds a replacement history of recent user messages + summary,
    matching Codex Rust's compaction strategy.

    Returns a ``CompactionResult`` with the enriched summary and replacement
    messages in ``details``, or ``None`` if *model* is not provided.
    """
    if model is None:
        return None

    summary = await generate_summary(
        model,
        preparation.messages_to_summarize,
        custom_instructions=CODEX_COMPACTION_PROMPT,
    )

    summary_text = f"{CODEX_SUMMARY_PREFIX}\n{summary}"

    # Build replacement history: recent user messages + summary.
    user_msgs = collect_user_messages(preparation.messages_to_summarize)
    replacement_messages = build_compacted_history(user_msgs, summary_text)

    return CompactionResult(
        summary=summary_text,
        first_kept_entry_id=preparation.first_kept_entry_id,
        tokens_before=preparation.tokens_before,
        details={"replacement_messages": replacement_messages},
    )


def configure_codex_compaction(session) -> None:
    """Configure Codex-style compaction on *session*.

    Unlike Zed, Codex keeps auto-compact enabled.  The compaction hook is
    set so that summarization uses the Codex compaction prompt and the
    generated summary is prefixed with the Codex handoff preamble.  The hook
    also returns a replacement message list (recent user messages + summary)
    that the session applies directly to the agent's state.
    """

    async def _hook(preparation: CompactionPreparation) -> CompactionResult | None:
        return await codex_compaction_hook(preparation, model=session.model)

    session.set_compaction_hook(_hook)
