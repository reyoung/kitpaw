from __future__ import annotations

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


async def codex_compaction_hook(
    preparation: CompactionPreparation,
    *,
    model=None,
) -> CompactionResult | None:
    """Compaction hook that uses the Codex compaction prompt.

    Generates a summary using the Codex-specific compaction instructions and
    prepends the Codex summary prefix so the resuming model has proper context.

    Returns a ``CompactionResult`` with the enriched summary, or ``None`` if
    *model* is not provided (falling back to default compaction).
    """
    if model is None:
        return None

    summary = await generate_summary(
        model,
        preparation.messages_to_summarize,
        custom_instructions=CODEX_COMPACTION_PROMPT,
    )

    enriched_summary = f"{CODEX_SUMMARY_PREFIX}\n\n{summary}"

    return CompactionResult(
        summary=enriched_summary,
        first_kept_entry_id=preparation.first_kept_entry_id,
        tokens_before=preparation.tokens_before,
    )


def configure_codex_compaction(session) -> None:
    """Configure Codex-style compaction on *session*.

    Unlike Zed, Codex keeps auto-compact enabled.  The compaction hook is
    set so that summarization uses the Codex compaction prompt and the
    generated summary is prefixed with the Codex handoff preamble.
    """
    session.set_compaction_hook(codex_compaction_hook)
