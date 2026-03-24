from __future__ import annotations


def configure_zed_compaction(session) -> None:
    """Disable automatic compaction for the Zed agent.

    Zed does not perform automatic context compaction.  Summaries are
    generated on-demand instead.  This helper disables the auto-compact
    behaviour after a session has been created.
    """
    session.set_compaction_enabled(False)
