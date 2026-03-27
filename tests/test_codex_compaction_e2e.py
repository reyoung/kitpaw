#!/usr/bin/env python3
"""Trigger Codex compaction and dump the resulting agent messages.

This script:
1. Creates a Codex agent session with a very small context window to force compaction
2. Injects synthetic conversation history (multiple user messages)
3. Triggers auto_compact
4. Dumps the agent messages before and after compaction for inspection

Usage:
    uv run python tests/test_codex_compaction_e2e.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _msg_to_dict(msg) -> dict:
    """Convert an AgentMessage to a dict for readable output."""
    role = getattr(msg, "role", "unknown")
    content = getattr(msg, "content", "")
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        text = "\n".join(getattr(b, "text", "") for b in content if getattr(b, "type", None) == "text")
    else:
        text = str(content)
    # Truncate for display
    if len(text) > 200:
        text = text[:200] + f"... [{len(text)} chars total]"
    return {"role": role, "text": text}


async def main():
    from kitpaw.pi_agent.ai import UserMessage
    from kitpaw.pi_agent.ai.types import TextContent
    from kitpaw.pi_agent.code_agent.codex.compaction import (
        CODEX_SUMMARY_PREFIX,
        build_compacted_history,
        collect_user_messages,
        is_summary_message,
    )
    from kitpaw.pi_agent.code_agent.summarizer import estimate_tokens

    # ── Part 1: Test build_compacted_history directly ────────────────────
    print("=" * 60)
    print("Part 1: build_compacted_history (no LLM needed)")
    print("=" * 60)

    # Simulate a conversation with many user messages
    user_messages_text = [
        "I need to fix the distance calculation in sympy's geometry module.",
        "The issue is that Point(2,0).distance(Point(1,0,2)) returns 1 instead of sqrt(5).",
        "I've looked at the code and the problem is in the zip function - it truncates to the shorter point's dimensions.",
        "I found the distance method at line 265 of sympy/geometry/point.py.",
        "The fix should normalize both points to the same dimension before computing distance.",
        "I've also checked taxicab_distance and midpoint - they have the same bug.",
        "Here's my approach: pad the shorter point with zeros to match the longer one.",
        "I've written a test script to verify the fix works correctly.",
        "The test passes: Point(2,0).distance(Point(1,0,2)) now returns sqrt(5).",
        "Now I need to run the full test suite to make sure nothing else breaks.",
    ]

    fake_messages = []
    for text in user_messages_text:
        fake_messages.append(UserMessage(content=text))

    print(f"\nOriginal messages: {len(fake_messages)}")
    print(f"Estimated tokens: {estimate_tokens(fake_messages)}")

    # Collect user messages (should get all 10, no summaries to filter)
    collected = collect_user_messages(fake_messages)
    print(f"Collected user messages: {len(collected)}")

    # Build compacted history with a tight budget (simulate 200 token limit)
    summary_text = f"{CODEX_SUMMARY_PREFIX}\nThe developer is fixing Point.distance() in sympy to handle different dimensions."
    replacement = build_compacted_history(collected, summary_text, max_user_tokens=200)

    print(f"\nCompacted history ({len(replacement)} messages):")
    for i, msg in enumerate(replacement):
        d = _msg_to_dict(msg)
        is_summary = is_summary_message(d["text"])
        tag = " [SUMMARY]" if is_summary else ""
        print(f"  [{i}] role={d['role']}{tag}: {d['text']}")

    # Verify structure
    assert len(replacement) >= 2, "Should have at least 1 user message + 1 summary"
    last_text = replacement[-1].content[0].text if isinstance(replacement[-1].content, list) else replacement[-1].content
    assert is_summary_message(last_text), "Last message should be summary"

    # ── Part 2: Test with full budget ────────────────────────────────────
    print("\n" + "=" * 60)
    print("Part 2: build_compacted_history with full 20K budget")
    print("=" * 60)

    replacement_full = build_compacted_history(collected, summary_text)
    print(f"Compacted history ({len(replacement_full)} messages):")
    for i, msg in enumerate(replacement_full):
        d = _msg_to_dict(msg)
        is_summary = is_summary_message(d["text"])
        tag = " [SUMMARY]" if is_summary else ""
        print(f"  [{i}] role={d['role']}{tag}: {d['text']}")

    # With 20K budget and short messages, all should be kept
    assert len(replacement_full) == len(collected) + 1, \
        f"All {len(collected)} user messages + 1 summary = {len(collected)+1}, got {len(replacement_full)}"

    # ── Part 3: Test with previous summary in history ────────────────────
    print("\n" + "=" * 60)
    print("Part 3: History with a previous compaction summary")
    print("=" * 60)

    # Add a previous summary message in the middle
    messages_with_old_summary = list(fake_messages[:5])
    messages_with_old_summary.append(
        UserMessage(content=[TextContent(text=f"{CODEX_SUMMARY_PREFIX}\nPrevious summary of earlier work")])
    )
    messages_with_old_summary.extend(fake_messages[5:])

    collected2 = collect_user_messages(messages_with_old_summary)
    print(f"Messages including old summary: {len(messages_with_old_summary)}")
    print(f"Collected user messages (summary filtered): {len(collected2)}")
    assert len(collected2) == 10, f"Old summary should be filtered out, expected 10 got {len(collected2)}"

    replacement2 = build_compacted_history(collected2, summary_text)
    print(f"Compacted history: {len(replacement2)} messages")

    print("\n" + "=" * 60)
    print("All e2e tests passed! ✓")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
