#!/usr/bin/env python3
"""Test script to verify Codex compaction produces replacement history
matching the Codex Rust strategy: recent user messages + summary.

Usage:
    python3 tests/test_codex_compaction.py
"""
from __future__ import annotations

import asyncio
import json
import sys
import os

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kitpaw.pi_agent.ai import UserMessage
from kitpaw.pi_agent.ai.types import TextContent
from kitpaw.pi_agent.code_agent.codex.compaction import (
    CODEX_SUMMARY_PREFIX,
    COMPACT_USER_MESSAGE_MAX_TOKENS,
    build_compacted_history,
    collect_user_messages,
    is_summary_message,
)


def test_is_summary_message():
    """Test is_summary_message correctly identifies summary messages."""
    summary = f"{CODEX_SUMMARY_PREFIX}\nSome summary content here"
    assert is_summary_message(summary), "Should identify summary message"
    assert not is_summary_message("Hello world"), "Should not identify normal message"
    assert not is_summary_message(""), "Should not identify empty string"
    print("  ✓ is_summary_message")


def test_collect_user_messages():
    """Test collect_user_messages filters correctly."""
    messages = [
        UserMessage(content="Hello, I need help with X"),
        UserMessage(content="Can you also do Y?"),
        # A previous summary message that should be filtered out
        UserMessage(content=[TextContent(text=f"{CODEX_SUMMARY_PREFIX}\nOld summary")]),
        UserMessage(content="Now do Z please"),
    ]
    result = collect_user_messages(messages)
    assert len(result) == 3, f"Expected 3 user messages, got {len(result)}: {result}"
    assert result[0] == "Hello, I need help with X"
    assert result[1] == "Can you also do Y?"
    assert result[2] == "Now do Z please"
    print("  ✓ collect_user_messages")


def test_build_compacted_history_basic():
    """Test build_compacted_history selects recent messages and appends summary."""
    user_msgs = ["First message", "Second message", "Third message"]
    summary = f"{CODEX_SUMMARY_PREFIX}\nTest summary"

    result = build_compacted_history(user_msgs, summary)

    # Should have all 3 user messages + 1 summary = 4 messages
    assert len(result) == 4, f"Expected 4 messages, got {len(result)}"

    # All should be UserMessage
    for msg in result:
        assert isinstance(msg, UserMessage), f"Expected UserMessage, got {type(msg)}"

    # Last message should be the summary
    last_text = result[-1].content[0].text if isinstance(result[-1].content, list) else result[-1].content
    assert CODEX_SUMMARY_PREFIX in last_text, "Last message should contain summary prefix"

    # First 3 should be the user messages in order
    for i, msg_text in enumerate(user_msgs):
        actual = result[i].content[0].text if isinstance(result[i].content, list) else result[i].content
        assert actual == msg_text, f"Message {i}: expected '{msg_text}', got '{actual}'"

    print("  ✓ build_compacted_history (basic)")


def test_build_compacted_history_token_limit():
    """Test build_compacted_history respects token budget, selecting from end."""
    # Create messages where total exceeds budget
    # Each "x" * 400 ≈ 100 tokens (400 / 4 = 100)
    user_msgs = [
        "A" * 400,  # ~100 tokens
        "B" * 400,  # ~100 tokens
        "C" * 400,  # ~100 tokens
        "D" * 400,  # ~100 tokens
    ]
    summary = "Test summary"

    # Budget of 250 tokens: should keep D (100) + C (100) + truncated B
    result = build_compacted_history(user_msgs, summary, max_user_tokens=250)

    # Should have kept some messages + summary
    assert len(result) >= 3, f"Expected at least 3 messages (2 full + 1 partial + summary), got {len(result)}"

    # Last message should be summary
    last_text = result[-1].content[0].text if isinstance(result[-1].content, list) else result[-1].content
    assert last_text == summary

    # The messages before summary should be from the END of user_msgs
    # (most recent first when selecting, then reversed to chronological)
    second_to_last = result[-2].content[0].text if isinstance(result[-2].content, list) else result[-2].content
    assert second_to_last.startswith("D"), "Second to last should be the most recent message (D)"

    print("  ✓ build_compacted_history (token limit)")


def test_build_compacted_history_empty():
    """Test build_compacted_history with no user messages."""
    result = build_compacted_history([], "Summary text")
    assert len(result) == 1, f"Expected 1 message (summary only), got {len(result)}"
    last_text = result[0].content[0].text if isinstance(result[0].content, list) else result[0].content
    assert last_text == "Summary text"
    print("  ✓ build_compacted_history (empty)")


def test_build_compacted_history_empty_summary():
    """Test build_compacted_history with empty summary."""
    result = build_compacted_history(["Hello"], "")
    assert len(result) == 2
    last_text = result[-1].content[0].text if isinstance(result[-1].content, list) else result[-1].content
    assert last_text == "(no summary available)"
    print("  ✓ build_compacted_history (empty summary)")


def main():
    print("Testing Codex compaction functions:")
    test_is_summary_message()
    test_collect_user_messages()
    test_build_compacted_history_basic()
    test_build_compacted_history_token_limit()
    test_build_compacted_history_empty()
    test_build_compacted_history_empty_summary()
    print("\nAll tests passed! ✓")


if __name__ == "__main__":
    main()
