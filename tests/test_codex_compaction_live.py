#!/usr/bin/env python3
"""Trigger real Codex compaction with local LLM and dump results.

This script creates a Codex session, injects enough messages to exceed
the compaction threshold, then triggers auto_compact and dumps the
before/after agent message state.

Usage:
    uv run python tests/test_codex_compaction_live.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _msg_summary(msg, idx: int) -> str:
    role = getattr(msg, "role", "unknown")
    content = getattr(msg, "content", "")
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        text = "\n".join(getattr(b, "text", "") for b in content if getattr(b, "type", None) == "text")
    else:
        text = str(content)
    preview = text[:120].replace("\n", "\\n")
    if len(text) > 120:
        preview += f"... [{len(text)} chars]"
    return f"  [{idx:2d}] role={role:12s} | {preview}"


async def main():
    from kitpaw.pi_agent.ai import UserMessage
    from kitpaw.pi_agent.ai.types import TextContent
    from kitpaw.pi_agent.code_agent.codex.compaction import (
        CODEX_SUMMARY_PREFIX,
        configure_codex_compaction,
        is_summary_message,
    )
    from kitpaw.pi_agent.code_agent.sdk import create_agent_session, CreateAgentSessionOptions
    from kitpaw.pi_agent.code_agent.codex.tools import create_codex_tools
    from kitpaw.pi_agent.code_agent.codex.resource_loader import CodexResourceLoader
    from kitpaw.pi_agent.code_agent.config import get_agent_dir
    from kitpaw.pi_agent.code_agent.summarizer import estimate_tokens

    print("Creating Codex agent session...")
    cwd = os.getcwd()
    agent_dir = str(get_agent_dir())

    codex_tools = create_codex_tools(cwd)
    codex_loader = CodexResourceLoader(cwd, agent_dir, None)
    codex_loader.set_tool_names([t.name for t in codex_tools])

    options = CreateAgentSessionOptions(
        cwd=cwd,
        agent_dir=agent_dir,
        resource_loader=codex_loader,
        tools=codex_tools,
    )
    result = await create_agent_session(options)
    session = result.session

    # Configure Codex compaction
    configure_codex_compaction(session)

    # Enable compaction (may be disabled by default)
    session.set_compaction_enabled(True)

    # Inject many synthetic messages to exceed threshold
    print("\nInjecting synthetic messages...")
    synthetic_messages = [
        "I need to fix the distance calculation in sympy's geometry module. The issue is in Point.distance().",
        "After looking at the code, I found that Point(2,0).distance(Point(1,0,2)) returns 1 instead of sqrt(5) because zip truncates to shorter dimension.",
        "I've checked the source code at sympy/geometry/point.py, line 265. The distance method uses zip(self.args, p.args) which silently drops extra dimensions.",
        "The same bug exists in taxicab_distance and midpoint methods. They all use zip which truncates.",
        "My approach: modify the distance, taxicab_distance, and midpoint methods to pad the shorter point with zeros before computing.",
        "I wrote a test: Point(2,0).distance(Point(1,0,2)) should be sqrt(5). Point(1,0,2).distance(Point(2,0)) should also be sqrt(5).",
        "I've applied the fix using apply_patch. The key change was replacing zip(self.args, p.args) with itertools.zip_longest(self.args, p.args, fillvalue=0).",
        "Running the test suite: python -m pytest sympy/geometry/tests/test_point.py -x. All tests pass.",
        "I also added new test cases for mixed-dimension point operations to prevent regression.",
        "Let me also check if there are any other methods in Point that use zip and might have the same bug.",
        "Found that __add__, __sub__, and __mul__ also use zip. But those already raise ValueError for dimension mismatch, which is correct behavior.",
        "Final check: running the full geometry test suite to make sure nothing else broke.",
    ]

    for text in synthetic_messages:
        session.agent._state.messages.append(UserMessage(content=text))

    estimated = estimate_tokens(session.messages)
    print(f"Injected {len(synthetic_messages)} messages, estimated tokens: {estimated}")

    # Show compaction state
    state = session.get_compaction_state()
    print(f"\nCompaction state:")
    print(f"  enabled:        {state['enabled']}")
    print(f"  estimatedTokens:{state['estimatedTokens']}")
    print(f"  thresholdTokens:{state['thresholdTokens']}")
    print(f"  contextWindow:  {state['contextWindow']}")
    print(f"  reserveTokens:  {state['reserveTokens']}")
    print(f"  shouldCompact:  {state['shouldCompact']}")

    # Force threshold low enough to trigger compaction
    # Set reserve_tokens high so threshold is very low
    reserve = max(session.model.context_window - 100, 0)
    session.set_compaction_reserve_tokens(reserve)
    state = session.get_compaction_state()
    print(f"\nAfter lowering threshold (reserve={reserve}):")
    print(f"  thresholdTokens:{state['thresholdTokens']}")
    print(f"  shouldCompact:  {state['shouldCompact']}")

    # Dump messages BEFORE compaction
    print(f"\n{'='*60}")
    print(f"BEFORE compaction: {len(session.agent._state.messages)} messages")
    print(f"{'='*60}")
    for i, msg in enumerate(session.agent._state.messages):
        print(_msg_summary(msg, i))

    # Trigger compaction
    print(f"\n{'='*60}")
    print("Triggering auto_compact...")
    print(f"{'='*60}")

    # Find the first_kept_entry_id
    first_kept_id = session._find_first_kept_entry_id(state['keepRecentTokens'])
    if first_kept_id is None:
        # If session_manager doesn't have entries (since we injected directly),
        # use a dummy ID
        first_kept_id = "synthetic-0"

    try:
        compact_result = await session.auto_compact(first_kept_id)
        print(f"Compaction result: {json.dumps(compact_result, indent=2)}")
    except Exception as e:
        print(f"Compaction error: {e}")
        import traceback
        traceback.print_exc()

    # Dump messages AFTER compaction
    print(f"\n{'='*60}")
    print(f"AFTER compaction: {len(session.agent._state.messages)} messages")
    print(f"{'='*60}")
    for i, msg in enumerate(session.agent._state.messages):
        print(_msg_summary(msg, i))

    # Verify structure
    print(f"\n{'='*60}")
    print("Verification:")
    print(f"{'='*60}")
    msgs = session.agent._state.messages
    if msgs:
        last_content = getattr(msgs[-1], "content", "")
        if isinstance(last_content, list):
            last_text = last_content[0].text if last_content else ""
        else:
            last_text = str(last_content)

        has_summary = is_summary_message(last_text)
        print(f"  Last message is summary: {has_summary}")

        user_count = sum(1 for m in msgs[:-1] if getattr(m, "role", None) == "user")
        print(f"  User messages before summary: {user_count}")
        print(f"  Total messages: {len(msgs)}")

        if has_summary and user_count > 0:
            print("\n  ✓ Compacted history matches Codex Rust strategy:")
            print("    [recent user messages] + [summary message]")
        else:
            print("\n  ✗ Compacted history does NOT match expected structure")
    else:
        print("  ✗ No messages after compaction")


if __name__ == "__main__":
    asyncio.run(main())
