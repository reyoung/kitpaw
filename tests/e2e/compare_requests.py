#!/usr/bin/env python3
"""Compare requests captured from Codex (Rust) and kitpaw (Python).

Usage:
    python compare_requests.py codex_requests.jsonl kitpaw_requests.jsonl
"""
from __future__ import annotations

import json
import sys


def load_requests(path: str) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def extract_chat_requests(records: list[dict]) -> list[dict]:
    """Filter to only chat completion requests."""
    return [
        r for r in records
        if r.get("request", {}).get("model") and "chat/completions" in r.get("path", "")
    ]


def compare_system_prompts(codex_reqs: list[dict], kitpaw_reqs: list[dict]) -> None:
    print("=" * 60)
    print("SYSTEM PROMPT COMPARISON")
    print("=" * 60)

    for label, reqs in [("Codex", codex_reqs), ("kitpaw", kitpaw_reqs)]:
        if not reqs:
            print(f"\n{label}: No requests found")
            continue
        messages = reqs[0]["request"].get("messages", [])
        system_msgs = [m for m in messages if m.get("role") in ("system", "developer")]
        if system_msgs:
            content = system_msgs[0].get("content", "")
            if isinstance(content, list):
                content = "\n".join(
                    c.get("text", "") for c in content if c.get("type") == "text"
                )
            print(f"\n{label} system prompt ({len(content)} chars):")
            print(content[:500] + "..." if len(content) > 500 else content)
        else:
            print(f"\n{label}: No system message found")


def compare_tools(codex_reqs: list[dict], kitpaw_reqs: list[dict]) -> None:
    print("\n" + "=" * 60)
    print("TOOL COMPARISON")
    print("=" * 60)

    for label, reqs in [("Codex", codex_reqs), ("kitpaw", kitpaw_reqs)]:
        if not reqs:
            print(f"\n{label}: No requests found")
            continue
        tools = reqs[0]["request"].get("tools", [])
        tool_names = [t.get("function", {}).get("name", "?") for t in tools]
        print(f"\n{label} tools ({len(tools)}):")
        for name in sorted(tool_names):
            print(f"  - {name}")


def compare_tool_schemas(codex_reqs: list[dict], kitpaw_reqs: list[dict]) -> None:
    print("\n" + "=" * 60)
    print("TOOL SCHEMA COMPARISON")
    print("=" * 60)

    codex_tools = {}
    kitpaw_tools = {}

    if codex_reqs:
        for t in codex_reqs[0]["request"].get("tools", []):
            name = t.get("function", {}).get("name", "?")
            codex_tools[name] = t.get("function", {})

    if kitpaw_reqs:
        for t in kitpaw_reqs[0]["request"].get("tools", []):
            name = t.get("function", {}).get("name", "?")
            kitpaw_tools[name] = t.get("function", {})

    all_names = sorted(set(codex_tools) | set(kitpaw_tools))
    for name in all_names:
        codex_t = codex_tools.get(name)
        kitpaw_t = kitpaw_tools.get(name)

        if not codex_t:
            print(f"\n  {name}: kitpaw ONLY")
        elif not kitpaw_t:
            print(f"\n  {name}: Codex ONLY")
        else:
            codex_params = codex_t.get("parameters", {})
            kitpaw_params = kitpaw_t.get("parameters", {})
            if codex_params == kitpaw_params:
                print(f"\n  {name}: MATCH ✓")
            else:
                print(f"\n  {name}: DIFFER ✗")
                print(f"    Codex params:  {json.dumps(codex_params, indent=2)[:200]}")
                print(f"    kitpaw params: {json.dumps(kitpaw_params, indent=2)[:200]}")


def compare_messages(codex_reqs: list[dict], kitpaw_reqs: list[dict]) -> None:
    print("\n" + "=" * 60)
    print("MESSAGE STRUCTURE COMPARISON")
    print("=" * 60)

    for label, reqs in [("Codex", codex_reqs), ("kitpaw", kitpaw_reqs)]:
        if not reqs:
            continue
        for i, req in enumerate(reqs):
            messages = req["request"].get("messages", [])
            roles = [m.get("role", "?") for m in messages]
            print(f"\n{label} request #{i+1}: {len(messages)} messages, roles: {roles}")


def main() -> None:
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} codex_requests.jsonl kitpaw_requests.jsonl")
        sys.exit(1)

    codex_records = load_requests(sys.argv[1])
    kitpaw_records = load_requests(sys.argv[2])

    codex_reqs = extract_chat_requests(codex_records)
    kitpaw_reqs = extract_chat_requests(kitpaw_records)

    print(f"Loaded {len(codex_records)} Codex records ({len(codex_reqs)} chat requests)")
    print(f"Loaded {len(kitpaw_records)} kitpaw records ({len(kitpaw_reqs)} chat requests)")

    compare_system_prompts(codex_reqs, kitpaw_reqs)
    compare_tools(codex_reqs, kitpaw_reqs)
    compare_tool_schemas(codex_reqs, kitpaw_reqs)
    compare_messages(codex_reqs, kitpaw_reqs)


if __name__ == "__main__":
    main()
