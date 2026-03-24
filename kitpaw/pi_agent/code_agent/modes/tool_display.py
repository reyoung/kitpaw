from __future__ import annotations

import json
import sys
from typing import Any

# ANSI colors
_DIM = "\033[2m"
_CYAN = "\033[36m"
_GREEN = "\033[32m"
_RED = "\033[31m"
_RESET = "\033[0m"


def _truncate(text: str, max_len: int = 60) -> str:
    text = text.replace("\n", "\\n")
    if len(text) > max_len:
        return text[:max_len] + "…"
    return text


def _format_args(args: Any) -> str:
    if args is None:
        return ""
    if isinstance(args, dict):
        parts = []
        for k, v in args.items():
            v_str = json.dumps(v, ensure_ascii=False) if not isinstance(v, str) else v
            parts.append(f"{k}={_truncate(v_str, 20)}")
        return ", ".join(parts)
    return _truncate(str(args))


def _format_result(result: Any) -> str:
    if result is None:
        return "(no output)"
    content = getattr(result, "content", None)
    if isinstance(content, list) and content:
        text = getattr(content[0], "text", str(content[0]))
        return _truncate(text, 80)
    return _truncate(str(result), 80)


def print_tool_start(tool_name: str, args: Any) -> None:
    """Print a compact tool invocation line to stderr."""
    args_str = _format_args(args)
    sys.stderr.write(f"{_DIM}▶ {_CYAN}{tool_name}{_RESET}{_DIM}({args_str}){_RESET}\n")
    sys.stderr.flush()


def print_tool_end(tool_name: str, result: Any, is_error: bool) -> None:
    """Print a compact tool result line to stderr."""
    result_str = _format_result(result)
    if is_error:
        sys.stderr.write(f"{_DIM}  ✗ {_RED}{result_str}{_RESET}\n")
    else:
        sys.stderr.write(f"{_DIM}  ✓ {_GREEN}{result_str}{_RESET}\n")
    sys.stderr.flush()


def make_tool_listener():
    """Create an event listener that prints tool call info to stderr.

    Returns a listener function suitable for ``session.subscribe()``.
    """

    def listener(event) -> None:
        event_type = getattr(event, "type", None)
        if event_type == "tool_execution_start":
            print_tool_start(event.tool_name, event.args)
        elif event_type == "tool_execution_end":
            print_tool_end(event.tool_name, event.result, event.is_error)

    return listener
