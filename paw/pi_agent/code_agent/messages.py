from __future__ import annotations

from datetime import datetime
from typing import Any

from ..ai.types import AssistantMessage, Message, TextContent, ToolResultMessage, UserMessage

COMPACTION_SUMMARY_PREFIX = """The conversation history before this point was compacted into the following summary:

<summary>
"""

COMPACTION_SUMMARY_SUFFIX = """
</summary>"""

BRANCH_SUMMARY_PREFIX = """The following is a summary of a branch that this conversation came back from:

<summary>
"""

BRANCH_SUMMARY_SUFFIX = "</summary>"


def create_branch_summary_message(summary: str, from_id: str, timestamp: str | int) -> dict[str, Any]:
    return {
        "role": "branchSummary",
        "summary": summary,
        "fromId": from_id,
        "timestamp": _to_timestamp_ms(timestamp),
    }


def create_compaction_summary_message(summary: str, tokens_before: int, timestamp: str | int) -> dict[str, Any]:
    return {
        "role": "compactionSummary",
        "summary": summary,
        "tokensBefore": tokens_before,
        "timestamp": _to_timestamp_ms(timestamp),
    }


def convert_to_llm(messages: list[Any]) -> list[Message]:
    converted: list[Message] = []
    for message in messages:
        if isinstance(message, (UserMessage, AssistantMessage, ToolResultMessage)):
            converted.append(message)
            continue
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role == "branchSummary":
            converted.append(
                UserMessage(
                    content=[TextContent(text=BRANCH_SUMMARY_PREFIX + message.get("summary", "") + BRANCH_SUMMARY_SUFFIX)],
                    timestamp=message.get("timestamp", 0),
                )
            )
        elif role == "compactionSummary":
            converted.append(
                UserMessage(
                    content=[TextContent(text=COMPACTION_SUMMARY_PREFIX + message.get("summary", "") + COMPACTION_SUMMARY_SUFFIX)],
                    timestamp=message.get("timestamp", 0),
                )
            )
        elif role == "custom":
            content = message.get("content", "")
            if isinstance(content, str):
                content = [TextContent(text=content)]
            converted.append(UserMessage(content=content, timestamp=message.get("timestamp", 0)))
        elif role == "bashExecution" and not message.get("excludeFromContext"):
            converted.append(
                UserMessage(content=[TextContent(text=_bash_execution_to_text(message))], timestamp=message.get("timestamp", 0))
            )
    return converted


def _to_timestamp_ms(value: str | int) -> int:
    if isinstance(value, int):
        return value
    return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() * 1000)


def _bash_execution_to_text(message: dict[str, Any]) -> str:
    text = f"Ran `{message.get('command', '')}`\n"
    output = message.get("output", "")
    if output:
        text += f"```\n{output}\n```"
    else:
        text += "(no output)"
    if message.get("cancelled"):
        text += "\n\n(command cancelled)"
    elif message.get("exitCode") not in (None, 0):
        text += f"\n\nCommand exited with code {message.get('exitCode')}"
    if message.get("truncated") and message.get("fullOutputPath"):
        text += f"\n\n[Output truncated. Full output: {message['fullOutputPath']}]"
    return text
