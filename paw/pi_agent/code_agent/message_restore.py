from __future__ import annotations

from dataclasses import fields as dataclass_fields
from typing import Any

from ..ai.types import (
    AssistantMessage,
    Cost,
    ImageContent,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResultMessage,
    Usage,
    UserMessage,
)
from .messages import create_branch_summary_message, create_compaction_summary_message


def _pick_fields(cls, data: dict[str, Any]) -> dict[str, Any]:
    """Filter dict to only contain keys that are valid fields of the dataclass."""
    valid = {f.name for f in dataclass_fields(cls)}
    return {k: v for k, v in data.items() if k in valid}


def restore_message(data: dict[str, Any]) -> Any:
    role = data.get("role")
    if role == "user":
        content = data.get("content")
        if isinstance(content, list):
            rebuilt = []
            for block in content:
                if block.get("type") == "text":
                    rebuilt.append(TextContent(**_pick_fields(TextContent, block)))
                elif block.get("type") == "image":
                    rebuilt.append(ImageContent(**_pick_fields(ImageContent, block)))
            content = rebuilt
        return UserMessage(content=content, timestamp=data.get("timestamp", 0))
    if role == "assistant":
        content = []
        for block in data.get("content", []):
            if block.get("type") == "text":
                content.append(TextContent(**_pick_fields(TextContent, block)))
            elif block.get("type") == "thinking":
                content.append(ThinkingContent(**_pick_fields(ThinkingContent, block)))
            elif block.get("type") == "toolCall":
                content.append(ToolCall(**_pick_fields(ToolCall, block)))
        usage_data = data.get("usage", {})
        return AssistantMessage(
            api=data.get("api", "openai-completions"),
            provider=data.get("provider", "openai"),
            model=data.get("model", ""),
            content=content,
            usage=Usage(
                input=usage_data.get("input", 0),
                output=usage_data.get("output", 0),
                cache_read=usage_data.get("cache_read", usage_data.get("cacheRead", 0)),
                cache_write=usage_data.get("cache_write", usage_data.get("cacheWrite", 0)),
                total_tokens=usage_data.get("total_tokens", usage_data.get("totalTokens", 0)),
                cost=Cost(**_pick_fields(Cost, usage_data.get("cost", {}))),
            ),
            stop_reason=data.get("stop_reason", data.get("stopReason", "stop")),
            error_message=data.get("error_message", data.get("errorMessage")),
            timestamp=data.get("timestamp", 0),
        )
    if role == "toolResult":
        content = []
        for block in data.get("content", []):
            if block.get("type") == "text":
                content.append(TextContent(**_pick_fields(TextContent, block)))
            elif block.get("type") == "image":
                content.append(ImageContent(**_pick_fields(ImageContent, block)))
        return ToolResultMessage(
            tool_call_id=data.get("tool_call_id", data.get("toolCallId", "")),
            tool_name=data.get("tool_name", data.get("toolName", "")),
            content=content,
            is_error=data.get("is_error", data.get("isError", False)),
            details=data.get("details"),
            timestamp=data.get("timestamp", 0),
        )
    if role == "branchSummary":
        return create_branch_summary_message(data.get("summary", ""), data.get("fromId", ""), data.get("timestamp", 0))
    if role == "compactionSummary":
        return create_compaction_summary_message(data.get("summary", ""), data.get("tokensBefore", 0), data.get("timestamp", 0))
    return data
