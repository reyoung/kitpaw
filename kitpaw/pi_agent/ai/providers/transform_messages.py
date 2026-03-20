from __future__ import annotations

from copy import deepcopy
from typing import Callable

from ..types import (
    AssistantMessage,
    Message,
    Model,
    TextContent,
    ToolCall,
    ToolResultMessage,
    now_ms,
)


def transform_messages(
    messages: list[Message],
    model: Model,
    normalize_tool_call_id: Callable[[str, Model, AssistantMessage], str] | None = None,
) -> list[Message]:
    tool_call_id_map: dict[str, str] = {}
    transformed: list[Message] = []

    for message in messages:
        if message.role == "user":
            transformed.append(message)
            continue

        if message.role == "toolResult":
            normalized_id = tool_call_id_map.get(message.tool_call_id)
            if normalized_id and normalized_id != message.tool_call_id:
                transformed.append(
                    ToolResultMessage(
                        tool_call_id=normalized_id,
                        tool_name=message.tool_name,
                        content=deepcopy(message.content),
                        is_error=message.is_error,
                        details=message.details,
                        timestamp=message.timestamp,
                    )
                )
            else:
                transformed.append(message)
            continue

        assistant = message
        is_same_model = (
            assistant.provider == model.provider
            and assistant.api == model.api
            and assistant.model == model.id
        )
        next_content = []
        for block in assistant.content:
            if block.type == "thinking":
                if block.redacted:
                    if is_same_model:
                        next_content.append(deepcopy(block))
                    continue
                if is_same_model and block.thinking_signature:
                    next_content.append(deepcopy(block))
                    continue
                if not block.thinking.strip():
                    continue
                if is_same_model:
                    next_content.append(deepcopy(block))
                else:
                    next_content.append(TextContent(text=block.thinking))
                continue

            if block.type == "text":
                next_content.append(deepcopy(block))
                continue

            if block.type == "toolCall":
                tool_call = deepcopy(block)
                if not is_same_model:
                    tool_call.thought_signature = None
                if not is_same_model and normalize_tool_call_id is not None:
                    normalized_id = normalize_tool_call_id(tool_call.id, model, assistant)
                    if normalized_id != tool_call.id:
                        tool_call_id_map[tool_call.id] = normalized_id
                        tool_call.id = normalized_id
                next_content.append(tool_call)
                continue

        transformed.append(
            AssistantMessage(
                api=assistant.api,
                provider=assistant.provider,
                model=assistant.model,
                content=next_content,
                usage=deepcopy(assistant.usage),
                stop_reason=assistant.stop_reason,
                error_message=assistant.error_message,
                timestamp=assistant.timestamp,
            )
        )

    result: list[Message] = []
    pending_tool_calls: list[ToolCall] = []
    existing_tool_result_ids: set[str] = set()

    def flush_pending() -> None:
        nonlocal pending_tool_calls, existing_tool_result_ids
        if not pending_tool_calls:
            return
        for tool_call in pending_tool_calls:
            if tool_call.id in existing_tool_result_ids:
                continue
            result.append(
                ToolResultMessage(
                    tool_call_id=tool_call.id,
                    tool_name=tool_call.name,
                    content=[TextContent(text="No result provided")],
                    is_error=True,
                    timestamp=now_ms(),
                )
            )
        pending_tool_calls = []
        existing_tool_result_ids = set()

    for message in transformed:
        if message.role == "assistant":
            flush_pending()
            if message.stop_reason in {"error", "aborted"}:
                continue
            tool_calls = [block for block in message.content if block.type == "toolCall"]
            if tool_calls:
                pending_tool_calls = [deepcopy(block) for block in tool_calls]
                existing_tool_result_ids = set()
            result.append(message)
            continue

        if message.role == "toolResult":
            existing_tool_result_ids.add(message.tool_call_id)
            result.append(message)
            continue

        flush_pending()
        result.append(message)

    return result
