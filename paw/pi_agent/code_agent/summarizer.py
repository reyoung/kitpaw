from __future__ import annotations

from typing import Any

from ..ai import Context, TextContent, UserMessage, complete
from .messages import convert_to_llm

SUMMARIZATION_SYSTEM_PROMPT = (
    "You are a context summarization assistant. Your task is to read a conversation between a user and an AI coding "
    "assistant, then produce a structured summary following the exact format specified.\n\n"
    "Do NOT continue the conversation. Do NOT respond to any questions in the conversation. ONLY output the structured summary."
)

SUMMARIZATION_PROMPT = (
    "Summarize the conversation for future continuation. Keep it concise but preserve:\n"
    "- user goals and constraints\n"
    "- important assistant conclusions\n"
    "- concrete file changes, commands, and outcomes\n"
    "- unresolved questions or follow-up work\n"
)


def serialize_conversation(messages: list[Any]) -> str:
    llm_messages = convert_to_llm(messages)
    parts: list[str] = []
    for message in llm_messages:
        role = getattr(message, "role", "unknown").capitalize()
        content = _message_text(message)
        if content:
            parts.append(f"{role}:\n{content}")
    return "\n\n".join(parts)


def estimate_tokens(messages: list[Any]) -> int:
    text = serialize_conversation(messages)
    return max(len(text) // 4, 1) if text else 0


async def generate_summary(model: Any, messages: list[Any], custom_instructions: str | None = None) -> str:
    conversation_text = serialize_conversation(messages)
    prompt = f"<conversation>\n{conversation_text}\n</conversation>\n\n{SUMMARIZATION_PROMPT}"
    if custom_instructions:
        prompt += f"\nAdditional focus: {custom_instructions.strip()}"
    response = await complete(
        model,
        Context(messages=[UserMessage(content=[TextContent(text=prompt)])], system_prompt=SUMMARIZATION_SYSTEM_PROMPT),
        {"max_tokens": 1024, **({"reasoning": "high"} if getattr(model, "reasoning", False) else {})},
    )
    text = "\n".join(block.text for block in response.content if getattr(block, "type", None) == "text").strip()
    return text or "No summary generated"


def _message_text(message: Any) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if getattr(block, "type", None) == "text":
                parts.append(getattr(block, "text", ""))
        return "\n".join(part for part in parts if part).strip()
    return ""
