from __future__ import annotations

import inspect
import json
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from typing import Any, Mapping

from openai import AsyncOpenAI

from ..env_api_keys import get_env_api_key
from ..event_stream import AssistantMessageEventStream
from ..models import calculate_cost, supports_xhigh
from ..types import (
    AssistantMessage,
    Context,
    DoneEvent,
    ErrorEvent,
    Model,
    OpenAICompletionsCompat,
    OpenAICompletionsOptions,
    SimpleStreamOptions,
    StartEvent,
    StopReason,
    StreamOptions,
    TextContent,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
    ThinkingContent,
    ThinkingDeltaEvent,
    ThinkingEndEvent,
    ThinkingStartEvent,
    Tool,
    ToolCall,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
    ToolResultMessage,
    Usage,
    now_ms,
)
from .simple_options import build_base_options, clamp_reasoning
from .transform_messages import transform_messages


def normalize_options(
    options: OpenAICompletionsOptions | StreamOptions | Mapping[str, Any] | None,
) -> OpenAICompletionsOptions:
    if options is None:
        return OpenAICompletionsOptions()
    if isinstance(options, OpenAICompletionsOptions):
        return options
    if is_dataclass(options):
        return OpenAICompletionsOptions(**asdict(options))
    if isinstance(options, Mapping):
        return OpenAICompletionsOptions(**dict(options))
    raise TypeError(f"Unsupported options type: {type(options)!r}")


def normalize_simple_options(options: SimpleStreamOptions | Mapping[str, Any] | None) -> SimpleStreamOptions:
    if options is None:
        return SimpleStreamOptions()
    if isinstance(options, SimpleStreamOptions):
        return options
    if is_dataclass(options):
        return SimpleStreamOptions(**asdict(options))
    if isinstance(options, Mapping):
        return SimpleStreamOptions(**dict(options))
    raise TypeError(f"Unsupported options type: {type(options)!r}")


def has_tool_history(messages: list[Any]) -> bool:
    for message in messages:
        if message.role == "toolResult":
            return True
        if message.role == "assistant" and any(block.type == "toolCall" for block in message.content):
            return True
    return False


def _stream_openai_completions_with_provider_options(
    model: Model,
    context: Context,
    options: OpenAICompletionsOptions | StreamOptions | Mapping[str, Any] | None = None,
) -> AssistantMessageEventStream:
    normalized = normalize_options(options)

    async def producer():
        output = AssistantMessage(
            api=model.api,
            provider=model.provider,
            model=model.id,
            timestamp=now_ms(),
        )
        current_block: TextContent | ThinkingContent | ToolCall | None = None
        current_tool_partial_args = ""

        def block_index() -> int:
            return len(output.content) - 1

        def finish_current_block() -> list[Any]:
            nonlocal current_block, current_tool_partial_args
            if current_block is None:
                return []

            index = block_index()
            if current_block.type == "text":
                events = [TextEndEvent(content_index=index, content=current_block.text, partial=output)]
            elif current_block.type == "thinking":
                events = [ThinkingEndEvent(content_index=index, content=current_block.thinking, partial=output)]
            else:
                current_block.arguments = parse_streaming_json(current_tool_partial_args)
                events = [ToolCallEndEvent(content_index=index, tool_call=current_block, partial=output)]

            current_block = None
            current_tool_partial_args = ""
            return events

        try:
            api_key = normalized.api_key or get_env_api_key(model.provider)
            client = create_client(model, api_key, normalized.headers)
            params = build_params(model, context, normalized)
            if normalized.on_payload is not None:
                maybe_next = normalized.on_payload(deepcopy(params), model)
                if inspect.isawaitable(maybe_next):
                    maybe_next = await maybe_next
                if maybe_next is not None:
                    params = maybe_next

            openai_stream = await client.chat.completions.create(**params)
            yield StartEvent(partial=output)

            async for chunk in openai_stream:
                chunk_data = to_dict(chunk)
                if chunk_data.get("usage"):
                    output.usage = parse_chunk_usage(chunk_data["usage"], model)

                choices = chunk_data.get("choices") or []
                if not choices:
                    continue

                choice = choices[0]
                if not chunk_data.get("usage") and choice.get("usage"):
                    output.usage = parse_chunk_usage(choice["usage"], model)

                finish_reason = choice.get("finish_reason")
                if finish_reason:
                    output.stop_reason = map_stop_reason(finish_reason)

                delta = choice.get("delta") or {}

                text_delta = delta.get("content")
                if text_delta:
                    if current_block is None or current_block.type != "text":
                        for event in finish_current_block():
                            yield event
                        current_block = TextContent(text="")
                        output.content.append(current_block)
                        yield TextStartEvent(content_index=block_index(), partial=output)
                    current_block.text += text_delta
                    yield TextDeltaEvent(content_index=block_index(), delta=text_delta, partial=output)

                reasoning_field = next(
                    (
                        field
                        for field in ("reasoning_content", "reasoning", "reasoning_text")
                        if delta.get(field)
                    ),
                    None,
                )
                if reasoning_field:
                    if current_block is None or current_block.type != "thinking":
                        for event in finish_current_block():
                            yield event
                        current_block = ThinkingContent(thinking="", thinking_signature=reasoning_field)
                        output.content.append(current_block)
                        yield ThinkingStartEvent(content_index=block_index(), partial=output)

                    reasoning_delta = delta[reasoning_field]
                    current_block.thinking += reasoning_delta
                    yield ThinkingDeltaEvent(content_index=block_index(), delta=reasoning_delta, partial=output)

                tool_calls = delta.get("tool_calls") or []
                for tool_call in tool_calls:
                    function = tool_call.get("function") or {}
                    needs_new_block = (
                        current_block is None
                        or current_block.type != "toolCall"
                        or (tool_call.get("id") and current_block.id != tool_call["id"])
                    )
                    if needs_new_block:
                        for event in finish_current_block():
                            yield event
                        current_block = ToolCall(
                            id=tool_call.get("id") or "",
                            name=function.get("name") or "",
                            arguments={},
                        )
                        current_tool_partial_args = ""
                        output.content.append(current_block)
                        yield ToolCallStartEvent(content_index=block_index(), partial=output)

                    if tool_call.get("id"):
                        current_block.id = tool_call["id"]
                    if function.get("name"):
                        current_block.name = function["name"]
                    argument_delta = function.get("arguments") or ""
                    if argument_delta:
                        current_tool_partial_args += argument_delta
                        current_block.arguments = parse_streaming_json(current_tool_partial_args)
                    yield ToolCallDeltaEvent(
                        content_index=block_index(),
                        delta=argument_delta,
                        partial=output,
                    )

                reasoning_details = delta.get("reasoning_details") or []
                for detail in reasoning_details:
                    if detail.get("type") != "reasoning.encrypted":
                        continue
                    detail_id = detail.get("id")
                    if not detail_id:
                        continue
                    matching = next(
                        (
                            block
                            for block in output.content
                            if block.type == "toolCall" and block.id == detail_id
                        ),
                        None,
                    )
                    if matching is not None:
                        matching.thought_signature = json.dumps(detail)

            for event in finish_current_block():
                yield event

            if output.stop_reason in {"aborted", "error"}:
                raise RuntimeError("The completion ended in an error state.")

            yield DoneEvent(reason=output.stop_reason, message=output)
        except Exception as exc:
            output.stop_reason = "error"
            output.error_message = str(exc)
            yield ErrorEvent(reason=output.stop_reason, error=output)

    return AssistantMessageEventStream(producer)


def stream_openai_completions(
    model: Model,
    context: Context,
    options: SimpleStreamOptions | Mapping[str, Any] | None = None,
) -> AssistantMessageEventStream:
    normalized = normalize_simple_options(options)
    api_key = normalized.api_key or get_env_api_key(model.provider)
    base_options = build_base_options(model, normalized, api_key)
    reasoning_effort = normalized.reasoning if supports_xhigh(model) else clamp_reasoning(normalized.reasoning)
    provider_options = OpenAICompletionsOptions(
        temperature=base_options.temperature,
        max_tokens=base_options.max_tokens,
        api_key=base_options.api_key,
        headers=base_options.headers,
        on_payload=base_options.on_payload,
        reasoning_effort=reasoning_effort,
    )
    return _stream_openai_completions_with_provider_options(model, context, provider_options)


def create_client(model: Model, api_key: str | None, headers: dict[str, str] | None) -> AsyncOpenAI:
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required.")

    merged_headers = dict(model.headers)
    if headers:
        merged_headers.update(headers)

    return AsyncOpenAI(
        api_key=api_key,
        base_url=model.base_url,
        default_headers=merged_headers or None,
    )


def build_params(model: Model, context: Context, options: OpenAICompletionsOptions) -> dict[str, Any]:
    compat = get_compat(model)
    messages = convert_messages(model, context, compat)
    params: dict[str, Any] = {
        "model": model.id,
        "messages": messages,
        "stream": True,
    }

    if compat.supports_usage_in_streaming:
        params["stream_options"] = {"include_usage": True}

    if compat.supports_store:
        params["store"] = False

    if options.max_tokens:
        params[compat.max_tokens_field] = options.max_tokens

    if options.temperature is not None:
        params["temperature"] = options.temperature

    if context.tools:
        params["tools"] = convert_tools(context.tools, compat)
    elif has_tool_history(context.messages):
        params["tools"] = []

    if options.tool_choice is not None:
        params["tool_choice"] = options.tool_choice

    if compat.thinking_format in {"zai", "qwen"} and model.reasoning:
        params["enable_thinking"] = bool(options.reasoning_effort)
    elif compat.thinking_format == "qwen-chat-template" and model.reasoning:
        params["chat_template_kwargs"] = {"enable_thinking": bool(options.reasoning_effort)}
    elif options.reasoning_effort and model.reasoning and compat.supports_reasoning_effort:
        params["reasoning_effort"] = map_reasoning_effort(
            options.reasoning_effort,
            compat.reasoning_effort_map,
        )

    return params


def convert_messages(
    model: Model,
    context: Context,
    compat: OpenAICompletionsCompat,
) -> list[dict[str, Any]]:
    params: list[dict[str, Any]] = []

    def normalize_tool_call_id(tool_call_id: str, current_model: Model, _: AssistantMessage) -> str:
        if "|" in tool_call_id:
            call_id = tool_call_id.split("|", 1)[0]
            return "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in call_id)[:40]
        if current_model.provider == "openai" and len(tool_call_id) > 40:
            return tool_call_id[:40]
        return tool_call_id

    transformed_messages = transform_messages(context.messages, model, normalize_tool_call_id)

    if context.system_prompt:
        role = "developer" if model.reasoning and compat.supports_developer_role else "system"
        params.append({"role": role, "content": sanitize_surrogates(context.system_prompt)})

    last_role: str | None = None
    i = 0
    while i < len(transformed_messages):
        message = transformed_messages[i]
        if (
            compat.requires_assistant_after_tool_result
            and last_role == "toolResult"
            and message.role == "user"
        ):
            params.append({"role": "assistant", "content": "I have processed the tool results."})

        if message.role == "user":
            if isinstance(message.content, str):
                params.append({"role": "user", "content": sanitize_surrogates(message.content)})
            else:
                content = []
                for item in message.content:
                    if item.type == "text":
                        content.append({"type": "text", "text": sanitize_surrogates(item.text)})
                    else:
                        content.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{item.mime_type};base64,{item.data}"},
                            }
                        )

                filtered_content = (
                    [part for part in content if part["type"] != "image_url"]
                    if "image" not in model.input
                    else content
                )
                if filtered_content:
                    params.append({"role": "user", "content": filtered_content})
            last_role = message.role
            i += 1
            continue

        if message.role == "assistant":
            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": "" if compat.requires_assistant_after_tool_result else None,
            }
            text_blocks = [
                block
                for block in message.content
                if block.type == "text" and block.text.strip()
            ]
            if text_blocks:
                assistant_msg["content"] = "".join(sanitize_surrogates(block.text) for block in text_blocks)

            thinking_blocks = [
                block
                for block in message.content
                if block.type == "thinking" and block.thinking.strip()
            ]
            if thinking_blocks:
                thinking_text = "\n\n".join(block.thinking for block in thinking_blocks)
                if compat.requires_thinking_as_text:
                    existing = assistant_msg.get("content") or ""
                    assistant_msg["content"] = thinking_text if not existing else f"{thinking_text}\n\n{existing}"
                else:
                    signature = thinking_blocks[0].thinking_signature
                    if signature:
                        assistant_msg[signature] = "\n".join(block.thinking for block in thinking_blocks)

            tool_calls = [block for block in message.content if block.type == "toolCall"]
            if tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.name,
                            "arguments": json.dumps(tool_call.arguments),
                        },
                    }
                    for tool_call in tool_calls
                ]
                reasoning_details = []
                for tool_call in tool_calls:
                    if not tool_call.thought_signature:
                        continue
                    try:
                        reasoning_details.append(json.loads(tool_call.thought_signature))
                    except json.JSONDecodeError:
                        continue
                if reasoning_details:
                    assistant_msg["reasoning_details"] = reasoning_details

            content = assistant_msg.get("content")
            has_content = content is not None and (len(content) > 0 if isinstance(content, str) else bool(content))
            if has_content or assistant_msg.get("tool_calls"):
                params.append(assistant_msg)
            last_role = message.role
            i += 1
            continue

        image_blocks: list[dict[str, Any]] = []
        while i < len(transformed_messages) and transformed_messages[i].role == "toolResult":
            tool_message = transformed_messages[i]
            assert isinstance(tool_message, ToolResultMessage)
            text_result = "\n".join(
                block.text
                for block in tool_message.content
                if block.type == "text"
            )
            has_images = any(block.type == "image" for block in tool_message.content)
            tool_result_message: dict[str, Any] = {
                "role": "tool",
                "content": sanitize_surrogates(text_result or "(see attached image)"),
                "tool_call_id": tool_message.tool_call_id,
            }
            if compat.requires_tool_result_name and tool_message.tool_name:
                tool_result_message["name"] = tool_message.tool_name
            params.append(tool_result_message)

            if has_images and "image" in model.input:
                for block in tool_message.content:
                    if block.type != "image":
                        continue
                    image_blocks.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{block.mime_type};base64,{block.data}"},
                        }
                    )
            i += 1

        if image_blocks:
            if compat.requires_assistant_after_tool_result:
                params.append({"role": "assistant", "content": "I have processed the tool results."})
            params.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Attached image(s) from tool result:"},
                        *image_blocks,
                    ],
                }
            )
            last_role = "user"
        else:
            last_role = "toolResult"

    return params


def convert_tools(tools: list[Tool], compat: OpenAICompletionsCompat) -> list[dict[str, Any]]:
    converted = []
    for tool in tools:
        function: dict[str, Any] = {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        }
        if compat.supports_strict_mode:
            function["strict"] = False
        converted.append({"type": "function", "function": function})
    return converted


def parse_chunk_usage(raw_usage: dict[str, Any], model: Model) -> Usage:
    cached_tokens = (raw_usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    reasoning_tokens = (raw_usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0)
    input_tokens = raw_usage.get("prompt_tokens", 0) - cached_tokens
    output_tokens = raw_usage.get("completion_tokens", 0) + reasoning_tokens
    usage = Usage(
        input=input_tokens,
        output=output_tokens,
        cache_read=cached_tokens,
        cache_write=0,
        total_tokens=input_tokens + output_tokens + cached_tokens,
    )
    return calculate_cost(model, usage)


def map_stop_reason(reason: str | None) -> StopReason:
    if reason in {None, "stop", "end"}:
        return "stop"
    if reason == "length":
        return "length"
    if reason in {"function_call", "tool_calls"}:
        return "toolUse"
    if reason == "content_filter":
        return "error"
    raise ValueError(f"Unhandled stop reason: {reason}")


def detect_compat(model: Model) -> OpenAICompletionsCompat:
    provider = model.provider
    base_url = model.base_url
    is_zai = provider == "zai" or "api.z.ai" in base_url
    is_bigmodel = "bigmodel.cn" in base_url
    is_non_standard = (
        provider in {"cerebras", "xai", "opencode"}
        or "cerebras.ai" in base_url
        or "api.x.ai" in base_url
        or "chutes.ai" in base_url
        or "deepseek.com" in base_url
        or is_bigmodel
        or is_zai
        or "opencode.ai" in base_url
    )
    is_groq = provider == "groq" or "groq.com" in base_url
    is_grok = provider == "xai" or "api.x.ai" in base_url
    reasoning_effort_map = (
        {
            "minimal": "default",
            "low": "default",
            "medium": "default",
            "high": "default",
            "xhigh": "default",
        }
        if is_groq and model.id == "qwen/qwen3-32b"
        else {}
    )
    return OpenAICompletionsCompat(
        supports_store=not is_non_standard,
        supports_developer_role=not is_non_standard,
        supports_reasoning_effort=not is_grok and not is_zai and not is_bigmodel,
        reasoning_effort_map=reasoning_effort_map,
        supports_usage_in_streaming=True,
        max_tokens_field="max_tokens" if "chutes.ai" in base_url or is_bigmodel else "max_completion_tokens",
        requires_tool_result_name=False,
        requires_assistant_after_tool_result=False,
        requires_thinking_as_text=False,
        thinking_format="zai" if is_zai else "openai",
        supports_strict_mode=True,
    )


def get_compat(model: Model) -> OpenAICompletionsCompat:
    detected = detect_compat(model)
    if model.compat is None:
        return detected

    override = model.compat
    return OpenAICompletionsCompat(
        supports_store=override.supports_store,
        supports_developer_role=override.supports_developer_role,
        supports_reasoning_effort=override.supports_reasoning_effort,
        reasoning_effort_map=override.reasoning_effort_map or detected.reasoning_effort_map,
        supports_usage_in_streaming=override.supports_usage_in_streaming,
        max_tokens_field=override.max_tokens_field,
        requires_tool_result_name=override.requires_tool_result_name,
        requires_assistant_after_tool_result=override.requires_assistant_after_tool_result,
        requires_thinking_as_text=override.requires_thinking_as_text,
        thinking_format=override.thinking_format,
        supports_strict_mode=override.supports_strict_mode,
    )


def map_reasoning_effort(effort: str, reasoning_effort_map: dict[str, str]) -> str:
    return reasoning_effort_map.get(effort, effort)


def parse_streaming_json(value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    except json.JSONDecodeError:
        return {}


def sanitize_surrogates(value: str) -> str:
    return value.encode("utf-16", "surrogatepass").decode("utf-16", "replace")


def to_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none=False)
    raise TypeError(f"Unsupported chunk type: {type(value)!r}")
