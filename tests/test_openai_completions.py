from __future__ import annotations

from dataclasses import replace

from kitpaw.pi_agent.ai import (
    Context,
    ImageContent,
    TextContent,
    Tool,
    ToolCall,
    ToolResultMessage,
    UserMessage,
    get_model,
)
from kitpaw.pi_agent.ai.providers.openai_completions import (
    build_params,
    convert_messages,
    convert_tools,
)
from kitpaw.pi_agent.ai.types import (
    AssistantMessage,
    OpenAICompletionsCompat,
    OpenAICompletionsOptions,
)


def test_build_params_forwards_tool_choice_and_reasoning_effort() -> None:
    model = replace(get_model("openai", "gpt-4o-mini"), base_url="https://api.openai.com/v1")
    context = Context(
        messages=[UserMessage(content="Call ping")],
        tools=[
            Tool(
                name="ping",
                description="Ping tool",
                parameters={
                    "type": "object",
                    "properties": {"ok": {"type": "boolean"}},
                    "required": ["ok"],
                },
            )
        ],
    )

    params = build_params(
        model,
        context,
        OpenAICompletionsOptions(tool_choice="required", reasoning_effort="medium"),
    )

    assert params["tool_choice"] == "required"
    assert params["reasoning_effort"] == "medium"
    assert params["tools"][0]["function"]["strict"] is False


def test_convert_tools_omits_strict_when_compat_disables_it() -> None:
    compat = OpenAICompletionsCompat(supports_strict_mode=False)
    converted = convert_tools(
        [
            Tool(
                name="ping",
                description="Ping tool",
                parameters={"type": "object", "properties": {}, "required": []},
            )
        ],
        compat,
    )

    assert "strict" not in converted[0]["function"]


def test_convert_messages_batches_tool_result_images() -> None:
    model = get_model("openai", "gpt-4o-mini")
    compat = OpenAICompletionsCompat()
    assistant = AssistantMessage(
        api=model.api,
        provider=model.provider,
        model=model.id,
        content=[
            ToolCall(id="tool-1", name="read", arguments={"path": "img-1.png"}),
            ToolCall(id="tool-2", name="read", arguments={"path": "img-2.png"}),
        ],
        stop_reason="toolUse",
    )
    context = Context(
        messages=[
            UserMessage(content="Read the images"),
            assistant,
            ToolResultMessage(
                tool_call_id="tool-1",
                tool_name="read",
                content=[
                    TextContent(text="Read image 1"),
                    ImageContent(data="ZmFrZQ==", mime_type="image/png"),
                ],
            ),
            ToolResultMessage(
                tool_call_id="tool-2",
                tool_name="read",
                content=[
                    TextContent(text="Read image 2"),
                    ImageContent(data="ZmFrZQ==", mime_type="image/png"),
                ],
            ),
        ]
    )

    messages = convert_messages(model, context, compat)

    assert [message["role"] for message in messages] == [
        "user",
        "assistant",
        "tool",
        "tool",
        "user",
    ]
    image_message = messages[-1]
    image_parts = [part for part in image_message["content"] if part["type"] == "image_url"]
    assert len(image_parts) == 2


def test_build_params_uses_max_tokens_for_bigmodel_compat() -> None:
    model = replace(
        get_model("openai", "glm-4.7"), base_url="https://open.bigmodel.cn/api/coding/paas/v4"
    )
    params = build_params(
        model,
        Context(messages=[UserMessage(content="hi")]),
        OpenAICompletionsOptions(max_tokens=64),
    )

    assert params["max_tokens"] == 64
    assert "max_completion_tokens" not in params
