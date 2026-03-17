from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

from paw.pi_agent.ai import (
    Context,
    TextContent,
    Tool,
    ToolCallEndEvent,
    UserMessage,
    complete,
    stream,
    get_model,
)
from paw.pi_agent.ai.providers.openai_completions import create_client
from paw.pi_agent.ai.types import StreamOptions


def make_chunk(*, delta: dict, finish_reason: str | None = None, usage: dict | None = None) -> dict:
    chunk = {
        "id": "chatcmpl-mock",
        "object": "chat.completion.chunk",
        "created": 1_710_000_000,
        "model": "mock-model",
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }
    if usage is not None:
        chunk["usage"] = usage
    return chunk


@contextmanager
def run_mock_openai_server(chunks: list[dict]):
    state: dict[str, object] = {"request": None}

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(length).decode("utf-8")
            state["request"] = json.loads(raw_body)

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()

            for chunk in chunks:
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode("utf-8"))
                self.wfile.flush()

            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

        def log_message(self, format: str, *args) -> None:  # noqa: A003
            return None

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/v1", state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.anyio
async def test_complete_mock_e2e_text_response() -> None:
    usage = {
        "prompt_tokens": 11,
        "completion_tokens": 7,
        "prompt_tokens_details": {"cached_tokens": 2},
        "completion_tokens_details": {"reasoning_tokens": 3},
    }
    with run_mock_openai_server(
        [
            make_chunk(delta={"content": "hello"}),
            make_chunk(delta={}, finish_reason="stop", usage=usage),
        ]
    ) as (base_url, state):
        model = replace(get_model("openai", "gpt-4o-mini"), base_url=base_url)
        result = await complete(
            model,
            Context(messages=[UserMessage(content="Say hello")]),
            {"api_key": "test-key", "max_tokens": 32},
        )

    request = state["request"]
    assert isinstance(request, dict)
    assert request["stream"] is True
    assert request["max_completion_tokens"] == 32
    assert result.stop_reason == "stop"
    assert [block.text for block in result.content if isinstance(block, TextContent)] == ["hello"]
    assert result.usage.input == 9
    assert result.usage.output == 10
    assert result.usage.total_tokens == 21


@pytest.mark.anyio
async def test_stream_mock_e2e_reasoning_and_tool_call() -> None:
    usage = {
        "prompt_tokens": 12,
        "completion_tokens": 6,
        "prompt_tokens_details": {"cached_tokens": 0},
        "completion_tokens_details": {"reasoning_tokens": 2},
    }
    chunks = [
        make_chunk(delta={"reasoning": "plan"}),
        make_chunk(
            delta={
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": "{\"city\":\"Bei"},
                    }
                ]
            }
        ),
        make_chunk(
            delta={
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"arguments": "jing\"}"},
                    }
                ],
                "reasoning_details": [
                    {"type": "reasoning.encrypted", "id": "call_1", "data": "opaque"}
                ],
            },
            finish_reason="tool_calls",
            usage=usage,
        ),
    ]

    with run_mock_openai_server(chunks) as (base_url, state):
        model = replace(get_model("openai", "gpt-4o-mini"), base_url=base_url)
        response_stream = stream(
            model,
            Context(
                messages=[UserMessage(content="Look up Beijing")],
                tools=[
                    Tool(
                        name="lookup",
                        description="Lookup weather",
                        parameters={
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    )
                ],
            ),
            {"api_key": "test-key", "reasoning": "medium"},
        )
        events = [event async for event in response_stream]
        result = await response_stream.result()

    request = state["request"]
    assert isinstance(request, dict)
    assert request["reasoning_effort"] == "medium"
    assert request["tools"][0]["function"]["strict"] is False
    assert [event.type for event in events] == [
        "start",
        "thinking_start",
        "thinking_delta",
        "thinking_end",
        "toolcall_start",
        "toolcall_delta",
        "toolcall_delta",
        "toolcall_end",
        "done",
    ]
    assert result.stop_reason == "toolUse"
    tool_calls = [block for block in result.content if getattr(block, "type", None) == "toolCall"]
    assert len(tool_calls) == 1
    assert tool_calls[0].arguments == {"city": "Beijing"}
    assert tool_calls[0].thought_signature is not None
    assert isinstance(events[-2], ToolCallEndEvent)


class TrackingAsyncClient(httpx.AsyncClient):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.closed_calls = 0

    async def aclose(self) -> None:
        self.closed_calls += 1
        await super().aclose()


@pytest.mark.anyio
async def test_create_client_uses_factory_without_owning_client() -> None:
    model = get_model("openai", "gpt-4o-mini")
    provided_client = TrackingAsyncClient(base_url=model.base_url)

    def factory(_model, _options):
        return provided_client

    client, owned_http_client = await create_client(
        model,
        StreamOptions(api_key="test-key", http_client_factory=factory),
        "test-key",
    )

    assert owned_http_client is None
    assert client._client is provided_client
    assert provided_client.closed_calls == 0
    await provided_client.aclose()
    assert provided_client.closed_calls == 1


@pytest.mark.anyio
async def test_create_client_supports_async_factory() -> None:
    model = get_model("openai", "gpt-4o-mini")
    provided_client = TrackingAsyncClient(base_url=model.base_url)

    async def factory(_model, _options):
        return provided_client

    client, owned_http_client = await create_client(
        model,
        StreamOptions(api_key="test-key", http_client_factory=factory),
        "test-key",
    )

    assert owned_http_client is None
    assert client._client is provided_client
    assert provided_client.closed_calls == 0
    await provided_client.aclose()
    assert provided_client.closed_calls == 1


@pytest.mark.anyio
async def test_create_client_owns_default_client() -> None:
    model = get_model("openai", "gpt-4o-mini")

    client, owned_http_client = await create_client(
        model,
        StreamOptions(api_key="test-key"),
        "test-key",
    )

    assert owned_http_client is not None
    assert client._client is owned_http_client
    await owned_http_client.aclose()


@pytest.mark.anyio
async def test_stream_does_not_close_factory_client() -> None:
    usage = {
        "prompt_tokens": 4,
        "completion_tokens": 2,
        "prompt_tokens_details": {"cached_tokens": 0},
        "completion_tokens_details": {"reasoning_tokens": 0},
    }
    with run_mock_openai_server(
        [
            make_chunk(delta={"content": "ok"}),
            make_chunk(delta={}, finish_reason="stop", usage=usage),
        ]
    ) as (base_url, _state):
        model = replace(get_model("openai", "gpt-4o-mini"), base_url=base_url)
        shared_client = TrackingAsyncClient(base_url=base_url)

        def factory(_model, _options):
            return shared_client

        response = await complete(
            model,
            Context(messages=[UserMessage(content="ping")]),
            {"api_key": "test-key", "http_client_factory": factory},
        )

        assert response.stop_reason == "stop"
        assert shared_client.closed_calls == 0
        assert shared_client.is_closed is False
        await shared_client.aclose()
