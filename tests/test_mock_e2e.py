from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from paw.pi_agent.ai import Context, TextContent, Tool, UserMessage, complete, get_model, stream_simple


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


def test_complete_mock_e2e_text_response() -> None:
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
        result = complete(
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


def test_stream_simple_mock_e2e_reasoning_and_tool_call() -> None:
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
        stream = stream_simple(
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
        events = list(stream)
        result = stream.result()

    request = state["request"]
    assert isinstance(request, dict)
    assert request["reasoning_effort"] == "medium"
    assert request["tools"][0]["function"]["strict"] is False
    assert [event["type"] for event in events] == [
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
