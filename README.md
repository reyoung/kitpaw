## paw

This repository now contains a standard Python project and the initial Python port of
`pi-mono/packages/ai` for the OpenAI Chat Completions interface.

## Async API

`paw.pi_agent.ai` is asyncio-first. Use `stream` / `complete`.

```python
import asyncio

from paw.pi_agent.ai import Context, UserMessage, complete, stream, get_model


async def main() -> None:
    model = get_model("openai", "gpt-4o-mini")

    response_stream = stream(
        model,
        Context(messages=[UserMessage(content="Say hello")]),
        {"max_tokens": 64},
    )
    async for event in response_stream:
        print(event.type)

    result = await complete(
        model,
        Context(messages=[UserMessage(content="Say hello")]),
        {"max_tokens": 64},
    )
    print(result)


asyncio.run(main())
```

## HTTP Client Reuse

`stream()` and `complete()` accept an optional `http_client_factory` in `options`.
Use it when the caller wants to reuse a shared `httpx.AsyncClient` and its connection pool.

If `http_client_factory` is omitted, the library creates and closes a fresh client per call.
If `http_client_factory` is provided, the library uses the returned client and does not close it.

## Agent Runtime

`paw.pi_agent.agent` provides the higher-level agent runtime on top of `paw.pi_agent.ai`.

```python
import asyncio

from paw.pi_agent.agent import Agent, AgentTool, AgentToolResult
from paw.pi_agent.ai import TextContent


async def main() -> None:
    agent = Agent()
    agent.set_system_prompt("You are a concise assistant.")

    agent.set_tools(
        [
            AgentTool(
                name="echo",
                label="Echo",
                description="Return the provided text",
                parameters={
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
                execute=lambda _tool_call_id, args, *_: AgentToolResult(
                    content=[TextContent(text=args["text"])],
                    details=None,
                ),
            )
        ]
    )

    await agent.prompt("Say hello")
    print(agent.state.messages[-1])


asyncio.run(main())
```

Use `agent.continue_()` to resume from the last non-assistant message, or `agent.follow_up()`
and `agent.steer()` to queue additional messages for the next turn.

## Code Agent

`paw.pi_agent.code_agent` is the Python port of the `pi` coding-agent package. It currently
ships:

- SDK session creation via `create_agent_session()`
- built-in coding tools (`read`, `bash`, `edit`, `write`, `grep`, `find`, `ls`)
- `pi` CLI entrypoint with print, JSON, RPC, and minimal interactive modes

Run a print-mode smoke:

```bash
pi -p "Reply with exactly the word pong."
```

Run JSON mode:

```bash
pi --mode json "List the files in this repository"
```

Run RPC mode:

```bash
pi --mode rpc
```

## Local environment

Put local credentials in the repository root `.env.local`. The package and tests load
this file automatically when present.

Recommended `.env.local`:

```dotenv
OPENAI_BASE_URL=https://open.bigmodel.cn/api/coding/paas/v4
OPENAI_API_KEY=your-local-key
OPENAI_MODEL=glm-4.7
OPENAI_FALLBACK_MODEL=glm-4.5
```

`.env.local` is ignored by git and must not be committed.

## Commands

Install dependencies and run the default test suite:

```bash
uv run pytest
```

Run the real upstream smoke test explicitly:

```bash
PAW_RUN_REAL_E2E=1 uv run pytest -m real_e2e
```
