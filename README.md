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
