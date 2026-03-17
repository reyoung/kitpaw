# Repository Notes

## Local credentials

- Use the repository root `.env.local` for local-only credentials.
- `.env.local` is gitignored and should never be committed.
- `paw.pi_agent.ai` automatically loads `.env.local` when it exists.
- Prefer `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL`, and `OPENAI_FALLBACK_MODEL`.

## Verification

- `uv run pytest` runs unit tests and the local mock-server end-to-end tests.
- `PAW_RUN_REAL_E2E=1 uv run pytest -m real_e2e` runs the real OpenAI-compatible smoke test.

## AI API shape

- `paw.pi_agent.ai` is asyncio-first.
- Prefer `stream` and `complete`.
- `AssistantMessageEventStream` is an async iterator of dataclass event objects.
- `stream` / `complete` accept optional `http_client_factory` for shared `httpx.AsyncClient` injection.
- The library closes only clients it created itself; caller-provided clients are caller-owned.
