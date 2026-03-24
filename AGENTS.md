# Repository Notes

## Local credentials

- Use `.kitpaw` in the repository root or CWD for local-only credentials.
- CWD `.kitpaw` takes precedence over repo-root `.kitpaw`.
- `.kitpaw` is gitignored and should never be committed.
- `paw.pi_agent.ai` automatically loads `.kitpaw` when it exists.
- Prefer `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL`, and `OPENAI_FALLBACK_MODEL`.

## Verification

- `uv run pytest` runs unit tests and the local mock-server end-to-end tests.
- `PAW_RUN_REAL_E2E=1 uv run pytest -m real_e2e` runs the real OpenAI-compatible smoke test.

## Development workflow

- For substantial work, proceed in this order: design document, interfaces, unit tests, implementation.
- Do not treat implementation as complete until the relevant tests have been run and are passing.
- Only deliver work after confirming the full required test suite passes for that change.

## AI API shape

- `paw.pi_agent.ai` is asyncio-first.
- Prefer `stream` and `complete`.
- `AssistantMessageEventStream` is an async iterator of dataclass event objects.
- `stream` / `complete` accept optional `http_client_factory` for shared `httpx.AsyncClient` injection.
- The library closes only clients it created itself; caller-provided clients are caller-owned.


## Autonomy Policy
If you're working towards goals, do NOT end your turn. This allows for continuous autonomous work.

The user will interrupt you when required, but they will mostly provide steering messages.

Do not pester the user by ending your turn after a unit of work, as that requires them to keep nudging you to keep working.

You MUST continue working autonomously towards any known objectives until the user interrupts you. Do NOT end your turn until there is absolutely nothing left to do.

