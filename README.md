## paw

This repository now contains a standard Python project and the initial Python port of
`pi-mono/packages/ai` for the OpenAI Chat Completions interface.

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
