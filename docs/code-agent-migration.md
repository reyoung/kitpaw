# Python Code Agent Migration

## Summary

This repository now contains an initial Python port of `pi-mono/packages/coding-agent`
under `paw.pi_agent.code_agent`.

The port keeps the `pi` CLI shape and `.pi` / `~/.pi/agent` layout, but currently limits
provider support to OpenAI-compatible endpoints through `paw.pi_agent.ai`.

## Implemented Surface

- `paw.pi_agent.code_agent` SDK exports for session creation and built-in tools
- `pi` CLI entrypoint via `pyproject.toml`
- built-in coding tools: `read`, `bash`, `edit`, `write`, `grep`, `find`, `ls`
- session persistence to JSONL under `~/.pi/agent/sessions/...`
- settings, auth, model registry, and resource discovery scaffolding
- print mode, JSON event stream mode, RPC mode, and a minimal interactive REPL
- real OpenAI-compatible smoke coverage for the CLI print path

## Intentional Differences From TS

- extensions are Python modules, not TypeScript modules
- package management is scaffolded through settings/resource loading, but npm/git TS
  package parity is not implemented yet
- the interactive mode is currently a minimal REPL, not the full TS TUI/session UI stack
- branch/tree/compaction/session-navigation parity is not implemented yet

## Validation

- `uv run pytest`
- `PAW_RUN_REAL_E2E=1 uv run pytest -m real_e2e`
- `uv run ruff check paw/pi_agent/code_agent tests/test_code_agent_tools.py tests/test_code_agent_session.py tests/test_code_agent_rpc.py tests/test_code_agent_real_e2e.py`
