# Claw Python Runtime Migration

## Goal

Turn `kitpaw.claw` into a top-level runtime that embeds the existing
Python coding engine, instead of treating it as another
`pi_agent.code_agent` variant.

The first runnable `claw` surface is still the local orchestration layer:

- `sessions_list`
- `sessions_history`
- `session_status`
- `sessions_spawn`
- `sessions_send`
- `sessions_yield`
- `subagents`

## Non-goals

- No `gateway` integration
- No channel/message delivery
- No plugin tool loading
- No ACP runtime
- No OpenClaw-specific exec sandbox/policy pipeline
- No `pi --agent claw`

## Public Interface

Expose a top-level package:

- `kitpaw.claw`

Export:

- `OpenClawToolContext`
- `ClawResourceLoader`
- `CreateClawSessionOptions`
- `CreateClawSessionResult`
- `create_openclaw_coding_tools`
- `create_claw_session`

Add runnable entrypoints:

- `claw`
- `python -m kitpaw.claw`

`create_claw_session()` is the top-level runtime factory. It owns:

- session creation / resume
- `ClawResourceLoader`
- final tool binding for `kitpaw.claw`
- final system-prompt binding for `kitpaw.claw`

Internally it may still reuse:

- `SessionManager`
- `create_agent_session()`
- the existing text/json/rpc mode runners

That reuse is an implementation detail, not the public runtime boundary.

## Runtime Semantics

### Runtime boundary

`claw` is not a code-agent mode.

- `pi` keeps its current `pi | zed | codex` split.
- `claw` is a separate runtime with its own entrypoint.
- the coding engine is embedded under `claw`, analogous to how OpenClaw
  embeds `pi-coding-agent`.

### Session setup

`create_claw_session()` creates or resolves a session manager first so
the runtime has a stable session id before the final tool binding.

Session setup happens in two phases:

1. Create a temporary `OpenClawToolContext` and temporary tool set.
2. Create the session and compute the final `system_prompt`, `model`,
   and `thinking_level`.
3. Rebuild the `OpenClawToolContext` and rebind the final tool set.

This final rebind is required so spawned child sessions inherit the real
parent prompt/model state instead of provisional defaults.

### Prompt / resource loading

`ClawResourceLoader` is responsible for `claw` prompt assembly.

It delegates skills/prompts/themes/extensions/AGENTS discovery to the
existing default loader, but builds a `claw`-specific prompt that:

- identifies the runtime as `claw`
- describes the current workspace
- lists the available tools
- explains local-only orchestration semantics
- does not advertise unimplemented OpenClaw features

### Session visibility

Visibility remains local-only and controller-scoped.

- a controller session can see the child sessions it spawned
- a session cannot enumerate or send into unrelated sessions
- no agent-to-agent or global label lookup exists in this change

### Spawn semantics

`sessions_spawn` still supports only `runtime="subagent"`.

- `mode="run"` is one-shot
- `mode="session"` keeps a persistent child session
- `cleanup="delete"` removes only the runtime registry handle

Child sessions inherit:

- model provider/id
- thinking level
- system prompt
- the tool set produced by `create_openclaw_coding_tools`

## Storage Model

Persisted conversation state continues to use the existing JSONL session
files managed by `SessionManager`.

Subagent registry state remains runtime-only and process-local.

## Testing Strategy

Add coverage for:

- `ClawResourceLoader` prompt generation
- `create_claw_session()` tool binding and final prompt binding
- resumed sessions rebinding `claw` tools
- `python -m kitpaw.claw --no-session`
- CLI rejection of `--agent`

Keep the existing orchestration coverage in `tests/claw`.
