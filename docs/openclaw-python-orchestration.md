# OpenClaw Python Orchestration Migration

## Goal

Add a Python-native OpenClaw orchestration layer on top of
`kitpaw.pi_agent.code_agent` without pulling in OpenClaw's gateway,
channel, media, or plugin runtime.

The first migration target is the local multi-session orchestration
surface:

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
- No new CLI mode in this change

## Public Interface

Expose a new package:

- `kitpaw.claw`

Export:

- `OpenClawToolContext`
- `create_openclaw_coding_tools`

`OpenClawToolContext` carries the runtime facts needed to create child
sessions and report orchestration events:

- `cwd`
- `workspace_dir`
- `spawn_workspace_dir`
- `agent_id`
- `session_id`
- `controller_session_id`
- `model_provider`
- `model_id`
- `thinking_level`
- `sandboxed`
- `system_prompt`
- `on_yield`

The extra `system_prompt` field is required so spawned sessions can
inherit the parent prompt exactly instead of rebuilding it heuristically.

## Runtime Semantics

### Session visibility

Visibility is local-only and controller-scoped.

- A controller session can see the child sessions it spawned.
- A session cannot enumerate or send into unrelated sessions.
- No agent-to-agent or global label lookup exists in this change.

### Spawn semantics

`sessions_spawn` supports only `runtime="subagent"`.

- `mode="run"` creates or reuses a child session for a single task run
  and returns the result after the run completes.
- `mode="session"` creates a persistent child session and keeps it in the
  registry for later `sessions_send` and `subagents` actions.
- `cleanup="delete"` removes only the in-memory registry handle after the
  run. Session files are retained.

Spawned sessions inherit:

- model provider/id
- thinking level
- system prompt
- tool set produced by `create_openclaw_coding_tools`

The child session workspace uses `spawn_workspace_dir` when present and
falls back to `workspace_dir`.

### Subagent control

`subagents` exposes:

- `list`
- `steer`
- `kill`

`kill` aborts an active run and removes the child from the active
registry. It does not delete persisted session files.

### Yield semantics

`sessions_yield` is a structured interrupt point.

- If `on_yield` is not bound, it returns a success payload describing the
  no-op.
- If `on_yield` is bound, it calls the callback and aborts the current run
  by raising `asyncio.CancelledError`.

## Storage Model

Persisted conversation state continues to use the existing JSONL session
files managed by `SessionManager`.

Subagent registry state is runtime-only and process-local. It stores:

- controller session id
- child session id
- label
- lifecycle status
- timestamps
- last task/result summary
- active asyncio task handle

The registry is an orchestration index, not a source of truth for message
history.

## Testing Strategy

Add unit/integration coverage for:

- tool inventory and schemas
- session listing/history/status
- `sessions_spawn` for `run` and `session`
- `sessions_send` into persistent child sessions
- `subagents list/steer/kill`
- `sessions_yield` callback + abort behavior

Use the existing mock OpenAI streaming server for child-session runs.
