# kitpaw TODO

## High Priority

### 1. Support OpenAI Responses API protocol
Currently kitpaw only speaks the Chat Completions API (`/v1/chat/completions`). Codex Rust has fully migrated to the Responses API (`/v1/responses`) as of v0.95.0-alpha.6. Adding Responses API support would enable:
- Direct e2e alignment with latest Codex Rust
- WebSocket streaming support
- Native tool result streaming

### 2. Verify Codex context compactor correctness
The Codex compaction hook uses a custom prompt ("CONTEXT CHECKPOINT COMPACTION") and summary prefix, but hasn't been validated against Codex Rust's actual compaction behavior:
- Compare compaction trigger thresholds (token estimation heuristics)
- Verify summary format matches what Codex Rust produces
- Test multi-turn conversations where compaction fires mid-session
- Check that the summary prefix ("Another language model started to solve this problem...") is correctly prepended when resuming

## Medium Priority

### 3. Port Gemini agent (`--agent gemini`)
Add a Gemini-style agent mode, similar to how Zed and Codex were ported. Investigate:
- Gemini's system prompt and tool conventions
- Native function calling format differences
- Grounding / search integration

### 4. Port OpenCode agent (`--agent opencode`)
[OpenCode](https://github.com/nicholasgasior/opencode) is another terminal-based coding agent. Port its:
- System prompt and tool definitions
- Session management approach
- Any unique compaction or context handling
