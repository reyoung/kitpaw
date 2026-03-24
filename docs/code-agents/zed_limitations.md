# Zed Agent Limitations

`pi --agent zed` currently provides a Zed-flavored system prompt and a set of Zed-compatible tool
names on top of the existing Python `code_agent` runtime.

This is intentionally not a full port of Zed's native agent runtime.

## Known Limitation: Runtime Architecture Is Not Ported

The current implementation does not port Zed's core agent runtime layers such as:

- thread / session lifecycle management built around Zed's native thread model
- ACP/UI event integration
- project-context and context-server runtime integration
- buffer/editor-backed tool execution

Instead, `--agent zed` runs inside the existing Python `AgentSession` and `Agent` stack and swaps
in:

- a Zed-style system prompt
- a Zed-style resource loader
- a Zed-compatible tool surface where feasible in CLI mode

## Practical Implications

Because the runtime architecture is not ported, `--agent zed` should currently be treated as a
compatibility layer rather than behavioral parity with upstream Zed. In particular:

- tool behavior may be CLI-oriented approximations rather than editor-integrated behavior
- permission flows, UI metadata, and tool replay semantics are not fully aligned
- project/worktree/context handling is narrower than upstream Zed
- some upstream Zed agent features are intentionally unavailable in this repository

This is a known gap, not an accidental regression.
