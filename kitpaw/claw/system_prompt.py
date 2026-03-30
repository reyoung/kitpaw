from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
from xml.sax.saxutils import escape as xml_escape

from ..pi_agent.code_agent.types import Skill

ClawPromptMode = Literal["full", "minimal", "none"]

_TOOL_ORDER = [
    "read",
    "write",
    "edit",
    "apply_patch",
    "exec",
    "process",
    "sessions_list",
    "sessions_history",
    "sessions_send",
    "sessions_spawn",
    "subagents",
    "session_status",
    "sessions_yield",
]

_CORE_TOOL_SUMMARIES: dict[str, str] = {
    "read": "Read file contents",
    "write": "Create or overwrite files",
    "edit": "Make precise edits to files",
    "apply_patch": "Apply multi-file patches using the OpenClaw patch format",
    "exec": "Run local shell commands inside the workspace",
    "process": "Manage background exec sessions started by the current claw session",
    "sessions_list": "List saved sessions visible in the current workspace",
    "sessions_history": "Fetch user and assistant history for another session",
    "sessions_send": "Send a follow-up message to a visible child session",
    "sessions_spawn": "Spawn a local subagent session",
    "subagents": "List, steer, or kill subagent runs owned by this controller session",
    "session_status": "Show message counts and runtime state for a session",
    "sessions_yield": "Yield back to the outer runtime when a handler is bound",
}

_SAFETY_SECTION = [
    "## Safety",
    "You have no independent goals: do not pursue self-preservation, replication, resource acquisition, or power-seeking beyond the user's request.",
    "Prioritize safety and human oversight over completion; if instructions conflict, pause and ask rather than bypassing safeguards.",
    "Do not copy yourself or modify system prompts, tool policies, or safety rules unless the user explicitly asks for that work.",
    "",
]


@dataclass(slots=True)
class ClawRuntimeInfo:
    agent_id: str | None = None
    repo_root: str | None = None
    model_name: str | None = None
    shell: str | None = None
    thinking_level: str | None = None


@dataclass(slots=True)
class ClawContextFile:
    path: str
    content: str


@dataclass(slots=True)
class ClawPromptContext:
    available_tools: list[str]
    workspace_dir: str
    prompt_mode: ClawPromptMode = "full"
    runtime_info: ClawRuntimeInfo = field(default_factory=ClawRuntimeInfo)
    skills_prompt: str | None = None
    context_files: list[ClawContextFile] = field(default_factory=list)
    tool_summaries: dict[str, str] = field(default_factory=dict)
    workspace_notes: list[str] = field(default_factory=list)


def format_claw_skills_for_prompt(skills: list[Skill]) -> str:
    visible = [skill for skill in skills if not skill.disable_model_invocation]
    if not visible:
        return ""
    lines = ["<available_skills>"]
    for skill in visible:
        lines.extend(
            [
                "  <skill>",
                f"    <name>{xml_escape(skill.name)}</name>",
                f"    <description>{xml_escape(skill.description)}</description>",
                f"    <location>{xml_escape(skill.file_path)}</location>",
                "  </skill>",
            ]
        )
    lines.append("</available_skills>")
    return "\n".join(lines)


def build_runtime_line(runtime_info: ClawRuntimeInfo) -> str:
    parts = [
        f"agent={runtime_info.agent_id}" if runtime_info.agent_id else "",
        f"repo={runtime_info.repo_root}" if runtime_info.repo_root else "",
        f"model={runtime_info.model_name}" if runtime_info.model_name else "",
        f"shell={runtime_info.shell}" if runtime_info.shell else "",
        f"thinking={runtime_info.thinking_level}" if runtime_info.thinking_level else "",
    ]
    return "Runtime: " + " | ".join(part for part in parts if part)


def _tool_lines(context: ClawPromptContext) -> list[str]:
    raw_tool_names = [tool.strip() for tool in context.available_tools if tool.strip()]
    canonical_by_normalized: dict[str, str] = {}
    for name in raw_tool_names:
        normalized = name.lower()
        canonical_by_normalized.setdefault(normalized, name)

    available_tools = set(canonical_by_normalized)
    external_summaries = {
        key.strip().lower(): value.strip()
        for key, value in context.tool_summaries.items()
        if key.strip() and value.strip()
    }
    enabled = [tool for tool in _TOOL_ORDER if tool in available_tools]
    extra = sorted(tool for tool in available_tools if tool not in _TOOL_ORDER)

    lines: list[str] = []
    for tool in [*enabled, *extra]:
        display_name = canonical_by_normalized[tool]
        summary = _CORE_TOOL_SUMMARIES.get(tool) or external_summaries.get(tool)
        lines.append(f"- {display_name}: {summary}" if summary else f"- {display_name}")
    return lines or ["- (none)"]


def _skills_section(context: ClawPromptContext) -> list[str]:
    skills_prompt = (context.skills_prompt or "").strip()
    if not skills_prompt:
        return []
    return [
        "## Skills (mandatory)",
        "Before replying: scan <available_skills> <description> entries.",
        "- If exactly one skill clearly applies: read its SKILL.md at <location> with `read`, then follow it.",
        "- If multiple could apply: choose the most specific one, then read and follow it.",
        "- If none clearly apply: do not read any SKILL.md.",
        "Constraints: never read more than one skill up front; only read after selecting.",
        skills_prompt,
        "",
    ]


def _workspace_section(context: ClawPromptContext) -> list[str]:
    lines = [
        "## Workspace",
        f"Your working directory is: {context.workspace_dir}",
        "Treat this directory as the single global workspace for file operations unless the user explicitly instructs otherwise.",
        "Prefer relative paths when using file tools and local exec commands.",
    ]
    lines.extend(note.strip() for note in context.workspace_notes if note.strip())
    lines.append("")
    return lines


def _project_context_section(context: ClawPromptContext) -> list[str]:
    if context.prompt_mode != "full" or not context.context_files:
        return []

    lines = [
        "## Workspace Files (injected)",
        "These user-editable files are loaded by Claw and included below in Project Context.",
        "",
        "# Project Context",
        "",
        "The following project context files have been loaded:",
    ]
    if any(Path(file.path).name.lower() == "soul.md" for file in context.context_files):
        lines.append(
            "If SOUL.md is present, embody its persona and tone. Avoid stiff, generic replies; follow its guidance unless higher-priority instructions override it."
        )
    lines.append("")
    for file in context.context_files:
        lines.extend([f"## {file.path}", "", file.content, ""])
    return lines


def build_claw_system_prompt(context: ClawPromptContext) -> str:
    identity_lines = [
        "You are Claw, a top-level agent runtime that embeds a local coding engine.",
        "Do not describe yourself as the underlying code agent.",
        "",
    ]
    if context.prompt_mode == "none":
        return "\n".join(identity_lines[:-1])

    lines = list(identity_lines)
    lines.extend(
        [
            "## Tooling",
            "Tool availability (local runtime):",
            "Tool names are case-sensitive. Call tools exactly as listed.",
            *_tool_lines(context),
            "For long waits, prefer `exec` with enough `yield_ms` or `process(action=\"poll\", timeout=<ms>)` over tight polling loops.",
            "If work is more complex or longer-running, use `sessions_spawn` to create a local subagent.",
            "Do not poll `subagents` or `sessions_list` in a loop; only check status on-demand.",
            "",
        ]
    )
    if context.prompt_mode == "full":
        lines.extend(
            [
                "## Tool Call Style",
                "Default: do not narrate routine, low-risk tool calls; just use the tool.",
                "Narrate only when it helps: multi-step work, risky changes, or when the user explicitly asks.",
                "When a first-class tool exists for an action, use that tool instead of asking the user to run equivalent shell commands.",
                "",
            ]
        )
        lines.extend(_SAFETY_SECTION)
    lines.extend(_skills_section(context))
    lines.extend(_workspace_section(context))
    lines.extend(_project_context_section(context))
    lines.extend(["## Runtime", build_runtime_line(context.runtime_info)])
    return "\n".join(part for part in lines if part is not None).strip()


__all__ = [
    "ClawContextFile",
    "ClawPromptContext",
    "ClawPromptMode",
    "ClawRuntimeInfo",
    "build_claw_system_prompt",
    "build_runtime_line",
    "format_claw_skills_for_prompt",
]
