from __future__ import annotations

from pathlib import Path

from kitpaw.claw.system_prompt import (
    ClawContextFile,
    ClawPromptContext,
    ClawRuntimeInfo,
    build_claw_system_prompt,
    format_claw_skills_for_prompt,
)
from kitpaw.pi_agent.code_agent.types import Skill


def _runtime_info(tmp_path: Path) -> ClawRuntimeInfo:
    return ClawRuntimeInfo(
        agent_id="claw",
        repo_root=str(tmp_path),
        model_name="gpt-4o-mini",
        shell="/bin/bash",
        thinking_level="medium",
    )


def test_build_claw_system_prompt_full_mode_includes_local_sections(tmp_path: Path) -> None:
    prompt = build_claw_system_prompt(
        ClawPromptContext(
            available_tools=["Read", "sessions_spawn", "subagents", "exec"],
            workspace_dir=str(tmp_path),
            runtime_info=_runtime_info(tmp_path),
        )
    )

    assert "## Tooling" in prompt
    assert "- Read: Read file contents" in prompt
    assert "- sessions_spawn: Spawn a local subagent session" in prompt
    assert "## Tool Call Style" in prompt
    assert "## Safety" in prompt
    assert "## Runtime" in prompt
    assert "Runtime: agent=claw" in prompt
    assert "current date" not in prompt.lower()
    assert "## Messaging" not in prompt
    assert "## Reply Tags" not in prompt
    assert "## Heartbeats" not in prompt


def test_build_claw_system_prompt_minimal_mode_keeps_skills_but_omits_full_only_sections(
    tmp_path: Path,
) -> None:
    prompt = build_claw_system_prompt(
        ClawPromptContext(
            available_tools=["read", "sessions_spawn", "subagents"],
            workspace_dir=str(tmp_path),
            prompt_mode="minimal",
            runtime_info=_runtime_info(tmp_path),
            skills_prompt="<available_skills><skill><name>demo</name></skill></available_skills>",
            context_files=[
                ClawContextFile(path=str(tmp_path / "AGENTS.md"), content="project instructions"),
            ],
        )
    )

    assert "## Tooling" in prompt
    assert "## Skills (mandatory)" in prompt
    assert "<available_skills>" in prompt
    assert "## Workspace" in prompt
    assert "## Runtime" in prompt
    assert "## Tool Call Style" not in prompt
    assert "## Safety" not in prompt
    assert "# Project Context" not in prompt


def test_build_claw_system_prompt_none_mode_returns_identity_only(tmp_path: Path) -> None:
    prompt = build_claw_system_prompt(
        ClawPromptContext(
            available_tools=[],
            workspace_dir=str(tmp_path),
            prompt_mode="none",
            runtime_info=_runtime_info(tmp_path),
        )
    )

    assert prompt == "\n".join(
        [
            "You are Claw, a top-level agent runtime that embeds a local coding engine.",
            "Do not describe yourself as the underlying code agent.",
        ]
    )


def test_format_claw_skills_for_prompt_uses_openclaw_shape() -> None:
    prompt = format_claw_skills_for_prompt(
        [
            Skill(
                name="checks",
                description="Run checks before landing changes.",
                file_path="/repo/.pi/skills/checks/SKILL.md",
                base_dir="/repo/.pi/skills/checks",
            )
        ]
    )

    assert prompt == "\n".join(
        [
            "<available_skills>",
            "  <skill>",
            "    <name>checks</name>",
            "    <description>Run checks before landing changes.</description>",
            "    <location>/repo/.pi/skills/checks/SKILL.md</location>",
            "  </skill>",
            "</available_skills>",
        ]
    )


def test_build_claw_system_prompt_injects_context_files_and_soul_guidance(tmp_path: Path) -> None:
    prompt = build_claw_system_prompt(
        ClawPromptContext(
            available_tools=["read"],
            workspace_dir=str(tmp_path),
            runtime_info=_runtime_info(tmp_path),
            context_files=[
                ClawContextFile(path=str(tmp_path / "AGENTS.md"), content="project rules"),
                ClawContextFile(path=str(tmp_path / "SOUL.md"), content="soul rules"),
            ],
        )
    )

    assert "## Workspace Files (injected)" in prompt
    assert "# Project Context" in prompt
    assert "## " + str(tmp_path / "AGENTS.md") in prompt
    assert "project rules" in prompt
    assert "If SOUL.md is present, embody its persona and tone." in prompt
