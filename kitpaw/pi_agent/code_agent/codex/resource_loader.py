from __future__ import annotations

import os
from pathlib import Path

from ..resource_loader import DefaultResourceLoader
from ..system_prompt import format_skills_for_prompt
from ..types import (
    LoadedAgentsFiles,
    LoadedExtensions,
    LoadedPrompts,
    LoadedSkills,
    LoadedThemes,
    Skill,
)
from .system_prompt import build_codex_system_prompt

# Sandbox permissions text matching Codex Rust's developer instructions.
_PERMISSIONS_TEXT = """\
<permissions instructions>
Filesystem sandboxing defines which files can be read or written. \
`sandbox_mode` is `workspace-write`: The sandbox permits reading files, \
and editing files in `cwd` and `writable_roots`. Editing files in other \
directories requires approval. Network access is disabled.

Approvals are your mechanism to get user consent to run shell commands \
without the sandbox. `approval_policy` is `never`: This is a non-interactive \
mode where you may NEVER ask the user for approval to run commands. Instead, \
you must always persist and work around constraints to solve the task for \
the user. You MUST do your utmost best to finish the task and validate your \
work before yielding.
</permissions instructions>"""


class CodexResourceLoader:
    """Resource loader that produces the Codex-flavoured system prompt.

    To match Codex Rust's message structure, the system prompt does NOT
    include AGENTS.md content.  Instead, AGENTS.md is provided as a
    separate user message via :meth:`get_agents_md_message`, and sandbox
    permissions are provided via :meth:`get_permissions_message`.
    """

    def __init__(self, cwd: str, agent_dir: str, settings_manager) -> None:
        self._delegate = DefaultResourceLoader(cwd, agent_dir, settings_manager)
        self._cwd = cwd
        self._tool_names: list[str] = []

    def set_tool_names(self, names: list[str]) -> None:
        self._tool_names = list(names)

    # --- Delegated properties / methods ---

    @property
    def agent_dir(self) -> Path:
        return self._delegate.agent_dir

    async def reload(self) -> None:
        await self._delegate.reload()

    def get_skills(self) -> LoadedSkills:
        return self._delegate.get_skills()

    def get_prompts(self) -> LoadedPrompts:
        return self._delegate.get_prompts()

    def get_themes(self) -> LoadedThemes:
        return self._delegate.get_themes()

    def get_extensions(self) -> LoadedExtensions:
        return self._delegate.get_extensions()

    def get_agents_files(self) -> LoadedAgentsFiles:
        return self._delegate.get_agents_files()

    # --- Codex-specific overrides ---

    def format_tool_not_found(self, tool_name: str) -> str:
        return self._delegate.format_tool_not_found(tool_name)

    def get_system_prompt(self) -> str | None:
        return None

    def build_system_prompt(self, base_prompt: str | None, skills: list[Skill]) -> str:
        """Build the base Codex system prompt (WITHOUT AGENTS.md — that is a separate message)."""
        prompt = build_codex_system_prompt(
            available_tools=self._tool_names,
            cwd=self._cwd,
        )
        skill_text = format_skills_for_prompt(skills)
        if skill_text:
            prompt = prompt + "\n\n" + skill_text
        return prompt

    def build_system_prompt_with_tools(
        self,
        available_tools: list[str],
        *,
        model_name: str | None = None,
    ) -> str:
        prompt = build_codex_system_prompt(
            available_tools=available_tools,
            cwd=self._cwd,
        )
        skill_text = format_skills_for_prompt(self.get_skills().skills)
        if skill_text:
            prompt = prompt + "\n\n" + skill_text
        return prompt

    # --- Extra messages (matching Codex Rust's 5-message structure) ---

    def get_permissions_message(self) -> str:
        """Return the developer-role permissions text."""
        return _PERMISSIONS_TEXT

    def get_agents_md_messages(self) -> list[dict[str, str]]:
        """Return AGENTS.md content as user messages (one per file)."""
        messages: list[dict[str, str]] = []
        for path_str in self._delegate.get_agents_files().agents_files:
            try:
                text = Path(path_str).read_text(encoding="utf-8")
                if text.strip():
                    messages.append({
                        "role": "user",
                        "content": (
                            f"# AGENTS.md instructions for {self._cwd}\n\n"
                            f"<INSTRUCTIONS>\n{text.strip()}\n</INSTRUCTIONS>"
                        ),
                    })
            except OSError:
                pass
        return messages

    def get_environment_context_message(self) -> dict[str, str]:
        """Return environment context as a user message."""
        shell = os.environ.get("SHELL", "bash")
        if "/" in shell:
            shell = Path(shell).name
        return {
            "role": "user",
            "content": (
                "<environment_context>\n"
                f"  <cwd>{self._cwd}</cwd>\n"
                f"  <shell>{shell}</shell>\n"
                "</environment_context>"
            ),
        }
