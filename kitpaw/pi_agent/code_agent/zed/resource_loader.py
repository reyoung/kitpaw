from __future__ import annotations

import os
import platform
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
from .system_prompt import build_zed_system_prompt


class ZedResourceLoader:
    """Resource loader that produces the Zed-flavoured system prompt.

    Delegates most behaviour to a ``DefaultResourceLoader`` but overrides
    prompt construction to use the Zed system prompt template.
    """

    def __init__(self, cwd: str, agent_dir: str, settings_manager) -> None:
        self._delegate = DefaultResourceLoader(cwd, agent_dir, settings_manager)
        self._cwd = cwd
        self._worktree_name = Path(cwd).resolve().name
        self._tool_names: list[str] = []

    def set_tool_names(self, names: list[str]) -> None:
        """Set the tool names so that ``build_system_prompt`` can inject them."""
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

    def _get_project_rules(self) -> list[tuple[str, str]]:
        """Collect project rules files (e.g. AGENTS.md) as (path, text) tuples."""
        rules: list[tuple[str, str]] = []
        for path_str in self._delegate.get_agents_files().agents_files:
            try:
                text = Path(path_str).read_text(encoding="utf-8")
                if text.strip():
                    rules.append((path_str, text))
            except OSError:
                pass
        return rules

    # --- Zed-specific overrides ---

    def get_system_prompt(self) -> str | None:
        """Always return ``None`` so the caller uses ``build_system_prompt``.

        In Zed mode, AGENTS.md content is injected as project rules inside
        the system prompt rather than replacing it.
        """
        return None

    def build_system_prompt(self, base_prompt: str | None, skills: list[Skill]) -> str:
        """Build the Zed system prompt with project rules and skills appended.

        *base_prompt* is ignored — in Zed mode the main prompt template is
        always used and AGENTS.md content is injected as project rules under
        ``## User's Custom Instructions``.
        """
        prompt = build_zed_system_prompt(
            available_tools=self._tool_names,
            worktrees=[self._worktree_name],
            os_name=platform.system(),
            shell=os.environ.get("SHELL"),
            project_rules=self._get_project_rules(),
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
        """Build the Zed system prompt with the actual tool list injected.

        This is the preferred entry-point once the tool names are known.
        AGENTS.md content is injected as project rules, not as a replacement.
        """
        prompt = build_zed_system_prompt(
            available_tools=available_tools,
            worktrees=[self._worktree_name],
            os_name=platform.system(),
            shell=os.environ.get("SHELL"),
            model_name=model_name,
            project_rules=self._get_project_rules(),
        )

        skill_text = format_skills_for_prompt(self.get_skills().skills)
        if skill_text:
            prompt = prompt + "\n\n" + skill_text

        return prompt
