from __future__ import annotations

from pathlib import Path

from ..pi_agent.code_agent.resource_loader import DefaultResourceLoader
from ..pi_agent.code_agent.system_prompt import format_skills_for_prompt
from ..pi_agent.code_agent.types import (
    LoadedAgentsFiles,
    LoadedExtensions,
    LoadedPrompts,
    LoadedSkills,
    LoadedThemes,
    Skill,
)
from .system_prompt import build_claw_system_prompt


class ClawResourceLoader:
    def __init__(self, cwd: str, agent_dir: str, settings_manager) -> None:
        self._delegate = DefaultResourceLoader(cwd, agent_dir, settings_manager)
        self._cwd = cwd
        self._tool_names: list[str] = []

    @property
    def agent_dir(self) -> Path:
        return self._delegate.agent_dir

    def set_tool_names(self, names: list[str]) -> None:
        self._tool_names = list(names)

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

    def format_tool_not_found(self, tool_name: str) -> str:
        return self._delegate.format_tool_not_found(tool_name)

    def get_system_prompt(self) -> str | None:
        return None

    def build_system_prompt(self, base_prompt: str | None, skills: list[Skill]) -> str:
        del base_prompt
        return self._build_prompt(self._tool_names, skills)

    def build_system_prompt_with_tools(
        self,
        available_tools: list[str],
        *,
        model_name: str | None = None,
    ) -> str:
        del model_name
        return self._build_prompt(available_tools, self.get_skills().skills)

    def _build_prompt(self, available_tools: list[str], skills: list[Skill]) -> str:
        prompt = build_claw_system_prompt(available_tools=available_tools, cwd=self._cwd)
        skill_text = format_skills_for_prompt(skills)
        if skill_text:
            prompt = prompt + "\n\n" + skill_text
        return prompt
