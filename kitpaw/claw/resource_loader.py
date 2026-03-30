from __future__ import annotations

import os
from pathlib import Path

from ..pi_agent.code_agent.resource_loader import DefaultResourceLoader
from ..pi_agent.code_agent.types import (
    LoadedAgentsFiles,
    LoadedExtensions,
    LoadedPrompts,
    LoadedSkills,
    LoadedThemes,
    Skill,
)
from .system_prompt import (
    ClawContextFile,
    ClawPromptContext,
    ClawRuntimeInfo,
    build_claw_system_prompt,
    format_claw_skills_for_prompt,
)


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
        thinking_level: str | None = None,
        agent_id: str = "claw",
        prompt_mode: str = "full",
    ) -> str:
        return self._build_prompt(
            available_tools,
            self.get_skills().skills,
            model_name=model_name,
            thinking_level=thinking_level,
            agent_id=agent_id,
            prompt_mode=prompt_mode,
        )

    def _get_context_files(self) -> list[ClawContextFile]:
        files: list[ClawContextFile] = []
        for path_str in self._delegate.get_agents_files().agents_files:
            path = Path(path_str)
            try:
                content = path.read_text(encoding="utf-8").strip()
            except OSError:
                continue
            if not content:
                continue
            files.append(ClawContextFile(path=str(path), content=content))
        return files

    def _build_prompt(
        self,
        available_tools: list[str],
        skills: list[Skill],
        *,
        model_name: str | None = None,
        thinking_level: str | None = None,
        agent_id: str = "claw",
        prompt_mode: str = "full",
    ) -> str:
        context = ClawPromptContext(
            available_tools=available_tools,
            workspace_dir=self._cwd,
            prompt_mode=prompt_mode,
            runtime_info=ClawRuntimeInfo(
                agent_id=agent_id,
                repo_root=self._cwd,
                model_name=model_name,
                shell=os.environ.get("SHELL"),
                thinking_level=thinking_level,
            ),
            skills_prompt=format_claw_skills_for_prompt(skills),
            context_files=self._get_context_files(),
        )
        return build_claw_system_prompt(context)
