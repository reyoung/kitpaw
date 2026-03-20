from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

from .config import (
    get_project_agents_path,
    get_project_extensions_dir,
    get_project_prompts_dir,
    get_project_skills_dir,
    get_project_themes_dir,
)
from .types import (
    LoadedAgentsFiles,
    LoadedExtensions,
    LoadedPrompts,
    LoadedSkills,
    LoadedThemes,
    PromptTemplate,
    ResourceDiagnostic,
    Skill,
    ThemeResource,
)


class DefaultResourceLoader:
    def __init__(self, cwd: str, agent_dir: str, settings_manager) -> None:
        self.cwd = Path(cwd).resolve()
        self.agent_dir = Path(agent_dir).resolve()
        self.settings_manager = settings_manager
        self._skills = LoadedSkills()
        self._prompts = LoadedPrompts()
        self._themes = LoadedThemes()
        self._extensions = LoadedExtensions()
        self._agents_files = LoadedAgentsFiles()

    async def reload(self) -> None:
        self._skills = self._load_skills()
        self._prompts = self._load_prompts()
        self._themes = self._load_themes()
        self._extensions = self._load_extensions()
        self._agents_files = self._load_agents_files()

    def get_skills(self) -> LoadedSkills:
        return self._skills

    def get_prompts(self) -> LoadedPrompts:
        return self._prompts

    def get_themes(self) -> LoadedThemes:
        return self._themes

    def get_extensions(self) -> LoadedExtensions:
        return self._extensions

    def get_agents_files(self) -> LoadedAgentsFiles:
        return self._agents_files

    def get_system_prompt(self) -> str | None:
        files = self._agents_files.agents_files
        if not files:
            return None
        return "\n\n".join(Path(path).read_text(encoding="utf-8") for path in files)

    def get_append_system_prompt(self) -> list[str]:
        return []

    def get_path_metadata(self) -> dict[str, Any]:
        return {}

    def extend_resources(self, *_args, **_kwargs) -> None:
        return None

    def _load_agents_files(self) -> LoadedAgentsFiles:
        files: list[str] = []
        project_agents = get_project_agents_path(self.cwd)
        if project_agents.exists():
            files.append(str(project_agents))
        return LoadedAgentsFiles(agents_files=files)

    def _iter_skill_files(self) -> list[Path]:
        paths = [self.agent_dir / "skills", get_project_skills_dir(self.cwd)]
        skill_files: list[Path] = []
        for root in paths:
            if not root.exists():
                continue
            skill_files.extend(root.glob("**/SKILL.md"))
            skill_files.extend([p for p in root.glob("*.md") if p.name != "SKILL.md"])
        return skill_files

    def _load_skills(self) -> LoadedSkills:
        skills: list[Skill] = []
        diagnostics: list[ResourceDiagnostic] = []
        seen: set[str] = set()
        for path in self._iter_skill_files():
            try:
                text = path.read_text(encoding="utf-8")
            except OSError as exc:
                diagnostics.append(ResourceDiagnostic(type="error", message=f"Cannot read skill file {path}: {exc}"))
                continue
            name = path.parent.name if path.name == "SKILL.md" else path.stem
            description = ""
            if text.startswith("---"):
                parts = text.split("---", 2)
                if len(parts) >= 3:
                    frontmatter, body = parts[1], parts[2]
                    for line in frontmatter.splitlines():
                        if line.startswith("name:"):
                            name = line.split(":", 1)[1].strip()
                        if line.startswith("description:"):
                            description = line.split(":", 1)[1].strip()
                    text = body
            if not description:
                diagnostics.append(ResourceDiagnostic(type="warning", message=f"Skill {path} missing description"))
                continue
            if name in seen:
                diagnostics.append(ResourceDiagnostic(type="collision", message=f"Duplicate skill name: {name}"))
                continue
            seen.add(name)
            skills.append(
                Skill(
                    name=name,
                    description=description,
                    file_path=str(path),
                    base_dir=str(path.parent),
                    source="project" if str(self.cwd) in str(path) else "global",
                )
            )
        return LoadedSkills(skills=skills, diagnostics=diagnostics)

    def _load_prompts(self) -> LoadedPrompts:
        prompts: list[PromptTemplate] = []
        for root in (get_project_prompts_dir(self.cwd), self.agent_dir / "prompts"):
            if not root.exists():
                continue
            for path in root.glob("**/*.md"):
                prompts.append(PromptTemplate(name=path.stem, text=path.read_text(encoding="utf-8"), file_path=str(path)))
        return LoadedPrompts(prompts=prompts, diagnostics=[])

    def _load_themes(self) -> LoadedThemes:
        themes: list[ThemeResource] = []
        diagnostics: list[ResourceDiagnostic] = []
        for root in (get_project_themes_dir(self.cwd), self.agent_dir / "themes"):
            if not root.exists():
                continue
            for path in root.glob("*.json"):
                try:
                    themes.append(ThemeResource(name=path.stem, file_path=str(path), data=json.loads(path.read_text(encoding="utf-8"))))
                except json.JSONDecodeError as exc:
                    diagnostics.append(ResourceDiagnostic(type="error", message=f"Theme parse error {path}: {exc}"))
        return LoadedThemes(themes=themes, diagnostics=diagnostics)

    def _load_extensions(self) -> LoadedExtensions:
        modules: list[Any] = []
        errors: list[str] = []
        for root in (get_project_extensions_dir(self.cwd), self.agent_dir / "extensions"):
            if not root.exists():
                continue
            for path in list(root.glob("*.py")) + list(root.glob("*/__init__.py")):
                module_name = f"pi_ext_{path.stem}_{abs(hash(path))}"
                spec = importlib.util.spec_from_file_location(module_name, path)
                if spec is None or spec.loader is None:
                    errors.append(f"Could not load extension {path}")
                    continue
                module = importlib.util.module_from_spec(spec)
                try:
                    spec.loader.exec_module(module)
                    modules.append(module)
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"{path}: {exc}")
        return LoadedExtensions(extensions=modules, errors=errors)
