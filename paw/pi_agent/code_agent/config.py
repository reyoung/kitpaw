from __future__ import annotations

import os
from pathlib import Path

APP_NAME = "pi"
CONFIG_DIR_NAME = ".pi"
ENV_AGENT_DIR = "PI_CODING_AGENT_DIR"
VERSION = "0.1.0"


def get_agent_dir() -> Path:
    override = os.getenv(ENV_AGENT_DIR)
    if override:
        return Path(override).expanduser().resolve()
    return (Path.home() / CONFIG_DIR_NAME / "agent").resolve()


def get_settings_path() -> Path:
    return get_agent_dir() / "settings.json"


def get_auth_path() -> Path:
    return get_agent_dir() / "auth.json"


def get_models_path() -> Path:
    return get_agent_dir() / "models.json"


def get_prompts_dir() -> Path:
    return get_agent_dir() / "prompts"


def get_sessions_dir() -> Path:
    return get_agent_dir() / "sessions"


def get_extensions_dir() -> Path:
    return get_agent_dir() / "extensions"


def get_skills_dir() -> Path:
    return get_agent_dir() / "skills"


def get_themes_dir() -> Path:
    return get_agent_dir() / "themes"


def get_project_pi_dir(cwd: str | Path) -> Path:
    return Path(cwd).resolve() / CONFIG_DIR_NAME


def get_project_settings_path(cwd: str | Path) -> Path:
    return get_project_pi_dir(cwd) / "settings.json"


def get_project_extensions_dir(cwd: str | Path) -> Path:
    return get_project_pi_dir(cwd) / "extensions"


def get_project_skills_dir(cwd: str | Path) -> Path:
    return get_project_pi_dir(cwd) / "skills"


def get_project_prompts_dir(cwd: str | Path) -> Path:
    return get_project_pi_dir(cwd) / "prompts"


def get_project_themes_dir(cwd: str | Path) -> Path:
    return get_project_pi_dir(cwd) / "themes"


def get_project_agents_path(cwd: str | Path) -> Path:
    return Path(cwd).resolve() / "AGENTS.md"


def encode_cwd_for_session_dir(cwd: str | Path) -> str:
    path = str(Path(cwd).resolve())
    return f"--{path.replace('/', '-').replace(':', '')}--"
