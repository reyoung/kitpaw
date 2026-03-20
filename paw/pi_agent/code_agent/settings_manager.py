from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import get_project_settings_path, get_settings_path
from .types import (
    CompactionSettings,
    ImageSettings,
    RetrySettings,
    Settings,
    TerminalSettings,
)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


class SettingsManager:
    def __init__(self, cwd: str, agent_dir: Path) -> None:
        self.cwd = cwd
        self.agent_dir = agent_dir
        self.global_path = get_settings_path() if agent_dir == get_settings_path().parent else agent_dir / "settings.json"
        self.project_path = get_project_settings_path(cwd)
        self._settings = self._load()

    @classmethod
    def create(cls, cwd: str, agent_dir: str | Path) -> "SettingsManager":
        return cls(cwd, Path(agent_dir))

    def _load_json(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            return {}
        return json.loads(text)

    def _from_dict(self, data: dict[str, Any]) -> Settings:
        retry_data = data.get("retry", {})
        compaction_data = data.get("compaction", {})
        images = ImageSettings(
            auto_resize=data.get("images", {}).get("autoResize", True),
            block_images=data.get("images", {}).get("blockImages", False),
        )
        terminal = TerminalSettings(
            show_images=data.get("terminal", {}).get("showImages", True),
            clear_on_shrink=data.get("terminal", {}).get("clearOnShrink", False),
        )
        retry = RetrySettings(
            enabled=retry_data.get("enabled", True),
            max_retries=retry_data.get("maxRetries", retry_data.get("max_retries", 3)),
            base_delay_ms=retry_data.get("baseDelayMs", retry_data.get("base_delay_ms", 2000)),
            max_delay_ms=retry_data.get("maxDelayMs", retry_data.get("max_delay_ms", 60_000)),
        )
        compaction = CompactionSettings(
            enabled=compaction_data.get("enabled", True),
            reserve_tokens=compaction_data.get("reserveTokens", compaction_data.get("reserve_tokens", 16_384)),
            keep_recent_tokens=compaction_data.get("keepRecentTokens", compaction_data.get("keep_recent_tokens", 20_000)),
        )
        return Settings(
            default_provider=data.get("defaultProvider"),
            default_model=data.get("defaultModel"),
            default_thinking_level=data.get("defaultThinkingLevel"),
            theme=data.get("theme", "dark"),
            quiet_startup=data.get("quietStartup", False),
            steering_mode=data.get("steeringMode", "one-at-a-time"),
            follow_up_mode=data.get("followUpMode", "one-at-a-time"),
            transport=data.get("transport", "sse"),
            shell_command_prefix=data.get("shellCommandPrefix"),
            shell_path=data.get("shellPath"),
            enabled_models=list(data.get("enabledModels", [])),
            packages=list(data.get("packages", [])),
            extensions=list(data.get("extensions", [])),
            skills=list(data.get("skills", [])),
            prompts=list(data.get("prompts", [])),
            themes=list(data.get("themes", [])),
            enable_skill_commands=data.get("enableSkillCommands", True),
            retry=retry,
            compaction=compaction,
            images=images,
            terminal=terminal,
        )

    def _load(self) -> Settings:
        merged = _deep_merge(self._load_json(self.global_path), self._load_json(self.project_path))
        return self._from_dict(merged)

    def reload(self) -> None:
        self._settings = self._load()

    def _to_dict(self) -> dict[str, Any]:
        return {
            "defaultProvider": self._settings.default_provider,
            "defaultModel": self._settings.default_model,
            "defaultThinkingLevel": self._settings.default_thinking_level,
            "theme": self._settings.theme,
            "quietStartup": self._settings.quiet_startup,
            "steeringMode": self._settings.steering_mode,
            "followUpMode": self._settings.follow_up_mode,
            "transport": self._settings.transport,
            "shellCommandPrefix": self._settings.shell_command_prefix,
            "shellPath": self._settings.shell_path,
            "enabledModels": self._settings.enabled_models,
            "packages": self._settings.packages,
            "extensions": self._settings.extensions,
            "skills": self._settings.skills,
            "prompts": self._settings.prompts,
            "themes": self._settings.themes,
            "enableSkillCommands": self._settings.enable_skill_commands,
            "retry": {
                "enabled": self._settings.retry.enabled,
                "maxRetries": self._settings.retry.max_retries,
                "baseDelayMs": self._settings.retry.base_delay_ms,
                "maxDelayMs": self._settings.retry.max_delay_ms,
            },
            "compaction": {
                "enabled": self._settings.compaction.enabled,
                "reserveTokens": self._settings.compaction.reserve_tokens,
                "keepRecentTokens": self._settings.compaction.keep_recent_tokens,
            },
            "images": {
                "autoResize": self._settings.images.auto_resize,
                "blockImages": self._settings.images.block_images,
            },
            "terminal": {
                "showImages": self._settings.terminal.show_images,
                "clearOnShrink": self._settings.terminal.clear_on_shrink,
            },
        }

    def save_project_settings(self) -> None:
        self.project_path.parent.mkdir(parents=True, exist_ok=True)
        self.project_path.write_text(json.dumps(self._to_dict(), indent=2), encoding="utf-8")

    def get_settings(self) -> Settings:
        return self._settings

    def get_default_provider(self) -> str | None:
        return self._settings.default_provider

    def get_default_model(self) -> str | None:
        return self._settings.default_model

    def get_default_thinking_level(self):
        return self._settings.default_thinking_level

    def get_theme(self) -> str:
        return self._settings.theme

    def set_theme(self, theme: str) -> None:
        self._settings.theme = theme
        self.save_project_settings()

    def get_quiet_startup(self) -> bool:
        return self._settings.quiet_startup

    def set_quiet_startup(self, enabled: bool) -> None:
        self._settings.quiet_startup = enabled
        self.save_project_settings()

    def get_steering_mode(self):
        return self._settings.steering_mode

    def set_steering_mode(self, mode: str) -> None:
        self._settings.steering_mode = mode
        self.save_project_settings()

    def get_follow_up_mode(self):
        return self._settings.follow_up_mode

    def set_follow_up_mode(self, mode: str) -> None:
        self._settings.follow_up_mode = mode
        self.save_project_settings()

    def get_transport(self):
        return self._settings.transport

    def set_transport(self, transport: str) -> None:
        self._settings.transport = transport
        self.save_project_settings()

    def get_shell_command_prefix(self) -> str | None:
        return self._settings.shell_command_prefix

    def get_block_images(self) -> bool:
        return self._settings.images.block_images

    def set_block_images(self, enabled: bool) -> None:
        self._settings.images.block_images = enabled
        self.save_project_settings()

    def set_show_images(self, enabled: bool) -> None:
        self._settings.terminal.show_images = enabled
        self.save_project_settings()

    def set_enable_skill_commands(self, enabled: bool) -> None:
        self._settings.enable_skill_commands = enabled
        self.save_project_settings()

    def set_retry_enabled(self, enabled: bool) -> None:
        self._settings.retry.enabled = enabled
        self.save_project_settings()

    def set_retry_max_retries(self, max_retries: int) -> None:
        self._settings.retry.max_retries = max_retries
        self.save_project_settings()

    def set_retry_base_delay_ms(self, base_delay_ms: int) -> None:
        self._settings.retry.base_delay_ms = base_delay_ms
        self.save_project_settings()

    def set_retry_max_delay_ms(self, max_delay_ms: int) -> None:
        self._settings.retry.max_delay_ms = max_delay_ms
        self.save_project_settings()

    def set_compaction_enabled(self, enabled: bool) -> None:
        self._settings.compaction.enabled = enabled
        self.save_project_settings()

    def set_compaction_reserve_tokens(self, reserve_tokens: int) -> None:
        self._settings.compaction.reserve_tokens = reserve_tokens
        self.save_project_settings()

    def set_compaction_keep_recent_tokens(self, keep_recent_tokens: int) -> None:
        self._settings.compaction.keep_recent_tokens = keep_recent_tokens
        self.save_project_settings()
