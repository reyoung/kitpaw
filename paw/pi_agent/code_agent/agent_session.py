from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..agent import Agent
from ..agent.types import AgentMessage, ThinkingLevel
from ..ai import UserMessage
from ..ai.env_api_keys import get_env_default_model
from ..ai.models import get_model, supports_xhigh
from ..ai.types import ImageContent, TextContent
from .export_html import export_from_file
from .message_restore import restore_message
from .model_registry import ModelRegistry
from .package_manager import PackageManager
from .session_manager import SessionManager
from .settings_manager import SettingsManager
from .summarizer import estimate_tokens, generate_summary
from .system_prompt import build_system_prompt
from .types import PromptOptions, QueueMode, SessionInfo, SessionStateSnapshot

_THINKING_LEVELS: tuple[ThinkingLevel, ...] = ("off", "minimal", "low", "medium", "high", "xhigh")


def _estimate_message_entry_tokens(message: dict[str, Any]) -> int:
    content = message.get("content", "")
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        text = "\n".join(block.get("text", "") for block in content if block.get("type") == "text")
    else:
        text = ""
    return max(len(text) // 4, 1) if text else 0


@dataclass(slots=True)
class AgentSessionConfig:
    agent: Agent
    session_manager: SessionManager
    settings_manager: SettingsManager
    cwd: str
    model_registry: ModelRegistry
    resource_loader: Any


class AgentSession:
    def __init__(self, config: AgentSessionConfig) -> None:
        self.agent = config.agent
        self.session_manager = config.session_manager
        self.settings_manager = config.settings_manager
        self.cwd = config.cwd
        self.model_registry = config.model_registry
        self.resource_loader = config.resource_loader
        self._listeners: list[Callable[[Any], None]] = []
        self.agent.subscribe(self._handle_agent_event)
        skills = self.resource_loader.get_skills().skills
        self.agent.set_system_prompt(build_system_prompt(self.agent.state.system_prompt, skills))

    def subscribe(self, listener: Callable[[Any], None]) -> Callable[[], None]:
        self._listeners.append(listener)
        return lambda: self._listeners.remove(listener)

    def _emit(self, event: Any) -> None:
        for listener in list(self._listeners):
            listener(event)

    def _handle_agent_event(self, event: Any) -> None:
        if getattr(event, "type", None) == "message_end":
            self.session_manager.append_message(event.message)
        self._emit(event)

    def _restore_messages_from_current_branch(self) -> None:
        context = self.session_manager.build_runtime_context()
        self.agent.replace_messages([restore_message(message) for message in context["messages"] if isinstance(message, dict)])
        model = context.get("model")
        if model is not None:
            self.agent.set_model(self.model_registry.find(model["provider"], model["modelId"]))
        thinking_level = context.get("thinkingLevel")
        if thinking_level is not None:
            self.agent.set_thinking_level(thinking_level)

    @property
    def messages(self) -> list[AgentMessage]:
        return list(self.agent.state.messages)

    @property
    def session_file(self) -> str | None:
        return self.session_manager.get_session_file()

    @property
    def session_id(self) -> str:
        return self.session_manager.get_session_id()

    @property
    def model(self):
        return self.agent.state.model

    @property
    def thinking_level(self) -> ThinkingLevel:
        return self.agent.state.thinking_level

    @property
    def is_streaming(self) -> bool:
        return self.agent.state.is_streaming

    @property
    def system_prompt(self) -> str:
        return self.agent.state.system_prompt

    async def prompt(self, text: str, options: PromptOptions | None = None) -> None:
        opts = options or PromptOptions()
        images = opts.images
        if self.agent.state.is_streaming:
            if opts.streaming_behavior == "steer":
                await self.steer(text, images=images)
                return
            if opts.streaming_behavior == "followUp":
                await self.follow_up(text, images=images)
                return
            raise ValueError("Agent is already streaming. Use streaming_behavior='steer' or 'followUp'.")
        await self.agent.prompt(text, images=images)
        await self._maybe_auto_compact()

    async def steer(self, text: str, images: list[ImageContent] | None = None) -> None:
        content: str | list[Any] = text
        if images:
            content = [TextContent(text=text), *images]
        self.agent.steer(UserMessage(content=content))
        if not self.agent.state.is_streaming:
            await self.agent.continue_()
            await self._maybe_auto_compact()

    async def follow_up(self, text: str, images: list[ImageContent] | None = None) -> None:
        content: str | list[Any] = text
        if images:
            content = [TextContent(text=text), *images]
        self.agent.follow_up(UserMessage(content=content))
        if not self.agent.state.is_streaming:
            await self.agent.continue_()
            await self._maybe_auto_compact()

    async def abort(self) -> None:
        self.agent.abort()
        await self.agent.wait_for_idle()

    async def bash(self, command: str) -> dict[str, Any]:
        for tool in self.agent.state.tools:
            if tool.name == "bash":
                result = await tool.execute("bash", {"command": command}, asyncio.Event(), None)
                message = {
                    "role": "bashExecution",
                    "command": command,
                    "output": "".join(block.text for block in result.content),
                    "exitCode": 0,
                    "cancelled": False,
                    "timestamp": 0,
                }
                self.session_manager.append_message(message)
                return message
        raise ValueError("bash tool is not enabled")

    async def new_session(self) -> None:
        self.agent.reset()
        if self.session_manager.session_file is not None:
            self.session_manager.entries = []
            self.session_manager.leaf_id = None

    def _apply_model(self, provider: str, model_id: str) -> None:
        model = self.model_registry.find(provider, model_id)
        self.agent.set_model(model)
        self.session_manager.append_model_change(provider, model_id)

    async def set_model(self, provider: str, model_id: str) -> None:
        self._apply_model(provider, model_id)

    async def cycle_model(self) -> Any:
        models = self.model_registry.list_models()
        if not models:
            raise ValueError("No models available")
        current_index = next(
            (
                index
                for index, model in enumerate(models)
                if model.provider == self.model.provider and model.id == self.model.id
            ),
            -1,
        )
        next_model = models[(current_index + 1) % len(models)]
        await self.set_model(next_model.provider, next_model.id)
        return self.model

    def set_thinking_level(self, level: ThinkingLevel) -> None:
        self.agent.set_thinking_level(level)
        self.session_manager.append_thinking_level_change(level)

    def cycle_thinking_level(self) -> ThinkingLevel:
        levels = list(_THINKING_LEVELS)
        if not supports_xhigh(self.model):
            levels.remove("xhigh")
        current_level = self.thinking_level
        current_index = levels.index(current_level) if current_level in levels else -1
        next_level = levels[(current_index + 1) % len(levels)]
        self.set_thinking_level(next_level)
        return next_level

    def get_thinking_selector_schema(self) -> dict[str, Any]:
        items = [
            {
                "id": level,
                "value": level,
                "label": level,
                "description": "supported" if level != "xhigh" or supports_xhigh(self.model) else "unsupported",
                "isCurrent": level == self.thinking_level,
                "isSupported": level != "xhigh" or supports_xhigh(self.model),
                "position": index,
            }
            for index, level in enumerate(_THINKING_LEVELS)
            if level != "xhigh" or supports_xhigh(self.model)
        ]
        return {
            "currentThinkingLevel": self.thinking_level,
            "itemOrder": [item["id"] for item in items],
            "items": items,
        }

    def set_steering_mode(self, mode: QueueMode) -> QueueMode:
        self.agent.set_steering_mode(mode)
        self.settings_manager.set_steering_mode(mode)
        return self.agent.get_steering_mode()

    def get_steering_selector_schema(self) -> dict[str, Any]:
        current_mode = self.agent.get_steering_mode()
        items = [
            {
                "id": mode,
                "value": mode,
                "label": mode,
                "description": "Queue every steering message" if mode == "all" else "Queue one steering message at a time",
                "isCurrent": mode == current_mode,
                "position": index,
            }
            for index, mode in enumerate(("one-at-a-time", "all"))
        ]
        return {
            "currentSteeringMode": current_mode,
            "itemOrder": [item["id"] for item in items],
            "items": items,
        }

    def set_follow_up_mode(self, mode: QueueMode) -> QueueMode:
        self.agent.set_follow_up_mode(mode)
        self.settings_manager.set_follow_up_mode(mode)
        return self.agent.get_follow_up_mode()

    def get_follow_up_selector_schema(self) -> dict[str, Any]:
        current_mode = self.agent.get_follow_up_mode()
        items = [
            {
                "id": mode,
                "value": mode,
                "label": mode,
                "description": "Queue every follow-up message" if mode == "all" else "Queue one follow-up message at a time",
                "isCurrent": mode == current_mode,
                "position": index,
            }
            for index, mode in enumerate(("one-at-a-time", "all"))
        ]
        return {
            "currentFollowUpMode": current_mode,
            "itemOrder": [item["id"] for item in items],
            "items": items,
        }

    def set_session_name(self, name: str | None) -> None:
        self.session_manager.set_session_name(name)

    def get_session_stats(self) -> dict[str, Any]:
        return self.session_manager.get_stats()

    def get_settings_snapshot(self) -> dict[str, Any]:
        settings = self.settings_manager.get_settings()
        return {
            "theme": self.get_theme(),
            "model": {"provider": self.model.provider, "id": self.model.id},
            "thinkingLevel": self.thinking_level,
            "steeringMode": self.agent.get_steering_mode(),
            "followUpMode": self.agent.get_follow_up_mode(),
            "quietStartup": settings.quiet_startup,
            "blockImages": settings.images.block_images,
            "showImages": settings.terminal.show_images,
            "enableSkillCommands": settings.enable_skill_commands,
            "transport": settings.transport,
            "retry": {
                "enabled": settings.retry.enabled,
                "maxRetries": settings.retry.max_retries,
                "baseDelayMs": settings.retry.base_delay_ms,
                "maxDelayMs": settings.retry.max_delay_ms,
            },
            "compaction": self.get_compaction_state(),
            "sessionName": self.session_manager.get_session_name(),
            "sessionFile": self.session_file,
        }

    def get_package_selector_schema(self) -> dict[str, Any]:
        manager = PackageManager(self.cwd, str(self.resource_loader.agent_dir), self.settings_manager)
        packages = manager.list()
        items = [
            {
                "id": package.source,
                "source": package.source,
                "scope": package.scope,
                "path": package.path,
                "label": package.source,
                "description": f"{package.scope}: {package.path}",
                "position": index,
            }
            for index, package in enumerate(packages)
        ]
        return {
            "itemOrder": [item["id"] for item in items],
            "items": items,
        }

    def list_packages(self) -> list[dict[str, Any]]:
        manager = PackageManager(self.cwd, str(self.resource_loader.agent_dir), self.settings_manager)
        return [
            {
                "source": package.source,
                "scope": package.scope,
                "path": package.path,
            }
            for package in manager.list()
        ]

    def install_package(self, source: str, local: bool = False) -> dict[str, Any]:
        manager = PackageManager(self.cwd, str(self.resource_loader.agent_dir), self.settings_manager)
        path = manager.install(source, local=local)
        return {"source": source, "scope": "project" if local else "user", "path": path}

    def remove_package(self, source: str, local: bool = False) -> dict[str, Any]:
        manager = PackageManager(self.cwd, str(self.resource_loader.agent_dir), self.settings_manager)
        removed = manager.remove(source, local=local)
        return {"source": source, "scope": "project" if local else "user", "removed": removed}

    def update_packages(self, source: str | None = None) -> dict[str, Any]:
        manager = PackageManager(self.cwd, str(self.resource_loader.agent_dir), self.settings_manager)
        updated = manager.update(source)
        return {"updated": updated}

    def get_package_selector_item(self, source: str) -> dict[str, Any]:
        payload = self.get_selector_item("packages", source)
        payload["kind"] = "packages"
        return payload

    def get_resource_schema(self) -> dict[str, Any]:
        skills = self.resource_loader.get_skills()
        prompts = self.resource_loader.get_prompts()
        themes = self.resource_loader.get_themes()
        extensions = self.resource_loader.get_extensions()
        agents_files = self.resource_loader.get_agents_files()
        return {
            "counts": {
                "skills": len(skills.skills),
                "prompts": len(prompts.prompts),
                "themes": len(themes.themes),
                "extensions": len(extensions.extensions),
                "agentsFiles": len(agents_files.agents_files),
            },
            "skills": [
                {
                    "id": skill.name,
                    "name": skill.name,
                    "description": skill.description,
                    "source": skill.source,
                    "filePath": skill.file_path,
                    "position": index,
                }
                for index, skill in enumerate(skills.skills)
            ],
            "prompts": [
                {
                    "id": prompt.name,
                    "name": prompt.name,
                    "filePath": prompt.file_path,
                    "position": index,
                }
                for index, prompt in enumerate(prompts.prompts)
            ],
            "themes": [
                {
                    "id": theme.name,
                    "name": theme.name,
                    "filePath": theme.file_path,
                    "position": index,
                }
                for index, theme in enumerate(themes.themes)
            ],
            "extensions": [
                {
                    "id": getattr(module, "__name__", f"extension-{index}"),
                    "name": getattr(module, "__name__", f"extension-{index}"),
                    "position": index,
                }
                for index, module in enumerate(extensions.extensions)
            ],
            "agentsFiles": list(agents_files.agents_files),
            "diagnostics": {
                "skills": [{"type": item.type, "message": item.message} for item in skills.diagnostics],
                "prompts": [{"type": item.type, "message": item.message} for item in prompts.diagnostics],
                "themes": [{"type": item.type, "message": item.message} for item in themes.diagnostics],
                "extensions": list(extensions.errors),
            },
        }

    def get_resource_item(self, kind: str, item_id: str) -> dict[str, Any]:
        schema = self.get_resource_schema()
        normalized = kind.strip().lower().replace("-", "")
        if normalized == "agentsfiles":
            normalized = "agentsFiles"
        if normalized not in {"skills", "prompts", "themes", "extensions", "agentsFiles"}:
            raise ValueError(f"Unknown resource kind: {kind}")
        values = schema.get(normalized, [])
        if normalized == "agentsFiles":
            for item in values:
                if str(item) == item_id:
                    return {"kind": normalized, "requestedItemId": item_id, "resolvedItemId": str(item), "item": item}
        else:
            for item in values:
                if str(item.get("id")) == item_id or str(item.get("name")) == item_id:
                    resolved_item_id = str(item.get("id") or item.get("name") or item_id)
                    return {"kind": normalized, "requestedItemId": item_id, "resolvedItemId": resolved_item_id, "item": item}
        raise ValueError(f"Unknown resource item for {kind}: {item_id}")

    def get_command_schema(self) -> dict[str, Any]:
        commands = [
            {"name": "help", "group": "general", "order": 10, "description": "Show interactive help", "usage": "/help"},
            {"name": "session", "group": "session", "order": 20, "description": "Show current session stats", "usage": "/session"},
            {"name": "selector-item", "group": "resources", "order": 25, "description": "Show a single selector item", "usage": "/selector-item SELECTOR_ID ITEM_ID"},
            {"name": "settings", "group": "settings", "order": 30, "description": "Show or update settings", "usage": "/settings"},
            {"name": "settings-schema", "group": "settings", "order": 40, "description": "Show editable settings schema", "usage": "/settings schema"},
            {"name": "compaction", "group": "session", "order": 50, "description": "Show or update compaction settings", "usage": "/compaction"},
            {"name": "theme", "group": "appearance", "order": 60, "description": "Show or set theme", "usage": "/theme [NAME]"},
            {"name": "theme-schema", "group": "appearance", "order": 70, "description": "Show flat theme selector schema", "usage": "/theme schema"},
            {"name": "model", "group": "model", "order": 80, "description": "Show or set model", "usage": "/model [MODEL_ID]"},
            {"name": "model-schema", "group": "model", "order": 90, "description": "Show flat model selector schema", "usage": "/model schema"},
            {"name": "cycle-model", "group": "model", "order": 100, "description": "Cycle to the next model", "usage": "/cycle-model"},
            {"name": "thinking", "group": "model", "order": 110, "description": "Show, set, or cycle thinking level", "usage": "/thinking [LEVEL|cycle]"},
            {"name": "steering", "group": "interaction", "order": 120, "description": "Show or set steering queue mode", "usage": "/steering [MODE]"},
            {"name": "followup", "group": "interaction", "order": 130, "description": "Show or set follow-up queue mode", "usage": "/followup [MODE]"},
            {"name": "name", "group": "session", "order": 140, "description": "Set session name", "usage": "/name [TEXT]"},
            {"name": "new", "group": "session", "order": 150, "description": "Start a new session", "usage": "/new"},
            {"name": "reload", "group": "resources", "order": 160, "description": "Reload resources", "usage": "/reload"},
            {"name": "resources", "group": "resources", "order": 170, "description": "Show loaded resource counts", "usage": "/resources"},
            {"name": "resources-schema", "group": "resources", "order": 180, "description": "Show loaded resource details", "usage": "/resources schema"},
            {"name": "packages-schema", "group": "resources", "order": 190, "description": "Show installed package selector schema", "usage": "/packages schema"},
            {"name": "packages", "group": "resources", "order": 195, "description": "List installed packages", "usage": "/packages"},
            {"name": "packages-install", "group": "resources", "order": 196, "description": "Install a package from a local path or git URL", "usage": "/packages install SOURCE [local]"},
            {"name": "packages-remove", "group": "resources", "order": 197, "description": "Remove an installed package", "usage": "/packages remove SOURCE [local]"},
            {"name": "packages-update", "group": "resources", "order": 198, "description": "Update installed packages", "usage": "/packages update [SOURCE]"},
            {"name": "packages-uninstall", "group": "resources", "order": 199, "description": "Alias for packages remove", "usage": "/packages uninstall SOURCE [local]"},
            {"name": "packages-item", "group": "resources", "order": 200, "description": "Show details for an installed package", "usage": "/packages item SOURCE"},
            {"name": "resources-item", "group": "resources", "order": 201, "description": "Show details for a loaded resource", "usage": "/resources item KIND ID"},
            {"name": "resume", "group": "session", "order": 210, "description": "Resume most recent or matching session", "usage": "/resume [QUERY]"},
            {"name": "sessions", "group": "session", "order": 220, "description": "List saved sessions", "usage": "/sessions"},
            {"name": "sessions-schema", "group": "session", "order": 230, "description": "Show flat saved-session selector schema", "usage": "/sessions schema"},
            {"name": "switch", "group": "session", "order": 240, "description": "Switch to a session path, id, or name", "usage": "/switch QUERY"},
            {"name": "fork", "group": "tree", "order": 250, "description": "List or fork from user message", "usage": "/fork [ENTRY_ID]"},
            {"name": "tree", "group": "tree", "order": 260, "description": "Show or switch current session tree", "usage": "/tree [ENTRY_ID]"},
            {"name": "tree-schema", "group": "tree", "order": 270, "description": "Show flat session tree schema", "usage": "/tree schema"},
            {"name": "branch-summary", "group": "tree", "order": 280, "description": "Branch and append a summary entry", "usage": "/branch-summary ID TEXT"},
            {"name": "branch-summary-auto", "group": "tree", "order": 290, "description": "Generate and append a branch summary", "usage": "/branch-summary-auto ID [TEXT]"},
            {"name": "compact", "group": "session", "order": 300, "description": "Append a compaction summary entry", "usage": "/compact ID TOKENS TEXT"},
            {"name": "compact-auto", "group": "session", "order": 310, "description": "Generate and append a compaction summary", "usage": "/compact-auto ID [TEXT]"},
            {"name": "summarize", "group": "session", "order": 320, "description": "Generate a summary of the current session", "usage": "/summarize [TEXT]"},
            {"name": "last", "group": "session", "order": 330, "description": "Show last assistant text", "usage": "/last"},
            {"name": "quit", "group": "general", "order": 340, "description": "Exit interactive mode", "usage": "/quit"},
        ]
        groups = [
            {"id": "general", "label": "General", "order": 10},
            {"id": "session", "label": "Session", "order": 20},
            {"id": "settings", "label": "Settings", "order": 30},
            {"id": "appearance", "label": "Appearance", "order": 40},
            {"id": "model", "label": "Model", "order": 50},
            {"id": "interaction", "label": "Interaction", "order": 60},
            {"id": "resources", "label": "Resources", "order": 70},
            {"id": "tree", "label": "Tree", "order": 80},
        ]
        ordered_commands = sorted(commands, key=lambda item: int(item["order"]))
        return {
            "groups": groups,
            "itemOrder": [item["name"] for item in ordered_commands],
            "commands": ordered_commands,
        }

    def get_selector_registry(self) -> dict[str, Any]:
        selectors = [
            {
                "id": "commands",
                "label": "Commands",
                "kind": "list",
                "getter": "get_command_schema",
                "currentKey": None,
                "itemOrder": self.get_command_schema()["itemOrder"],
                "group": "general",
            },
            {
                "id": "settings",
                "label": "Settings",
                "kind": "fields",
                "getter": "get_settings_schema",
                "currentKey": None,
                "itemOrder": self.get_settings_schema()["fieldOrder"],
                "group": "settings",
            },
            {
                "id": "compaction",
                "label": "Compaction",
                "kind": "fields",
                "getter": "get_compaction_schema",
                "currentKey": "enabled",
                "itemOrder": self.get_compaction_schema()["fieldOrder"],
                "group": "settings",
            },
            {
                "id": "theme",
                "label": "Themes",
                "kind": "list",
                "getter": "get_theme_selector_schema",
                "currentKey": "currentTheme",
                "itemOrder": self.get_theme_selector_schema()["itemOrder"],
                "group": "appearance",
            },
            {
                "id": "model",
                "label": "Models",
                "kind": "list",
                "getter": "get_model_selector_schema",
                "currentKey": "currentModel",
                "itemOrder": self.get_model_selector_schema()["itemOrder"],
                "group": "model",
            },
            {
                "id": "thinking",
                "label": "Thinking",
                "kind": "list",
                "getter": "get_thinking_selector_schema",
                "currentKey": "currentThinkingLevel",
                "itemOrder": self.get_thinking_selector_schema()["itemOrder"],
                "group": "model",
            },
            {
                "id": "steering",
                "label": "Steering",
                "kind": "list",
                "getter": "get_steering_selector_schema",
                "currentKey": "currentSteeringMode",
                "itemOrder": self.get_steering_selector_schema()["itemOrder"],
                "group": "interaction",
            },
            {
                "id": "followup",
                "label": "Follow-up",
                "kind": "list",
                "getter": "get_follow_up_selector_schema",
                "currentKey": "currentFollowUpMode",
                "itemOrder": self.get_follow_up_selector_schema()["itemOrder"],
                "group": "interaction",
            },
            {
                "id": "sessions",
                "label": "Sessions",
                "kind": "list",
                "getter": "get_session_selector_schema",
                "currentKey": "currentSessionFile",
                "itemOrder": self.get_session_selector_schema()["itemOrder"],
                "group": "session",
            },
            {
                "id": "tree",
                "label": "Tree",
                "kind": "list",
                "getter": "get_tree_schema",
                "currentKey": "currentLeafId",
                "itemOrder": self.get_tree_schema()["itemOrder"],
                "group": "session",
            },
            {
                "id": "packages",
                "label": "Packages",
                "kind": "list",
                "getter": "get_package_selector_schema",
                "currentKey": None,
                "itemOrder": self.get_package_selector_schema()["itemOrder"],
                "group": "resources",
            },
            {
                "id": "resources",
                "label": "Resources",
                "kind": "object",
                "getter": "get_resource_schema",
                "currentKey": None,
                "itemOrder": ["skills", "prompts", "themes", "extensions", "agentsFiles"],
                "group": "resources",
            },
        ]
        groups = [
            {"id": "general", "label": "General", "order": 10},
            {"id": "settings", "label": "Settings", "order": 20},
            {"id": "appearance", "label": "Appearance", "order": 30},
            {"id": "model", "label": "Model", "order": 40},
            {"id": "interaction", "label": "Interaction", "order": 50},
            {"id": "session", "label": "Session", "order": 60},
            {"id": "resources", "label": "Resources", "order": 70},
        ]
        ordered = sorted(selectors, key=lambda item: (next(group["order"] for group in groups if group["id"] == item["group"]), item["label"]))
        return {
            "groups": groups,
            "itemOrder": [item["id"] for item in ordered],
            "selectors": ordered,
        }

    def get_selector(self, selector_id: str) -> dict[str, Any]:
        registry = self.get_selector_registry()
        selector = next((item for item in registry["selectors"] if item["id"] == selector_id), None)
        if selector is None:
            raise ValueError(f"Unknown selector: {selector_id}")
        getter = getattr(self, selector["getter"])
        payload = getter()
        return {
            "selector": selector,
            "preview": self._selector_preview(selector, payload),
            "data": payload,
        }

    def get_selector_item(self, selector_id: str, item_id: str) -> dict[str, Any]:
        payload = self.get_selector(selector_id)
        selector = payload["selector"]
        data = payload["data"]
        item = self._find_selector_item(data, item_id)
        if item is None:
            raise ValueError(f"Unknown item for selector {selector_id}: {item_id}")
        resolved_item_id = str(item.get("id") or item.get("name") or item_id) if isinstance(item, dict) else str(item)
        return {
            "selector": selector,
            "requestedItemId": item_id,
            "resolvedItemId": resolved_item_id,
            "item": item,
        }

    def _find_selector_item(self, payload: dict[str, Any], item_id: str) -> Any | None:
        kind_hint: str | None = None
        item_key = item_id
        known_kinds = {"skills", "prompts", "themes", "extensions", "agentsfiles"}
        for separator in (":", "/"):
            if separator in item_id:
                if separator == "/" and item_id.startswith("/"):
                    continue
                possible_kind, possible_item = item_id.split(separator, 1)
                possible_kind = possible_kind.strip().lower().replace("-", "")
                if possible_kind in known_kinds:
                    kind_hint = possible_kind
                    item_key = possible_item.strip()
                    break
        for key in ("items", "commands", "fields", "selectors", "groups", "skills", "prompts", "themes", "extensions", "agentsFiles"):
            values = payload.get(key)
            if isinstance(values, list):
                if kind_hint is not None:
                    normalized_key = key.lower().replace("-", "")
                    if normalized_key != kind_hint and not (key == "agentsFiles" and kind_hint == "agentsfiles"):
                        continue
                for value in values:
                    if key in {"extensions", "agentsFiles"} and str(value) == item_key:
                        return value
                    if isinstance(value, dict) and str(value.get("id")) == item_key:
                        return value
                    if isinstance(value, dict) and str(value.get("name")) == item_key:
                        return value
        return None

    def _selector_preview(self, selector: dict[str, Any], payload: dict[str, Any]) -> str:
        selector_id = selector["id"]
        if selector_id == "commands":
            return f'{len(payload.get("commands", []))} commands'
        if selector_id == "settings":
            return f'{len(payload.get("fields", []))} fields'
        if selector_id == "compaction":
            state = payload.get("state", {})
            return f'enabled={state.get("enabled")} reserve={state.get("reserveTokens")} keep={state.get("keepRecentTokens")}'
        if selector_id == "theme":
            return str(payload.get("currentTheme"))
        if selector_id == "model":
            current = payload.get("currentModel", {})
            return f'{current.get("provider")}/{current.get("id")}'
        if selector_id == "thinking":
            return str(payload.get("currentThinkingLevel"))
        if selector_id == "steering":
            return str(payload.get("currentSteeringMode"))
        if selector_id == "followup":
            return str(payload.get("currentFollowUpMode"))
        if selector_id == "sessions":
            return str(payload.get("currentSessionFile") or "none")
        if selector_id == "tree":
            return str(payload.get("currentLeafId") or "root")
        if selector_id == "packages":
            return f'{len(payload.get("items", []))} packages'
        if selector_id == "resources":
            counts = payload.get("counts", {})
            return (
                f"skills={counts.get('skills', 0)} "
                f"prompts={counts.get('prompts', 0)} "
                f"themes={counts.get('themes', 0)}"
            )
        return selector["label"]

    def get_settings_schema(self) -> dict[str, Any]:
        fields = [
                {
                    "id": "name",
                    "order": 10,
                    "label": "Session name",
                    "description": "Human-readable name for the current session",
                    "group": "session",
                    "type": "string",
                    "updatePath": "sessionName",
                    "command": "/settings name <text>",
                    "current": self.session_manager.get_session_name(),
                },
                {
                    "id": "model",
                    "order": 20,
                    "label": "Model",
                    "description": "Default model for new prompts in this session",
                    "group": "model",
                    "type": "select",
                    "updatePath": "model",
                    "command": "/settings model <model-id>",
                    "current": {"provider": self.model.provider, "id": self.model.id},
                    "options": [
                        {
                            "provider": model.provider,
                            "id": model.id,
                            "value": model.id,
                            "label": f"{model.provider}/{model.id}",
                        }
                        for model in self.get_available_models()
                    ],
                },
                {
                    "id": "theme",
                    "order": 30,
                    "label": "Theme",
                    "description": "Color theme for the interface",
                    "group": "appearance",
                    "type": "select",
                    "updatePath": "theme",
                    "command": "/settings theme <name>",
                    "current": self.get_theme(),
                    "options": [{"value": item["name"], "label": item["name"]} for item in self.get_themes()["themes"]],
                },
                {
                    "id": "thinking",
                    "order": 40,
                    "label": "Thinking level",
                    "description": "Reasoning intensity for the model",
                    "group": "model",
                    "type": "select",
                    "updatePath": "thinkingLevel",
                    "command": "/settings thinking <level>",
                    "current": self.thinking_level,
                    "options": [{"value": level, "label": level} for level in _THINKING_LEVELS if level != "xhigh" or supports_xhigh(self.model)],
                },
                {
                    "id": "steering",
                    "order": 50,
                    "label": "Steering mode",
                    "description": "How steering messages are queued while streaming",
                    "group": "interaction",
                    "type": "select",
                    "updatePath": "steeringMode",
                    "command": "/settings steering <mode>",
                    "current": self.agent.get_steering_mode(),
                    "options": [{"value": mode, "label": mode} for mode in ("one-at-a-time", "all")],
                },
                {
                    "id": "followup",
                    "order": 60,
                    "label": "Follow-up mode",
                    "description": "How follow-up messages are queued while streaming",
                    "group": "interaction",
                    "type": "select",
                    "updatePath": "followUpMode",
                    "command": "/settings followup <mode>",
                    "current": self.agent.get_follow_up_mode(),
                    "options": [{"value": mode, "label": mode} for mode in ("one-at-a-time", "all")],
                },
                {
                    "id": "quiet",
                    "order": 70,
                    "label": "Quiet startup",
                    "description": "Reduce startup output in interactive mode",
                    "group": "interaction",
                    "type": "boolean",
                    "updatePath": "quietStartup",
                    "command": "/settings quiet <true|false>",
                    "current": self.settings_manager.get_settings().quiet_startup,
                },
                {
                    "id": "block-images",
                    "order": 80,
                    "label": "Block images",
                    "description": "Prevent image inputs from being sent to the model",
                    "group": "appearance",
                    "type": "boolean",
                    "updatePath": "blockImages",
                    "command": "/settings block-images <true|false>",
                    "current": self.settings_manager.get_settings().images.block_images,
                },
                {
                    "id": "show-images",
                    "order": 90,
                    "label": "Show images",
                    "description": "Render images in supported terminals",
                    "group": "appearance",
                    "type": "boolean",
                    "updatePath": "showImages",
                    "command": "/settings show-images <true|false>",
                    "current": self.settings_manager.get_settings().terminal.show_images,
                },
                {
                    "id": "skill-commands",
                    "order": 100,
                    "label": "Skill commands",
                    "description": "Expose loaded skills as slash commands",
                    "group": "resources",
                    "type": "boolean",
                    "updatePath": "enableSkillCommands",
                    "command": "/settings skill-commands <true|false>",
                    "current": self.settings_manager.get_settings().enable_skill_commands,
                },
                {
                    "id": "transport",
                    "order": 110,
                    "label": "Transport",
                    "description": "Preferred provider transport",
                    "group": "network",
                    "type": "select",
                    "updatePath": "transport",
                    "command": "/settings transport <value>",
                    "current": self.settings_manager.get_settings().transport,
                    "options": [{"value": value, "label": value} for value in ("sse", "websocket", "auto")],
                },
                {
                    "id": "retry",
                    "order": 120,
                    "label": "Retry policy",
                    "description": "Automatic retry behavior for provider requests",
                    "group": "network",
                    "type": "object",
                    "updatePath": "retry",
                    "current": self.get_settings_snapshot()["retry"],
                    "fields": [
                        {"id": "enabled", "label": "Enabled", "type": "boolean", "updatePath": "retry.enabled", "command": "/settings retry enabled <true|false>"},
                        {"id": "maxRetries", "label": "Max retries", "type": "number", "updatePath": "retry.maxRetries", "command": "/settings retry max-retries <n>"},
                        {"id": "baseDelayMs", "label": "Base delay (ms)", "type": "number", "updatePath": "retry.baseDelayMs", "command": "/settings retry base-delay-ms <ms>"},
                        {"id": "maxDelayMs", "label": "Max delay (ms)", "type": "number", "updatePath": "retry.maxDelayMs", "command": "/settings retry max-delay-ms <ms>"},
                    ],
                },
                {
                    "id": "compaction",
                    "order": 130,
                    "label": "Compaction",
                    "description": "Conversation compaction thresholds and state",
                    "group": "session",
                    "type": "object",
                    "updatePath": "compaction",
                    "current": self.get_compaction_state(),
                    "fields": [
                        {"id": "enabled", "label": "Enabled", "type": "boolean", "updatePath": "compaction.enabled", "command": "/settings compaction enabled <true|false>"},
                        {"id": "reserveTokens", "label": "Reserve tokens", "type": "number", "updatePath": "compaction.reserveTokens", "command": "/settings compaction reserve <n>"},
                        {"id": "keepRecentTokens", "label": "Keep recent tokens", "type": "number", "updatePath": "compaction.keepRecentTokens", "command": "/settings compaction keep <n>"},
                        {"id": "estimatedTokens", "label": "Estimated tokens", "type": "number", "readonly": True},
                        {"id": "thresholdTokens", "label": "Threshold tokens", "type": "number", "readonly": True},
                        {"id": "shouldCompact", "label": "Should compact", "type": "boolean", "readonly": True},
                        {"id": "contextWindow", "label": "Context window", "type": "number", "readonly": True},
                    ],
                },
            ]
        groups = [
            {"id": "session", "label": "Session", "order": 10},
            {"id": "model", "label": "Model", "order": 20},
            {"id": "interaction", "label": "Interaction", "order": 30},
            {"id": "appearance", "label": "Appearance", "order": 40},
            {"id": "resources", "label": "Resources", "order": 50},
            {"id": "network", "label": "Network", "order": 60},
        ]
        return {
            "groups": sorted(groups, key=lambda item: int(item.get("order", 0))),
            "fieldOrder": [field["id"] for field in sorted(fields, key=lambda item: int(item.get("order", 0)))],
            "fields": fields,
        }

    def update_settings(self, patch: dict[str, Any]) -> dict[str, Any]:
        if "model" in patch:
            model_patch = patch["model"]
            if isinstance(model_patch, dict):
                provider = str(model_patch.get("provider") or self.model.provider)
                model_id = str(model_patch.get("id") or self.model.id)
            else:
                provider = self.model.provider
                model_id = str(model_patch)
            self._apply_model(provider, model_id)
        if "theme" in patch:
            self.set_theme(str(patch["theme"]))
        if "sessionName" in patch:
            self.set_session_name(None if patch["sessionName"] is None else str(patch["sessionName"]))
        if "quietStartup" in patch:
            self.settings_manager.set_quiet_startup(bool(patch["quietStartup"]))
        if "blockImages" in patch:
            self.settings_manager.set_block_images(bool(patch["blockImages"]))
        if "showImages" in patch:
            self.settings_manager.set_show_images(bool(patch["showImages"]))
        if "enableSkillCommands" in patch:
            self.settings_manager.set_enable_skill_commands(bool(patch["enableSkillCommands"]))
        if "transport" in patch:
            self.settings_manager.set_transport(str(patch["transport"]))
        if "thinkingLevel" in patch:
            self.set_thinking_level(patch["thinkingLevel"])
        if "steeringMode" in patch:
            self.set_steering_mode(patch["steeringMode"])
        if "followUpMode" in patch:
            self.set_follow_up_mode(patch["followUpMode"])
        retry = patch.get("retry")
        if isinstance(retry, dict):
            if "enabled" in retry:
                self.settings_manager.set_retry_enabled(bool(retry["enabled"]))
            if "maxRetries" in retry:
                self.settings_manager.set_retry_max_retries(int(retry["maxRetries"]))
            if "baseDelayMs" in retry:
                self.settings_manager.set_retry_base_delay_ms(int(retry["baseDelayMs"]))
            if "maxDelayMs" in retry:
                self.settings_manager.set_retry_max_delay_ms(int(retry["maxDelayMs"]))
        compaction = patch.get("compaction")
        if isinstance(compaction, dict):
            if "enabled" in compaction:
                self.set_compaction_enabled(bool(compaction["enabled"]))
            if "reserveTokens" in compaction:
                self.set_compaction_reserve_tokens(int(compaction["reserveTokens"]))
            if "keepRecentTokens" in compaction:
                self.set_compaction_keep_recent_tokens(int(compaction["keepRecentTokens"]))
        return self.get_settings_snapshot()

    def get_theme(self) -> str:
        return self.settings_manager.get_theme()

    def get_themes(self) -> dict[str, Any]:
        loaded = self.resource_loader.get_themes()
        return {
            "currentTheme": self.get_theme(),
            "themes": [{"name": theme.name, "filePath": theme.file_path} for theme in loaded.themes],
            "diagnostics": [{"type": item.type, "message": item.message} for item in loaded.diagnostics],
        }

    def get_theme_selector_schema(self) -> dict[str, Any]:
        themes = self.get_themes()
        current_theme = themes["currentTheme"]
        items = [
            {
                "id": item["name"],
                "value": item["name"],
                "label": item["name"],
                "description": item["filePath"],
                "isCurrent": item["name"] == current_theme,
                "position": index,
            }
            for index, item in enumerate(themes["themes"])
        ]
        return {
            "currentTheme": current_theme,
            "itemOrder": [item["id"] for item in items],
            "items": items,
            "diagnostics": themes["diagnostics"],
        }

    def set_theme(self, theme: str) -> dict[str, Any]:
        available = {item["name"] for item in self.get_themes()["themes"]}
        if available and theme not in available:
            raise ValueError(f"Unknown theme: {theme}")
        self.settings_manager.set_theme(theme)
        return self.get_themes()

    def get_compaction_state(self) -> dict[str, Any]:
        compaction = self.settings_manager.get_settings().compaction
        estimated_tokens = estimate_tokens(self.messages)
        threshold = max(self.model.context_window - compaction.reserve_tokens, 0)
        return {
            "enabled": compaction.enabled,
            "estimatedTokens": estimated_tokens,
            "thresholdTokens": threshold,
            "reserveTokens": compaction.reserve_tokens,
            "keepRecentTokens": compaction.keep_recent_tokens,
            "shouldCompact": compaction.enabled and estimated_tokens > threshold,
            "contextWindow": self.model.context_window,
        }

    def get_compaction_schema(self) -> dict[str, Any]:
        state = self.get_compaction_state()
        fields = [
            {
                "id": "enabled",
                "label": "Enabled",
                "type": "boolean",
                "current": state["enabled"],
                "updatePath": "compaction.enabled",
                "command": "/compaction enabled <true|false>",
                "readonly": False,
                "order": 10,
            },
            {
                "id": "reserveTokens",
                "label": "Reserve tokens",
                "type": "number",
                "current": state["reserveTokens"],
                "updatePath": "compaction.reserveTokens",
                "command": "/compaction reserve <n>",
                "readonly": False,
                "order": 20,
            },
            {
                "id": "keepRecentTokens",
                "label": "Keep recent tokens",
                "type": "number",
                "current": state["keepRecentTokens"],
                "updatePath": "compaction.keepRecentTokens",
                "command": "/compaction keep <n>",
                "readonly": False,
                "order": 30,
            },
            {
                "id": "estimatedTokens",
                "label": "Estimated tokens",
                "type": "number",
                "current": state["estimatedTokens"],
                "readonly": True,
                "order": 40,
            },
            {
                "id": "thresholdTokens",
                "label": "Threshold tokens",
                "type": "number",
                "current": state["thresholdTokens"],
                "readonly": True,
                "order": 50,
            },
            {
                "id": "shouldCompact",
                "label": "Should compact",
                "type": "boolean",
                "current": state["shouldCompact"],
                "readonly": True,
                "order": 60,
            },
            {
                "id": "contextWindow",
                "label": "Context window",
                "type": "number",
                "current": state["contextWindow"],
                "readonly": True,
                "order": 70,
            },
        ]
        return {
            "fieldOrder": [field["id"] for field in sorted(fields, key=lambda item: int(item["order"]))],
            "fields": fields,
            "state": state,
        }

    def set_compaction_enabled(self, enabled: bool) -> dict[str, Any]:
        self.settings_manager.set_compaction_enabled(enabled)
        return self.get_compaction_state()

    def set_compaction_reserve_tokens(self, reserve_tokens: int) -> dict[str, Any]:
        self.settings_manager.set_compaction_reserve_tokens(reserve_tokens)
        return self.get_compaction_state()

    def set_compaction_keep_recent_tokens(self, keep_recent_tokens: int) -> dict[str, Any]:
        self.settings_manager.set_compaction_keep_recent_tokens(keep_recent_tokens)
        return self.get_compaction_state()

    def get_available_models(self) -> list[Any]:
        return self.model_registry.list_models()

    def get_model_selector_schema(self) -> dict[str, Any]:
        models = self.get_available_models()
        current_model = {"provider": self.model.provider, "id": self.model.id}
        items = [
            {
                "id": f"{model.provider}/{model.id}",
                "provider": model.provider,
                "modelId": model.id,
                "label": f"{model.provider}/{model.id}",
                "description": f"contextWindow={model.context_window}",
                "isCurrent": model.provider == current_model["provider"] and model.id == current_model["id"],
                "position": index,
            }
            for index, model in enumerate(models)
        ]
        return {
            "currentModel": current_model,
            "itemOrder": [item["id"] for item in items],
            "items": items,
        }

    async def export_to_html(self, output_path: str | None = None) -> str:
        session_file = self.session_file
        if session_file is None:
            raise ValueError("Cannot export in-memory session to HTML")
        return export_from_file(session_file, output_path)

    async def switch_session(self, session_path: str) -> bool:
        manager = SessionManager.open(session_path)
        self.session_manager = manager
        self.agent.reset()
        self._restore_messages_from_current_branch()
        return True

    def resolve_session(self, query: str) -> str:
        current_file = self.session_file
        if current_file is not None:
            current_info = SessionManager.read_session_info(current_file)
            normalized_query = query.strip().lower()
            current_path = Path(current_file)
            if normalized_query in {
                current_path.name.lower(),
                current_path.stem.lower(),
                self.session_id.lower(),
            }:
                return current_file
            if current_info is not None:
                if current_info.name and normalized_query in current_info.name.lower():
                    return current_file
                if normalized_query and normalized_query in current_info.first_message.lower():
                    return current_file
        return str(SessionManager.resolve_session(self.cwd, query))

    async def resolve_and_switch_session(self, query: str) -> str:
        session_path = self.resolve_session(query)
        await self.switch_session(session_path)
        return session_path

    def get_fork_messages(self) -> list[dict[str, str]]:
        return self.session_manager.get_user_messages_for_forking()

    async def fork(self, entry_id: str) -> dict[str, Any]:
        new_manager, selected_text = self.session_manager.fork_to_new_manager(entry_id)
        self.session_manager = new_manager
        self._restore_messages_from_current_branch()
        return {"selectedText": selected_text, "cancelled": False}

    def branch_with_summary(self, entry_id: str | None, summary: str) -> dict[str, Any]:
        summary = summary.strip()
        if not summary:
            raise ValueError("Summary cannot be empty")
        summary_entry_id = self.session_manager.branch_with_summary(entry_id, summary)
        self._restore_messages_from_current_branch()
        return {
            "entryId": entry_id,
            "summaryEntryId": summary_entry_id,
            "leafId": self.session_manager.get_leaf_id(),
            "cancelled": False,
        }

    def compact(self, first_kept_entry_id: str, summary: str, tokens_before: int) -> dict[str, Any]:
        summary = summary.strip()
        if not summary:
            raise ValueError("Summary cannot be empty")
        compaction_entry_id = self.session_manager.append_compaction(summary, first_kept_entry_id, tokens_before)
        self._restore_messages_from_current_branch()
        return {
            "compactionEntryId": compaction_entry_id,
            "leafId": self.session_manager.get_leaf_id(),
            "cancelled": False,
        }

    async def generate_summary(self, custom_instructions: str | None = None) -> dict[str, Any]:
        summary = await generate_summary(self.model, self.messages, custom_instructions)
        return {"summary": summary}

    async def auto_branch_with_summary(self, entry_id: str | None, custom_instructions: str | None = None) -> dict[str, Any]:
        summary = await generate_summary(self.model, self.messages, custom_instructions)
        return self.branch_with_summary(entry_id, summary)

    async def auto_compact(self, first_kept_entry_id: str, custom_instructions: str | None = None) -> dict[str, Any]:
        summary = await generate_summary(self.model, self.messages, custom_instructions)
        tokens_before = max(len(summary) // 4, 1)
        return self.compact(first_kept_entry_id, summary, tokens_before)

    async def _maybe_auto_compact(self) -> None:
        settings = self.settings_manager.get_settings().compaction
        if not settings.enabled:
            return
        current_messages = self.messages
        estimated_tokens = estimate_tokens(current_messages)
        threshold = max(self.model.context_window - settings.reserve_tokens, 0)
        if estimated_tokens <= threshold:
            return

        first_kept_entry_id = self._find_first_kept_entry_id(settings.keep_recent_tokens)
        if first_kept_entry_id is None:
            return

        result = await self.auto_compact(first_kept_entry_id)
        self._emit({"type": "auto_compaction_end", "data": result})

    def _find_first_kept_entry_id(self, keep_recent_tokens: int) -> str | None:
        branch_entries = self.session_manager.get_branch_entries()
        message_entries = [entry for entry in branch_entries if entry["type"] == "message"]
        if len(message_entries) < 2:
            return None

        kept: list[dict[str, Any]] = []
        kept_tokens = 0
        for entry in reversed(message_entries):
            kept.insert(0, entry)
            kept_tokens += _estimate_message_entry_tokens(entry["message"])
            if kept_tokens >= keep_recent_tokens:
                break

        if not kept or kept[0]["id"] == message_entries[0]["id"]:
            return None
        return kept[0]["id"]

    def get_last_assistant_text(self) -> str | None:
        return self.session_manager.get_last_assistant_text()

    def list_sessions(self) -> list[str]:
        return [str(path) for path in SessionManager.list_sessions(self.cwd)]

    def list_session_infos(self) -> list[SessionInfo]:
        return SessionManager.list_session_infos(self.cwd)

    def get_session_selector_schema(self) -> dict[str, Any]:
        current_file = self.session_file
        infos = self.list_session_infos()
        items: list[dict[str, Any]] = []
        seen_paths: set[str] = set()

        for index, info in enumerate(infos):
            seen_paths.add(info.path)
            items.append(
                {
                    "id": info.id,
                    "path": info.path,
                    "name": info.name,
                    "label": info.name or Path(info.path).stem,
                    "description": info.first_message,
                    "messageCount": info.message_count,
                    "created": info.created,
                    "modified": info.modified,
                    "firstMessage": info.first_message,
                    "cwd": info.cwd,
                    "isCurrent": info.path == current_file,
                    "position": index,
                }
            )

        if current_file is not None and current_file not in seen_paths:
            current_info = SessionManager.read_session_info(current_file)
            if current_info is not None:
                items.insert(
                    0,
                    {
                        "id": current_info.id,
                        "path": current_info.path,
                        "name": current_info.name,
                        "label": current_info.name or Path(current_info.path).stem,
                        "description": current_info.first_message,
                        "messageCount": current_info.message_count,
                        "created": current_info.created,
                        "modified": current_info.modified,
                        "firstMessage": current_info.first_message,
                        "cwd": current_info.cwd,
                        "isCurrent": True,
                        "position": 0,
                    },
                )
                for index, item in enumerate(items):
                    item["position"] = index

        return {
            "currentSessionFile": current_file,
            "itemOrder": [item["path"] for item in items],
            "items": items,
        }

    def get_tree(self) -> list[dict[str, Any]]:
        return self.session_manager.get_tree()

    def get_tree_schema(self) -> dict[str, Any]:
        branch_ids = {entry["id"] for entry in self.session_manager.get_branch_entries()}
        items: list[dict[str, Any]] = []

        def visit(nodes: list[dict[str, Any]], depth: int) -> None:
            for index, node in enumerate(nodes):
                entry = node["entry"]
                children = node["children"]
                items.append(
                    {
                        "id": entry["id"],
                        "parentId": entry.get("parentId"),
                        "type": entry["type"],
                        "label": entry["label"],
                        "timestamp": entry["timestamp"],
                        "depth": depth,
                        "childCount": len(children),
                        "isLeaf": bool(node.get("isLeaf")),
                        "isOnCurrentBranch": entry["id"] in branch_ids,
                        "position": index,
                    }
                )
                visit(children, depth + 1)

        tree = self.get_tree()
        visit(tree, 0)
        return {
            "currentLeafId": self.session_manager.get_leaf_id(),
            "itemOrder": [item["id"] for item in items],
            "items": items,
        }

    def branch(self, entry_id: str | None) -> dict[str, Any]:
        self.session_manager.branch(entry_id)
        self._restore_messages_from_current_branch()
        return {"entryId": entry_id, "leafId": self.session_manager.get_leaf_id(), "cancelled": False}

    def get_state(self) -> SessionStateSnapshot:
        return SessionStateSnapshot(
            model=self.agent.state.model,
            thinking_level=self.agent.state.thinking_level,
            is_streaming=self.agent.state.is_streaming,
            session_file=self.session_file,
            session_id=self.session_id,
            message_count=len(self.agent.state.messages),
            steering_mode=self.agent.get_steering_mode(),
            follow_up_mode=self.agent.get_follow_up_mode(),
            pending_message_count=self.agent.get_pending_message_count(),
        )


def create_default_model():
    return get_model("openai", get_env_default_model() or "gpt-4o-mini")
