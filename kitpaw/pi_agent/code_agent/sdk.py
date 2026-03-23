from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..agent import Agent
from .agent_session import AgentSession, AgentSessionConfig, create_default_model
from .auth_storage import AuthStorage
from .config import get_agent_dir, get_auth_path
from .message_restore import restore_message
from .messages import convert_to_llm
from .model_registry import ModelRegistry
from .resource_loader import DefaultResourceLoader, ResourceLoader
from .session_manager import SessionManager, infer_session_dir
from .settings_manager import SettingsManager
from .tools import (
    create_all_tools,
    create_bash_tool,
    create_coding_tools,
    create_edit_tool,
    create_find_tool,
    create_grep_tool,
    create_ls_tool,
    create_read_only_tools,
    create_read_tool,
    create_uv_tool,
    create_write_tool,
)


@dataclass(slots=True)
class CreateAgentSessionOptions:
    cwd: str | None = None
    agent_dir: str | None = None
    session_dir: str | None = None
    model: Any = None
    thinking_level: str | None = None
    tools: Any = None
    session_manager: Any = None
    settings_manager: Any = None
    auth_storage: Any = None
    model_registry: Any = None
    resource_loader: ResourceLoader | None = None


@dataclass(slots=True)
class CreateAgentSessionResult:
    session: AgentSession
    extensions_result: object | None = None
    model_fallback_message: str | None = None


async def create_agent_session(options: CreateAgentSessionOptions | None = None) -> CreateAgentSessionResult:
    opts = options or CreateAgentSessionOptions()
    cwd = str(Path(opts.cwd or ".").resolve())
    agent_dir = opts.agent_dir or str(get_agent_dir())
    auth_storage = opts.auth_storage or AuthStorage.create(get_auth_path() if opts.agent_dir is None else Path(agent_dir) / "auth.json")
    model_registry = opts.model_registry or ModelRegistry(auth_storage)
    settings_manager = opts.settings_manager or SettingsManager.create(cwd, agent_dir)
    resource_loader = opts.resource_loader or DefaultResourceLoader(cwd, agent_dir, settings_manager)
    await resource_loader.reload()
    session_manager = opts.session_manager or SessionManager.create(cwd, opts.session_dir)
    effective_session_dir = opts.session_dir
    if effective_session_dir is None:
        session_file = session_manager.get_session_file()
        if session_file is not None:
            effective_session_dir = infer_session_dir(session_file)
    context = session_manager.build_runtime_context() if session_manager.entries else {"messages": [], "model": None, "thinkingLevel": None}
    restored_messages = [restore_message(message) for message in context["messages"] if isinstance(message, dict)]
    tools = opts.tools or create_coding_tools(cwd, command_prefix=settings_manager.get_shell_command_prefix())
    restored_model = context.get("model")
    model = opts.model or (
        model_registry.find(restored_model["provider"], restored_model["modelId"]) if restored_model else create_default_model()
    )
    thinking_level = opts.thinking_level or context.get("thinkingLevel") or settings_manager.get_default_thinking_level() or "medium"
    agent = Agent(
        {
            "initial_state": {
                "model": model,
                "system_prompt": resource_loader.get_system_prompt() or "",
                "thinking_level": thinking_level,
                "tools": tools,
                "messages": restored_messages,
            },
            "convert_to_llm": convert_to_llm,
            "steering_mode": settings_manager.get_steering_mode(),
            "follow_up_mode": settings_manager.get_follow_up_mode(),
            "get_api_key": lambda provider: auth_storage.get_api_key(provider),
        }
    )
    session = AgentSession(
        AgentSessionConfig(
            agent=agent,
            session_manager=session_manager,
            settings_manager=settings_manager,
            cwd=cwd,
            session_dir=effective_session_dir,
            model_registry=model_registry,
            resource_loader=resource_loader,
        )
    )
    return CreateAgentSessionResult(session=session, extensions_result=resource_loader.get_extensions(), model_fallback_message=None)


__all__ = [
    "AgentSession",
    "CreateAgentSessionOptions",
    "CreateAgentSessionResult",
    "create_agent_session",
    "create_all_tools",
    "create_bash_tool",
    "create_coding_tools",
    "create_edit_tool",
    "create_find_tool",
    "create_grep_tool",
    "create_ls_tool",
    "create_read_only_tools",
    "create_read_tool",
    "create_uv_tool",
    "create_write_tool",
]
