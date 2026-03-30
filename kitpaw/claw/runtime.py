from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..pi_agent.code_agent.agent_session import AgentSession, create_default_model
from ..pi_agent.code_agent.config import get_agent_dir
from ..pi_agent.code_agent.resource_loader import ResourceLoader
from ..pi_agent.code_agent.sdk import CreateAgentSessionOptions, create_agent_session
from ..pi_agent.code_agent.session_manager import SessionManager
from .context import OpenClawToolContext, YieldCallback
from .resource_loader import ClawResourceLoader
from .tools import create_openclaw_coding_tools


@dataclass(slots=True)
class CreateClawSessionOptions:
    cwd: str | None = None
    workspace_dir: str | None = None
    spawn_workspace_dir: str | None = None
    agent_dir: str | None = None
    session_dir: str | None = None
    model: Any = None
    thinking_level: str | None = None
    system_prompt: str | None = None
    session_manager: SessionManager | None = None
    settings_manager: Any = None
    auth_storage: Any = None
    model_registry: Any = None
    resource_loader: ResourceLoader | None = None
    agent_id: str = "claw"
    sandboxed: bool = False
    on_yield: YieldCallback | None = None
    prompt_mode: str = "full"


@dataclass(slots=True)
class CreateClawSessionResult:
    session: AgentSession
    resource_loader: ResourceLoader


def _resolve_path(value: str | None, fallback: str) -> str:
    return str(Path(value or fallback).resolve())


def _build_tool_context(
    *,
    workspace_dir: str,
    spawn_workspace_dir: str,
    agent_id: str,
    session_id: str,
    model_provider: str,
    model_id: str,
    thinking_level: str,
    sandboxed: bool,
    prompt_mode: str,
    system_prompt: str | None,
    system_prompt_is_override: bool,
    on_yield: YieldCallback | None,
) -> OpenClawToolContext:
    return OpenClawToolContext(
        cwd=workspace_dir,
        workspace_dir=workspace_dir,
        spawn_workspace_dir=spawn_workspace_dir,
        agent_id=agent_id,
        session_id=session_id,
        controller_session_id=session_id,
        model_provider=model_provider,
        model_id=model_id,
        thinking_level=thinking_level,
        sandboxed=sandboxed,
        prompt_mode=prompt_mode,
        system_prompt=system_prompt,
        system_prompt_is_override=system_prompt_is_override,
        on_yield=on_yield,
    )


def _resolve_final_prompt(
    resource_loader: ResourceLoader,
    session: AgentSession,
    tool_names: list[str],
    override: str | None,
    *,
    prompt_mode: str,
    agent_id: str,
) -> str:
    if override is not None:
        return override
    if hasattr(resource_loader, "build_system_prompt_with_tools"):
        return resource_loader.build_system_prompt_with_tools(
            tool_names,
            model_name=getattr(session.model, "name", None),
            thinking_level=session.thinking_level,
            agent_id=agent_id,
            prompt_mode=prompt_mode,
        )
    return session.system_prompt


async def create_claw_session(
    options: CreateClawSessionOptions | None = None,
) -> CreateClawSessionResult:
    opts = options or CreateClawSessionOptions()
    workspace_dir = _resolve_path(opts.workspace_dir, opts.cwd or ".")
    spawn_workspace_dir = _resolve_path(opts.spawn_workspace_dir, workspace_dir)
    agent_dir = _resolve_path(opts.agent_dir, str(get_agent_dir()))
    session_manager = opts.session_manager or SessionManager.create(workspace_dir, opts.session_dir)

    provisional_model = opts.model or create_default_model()
    provisional_thinking = opts.thinking_level or "medium"
    provisional_context = _build_tool_context(
        workspace_dir=workspace_dir,
        spawn_workspace_dir=spawn_workspace_dir,
        agent_id=opts.agent_id,
        session_id=session_manager.get_session_id(),
        model_provider=provisional_model.provider,
        model_id=provisional_model.id,
        thinking_level=provisional_thinking,
        sandboxed=opts.sandboxed,
        prompt_mode=opts.prompt_mode,
        system_prompt=None,
        system_prompt_is_override=opts.system_prompt is not None,
        on_yield=opts.on_yield,
    )
    resource_loader = opts.resource_loader or ClawResourceLoader(
        workspace_dir,
        agent_dir,
        opts.settings_manager,
    )
    initial_tools = create_openclaw_coding_tools(workspace_dir, provisional_context)

    result = await create_agent_session(
        CreateAgentSessionOptions(
            cwd=workspace_dir,
            agent_dir=agent_dir,
            session_dir=opts.session_dir,
            model=opts.model,
            thinking_level=opts.thinking_level,
            tools=initial_tools,
            session_manager=session_manager,
            settings_manager=opts.settings_manager,
            auth_storage=opts.auth_storage,
            model_registry=opts.model_registry,
            resource_loader=resource_loader,
        )
    )
    session = result.session
    final_prompt = _resolve_final_prompt(
        resource_loader,
        session,
        [tool.name for tool in initial_tools],
        opts.system_prompt,
        prompt_mode=opts.prompt_mode,
        agent_id=opts.agent_id,
    )
    final_model = session.model or provisional_model
    final_context = _build_tool_context(
        workspace_dir=workspace_dir,
        spawn_workspace_dir=spawn_workspace_dir,
        agent_id=opts.agent_id,
        session_id=session.session_id,
        model_provider=final_model.provider,
        model_id=final_model.id,
        thinking_level=session.thinking_level,
        sandboxed=opts.sandboxed,
        prompt_mode=opts.prompt_mode,
        system_prompt=final_prompt,
        system_prompt_is_override=opts.system_prompt is not None,
        on_yield=opts.on_yield,
    )
    final_tools = create_openclaw_coding_tools(workspace_dir, final_context)
    session.agent.set_tools(final_tools)
    session.agent.set_system_prompt(final_prompt)
    return CreateClawSessionResult(session=session, resource_loader=resource_loader)
