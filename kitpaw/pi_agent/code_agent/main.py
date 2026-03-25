from __future__ import annotations

import asyncio
import json
import os
import sys

from .cli_args import build_parser
from .config import get_agent_dir
from .export_html import export_from_file
from .modes.interactive_mode import run_interactive_mode
from .modes.json_mode import run_json_mode
from .modes.print_mode import run_print_mode
from .modes.rpc_mode import run_rpc_mode
from .package_manager import PackageManager
from .sdk import CreateAgentSessionOptions, create_agent_session
from .session_manager import SessionManager
from .session_picker import select_session
from .tools import create_all_tools


def _find_env_source(key: str) -> str | None:
    """Find which .kitpaw file defines *key* (first match wins, same order as load)."""
    from ..ai.local_env import kitpaw_env_files

    for path in kitpaw_env_files():
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if stripped.startswith(key) and "=" in stripped:
                    return str(path)
        except OSError:
            pass
    return None


def _mask_key(key: str) -> str:
    if len(key) > 12:
        return key[:8] + "..." + key[-4:]
    return "***"


def _print_openai_provider_config(session, args) -> None:
    """Print the resolved OpenAI provider configuration and its sources."""
    from ..ai.env_api_keys import DEFAULT_OPENAI_BASE_URL
    from .config import get_auth_path

    model = session.model
    settings_manager = session.settings_manager
    auth_storage = session.model_registry.auth_storage

    print("\n╭─ OpenAI Provider Configuration ─────────────────────────────╮")

    # --- Provider ---
    provider_value = model.provider if model else "openai"
    if args.provider:
        provider_source = f"CLI --provider={args.provider}"
    elif settings_manager.get_default_provider():
        project_raw = settings_manager._load_json(settings_manager.project_path)
        if "defaultProvider" in project_raw:
            provider_source = f"project settings ({settings_manager.project_path})"
        else:
            provider_source = f"global settings ({settings_manager.global_path})"
    else:
        provider_source = "default"
    print(f"│  provider:   {provider_value:<20s}  ← {provider_source}")

    # --- Model ---
    model_id = model.id if model else "gpt-4o-mini"
    if args.model:
        model_source = f"CLI --model={args.model}"
    else:
        sm = session.session_manager
        context = sm.build_runtime_context() if sm.entries else {}
        restored_model = context.get("model")
        if restored_model:
            model_source = "restored session"
        elif settings_manager.get_default_model():
            project_raw = settings_manager._load_json(settings_manager.project_path)
            if "defaultModel" in project_raw:
                model_source = f"project settings ({settings_manager.project_path})"
            else:
                model_source = f"global settings ({settings_manager.global_path})"
        else:
            env_model = os.getenv("OPENAI_MODEL")
            if env_model:
                src = _find_env_source("OPENAI_MODEL")
                model_source = f"env OPENAI_MODEL (from {src})" if src else "env OPENAI_MODEL"
            else:
                model_source = "default (gpt-4o-mini)"
    print(f"│  model:      {model_id:<20s}  ← {model_source}")

    # --- Base URL ---
    base_url = model.base_url if model else DEFAULT_OPENAI_BASE_URL
    env_base = os.getenv("OPENAI_BASE_URL")
    if env_base:
        src = _find_env_source("OPENAI_BASE_URL")
        base_url_source = f"env OPENAI_BASE_URL (from {src})" if src else "env OPENAI_BASE_URL"
    else:
        base_url_source = f"default ({DEFAULT_OPENAI_BASE_URL})"
    print(f"│  base_url:   {base_url:<20s}  ← {base_url_source}")

    # --- API Key ---
    if args.api_key:
        api_key_source = "CLI --api-key"
        api_key_display = _mask_key(args.api_key)
    else:
        runtime_key = auth_storage._runtime_keys.get("openai")
        if runtime_key:
            api_key_source = "runtime (set via CLI)"
            api_key_display = _mask_key(runtime_key)
        else:
            auth_path = get_auth_path()
            auth_data = {}
            if auth_path.exists():
                try:
                    auth_data = json.loads(auth_path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    pass
            cred = auth_data.get("openai")
            if isinstance(cred, dict) and cred.get("type") == "api_key" and cred.get("key"):
                key = cred["key"]
                api_key_source = f"auth.json ({auth_path})"
                api_key_display = _mask_key(key)
            else:
                env_key = os.getenv("OPENAI_API_KEY")
                if env_key:
                    src = _find_env_source("OPENAI_API_KEY")
                    api_key_source = f"env OPENAI_API_KEY (from {src})" if src else "env OPENAI_API_KEY"
                    api_key_display = _mask_key(env_key)
                else:
                    api_key_source = "⚠ NOT SET"
                    api_key_display = "(none)"
    print(f"│  api_key:    {api_key_display:<20s}  ← {api_key_source}")

    # --- Thinking level ---
    thinking = session.agent.state.thinking_level if hasattr(session.agent.state, 'thinking_level') else "medium"
    if args.thinking:
        thinking_source = f"CLI --thinking={args.thinking}"
    elif settings_manager.get_default_thinking_level():
        project_raw = settings_manager._load_json(settings_manager.project_path)
        if "defaultThinkingLevel" in project_raw:
            thinking_source = f"project settings ({settings_manager.project_path})"
        else:
            thinking_source = f"global settings ({settings_manager.global_path})"
    else:
        thinking_source = "default (medium)"
    print(f"│  thinking:   {thinking:<20s}  ← {thinking_source}")

    print("╰─────────────────────────────────────────────────────────────╯\n")


async def amain(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    command = args.command
    package_commands = {"install", "remove", "uninstall", "update", "list"}
    if command in package_commands:
        package_manager = PackageManager(os.getcwd(), str(get_agent_dir()), None)
        if command == "install":
            if not args.messages:
                raise SystemExit("install requires a source")
            path = package_manager.install(args.messages[0], local=args.local)
            print(f"Installed {args.messages[0]} -> {path}")
            return 0
        if command in {"remove", "uninstall"}:
            if not args.messages:
                raise SystemExit("remove requires a source")
            removed = package_manager.remove(args.messages[0], local=args.local)
            if not removed:
                raise SystemExit(f"No matching package found for {args.messages[0]}")
            print(f"Removed {args.messages[0]}")
            return 0
        if command == "update":
            updated = package_manager.update(args.messages[0] if args.messages else None)
            print("\n".join(updated) if updated else "No packages updated.")
            return 0
        if command == "list":
            packages = package_manager.list()
            if not packages:
                print("No packages installed.")
                return 0
            for package in packages:
                print(f"{package.scope}: {package.source}")
                print(f"  {package.path}")
            return 0
    if command and command not in package_commands:
        args.messages = [command, *args.messages]

    if args.export:
        print(export_from_file(args.export))
        return 0

    session_manager = None
    if args.no_session:
        session_manager = SessionManager.in_memory()
    elif args.session:
        resolved = SessionManager.resolve_session(os.getcwd(), args.session, args.session_dir)
        session_manager = SessionManager.open(resolved)
    elif args.continue_session or args.resume:
        if args.resume and args.resume != "__latest__":
            session_file = select_session(os.getcwd(), args.session_dir, args.resume)
        else:
            session_file = select_session(os.getcwd(), args.session_dir, None)
        if session_file is not None:
            session_manager = SessionManager.open(session_file)

    options = CreateAgentSessionOptions(
        cwd=os.getcwd(),
        agent_dir=str(get_agent_dir()),
        session_dir=args.session_dir,
        session_manager=session_manager,
    )

    if args.agent == "zed":
        from .zed.compaction import configure_zed_compaction
        from .zed.resource_loader import ZedResourceLoader
        from .zed.tools import create_zed_tools

        zed_tools = create_zed_tools(os.getcwd())
        zed_loader = ZedResourceLoader(
            os.getcwd(), str(get_agent_dir()), None,
        )
        zed_loader.set_tool_names([t.name for t in zed_tools])
        options.resource_loader = zed_loader
        options.tools = zed_tools

    elif args.agent == "codex":
        from .codex.resource_loader import CodexResourceLoader
        from .codex.tools import create_codex_tools

        codex_tools = create_codex_tools(os.getcwd())
        codex_loader = CodexResourceLoader(
            os.getcwd(), str(get_agent_dir()), None,
        )
        codex_loader.set_tool_names([t.name for t in codex_tools])
        options.resource_loader = codex_loader
        options.tools = codex_tools

    result = await create_agent_session(options)
    session = result.session

    if args.agent == "zed":
        configure_zed_compaction(session)
        # Rebuild tools with parent_agent now available for spawn_agent
        zed_tools_with_parent = create_zed_tools(os.getcwd(), parent_agent=session.agent)
        session.agent.set_tools(zed_tools_with_parent)
        # Rebuild system prompt with model name now available
        tool_names = [t.name for t in session.agent.state.tools]
        zed_loader = session.resource_loader
        if hasattr(zed_loader, "build_system_prompt_with_tools"):
            session.agent.set_system_prompt(
                zed_loader.build_system_prompt_with_tools(
                    tool_names, model_name=session.model.name,
                )
            )

    elif args.agent == "codex":
        from .codex.compaction import configure_codex_compaction

        configure_codex_compaction(session)

        # Force system role (not developer) to match Codex Rust
        from dataclasses import replace as _replace
        from ..ai.types import OpenAICompletionsCompat

        model = session.model
        current_compat = model.compat or OpenAICompletionsCompat()
        session.agent._state.model = _replace(
            model, compat=_replace(current_compat, supports_developer_role=False),
        )

        # Wrap convert_to_llm to inject extra messages matching Codex Rust's
        # message structure: permissions in system prompt, AGENTS.md + env_context
        # as user messages before the user's prompt.
        codex_loader = session.resource_loader
        if hasattr(codex_loader, "get_permissions_message"):
            from ..ai import UserMessage
            from ..ai.types import TextContent

            # Append permissions to system prompt (Codex sends as developer role,
            # but since some providers don't support developer role, we merge it)
            perms_text = codex_loader.get_permissions_message()
            current_sp = session.agent.state.system_prompt
            session.agent.set_system_prompt(current_sp + "\n\n" + perms_text)

            # Build prefix user messages: AGENTS.md + env_context
            agents_md_msgs = codex_loader.get_agents_md_messages()
            env_msg = codex_loader.get_environment_context_message()

            prefix_messages = []
            for am in agents_md_msgs:
                prefix_messages.append(UserMessage(content=[TextContent(text=am["content"])]))
            prefix_messages.append(UserMessage(content=[TextContent(text=env_msg["content"])]))

            original_convert = session.agent.convert_to_llm

            def codex_convert_to_llm(messages):
                converted = original_convert(messages)
                return list(prefix_messages) + list(converted)

            session.agent.convert_to_llm = codex_convert_to_llm

    if args.provider and args.model:
        await session.set_model(args.provider, args.model)
    elif args.model:
        await session.set_model("openai", args.model)
    if args.theme:
        session.set_theme(args.theme)
    if args.thinking:
        session.set_thinking_level(args.thinking)
    if args.api_key:
        # Store API key in auth_storage in-memory override rather than
        # setting it as an environment variable (which leaks to child processes).
        session.model_registry.auth_storage.set_runtime_api_key(
            args.provider or "openai", args.api_key
        )
    if args.tools:
        tool_names = [name.strip() for name in args.tools.split(",") if name.strip()]
        if args.agent == "zed":
            from .zed.tools import create_zed_tools as _create_zed_tools

            all_zed = {t.name: t for t in _create_zed_tools(session.cwd)}
            session.agent.set_tools([all_zed[n] for n in tool_names if n in all_zed])
        elif args.agent == "codex":
            from .codex.tools import create_codex_tools as _create_codex_tools

            all_codex = {t.name: t for t in _create_codex_tools(session.cwd)}
            session.agent.set_tools([all_codex[n] for n in tool_names if n in all_codex])
        else:
            all_tools = create_all_tools(session.cwd, command_prefix=session.settings_manager.get_shell_command_prefix())
            session.agent.set_tools([all_tools[name] for name in tool_names if name in all_tools])

    if args.system_prompt:
        sp = args.system_prompt
        if sp.startswith("@"):
            path = sp[1:]
            try:
                with open(path, encoding="utf-8") as f:
                    sp = f.read()
            except OSError as e:
                print(f"Error reading system prompt file: {e}", file=sys.stderr)
                return 1
        session.agent.set_system_prompt(sp)

    message = " ".join(args.messages).strip()

    # Check that an API key is available before entering any mode.
    provider = args.provider or "openai"
    api_key = session.model_registry.auth_storage.get_api_key(provider)
    if not api_key:
        print(
            f"Error: No API key found for provider '{provider}'.\n"
            f"\n"
            f"Set one of the following:\n"
            f"  1. OPENAI_API_KEY environment variable\n"
            f"  2. --api-key flag\n"
            f"  3. .env.local file with OPENAI_API_KEY=...",
            file=sys.stderr,
        )
        return 1

    # Set up JSONL error logging if requested.
    error_log_cleanup = None
    if args.error_log_jsonl:
        from .error_logger import setup_error_logger

        error_log_cleanup = setup_error_logger(session, args.error_log_jsonl)

    try:
        if args.mode == "rpc":
            return await run_rpc_mode(session)
        if args.mode == "json":
            return await run_json_mode(session, message)
        if args.print_mode:
            return await run_print_mode(session, message)
        if message:
            return await run_print_mode(session, message)

        _print_openai_provider_config(session, args)
        return await run_interactive_mode(session)
    finally:
        if error_log_cleanup is not None:
            error_log_cleanup()


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(amain(argv))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
