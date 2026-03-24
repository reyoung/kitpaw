from __future__ import annotations

import asyncio
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
        else:
            all_tools = create_all_tools(session.cwd, command_prefix=session.settings_manager.get_shell_command_prefix())
            session.agent.set_tools([all_tools[name] for name in tool_names if name in all_tools])

    message = " ".join(args.messages).strip()
    if args.mode == "rpc":
        return await run_rpc_mode(session)
    if args.mode == "json":
        return await run_json_mode(session, message)
    if args.print_mode:
        return await run_print_mode(session, message)
    if message:
        return await run_print_mode(session, message)
    return await run_interactive_mode(session)


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(amain(argv))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
