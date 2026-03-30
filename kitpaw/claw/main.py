from __future__ import annotations

import asyncio
import os
import sys

from ..pi_agent.ai import get_model
from ..pi_agent.code_agent.config import get_agent_dir
from ..pi_agent.code_agent.export_html import export_from_file
from ..pi_agent.code_agent.main import _print_openai_provider_config
from ..pi_agent.code_agent.modes.interactive_mode import run_interactive_mode
from ..pi_agent.code_agent.modes.json_mode import run_json_mode
from ..pi_agent.code_agent.modes.print_mode import run_print_mode
from ..pi_agent.code_agent.modes.rpc_mode import run_rpc_mode
from ..pi_agent.code_agent.session_manager import SessionManager
from ..pi_agent.code_agent.session_picker import select_session
from ..pi_agent.code_agent.tool_error_limit import (
    ToolErrorLimitExceededError,
    configure_tool_error_limit,
    consume_tool_error_limit_exception,
)
from .cli_args import build_parser
from .runtime import CreateClawSessionOptions, create_claw_session


def _load_system_prompt(path_or_text: str | None) -> str | None:
    if not path_or_text:
        return None
    if not path_or_text.startswith("@"):
        return path_or_text
    path = path_or_text[1:]
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def _build_session_manager(args, cwd: str) -> SessionManager | None:
    if args.no_session:
        return SessionManager.in_memory(cwd)
    if args.session:
        resolved = SessionManager.resolve_session(cwd, args.session, args.session_dir)
        return SessionManager.open(resolved)
    if args.continue_session or args.resume:
        if args.resume and args.resume != "__latest__":
            session_file = select_session(cwd, args.session_dir, args.resume)
        else:
            session_file = select_session(cwd, args.session_dir, None)
        if session_file is not None:
            return SessionManager.open(session_file)
    return None


async def amain(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command:
        args.messages = [args.command, *args.messages]

    if args.export:
        print(export_from_file(args.export))
        return 0

    cwd = os.getcwd()
    session_manager = _build_session_manager(args, cwd)

    model = None
    if args.provider and args.model:
        model = get_model(args.provider, args.model)
    elif args.model:
        model = get_model("openai", args.model)

    try:
        system_prompt = _load_system_prompt(args.system_prompt)
    except OSError as exc:
        print(f"Error reading system prompt file: {exc}", file=sys.stderr)
        return 1

    result = await create_claw_session(
        CreateClawSessionOptions(
            cwd=cwd,
            agent_dir=str(get_agent_dir()),
            session_dir=args.session_dir,
            model=model,
            thinking_level=args.thinking,
            system_prompt=system_prompt,
            session_manager=session_manager,
        )
    )
    session = result.session

    runtime_provider = session.model.provider if session.model is not None else "openai"
    if args.api_key:
        session.model_registry.auth_storage.set_runtime_api_key(args.provider or runtime_provider, args.api_key)

    if args.max_tool_errors is not None:
        if args.max_tool_errors < 1:
            print("Error: --max-tool-errors must be at least 1.", file=sys.stderr)
            return 1
        configure_tool_error_limit(session, args.max_tool_errors)

    api_key = session.model_registry.auth_storage.get_api_key(runtime_provider)
    if not api_key:
        print(
            f"Error: No API key found for provider '{runtime_provider}'.\n"
            f"\n"
            f"Set one of the following:\n"
            f"  1. OPENAI_API_KEY environment variable\n"
            f"  2. --api-key flag\n"
            f"  3. .env.local file with OPENAI_API_KEY=...",
            file=sys.stderr,
        )
        return 1

    error_log_cleanup = None
    if args.error_log_jsonl:
        from ..pi_agent.code_agent.error_logger import setup_error_logger

        error_log_cleanup = setup_error_logger(session, args.error_log_jsonl)

    message = " ".join(args.messages).strip()
    try:
        try:
            if args.mode == "rpc":
                return await run_rpc_mode(session)
            if args.mode == "json":
                exit_code = await run_json_mode(session, message)
                if exit_code != 0 and (limit_error := consume_tool_error_limit_exception(session)) is not None:
                    print(f"Error: {limit_error}", file=sys.stderr)
                return exit_code
            if args.print_mode:
                exit_code = await run_print_mode(session, message)
                if exit_code != 0 and (limit_error := consume_tool_error_limit_exception(session)) is not None:
                    print(f"Error: {limit_error}", file=sys.stderr)
                return exit_code
            if message:
                exit_code = await run_print_mode(session, message)
                if exit_code != 0 and (limit_error := consume_tool_error_limit_exception(session)) is not None:
                    print(f"Error: {limit_error}", file=sys.stderr)
                return exit_code

            _print_openai_provider_config(session, args)
            exit_code = await run_interactive_mode(
                session,
                banner_lines=("claw interactive mode", "Type /help for commands."),
            )
            if exit_code != 0 and (limit_error := consume_tool_error_limit_exception(session)) is not None:
                print(f"Error: {limit_error}", file=sys.stderr)
            return exit_code
        except ToolErrorLimitExceededError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
    finally:
        if error_log_cleanup is not None:
            error_log_cleanup()


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(amain(argv))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
