from __future__ import annotations

from pathlib import Path

from .session_manager import SessionManager
from .types import SessionInfo


def _shorten_path(path: str) -> str:
    if not path:
        return path
    home = str(Path.home())
    if path == home:
        return "~"
    if path.startswith(home):
        return "~" + path[len(home) :]
    return path


def _print_session_info(index: int, info: SessionInfo) -> None:
    name = f" [{info.name}]" if info.name else ""
    print(f"{index}. {_shorten_path(info.path)}{name}")
    print(f"   modified={info.modified} messages={info.message_count} first={info.first_message}")


def _print_session_list(infos: list[SessionInfo]) -> None:
    for index, info in enumerate(infos, start=1):
        _print_session_info(index, info)


def _resolve_scope(query: str | None) -> tuple[str, str | None]:
    if not query:
        return "current", None
    lowered = query.strip().lower()
    if lowered.startswith("all:"):
        return "all", query.split(":", 1)[1].strip()
    if lowered.startswith("current:"):
        return "current", query.split(":", 1)[1].strip()
    return "current", query


def _resolve_session_infos_for_scope(
    scope: str,
    cwd: str,
    query: str,
    session_dir: str | Path | None,
) -> list[SessionInfo]:
    if scope == "all":
        return SessionManager.resolve_all_session_infos(query, session_dir)
    return SessionManager.resolve_session_infos(cwd, query, session_dir)


def _search_session_infos_for_scope(
    scope: str,
    cwd: str,
    query: str,
    session_dir: str | Path | None,
) -> list[SessionInfo]:
    if scope == "all":
        return SessionManager.search_all_session_infos(query, session_dir)
    return SessionManager.search_session_infos(cwd, query, session_dir)


def _resolve_session_path_for_scope(
    scope: str,
    cwd: str,
    query: str,
    session_dir: str | Path | None,
) -> tuple[Path | None, list[SessionInfo]]:
    candidate = Path(query).expanduser()
    if candidate.exists():
        return candidate.resolve(), []

    strict_matches = _resolve_session_infos_for_scope(scope, cwd, query, session_dir)
    if len(strict_matches) == 1:
        return Path(strict_matches[0].path), []
    if strict_matches:
        return None, strict_matches

    search_matches = _search_session_infos_for_scope(scope, cwd, query, session_dir)
    if len(search_matches) == 1:
        return Path(search_matches[0].path), []
    if search_matches:
        return None, search_matches

    return None, []


def select_session(cwd: str, session_dir: str | Path | None = None, query: str | None = None) -> Path | None:
    scope, scoped_query = _resolve_scope(query)
    infos = SessionManager.list_all_session_infos(session_dir) if scope == "all" else SessionManager.list_session_infos(cwd, session_dir)
    if not infos:
        return None

    if scoped_query:
        resolved_path, matches = _resolve_session_path_for_scope(scope, cwd, scoped_query, session_dir)
        if resolved_path is not None:
            return resolved_path
        if matches:
            infos = matches

    title = "Resume Session (All)" if scope == "all" else "Resume Session (Current Folder)"
    print(title)
    _print_session_list(infos)
    print("Enter a number, session query, or blank to cancel. Use all:<query> or current:<query> to switch scope.")

    while True:
        choice = input("> ").strip()
        if not choice:
            return None
        if choice.isdigit():
            index = int(choice)
            if 1 <= index <= len(infos):
                return Path(infos[index - 1].path)
            print("Invalid selection.")
            continue
        next_scope, next_query = _resolve_scope(choice)
        resolved_path, matches = _resolve_session_path_for_scope(next_scope, cwd, next_query or "", session_dir)
        if resolved_path is not None:
            return resolved_path
        if matches:
            infos = matches
            print("Select a session:")
            _print_session_list(infos)
            continue
        print(f"No session found for query: {choice}")
