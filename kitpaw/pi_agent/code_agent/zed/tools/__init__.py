from __future__ import annotations

from typing import Any

from ....agent.types import AgentTool

from .copy_path import create_copy_path_tool
from .create_directory import create_create_directory_tool
from .delete_path import create_delete_path_tool
from .diagnostics import create_diagnostics_tool
from .edit_file import create_edit_file_tool
from .fetch import create_fetch_tool
from .find_path import create_find_path_tool
from .grep import create_grep_tool
from .list_directory import create_list_directory_tool
from .move_path import create_move_path_tool
from .now import create_now_tool
from .open_tool import create_open_tool
from .read_file import create_read_file_tool
from .restore_file_from_disk import create_restore_file_from_disk_tool
from .save_file import create_save_file_tool
from .spawn_agent import create_spawn_agent_tool
from .terminal import create_terminal_tool
from .update_plan import create_update_plan_tool
from .web_search import create_web_search_tool

# Tools that require only ``cwd``.
_SIMPLE_TOOL_FACTORIES: dict[str, Any] = {
    "read_file": create_read_file_tool,
    "edit_file": create_edit_file_tool,
    "terminal": create_terminal_tool,
    "grep": create_grep_tool,
    "find_path": create_find_path_tool,
    "list_directory": create_list_directory_tool,
    "copy_path": create_copy_path_tool,
    "move_path": create_move_path_tool,
    "delete_path": create_delete_path_tool,
    "create_directory": create_create_directory_tool,
    "save_file": create_save_file_tool,
    "restore_file_from_disk": create_restore_file_from_disk_tool,
    "fetch": create_fetch_tool,
    "web_search": create_web_search_tool,
    "open": create_open_tool,
    "now": create_now_tool,
    "update_plan": create_update_plan_tool,
    "diagnostics": create_diagnostics_tool,
}

# All tool names (simple + special).
ALL_TOOL_NAMES: set[str] = {*_SIMPLE_TOOL_FACTORIES, "spawn_agent"}

# Tools with a real CLI implementation.  Stubs (save_file,
# restore_file_from_disk) are still excluded because they have no
# meaningful behaviour outside an editor buffer model.
_CLI_SUPPORTED_TOOLS: set[str] = {
    "read_file",
    "edit_file",
    "terminal",
    "grep",
    "find_path",
    "list_directory",
    "copy_path",
    "move_path",
    "delete_path",
    "create_directory",
    "fetch",
    "open",
    "now",
    "update_plan",
    "web_search",
    "diagnostics",
    "spawn_agent",
}


def create_zed_tools(
    cwd: str,
    *,
    parent_agent: Any = None,
    enabled: set[str] | None = None,
) -> list[AgentTool]:
    """Create Zed-compatible tools for the given working directory.

    *parent_agent* is passed to ``spawn_agent`` so that sub-agents can
    inherit the model, system prompt, and tools of the parent.

    *enabled* selects which tools to include.  When ``None`` (the default),
    all CLI-supported tools are returned.
    """
    names = enabled if enabled is not None else _CLI_SUPPORTED_TOOLS
    tools: list[AgentTool] = []
    for name, factory in _SIMPLE_TOOL_FACTORIES.items():
        if name in names:
            tools.append(factory(cwd))
    if "spawn_agent" in names:
        tools.append(create_spawn_agent_tool(cwd, parent_agent=parent_agent))
    return tools


__all__ = [
    "create_zed_tools",
    "ALL_TOOL_NAMES",
    "create_copy_path_tool",
    "create_create_directory_tool",
    "create_delete_path_tool",
    "create_diagnostics_tool",
    "create_edit_file_tool",
    "create_fetch_tool",
    "create_find_path_tool",
    "create_grep_tool",
    "create_list_directory_tool",
    "create_move_path_tool",
    "create_now_tool",
    "create_open_tool",
    "create_read_file_tool",
    "create_restore_file_from_disk_tool",
    "create_save_file_tool",
    "create_spawn_agent_tool",
    "create_terminal_tool",
    "create_update_plan_tool",
    "create_web_search_tool",
]
