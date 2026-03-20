from __future__ import annotations

from .bash import create_bash_tool
from .edit import create_edit_tool
from .find import create_find_tool
from .grep import create_grep_tool
from .ls import create_ls_tool
from .read import create_read_tool
from .write import create_write_tool


def create_coding_tools(cwd: str, command_prefix: str | None = None):
    return [
        create_read_tool(cwd),
        create_bash_tool(cwd, command_prefix=command_prefix),
        create_edit_tool(cwd),
        create_write_tool(cwd),
    ]


def create_read_only_tools(cwd: str):
    return [
        create_read_tool(cwd),
        create_grep_tool(cwd),
        create_find_tool(cwd),
        create_ls_tool(cwd),
    ]


def create_all_tools(cwd: str, command_prefix: str | None = None):
    return {
        "read": create_read_tool(cwd),
        "bash": create_bash_tool(cwd, command_prefix=command_prefix),
        "edit": create_edit_tool(cwd),
        "write": create_write_tool(cwd),
        "grep": create_grep_tool(cwd),
        "find": create_find_tool(cwd),
        "ls": create_ls_tool(cwd),
    }
