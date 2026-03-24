from __future__ import annotations

from ....agent.types import AgentTool
from .apply_patch import create_apply_patch_tool
from .request_user_input import create_request_user_input_tool
from .shell import create_shell_tool
from .update_plan import create_update_plan_tool
from .view_image import create_view_image_tool


def create_codex_tools(cwd: str) -> list[AgentTool]:
    return [
        create_shell_tool(cwd),
        create_apply_patch_tool(cwd),
        create_update_plan_tool(cwd),
        create_view_image_tool(cwd),
        create_request_user_input_tool(cwd),
    ]
