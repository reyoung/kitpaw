from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

YieldCallback = Callable[[str], None | Awaitable[None]]


@dataclass(slots=True)
class OpenClawToolContext:
    cwd: str
    workspace_dir: str
    spawn_workspace_dir: str
    agent_id: str
    session_id: str
    controller_session_id: str
    model_provider: str
    model_id: str
    thinking_level: str
    sandboxed: bool
    prompt_mode: str = "full"
    system_prompt: str | None = None
    system_prompt_is_override: bool = False
    on_yield: YieldCallback | None = None
