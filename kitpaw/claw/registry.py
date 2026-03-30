from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from ..pi_agent.code_agent.agent_session import AgentSession


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(slots=True)
class SubagentHandle:
    controller_session_id: str
    session_id: str
    label: str | None
    session: AgentSession
    mode: str
    cleanup: str
    created_at: str = field(default_factory=now_iso)
    updated_at: str = field(default_factory=now_iso)
    status: str = "idle"
    last_task: str | None = None
    last_output: str | None = None
    stop_reason: str | None = None
    error_message: str | None = None
    current_task: asyncio.Task[dict[str, Any]] | None = None

    def to_summary(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "label": self.label,
            "mode": self.mode,
            "cleanup": self.cleanup,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_task": self.last_task,
            "last_output": self.last_output,
            "stop_reason": self.stop_reason,
            "error_message": self.error_message,
            "session_file": self.session.session_file,
        }


class SubagentRegistry:
    def __init__(self) -> None:
        self._by_session_id: dict[str, SubagentHandle] = {}
        self._controller_children: dict[str, set[str]] = defaultdict(set)

    def register(self, handle: SubagentHandle) -> None:
        self._by_session_id[handle.session_id] = handle
        self._controller_children[handle.controller_session_id].add(handle.session_id)

    def list_for_controller(self, controller_session_id: str) -> list[SubagentHandle]:
        session_ids = self._controller_children.get(controller_session_id, set())
        handles = [self._by_session_id[session_id] for session_id in session_ids if session_id in self._by_session_id]
        handles.sort(key=lambda item: item.updated_at, reverse=True)
        return handles

    def resolve(self, controller_session_id: str, target: str) -> SubagentHandle | None:
        normalized = target.strip()
        if not normalized:
            return None
        for handle in self.list_for_controller(controller_session_id):
            if handle.session_id == normalized or (handle.label and handle.label == normalized):
                return handle
        return None

    def remove(self, session_id: str) -> SubagentHandle | None:
        handle = self._by_session_id.pop(session_id, None)
        if handle is None:
            return None
        children = self._controller_children.get(handle.controller_session_id)
        if children is not None:
            children.discard(session_id)
            if not children:
                self._controller_children.pop(handle.controller_session_id, None)
        return handle

    def clear(self) -> None:
        self._by_session_id.clear()
        self._controller_children.clear()


_REGISTRY = SubagentRegistry()


def get_subagent_registry() -> SubagentRegistry:
    return _REGISTRY


@dataclass(slots=True)
class ProcessHandle:
    owner_session_id: str
    session_id: str
    command: str
    cwd: str
    name: str
    process: asyncio.subprocess.Process | None = None
    pid: int | None = None
    timeout_seconds: float | None = None
    created_at: str = field(default_factory=now_iso)
    updated_at: str = field(default_factory=now_iso)
    status: str = "running"
    exit_code: int | None = None
    stop_reason: str | None = None
    error_message: str | None = None
    output: str = ""
    ended_at: str | None = None
    reader_task: asyncio.Task[None] | None = None
    wait_task: asyncio.Task[None] | None = None
    termination_reason: str | None = None
    activity_event: asyncio.Event = field(default_factory=asyncio.Event)

    def mark_activity(self) -> None:
        self.updated_at = now_iso()
        self.activity_event.set()

    def append_output(self, text: str) -> None:
        if not text:
            return
        self.output += text
        self.mark_activity()

    def mark_finished(
        self,
        *,
        status: str,
        exit_code: int | None,
        stop_reason: str | None,
        error_message: str | None,
    ) -> None:
        self.status = status
        self.exit_code = exit_code
        self.stop_reason = stop_reason
        self.error_message = error_message
        self.ended_at = now_iso()
        self.mark_activity()

    async def wait_for_activity(self, timeout_ms: int) -> None:
        if timeout_ms <= 0 or self.status != "running":
            return
        try:
            await asyncio.wait_for(self.activity_event.wait(), timeout=timeout_ms / 1000)
        except asyncio.TimeoutError:
            return
        finally:
            self.activity_event.clear()

    def to_summary(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "command": self.command,
            "name": self.name,
            "cwd": self.cwd,
            "pid": self.pid,
            "status": self.status,
            "exit_code": self.exit_code,
            "stop_reason": self.stop_reason,
            "error_message": self.error_message,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "ended_at": self.ended_at,
        }


class ProcessRegistry:
    def __init__(self) -> None:
        self._by_session_id: dict[str, ProcessHandle] = {}
        self._owner_children: dict[str, set[str]] = defaultdict(set)

    def register(self, handle: ProcessHandle) -> None:
        self._by_session_id[handle.session_id] = handle
        self._owner_children[handle.owner_session_id].add(handle.session_id)

    def list_for_owner(self, owner_session_id: str) -> list[ProcessHandle]:
        session_ids = self._owner_children.get(owner_session_id, set())
        handles = [self._by_session_id[session_id] for session_id in session_ids if session_id in self._by_session_id]
        handles.sort(key=lambda item: item.updated_at, reverse=True)
        return handles

    def resolve(self, owner_session_id: str, target: str) -> ProcessHandle | None:
        normalized = target.strip()
        if not normalized:
            return None
        for handle in self.list_for_owner(owner_session_id):
            if handle.session_id == normalized:
                return handle
        return None

    def remove(self, session_id: str) -> ProcessHandle | None:
        handle = self._by_session_id.pop(session_id, None)
        if handle is None:
            return None
        children = self._owner_children.get(handle.owner_session_id)
        if children is not None:
            children.discard(session_id)
            if not children:
                self._owner_children.pop(handle.owner_session_id, None)
        return handle

    def clear(self) -> None:
        self._by_session_id.clear()
        self._owner_children.clear()


_PROCESS_REGISTRY = ProcessRegistry()


def get_process_registry() -> ProcessRegistry:
    return _PROCESS_REGISTRY
