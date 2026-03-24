from __future__ import annotations

import json
import uuid
from typing import Any

from ....agent.agent import Agent
from ....agent.types import AgentTool, AgentToolResult
from ....ai.types import TextContent


def create_spawn_agent_tool(cwd: str, *, parent_agent: Agent | None = None) -> AgentTool:
    """Spawn a sub-agent to handle a task in an independent session."""

    active_sessions: dict[str, Agent] = {}

    async def execute(tool_call_id: str, args: Any, cancel_event=None, on_update=None) -> AgentToolResult:
        label = args.get("label", "")
        message = args.get("message", "")
        session_id = args.get("session_id")

        if session_id is not None:
            # Resume an existing session
            agent = active_sessions.get(session_id)
            if agent is None:
                return AgentToolResult(
                    content=[TextContent(text=json.dumps({
                        "error": f"No active session found for session_id: {session_id}",
                    }))],
                    details=None,
                )
        else:
            # Create a new session
            session_id = str(uuid.uuid4())

            if parent_agent is None:
                return AgentToolResult(
                    content=[TextContent(text=json.dumps({
                        "error": "Cannot spawn sub-agent: no parent agent provided.",
                    }))],
                    details=None,
                )

            parent_state = parent_agent.state

            # Filter out spawn_agent tool to prevent infinite recursion
            child_tools = [
                tool for tool in parent_state.tools
                if tool.name != "spawn_agent"
            ]

            agent = Agent({
                "initial_state": {
                    "model": parent_state.model,
                    "system_prompt": parent_state.system_prompt,
                    "thinking_level": parent_state.thinking_level,
                    "tools": child_tools,
                },
                "convert_to_llm": parent_agent.convert_to_llm,
                "get_api_key": parent_agent.get_api_key,
                "stream_fn": parent_agent.stream_fn,
            })

            active_sessions[session_id] = agent

        # Collect output from text_delta events
        output_parts: list[str] = []

        def listener(event: Any) -> None:
            if getattr(event, "type", None) == "message_update":
                assistant_event = getattr(event, "assistant_message_event", None)
                if getattr(assistant_event, "type", None) == "text_delta":
                    output_parts.append(assistant_event.delta)

        unsubscribe = agent.subscribe(listener)
        try:
            await agent.prompt(message)
        finally:
            unsubscribe()

        output = "".join(output_parts)

        return AgentToolResult(
            content=[TextContent(text=json.dumps({
                "session_id": session_id,
                "output": output,
            }))],
            details=None,
        )

    return AgentTool(
        name="spawn_agent",
        label="Spawn Agent",
        description=(
            "Spawn a sub-agent to work on a task. The sub-agent has access to "
            "the same tools (except spawn_agent) and can maintain conversation "
            "state across calls via session_id. Omit session_id to start a new "
            "session; provide it to continue an existing one."
        ),
        parameters={
            "type": "object",
            "properties": {
                "label": {
                    "type": "string",
                    "description": "Short label displayed while the agent runs.",
                },
                "message": {
                    "type": "string",
                    "description": "The prompt for the sub-agent.",
                },
                "session_id": {
                    "type": "string",
                    "description": "Session ID to continue an existing agent session.",
                },
            },
            "required": ["label", "message"],
        },
        execute=execute,
    )
