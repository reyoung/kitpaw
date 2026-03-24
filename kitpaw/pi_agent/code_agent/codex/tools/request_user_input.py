from __future__ import annotations

import asyncio
import json
import sys
from functools import partial

from ....agent.types import AgentTool, AgentToolResult
from ....ai.types import TextContent


def create_request_user_input_tool(cwd: str) -> AgentTool:
    """Request input from the user. Each question can optionally have predefined options."""

    async def execute(tool_call_id, args, cancel_event=None, on_update=None):
        questions: list[dict] = args.get("questions", [])

        if not questions:
            return AgentToolResult(
                content=[TextContent(text=json.dumps({"error": "No questions provided."}))],
                details=None,
            )

        loop = asyncio.get_running_loop()
        answers: list[dict] = []

        for item in questions:
            question = item.get("question", "")
            options = item.get("options")

            # Print question to stderr so it doesn't mix with stdout piping
            print(f"\n{question}", file=sys.stderr, flush=True)
            if options:
                for i, opt in enumerate(options, 1):
                    print(f"  {i}. {opt}", file=sys.stderr, flush=True)
                print("Enter choice number or text: ", file=sys.stderr, end="", flush=True)
            else:
                print("Answer: ", file=sys.stderr, end="", flush=True)

            # Read from stdin in executor to avoid blocking the event loop
            answer = await loop.run_in_executor(None, partial(sys.stdin.readline))
            answer = answer.strip()

            answers.append({"question": question, "answer": answer})

        return AgentToolResult(
            content=[TextContent(text=json.dumps({"answers": answers}))],
            details=None,
        )

    return AgentTool(
        name="request_user_input",
        label="Request User Input",
        description=(
            "Request input from the user. Each question can optionally have "
            "predefined options."
        ),
        parameters={
            "type": "object",
            "properties": {
                "questions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string",
                                "description": "The question to ask the user.",
                            },
                            "options": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Optional predefined answer options.",
                            },
                        },
                        "required": ["question"],
                    },
                    "description": "The list of questions to ask.",
                },
            },
            "required": ["questions"],
        },
        execute=execute,
    )
