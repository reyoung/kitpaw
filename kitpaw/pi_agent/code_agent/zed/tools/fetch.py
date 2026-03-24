from __future__ import annotations

from urllib.parse import urlparse

import httpx

from ....agent.types import AgentTool, AgentToolResult
from ....ai.types import TextContent
from ._http_client import get_http_client

_MAX_SIZE = 32 * 1024  # 32KB
_ALLOWED_SCHEMES = {"http", "https"}


def create_fetch_tool(cwd: str) -> AgentTool:
    """Fetch the content of a URL."""

    async def execute(tool_call_id, args, cancel_event=None, on_update=None):
        url = args.get("url", "")

        if not url:
            return AgentToolResult(
                content=[TextContent(text="Error: No URL provided.")],
                details=None,
            )

        parsed = urlparse(url)
        if parsed.scheme not in _ALLOWED_SCHEMES:
            return AgentToolResult(
                content=[TextContent(text=f"Error: Only http and https URLs are allowed, got '{parsed.scheme}'.")],
                details=None,
            )

        try:
            client = get_http_client()
            response = await client.get(url)
            response.raise_for_status()

            content_type = response.headers.get("content-type", "")
            data = response.content
            text = data.decode("utf-8", errors="replace")
            if len(text) > _MAX_SIZE:
                text = text[:_MAX_SIZE] + "\n\n... (content truncated at 32KB)"

            return AgentToolResult(
                content=[TextContent(text=f"URL: {url}\nContent-Type: {content_type}\n\n{text}")],
                details=None,
            )
        except httpx.HTTPStatusError as e:
            return AgentToolResult(
                content=[TextContent(text=f"HTTP Error {e.response.status_code} for URL: {url}")],
                details=None,
            )
        except httpx.RequestError as e:
            return AgentToolResult(
                content=[TextContent(text=f"Request Error: {e} for URL: {url}")],
                details=None,
            )
        except Exception as e:
            return AgentToolResult(
                content=[TextContent(text=f"Error fetching URL: {e}")],
                details=None,
            )

    return AgentTool(
        name="fetch",
        label="Fetch",
        description=(
            "Fetch the content of a URL and return it as text. "
            "Only http and https URLs are supported. "
            "Content is truncated at 32KB. "
            "Useful for reading web pages, API responses, or other online resources."
        ),
        parameters={
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to fetch content from.",
                },
            },
            "required": ["url"],
        },
        execute=execute,
    )
