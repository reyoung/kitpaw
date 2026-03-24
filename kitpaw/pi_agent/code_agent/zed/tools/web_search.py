from __future__ import annotations

import re
from html import unescape
from urllib.parse import unquote, urlparse, parse_qs

import httpx

from ....agent.types import AgentTool, AgentToolResult
from ....ai.types import TextContent
from ._http_client import get_http_client

_MAX_RESULTS = 10
_TIMEOUT = 15.0
_SEARCH_URL = "https://html.duckduckgo.com/html/"

# Patterns for parsing DuckDuckGo HTML results
_RESULT_BLOCK_RE = re.compile(
    r'<div\s+class="result\s+results_links[^"]*">(.*?)</div>\s*</div>\s*</div>',
    re.DOTALL,
)
_TITLE_RE = re.compile(
    r'<a\s+rel="nofollow"\s+class="result__a"\s+href="([^"]*)"[^>]*>(.*?)</a>',
    re.DOTALL,
)
_SNIPPET_RE = re.compile(
    r'<a\s+class="result__snippet"[^>]*>(.*?)</a>',
    re.DOTALL,
)


def _strip_tags(html: str) -> str:
    """Remove HTML tags and decode entities."""
    text = re.sub(r"<[^>]+>", "", html)
    return unescape(text).strip()


def _extract_real_url(ddg_href: str) -> str:
    """Extract the actual URL from DuckDuckGo's redirect wrapper.

    DDG wraps links like: //duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com&...
    """
    if "duckduckgo.com/l/" in ddg_href:
        parsed = urlparse(ddg_href)
        qs = parse_qs(parsed.query)
        if "uddg" in qs:
            return unquote(qs["uddg"][0])
    # Already a direct URL or relative path
    if ddg_href.startswith("//"):
        return "https:" + ddg_href
    return ddg_href


def _parse_results(html: str) -> list[dict[str, str]]:
    """Parse search results from DuckDuckGo HTML response."""
    results: list[dict[str, str]] = []

    for block_match in _RESULT_BLOCK_RE.finditer(html):
        block = block_match.group(1)

        title_match = _TITLE_RE.search(block)
        if not title_match:
            continue

        raw_url = title_match.group(1)
        url = _extract_real_url(raw_url)
        title = _strip_tags(title_match.group(2))

        snippet = ""
        snippet_match = _SNIPPET_RE.search(block)
        if snippet_match:
            snippet = _strip_tags(snippet_match.group(1))

        if title and url:
            results.append({"title": title, "url": url, "snippet": snippet})

        if len(results) >= _MAX_RESULTS:
            break

    return results


def create_web_search_tool(cwd: str) -> AgentTool:
    """Search the web for information using DuckDuckGo."""

    async def execute(tool_call_id, args, cancel_event=None, on_update=None):
        query = args.get("query", "").strip()

        if not query:
            return AgentToolResult(
                content=[TextContent(text="Error: No search query provided.")],
                details=None,
            )

        try:
            client = get_http_client()
            response = await client.post(
                _SEARCH_URL,
                data={"q": query, "b": ""},
            )
            response.raise_for_status()

            html = response.text
            results = _parse_results(html)

            if not results:
                return AgentToolResult(
                    content=[TextContent(text=f"No results found for: {query}")],
                    details=None,
                )

            lines: list[str] = []
            for i, r in enumerate(results, 1):
                lines.append(
                    f"{i}. {r['title']}\n"
                    f"   URL: {r['url']}\n"
                    f"   {r['snippet']}\n"
                )

            return AgentToolResult(
                content=[TextContent(text="\n".join(lines))],
                details=None,
            )

        except httpx.HTTPStatusError as e:
            return AgentToolResult(
                content=[TextContent(text=f"Search error: HTTP {e.response.status_code} from DuckDuckGo.")],
                details=None,
            )
        except httpx.RequestError as e:
            return AgentToolResult(
                content=[TextContent(text=f"Search error: Could not reach DuckDuckGo ({e}).")],
                details=None,
            )
        except Exception as e:
            return AgentToolResult(
                content=[TextContent(text=f"Search error: {e}")],
                details=None,
            )

    return AgentTool(
        name="web_search",
        label="Web Search",
        description=(
            "Search the web for information using DuckDuckGo. "
            "Returns up to 10 results with titles, URLs, and snippets. "
            "Useful for finding current information, documentation, or answers to questions."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query.",
                },
            },
            "required": ["query"],
        },
        execute=execute,
    )
