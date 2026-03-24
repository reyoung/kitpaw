from __future__ import annotations

import httpx

_client: httpx.AsyncClient | None = None


def get_http_client() -> httpx.AsyncClient:
    """Return a shared ``httpx.AsyncClient`` for Zed tools.

    The client is lazily created on first call and reused for all
    subsequent requests, keeping the underlying connection pool alive.
    """
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(
            follow_redirects=True,
            timeout=30.0,
            headers={"User-Agent": "Mozilla/5.0 (compatible; ZedAgent/1.0)"},
        )
    return _client
