from __future__ import annotations

import html
import json
from pathlib import Path

from .session_manager import SessionManager


def _render_message(entry: dict) -> str:
    message = entry["message"]
    role = html.escape(message.get("role", "unknown"))
    content = message.get("content", "")
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif block.get("type") == "thinking":
                parts.append(block.get("thinking", ""))
            elif block.get("type") == "toolCall":
                parts.append(json.dumps({"tool": block.get("name"), "arguments": block.get("arguments")}, ensure_ascii=False))
            else:
                parts.append(json.dumps(block, ensure_ascii=False))
        rendered = "\n".join(parts)
    else:
        rendered = str(content)
    return f"<section><h3>{role}</h3><pre>{html.escape(rendered)}</pre></section>"


def _render_entry(entry: dict) -> str:
    if entry["type"] == "message":
        return _render_message(entry)
    if entry["type"] == "branch_summary":
        return f"<section><h3>branch_summary</h3><pre>{html.escape(entry.get('summary', ''))}</pre></section>"
    if entry["type"] == "compaction":
        summary = html.escape(entry.get("summary", ""))
        tokens_before = entry.get("tokensBefore", 0)
        return f"<section><h3>compaction</h3><pre>Compacted from {tokens_before} tokens\n\n{summary}</pre></section>"
    if entry["type"] == "session_info":
        return f"<section><h3>session_info</h3><pre>{html.escape(str(entry.get('name') or ''))}</pre></section>"
    return ""


def export_from_file(input_path: str, output_path: str | None = None) -> str:
    session = SessionManager.open(input_path)
    output = Path(output_path) if output_path else Path(input_path).with_suffix(".html")
    body = "\n".join(rendered for rendered in (_render_entry(entry) for entry in session.entries) if rendered)
    title = session.get_session_name() or session.get_session_id()
    html_doc = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{html.escape(title)}</title>"
        "<style>body{font-family:monospace;max-width:900px;margin:2rem auto;padding:0 1rem;}pre{white-space:pre-wrap;background:#f5f5f5;padding:1rem;border-radius:8px;}section{margin-bottom:1rem;}</style>"
        f"</head><body><h1>{html.escape(title)}</h1>{body}</body></html>"
    )
    output.write_text(html_doc, encoding="utf-8")
    return str(output)
