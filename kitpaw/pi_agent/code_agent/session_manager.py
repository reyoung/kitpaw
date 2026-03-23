from __future__ import annotations

import json
import re
import uuid
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..agent.types import AgentMessage
from .config import encode_cwd_for_session_dir, get_sessions_dir
from .messages import create_branch_summary_message, create_compaction_summary_message
from .types import SessionHeader, SessionInfo


def infer_session_dir(session_file: str | Path) -> str | None:
    path = Path(session_file).resolve()
    parent = path.parent
    if parent.name.startswith("--") and parent.name.endswith("--"):
        return str(parent.parent)
    return None


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _parse_iso_datetime(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return datetime.min.replace(tzinfo=UTC)


def _extract_message_text(message: dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "\n".join(block.get("text", "") for block in content if block.get("type") == "text").strip()
    return ""


def _normalize_whitespace_lower(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _get_session_search_text(info: SessionInfo) -> str:
    return f"{info.id} {info.name or ''} {info.all_messages_text} {info.cwd}"


def _fuzzy_match(pattern: str, text: str) -> tuple[bool, float]:
    pattern = pattern.strip().lower()
    if not pattern:
        return True, 0.0

    text_lower = text.lower()
    pos = 0
    total = 0.0
    for ch in pattern:
        idx = text_lower.find(ch, pos)
        if idx < 0:
            return False, 0.0
        total += idx - pos
        pos = idx + 1
    return True, total


def _parse_search_query(query: str) -> dict[str, Any]:
    trimmed = query.strip()
    if not trimmed:
        return {"mode": "tokens", "tokens": [], "regex": None}

    if trimmed.startswith("re:"):
        pattern = trimmed[3:].strip()
        if not pattern:
            return {"mode": "regex", "tokens": [], "regex": None, "error": "Empty regex"}
        try:
            return {"mode": "regex", "tokens": [], "regex": re.compile(pattern, re.IGNORECASE)}
        except re.error as exc:
            return {"mode": "regex", "tokens": [], "regex": None, "error": str(exc)}

    tokens: list[dict[str, str]] = []
    buf = ""
    in_quote = False
    had_unclosed_quote = False

    def flush(kind: str) -> None:
        nonlocal buf
        value = buf.strip()
        buf = ""
        if value:
            tokens.append({"kind": kind, "value": value})

    for ch in trimmed:
        if ch == '"':
            if in_quote:
                flush("phrase")
                in_quote = False
            else:
                flush("fuzzy")
                in_quote = True
            continue
        if not in_quote and ch.isspace():
            flush("fuzzy")
            continue
        buf += ch

    if in_quote:
        had_unclosed_quote = True

    if had_unclosed_quote:
        return {
            "mode": "tokens",
            "tokens": [{"kind": "fuzzy", "value": token} for token in trimmed.split() if token.strip()],
            "regex": None,
        }

    flush("phrase" if in_quote else "fuzzy")
    return {"mode": "tokens", "tokens": tokens, "regex": None}


def _match_session(info: SessionInfo, parsed: dict[str, Any]) -> tuple[bool, float]:
    text = _get_session_search_text(info)
    if parsed["mode"] == "regex":
        regex = parsed.get("regex")
        if regex is None:
            return False, 0.0
        match = regex.search(text)
        if match is None:
            return False, 0.0
        return True, float(match.start()) * 0.1

    tokens = parsed.get("tokens", [])
    if not tokens:
        return True, 0.0

    total_score = 0.0
    normalized_text: str | None = None
    for token in tokens:
        kind = token["kind"]
        value = token["value"]
        if kind == "phrase":
            if normalized_text is None:
                normalized_text = _normalize_whitespace_lower(text)
            phrase = _normalize_whitespace_lower(value)
            if not phrase:
                continue
            idx = normalized_text.find(phrase)
            if idx < 0:
                return False, 0.0
            total_score += float(idx) * 0.1
            continue

        matches, score = _fuzzy_match(value, text)
        if not matches:
            return False, 0.0
        total_score += score

    return True, total_score


def _match_session_identity(info: SessionInfo, normalized_query: str) -> bool:
    path = Path(info.path)
    basename = path.name.lower()
    stem = path.stem.lower()
    session_id = info.id.lower()
    name = (info.name or "").lower()
    return (
        normalized_query in {basename, stem}
        or session_id.startswith(normalized_query)
        or (name and normalized_query == name)
    )


def _search_session_infos(infos: list[SessionInfo], query: str) -> list[SessionInfo]:
    parsed = _parse_search_query(query)
    if parsed.get("error"):
        return []

    normalized_query = query.strip().lower()
    scored: list[tuple[SessionInfo, float]] = []
    for info in infos:
        if normalized_query and _match_session_identity(info, normalized_query):
            scored.append((info, 0.0))
            continue
        matches, score = _match_session(info, parsed)
        if matches:
            scored.append((info, score))

    scored.sort(key=lambda item: (item[1], -_parse_iso_datetime(item[0].modified).timestamp()))
    return [item[0] for item in scored]


def _resolve_session_infos(infos: list[SessionInfo], query: str) -> list[SessionInfo]:
    normalized_query = query.strip().lower()
    if not normalized_query:
        return []

    matches: list[SessionInfo] = []
    for info in infos:
        path = Path(info.path)
        basename = path.name.lower()
        stem = path.stem.lower()
        session_id = info.id.lower()
        name = (info.name or "").lower()
        first_message = info.first_message.lower()
        all_messages_text = info.all_messages_text.lower()
        if normalized_query in {basename, stem}:
            matches.append(info)
            continue
        if session_id.startswith(normalized_query):
            matches.append(info)
            continue
        if name == normalized_query:
            matches.append(info)
            continue
        if normalized_query in first_message:
            matches.append(info)
            continue
        if normalized_query in all_messages_text:
            matches.append(info)
            continue
    return matches


class SessionManager:
    def __init__(self, cwd: str, session_file: Path | None = None) -> None:
        self.cwd = str(Path(cwd).resolve())
        self.session_id = uuid.uuid4().hex
        self.session_file = session_file
        self.entries: list[dict[str, Any]] = []
        self.leaf_id: str | None = None
        self.session_name: str | None = None
        if self.session_file is not None and self.session_file.exists():
            self._load_existing()

    @classmethod
    def create(cls, cwd: str, session_dir: str | Path | None = None) -> "SessionManager":
        root = Path(session_dir) if session_dir is not None else get_sessions_dir()
        root = root / encode_cwd_for_session_dir(cwd)
        root.mkdir(parents=True, exist_ok=True)
        session_file = root / f"{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.jsonl"
        return cls(cwd, session_file)

    @classmethod
    def in_memory(cls, cwd: str = ".") -> "SessionManager":
        return cls(cwd, None)

    @classmethod
    def open(cls, session_file: str | Path) -> "SessionManager":
        path = Path(session_file).resolve()
        lines = path.read_text(encoding="utf-8").splitlines()
        if not lines:
            raise ValueError(f"Session file is empty: {path}")
        header = json.loads(lines[0])
        return cls(header["cwd"], path)

    @classmethod
    def list_sessions(cls, cwd: str, session_dir: str | Path | None = None) -> list[Path]:
        root = Path(session_dir) if session_dir is not None else get_sessions_dir()
        target = root / encode_cwd_for_session_dir(cwd)
        if not target.exists():
            return []
        return sorted(target.glob("*.jsonl"), key=lambda path: path.stat().st_mtime, reverse=True)

    @classmethod
    def list_session_infos(cls, cwd: str, session_dir: str | Path | None = None) -> list[SessionInfo]:
        infos: list[SessionInfo] = []
        for path in cls.list_sessions(cwd, session_dir):
            info = cls.read_session_info(path)
            if info is not None:
                infos.append(info)
        return infos

    @classmethod
    def list_all_sessions(cls, session_dir: str | Path | None = None) -> list[Path]:
        root = Path(session_dir) if session_dir is not None else get_sessions_dir()
        if not root.exists():
            return []
        paths: list[Path] = []
        for subdir in root.iterdir():
            if not subdir.is_dir():
                continue
            paths.extend(path for path in sorted(subdir.glob("*.jsonl"), key=lambda path: path.stat().st_mtime, reverse=True))
        return sorted(paths, key=lambda path: path.stat().st_mtime, reverse=True)

    @classmethod
    def list_all_session_infos(cls, session_dir: str | Path | None = None) -> list[SessionInfo]:
        infos: list[SessionInfo] = []
        for path in cls.list_all_sessions(session_dir):
            info = cls.read_session_info(path)
            if info is not None:
                infos.append(info)
        return infos

    @classmethod
    def find_most_recent_session(cls, cwd: str, session_dir: str | Path | None = None) -> Path | None:
        sessions = cls.list_sessions(cwd, session_dir)
        return sessions[0] if sessions else None

    @classmethod
    def resolve_session(cls, cwd: str, query: str, session_dir: str | Path | None = None) -> Path:
        candidate = Path(query).expanduser()
        if candidate.exists():
            return candidate.resolve()

        matches = cls.resolve_session_infos(cwd, query, session_dir)

        if not matches:
            raise ValueError(f"No session found for query: {query}")
        if len(matches) > 1:
            raise ValueError(f"Multiple sessions match query: {query}")
        return Path(matches[0].path)

    @classmethod
    def resolve_session_infos(cls, cwd: str, query: str, session_dir: str | Path | None = None) -> list[SessionInfo]:
        return _resolve_session_infos(cls.list_session_infos(cwd, session_dir), query)

    @classmethod
    def search_session_infos(cls, cwd: str, query: str, session_dir: str | Path | None = None) -> list[SessionInfo]:
        return _search_session_infos(cls.list_session_infos(cwd, session_dir), query)

    @classmethod
    def resolve_all_session_infos(cls, query: str, session_dir: str | Path | None = None) -> list[SessionInfo]:
        return _resolve_session_infos(cls.list_all_session_infos(session_dir), query)

    @classmethod
    def search_all_session_infos(cls, query: str, session_dir: str | Path | None = None) -> list[SessionInfo]:
        return _search_session_infos(cls.list_all_session_infos(session_dir), query)

    @classmethod
    def read_session_info(cls, session_file: str | Path) -> SessionInfo | None:
        path = Path(session_file).resolve()
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return None
        if not lines:
            return None
        try:
            header = json.loads(lines[0])
        except json.JSONDecodeError:
            return None
        if header.get("type") != "session":
            return None

        name: str | None = None
        message_count = 0
        first_message = ""
        all_messages: list[str] = []
        last_activity = header.get("timestamp")
        for line in lines[1:]:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            timestamp = entry.get("timestamp")
            if isinstance(timestamp, str):
                last_activity = timestamp
            if entry.get("type") == "session_info":
                value = entry.get("name")
                name = value.strip() if isinstance(value, str) and value.strip() else None
                continue
            if entry.get("type") != "message":
                continue
            message = entry.get("message", {})
            role = message.get("role")
            if role not in {"user", "assistant"}:
                continue
            text = _extract_message_text(message)
            message_count += 1
            if text:
                all_messages.append(text)
                if not first_message and role == "user":
                    first_message = text

        created = header.get("timestamp") if isinstance(header.get("timestamp"), str) else _now_iso()
        modified = last_activity if isinstance(last_activity, str) else created
        return SessionInfo(
            path=str(path),
            id=header.get("id", ""),
            cwd=header.get("cwd", ""),
            name=name,
            parent_session_path=header.get("parentSession"),
            created=created,
            modified=modified,
            message_count=message_count,
            first_message=first_message or "(no messages)",
            all_messages_text=" ".join(all_messages),
        )

    def _load_existing(self) -> None:
        lines = self.session_file.read_text(encoding="utf-8").splitlines()
        for index, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if index == 0 and entry.get("type") == "session":
                self.session_id = entry["id"]
                continue
            self.entries.append(entry)
            if entry.get("type") == "session_info":
                self.session_name = entry.get("name") or None
        if self.entries:
            self.leaf_id = self.entries[-1].get("id")

    def _ensure_header(self) -> None:
        if self.session_file is None or self.session_file.exists():
            return
        header = SessionHeader(id=self.session_id, timestamp=_now_iso(), cwd=self.cwd)
        self.session_file.write_text(json.dumps(asdict(header)) + "\n", encoding="utf-8")

    def append_message(self, message: AgentMessage) -> dict[str, Any]:
        entry = {
            "type": "message",
            "id": uuid.uuid4().hex[:8],
            "parentId": self.leaf_id,
            "timestamp": _now_iso(),
            "message": _message_to_dict(message),
        }
        self.entries.append(entry)
        self.leaf_id = entry["id"]
        if self.session_file is not None:
            self._ensure_header()
            with self.session_file.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry) + "\n")
        return entry

    def append_model_change(self, provider: str, model_id: str) -> None:
        self._append_entry({"provider": provider, "modelId": model_id}, "model_change")

    def append_thinking_level_change(self, thinking_level: str) -> None:
        self._append_entry({"thinkingLevel": thinking_level}, "thinking_level_change")

    def append_compaction(
        self,
        summary: str,
        first_kept_entry_id: str,
        tokens_before: int,
        details: object | None = None,
        from_hook: bool = False,
    ) -> str:
        data: dict[str, object] = {
            "summary": summary,
            "firstKeptEntryId": first_kept_entry_id,
            "tokensBefore": tokens_before,
        }
        if details is not None:
            data["details"] = details
        if from_hook:
            data["fromHook"] = True
        entry = self._append_entry(data, "compaction")
        return entry["id"]

    def branch_with_summary(self, branch_from_id: str | None, summary: str) -> str:
        self.branch(branch_from_id)
        entry = self._append_entry({"fromId": branch_from_id or "root", "summary": summary}, "branch_summary")
        return entry["id"]

    def set_session_name(self, name: str | None) -> None:
        self.session_name = name or None
        self._append_entry({"name": self.session_name}, "session_info")

    def get_session_name(self) -> str | None:
        return self.session_name

    def _append_entry(self, payload: dict[str, Any], type_name: str) -> dict[str, Any]:
        entry = {
            "type": type_name,
            "id": uuid.uuid4().hex[:8],
            "parentId": self.leaf_id,
            "timestamp": _now_iso(),
            **payload,
        }
        self.entries.append(entry)
        self.leaf_id = entry["id"]
        if self.session_file is not None:
            self._ensure_header()
            with self.session_file.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry) + "\n")
        return entry

    def get_session_file(self) -> str | None:
        return None if self.session_file is None else str(self.session_file)

    def get_session_id(self) -> str:
        return self.session_id

    def build_session_context(self) -> list[dict[str, Any]]:
        context = self.build_runtime_context()
        return [message for message in context["messages"] if isinstance(message, dict)]

    def get_stats(self) -> dict[str, Any]:
        messages = [entry for entry in self.entries if entry["type"] == "message"]
        role_counts: dict[str, int] = {}
        for entry in messages:
            role = entry["message"].get("role", "unknown")
            role_counts[role] = role_counts.get(role, 0) + 1
        return {
            "sessionFile": self.get_session_file(),
            "sessionId": self.get_session_id(),
            "sessionName": self.get_session_name(),
            "messageCount": len(messages),
            "userMessages": role_counts.get("user", 0),
            "assistantMessages": role_counts.get("assistant", 0),
            "toolResults": role_counts.get("toolResult", 0),
            "bashMessages": role_counts.get("bashExecution", 0),
            "branchSummaries": sum(1 for entry in self.entries if entry["type"] == "branch_summary"),
            "compactions": sum(1 for entry in self.entries if entry["type"] == "compaction"),
        }

    def get_messages(self) -> list[dict[str, Any]]:
        return [entry["message"] for entry in self.entries if entry["type"] == "message"]

    def get_entry(self, entry_id: str) -> dict[str, Any] | None:
        return next((entry for entry in self.entries if entry["id"] == entry_id), None)

    def get_leaf_id(self) -> str | None:
        return self.leaf_id

    def get_branch_entries(self, leaf_id: str | None = None) -> list[dict[str, Any]]:
        target_leaf = self.leaf_id if leaf_id is None else leaf_id
        if target_leaf is None:
            return []
        by_id = {entry["id"]: entry for entry in self.entries}
        path: list[dict[str, Any]] = []
        current_id = target_leaf
        while current_id is not None:
            entry = by_id.get(current_id)
            if entry is None:
                break
            path.append(entry)
            current_id = entry.get("parentId")
        path.reverse()
        return path

    def get_branch_messages(self, leaf_id: str | None = None) -> list[dict[str, Any]]:
        return [entry["message"] for entry in self.get_branch_entries(leaf_id) if entry["type"] == "message"]

    def build_runtime_context(self, leaf_id: str | None = None) -> dict[str, Any]:
        path = self.get_branch_entries(leaf_id)
        thinking_level: str | None = None
        model: dict[str, str] | None = None
        compaction: dict[str, Any] | None = None

        for entry in path:
            if entry["type"] == "thinking_level_change":
                thinking_level = entry["thinkingLevel"]
            elif entry["type"] == "model_change":
                model = {"provider": entry["provider"], "modelId": entry["modelId"]}
            elif entry["type"] == "message" and entry["message"].get("role") == "assistant":
                model = {
                    "provider": entry["message"].get("provider", "openai"),
                    "modelId": entry["message"].get("model", ""),
                }
            elif entry["type"] == "compaction":
                compaction = entry

        messages: list[Any] = []

        def append_context_message(entry: dict[str, Any]) -> None:
            if entry["type"] == "message":
                messages.append(entry["message"])
            elif entry["type"] == "branch_summary" and entry.get("summary"):
                messages.append(create_branch_summary_message(entry["summary"], entry.get("fromId", "root"), entry["timestamp"]))

        if compaction is not None:
            messages.append(
                create_compaction_summary_message(
                    compaction.get("summary", ""),
                    compaction.get("tokensBefore", 0),
                    compaction["timestamp"],
                )
            )
            compaction_index = next(
                index for index, entry in enumerate(path) if entry["type"] == "compaction" and entry["id"] == compaction["id"]
            )
            found_first_kept = False
            for entry in path[:compaction_index]:
                if entry["id"] == compaction.get("firstKeptEntryId"):
                    found_first_kept = True
                if found_first_kept:
                    append_context_message(entry)
            for entry in path[compaction_index + 1 :]:
                append_context_message(entry)
        else:
            for entry in path:
                append_context_message(entry)

        return {"messages": messages, "thinkingLevel": thinking_level, "model": model}

    def get_tree(self) -> list[dict[str, Any]]:
        nodes: dict[str, dict[str, Any]] = {}
        roots: list[dict[str, Any]] = []
        for entry in self.entries:
            label = entry["type"]
            if entry["type"] == "message":
                label = entry["message"].get("role", "message")
            nodes[entry["id"]] = {
                "entry": {
                    "id": entry["id"],
                    "type": entry["type"],
                    "parentId": entry.get("parentId"),
                    "timestamp": entry["timestamp"],
                    "label": label,
                },
                "children": [],
                "isLeaf": entry["id"] == self.leaf_id,
            }
        for entry in self.entries:
            node = nodes[entry["id"]]
            parent_id = entry.get("parentId")
            if parent_id is None or parent_id not in nodes or parent_id == entry["id"]:
                roots.append(node)
            else:
                nodes[parent_id]["children"].append(node)
        return roots

    def branch(self, entry_id: str | None) -> None:
        if entry_id is not None and self.get_entry(entry_id) is None:
            raise ValueError(f"Unknown entry id: {entry_id}")
        self.leaf_id = entry_id

    def get_user_messages_for_forking(self) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        for entry in self.get_branch_entries():
            if entry["type"] != "message":
                continue
            message = entry["message"]
            if message.get("role") != "user":
                continue
            content = message.get("content")
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                text = "\n".join(block.get("text", "") for block in content if block.get("type") == "text")
            else:
                text = ""
            messages.append({"entryId": entry["id"], "text": text})
        return messages

    def get_last_assistant_text(self) -> str | None:
        for entry in reversed(self.get_branch_entries()):
            if entry["type"] != "message":
                continue
            message = entry["message"]
            if message.get("role") != "assistant":
                continue
            parts: list[str] = []
            for block in message.get("content", []):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
            return "".join(parts).strip() or None
        return None

    def _entries_up_to_parent_of(self, entry_id: str) -> list[dict[str, Any]]:
        selected = next((entry for entry in self.entries if entry.get("id") == entry_id), None)
        if selected is None:
            raise ValueError(f"Unknown entry id: {entry_id}")
        keep_until = selected.get("parentId")
        if keep_until is None:
            return []
        index_by_id = {entry["id"]: index for index, entry in enumerate(self.entries)}
        if keep_until not in index_by_id:
            return []
        return self.entries[: index_by_id[keep_until] + 1]

    def fork_to_new_manager(self, entry_id: str, session_dir: str | Path | None = None) -> tuple["SessionManager", str]:
        selected = next((entry for entry in self.entries if entry.get("id") == entry_id), None)
        if selected is None or selected["type"] != "message":
            raise ValueError(f"Unknown message entry id: {entry_id}")
        new_manager = SessionManager.create(self.cwd, session_dir)
        new_manager.session_name = self.session_name
        retained = self._entries_up_to_parent_of(entry_id)
        if retained:
            new_manager.entries = json.loads(json.dumps(retained))
            new_manager.leaf_id = new_manager.entries[-1]["id"]
            if new_manager.session_file is not None:
                new_manager._ensure_header()
                with new_manager.session_file.open("a", encoding="utf-8") as handle:
                    for entry in new_manager.entries:
                        handle.write(json.dumps(entry) + "\n")
        content = selected["message"].get("content", "")
        if isinstance(content, str):
            return new_manager, content
        text = "\n".join(block.get("text", "") for block in content if block.get("type") == "text")
        return new_manager, text


def _message_to_dict(message: AgentMessage) -> dict[str, Any]:
    return _to_jsonable(message)


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _to_jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {key: _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if hasattr(value, "__slots__"):
        return {slot: _to_jsonable(getattr(value, slot)) for slot in value.__slots__}
    return value
