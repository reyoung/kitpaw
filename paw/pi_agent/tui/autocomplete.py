from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Protocol


@dataclass(slots=True)
class AutocompleteItem:
    value: str
    label: str | None = None
    description: str | None = None


@dataclass(slots=True)
class SlashCommand:
    value: str
    label: str
    description: str | None = None
    get_argument_completions: Callable[[str], list[AutocompleteItem] | None] | None = None


class AutocompleteProvider(Protocol):
    def get_suggestions(self, lines: list[str], cursor_line: int, cursor_col: int): ...
    def apply_completion(self, lines: list[str], cursor_line: int, cursor_col: int, item: AutocompleteItem, prefix: str): ...
    def get_force_file_suggestions(self, lines: list[str], cursor_line: int, cursor_col: int): ...


_PATH_DELIMITERS = {" ", "\t", '"', "'", "="}


def _find_last_delimiter(text: str) -> int:
    for index in range(len(text) - 1, -1, -1):
        if text[index] in _PATH_DELIMITERS:
            return index
    return -1


def _find_unclosed_quote_start(text: str) -> int | None:
    in_quotes = False
    quote_start = -1
    for index, char in enumerate(text):
        if char == '"':
            in_quotes = not in_quotes
            if in_quotes:
                quote_start = index
    return quote_start if in_quotes else None


def _is_token_start(text: str, index: int) -> bool:
    return index == 0 or text[index - 1] in _PATH_DELIMITERS


def _extract_quoted_prefix(text: str) -> str | None:
    quote_start = _find_unclosed_quote_start(text)
    if quote_start is None:
        return None
    if quote_start > 0 and text[quote_start - 1] == "@":
        if not _is_token_start(text, quote_start - 1):
            return None
        return text[quote_start - 1 :]
    if not _is_token_start(text, quote_start):
        return None
    return text[quote_start:]


def _parse_path_prefix(prefix: str) -> tuple[str, bool, bool]:
    if prefix.startswith('@"'):
        return prefix[2:], True, True
    if prefix.startswith('"'):
        return prefix[1:], True, False
    if prefix.startswith("@"):
        return prefix[1:], False, True
    return prefix, False, False


def _to_display_path(value: str) -> str:
    return value.replace("\\", "/")


def _build_completion_value(path: str, *, is_directory: bool, is_at_prefix: bool, is_quoted_prefix: bool) -> str:
    needs_quotes = is_quoted_prefix or " " in path
    prefix = "@" if is_at_prefix else ""
    if not needs_quotes:
        return f"{prefix}{path}"
    return f'{prefix}"{path}"'


class CombinedAutocompleteProvider:
    def __init__(
        self,
        providers: list[AutocompleteProvider | SlashCommand | AutocompleteItem] | None = None,
        base_path: str | None = None,
    ) -> None:
        self.providers = providers or []
        self.base_path = base_path or os.getcwd()

    def add_provider(self, provider: AutocompleteProvider | SlashCommand | AutocompleteItem) -> None:
        self.providers.append(provider)

    def get_suggestions(self, lines: list[str], cursor_line: int, cursor_col: int):
        current_line = lines[cursor_line] if cursor_line < len(lines) else ""
        text_before_cursor = current_line[:cursor_col]

        at_prefix = self._extract_at_prefix(text_before_cursor)
        if at_prefix is not None:
            suggestions = self._get_at_file_suggestions(at_prefix)
            if suggestions:
                return {"items": suggestions, "prefix": at_prefix}

        if text_before_cursor.startswith("/"):
            space_index = text_before_cursor.find(" ")
            commands = [provider for provider in self.providers if isinstance(provider, SlashCommand)]
            if space_index == -1:
                prefix = text_before_cursor[1:]
                items = [
                    AutocompleteItem(value=command.value, label=command.label, description=command.description)
                    for command in commands
                    if command.value.startswith(prefix)
                ]
                return {"items": items, "prefix": text_before_cursor} if items else None

            command_name = text_before_cursor[1:space_index]
            argument_prefix = text_before_cursor[space_index + 1 :]
            for command in commands:
                if command.value == command_name and command.get_argument_completions is not None:
                    items = command.get_argument_completions(argument_prefix) or []
                    return {"items": items, "prefix": argument_prefix} if items else None
            return None

        for provider in self.providers:
            if isinstance(provider, (SlashCommand, AutocompleteItem)):
                continue
            result = provider.get_suggestions(lines, cursor_line, cursor_col)
            if result:
                return result
        path_prefix = self._extract_path_prefix(text_before_cursor, force_extract=False)
        if path_prefix is not None:
            suggestions = self._get_file_suggestions(path_prefix)
            if suggestions:
                return {"items": suggestions, "prefix": path_prefix}
        return None

    def get_force_file_suggestions(self, lines: list[str], cursor_line: int, cursor_col: int):
        current_line = lines[cursor_line] if cursor_line < len(lines) else ""
        text_before_cursor = current_line[:cursor_col]

        for provider in self.providers:
            if isinstance(provider, (SlashCommand, AutocompleteItem)):
                continue
            getter = getattr(provider, "get_force_file_suggestions", None)
            if getter is None:
                continue
            result = getter(lines, cursor_line, cursor_col)
            if result:
                return result
        if text_before_cursor.startswith("/") and " " not in text_before_cursor and "/" not in text_before_cursor[1:]:
            return None
        path_prefix = self._extract_path_prefix(text_before_cursor, force_extract=True)
        if path_prefix is not None:
            suggestions = self._get_file_suggestions(path_prefix)
            if suggestions:
                return {"items": suggestions, "prefix": path_prefix}
        return self.get_suggestions(lines, cursor_line, cursor_col)

    def apply_completion(
        self,
        lines: list[str],
        cursor_line: int,
        cursor_col: int,
        item: AutocompleteItem,
        prefix: str,
    ):
        current_line = lines[cursor_line] if cursor_line < len(lines) else ""
        before_prefix = current_line[: cursor_col - len(prefix)]
        after_cursor = current_line[cursor_col:]
        is_quoted_prefix = prefix.startswith('"') or prefix.startswith('@"')
        has_leading_quote_after_cursor = after_cursor.startswith('"')
        has_trailing_quote_in_item = item.value.endswith('"')
        adjusted_after_cursor = after_cursor[1:] if is_quoted_prefix and has_leading_quote_after_cursor and has_trailing_quote_in_item else after_cursor

        is_slash_command = prefix.startswith("/") and before_prefix.strip() == "" and "/" not in prefix[1:]
        if is_slash_command:
            new_line = f"{before_prefix}/{item.value} {adjusted_after_cursor}"
            new_lines = list(lines)
            new_lines[cursor_line] = new_line
            return {"lines": new_lines, "cursor_line": cursor_line, "cursor_col": len(before_prefix) + len(item.value) + 2}

        if prefix.startswith("@"):
            is_directory = (item.label or item.value).endswith("/")
            suffix = "" if is_directory else " "
            new_line = before_prefix + item.value + suffix + adjusted_after_cursor
            new_lines = list(lines)
            new_lines[cursor_line] = new_line
            cursor_offset = len(item.value) - 1 if is_directory and item.value.endswith('"') else len(item.value)
            return {"lines": new_lines, "cursor_line": cursor_line, "cursor_col": len(before_prefix) + cursor_offset + len(suffix)}

        new_line = before_prefix + item.value + adjusted_after_cursor
        new_lines = list(lines)
        new_lines[cursor_line] = new_line
        return {"lines": new_lines, "cursor_line": cursor_line, "cursor_col": len(before_prefix) + len(item.value)}

    def _extract_at_prefix(self, text: str) -> str | None:
        quoted_prefix = _extract_quoted_prefix(text)
        if quoted_prefix is not None and quoted_prefix.startswith('@"'):
            return quoted_prefix
        last_delimiter_index = _find_last_delimiter(text)
        token_start = 0 if last_delimiter_index == -1 else last_delimiter_index + 1
        if token_start < len(text) and text[token_start] == "@":
            return text[token_start:]
        return None

    def _extract_path_prefix(self, text: str, force_extract: bool = False) -> str | None:
        quoted_prefix = _extract_quoted_prefix(text)
        if quoted_prefix is not None:
            return quoted_prefix
        last_delimiter_index = _find_last_delimiter(text)
        path_prefix = text if last_delimiter_index == -1 else text[last_delimiter_index + 1 :]
        if force_extract:
            return path_prefix
        if "/" in path_prefix or path_prefix.startswith(".") or path_prefix.startswith("~/"):
            return path_prefix
        if path_prefix == "" and text.endswith(" "):
            return path_prefix
        return None

    def _get_file_suggestions(self, prefix: str) -> list[AutocompleteItem]:
        raw_prefix, is_quoted_prefix, is_at_prefix = _parse_path_prefix(prefix)
        normalized = raw_prefix.replace("\\", "/")
        expanded = os.path.expanduser(normalized)

        if normalized.endswith("/"):
            display_base = normalized
            directory = expanded if os.path.isabs(expanded) else os.path.join(self.base_path, expanded)
            partial = ""
        else:
            display_base = normalized[: normalized.rfind("/") + 1] if "/" in normalized else ""
            partial = normalized.split("/")[-1] if normalized else ""
            if normalized.startswith("/"):
                directory = os.path.dirname(expanded) or "/"
            else:
                directory_part = expanded[: expanded.rfind(os.sep) + 1] if os.sep in expanded else ""
                if directory_part:
                    directory = os.path.join(self.base_path, directory_part)
                else:
                    directory = self.base_path

        try:
            entries = list(os.scandir(directory))
        except OSError:
            return []

        items: list[AutocompleteItem] = []
        for entry in sorted(entries, key=lambda item: item.name):
            if partial and not entry.name.startswith(partial):
                continue
            suffix = "/" if entry.is_dir() else ""
            display_path = f"{display_base}{entry.name}{suffix}"
            value = _build_completion_value(
                display_path,
                is_directory=entry.is_dir(),
                is_at_prefix=is_at_prefix,
                is_quoted_prefix=is_quoted_prefix,
            )
            items.append(AutocompleteItem(value=value, label=value))
        return items

    def _iter_project_entries(self, root: str):
        for current_root, dirnames, filenames in os.walk(root):
            dirnames[:] = [dirname for dirname in dirnames if dirname != ".git"]
            rel_root = os.path.relpath(current_root, root)
            rel_root = "" if rel_root == "." else _to_display_path(rel_root)

            for dirname in sorted(dirnames):
                rel_path = f"{rel_root}/{dirname}" if rel_root else dirname
                yield rel_path + "/", True
            for filename in sorted(filenames):
                rel_path = f"{rel_root}/{filename}" if rel_root else filename
                if rel_path == ".git" or rel_path.startswith(".git/"):
                    continue
                yield rel_path, False

    def _get_at_file_suggestions(self, prefix: str) -> list[AutocompleteItem]:
        raw_prefix, is_quoted_prefix, _is_at_prefix = _parse_path_prefix(prefix)
        query = _to_display_path(raw_prefix)

        items: list[tuple[str, bool]] = []
        if "/" in query:
            base_part = query[: query.rfind("/") + 1]
            tail = query[query.rfind("/") + 1 :]
            scoped_root = os.path.join(self.base_path, os.path.expanduser(base_part))
            if os.path.isdir(scoped_root):
                entries = list(self._iter_project_entries(scoped_root))
                for rel_path, is_dir in entries:
                    if tail.lower() in rel_path.lower():
                        display = _to_display_path(base_part) + rel_path
                        items.append((display, is_dir))
            else:
                lowered = query.lower()
                for rel_path, is_dir in self._iter_project_entries(self.base_path):
                    if lowered in rel_path.lower():
                        items.append((rel_path, is_dir))
        else:
            lowered = query.lower()
            for rel_path, is_dir in self._iter_project_entries(self.base_path):
                haystack = rel_path.lower()
                basename = os.path.basename(rel_path.rstrip("/")).lower()
                if not lowered or lowered in haystack or lowered in basename:
                    items.append((rel_path, is_dir))

        items.sort(key=lambda item: (not item[1], item[0]))
        return [
            AutocompleteItem(
                value=_build_completion_value(path, is_directory=is_dir, is_at_prefix=True, is_quoted_prefix=is_quoted_prefix),
                label=_build_completion_value(path, is_directory=is_dir, is_at_prefix=True, is_quoted_prefix=is_quoted_prefix),
            )
            for path, is_dir in items
        ]
