from __future__ import annotations

from dataclasses import dataclass

DEFAULT_MAX_LINES = 2000
DEFAULT_MAX_BYTES = 50 * 1024
GREP_MAX_LINE_LENGTH = 500


@dataclass(slots=True)
class TruncationResult:
    content: str
    truncated: bool
    truncated_by: str | None
    total_lines: int
    total_bytes: int
    output_lines: int
    output_bytes: int
    last_line_partial: bool
    first_line_exceeds_limit: bool
    max_lines: int
    max_bytes: int

    def to_dict(self) -> dict[str, object]:
        return {
            "content": self.content,
            "truncated": self.truncated,
            "truncated_by": self.truncated_by,
            "total_lines": self.total_lines,
            "total_bytes": self.total_bytes,
            "output_lines": self.output_lines,
            "output_bytes": self.output_bytes,
            "last_line_partial": self.last_line_partial,
            "first_line_exceeds_limit": self.first_line_exceeds_limit,
            "max_lines": self.max_lines,
            "max_bytes": self.max_bytes,
        }


def format_size(bytes_count: int) -> str:
    if bytes_count < 1024:
        return f"{bytes_count}B"
    if bytes_count < 1024 * 1024:
        return f"{bytes_count / 1024:.1f}KB"
    return f"{bytes_count / (1024 * 1024):.1f}MB"


def truncate_head(content: str, max_lines: int = DEFAULT_MAX_LINES, max_bytes: int = DEFAULT_MAX_BYTES) -> TruncationResult:
    total_bytes = len(content.encode("utf-8"))
    lines = content.split("\n")
    total_lines = len(lines)
    if total_lines <= max_lines and total_bytes <= max_bytes:
        return TruncationResult(
            content=content,
            truncated=False,
            truncated_by=None,
            total_lines=total_lines,
            total_bytes=total_bytes,
            output_lines=total_lines,
            output_bytes=total_bytes,
            last_line_partial=False,
            first_line_exceeds_limit=False,
            max_lines=max_lines,
            max_bytes=max_bytes,
        )

    if len(lines[0].encode("utf-8")) > max_bytes:
        return TruncationResult(
            content="",
            truncated=True,
            truncated_by="bytes",
            total_lines=total_lines,
            total_bytes=total_bytes,
            output_lines=0,
            output_bytes=0,
            last_line_partial=False,
            first_line_exceeds_limit=True,
            max_lines=max_lines,
            max_bytes=max_bytes,
        )

    out_lines: list[str] = []
    used = 0
    truncated_by = "lines"
    for index, line in enumerate(lines[:max_lines]):
        extra = len(line.encode("utf-8")) + (1 if index > 0 else 0)
        if used + extra > max_bytes:
            truncated_by = "bytes"
            break
        out_lines.append(line)
        used += extra

    output = "\n".join(out_lines)
    return TruncationResult(
        content=output,
        truncated=True,
        truncated_by=truncated_by,
        total_lines=total_lines,
        total_bytes=total_bytes,
        output_lines=len(out_lines),
        output_bytes=len(output.encode("utf-8")),
        last_line_partial=False,
        first_line_exceeds_limit=False,
        max_lines=max_lines,
        max_bytes=max_bytes,
    )


def _truncate_string_to_bytes_from_end(text: str, max_bytes: int) -> str:
    data = text.encode("utf-8")
    if len(data) <= max_bytes:
        return text
    chunk = data[-max_bytes:]
    while chunk and (chunk[0] & 0xC0) == 0x80:
        chunk = chunk[1:]
    return chunk.decode("utf-8", errors="ignore")


def truncate_tail(content: str, max_lines: int = DEFAULT_MAX_LINES, max_bytes: int = DEFAULT_MAX_BYTES) -> TruncationResult:
    total_bytes = len(content.encode("utf-8"))
    lines = content.split("\n")
    total_lines = len(lines)
    if total_lines <= max_lines and total_bytes <= max_bytes:
        return TruncationResult(
            content=content,
            truncated=False,
            truncated_by=None,
            total_lines=total_lines,
            total_bytes=total_bytes,
            output_lines=total_lines,
            output_bytes=total_bytes,
            last_line_partial=False,
            first_line_exceeds_limit=False,
            max_lines=max_lines,
            max_bytes=max_bytes,
        )

    out_lines: list[str] = []
    used = 0
    truncated_by = "lines"
    partial = False
    for index in range(len(lines) - 1, -1, -1):
        if len(out_lines) >= max_lines:
            break
        line = lines[index]
        extra = len(line.encode("utf-8")) + (1 if out_lines else 0)
        if used + extra > max_bytes:
            truncated_by = "bytes"
            if not out_lines:
                out_lines.insert(0, _truncate_string_to_bytes_from_end(line, max_bytes))
                used = len(out_lines[0].encode("utf-8"))
                partial = True
            break
        out_lines.insert(0, line)
        used += extra

    output = "\n".join(out_lines)
    return TruncationResult(
        content=output,
        truncated=True,
        truncated_by=truncated_by,
        total_lines=total_lines,
        total_bytes=total_bytes,
        output_lines=len(out_lines),
        output_bytes=len(output.encode("utf-8")),
        last_line_partial=partial,
        first_line_exceeds_limit=False,
        max_lines=max_lines,
        max_bytes=max_bytes,
    )


def truncate_line(line: str, max_chars: int = GREP_MAX_LINE_LENGTH) -> tuple[str, bool]:
    if len(line) <= max_chars:
        return line, False
    return f"{line[:max_chars]}... [truncated]", True
