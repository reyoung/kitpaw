from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..pi_agent.agent.types import AgentTool
from ..pi_agent.code_agent.tools.path_utils import PathTraversalError, resolve_to_cwd
from .context import OpenClawToolContext
from .result_utils import error_result, json_result

BEGIN_PATCH_MARKER = "*** Begin Patch"
END_PATCH_MARKER = "*** End Patch"
ADD_FILE_MARKER = "*** Add File: "
DELETE_FILE_MARKER = "*** Delete File: "
UPDATE_FILE_MARKER = "*** Update File: "
MOVE_TO_MARKER = "*** Move to: "
EOF_MARKER = "*** End of File"


@dataclass(slots=True)
class AddFileHunk:
    path: str
    contents: str


@dataclass(slots=True)
class DeleteFileHunk:
    path: str


@dataclass(slots=True)
class UpdateChunk:
    old_lines: list[str]
    new_lines: list[str]
    is_end_of_file: bool


@dataclass(slots=True)
class UpdateFileHunk:
    path: str
    move_to: str | None
    chunks: list[UpdateChunk]


@dataclass(slots=True)
class ApplyPatchResult:
    summary: dict[str, list[str]]
    text: str


def _is_hunk_start(line: str) -> bool:
    return (
        line.startswith(ADD_FILE_MARKER)
        or line.startswith(DELETE_FILE_MARKER)
        or line.startswith(UPDATE_FILE_MARKER)
    )


def _resolve_patch_path(path: str, workspace_dir: str) -> tuple[Path, str]:
    try:
        resolved = Path(resolve_to_cwd(path, workspace_dir))
    except PathTraversalError as exc:
        raise ValueError(str(exc)) from exc
    display = str(resolved.relative_to(Path(workspace_dir).resolve()))
    return resolved, display


def _format_summary(summary: dict[str, list[str]]) -> str:
    lines = ["Success. Updated the following files:"]
    for path in summary["added"]:
        lines.append(f"A {path}")
    for path in summary["modified"]:
        lines.append(f"M {path}")
    for path in summary["deleted"]:
        lines.append(f"D {path}")
    return "\n".join(lines)


def _join_patch_lines(lines: list[str], *, trailing_newline: bool) -> str:
    if not lines:
        return ""
    text = "\n".join(lines)
    if trailing_newline:
        text += "\n"
    return text


def _apply_update_chunks(original: str, chunks: list[UpdateChunk], display_path: str) -> str:
    updated = original
    for chunk in chunks:
        old_block = _join_patch_lines(chunk.old_lines, trailing_newline=not chunk.is_end_of_file)
        new_block = _join_patch_lines(chunk.new_lines, trailing_newline=not chunk.is_end_of_file)
        occurrences = updated.count(old_block)
        if occurrences == 0:
            raise ValueError(f"Could not find expected content in {display_path}.")
        if occurrences > 1:
            raise ValueError(f"Patch context is ambiguous in {display_path}.")
        updated = updated.replace(old_block, new_block, 1)
    return updated


def _parse_patch(input_text: str) -> list[AddFileHunk | DeleteFileHunk | UpdateFileHunk]:
    lines = input_text.splitlines()
    if not lines or lines[0] != BEGIN_PATCH_MARKER:
        raise ValueError("Patch must start with '*** Begin Patch'.")

    hunks: list[AddFileHunk | DeleteFileHunk | UpdateFileHunk] = []
    index = 1
    saw_end = False

    while index < len(lines):
        line = lines[index]
        if line == END_PATCH_MARKER:
            saw_end = True
            index += 1
            break

        if line.startswith(ADD_FILE_MARKER):
            path = line[len(ADD_FILE_MARKER) :].strip()
            if not path:
                raise ValueError("Add File hunk requires a path.")
            index += 1
            contents: list[str] = []
            while index < len(lines) and lines[index] != END_PATCH_MARKER and not _is_hunk_start(lines[index]):
                current = lines[index]
                if not current.startswith("+"):
                    raise ValueError(f"Invalid add-file line: {current}")
                contents.append(current[1:])
                index += 1
            if not contents:
                raise ValueError(f"Add File hunk for {path} is empty.")
            hunks.append(AddFileHunk(path=path, contents="\n".join(contents) + "\n"))
            continue

        if line.startswith(DELETE_FILE_MARKER):
            path = line[len(DELETE_FILE_MARKER) :].strip()
            if not path:
                raise ValueError("Delete File hunk requires a path.")
            hunks.append(DeleteFileHunk(path=path))
            index += 1
            continue

        if line.startswith(UPDATE_FILE_MARKER):
            path = line[len(UPDATE_FILE_MARKER) :].strip()
            if not path:
                raise ValueError("Update File hunk requires a path.")
            index += 1
            move_to = None
            if index < len(lines) and lines[index].startswith(MOVE_TO_MARKER):
                move_to = lines[index][len(MOVE_TO_MARKER) :].strip()
                if not move_to:
                    raise ValueError("Move to target cannot be empty.")
                index += 1

            chunks: list[UpdateChunk] = []
            while index < len(lines) and lines[index] != END_PATCH_MARKER and not _is_hunk_start(lines[index]):
                if lines[index] not in {"@@"} and not lines[index].startswith("@@ "):
                    raise ValueError(f"Expected update chunk header, got: {lines[index]}")
                index += 1
                old_lines: list[str] = []
                new_lines: list[str] = []
                is_end_of_file = False
                while index < len(lines):
                    current = lines[index]
                    if current == EOF_MARKER:
                        is_end_of_file = True
                        index += 1
                        break
                    if current == END_PATCH_MARKER or _is_hunk_start(current) or current in {"@@"} or current.startswith("@@ "):
                        break
                    if not current or current[0] not in {" ", "+", "-"}:
                        raise ValueError(f"Invalid update line: {current}")
                    if current[0] in {" ", "-"}:
                        old_lines.append(current[1:])
                    if current[0] in {" ", "+"}:
                        new_lines.append(current[1:])
                    index += 1
                chunks.append(UpdateChunk(old_lines=old_lines, new_lines=new_lines, is_end_of_file=is_end_of_file))
            if not chunks and move_to is None:
                raise ValueError(f"Update File hunk for {path} has no changes.")
            hunks.append(UpdateFileHunk(path=path, move_to=move_to, chunks=chunks))
            continue

        raise ValueError(f"Unexpected patch line: {line}")

    if not saw_end:
        raise ValueError("Patch must end with '*** End Patch'.")
    if index != len(lines):
        trailing = "\n".join(lines[index:])
        if trailing.strip():
            raise ValueError("Patch contains unexpected content after '*** End Patch'.")
    if not hunks:
        raise ValueError("Patch did not modify any files.")
    return hunks


def apply_patch_text(input_text: str, workspace_dir: str) -> ApplyPatchResult:
    hunks = _parse_patch(input_text)
    summary = {"added": [], "modified": [], "deleted": []}

    def record(bucket: str, path: str) -> None:
        if path not in summary[bucket]:
            summary[bucket].append(path)

    for hunk in hunks:
        if isinstance(hunk, AddFileHunk):
            target, display = _resolve_patch_path(hunk.path, workspace_dir)
            if target.exists():
                raise ValueError(f"Cannot add {display}: file already exists.")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(hunk.contents, encoding="utf-8")
            record("added", display)
            continue

        if isinstance(hunk, DeleteFileHunk):
            target, display = _resolve_patch_path(hunk.path, workspace_dir)
            if not target.exists():
                raise ValueError(f"Cannot delete {display}: file does not exist.")
            if target.is_dir():
                raise ValueError(f"Cannot delete {display}: directories are not supported.")
            target.unlink()
            record("deleted", display)
            continue

        source, display = _resolve_patch_path(hunk.path, workspace_dir)
        if not source.exists():
            raise ValueError(f"Cannot update {display}: file does not exist.")
        if source.is_dir():
            raise ValueError(f"Cannot update {display}: directories are not supported.")

        updated_text = source.read_text(encoding="utf-8")
        if hunk.chunks:
            updated_text = _apply_update_chunks(updated_text, hunk.chunks, display)

        if hunk.move_to is not None:
            target, moved_display = _resolve_patch_path(hunk.move_to, workspace_dir)
            if target.exists() and target != source:
                raise ValueError(f"Cannot move to {moved_display}: target already exists.")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(updated_text, encoding="utf-8")
            if target != source:
                source.unlink()
            record("modified", moved_display)
        else:
            source.write_text(updated_text, encoding="utf-8")
            record("modified", display)

    return ApplyPatchResult(summary=summary, text=_format_summary(summary))


def create_apply_patch_tool(context: OpenClawToolContext) -> AgentTool:
    async def execute(tool_call_id, args, cancel_event=None, on_update=None):
        del tool_call_id, cancel_event, on_update
        input_text = str(args.get("input", ""))
        if not input_text.strip():
            return error_result("Missing required field: input.")
        try:
            result = apply_patch_text(input_text, context.workspace_dir)
        except ValueError as exc:
            return error_result(str(exc))
        return json_result(
            {
                "status": "ok",
                "summary": result.summary,
                "text": result.text,
            }
        )

    return AgentTool(
        name="apply_patch",
        label="apply_patch",
        description=(
            "Apply a patch using the OpenClaw apply_patch format with *** Begin Patch and "
            "*** End Patch markers."
        ),
        parameters={
            "type": "object",
            "properties": {
                "input": {
                    "type": "string",
                    "description": "Patch content using the *** Begin Patch/End Patch format.",
                },
            },
            "required": ["input"],
        },
        execute=execute,
    )
