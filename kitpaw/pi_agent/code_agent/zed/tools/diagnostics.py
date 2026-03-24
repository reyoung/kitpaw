from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Sequence

from ....agent.types import AgentTool, AgentToolResult
from ....ai.types import TextContent

_OUTPUT_LIMIT = 16 * 1024  # 16KB
_TIMEOUT_SEC = 60.0


def _truncate(text: str) -> str:
    if len(text) > _OUTPUT_LIMIT:
        return text[:_OUTPUT_LIMIT] + "\n... (output truncated at 16KB)"
    return text


async def _run_cmd(
    cmd: Sequence[str], cwd: str
) -> tuple[int | None, str] | None:
    """Run *cmd* and return (exit_code, combined output), or None on FileNotFoundError."""
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
    except FileNotFoundError:
        return None

    try:
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=_TIMEOUT_SEC)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return (
            -1,
            f"Error: Command timed out after {_TIMEOUT_SEC:.0f}s: {' '.join(cmd)}",
        )

    output = stdout.decode("utf-8", errors="replace") if stdout else ""
    return (proc.returncode, output)


def _detect_project_type(cwd: str) -> str | None:
    """Return a project type key based on config files present in *cwd*."""
    root = Path(cwd)
    checks: list[tuple[str, list[str]]] = [
        ("python", ["pyproject.toml", "setup.py"]),
        ("typescript", ["tsconfig.json", "package.json"]),
        ("rust", ["Cargo.toml"]),
        ("go", ["go.mod"]),
    ]
    for project_type, markers in checks:
        for marker in markers:
            if (root / marker).exists():
                return project_type
    return None


async def _run_python_diagnostics(cwd: str, path: str | None) -> str:
    """Try ruff first, fall back to py_compile for single files."""
    path_args = [path] if path else ["."]
    result = await _run_cmd(
        ["ruff", "check", "--output-format=text", *path_args], cwd
    )
    if result is not None:
        exit_code, output = result
        header = f"ruff check {' '.join(path_args)}"
        if not output.strip() and exit_code == 0:
            return f"$ {header}\nNo issues found."
        return f"$ {header}\n{output}"

    # ruff not found — fall back to py_compile for a single file
    if path:
        result = await _run_cmd(
            ["python", "-m", "py_compile", path], cwd
        )
        if result is not None:
            exit_code, output = result
            if not output.strip() and exit_code == 0:
                return f"$ python -m py_compile {path}\nNo syntax errors found."
            return f"$ python -m py_compile {path}\n{output}"

    return (
        "Could not run Python diagnostics: `ruff` is not installed.\n"
        "Install it with `pip install ruff`, or use the terminal tool to run a linter manually."
    )


async def _run_typescript_diagnostics(cwd: str, path: str | None) -> str:
    """Run tsc --noEmit."""
    path_args = [path] if path else []
    cmd = ["npx", "tsc", "--noEmit", *path_args]
    result = await _run_cmd(cmd, cwd)
    if result is not None:
        exit_code, output = result
        label = " ".join(cmd)
        if not output.strip() and exit_code == 0:
            return f"$ {label}\nNo issues found."
        return f"$ {label}\n{output}"
    return (
        "Could not run TypeScript diagnostics: `npx` / `tsc` not found.\n"
        "Use the terminal tool to run a type checker manually."
    )


async def _run_rust_diagnostics(cwd: str, path: str | None) -> str:
    """Run cargo check. Cargo doesn't accept individual files, so *path* is ignored."""
    cmd = ["cargo", "check", "--message-format=short"]
    result = await _run_cmd(cmd, cwd)
    if result is not None:
        exit_code, output = result
        label = " ".join(cmd)
        if not output.strip() and exit_code == 0:
            return f"$ {label}\nNo issues found."
        return f"$ {label}\n{output}"
    return (
        "Could not run Rust diagnostics: `cargo` not found.\n"
        "Use the terminal tool to run `cargo check` manually."
    )


async def _run_go_diagnostics(cwd: str, path: str | None) -> str:
    """Run go vet."""
    target = [path] if path else ["./..."]
    cmd = ["go", "vet", *target]
    result = await _run_cmd(cmd, cwd)
    if result is not None:
        exit_code, output = result
        label = " ".join(cmd)
        if not output.strip() and exit_code == 0:
            return f"$ {label}\nNo issues found."
        return f"$ {label}\n{output}"
    return (
        "Could not run Go diagnostics: `go` not found.\n"
        "Use the terminal tool to run `go vet` manually."
    )


_RUNNERS = {
    "python": _run_python_diagnostics,
    "typescript": _run_typescript_diagnostics,
    "rust": _run_rust_diagnostics,
    "go": _run_go_diagnostics,
}


def create_diagnostics_tool(cwd: str) -> AgentTool:
    """Get diagnostics (errors, warnings) for a file or the project."""

    async def execute(tool_call_id, args, cancel_event=None, on_update=None):
        path: str | None = args.get("path")

        project_type = _detect_project_type(cwd)
        if project_type is None:
            return AgentToolResult(
                content=[TextContent(text=(
                    "Could not auto-detect project type (no pyproject.toml, setup.py, "
                    "tsconfig.json, package.json, Cargo.toml, or go.mod found).\n"
                    "Use the 'terminal' tool to run a linter or type checker directly "
                    "(e.g., 'ruff check .', 'mypy .', 'tsc --noEmit')."
                ))],
                details=None,
            )

        runner = _RUNNERS[project_type]
        try:
            output = await runner(cwd, path)
        except Exception as e:
            output = f"Error running diagnostics: {e}"

        return AgentToolResult(
            content=[TextContent(text=_truncate(output))],
            details=None,
        )

    return AgentTool(
        name="diagnostics",
        label="Diagnostics",
        description=(
            "Get diagnostics (errors, warnings, hints) for a file or the entire project. "
            "Auto-detects the project type (Python, TypeScript, Rust, Go) and runs "
            "the appropriate linter or type checker. Returns raw linter output."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Optional file path to get diagnostics for. If omitted, returns project-wide diagnostics.",
                },
            },
            "required": [],
        },
        execute=execute,
    )
