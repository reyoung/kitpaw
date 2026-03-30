from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from kitpaw.claw import OpenClawToolContext, create_openclaw_coding_tools
from kitpaw.pi_agent.code_agent.session_manager import SessionManager


def _make_context(tmp_path: Path, *, session_id: str) -> OpenClawToolContext:
    return OpenClawToolContext(
        cwd=str(tmp_path),
        workspace_dir=str(tmp_path),
        spawn_workspace_dir=str(tmp_path),
        agent_id="claw",
        session_id=session_id,
        controller_session_id=session_id,
        model_provider="openai",
        model_id="gpt-4o-mini",
        thinking_level="medium",
        sandboxed=False,
        system_prompt="You are Claw.",
    )


def _tool_by_name(tools: list[Any], name: str):
    return next(tool for tool in tools if tool.name == name)


def _json_payload(result) -> dict[str, Any]:
    return json.loads(result.content[0].text)


@pytest.mark.anyio
async def test_apply_patch_add_update_move_and_delete(tmp_path: Path) -> None:
    manager = SessionManager.in_memory(str(tmp_path))
    context = _make_context(tmp_path, session_id=manager.get_session_id())
    tools = create_openclaw_coding_tools(str(tmp_path), context)
    apply_patch_tool = _tool_by_name(tools, "apply_patch")

    added = _json_payload(
        await apply_patch_tool.execute(
            "patch-add",
            {
                "input": "\n".join(
                    [
                        "*** Begin Patch",
                        "*** Add File: alpha.txt",
                        "+hello",
                        "*** End Patch",
                    ]
                )
            },
            None,
            None,
        )
    )
    assert added["status"] == "ok"
    assert (tmp_path / "alpha.txt").read_text(encoding="utf-8") == "hello\n"

    updated = _json_payload(
        await apply_patch_tool.execute(
            "patch-update",
            {
                "input": "\n".join(
                    [
                        "*** Begin Patch",
                        "*** Update File: alpha.txt",
                        "@@",
                        "-hello",
                        "+world",
                        "*** End Patch",
                    ]
                )
            },
            None,
            None,
        )
    )
    assert updated["status"] == "ok"
    assert (tmp_path / "alpha.txt").read_text(encoding="utf-8") == "world\n"

    moved = _json_payload(
        await apply_patch_tool.execute(
            "patch-move",
            {
                "input": "\n".join(
                    [
                        "*** Begin Patch",
                        "*** Update File: alpha.txt",
                        "*** Move to: beta.txt",
                        "@@",
                        "-world",
                        "+moved",
                        "*** End Patch",
                    ]
                )
            },
            None,
            None,
        )
    )
    assert moved["status"] == "ok"
    assert not (tmp_path / "alpha.txt").exists()
    assert (tmp_path / "beta.txt").read_text(encoding="utf-8") == "moved\n"

    deleted = _json_payload(
        await apply_patch_tool.execute(
            "patch-delete",
            {
                "input": "\n".join(
                    [
                        "*** Begin Patch",
                        "*** Delete File: beta.txt",
                        "*** End Patch",
                    ]
                )
            },
            None,
            None,
        )
    )
    assert deleted["status"] == "ok"
    assert not (tmp_path / "beta.txt").exists()


@pytest.mark.anyio
async def test_apply_patch_rejects_invalid_syntax_and_workspace_escape(tmp_path: Path) -> None:
    manager = SessionManager.in_memory(str(tmp_path))
    context = _make_context(tmp_path, session_id=manager.get_session_id())
    tools = create_openclaw_coding_tools(str(tmp_path), context)
    apply_patch_tool = _tool_by_name(tools, "apply_patch")

    invalid = _json_payload(
        await apply_patch_tool.execute(
            "patch-invalid",
            {
                "input": "\n".join(
                    [
                        "*** Begin Patch",
                        "*** Add File: alpha.txt",
                        "+hello",
                    ]
                )
            },
            None,
            None,
        )
    )
    assert invalid["status"] == "error"
    assert "End Patch" in invalid["error"]

    escaped = _json_payload(
        await apply_patch_tool.execute(
            "patch-escape",
            {
                "input": "\n".join(
                    [
                        "*** Begin Patch",
                        "*** Add File: ../oops.txt",
                        "+oops",
                        "*** End Patch",
                    ]
                )
            },
            None,
            None,
        )
    )
    assert escaped["status"] == "error"
    assert "outside the working directory" in escaped["error"]
