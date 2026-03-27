from __future__ import annotations

from pathlib import Path

import pytest

from kitpaw.pi_agent.code_agent.tools import create_all_tools


@pytest.mark.anyio
async def test_code_agent_read_write_edit_tools(tmp_path: Path) -> None:
    tools = create_all_tools(str(tmp_path))

    write_result = await tools["write"].execute("call-1", {"path": "notes.txt", "content": "hello\nworld"})
    assert "Successfully wrote" in write_result.content[0].text

    read_result = await tools["read"].execute("call-2", {"path": "notes.txt"})
    assert read_result.content[0].text == "hello\nworld"

    edit_result = await tools["edit"].execute(
        "call-3",
        {"path": "notes.txt", "oldText": "world", "newText": "paw"},
    )
    assert "Successfully replaced text" in edit_result.content[0].text
    assert "paw" in edit_result.details["diff"]

    assert (tmp_path / "notes.txt").read_text(encoding="utf-8") == "hello\npaw"


@pytest.mark.anyio
async def test_code_agent_find_grep_ls_tools(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("print('hello')\nprint('world')\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("hello paw\n", encoding="utf-8")
    tools = create_all_tools(str(tmp_path))

    ls_result = await tools["ls"].execute("call-ls", {"path": "."})
    assert "src/" in ls_result.content[0].text
    assert "README.md" in ls_result.content[0].text

    find_result = await tools["find"].execute("call-find", {"pattern": "*.py"})
    assert "src/main.py" in find_result.content[0].text

    grep_result = await tools["grep"].execute("call-grep", {"pattern": "world", "path": "src"})
    assert "main.py:2:" in grep_result.content[0].text


@pytest.mark.anyio
async def test_code_agent_bash_tool(tmp_path: Path) -> None:
    tools = create_all_tools(str(tmp_path))
    result = await tools["bash"].execute("call-bash", {"command": "printf 'hello\\nworld'"})
    assert result.content[0].text == "hello\nworld"


@pytest.mark.anyio
async def test_code_agent_bash_tool_exposes_truncation_details(tmp_path: Path) -> None:
    tools = create_all_tools(str(tmp_path))
    long_output = "for i in $(seq 1 2505); do echo line-$i; done"
    result = await tools["bash"].execute("call-bash-long", {"command": long_output})

    assert "Showing lines" in result.content[0].text
    assert result.details is not None
    assert result.details["truncation"]["truncated"] is True
    assert result.details["truncation"]["truncated_by"] == "lines"
