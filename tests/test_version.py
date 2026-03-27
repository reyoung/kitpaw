from __future__ import annotations

from kitpaw.version import normalize_git_describe


def test_normalize_git_describe_exact_tag() -> None:
    assert normalize_git_describe("v0.1.4") == "0.1.4"


def test_normalize_git_describe_with_distance_and_sha() -> None:
    assert normalize_git_describe("v0.1.4-2-g18d07c0") == "0.1.4+2.g18d07c0"


def test_normalize_git_describe_with_distance_sha_and_dirty() -> None:
    assert normalize_git_describe("v0.1.4-2-g18d07c0-dirty") == "0.1.4+2.g18d07c0.dirty"


def test_normalize_git_describe_exact_dirty_uses_sha() -> None:
    assert normalize_git_describe("v0.1.4-dirty", short_sha="18d07c0") == "0.1.4+g18d07c0.dirty"
