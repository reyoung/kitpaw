from __future__ import annotations

from paw.pi_agent.tui import fuzzy_filter, fuzzy_match


def test_fuzzy_match_empty_query_matches_everything() -> None:
    """empty query matches everything with score 0"""
    result = fuzzy_match("", "anything")
    assert result.matches is True
    assert result.score == 0


def test_fuzzy_match_query_longer_than_text_does_not_match() -> None:
    """query longer than text does not match"""
    result = fuzzy_match("longquery", "short")
    assert result.matches is False


def test_fuzzy_match_exact_match_scores_well() -> None:
    """exact match has good score"""
    result = fuzzy_match("test", "test")
    assert result.matches is True
    assert result.score < 0


def test_fuzzy_match_requires_in_order_characters() -> None:
    """characters must appear in order"""
    assert fuzzy_match("abc", "aXbXc").matches is True
    assert fuzzy_match("abc", "cba").matches is False


def test_fuzzy_match_is_case_insensitive() -> None:
    """case insensitive matching"""
    assert fuzzy_match("ABC", "abc").matches is True
    assert fuzzy_match("abc", "ABC").matches is True


def test_fuzzy_match_rewards_consecutive_matches() -> None:
    """consecutive matches score better than scattered matches"""
    consecutive = fuzzy_match("foo", "foobar")
    scattered = fuzzy_match("foo", "f_o_o_bar")
    assert consecutive.matches is True
    assert scattered.matches is True
    assert consecutive.score < scattered.score


def test_fuzzy_match_rewards_word_boundary_matches() -> None:
    """word boundary matches score better"""
    at_boundary = fuzzy_match("fb", "foo-bar")
    not_at_boundary = fuzzy_match("fb", "afbx")
    assert at_boundary.matches is True
    assert not_at_boundary.matches is True
    assert at_boundary.score < not_at_boundary.score


def test_fuzzy_match_handles_swapped_alpha_numeric_tokens() -> None:
    """matches swapped alpha numeric tokens"""
    result = fuzzy_match("codex52", "gpt-5.2-codex")
    assert result.matches is True


def test_fuzzy_filter_returns_all_items_for_empty_query() -> None:
    """empty query returns all items unchanged"""
    items = ["apple", "banana", "cherry"]
    assert fuzzy_filter(items, "", lambda item: item) == items


def test_fuzzy_filter_filters_non_matches() -> None:
    """filters out non-matching items"""
    items = ["apple", "banana", "cherry"]
    result = fuzzy_filter(items, "an", lambda item: item)
    assert "banana" in result
    assert "apple" not in result
    assert "cherry" not in result


def test_fuzzy_filter_sorts_by_match_quality() -> None:
    """sorts results by match quality"""
    items = ["a_p_p", "app", "application"]
    result = fuzzy_filter(items, "app", lambda item: item)
    assert result[0] == "app"


def test_fuzzy_filter_supports_custom_get_text() -> None:
    """works with custom getText function"""
    items = [
        {"name": "foo", "id": 1},
        {"name": "bar", "id": 2},
        {"name": "foobar", "id": 3},
    ]
    result = fuzzy_filter(items, "foo", lambda item: item["name"])
    assert len(result) == 2
    assert {item["name"] for item in result} == {"foo", "foobar"}
