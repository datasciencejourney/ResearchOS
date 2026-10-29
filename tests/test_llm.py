"""Unit tests for JSON extraction and resilient structured calls."""

from __future__ import annotations

import pytest

from app.llm import extract_json, structured
from app.schemas.agents import SearchQueries
from tests.fakes import ScriptedChatModel


def test_extract_json_plain_object():
    assert cit_extract('{"a": 1}') == {"a": 1}


def test_extract_json_strips_markdown_fence():
    assert cit_extract('```json\n{"queries": ["x"]}\n```') == {"queries": ["x"]}


def test_extract_json_finds_object_inside_prose():
    assert cit_extract('Sure, here it is: {"a": {"b": 2}} hope that helps') == {"a": {"b": 2}}


def test_extract_json_handles_braces_inside_strings():
    assert cit_extract('{"note": "a { brace", "n": 1}')["n"] == 1


def test_extract_json_raises_without_object():
    import pytest

    from app.llm import extract_json

    with pytest.raises(ValueError):
        extract_json("no json here")


def cit_extract(text: str) -> dict:
    from app.llm import extract_json

    return extract_json(text)


async def test_structured_retries_once_with_valid_schema():
    responses = iter(["not json at all", '{"queries": ["postgres benchmarks"]}'])
    model = ScriptedChatModel(lambda prompt: next(responses))
    result = await structured("pick search queries", SearchQueries, model=model)
    assert result.queries == ["postgres benchmarks"]


async def test_structured_raises_after_repeated_failures():
    model = ScriptedChatModel(lambda prompt: "still not json")
    with pytest.raises(ValueError):
        await structured("pick search queries", SearchQueries, model=model)
