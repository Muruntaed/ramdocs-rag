"""Tests for ``MockLLM`` — must itself be deterministic."""

from __future__ import annotations

import pytest

from ramdocs_rag.core.llm import MockLLM


def test_mock_returns_scripted_response():
    mock = MockLLM()
    mock.script(system="sys", user="hello", response={"x": 1})
    out = mock.complete_json(system="sys", user="hello", schema={}, schema_name="s")
    assert out.parsed == {"x": 1}
    assert out.model == "mock-llm"
    assert mock.calls == 1


def test_mock_falls_back_to_default():
    mock = MockLLM(default={"stance": "no_answer"})
    out = mock.complete_json(system="s", user="anything", schema={}, schema_name="x")
    assert out.parsed == {"stance": "no_answer"}


def test_mock_raises_without_default_and_script():
    mock = MockLLM()
    with pytest.raises(KeyError):
        mock.complete_json(system="s", user="u", schema={}, schema_name="x")


def test_mock_records_prompts():
    mock = MockLLM(default={})
    mock.complete_json(system="A", user="1", schema={}, schema_name="x")
    mock.complete_json(system="B", user="2", schema={}, schema_name="x")
    assert mock.last_prompts == [("A", "1"), ("B", "2")]
    assert mock.calls == 2
