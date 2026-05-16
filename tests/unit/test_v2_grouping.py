"""Unit tests for v2.0 entity grouping. No LLM calls."""

from __future__ import annotations

from ramdocs_rag.core.types import Claim
from ramdocs_rag.pipelines.v2_0_entity_first.grouping import (
    canonicalize_entity,
    display_entity,
    group_by_entity,
)


def _c(doc_id: str, entity: str, text: str, stance: str = "supports") -> Claim:
    return Claim(
        doc_id=doc_id, entity=entity, text=text, stance=stance,  # type: ignore[arg-type]
        confidence=0.9, supporting_quote="q",
    )


def test_canonicalize_basic():
    assert canonicalize_entity("Placebo") == "placebo"
    assert canonicalize_entity("  PLACEBO  ") == "placebo"
    assert canonicalize_entity("Placebo (album)") == "placebo (album)"


def test_canonicalize_normalizes_punctuation():
    a = canonicalize_entity("Without You I'm Nothing (Placebo album)")
    b = canonicalize_entity("Without You I'm Nothing (Placebo album)  ")
    assert a == b


def test_canonicalize_empty():
    assert canonicalize_entity("") == ""
    assert canonicalize_entity("   ") == ""


def test_group_two_distinct_entities():
    claims = [
        _c("d0", "Without You I'm Nothing (Placebo album)", "Placebo"),
        _c("d1", "Without You I'm Nothing (Placebo album)", "Placebo"),
        _c("d2", "Without You I'm Nothing (Sandra Bernhard album)", "Sandra Bernhard"),
    ]
    groups = group_by_entity(claims)
    assert len(groups) == 2
    # 2 docs about Placebo, 1 about Bernhard
    sizes = sorted(len(g) for g in groups.values())
    assert sizes == [1, 2]


def test_group_substring_merge():
    """The short title ``"Placebo album"`` should merge into the full
    ``"Without You I'm Nothing (Placebo album)"``."""
    claims = [
        _c("d0", "Without You I'm Nothing (Placebo album)", "Placebo"),
        _c("d1", "Placebo album", "Placebo"),
    ]
    groups = group_by_entity(claims)
    assert len(groups) == 1
    # both claims end up in the same group
    only_group = next(iter(groups.values()))
    assert len(only_group) == 2


def test_group_ignores_non_supports():
    claims = [
        _c("d0", "Placebo", "Placebo", stance="supports"),
        _c("d1", "", "", stance="no_answer"),
        _c("d2", "Beatles", "Beatles", stance="contradicts"),
    ]
    groups = group_by_entity(claims)
    assert len(groups) == 1  # only supports claims are kept


def test_group_empty_entity_supports_dropped():
    """A supports claim without an entity (analyzer bug) must not form a group."""
    claims = [_c("d0", "", "Placebo", stance="supports")]
    assert group_by_entity(claims) == {}


def test_display_entity_picks_longest():
    claims = [
        _c("d0", "Placebo album", "x"),
        _c("d1", "Without You I'm Nothing (Placebo album)", "x"),
    ]
    assert display_entity(claims) == "Without You I'm Nothing (Placebo album)"
