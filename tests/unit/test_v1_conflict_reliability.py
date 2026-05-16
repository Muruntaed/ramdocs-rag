"""Unit tests for v1.0 deterministic modules: conflict + reliability.

No LLM calls. The sentence-transformers embedder loads on first use of
the clusterer; the suite still runs in the fast pytest pass by default
but can be opted out via ``pytest -m "not slow"`` if needed.
"""

from __future__ import annotations

import pytest

from ramdocs_rag.core.types import Claim, RetrievedDoc
from ramdocs_rag.pipelines.v1_0_madam_lite.conflict import (
    MAJORITY_RATIO,
    detect_conflict,
    minority_doc_ids,
)
from ramdocs_rag.pipelines.v1_0_madam_lite.reliability import (
    W_AUTHORITY,
    W_MINORITY_PENALTY,
    W_RECENCY,
    W_RETRIEVAL,
    W_SELF_CONFIDENCE,
    compute_reliability,
)


# ---------- Reliability ----------


def test_reliability_weights_sum_to_one_excluding_penalty():
    # Penalty is a subtractive term, not part of the convex combination.
    s = W_RETRIEVAL + W_SELF_CONFIDENCE + W_RECENCY + W_AUTHORITY
    assert abs(s - 1.0) < 1e-9


def test_reliability_provisional_no_minority_when_one_cluster():
    docs = [
        RetrievedDoc(doc_id="d0", text="x", score=0.8),
        RetrievedDoc(doc_id="d1", text="y", score=0.6),
    ]
    claims = [
        Claim(doc_id="d0", text="Placebo", stance="supports", confidence=0.9, supporting_quote="q"),
        Claim(doc_id="d1", text="Placebo", stance="supports", confidence=0.7, supporting_quote="q"),
    ]
    rel, _ = compute_reliability(docs, claims)
    # With a single cluster, the penalty term is 0 for both docs.
    # score = 0.40*ret + 0.25*conf + 0.20*0.5 + 0.15*0.5
    expected_d0 = 0.40 * 0.8 + 0.25 * 0.9 + 0.20 * 0.5 + 0.15 * 0.5
    assert abs(rel["d0"] - expected_d0) < 1e-9


def test_reliability_clipped_to_unit_interval():
    docs = [RetrievedDoc(doc_id="d0", text="x", score=1.0)]
    claims = [
        Claim(doc_id="d0", text="X", stance="supports", confidence=1.0, supporting_quote="q"),
    ]
    rel, _ = compute_reliability(docs, claims)
    assert 0.0 <= rel["d0"] <= 1.0


def test_reliability_no_claim_defaults_self_conf_05():
    docs = [RetrievedDoc(doc_id="d0", text="x", score=0.5)]
    rel, _ = compute_reliability(docs, [])
    expected = 0.40 * 0.5 + 0.25 * 0.5 + 0.20 * 0.5 + 0.15 * 0.5  # = 0.50
    assert abs(rel["d0"] - expected) < 1e-9


# ---------- Conflict detection ----------


def test_detect_conflict_no_claims():
    report = detect_conflict([], {})
    assert report.clusters == ()
    assert report.winner is None


def test_detect_conflict_single_cluster_clear_winner():
    """All claims agree on one answer → one cluster, that cluster is the winner."""
    claims = [
        Claim(doc_id="d0", text="Placebo", stance="supports", confidence=0.9, supporting_quote="q"),
        Claim(doc_id="d1", text="Placebo", stance="supports", confidence=0.8, supporting_quote="q"),
    ]
    rel = {"d0": 0.7, "d1": 0.6}
    report = detect_conflict(claims, rel)
    assert report.winner is not None
    assert set(report.winner.members) == {"d0", "d1"}
    assert minority_doc_ids(report) == set()


def test_detect_conflict_two_clusters_dominant():
    """Two clusters with a weight gap above ``MAJORITY_RATIO`` → there is a clear winner."""
    claims = [
        Claim(doc_id="d0", text="Placebo", stance="supports", confidence=0.9, supporting_quote="q"),
        Claim(doc_id="d1", text="Placebo", stance="supports", confidence=0.9, supporting_quote="q"),
        Claim(doc_id="d2", text="The Beatles", stance="supports", confidence=0.9, supporting_quote="q"),
    ]
    rel = {"d0": 0.9, "d1": 0.9, "d2": 0.3}
    report = detect_conflict(claims, rel)
    # ratio = (0.9+0.9)/0.3 = 6.0 >> 1.5
    assert report.winner is not None
    assert report.ratio >= MAJORITY_RATIO
    assert "d2" in minority_doc_ids(report)


def test_detect_conflict_two_clusters_inconclusive():
    """Clusters tied on weight → no deterministic winner, fall through to the LLM mediator."""
    claims = [
        Claim(doc_id="d0", text="Placebo", stance="supports", confidence=0.9, supporting_quote="q"),
        Claim(doc_id="d1", text="The Beatles", stance="supports", confidence=0.9, supporting_quote="q"),
    ]
    rel = {"d0": 0.5, "d1": 0.5}
    report = detect_conflict(claims, rel)
    assert report.winner is None
    assert len(report.clusters) == 2
    assert report.ratio < MAJORITY_RATIO


def test_no_answer_claims_excluded():
    """``no_answer`` claims never enter a cluster — even when they are the majority."""
    claims = [
        Claim(doc_id="d0", text="", stance="no_answer", confidence=0.5, supporting_quote=""),
        Claim(doc_id="d1", text="", stance="no_answer", confidence=0.5, supporting_quote=""),
        Claim(doc_id="d2", text="Placebo", stance="supports", confidence=0.9, supporting_quote="q"),
    ]
    rel = {"d0": 0.5, "d1": 0.5, "d2": 0.5}
    report = detect_conflict(claims, rel)
    # Only one supports claim → one cluster → winner exists and contains only d2.
    assert report.winner is not None
    assert set(report.winner.members) == {"d2"}


# ---------- Property: adding a misinfo cluster must not steal the win ----------


def test_property_adding_minority_misinfo_doesnt_steal_win():
    base_claims = [
        Claim(doc_id="d0", text="Placebo", stance="supports", confidence=0.9, supporting_quote="q"),
        Claim(doc_id="d1", text="Placebo", stance="supports", confidence=0.9, supporting_quote="q"),
    ]
    rel_base = {"d0": 0.8, "d1": 0.8}
    win0 = detect_conflict(base_claims, rel_base).winner
    assert win0 is not None

    extended = base_claims + [
        Claim(doc_id="dm", text="The Beatles", stance="supports", confidence=0.9, supporting_quote="q"),
    ]
    rel_ext = {**rel_base, "dm": 0.3}
    win1 = detect_conflict(extended, rel_ext).winner
    assert win1 is not None
    assert win0.representative_text == win1.representative_text
