"""Unit tests for metrics. Each metric is exercised on hand-crafted fixtures."""

from __future__ import annotations

import math

import pytest

from ramdocs_rag.eval.metrics import (
    citation_faithfulness,
    compute_metrics,
    correct_citation,
    coverage,
    em_any_gold,
    em_substring,
    f1_multi_answer,
    misinfo_rejection,
    noise_rejection,
    precision_answers,
    recall_all_gold,
)

# ---------- EM ----------


def test_em_any_gold_hit(mk_q, mk_answer):
    q = mk_q(gold=["Placebo"])
    a = mk_answer(variants=[("Placebo", [])])
    assert em_any_gold(q, a) == 1.0


def test_em_any_gold_case_insensitive(mk_q, mk_answer):
    q = mk_q(gold=["Placebo"])
    a = mk_answer(variants=[("placebo", [])])
    assert em_any_gold(q, a) == 1.0


def test_em_any_gold_miss(mk_q, mk_answer):
    q = mk_q(gold=["Placebo"])
    a = mk_answer(variants=[("The Beatles", [])])
    assert em_any_gold(q, a) == 0.0


def test_em_any_gold_no_answer(mk_q, mk_answer):
    q = mk_q(gold=["Placebo"])
    a = mk_answer(abstained=True)
    assert em_any_gold(q, a) == 0.0


def test_em_substring_matches_long_answer(mk_q, mk_answer):
    q = mk_q(gold=["Placebo"])
    a = mk_answer(variants=[("The artist is Placebo, the British band.", [])])
    assert em_substring(q, a) == 1.0
    assert em_any_gold(q, a) == 0.0  # exact match should not fire here


# ---------- Recall / Precision / F1 (multi-answer) ----------


def test_recall_full_when_both_gold_captured(mk_q, mk_answer):
    q = mk_q(gold=["Placebo", "Sandra Bernhard"])
    a = mk_answer(variants=[("Placebo", []), ("Sandra Bernhard", [])])
    assert recall_all_gold(q, a) == 1.0


def test_recall_half_when_only_one_captured(mk_q, mk_answer):
    q = mk_q(gold=["Placebo", "Sandra Bernhard"])
    a = mk_answer(variants=[("Placebo", [])])
    assert recall_all_gold(q, a) == 0.5


def test_precision_full_when_all_variants_are_gold(mk_q, mk_answer):
    q = mk_q(gold=["Placebo", "Sandra Bernhard"])
    a = mk_answer(variants=[("Placebo", []), ("Sandra Bernhard", [])])
    assert precision_answers(q, a) == 1.0


def test_precision_penalises_extra_wrong_variant(mk_q, mk_answer):
    q = mk_q(gold=["Placebo"])
    a = mk_answer(variants=[("Placebo", []), ("The Beatles", [])])
    assert precision_answers(q, a) == 0.5


def test_f1_harmonic_mean(mk_q, mk_answer):
    q = mk_q(gold=["Placebo", "Sandra Bernhard"])
    # recall=0.5, precision=1.0 → F1 = 2/3
    a = mk_answer(variants=[("Placebo", [])])
    assert math.isclose(f1_multi_answer(q, a), 2 / 3, rel_tol=1e-9)


def test_f1_zero_when_abstained(mk_q, mk_answer):
    q = mk_q(gold=["Placebo"])
    a = mk_answer(abstained=True)
    assert f1_multi_answer(q, a) == 0.0


# ---------- Source handling: misinfo / noise rejection ----------


def test_misinfo_rejection_full(mk_q, mk_answer):
    q = mk_q(
        meta=[
            ("d0", "correct", "Placebo"),
            ("d1", "misinfo", "The Beatles"),
            ("d2", "misinfo", "Queen"),
        ]
    )
    a = mk_answer(variants=[("Placebo", ["d0"])], rejected=["d1", "d2"])
    assert misinfo_rejection(q, a) == 1.0


def test_misinfo_rejection_partial(mk_q, mk_answer):
    q = mk_q(
        meta=[
            ("d0", "correct", "Placebo"),
            ("d1", "misinfo", "The Beatles"),
            ("d2", "misinfo", "Queen"),
        ]
    )
    a = mk_answer(variants=[("Placebo", ["d0"])], rejected=["d1"])
    assert misinfo_rejection(q, a) == 0.5


def test_misinfo_rejection_returns_1_when_no_misinfo(mk_q, mk_answer):
    """When the pool has no misinfo at all the metric must not penalise the system."""
    q = mk_q(meta=[("d0", "correct", "Placebo")])
    a = mk_answer(variants=[("Placebo", ["d0"])])
    assert misinfo_rejection(q, a) == 1.0


def test_noise_rejection(mk_q, mk_answer):
    q = mk_q(meta=[("d0", "correct", "Placebo"), ("d1", "noise", None)])
    a = mk_answer(variants=[("Placebo", ["d0"])], rejected=["d1"])
    assert noise_rejection(q, a) == 1.0


# ---------- Citations ----------


def test_correct_citation_precision(mk_q, mk_answer):
    q = mk_q(
        meta=[
            ("d0", "correct", "Placebo"),
            ("d1", "misinfo", "Queen"),
        ]
    )
    a = mk_answer(variants=[("Placebo", ["d0", "d1"])])
    assert correct_citation(q, a) == 0.5


def test_citation_faithfulness_matches_gt_answer(mk_q, mk_answer):
    q = mk_q(
        gold=["Placebo", "Sandra Bernhard"],
        meta=[
            ("d0", "correct", "Placebo"),
            ("d1", "correct", "Sandra Bernhard"),
        ],
    )
    a = mk_answer(variants=[("Placebo", ["d0"]), ("Sandra Bernhard", ["d1"])])
    assert citation_faithfulness(q, a) == 1.0


def test_citation_faithfulness_wrong_cite(mk_q, mk_answer):
    """Variant ``'Placebo'`` but the citation points at the Sandra Bernhard doc — faithfulness must drop."""
    q = mk_q(
        gold=["Placebo", "Sandra Bernhard"],
        meta=[
            ("d0", "correct", "Placebo"),
            ("d1", "correct", "Sandra Bernhard"),
        ],
    )
    a = mk_answer(variants=[("Placebo", ["d1"])])  # wrong citation on purpose
    assert citation_faithfulness(q, a) == 0.0


def test_coverage_recall_of_correct_docs(mk_q, mk_answer):
    q = mk_q(
        meta=[
            ("d0", "correct", "Placebo"),
            ("d1", "correct", "Placebo"),
            ("d2", "correct", "Sandra Bernhard"),
            ("d3", "misinfo", "Queen"),
        ]
    )
    a = mk_answer(variants=[("Placebo", ["d0", "d1"])])
    # 2 out of 3 correct docs are covered
    assert math.isclose(coverage(q, a), 2 / 3, rel_tol=1e-9)


# ---------- Property-like ----------


def test_adding_misinfo_doc_does_not_change_correct_em(mk_q, mk_answer):
    """If EM=1 on the correct answer, adding (rejected) misinfo docs must not drop EM —
    this is just an invariant of the metric."""
    base = mk_q(gold=["Placebo"])
    a = mk_answer(variants=[("Placebo", ["d0"])], rejected=["d_misinfo"])
    assert em_any_gold(base, a) == 1.0


# ---------- Aggregator: compute_metrics ----------


def test_compute_metrics_smoke(mk_q, mk_answer, mk_result):
    q1 = mk_q(qid="q1", gold=["Placebo"], meta=[("d0", "correct", "Placebo")])
    q2 = mk_q(qid="q2", gold=["Sandra Bernhard"], meta=[("d0", "correct", "Sandra Bernhard")])
    r1 = mk_result(mk_answer(variants=[("Placebo", ["d0"])]), qid="q1")
    r2 = mk_result(mk_answer(abstained=True), qid="q2")
    m = compute_metrics([(q1, r1), (q2, r2)])
    assert m.n_questions == 2
    assert m.em_any_gold == 0.5
    assert m.abstention_rate == 0.5


def test_compute_metrics_empty_raises():
    with pytest.raises(ValueError):
        compute_metrics([])
