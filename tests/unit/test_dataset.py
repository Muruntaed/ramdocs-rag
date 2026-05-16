"""Sanity checks: the dataset loads and its structure matches expectations."""

from __future__ import annotations

from ramdocs_rag.core.dataset import load_by_id, load_subset


def test_subset_loads():
    subset = load_subset()
    assert len(subset) == 12


def test_categories_distribution():
    subset = load_subset()
    cats = {q.category for q in subset}
    assert cats == {"pure_correct", "has_misinfo", "has_noise", "mixed_conflict"}


def test_each_q_has_eval_meta_for_each_doc():
    for q in load_subset():
        doc_ids = {d.doc_id for d in q.docs}
        meta_ids = {m.doc_id for m in q.eval_metadata}
        assert meta_ids == doc_ids, f"meta mismatch for {q.question_id}"


def test_load_by_id_roundtrip():
    subset = load_subset()
    qid = subset[0].question_id
    assert load_by_id(qid).question_id == qid


def test_at_least_one_ambiguous_question():
    """Our subset contains at least one question with ≥ 2 distinct gold answers."""
    has_ambig = any(len(set(q.gold_answers)) >= 2 for q in load_subset())
    assert has_ambig
