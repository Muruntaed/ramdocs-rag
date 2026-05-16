"""Pipeline-quality metrics. All deterministic, no LLM-judge.

Built on top of ``Question.eval_metadata`` (per-doc RAMDocs ground truth)
and the ``FinalAnswer.variants`` / ``rejected_doc_ids`` returned by the
pipeline.

Design:
- Each metric is a pure function ``(Question, FinalAnswer) -> float``.
- ``compute_metrics`` aggregates a list of ``(Question, RunResult)``
  into a ``RunMetrics`` instance.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from ..core.types import FinalAnswer, Question, RunMetrics, RunResult


def _norm(s: str) -> str:
    """Canonicalise a string for answer comparison: ``lower().strip()``."""
    return s.strip().lower()


def _norm_set(items: Iterable[str]) -> set[str]:
    return {_norm(x) for x in items if x}


# ---------- Answer quality ----------


def em_any_gold(q: Question, ans: FinalAnswer) -> float:
    """1.0 if ``primary_answer`` is an **exact** match against any gold answer."""
    if ans.primary_answer is None:
        return 0.0
    return 1.0 if _norm(ans.primary_answer) in _norm_set(q.gold_answers) else 0.0


def em_substring(q: Question, ans: FinalAnswer) -> float:
    """Tolerant to verbose answers: ``gold ⊆ produced`` as a substring."""
    if ans.primary_answer is None:
        return 0.0
    p = _norm(ans.primary_answer)
    return 1.0 if any(_norm(g) in p for g in q.gold_answers) else 0.0


def recall_all_gold(q: Question, ans: FinalAnswer) -> float:
    """Fraction of gold answers covered by the pipeline's variants.

    Substring comparison: a variant covers a gold answer if the gold
    appears inside the variant text (case-insensitive). Gives fair
    credit to multi-answer pipelines and does not penalise polite
    wording around the answer.
    """
    gold = q.gold_answers
    if not gold:
        return 0.0
    produced = [_norm(v.answer) for v in ans.variants]
    if not produced:
        return 0.0
    hit = sum(1 for g in gold if any(_norm(g) in p for p in produced))
    return hit / len(gold)


def precision_answers(q: Question, ans: FinalAnswer) -> float:
    """Fraction of produced variants that cover at least one gold answer."""
    if not ans.variants:
        return 0.0
    gold = [_norm(g) for g in q.gold_answers]
    if not gold:
        return 0.0
    hit = sum(1 for v in ans.variants if any(g in _norm(v.answer) for g in gold))
    return hit / len(ans.variants)


def f1_multi_answer(q: Question, ans: FinalAnswer) -> float:
    """Harmonic mean of ``recall_all_gold`` and ``precision_answers``."""
    r = recall_all_gold(q, ans)
    p = precision_answers(q, ans)
    if r + p == 0:
        return 0.0
    return 2 * p * r / (p + r)


def abstained(_q: Question, ans: FinalAnswer) -> float:
    return 1.0 if ans.abstained else 0.0


# ---------- Source-handling quality ----------


def _eval_meta_by_id(q: Question) -> dict[str, str]:
    """``doc_id -> type`` ('correct' | 'misinfo' | 'noise')."""
    return {m.doc_id: m.type for m in q.eval_metadata}


def _doc_to_answer(q: Question) -> dict[str, str | None]:
    """``doc_id -> ground-truth answer`` (only set for correct documents)."""
    return {m.doc_id: m.answer for m in q.eval_metadata}


def misinfo_rejection(q: Question, ans: FinalAnswer) -> float:
    """Fraction of misinfo documents that ended up in ``rejected_doc_ids``."""
    meta = _eval_meta_by_id(q)
    misinfo_ids = {d for d, t in meta.items() if t == "misinfo"}
    if not misinfo_ids:
        return 1.0  # nothing to reject — trivially perfect
    rejected = set(ans.rejected_doc_ids)
    return len(misinfo_ids & rejected) / len(misinfo_ids)


def noise_rejection(q: Question, ans: FinalAnswer) -> float:
    meta = _eval_meta_by_id(q)
    noise_ids = {d for d, t in meta.items() if t == "noise"}
    if not noise_ids:
        return 1.0
    rejected = set(ans.rejected_doc_ids)
    return len(noise_ids & rejected) / len(noise_ids)


def correct_citation(q: Question, ans: FinalAnswer) -> float:
    """Citation precision: fraction of supporting docs that are correct documents."""
    cited = ans.all_supporting_doc_ids
    if not cited:
        return 0.0
    meta = _eval_meta_by_id(q)
    correct_cited = sum(1 for d in cited if meta.get(d) == "correct")
    return correct_cited / len(cited)


def citation_faithfulness(q: Question, ans: FinalAnswer) -> float:
    """Does the cited document actually support the produced variant?

    For each variant we check whether at least one of its supporting
    citations carries the same entity in ``eval_metadata.answer``
    (substring match). The metric averages across variants.
    """
    if not ans.variants:
        return 0.0
    doc_to_ans = _doc_to_answer(q)
    scores: list[float] = []
    for v in ans.variants:
        if not v.supporting_doc_ids:
            scores.append(0.0)
            continue
        v_norm = _norm(v.answer)
        hits = 0
        for d in v.supporting_doc_ids:
            gt = doc_to_ans.get(d)
            if gt and _norm(gt) in v_norm:
                hits += 1
        scores.append(hits / len(v.supporting_doc_ids))
    return sum(scores) / len(scores)


def coverage(q: Question, ans: FinalAnswer) -> float:
    """Citation recall: fraction of correct documents from the pool that were cited."""
    meta = _eval_meta_by_id(q)
    correct_ids = {d for d, t in meta.items() if t == "correct"}
    if not correct_ids:
        return 1.0
    cited = set(ans.all_supporting_doc_ids)
    return len(correct_ids & cited) / len(correct_ids)


# ---------- Aggregator ----------


_METRIC_FNS = {
    "em_any_gold": em_any_gold,
    "em_substring": em_substring,
    "recall_all_gold": recall_all_gold,
    "precision_answers": precision_answers,
    "f1_multi_answer": f1_multi_answer,
    "abstention_rate": abstained,
    "misinfo_rejection": misinfo_rejection,
    "noise_rejection": noise_rejection,
    "correct_citation": correct_citation,
    "citation_faithfulness": citation_faithfulness,
    "coverage": coverage,
}


def compute_metrics(pairs: list[tuple[Question, RunResult]]) -> RunMetrics:
    """Compute ``RunMetrics`` from a list of ``(Question, RunResult)``. Pure function."""
    n = len(pairs)
    if n == 0:
        raise ValueError("empty pairs")

    sums: dict[str, float] = dict.fromkeys(_METRIC_FNS, 0.0)
    by_cat_sums: dict[str, dict[str, float]] = defaultdict(lambda: dict.fromkeys(_METRIC_FNS, 0.0))
    by_cat_counts: dict[str, int] = defaultdict(int)

    total_cost = 0.0
    total_calls = 0
    total_latency = 0.0

    for q, r in pairs:
        for name, fn in _METRIC_FNS.items():
            val = fn(q, r.final_answer)
            sums[name] += val
            by_cat_sums[q.category][name] += val
        by_cat_counts[q.category] += 1
        total_cost += r.cost_usd
        total_calls += r.llm_calls
        total_latency += r.latency_s

    avg = {name: sums[name] / n for name in _METRIC_FNS}
    by_category = {
        cat: {name: by_cat_sums[cat][name] / by_cat_counts[cat] for name in _METRIC_FNS}
        for cat in by_cat_counts
    }

    return RunMetrics(
        n_questions=n,
        em_any_gold=avg["em_any_gold"],
        em_substring=avg["em_substring"],
        recall_all_gold=avg["recall_all_gold"],
        precision_answers=avg["precision_answers"],
        f1_multi_answer=avg["f1_multi_answer"],
        abstention_rate=avg["abstention_rate"],
        misinfo_rejection=avg["misinfo_rejection"],
        noise_rejection=avg["noise_rejection"],
        correct_citation=avg["correct_citation"],
        citation_faithfulness=avg["citation_faithfulness"],
        coverage=avg["coverage"],
        total_cost_usd=total_cost,
        avg_cost_per_question_usd=total_cost / n,
        avg_llm_calls_per_question=total_calls / n,
        avg_latency_s=total_latency / n,
        by_category=by_category,
    )
