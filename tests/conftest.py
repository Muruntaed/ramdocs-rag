"""Shared pytest fixtures. No imports from ``pipelines/`` — only ``core/``."""

from __future__ import annotations

import pytest

from ramdocs_rag.core.dataset import load_subset
from ramdocs_rag.core.types import (
    AnswerVariant,
    DocEvalMeta,
    FinalAnswer,
    Question,
    RAMDoc,
    RunResult,
)


# ---------- Fixtures sourced from the real dataset ----------


@pytest.fixture(scope="session")
def subset() -> tuple[Question, ...]:
    return load_subset()


@pytest.fixture(scope="session")
def pure_correct_q(subset: tuple[Question, ...]) -> Question:
    return next(q for q in subset if q.category == "pure_correct")


@pytest.fixture(scope="session")
def has_misinfo_q(subset: tuple[Question, ...]) -> Question:
    return next(q for q in subset if q.category == "has_misinfo")


@pytest.fixture(scope="session")
def has_noise_q(subset: tuple[Question, ...]) -> Question:
    return next(q for q in subset if q.category == "has_noise")


# ---------- Synthetic fixtures for the metrics layer ----------


def make_q(
    *,
    qid: str = "qtest",
    question: str = "Who is the artist of X?",
    gold: list[str] | None = None,
    docs: list[tuple[str, str]] | None = None,
    meta: list[tuple[str, str, str | None]] | None = None,
    category: str = "has_misinfo",
) -> Question:
    """Build a synthetic ``Question`` for unit tests of the metrics layer."""
    gold = gold or ["Placebo"]
    docs = docs or [("d0", "doc text 0")]
    meta = meta or [("d0", "correct", "Placebo")]
    return Question(
        question_id=qid,
        question=question,
        category=category,  # type: ignore[arg-type]
        disambig_entity=[],
        gold_answers=gold,
        wrong_answers=[],
        docs=[RAMDoc(doc_id=d, text=t) for d, t in docs],
        eval_metadata=[DocEvalMeta(doc_id=d, type=t, answer=a) for d, t, a in meta],  # type: ignore[arg-type]
    )


def make_answer(
    *,
    variants: list[tuple[str, list[str]]] | None = None,
    rejected: list[str] | None = None,
    abstained: bool = False,
) -> FinalAnswer:
    """Build a ``FinalAnswer`` for unit tests. ``variants``: ``[(text, [doc_ids])]``."""
    return FinalAnswer(
        variants=[
            AnswerVariant(answer=a, confidence=1.0, supporting_doc_ids=ids)
            for a, ids in (variants or [])
        ],
        rejected_doc_ids=rejected or [],
        abstained=abstained,
    )


def make_result(answer: FinalAnswer, *, qid: str = "qtest") -> RunResult:
    return RunResult(question_id=qid, final_answer=answer)


@pytest.fixture()
def mk_q():
    return make_q


@pytest.fixture()
def mk_answer():
    return make_answer


@pytest.fixture()
def mk_result():
    return make_result
