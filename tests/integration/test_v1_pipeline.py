"""End-to-end integration tests for v1.0 with ``MockLLM`` — no real OpenAI.

Scenarios:
1. ``pure_correct``: every doc agrees on one entity → the deterministic
   mediator returns it; no LLM call beyond the analyzer.
2. ``has_misinfo``: 3 correct + 1 misinfo with a different answer → the
   reliability formula pushes the misinfo down and the mediator picks
   the correct answer.
3. ``has_noise``: 2 correct + 1 noise (``no_answer``) → the noise doc
   never enters a cluster, lands in ``rejected`` or is simply ignored.
4. ``all_no_answer``: the analyzer returned ``no_answer`` everywhere → abstain.
"""

from __future__ import annotations

from ramdocs_rag.core.llm import MockLLM
from ramdocs_rag.core.types import DocEvalMeta, Question, RAMDoc
from ramdocs_rag.pipelines.v1_0_madam_lite import V1MadamLite


def _mk_question(
    qid: str, docs: list[tuple[str, str]], golds: list[str], meta: list[tuple[str, str, str | None]]
) -> Question:
    return Question(
        question_id=qid,
        question="Who is the artist of the album 'X'?",
        category="has_misinfo",
        disambig_entity=[],
        gold_answers=golds,
        wrong_answers=[],
        docs=[RAMDoc(doc_id=d, text=t) for d, t in docs],
        eval_metadata=[DocEvalMeta(doc_id=d, type=t, answer=a) for d, t, a in meta],  # type: ignore[arg-type]
    )


def _mock_supports(answer: str, doc_id: str, conf: float = 0.9) -> dict:
    return {
        "doc_id": doc_id,
        "text": answer,
        "stance": "supports",
        "confidence": conf,
        "supporting_quote": f"quote for {answer}",
    }


def _mock_no_answer(doc_id: str) -> dict:
    return {
        "doc_id": doc_id,
        "text": "",
        "stance": "no_answer",
        "confidence": 0.3,
        "supporting_quote": "",
    }


def test_v1_deterministic_path_all_agree():
    """3 documents → 3 supports on 'Placebo' → one cluster → no LLM mediator call."""
    q = _mk_question(
        "q_test_1",
        docs=[
            ("d0", "Album X by Placebo, 1998."),
            ("d1", "Placebo released X in 1998."),
            ("d2", "X is a Placebo album."),
        ],
        golds=["Placebo"],
        meta=[
            ("d0", "correct", "Placebo"),
            ("d1", "correct", "Placebo"),
            ("d2", "correct", "Placebo"),
        ],
    )
    mock = MockLLM(default=_mock_supports("Placebo", "X"))
    pipe = V1MadamLite(llm=mock)
    result = pipe.run(q)

    assert result.error is None
    assert result.final_answer.abstained is False
    assert result.final_answer.primary_answer is not None
    assert "Placebo" in result.final_answer.primary_answer
    # 3 analyzer calls; the LLM mediator is not invoked (deterministic path).
    assert result.llm_calls == 3
    # No conflict → no rejected docs.
    assert result.final_answer.rejected_doc_ids == []


def test_v1_misinfo_rejected_by_minority_penalty():
    """3 correct + 1 misinfo. After the two-pass reliability with
    ``minority_penalty``, the misinfo doc should sink and the 'Placebo'
    cluster wins."""
    q = _mk_question(
        "q_test_2",
        docs=[
            ("d0", "Album X by Placebo."),
            ("d1", "X — Placebo album."),
            ("d2", "X by Placebo."),
            ("dm", "Album X is by The Beatles."),
        ],
        golds=["Placebo"],
        meta=[
            ("d0", "correct", "Placebo"),
            ("d1", "correct", "Placebo"),
            ("d2", "correct", "Placebo"),
            ("dm", "misinfo", "The Beatles"),
        ],
    )

    # ``MockLLM`` hashes the (system, user) pair — for per-doc_id responses
    # it is cleaner to write a small local mock that inspects the user prompt.
    class _ByDocLLM:
        """A mock that returns different answers based on the ``doc_id`` seen in the user prompt."""

        model = "mock-by-doc"

        def __init__(self) -> None:
            self.calls = 0

        def complete_json(self, *, system, user, schema, schema_name, temperature=0.0):
            self.calls += 1
            # Parse doc_id out of the user prompt.
            doc_id = None
            for line in user.splitlines():
                if line.startswith("Document id: "):
                    doc_id = line.removeprefix("Document id: ").strip()
                    break
            if doc_id == "dm":
                parsed = _mock_supports("The Beatles", "dm", conf=0.9)
            elif doc_id is not None:
                parsed = _mock_supports("Placebo", doc_id, conf=0.9)
            else:
                # This branch is the mediator path — not expected in this test.
                parsed = {
                    "answer": "Placebo",
                    "confidence": 0.5,
                    "supporting_doc_ids": [],
                    "rejected_doc_ids": [],
                    "reconciliation_explanation": "fallback",
                }
            import json

            from ramdocs_rag.core.llm import LLMCallResult

            return LLMCallResult(
                parsed=parsed,
                raw_text=json.dumps(parsed),
                cost_usd=0.0,
                tokens_in=0,
                tokens_out=0,
                model=self.model,
            )

    pipe = V1MadamLite(llm=_ByDocLLM())
    result = pipe.run(q)

    assert result.error is None
    assert result.final_answer.primary_answer is not None
    assert "Placebo" in result.final_answer.primary_answer
    # The misinfo cluster loses; "dm" lands in the minority set → rejected_doc_ids.
    assert "dm" in result.final_answer.rejected_doc_ids


def test_v1_abstains_when_all_no_answer():
    q = _mk_question(
        "q_test_3",
        docs=[("d0", "Unrelated text."), ("d1", "Also unrelated.")],
        golds=["Placebo"],
        meta=[("d0", "noise", None), ("d1", "noise", None)],
    )
    mock = MockLLM(default=_mock_no_answer("dummy"))
    pipe = V1MadamLite(llm=mock)
    result = pipe.run(q)

    assert result.error is None
    assert result.final_answer.abstained is True
    assert result.final_answer.primary_answer is None


def test_v1_describe_contains_config():
    mock = MockLLM(default=_mock_no_answer("d"))
    pipe = V1MadamLite(llm=mock, config={"top_k": 3})
    desc = pipe.describe()
    assert desc["name"] == "v1.0_madam_lite"
    assert desc["version"] == "1.0.0"
    assert desc["retrieval"]["top_k"] == 3
