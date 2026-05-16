"""End-to-end integration tests for v2.0 with ``MockLLM`` — no real OpenAI."""

from __future__ import annotations

import json

from ramdocs_rag.core.llm import LLMCallResult
from ramdocs_rag.core.types import DocEvalMeta, Question, RAMDoc
from ramdocs_rag.pipelines.v2_0_entity_first import V2EntityFirst


class _EntityAwareMockLLM:
    """Mock that returns different answers depending on the document text.

    Detects marker words in the user prompt to decide which ``entity``
    and ``text`` to put into the ``Claim``. Also handles mediator calls
    (the intra-entity LLM path).
    """

    model = "mock-entity-aware"

    def __init__(self) -> None:
        self.calls = 0

    def complete_json(self, *, system, user, schema, schema_name, temperature=0.0):
        self.calls += 1
        # mediator path?
        if schema_name == "IntraEntityMediator":
            # Simple strategy: return the first candidate ``text``.
            # Parse the first ``text=...`` token from the user prompt.
            import re

            m = re.search(r"text=([\"'])(.*?)\1", user)
            text = m.group(2) if m else "unknown"
            parsed = {
                "answer": text,
                "confidence": 0.8,
                "supporting_doc_ids": ["d0"],
                "rejected_doc_ids": [],
                "reconciliation_explanation": "mock pick",
            }
        else:  # analyzer
            doc_id = ""
            for line in user.splitlines():
                if line.startswith("Document id: "):
                    doc_id = line.removeprefix("Document id: ").strip()
            doc_text = user.split("---", 1)[1] if "---" in user else user
            entity = ""
            text = ""
            stance = "supports"
            if "Placebo" in doc_text:
                entity = "Without You I'm Nothing (Placebo album)"
                text = "Placebo"
            elif "Sandra Bernhard" in doc_text:
                entity = "Without You I'm Nothing (Sandra Bernhard album)"
                text = "Sandra Bernhard"
            elif "Beatles" in doc_text:
                entity = "Without You I'm Nothing (Beatles album)"
                text = "The Beatles"
            else:
                stance = "no_answer"
            parsed = {
                "doc_id": doc_id,
                "entity": entity,
                "text": text,
                "stance": stance,
                "confidence": 0.9,
                "supporting_quote": "mock-quote",
            }
        return LLMCallResult(
            parsed=parsed,
            raw_text=json.dumps(parsed),
            cost_usd=0.0,
            tokens_in=0,
            tokens_out=0,
            model=self.model,
        )


def _mk_q(docs, golds, meta, qid="qt", cat="pure_correct") -> Question:
    return Question(
        question_id=qid,
        question="Who is the artist of 'Without You I'm Nothing'?",
        category=cat,
        disambig_entity=[],
        gold_answers=golds,
        wrong_answers=[],
        docs=[RAMDoc(doc_id=d, text=t) for d, t in docs],
        eval_metadata=[DocEvalMeta(doc_id=d, type=t, answer=a) for d, t, a in meta],  # type: ignore[arg-type]
    )


def test_v2_multi_answer_two_entities():
    """3 Placebo docs + 3 Bernhard docs → both variants must survive."""
    q = _mk_q(
        docs=[
            ("d0", "Without You I'm Nothing is by Placebo, 1998."),
            ("d1", "Placebo released this album."),
            ("d2", "Album by Placebo."),
            ("d3", "Without You I'm Nothing is by Sandra Bernhard, comedy."),
            ("d4", "Sandra Bernhard recorded this."),
            ("d5", "Sandra Bernhard live album."),
        ],
        golds=["Placebo", "Sandra Bernhard"],
        meta=[
            ("d0", "correct", "Placebo"),
            ("d1", "correct", "Placebo"),
            ("d2", "correct", "Placebo"),
            ("d3", "correct", "Sandra Bernhard"),
            ("d4", "correct", "Sandra Bernhard"),
            ("d5", "correct", "Sandra Bernhard"),
        ],
    )
    pipe = V2EntityFirst(llm=_EntityAwareMockLLM())
    result = pipe.run(q)

    assert result.error is None
    assert len(result.final_answer.variants) == 2
    answers = {v.answer for v in result.final_answer.variants}
    assert "Placebo" in answers
    assert "Sandra Bernhard" in answers


def test_v2_misinfo_entity_filtered():
    """4 Placebo + 1 Beatles (misinfo). The Beatles group must be pruned
    by ``min_relative_weight`` (1 doc << 4 docs)."""
    q = _mk_q(
        docs=[
            ("d0", "Without You I'm Nothing by Placebo."),
            ("d1", "Placebo album from 1998."),
            ("d2", "Placebo released this."),
            ("d3", "Without You I'm Nothing - Placebo."),
            ("dm", "Album by The Beatles."),
        ],
        golds=["Placebo"],
        meta=[
            ("d0", "correct", "Placebo"),
            ("d1", "correct", "Placebo"),
            ("d2", "correct", "Placebo"),
            ("d3", "correct", "Placebo"),
            ("dm", "misinfo", "The Beatles"),
        ],
        cat="has_misinfo",
    )
    pipe = V2EntityFirst(llm=_EntityAwareMockLLM())
    result = pipe.run(q)

    assert result.error is None
    answers = {v.answer for v in result.final_answer.variants}
    assert "Placebo" in answers
    assert "The Beatles" not in answers
    assert "dm" in result.final_answer.rejected_doc_ids


def test_v2_abstains_when_all_no_answer():
    q = _mk_q(
        docs=[("d0", "Some unrelated text."), ("d1", "Other unrelated text.")],
        golds=["Placebo"],
        meta=[("d0", "noise", None), ("d1", "noise", None)],
        cat="has_noise",
    )
    pipe = V2EntityFirst(llm=_EntityAwareMockLLM())
    result = pipe.run(q)

    assert result.error is None
    assert result.final_answer.abstained is True
    assert result.final_answer.variants == []


def test_v2_describe_contains_config():
    pipe = V2EntityFirst(llm=_EntityAwareMockLLM(), config={"min_relative_weight": 0.5})
    desc = pipe.describe()
    assert desc["name"] == "v2.0_entity_first"
    assert desc["version"] == "2.0.0"
    assert desc["min_relative_weight"] == 0.5
