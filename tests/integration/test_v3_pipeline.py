"""Integration tests for v3.0 (Entity-First + Skeptic) on MockLLM."""

from __future__ import annotations

import json

from ramdocs_rag.core.llm import LLMCallResult
from ramdocs_rag.core.types import DocEvalMeta, Question, RAMDoc
from ramdocs_rag.pipelines.v3_0_skeptic import V3Skeptic


class _Mock:
    """LLM mock that handles analyzer + intra-mediator + skeptic by schema_name."""

    model = "mock-v3"

    def __init__(self, skeptic_rejects: set[str] | None = None) -> None:
        self.calls = 0
        self.skeptic_rejects = skeptic_rejects or set()

    def complete_json(self, *, system, user, schema, schema_name, temperature=0.0):
        self.calls += 1

        if schema_name == "SkepticVerdicts":
            # Parse entities from user prompt and apply self.skeptic_rejects
            import re
            entities = re.findall(r"entity=([\"'])(.*?)\1", user)
            decisions = []
            for _, ent in entities:
                verdict = "reject" if ent in self.skeptic_rejects else "keep"
                decisions.append({"entity": ent, "verdict": verdict, "reason": "mock"})
            parsed = {"decisions": decisions}

        elif schema_name == "IntraEntityMediator":
            # Pick the first claim's text
            import re
            m = re.search(r"text=([\"'])(.*?)\1", user)
            text = m.group(2) if m else "unknown"
            parsed = {
                "answer": text, "confidence": 0.8,
                "supporting_doc_ids": ["d0"], "rejected_doc_ids": [],
                "reconciliation_explanation": "mock",
            }

        else:  # EntityClaim (analyzer)
            doc_id = ""
            for line in user.splitlines():
                if line.startswith("Document id: "):
                    doc_id = line.removeprefix("Document id: ").strip()
            doc_text = user.split("---", 1)[1] if "---" in user else user
            if "Placebo" in doc_text:
                entity = "Without You I'm Nothing (Placebo album)"
                text = "Placebo"
                stance = "supports"
            elif "Sandra Bernhard" in doc_text:
                entity = "Without You I'm Nothing (Sandra Bernhard album)"
                text = "Sandra Bernhard"
                stance = "supports"
            elif "Beatles" in doc_text:
                entity = "Without You I'm Nothing (Beatles album)"
                text = "The Beatles"
                stance = "supports"
            else:
                entity = ""
                text = ""
                stance = "no_answer"
            parsed = {
                "doc_id": doc_id, "entity": entity, "text": text,
                "stance": stance, "confidence": 0.9, "supporting_quote": "mock",
            }

        return LLMCallResult(
            parsed=parsed, raw_text=json.dumps(parsed),
            cost_usd=0.0, tokens_in=0, tokens_out=0, model=self.model,
        )


def _mk_q(docs, golds, meta, qid="qt", cat="pure_correct") -> Question:
    return Question(
        question_id=qid,
        question="Who is the artist of 'Without You I'm Nothing'?",
        category=cat, disambig_entity=[], gold_answers=golds, wrong_answers=[],
        docs=[RAMDoc(doc_id=d, text=t) for d, t in docs],
        eval_metadata=[DocEvalMeta(doc_id=d, type=t, answer=a) for d, t, a in meta],  # type: ignore[arg-type]
    )


def test_v3_keeps_all_when_skeptic_approves():
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
        meta=[("d0", "correct", "Placebo"), ("d1", "correct", "Placebo"),
              ("d2", "correct", "Placebo"), ("d3", "correct", "Sandra Bernhard"),
              ("d4", "correct", "Sandra Bernhard"), ("d5", "correct", "Sandra Bernhard")],
    )
    pipe = V3Skeptic(llm=_Mock())
    result = pipe.run(q)
    assert result.error is None
    answers = {v.answer for v in result.final_answer.variants}
    assert "Placebo" in answers
    assert "Sandra Bernhard" in answers


def test_v3_skeptic_rejects_misinfo_variant():
    """Skeptic should be able to reject a misinfo entity entirely."""
    q = _mk_q(
        docs=[
            ("d0", "Without You I'm Nothing by Placebo."),
            ("d1", "Placebo album from 1998."),
            ("d2", "Placebo released this."),
            ("dm", "Album by The Beatles."),
        ],
        golds=["Placebo"],
        meta=[("d0", "correct", "Placebo"), ("d1", "correct", "Placebo"),
              ("d2", "correct", "Placebo"), ("dm", "misinfo", "The Beatles")],
        cat="has_misinfo",
    )
    pipe = V3Skeptic(llm=_Mock(skeptic_rejects={"Without You I'm Nothing (Beatles album)"}))
    result = pipe.run(q)
    answers = {v.answer for v in result.final_answer.variants}
    assert "The Beatles" not in answers
    assert "dm" in result.final_answer.rejected_doc_ids


def test_v3_abstains_when_all_no_answer():
    q = _mk_q(
        docs=[("d0", "Some unrelated text."), ("d1", "Other unrelated text.")],
        golds=["Placebo"],
        meta=[("d0", "noise", None), ("d1", "noise", None)],
        cat="has_noise",
    )
    pipe = V3Skeptic(llm=_Mock())
    result = pipe.run(q)
    assert result.final_answer.abstained is True
    assert result.final_answer.variants == []


def test_v3_describe_lists_three_models():
    pipe = V3Skeptic(llm=_Mock())
    desc = pipe.describe()
    assert desc["name"] == "v3.0_skeptic"
    assert desc["version"] == "3.0.0"
    assert "skeptic_llm_model" in desc
    assert "mediator_llm_model" in desc
