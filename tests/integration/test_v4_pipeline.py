"""Integration tests for v4.0 (Entity-First + Evidence Evaluator + Skeptic) on MockLLM."""

from __future__ import annotations

import json
import re

from ramdocs_rag.core.llm import LLMCallResult
from ramdocs_rag.core.types import DocEvalMeta, Question, RAMDoc
from ramdocs_rag.pipelines.v4_0_evidence_quality import V4EvidenceQuality


class _Mock:
    """LLM mock that handles analyzer + evaluator + intra-mediator + skeptic by schema_name.

    The mock implements just enough behaviour for the pipeline to type-check
    and produce variants. trust_score is set per-doc via the constructor so
    we can probe how the new W_TRUST slot interacts with grouping.
    """

    model = "mock-v4"

    def __init__(
        self,
        trust_by_doc: dict[str, float] | None = None,
        flags_by_doc: dict[str, list[str]] | None = None,
        skeptic_rejects: set[str] | None = None,
    ) -> None:
        self.calls = 0
        self.trust_by_doc = trust_by_doc or {}
        self.flags_by_doc = flags_by_doc or {}
        self.skeptic_rejects = skeptic_rejects or set()

    def complete_json(self, *, system, user, schema, schema_name, temperature=0.0):
        self.calls += 1

        if schema_name == "SkepticVerdicts":
            entities = re.findall(r"entity=([\"'])(.*?)\1", user)
            decisions = []
            for _, ent in entities:
                verdict = "reject" if ent in self.skeptic_rejects else "keep"
                decisions.append({"entity": ent, "verdict": verdict, "reason": "mock"})
            parsed = {"decisions": decisions}

        elif schema_name == "IntraEntityMediator":
            m = re.search(r"text=([\"'])(.*?)\1", user)
            text = m.group(2) if m else "unknown"
            parsed = {
                "answer": text,
                "confidence": 0.8,
                "supporting_doc_ids": ["d0"],
                "rejected_doc_ids": [],
                "reconciliation_explanation": "mock",
            }

        elif schema_name == "DocTrust":
            doc_id = ""
            for line in user.splitlines():
                if line.startswith("Document id: "):
                    doc_id = line.removeprefix("Document id: ").strip()
            trust = self.trust_by_doc.get(doc_id, 0.7)
            flags = self.flags_by_doc.get(doc_id, [])
            parsed = {
                "doc_id": doc_id,
                "internal_consistency": trust,
                "encyclopedic_quality": trust,
                "specificity": trust,
                "relevance": trust,
                "trust_score": trust,
                "red_flags": flags,
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
                "doc_id": doc_id,
                "entity": entity,
                "text": text,
                "stance": stance,
                "confidence": 0.9,
                "supporting_quote": "mock",
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


def test_v4_keeps_all_homonyms_when_skeptic_approves():
    """Two legitimate same-name albums with comparable trust should both survive."""
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
    pipe = V4EvidenceQuality(
        llm=_Mock(trust_by_doc={d: 0.8 for d in ["d0", "d1", "d2", "d3", "d4", "d5"]})
    )
    result = pipe.run(q)
    assert result.error is None
    answers = {v.answer for v in result.final_answer.variants}
    assert "Placebo" in answers
    assert "Sandra Bernhard" in answers


def test_v4_skeptic_rejects_misinfo_variant():
    """Skeptic should be able to reject a misinfo entity entirely."""
    q = _mk_q(
        docs=[
            ("d0", "Without You I'm Nothing by Placebo."),
            ("d1", "Placebo album from 1998."),
            ("d2", "Placebo released this."),
            ("dm", "Album by The Beatles."),
        ],
        golds=["Placebo"],
        meta=[
            ("d0", "correct", "Placebo"),
            ("d1", "correct", "Placebo"),
            ("d2", "correct", "Placebo"),
            ("dm", "misinfo", "The Beatles"),
        ],
        cat="has_misinfo",
    )
    pipe = V4EvidenceQuality(
        llm=_Mock(
            trust_by_doc={"d0": 0.85, "d1": 0.85, "d2": 0.85, "dm": 0.2},
            flags_by_doc={"dm": ["short_stub", "no_specifics"]},
            skeptic_rejects={"Without You I'm Nothing (Beatles album)"},
        )
    )
    result = pipe.run(q)
    answers = {v.answer for v in result.final_answer.variants}
    assert "The Beatles" not in answers
    assert "dm" in result.final_answer.rejected_doc_ids


def test_v4_abstains_when_all_no_answer():
    q = _mk_q(
        docs=[("d0", "Some unrelated text."), ("d1", "Other unrelated text.")],
        golds=["Placebo"],
        meta=[("d0", "noise", None), ("d1", "noise", None)],
        cat="has_noise",
    )
    pipe = V4EvidenceQuality(llm=_Mock(trust_by_doc={"d0": 0.3, "d1": 0.3}))
    result = pipe.run(q)
    assert result.final_answer.abstained is True
    assert result.final_answer.variants == []


def test_v4_describe_lists_four_models():
    """v4 has four model slots: analyzer, evaluator, mediator, skeptic."""
    pipe = V4EvidenceQuality(llm=_Mock())
    desc = pipe.describe()
    assert desc["name"] == "v4.0_evidence_quality"
    assert desc["version"] == "4.0.0"
    assert "evaluator_llm_model" in desc
    assert "skeptic_llm_model" in desc
    assert "mediator_llm_model" in desc


def test_v4_calls_evaluator_per_retrieved_doc():
    """Smoke: evaluator must be invoked once per retrieved doc (parallel to analyzer)."""
    q = _mk_q(
        docs=[
            ("d0", "Without You I'm Nothing by Placebo."),
            ("d1", "Placebo album from 1998."),
            ("d2", "Placebo released this."),
        ],
        golds=["Placebo"],
        meta=[
            ("d0", "correct", "Placebo"),
            ("d1", "correct", "Placebo"),
            ("d2", "correct", "Placebo"),
        ],
    )
    mock = _Mock(trust_by_doc={"d0": 0.8, "d1": 0.8, "d2": 0.8})
    pipe = V4EvidenceQuality(llm=mock)
    result = pipe.run(q)
    assert result.error is None
    # Pipeline calls one analyzer + one evaluator per retrieved doc, plus the
    # Skeptic. With three retrieved docs that's ≥ 3 analyzer + 3 evaluator + 1 skeptic = 7+.
    # Some retrieval paths may include a mediator call; we only assert the floor.
    assert mock.calls >= 7
