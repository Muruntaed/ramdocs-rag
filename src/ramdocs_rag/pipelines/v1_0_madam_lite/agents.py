"""v1.0 Analyzer and Mediator. Both consume ``LLMClient`` through DI.

- Analyzer: per-doc → Claim. Default model: ``gpt-4o-mini``.
- Mediator: claims + reliability → FinalAnswer. Two-stage:
  1) deterministic weighted vote (no LLM);
  2) LLM adjudication (``gpt-4o``) when the vote is inconclusive.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from ...core.llm import LLMClient
from ...core.safety import apply_safety
from ...core.types import AnswerVariant, Claim, FinalAnswer, RetrievedDoc
from .conflict import ConflictReport, detect_conflict, minority_doc_ids

_PROMPT_DIR = Path(__file__).resolve().parent / "prompts"


@lru_cache(maxsize=2)
def _read_prompt(name: str) -> str:
    raw = (_PROMPT_DIR / name).read_text(encoding="utf-8")
    return apply_safety(raw, name)


# ---------- JSON schemas (for OpenAI strict mode) ----------

_CLAIM_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["doc_id", "text", "stance", "confidence", "supporting_quote"],
    "properties": {
        "doc_id": {"type": "string"},
        "text": {"type": "string"},
        "stance": {"type": "string", "enum": ["supports", "contradicts", "no_answer"]},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "supporting_quote": {"type": "string"},
    },
}

_MEDIATOR_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "answer",
        "confidence",
        "supporting_doc_ids",
        "rejected_doc_ids",
        "reconciliation_explanation",
    ],
    "properties": {
        "answer": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "supporting_doc_ids": {"type": "array", "items": {"type": "string"}},
        "rejected_doc_ids": {"type": "array", "items": {"type": "string"}},
        "reconciliation_explanation": {"type": "string"},
    },
}


# ---------- Analyzer ----------


def analyze_doc(llm: LLMClient, query: str, doc: RetrievedDoc) -> tuple[Claim, float, int]:
    """One document → one ``Claim``. Returns ``(claim, cost_usd, n_calls=1)``."""
    system = _read_prompt("analyzer.txt")
    user = (
        f"Question: {query}\n\n"
        f"Document id: {doc.doc_id}\n"
        f"Document text:\n---\n{doc.text}\n---\n"
    )
    out = llm.complete_json(
        system=system, user=user, schema=_CLAIM_SCHEMA, schema_name="Claim"
    )
    parsed = dict(out.parsed)
    # Guard against a hallucinated ``doc_id``.
    parsed["doc_id"] = doc.doc_id
    claim = Claim.model_validate(parsed)
    return claim, out.cost_usd, 1


# ---------- Mediator: deterministic path ----------


def _deterministic_answer(report: ConflictReport) -> FinalAnswer:
    """No LLM: the top cluster won decisively → assemble ``FinalAnswer``."""
    assert report.winner is not None
    winner = report.winner
    total_weight = sum(c.weight for c in report.clusters) or 1.0
    confidence = min(1.0, winner.weight / total_weight)
    if report.runner_up is not None:
        explanation = (
            f"Weighted-majority vote: '{winner.representative_text}' "
            f"summed reliability {winner.weight:.3f} vs runner-up "
            f"'{report.runner_up.representative_text}' {report.runner_up.weight:.3f} "
            f"(ratio {report.ratio:.2f} ≥ threshold)."
        )
    else:
        explanation = (
            f"All supports-claims agree on '{winner.representative_text}'. "
            f"No conflict detected; LLM not invoked."
        )
    return FinalAnswer(
        variants=[
            AnswerVariant(
                answer=winner.representative_text,
                confidence=confidence,
                supporting_doc_ids=list(winner.members),
            )
        ],
        rejected_doc_ids=sorted(minority_doc_ids(report)),
        explanation=explanation,
    )


def _format_claims_for_mediator(claims: list[Claim], reliability: dict[str, float]) -> str:
    lines = []
    for c in claims:
        lines.append(
            f"- doc_id={c.doc_id} stance={c.stance} "
            f"confidence={c.confidence:.2f} "
            f"reliability={reliability.get(c.doc_id, 0.0):.3f}\n"
            f"  text: {c.text!r}\n"
            f"  quote: {c.supporting_quote!r}"
        )
    return "\n".join(lines)


def _format_clusters(report: ConflictReport) -> str:
    if not report.clusters:
        return "(no supports-clusters — all claims were no_answer/contradicts)"
    return "\n".join(
        f"- '{c.representative_text}': weight={c.weight:.3f}, members={list(c.members)}"
        for c in report.clusters
    )


# ---------- Mediator: LLM path ----------


def mediate(
    llm: LLMClient,
    query: str,
    claims: list[Claim],
    reliability: dict[str, float],
) -> tuple[FinalAnswer, bool, float, int]:
    """Returns ``(answer, used_llm, cost_usd, n_calls)``."""
    report = detect_conflict(claims, reliability)

    # Deterministic path
    if report.winner is not None:
        return _deterministic_answer(report), False, 0.0, 0

    # Edge case: no supports clusters at all → abstain.
    if not report.clusters:
        return (
            FinalAnswer(
                variants=[],
                rejected_doc_ids=sorted({c.doc_id for c in claims}),
                abstained=True,
                explanation="No supports-claims; analyzer found no grounded answer in any doc.",
            ),
            False,
            0.0,
            0,
        )

    # LLM path
    system = _read_prompt("mediator.txt")
    user = (
        f"Question: {query}\n\n"
        f"Claims:\n{_format_claims_for_mediator(claims, reliability)}\n\n"
        f"Cluster summary (weighted-vote inconclusive — ratio={report.ratio:.2f}):\n"
        f"{_format_clusters(report)}\n"
    )
    out = llm.complete_json(
        system=system, user=user, schema=_MEDIATOR_SCHEMA, schema_name="MediatorOutput"
    )
    parsed = out.parsed
    answer = FinalAnswer(
        variants=[
            AnswerVariant(
                answer=parsed["answer"],
                confidence=float(parsed["confidence"]),
                supporting_doc_ids=list(parsed["supporting_doc_ids"]),
            )
        ],
        rejected_doc_ids=list(parsed["rejected_doc_ids"]),
        explanation=parsed["reconciliation_explanation"],
    )
    return answer, True, out.cost_usd, 1
