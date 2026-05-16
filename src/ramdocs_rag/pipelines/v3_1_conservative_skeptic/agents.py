"""Agents for v3.0: analyzer + intra-entity mediator + Skeptic.

v3.0 is v2.0 plus a Skeptic agent that runs after the mediator. The Skeptic
sees every variant of the draft answer together with the full document pool
and decides which variants survive.
"""

from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from pathlib import Path

from ramdocs_rag.core.llm import LLMClient
from ramdocs_rag.core.safety import apply_safety
from ramdocs_rag.core.types import AnswerVariant, Claim, FinalAnswer, RetrievedDoc

_PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
INTRA_MAJORITY_RATIO = 1.5


@lru_cache(maxsize=3)
def _read_prompt(name: str) -> str:
    raw = (_PROMPT_DIR / name).read_text(encoding="utf-8")
    return apply_safety(raw, name)


# ---------- JSON schemas ----------

_CLAIM_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["doc_id", "entity", "text", "stance", "confidence", "supporting_quote"],
    "properties": {
        "doc_id": {"type": "string"},
        "entity": {"type": "string"},
        "text": {"type": "string"},
        "stance": {"type": "string", "enum": ["supports", "contradicts", "no_answer"]},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "supporting_quote": {"type": "string"},
    },
}

_INTRA_MEDIATOR_SCHEMA = {
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

_SKEPTIC_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["decisions"],
    "properties": {
        "decisions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["entity", "verdict", "reason"],
                "properties": {
                    "entity": {"type": "string"},
                    "verdict": {"type": "string", "enum": ["keep", "reject"]},
                    "reason": {"type": "string"},
                },
            },
        },
    },
}


# ---------- Analyzer ----------


def analyze_doc(llm: LLMClient, query: str, doc: RetrievedDoc) -> tuple[Claim, float, int]:
    system = _read_prompt("analyzer.txt")
    user = f"Question: {query}\n\nDocument id: {doc.doc_id}\nDocument text:\n---\n{doc.text}\n---\n"
    out = llm.complete_json(
        system=system, user=user, schema=_CLAIM_SCHEMA, schema_name="EntityClaim"
    )
    parsed = dict(out.parsed)
    parsed["doc_id"] = doc.doc_id
    if parsed["stance"] != "supports":
        parsed.setdefault("entity", "")
    claim = Claim.model_validate(parsed)
    return claim, out.cost_usd, 1


# ---------- Intra-group resolution ----------


def _norm_text(t: str) -> str:
    return " ".join(t.lower().split())


def _intra_group_deterministic(
    claims: list[Claim], reliability: dict[str, float]
) -> tuple[str, list[str], list[str]] | None:
    if not claims:
        return None
    if len(claims) == 1:
        c = claims[0]
        return c.text, [c.doc_id], []

    by_text: dict[str, list[Claim]] = defaultdict(list)
    for c in claims:
        by_text[_norm_text(c.text)].append(c)

    if len(by_text) == 1:
        members = next(iter(by_text.values()))
        members.sort(key=lambda c: reliability.get(c.doc_id, 0.0), reverse=True)
        return members[0].text, [m.doc_id for m in members], []

    text_weights = {t: sum(reliability.get(c.doc_id, 0.0) for c in cs) for t, cs in by_text.items()}
    sorted_texts = sorted(text_weights.items(), key=lambda x: -x[1])
    top_t, top_w = sorted_texts[0]
    runner_w = sorted_texts[1][1] if len(sorted_texts) > 1 else 0.0
    ratio = top_w / runner_w if runner_w > 0 else float("inf")
    if ratio < INTRA_MAJORITY_RATIO:
        return None

    winners = by_text[top_t]
    losers = [c for c in claims if _norm_text(c.text) != top_t]
    winners.sort(key=lambda c: reliability.get(c.doc_id, 0.0), reverse=True)
    return winners[0].text, [w.doc_id for w in winners], [loser.doc_id for loser in losers]


def _intra_group_llm(
    llm: LLMClient,
    query: str,
    entity: str,
    claims: list[Claim],
    reliability: dict[str, float],
) -> tuple[AnswerVariant, list[str], float, int]:
    system = _read_prompt("mediator.txt")
    claim_lines = []
    for c in claims:
        claim_lines.append(
            f"- doc_id={c.doc_id} text={c.text!r} "
            f"confidence={c.confidence:.2f} reliability={reliability.get(c.doc_id, 0.0):.3f}\n"
            f"  quote: {c.supporting_quote!r}"
        )
    user = (
        f"Question: {query}\n\n"
        f"Entity: {entity}\n\n"
        f"Conflicting claims (all about this entity):\n" + "\n".join(claim_lines)
    )
    out = llm.complete_json(
        system=system, user=user, schema=_INTRA_MEDIATOR_SCHEMA, schema_name="IntraEntityMediator"
    )
    p = out.parsed
    variant = AnswerVariant(
        answer=p["answer"],
        confidence=float(p["confidence"]),
        supporting_doc_ids=list(p["supporting_doc_ids"]),
        entity=entity,
    )
    return variant, list(p["rejected_doc_ids"]), out.cost_usd, 1


def resolve_entity_group(
    llm: LLMClient,
    query: str,
    entity: str,
    claims: list[Claim],
    reliability: dict[str, float],
) -> tuple[AnswerVariant, list[str], float, int]:
    det = _intra_group_deterministic(claims, reliability)
    if det is not None:
        text, supporting, rejected = det
        avg_rel = sum(reliability.get(d, 0.0) for d in supporting) / max(1, len(supporting))
        variant = AnswerVariant(
            answer=text,
            confidence=min(1.0, max(0.1, avg_rel)),
            supporting_doc_ids=supporting,
            entity=entity,
        )
        return variant, rejected, 0.0, 0
    return _intra_group_llm(llm, query, entity, claims, reliability)


# ---------- Skeptic (the NEW agent in v3.0) ----------


def skeptic_verify(
    llm: LLMClient,
    query: str,
    draft: FinalAnswer,
    retrieved: list[RetrievedDoc],
    reliability: dict[str, float],
) -> tuple[FinalAnswer, list[dict], float, int]:
    """Run a single-pass Skeptic verification over draft.variants.

    Returns (verified FinalAnswer, raw decisions list, cost, n_calls).

    Skeptic is skipped (0 calls) when draft has 0 variants or all variants
    are unanimous and the pool is tiny — but for simplicity we ALWAYS run it
    when there is ≥ 1 variant. Cost: 1 LLM call regardless of variant count.
    """
    if not draft.variants:
        return draft, [], 0.0, 0

    system = _read_prompt("skeptic.txt")

    # Build a compact corpus view: doc_id, reliability, full text, current role.
    supporting_set = set(draft.all_supporting_doc_ids)
    rejected_set = set(draft.rejected_doc_ids)
    pool_lines = []
    for d in retrieved:
        role = (
            "supporting"
            if d.doc_id in supporting_set
            else "rejected"
            if d.doc_id in rejected_set
            else "neutral"
        )
        pool_lines.append(
            f"- doc_id={d.doc_id} role={role} reliability={reliability.get(d.doc_id, 0.0):.3f}\n"
            f"  text: {d.text!r}"
        )

    variant_lines = []
    for v in draft.variants:
        variant_lines.append(
            f"- entity={v.entity!r}\n"
            f"  answer={v.answer!r}\n"
            f"  confidence={v.confidence:.2f}\n"
            f"  supporting_doc_ids={list(v.supporting_doc_ids)}"
        )

    user = (
        f"Question: {query}\n\n"
        f"DRAFT FINAL ANSWER (variants to verify):\n" + "\n".join(variant_lines) + "\n\n"
        "DOCUMENT POOL (every retrieved document, with role and reliability):\n"
        + "\n".join(pool_lines)
    )
    out = llm.complete_json(
        system=system, user=user, schema=_SKEPTIC_SCHEMA, schema_name="SkepticVerdicts"
    )
    decisions = list(out.parsed["decisions"])

    # Apply decisions
    verdict_by_entity: dict[str, str] = {}
    for d in decisions:
        verdict_by_entity[d["entity"]] = d["verdict"]

    kept: list[AnswerVariant] = []
    newly_rejected_docs: set[str] = set()
    for v in draft.variants:
        verdict = verdict_by_entity.get(v.entity or "", "keep")
        if verdict == "keep":
            kept.append(v)
        else:
            newly_rejected_docs.update(v.supporting_doc_ids)

    # Compose verified FinalAnswer
    merged_rejected = sorted(set(draft.rejected_doc_ids) | newly_rejected_docs)
    # remove anything still in surviving supporting from rejected
    surviving_supp = {d for v in kept for d in v.supporting_doc_ids}
    merged_rejected = sorted(set(merged_rejected) - surviving_supp)

    explanation_lines = [draft.explanation, "Skeptic decisions:"]
    for d in decisions:
        explanation_lines.append(f"  · [{d['verdict']}] {d['entity']!r}: {d['reason']}")
    new_explanation = "\n".join(explanation_lines)

    verified = FinalAnswer(
        variants=kept,
        rejected_doc_ids=merged_rejected,
        abstained=(len(kept) == 0),
        explanation=new_explanation,
    )
    return verified, decisions, out.cost_usd, 1
