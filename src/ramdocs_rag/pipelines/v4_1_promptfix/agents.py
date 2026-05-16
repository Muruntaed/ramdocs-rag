"""Agents for v4: analyzer + NEW Evidence Evaluator + intra-mediator + Skeptic.

The Evaluator is the architectural addition vs v3. It runs alongside the
analyzer on every retrieved document and emits a `DocTrust` report. The
trust score lands in `reliability` via the W_TRUST slot, and the structured
`red_flags` list is surfaced to the Skeptic for downstream auditing.
"""

from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ramdocs_rag.core.llm import LLMClient
from ramdocs_rag.core.safety import apply_safety
from ramdocs_rag.core.types import AnswerVariant, Claim, FinalAnswer, RetrievedDoc

_PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
INTRA_MAJORITY_RATIO = 1.5


@lru_cache(maxsize=4)
def _read_prompt(name: str) -> str:
    raw = (_PROMPT_DIR / name).read_text(encoding="utf-8")
    return apply_safety(raw, name)


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

_RED_FLAGS = [
    "short_stub",
    "self_contradiction",
    "off_topic",
    "no_specifics",
    "category_page",
    "formatting_cruft",
]

_DOCTRUST_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "doc_id",
        "internal_consistency",
        "encyclopedic_quality",
        "specificity",
        "relevance",
        "trust_score",
        "red_flags",
    ],
    "properties": {
        "doc_id": {"type": "string"},
        "internal_consistency": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "encyclopedic_quality": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "specificity": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "relevance": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "trust_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "red_flags": {"type": "array", "items": {"type": "string", "enum": _RED_FLAGS}},
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


class DocTrust(BaseModel):
    model_config = ConfigDict(extra="forbid")

    doc_id: str
    internal_consistency: float = Field(ge=0.0, le=1.0)
    encyclopedic_quality: float = Field(ge=0.0, le=1.0)
    specificity: float = Field(ge=0.0, le=1.0)
    relevance: float = Field(ge=0.0, le=1.0)
    trust_score: float = Field(ge=0.0, le=1.0)
    red_flags: list[
        Literal[
            "short_stub",
            "self_contradiction",
            "off_topic",
            "no_specifics",
            "category_page",
            "formatting_cruft",
        ]
    ] = Field(default_factory=list)


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
    return Claim.model_validate(parsed), out.cost_usd, 1


def evaluate_doc(llm: LLMClient, query: str, doc: RetrievedDoc) -> tuple[DocTrust, float, int]:
    """Score one document on internal qualities (no cross-doc reasoning)."""
    system = _read_prompt("evaluator.txt")
    user = f"Question: {query}\n\nDocument id: {doc.doc_id}\nDocument text:\n---\n{doc.text}\n---\n"
    out = llm.complete_json(
        system=system, user=user, schema=_DOCTRUST_SCHEMA, schema_name="DocTrust"
    )
    parsed = dict(out.parsed)
    parsed["doc_id"] = doc.doc_id
    return DocTrust.model_validate(parsed), out.cost_usd, 1


def norm_text(t: str) -> str:
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
        by_text[norm_text(c.text)].append(c)
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
    losers = [c for c in claims if norm_text(c.text) != top_t]
    winners.sort(key=lambda c: reliability.get(c.doc_id, 0.0), reverse=True)
    return winners[0].text, [w.doc_id for w in winners], [loser.doc_id for loser in losers]


def _intra_group_llm(llm, query, entity, claims, reliability):
    system = _read_prompt("mediator.txt")
    lines = []
    for c in claims:
        lines.append(
            f"- doc_id={c.doc_id} text={c.text!r} "
            f"confidence={c.confidence:.2f} reliability={reliability.get(c.doc_id, 0.0):.3f}\n"
            f"  quote: {c.supporting_quote!r}"
        )
    user = (
        f"Question: {query}\n\nEntity: {entity}\n\n"
        f"Conflicting claims (all about this entity):\n" + "\n".join(lines)
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


def resolve_entity_group(llm, query, entity, claims, reliability):
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


def skeptic_verify(
    llm: LLMClient,
    query: str,
    draft: FinalAnswer,
    retrieved: list[RetrievedDoc],
    reliability: dict[str, float],
    trust_by_doc: dict[str, float] | None = None,
    flags_by_doc: dict[str, list[str]] | None = None,
) -> tuple[FinalAnswer, list[dict], float, int]:
    if not draft.variants:
        return draft, [], 0.0, 0

    trust_by_doc = trust_by_doc or {}
    flags_by_doc = flags_by_doc or {}

    system = _read_prompt("skeptic.txt")
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
        flags = flags_by_doc.get(d.doc_id, [])
        flag_str = f" red_flags={flags}" if flags else ""
        pool_lines.append(
            f"- doc_id={d.doc_id} role={role} "
            f"reliability={reliability.get(d.doc_id, 0.0):.3f} "
            f"trust={trust_by_doc.get(d.doc_id, 0.5):.2f}{flag_str}\n"
            f"  text: {d.text!r}"
        )
    variant_lines = [
        f"- entity={v.entity!r}\n  answer={v.answer!r}\n"
        f"  confidence={v.confidence:.2f}\n  supporting_doc_ids={list(v.supporting_doc_ids)}"
        for v in draft.variants
    ]
    user = (
        f"Question: {query}\n\n"
        f"DRAFT FINAL ANSWER (variants to verify):\n" + "\n".join(variant_lines) + "\n\n"
        "DOCUMENT POOL (with role, reliability, trust and red_flags):\n" + "\n".join(pool_lines)
    )
    out = llm.complete_json(
        system=system, user=user, schema=_SKEPTIC_SCHEMA, schema_name="SkepticVerdicts"
    )
    decisions = list(out.parsed["decisions"])
    verdict_by_entity = {d["entity"]: d["verdict"] for d in decisions}

    kept: list[AnswerVariant] = []
    newly_rejected: set[str] = set()
    for v in draft.variants:
        if verdict_by_entity.get(v.entity or "", "keep") == "keep":
            kept.append(v)
        else:
            newly_rejected.update(v.supporting_doc_ids)

    merged_rejected = sorted(set(draft.rejected_doc_ids) | newly_rejected)
    surviving = {d for v in kept for d in v.supporting_doc_ids}
    merged_rejected = sorted(set(merged_rejected) - surviving)

    lines = [draft.explanation, "Skeptic decisions:"]
    for d in decisions:
        lines.append(f"  · [{d['verdict']}] {d['entity']!r}: {d['reason']}")
    return (
        FinalAnswer(
            variants=kept,
            rejected_doc_ids=merged_rejected,
            abstained=(len(kept) == 0),
            explanation="\n".join(lines),
        ),
        decisions,
        out.cost_usd,
        1,
    )
