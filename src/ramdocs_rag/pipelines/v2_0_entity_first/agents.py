"""v2.0 agents: analyzer (with a mandatory entity) and intra-entity mediator.

Flow::

  analyze_doc          one document → a Claim with a mandatory ``entity``
                       field (for ``supports`` claims).
  resolve_entity_group inside a single entity group:
                       - one claim → take it;
                       - all claims agree (on the normalised text) → take it;
                       - partial agreement (top weight ≥ 1.5 × runner-up)
                         → deterministic winner;
                       - otherwise → LLM mediator.
"""

from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from pathlib import Path

from ramdocs_rag.core.llm import LLMClient
from ramdocs_rag.core.safety import apply_safety
from ramdocs_rag.core.types import AnswerVariant, Claim, RetrievedDoc

_PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
INTRA_MAJORITY_RATIO = 1.5


@lru_cache(maxsize=2)
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
        "answer", "confidence", "supporting_doc_ids", "rejected_doc_ids",
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


def analyze_doc(
    llm: LLMClient, query: str, doc: RetrievedDoc
) -> tuple[Claim, float, int]:
    """One document → one ``Claim`` with an entity. Returns ``(claim, cost, calls=1)``."""
    system = _read_prompt("analyzer.txt")
    user = (
        f"Question: {query}\n\n"
        f"Document id: {doc.doc_id}\n"
        f"Document text:\n---\n{doc.text}\n---\n"
    )
    out = llm.complete_json(
        system=system, user=user, schema=_CLAIM_SCHEMA, schema_name="EntityClaim"
    )
    parsed = dict(out.parsed)
    parsed["doc_id"] = doc.doc_id  # guard against hallucinated id
    # Enforce entity-stance consistency: no_answer ⇒ entity may be empty.
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
    """If the group has a deterministic winner, return
    ``(answer_text, supporting_doc_ids, rejected_doc_ids)``; otherwise
    return ``None`` (meaning the caller must invoke the LLM mediator).

    Logic:
      - 1 claim → winner.
      - All answers agree (after ``_norm_text``) → winner.
      - Otherwise: weighted vote by reliability; requires
        ``top_weight / runner_up_weight ≥ 1.5``.
    """
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

    text_weights = {
        t: sum(reliability.get(c.doc_id, 0.0) for c in cs)
        for t, cs in by_text.items()
    }
    sorted_texts = sorted(text_weights.items(), key=lambda x: -x[1])
    top_t, top_w = sorted_texts[0]
    runner_w = sorted_texts[1][1] if len(sorted_texts) > 1 else 0.0
    ratio = top_w / runner_w if runner_w > 0 else float("inf")
    if ratio < INTRA_MAJORITY_RATIO:
        return None

    winners = by_text[top_t]
    losers = [c for c in claims if _norm_text(c.text) != top_t]
    winners.sort(key=lambda c: reliability.get(c.doc_id, 0.0), reverse=True)
    return winners[0].text, [w.doc_id for w in winners], [l.doc_id for l in losers]


def _intra_group_llm(
    llm: LLMClient,
    query: str,
    entity: str,
    claims: list[Claim],
    reliability: dict[str, float],
) -> tuple[AnswerVariant, list[str], float, int]:
    """LLM-driven intra-group conflict resolution.

    Returns ``(AnswerVariant, rejected_doc_ids, cost, calls=1)``.
    """
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
        f"Conflicting claims (all about this entity):\n"
        + "\n".join(claim_lines)
    )
    out = llm.complete_json(
        system=system, user=user, schema=_INTRA_MEDIATOR_SCHEMA,
        schema_name="IntraEntityMediator",
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
    """Resolve a single entity group. Returns ``(variant, rejected, cost, calls)``.

    Tries the deterministic path first; falls back to the LLM mediator
    when there is no clear winner.
    """
    det = _intra_group_deterministic(claims, reliability)
    if det is not None:
        text, supporting, rejected = det
        # confidence = average reliability of the winners, clipped to [0.1, 1.0].
        avg_rel = (
            sum(reliability.get(d, 0.0) for d in supporting) / max(1, len(supporting))
        )
        variant = AnswerVariant(
            answer=text,
            confidence=min(1.0, max(0.1, avg_rel)),
            supporting_doc_ids=supporting,
            entity=entity,
        )
        return variant, rejected, 0.0, 0

    return _intra_group_llm(llm, query, entity, claims, reliability)
