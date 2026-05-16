"""Reliability formula for v4 — trust_score replaces the dormant
recency/authority slots used in v1–v3.

Old formula (v1–v3):
  reliability(doc) =
        0.40 * retrieval_score
      + 0.25 * analyzer_confidence
      + 0.20 * recency              (dormant: default 0.5 on RAMDocs)
      + 0.15 * source_authority     (dormant: default 0.5 on RAMDocs)
      - 0.10 * minority_penalty

New formula (v4):
  reliability(doc) =
        0.40 * retrieval_score
      + 0.25 * analyzer_confidence
      + 0.35 * doc_trust_score      (NEW: from the Evaluator agent)
      - 0.10 * minority_penalty

The 0.35 weight is the combined budget that was previously split between
recency and authority — 0.20 + 0.15 = 0.35. The composite trust score thus
preserves the overall scale of reliability(): when trust_score = 0.5 the
new formula yields exactly the same number as the v1–v3 formula did with
the default 0.5 dormants. Above-baseline trust documents are boosted,
suspicious ones (thin stubs, off-topic) are penalised numerically.
"""

from __future__ import annotations

from ramdocs_rag.core.types import Claim, RetrievedDoc

W_RETRIEVAL = 0.40
W_CONFIDENCE = 0.25
W_TRUST = 0.35
W_MINORITY_PENALTY = 0.10

DEFAULT_TRUST = 0.5  # when evaluator has no opinion (should not happen in practice)


def score_doc(
    doc: RetrievedDoc,
    claim: Claim | None,
    trust_score: float,
    *,
    in_minority: bool,
) -> float:
    conf = claim.confidence if claim is not None else 0.5
    penalty = 1.0 if in_minority else 0.0
    s = (
        W_RETRIEVAL * doc.score
        + W_CONFIDENCE * conf
        + W_TRUST * trust_score
        - W_MINORITY_PENALTY * penalty
    )
    return max(0.0, min(1.0, s))


def initial_reliability(
    docs: list[RetrievedDoc],
    claims: list[Claim],
    trust_by_doc: dict[str, float],
) -> dict[str, float]:
    """First pass — without intra-group minority penalty."""
    claim_by_doc = {c.doc_id: c for c in claims}
    return {
        d.doc_id: score_doc(
            d,
            claim_by_doc.get(d.doc_id),
            trust_by_doc.get(d.doc_id, DEFAULT_TRUST),
            in_minority=False,
        )
        for d in docs
    }


def final_reliability(
    docs: list[RetrievedDoc],
    claims: list[Claim],
    trust_by_doc: dict[str, float],
    minority_doc_ids: set[str],
) -> dict[str, float]:
    """Second pass — with intra-group minority penalty applied."""
    claim_by_doc = {c.doc_id: c for c in claims}
    return {
        d.doc_id: score_doc(
            d,
            claim_by_doc.get(d.doc_id),
            trust_by_doc.get(d.doc_id, DEFAULT_TRUST),
            in_minority=d.doc_id in minority_doc_ids,
        )
        for d in docs
    }
