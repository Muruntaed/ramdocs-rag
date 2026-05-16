"""5-factor reliability formula (introduced in v2.0).

Same numerical formula as v1.0, but the ``minority_penalty`` is applied
INSIDE an entity group (not globally) so that intra-group losers are
the only ones penalised::

  reliability(doc) =
        0.40 * retrieval_score
      + 0.25 * confidence
      + 0.20 * recency             (dormant, default 0.5; RAMDocs has no dates)
      + 0.15 * authority           (dormant, default 0.5)
      - 0.10 * minority_penalty    (1.0 if the claim is an intra-group loser)
"""

from __future__ import annotations

from ramdocs_rag.core.types import Claim, RetrievedDoc

W_RETRIEVAL = 0.40
W_CONFIDENCE = 0.25
W_RECENCY = 0.20
W_AUTHORITY = 0.15
W_MINORITY_PENALTY = 0.10
DEFAULT_RECENCY = 0.5
DEFAULT_AUTHORITY = 0.5


def score_doc(
    doc: RetrievedDoc, claim: Claim | None, *, in_minority: bool
) -> float:
    conf = claim.confidence if claim is not None else 0.5
    penalty = 1.0 if in_minority else 0.0
    s = (
        W_RETRIEVAL * doc.score
        + W_CONFIDENCE * conf
        + W_RECENCY * DEFAULT_RECENCY
        + W_AUTHORITY * DEFAULT_AUTHORITY
        - W_MINORITY_PENALTY * penalty
    )
    return max(0.0, min(1.0, s))


def initial_reliability(
    docs: list[RetrievedDoc], claims: list[Claim]
) -> dict[str, float]:
    """First pass (no penalty) — needed for intra-group voting."""
    claim_by_doc = {c.doc_id: c for c in claims}
    return {
        d.doc_id: score_doc(d, claim_by_doc.get(d.doc_id), in_minority=False)
        for d in docs
    }


def final_reliability(
    docs: list[RetrievedDoc],
    claims: list[Claim],
    minority_doc_ids: set[str],
) -> dict[str, float]:
    """Final pass — applies the penalty to intra-group losers."""
    claim_by_doc = {c.doc_id: c for c in claims}
    return {
        d.doc_id: score_doc(
            d, claim_by_doc.get(d.doc_id), in_minority=d.doc_id in minority_doc_ids
        )
        for d in docs
    }
