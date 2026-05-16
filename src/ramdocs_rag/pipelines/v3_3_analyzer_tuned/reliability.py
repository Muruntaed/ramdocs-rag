"""5-factor reliability formula used through v3.

The ``minority_penalty`` is applied INSIDE an entity group, not globally
(as it was in v1). The four positive weights match the v1.0 baseline;
they are split between retrieval, analyzer confidence and two dormant
slots (recency / authority) which default to 0.5 on RAMDocs because the
corpus carries no per-document dates or source-authority labels::

  reliability(doc) =
        0.40 * retrieval_score
      + 0.25 * confidence
      + 0.20 * recency             (dormant, default 0.5)
      + 0.15 * authority           (dormant, default 0.5)
      - 0.10 * minority_penalty    (1.0 if the claim is an intra-group loser)

v4 collapses the two dormant slots into a single ``W_TRUST = 0.35``
populated by the new Evidence Evaluator agent.
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


def score_doc(doc: RetrievedDoc, claim: Claim | None, *, in_minority: bool) -> float:
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


def initial_reliability(docs: list[RetrievedDoc], claims: list[Claim]) -> dict[str, float]:
    """First pass (no penalty) — needed for intra-group voting."""
    claim_by_doc = {c.doc_id: c for c in claims}
    return {d.doc_id: score_doc(d, claim_by_doc.get(d.doc_id), in_minority=False) for d in docs}


def final_reliability(
    docs: list[RetrievedDoc],
    claims: list[Claim],
    minority_doc_ids: set[str],
) -> dict[str, float]:
    """Final pass — applies the penalty to intra-group losers."""
    claim_by_doc = {c.doc_id: c for c in claims}
    return {
        d.doc_id: score_doc(d, claim_by_doc.get(d.doc_id), in_minority=d.doc_id in minority_doc_ids)
        for d in docs
    }
