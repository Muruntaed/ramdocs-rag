"""5-factor reliability formula (ported from the legacy prototype)::

reliability(doc) =
      0.40 * retrieval_score
    + 0.25 * self_confidence
    + 0.20 * recency            (exp-decay; default 0.5 without metadata)
    + 0.15 * source_authority   (whitelist; default 0.5)
    - 0.10 * minority_penalty   (1.0 if the doc is in the losing cluster)
"""

from __future__ import annotations

from dataclasses import dataclass

from ...core.types import Claim, RetrievedDoc
from .conflict import detect_conflict, minority_doc_ids

W_RETRIEVAL = 0.40
W_SELF_CONFIDENCE = 0.25
W_RECENCY = 0.20
W_AUTHORITY = 0.15
W_MINORITY_PENALTY = 0.10

# RAMDocs has no date / source metadata — recency and authority default to 0.5.
DEFAULT_RECENCY = 0.5
DEFAULT_AUTHORITY = 0.5


@dataclass(frozen=True)
class ReliabilityBreakdown:
    doc_id: str
    retrieval: float
    self_confidence: float
    recency: float
    authority: float
    minority_penalty: float
    score: float


def _score_one(
    doc: RetrievedDoc, claim: Claim | None, *, in_minority: bool
) -> ReliabilityBreakdown:
    self_conf = claim.confidence if claim is not None else 0.5
    penalty = 1.0 if in_minority else 0.0
    score = (
        W_RETRIEVAL * doc.score
        + W_SELF_CONFIDENCE * self_conf
        + W_RECENCY * DEFAULT_RECENCY
        + W_AUTHORITY * DEFAULT_AUTHORITY
        - W_MINORITY_PENALTY * penalty
    )
    return ReliabilityBreakdown(
        doc_id=doc.doc_id,
        retrieval=doc.score,
        self_confidence=self_conf,
        recency=DEFAULT_RECENCY,
        authority=DEFAULT_AUTHORITY,
        minority_penalty=penalty,
        score=max(0.0, min(1.0, score)),
    )


def compute_reliability(
    docs: list[RetrievedDoc], claims: list[Claim]
) -> tuple[dict[str, float], list[ReliabilityBreakdown]]:
    """Two-pass computation: provisional → cluster → final with penalty.

    Preserves the legacy behaviour: ``minority_penalty`` is computed
    AFTER the first pass, because clustering itself depends on
    reliability scores.
    """
    claim_by_doc = {c.doc_id: c for c in claims}

    # 1) Provisional scores without the minority penalty.
    provisional: dict[str, float] = {}
    for d in docs:
        b = _score_one(d, claim_by_doc.get(d.doc_id), in_minority=False)
        provisional[d.doc_id] = b.score

    # 2) Cluster + minority detection over provisional scores.
    report = detect_conflict(claims, provisional)
    minority = minority_doc_ids(report)

    # 3) Final pass.
    final: dict[str, float] = {}
    breakdowns: list[ReliabilityBreakdown] = []
    for d in docs:
        b = _score_one(d, claim_by_doc.get(d.doc_id), in_minority=d.doc_id in minority)
        final[d.doc_id] = b.score
        breakdowns.append(b)
    return final, breakdowns
