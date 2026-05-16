"""v4.0 — Evidence Quality Scoring.

Major over v3: a per-document Evaluator agent runs alongside the Analyzer
and emits a structured DocTrust report (internal_consistency,
encyclopedic_quality, specificity, relevance, composite trust_score, plus
a closed-enum list of red_flags). The trust_score replaces the v1–v3
dormant recency/authority slots in the reliability formula (W_TRUST=0.35),
and the red_flags are surfaced to the Skeptic for sharper rejection.
"""

from .pipeline import V4EvidenceQuality

__all__ = ["V4EvidenceQuality"]
