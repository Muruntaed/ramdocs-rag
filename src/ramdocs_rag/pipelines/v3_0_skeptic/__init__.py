"""v3.0 — Entity-First + Skeptic verification agent.

Builds on v2.0 by adding a final Skeptic stage that sees every draft variant
together with the full document pool and decides keep/reject for each.
Goal: recover the misinfo-rejection and citation-faithfulness regressions
that surfaced in v2.0 without giving up the v2.0 EM/Recall/cost gains.
"""

from .pipeline import V3Skeptic

__all__ = ["V3Skeptic"]
