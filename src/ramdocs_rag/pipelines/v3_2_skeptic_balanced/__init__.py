"""v3.2 — Balanced Skeptic.

Minor over v3.1: the Skeptic prompt now uses a depth-asymmetry check on top
of the homonym tolerance, and the entity-group threshold is loosened
(0.40 → 0.30) to lift multi-answer recall. The pipeline-level abstention
fallback from v3.1 is kept.

Goal: combine v2.0's recall with v3.0's misinfo robustness — a balanced
operating point on the Pareto frontier.
"""

from .pipeline import V32SkepticBalanced

__all__ = ["V32SkepticBalanced"]
