"""v3.1 — Conservative Skeptic + abstention fallback.

Minor iteration over v3.0:
- Skeptic prompt explicitly handles homonym/namesake cases (the v3.0 failure
  mode on q131 and q113 where two valid same-named entities triggered total
  rejection).
- Pipeline-level safeguard: if Skeptic rejects every variant while the draft
  had at least one, restore the draft instead of abstaining.
"""

from .pipeline import V31ConservativeSkeptic

__all__ = ["V31ConservativeSkeptic"]
