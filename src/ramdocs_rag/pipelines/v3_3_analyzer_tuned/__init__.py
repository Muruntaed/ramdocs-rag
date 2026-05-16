"""v3.3 — v3 with an analyzer prompt tuned for multi-candidate disambiguation.

Minor over v3.2. The pipeline architecture, mediator and Skeptic are
identical to v3.2; the only change is the analyzer prompt, which now
explicitly tells the model to pick the value that DIRECTLY answers the
question (not just any salient value mentioned in the document). Targets
v3.2 errors on q306 (wrong-number extraction) and q094 (multi-candidate
doc).
"""

from .pipeline import V33AnalyzerTuned

__all__ = ["V33AnalyzerTuned"]
