"""v4.1 — Evidence Quality Scoring · prompt-fix iteration of v4.0.

Same architecture and code as v4.0 (parallel Evaluator + DocTrust + Skeptic
seeing trust + red_flags). Only the four prompts are revised:

- evaluator.txt: conservative red_flags calibration (legitimate short
  encyclopedic entries no longer flagged as short_stub / no_specifics),
  trust_score floor 0.55 for any coherent short entry.
- analyzer.txt: three new rules — A1 category-mirror (q094/q131),
  A2 population main-line value (q306), A3 entity↔text same-sentence
  binding (q113); plus light entity canonicalization (q021/q176).
- mediator.txt: M1 disambig-coherence guard — reject claims whose text
  is incompatible with the entity disambiguator before promoting them.
- skeptic.txt: S1 disambig-faithfulness (sharper citation check),
  S2 agreement-is-not-proof (anti-coordinated-misinfo), explicit
  "trust / red_flags are ADVISORY only" rule.
"""

from .pipeline import V41PromptFix

__all__ = ["V41PromptFix"]
