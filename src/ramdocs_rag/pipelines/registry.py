"""Register every known pipeline with the runner's registry.

This module is imported by ``eval.runner._ensure_registry_loaded`` and
the import itself is the only side-effect (the ``register(...)`` calls).

Every new pipeline version is wired up here with a single ``register`` call.
"""

from __future__ import annotations

from ..eval.runner import register
from .v1_0_madam_lite import V1MadamLite
from .v2_0_entity_first import V2EntityFirst


def _make_v1_madam_lite() -> V1MadamLite:
    """Production factory: analyzer on ``gpt-4o-mini``, mediator on ``gpt-4o``.

    Models are read from env (``OPENAI_MODEL_ANALYZER`` /
    ``OPENAI_MODEL_MEDIATOR``) with defaults matching the legacy prototype.
    Tests do not use this factory — they instantiate the pipeline directly
    with ``MockLLM``.
    """
    import os

    from ..core.llm import OpenAIClient

    analyzer = OpenAIClient(model=os.environ.get("OPENAI_MODEL_ANALYZER", "gpt-4o-mini"))
    mediator = OpenAIClient(model=os.environ.get("OPENAI_MODEL_MEDIATOR", "gpt-4o"))
    return V1MadamLite(llm=analyzer, mediator_llm=mediator)


register("v1.0_madam_lite", _make_v1_madam_lite)


def _make_v2_entity_first() -> V2EntityFirst:
    """v2.0: entity-first decomposition, multi-answer aware."""
    import os
    from ..core.llm import OpenAIClient

    analyzer = OpenAIClient(model=os.environ.get("OPENAI_MODEL_ANALYZER", "gpt-4o-mini"))
    mediator = OpenAIClient(model=os.environ.get("OPENAI_MODEL_MEDIATOR", "gpt-4o"))
    return V2EntityFirst(llm=analyzer, mediator_llm=mediator)


register("v2.0_entity_first", _make_v2_entity_first)


def _make_v3_skeptic():
    """v3.0: v2.0 + Skeptic verification agent (single-pass)."""
    import os

    from ..core.llm import OpenAIClient
    from .v3_0_skeptic import V3Skeptic

    analyzer = OpenAIClient(model=os.environ.get("OPENAI_MODEL_ANALYZER", "gpt-4o-mini"))
    mediator = OpenAIClient(model=os.environ.get("OPENAI_MODEL_MEDIATOR", "gpt-4o"))
    # The Skeptic shares the mediator's stronger model by default.
    skeptic = OpenAIClient(model=os.environ.get("OPENAI_MODEL_MEDIATOR", "gpt-4o"))
    return V3Skeptic(llm=analyzer, mediator_llm=mediator, skeptic_llm=skeptic)


register("v3.0_skeptic", _make_v3_skeptic)


def _make_v3_1_conservative_skeptic():
    """v3.1: v3.0 + conservative Skeptic prompt + abstention fallback."""
    import os

    from ..core.llm import OpenAIClient
    from .v3_1_conservative_skeptic import V31ConservativeSkeptic

    analyzer = OpenAIClient(model=os.environ.get("OPENAI_MODEL_ANALYZER", "gpt-4o-mini"))
    mediator = OpenAIClient(model=os.environ.get("OPENAI_MODEL_MEDIATOR", "gpt-4o"))
    skeptic = OpenAIClient(model=os.environ.get("OPENAI_MODEL_MEDIATOR", "gpt-4o"))
    return V31ConservativeSkeptic(llm=analyzer, mediator_llm=mediator, skeptic_llm=skeptic)


register("v3.1_conservative_skeptic", _make_v3_1_conservative_skeptic)


def _make_v3_2_skeptic_balanced():
    """v3.2: balanced Skeptic (homonyms + depth-asymmetry) + threshold 0.30."""
    import os

    from ..core.llm import OpenAIClient
    from .v3_2_skeptic_balanced import V32SkepticBalanced

    analyzer = OpenAIClient(model=os.environ.get("OPENAI_MODEL_ANALYZER", "gpt-4o-mini"))
    mediator = OpenAIClient(model=os.environ.get("OPENAI_MODEL_MEDIATOR", "gpt-4o"))
    skeptic = OpenAIClient(model=os.environ.get("OPENAI_MODEL_MEDIATOR", "gpt-4o"))
    return V32SkepticBalanced(llm=analyzer, mediator_llm=mediator, skeptic_llm=skeptic)


register("v3.2_skeptic_balanced", _make_v3_2_skeptic_balanced)


def _make_v3_3_analyzer_tuned():
    """v3.3: v3.2 + analyzer prompt tuned for multi-candidate disambiguation."""
    import os

    from ..core.llm import OpenAIClient
    from .v3_3_analyzer_tuned import V33AnalyzerTuned

    analyzer = OpenAIClient(model=os.environ.get("OPENAI_MODEL_ANALYZER", "gpt-4o-mini"))
    mediator = OpenAIClient(model=os.environ.get("OPENAI_MODEL_MEDIATOR", "gpt-4o"))
    skeptic = OpenAIClient(model=os.environ.get("OPENAI_MODEL_MEDIATOR", "gpt-4o"))
    return V33AnalyzerTuned(llm=analyzer, mediator_llm=mediator, skeptic_llm=skeptic)


register("v3.3_analyzer_tuned", _make_v3_3_analyzer_tuned)


def _make_v4_evidence_quality():
    """v4.0: v3 + per-doc Evidence Evaluator + trust-weighted reliability."""
    import os

    from ..core.llm import OpenAIClient
    from .v4_0_evidence_quality import V4EvidenceQuality

    analyzer = OpenAIClient(model=os.environ.get("OPENAI_MODEL_ANALYZER", "gpt-4o-mini"))
    # Evaluator on the cheap model — its job is local style/consistency rating.
    evaluator = OpenAIClient(model=os.environ.get("OPENAI_MODEL_ANALYZER", "gpt-4o-mini"))
    mediator = OpenAIClient(model=os.environ.get("OPENAI_MODEL_MEDIATOR", "gpt-4o"))
    skeptic = OpenAIClient(model=os.environ.get("OPENAI_MODEL_MEDIATOR", "gpt-4o"))
    return V4EvidenceQuality(
        llm=analyzer, evaluator_llm=evaluator,
        mediator_llm=mediator, skeptic_llm=skeptic,
    )


register("v4.0_evidence_quality", _make_v4_evidence_quality)


def _make_v4_1_promptfix():
    """v4.1: v4.0 code unchanged, all four prompts rewritten."""
    import os

    from ..core.llm import OpenAIClient
    from .v4_1_promptfix import V41PromptFix

    analyzer = OpenAIClient(model=os.environ.get("OPENAI_MODEL_ANALYZER", "gpt-4o-mini"))
    evaluator = OpenAIClient(model=os.environ.get("OPENAI_MODEL_ANALYZER", "gpt-4o-mini"))
    mediator = OpenAIClient(model=os.environ.get("OPENAI_MODEL_MEDIATOR", "gpt-4o"))
    skeptic = OpenAIClient(model=os.environ.get("OPENAI_MODEL_MEDIATOR", "gpt-4o"))
    return V41PromptFix(
        llm=analyzer, evaluator_llm=evaluator,
        mediator_llm=mediator, skeptic_llm=skeptic,
    )


register("v4.1_promptfix", _make_v4_1_promptfix)
