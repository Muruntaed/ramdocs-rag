"""Tests for the shared safety/prompt-hardening layer.

Validates two things:

1. ``apply_safety`` produces the right block per role, is idempotent, and
   stays a pure-text transformation.
2. Every pipeline version (v1.0 … v4.1) loads its prompts through the
   safety layer — i.e. the safety marker appears in every rendered prompt
   the agents will actually send to the LLM.
"""

from __future__ import annotations

import importlib

import pytest

from ramdocs_rag.core.safety import (
    SAFETY_ANALYZER,
    SAFETY_EVALUATOR,
    SAFETY_MEDIATOR,
    SAFETY_SKEPTIC,
    _SAFETY_MARKER,
    apply_safety,
)

# ---------- unit: the safety helper itself ----------


def test_marker_present_in_every_role_block():
    for block in (SAFETY_ANALYZER, SAFETY_EVALUATOR, SAFETY_MEDIATOR, SAFETY_SKEPTIC):
        assert _SAFETY_MARKER in block


def test_role_blocks_include_their_subset_of_rules():
    # All roles get grounding + leak + schema lock
    for block in (SAFETY_ANALYZER, SAFETY_EVALUATOR, SAFETY_MEDIATOR, SAFETY_SKEPTIC):
        assert "GROUNDING" in block
        assert "PROMPT-INJECTION & LEAK REFUSAL" in block
        assert "OUTPUT-SCHEMA LOCK" in block
    # Off-topic refusal only on the user-facing role (mediator)
    assert "OFF-TOPIC REFUSAL" in SAFETY_MEDIATOR
    for block in (SAFETY_ANALYZER, SAFETY_EVALUATOR, SAFETY_SKEPTIC):
        assert "OFF-TOPIC REFUSAL" not in block


def test_apply_safety_appends_block_after_original_text():
    original = "Do the analyzer thing.\n"
    out = apply_safety(original, "analyzer.txt")
    assert out.startswith(original)
    assert _SAFETY_MARKER in out
    assert out.endswith("\n")


def test_apply_safety_handles_missing_trailing_newline():
    original = "no newline at end"
    out = apply_safety(original, "mediator.txt")
    # Original text is preserved verbatim, then a separator, then the block.
    assert out.startswith(original)
    assert _SAFETY_MARKER in out


def test_apply_safety_is_idempotent():
    original = "Mediator prompt body."
    once = apply_safety(original, "mediator.txt")
    twice = apply_safety(once, "mediator.txt")
    assert once == twice
    # Marker must appear exactly once after a second pass.
    assert twice.count(_SAFETY_MARKER) == 1


def test_apply_safety_rejects_unknown_role():
    with pytest.raises(KeyError):
        apply_safety("x", "unknown.txt")


def test_apply_safety_uses_role_specific_block():
    analyzer_out = apply_safety("body", "analyzer.txt")
    mediator_out = apply_safety("body", "mediator.txt")
    # Only mediator should contain the off-topic clause.
    assert "OFF-TOPIC REFUSAL" not in analyzer_out
    assert "OFF-TOPIC REFUSAL" in mediator_out


# ---------- integration: every pipeline version wires safety in ----------

# (version_module, set_of_prompt_filenames) — covers every version in
# pipelines/. If a new vN_M_* is added without registering its prompts here
# the suite will start failing on the next pipeline's _read_prompt call.
_PIPELINE_AGENTS = [
    ("ramdocs_rag.pipelines.v1_0_madam_lite.agents", {"analyzer.txt", "mediator.txt"}),
    ("ramdocs_rag.pipelines.v2_0_entity_first.agents", {"analyzer.txt", "mediator.txt"}),
    (
        "ramdocs_rag.pipelines.v3_0_skeptic.agents",
        {"analyzer.txt", "mediator.txt", "skeptic.txt"},
    ),
    (
        "ramdocs_rag.pipelines.v3_1_conservative_skeptic.agents",
        {"analyzer.txt", "mediator.txt", "skeptic.txt"},
    ),
    (
        "ramdocs_rag.pipelines.v3_2_skeptic_balanced.agents",
        {"analyzer.txt", "mediator.txt", "skeptic.txt"},
    ),
    (
        "ramdocs_rag.pipelines.v3_3_analyzer_tuned.agents",
        {"analyzer.txt", "mediator.txt", "skeptic.txt"},
    ),
    (
        "ramdocs_rag.pipelines.v4_0_evidence_quality.agents",
        {"analyzer.txt", "evaluator.txt", "mediator.txt", "skeptic.txt"},
    ),
    (
        "ramdocs_rag.pipelines.v4_1_promptfix.agents",
        {"analyzer.txt", "evaluator.txt", "mediator.txt", "skeptic.txt"},
    ),
]


@pytest.mark.parametrize("module_path, prompt_names", _PIPELINE_AGENTS)
def test_every_pipeline_renders_prompts_with_safety_block(module_path, prompt_names):
    mod = importlib.import_module(module_path)
    read_prompt = mod._read_prompt
    # Clear the per-module lru_cache so we observe the wired behaviour.
    read_prompt.cache_clear()
    for name in prompt_names:
        rendered = read_prompt(name)
        assert _SAFETY_MARKER in rendered, (
            f"{module_path}::{name} rendered prompt is missing the safety marker; "
            "the version probably bypasses apply_safety in its _read_prompt."
        )
        # The original prompt body must come first; safety block sits at the end.
        marker_pos = rendered.index(_SAFETY_MARKER)
        assert marker_pos > 0, (
            f"{module_path}::{name} starts with the safety block — the original "
            "prompt body was lost. apply_safety must append, not prepend."
        )


def test_mediator_prompts_get_offtopic_clause_specifically():
    # Spot-check: every mediator.txt in every version pulls in the
    # mediator-specific block (which is the only one with OFF-TOPIC REFUSAL).
    for module_path, prompt_names in _PIPELINE_AGENTS:
        if "mediator.txt" not in prompt_names:
            continue
        mod = importlib.import_module(module_path)
        mod._read_prompt.cache_clear()
        rendered = mod._read_prompt("mediator.txt")
        assert "OFF-TOPIC REFUSAL" in rendered, (
            f"{module_path}::mediator.txt did not receive the mediator-specific "
            "safety block (off-topic clause missing)."
        )
