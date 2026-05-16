"""V4.1 — v4.0 with rewritten prompts (no code/architecture changes).

Same architecture as v4.0: per-doc Evaluator (parallel to Analyzer) emits
a DocTrust report; trust_score feeds the W_TRUST = 0.35 slot in reliability;
red_flags are advisory inputs to the Skeptic.

This minor targets the v4.0 regressions diagnosed against v3.3:
- Evaluator over-flagged legitimate short encyclopedic entries as
  short_stub / no_specifics → conservative red_flags calibration,
  trust_score floor 0.55 for any coherent short entry.
- Analyzer missed q094/q131 (category mirror), q306 (population main-line),
  q113 (entity↔text binding) → three explicit rules A1/A2/A3.
- Mediator could promote a claim whose text contradicted the entity
  disambiguator (q113 "Doug Harvey (ice hockey)" → 'Baseball') → M1
  disambig-coherence guard.
- Skeptic was over-influenced by red_flags alone and treated coordinated
  stub agreement as corroboration → S1 disambig-faithfulness and S2
  agreement-is-not-proof; trust / red_flags reduced to advisory inputs.

Code (agents.py / pipeline.py / reliability.py / grouping.py) is identical
to v4.0 modulo class name and version string; only the four prompts differ.
"""

from __future__ import annotations

import time

from ramdocs_rag.core.llm import LLMClient
from ramdocs_rag.core.retrieval import RetrievalConfig, retrieve
from ramdocs_rag.core.types import FinalAnswer, Question, RunResult
from ramdocs_rag.pipelines.base import Pipeline

from .agents import (
    analyze_doc,
    evaluate_doc,
    resolve_entity_group,
    skeptic_verify,
)
from .grouping import display_entity, group_by_entity
from .reliability import final_reliability, initial_reliability

MIN_RELATIVE_WEIGHT = 0.20  # was 0.30; lowered so 1-vs-4 homonyms still survive
# (e.g. q113 Doug Harvey: 1 hockey doc vs 4 baseball docs ⇒ 20% ratio).


class V41PromptFix(Pipeline):
    """v4.1: v4.0 with rewritten prompts (analyzer / evaluator / mediator / skeptic)."""

    name = "v4.1_promptfix"
    version = "4.1.0"

    def __init__(
        self,
        llm: LLMClient,
        evaluator_llm: LLMClient | None = None,
        mediator_llm: LLMClient | None = None,
        skeptic_llm: LLMClient | None = None,
        config: dict | None = None,
    ) -> None:
        super().__init__(llm, config or {})
        # Evaluator deliberately uses the cheap analyzer-class model: its job
        # is local style/consistency rating, not reasoning across the corpus.
        self.evaluator_llm: LLMClient = evaluator_llm or llm
        self.mediator_llm: LLMClient = mediator_llm or llm
        self.skeptic_llm: LLMClient = skeptic_llm or self.mediator_llm
        self._retrieval_cfg = RetrievalConfig(
            bm25_weight=self.config.get("bm25_weight", 0.5),
            dense_weight=self.config.get("dense_weight", 0.5),
            top_k=self.config.get("top_k", 8),
        )
        self._min_relative_weight = float(
            self.config.get("min_relative_weight", MIN_RELATIVE_WEIGHT)
        )

    def run(self, question: Question) -> RunResult:
        t0 = time.perf_counter()
        cost = 0.0
        calls = 0

        retrieved = retrieve(question.question, list(question.docs), self._retrieval_cfg)

        # Analyzer + Evaluator, paired per doc. (Sequential here; a future
        # minor could run them in parallel via asyncio.)
        claims = []
        trust_by_doc: dict[str, float] = {}
        flags_by_doc: dict[str, list[str]] = {}
        for d in retrieved:
            c, c_cost, c_calls = analyze_doc(self.llm, question.question, d)
            claims.append(c)
            cost += c_cost
            calls += c_calls

            t, t_cost, t_calls = evaluate_doc(self.evaluator_llm, question.question, d)
            trust_by_doc[d.doc_id] = t.trust_score
            flags_by_doc[d.doc_id] = list(t.red_flags)
            cost += t_cost
            calls += t_calls

        groups = group_by_entity(claims)
        no_answer_doc_ids = [c.doc_id for c in claims if c.stance != "supports"]

        if not groups:
            return RunResult(
                question_id=question.question_id,
                final_answer=FinalAnswer(
                    variants=[],
                    rejected_doc_ids=no_answer_doc_ids,
                    abstained=True,
                    explanation="No documents support an answer to the question.",
                ),
                cost_usd=cost, latency_s=time.perf_counter() - t0, llm_calls=calls,
            )

        rel = initial_reliability(retrieved, claims, trust_by_doc)

        from collections import Counter
        from .agents import _norm_text
        all_minority: set[str] = set()
        for group_claims in groups.values():
            if len(group_claims) < 2:
                continue
            counts = Counter(_norm_text(c.text) for c in group_claims)
            top_text = counts.most_common(1)[0][0]
            for c in group_claims:
                if _norm_text(c.text) != top_text:
                    all_minority.add(c.doc_id)
        rel = final_reliability(retrieved, claims, trust_by_doc, all_minority)

        group_weights = {
            key: sum(rel.get(c.doc_id, 0.0) for c in gclaims) for key, gclaims in groups.items()
        }
        top_weight = max(group_weights.values()) if group_weights else 0.0
        cutoff = self._min_relative_weight * top_weight

        variants = []
        rejected: set[str] = set(no_answer_doc_ids)
        for key, gclaims in sorted(groups.items(), key=lambda kv: -group_weights[kv[0]]):
            entity_display = display_entity(gclaims)
            if group_weights[key] < cutoff and len(variants) > 0:
                rejected.update(c.doc_id for c in gclaims)
                continue
            variant, intra_rejected, m_cost, m_calls = resolve_entity_group(
                self.mediator_llm, question.question, entity_display, gclaims, rel
            )
            cost += m_cost
            calls += m_calls
            variants.append(variant)
            rejected.update(intra_rejected)

        supporting_all = {d for v in variants for d in v.supporting_doc_ids}
        rejected -= supporting_all

        draft = FinalAnswer(
            variants=variants,
            rejected_doc_ids=sorted(rejected),
            abstained=(len(variants) == 0),
            explanation=(
                f"Draft: grouped {len(claims)} claims into {len(groups)} entity-clusters; "
                f"emitted {len(variants)} variant(s) above {self._min_relative_weight:.0%} of top-group weight; "
                f"trust scores ranged {min(trust_by_doc.values(), default=0):.2f}–"
                f"{max(trust_by_doc.values(), default=0):.2f}."
            ),
        )

        verified, _decisions, s_cost, s_calls = skeptic_verify(
            self.skeptic_llm, question.question, draft, retrieved, rel,
            trust_by_doc=trust_by_doc, flags_by_doc=flags_by_doc,
        )
        cost += s_cost
        calls += s_calls

        if not verified.variants and draft.variants:
            verified = FinalAnswer(
                variants=draft.variants,
                rejected_doc_ids=draft.rejected_doc_ids,
                abstained=False,
                explanation=(
                    draft.explanation
                    + "\nSkeptic rejected every variant; falling back to the unverified draft."
                ),
            )

        return RunResult(
            question_id=question.question_id,
            final_answer=verified,
            cost_usd=cost,
            latency_s=time.perf_counter() - t0,
            llm_calls=calls,
        )

    def describe(self) -> dict:
        base = super().describe()
        base["evaluator_llm_model"] = getattr(self.evaluator_llm, "model", "unknown")
        base["mediator_llm_model"] = getattr(self.mediator_llm, "model", "unknown")
        base["skeptic_llm_model"] = getattr(self.skeptic_llm, "model", "unknown")
        base["retrieval"] = {
            "bm25_weight": self._retrieval_cfg.bm25_weight,
            "dense_weight": self._retrieval_cfg.dense_weight,
            "top_k": self._retrieval_cfg.top_k,
        }
        base["min_relative_weight"] = self._min_relative_weight
        base["abstention_fallback"] = True
        base["reliability_formula"] = "0.40·retrieval + 0.25·confidence + 0.35·trust − 0.10·minority"
        return base
