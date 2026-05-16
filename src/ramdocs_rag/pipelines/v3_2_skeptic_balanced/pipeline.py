"""V3.2 — Entity-First + Balanced Skeptic + lowered threshold.

Two minor changes over v3.1:
1. Skeptic prompt adds a depth-asymmetry check on top of the homonym rule
   (catches thin-stub misinfo without re-introducing homonym over-rejection).
2. ``min_relative_weight`` lowered from 0.40 to 0.30 to let more legitimate
   entity groups through for ambiguous questions.

The pipeline-level abstention fallback from v3.1 is preserved.
"""

from __future__ import annotations

import time

from ramdocs_rag.core.llm import LLMClient
from ramdocs_rag.core.retrieval import RetrievalConfig, retrieve
from ramdocs_rag.core.types import FinalAnswer, Question, RunResult
from ramdocs_rag.pipelines.base import Pipeline

from .agents import analyze_doc, resolve_entity_group, skeptic_verify
from .grouping import display_entity, group_by_entity
from .reliability import final_reliability, initial_reliability

MIN_RELATIVE_WEIGHT = 0.30  # loosened from 0.40 in v3.1 to lift recall


class V32SkepticBalanced(Pipeline):
    """v3.2: balanced Skeptic + loosened entity-group threshold."""

    name = "v3.2_skeptic_balanced"
    version = "3.2.0"

    def __init__(
        self,
        llm: LLMClient,
        mediator_llm: LLMClient | None = None,
        skeptic_llm: LLMClient | None = None,
        config: dict | None = None,
    ) -> None:
        super().__init__(llm, config or {})
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

        claims = []
        for d in retrieved:
            c, c_cost, c_calls = analyze_doc(self.llm, question.question, d)
            claims.append(c)
            cost += c_cost
            calls += c_calls

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

        rel = initial_reliability(retrieved, claims)

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
        rel = final_reliability(retrieved, claims, all_minority)

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
                f"emitted {len(variants)} variant(s) above {self._min_relative_weight:.0%} of top-group weight."
            ),
        )

        verified, _decisions, s_cost, s_calls = skeptic_verify(
            self.skeptic_llm, question.question, draft, retrieved, rel
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
        base["mediator_llm_model"] = getattr(self.mediator_llm, "model", "unknown")
        base["skeptic_llm_model"] = getattr(self.skeptic_llm, "model", "unknown")
        base["retrieval"] = {
            "bm25_weight": self._retrieval_cfg.bm25_weight,
            "dense_weight": self._retrieval_cfg.dense_weight,
            "top_k": self._retrieval_cfg.top_k,
        }
        base["min_relative_weight"] = self._min_relative_weight
        base["abstention_fallback"] = True
        base["skeptic_rules"] = ["homonyms", "depth_asymmetry"]
        return base
