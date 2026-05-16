"""V2.0 — Entity-First Decomposition.

Flow::

  retrieve → analyze ×K → group_by_entity → reliability (2-pass)
  → resolve_entity_group per group → assemble AnswerVariants.

Key difference vs v1.0: the answer is a **list of variants** (one per
entity), not a single winner. That breaks the structural Recall@gold
cap v1.0 hits on ambiguous RAMDocs questions.

Thresholds for including an entity in the final answer:
- the top entity is always included;
- the rest pass only if their weight ≥ ``MIN_RELATIVE_WEIGHT × top_weight``.
This trims misinfo entities with low aggregate reliability.
"""

from __future__ import annotations

import time

from ramdocs_rag.core.llm import LLMClient
from ramdocs_rag.core.retrieval import RetrievalConfig, retrieve
from ramdocs_rag.core.types import FinalAnswer, Question, RunResult
from ramdocs_rag.pipelines.base import Pipeline

from .agents import analyze_doc, resolve_entity_group
from .grouping import display_entity, group_by_entity
from .reliability import final_reliability, initial_reliability

MIN_RELATIVE_WEIGHT = 0.4  # an entity group passes if its weight ≥ 40% of the top group


class V2EntityFirst(Pipeline):
    """Entity-first multi-answer pipeline. Group by entity, vote intra-entity."""

    name = "v2.0_entity_first"
    version = "2.0.0"

    def __init__(
        self,
        llm: LLMClient,
        mediator_llm: LLMClient | None = None,
        config: dict | None = None,
    ) -> None:
        super().__init__(llm, config or {})
        self.mediator_llm: LLMClient = mediator_llm or llm
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

        # 1. Retrieval
        retrieved = retrieve(question.question, list(question.docs), self._retrieval_cfg)

        # 2. Analyzer × K
        claims = []
        for d in retrieved:
            c, c_cost, c_calls = analyze_doc(self.llm, question.question, d)
            claims.append(c)
            cost += c_cost
            calls += c_calls

        # 3. Group claims by entity (supports-only)
        groups = group_by_entity(claims)
        no_answer_doc_ids = [c.doc_id for c in claims if c.stance != "supports"]

        # No supports-entity at all → abstain
        if not groups:
            return RunResult(
                question_id=question.question_id,
                final_answer=FinalAnswer(
                    variants=[],
                    rejected_doc_ids=no_answer_doc_ids,
                    abstained=True,
                    explanation="No documents support an answer to the question.",
                ),
                cost_usd=cost,
                latency_s=time.perf_counter() - t0,
                llm_calls=calls,
            )

        # 4. Initial reliability (no penalty)
        rel = initial_reliability(retrieved, claims)

        # 5. Per-group resolution → variants; intra-group minority → minority set.
        all_minority: set[str] = set()
        for _, group_claims in groups.items():
            if len(group_claims) < 2:
                continue
            # Find the intra-group minority by normalised answer text: every
            # claim that does NOT belong to the top-text cluster is a loser.
            from collections import Counter

            from .agents import _norm_text

            counts = Counter(_norm_text(c.text) for c in group_claims)
            top_text = counts.most_common(1)[0][0]
            for c in group_claims:
                if _norm_text(c.text) != top_text:
                    all_minority.add(c.doc_id)

        # 6. Final reliability (with intra-group minority penalty)
        rel = final_reliability(retrieved, claims, all_minority)

        # 7. Compute group weights for relative filtering
        group_weights = {
            key: sum(rel.get(c.doc_id, 0.0) for c in gclaims) for key, gclaims in groups.items()
        }
        top_weight = max(group_weights.values()) if group_weights else 0.0
        cutoff = self._min_relative_weight * top_weight

        # 8. Resolve each group (above cutoff); rejected groups → rejected docs
        variants = []
        rejected: set[str] = set(no_answer_doc_ids)
        for key, gclaims in sorted(groups.items(), key=lambda kv: -group_weights[kv[0]]):
            entity_display = display_entity(gclaims)
            if group_weights[key] < cutoff and len(variants) > 0:
                # Below threshold — entire group rejected
                rejected.update(c.doc_id for c in gclaims)
                continue
            variant, intra_rejected, m_cost, m_calls = resolve_entity_group(
                self.mediator_llm, question.question, entity_display, gclaims, rel
            )
            cost += m_cost
            calls += m_calls
            variants.append(variant)
            rejected.update(intra_rejected)

        # Defensive: strip any doc that ended up both in supporting and rejected.
        supporting_all = {d for v in variants for d in v.supporting_doc_ids}
        rejected -= supporting_all

        explanation = (
            f"Grouped {len(claims)} claims into {len(groups)} entity-clusters; "
            f"emitted {len(variants)} answer variant(s) above {self._min_relative_weight:.0%} "
            f"of top-group weight; rejected {len(rejected)} docs."
        )

        return RunResult(
            question_id=question.question_id,
            final_answer=FinalAnswer(
                variants=variants,
                rejected_doc_ids=sorted(rejected),
                abstained=False,
                explanation=explanation,
            ),
            cost_usd=cost,
            latency_s=time.perf_counter() - t0,
            llm_calls=calls,
        )

    def describe(self) -> dict:
        base = super().describe()
        base["mediator_llm_model"] = getattr(self.mediator_llm, "model", "unknown")
        base["retrieval"] = {
            "bm25_weight": self._retrieval_cfg.bm25_weight,
            "dense_weight": self._retrieval_cfg.dense_weight,
            "top_k": self._retrieval_cfg.top_k,
        }
        base["min_relative_weight"] = self._min_relative_weight
        return base
