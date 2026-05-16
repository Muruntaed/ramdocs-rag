"""v1.0 main class. Linear pipeline: retrieve → analyze × K → reliability → mediate."""

from __future__ import annotations

import time

from ...core.llm import LLMClient
from ...core.types import Question, RunResult
from ..base import Pipeline
from .agents import analyze_doc, mediate
from ramdocs_rag.core.retrieval import RetrievalConfig, retrieve


class V1MadamLite(Pipeline):
    """MADAM-lite baseline. Hyperparameters match ``_legacy/ramdocs_rag_v0``."""

    name = "v1.0_madam_lite"
    version = "1.0.0"

    def __init__(
        self,
        llm: LLMClient,
        mediator_llm: LLMClient | None = None,
        config: dict | None = None,
    ) -> None:
        super().__init__(llm, config or {})
        # The mediator may use a different model (gpt-4o in the legacy
        # prototype). By default it shares the analyzer client.
        self.mediator_llm: LLMClient = mediator_llm or llm
        self._retrieval_cfg = RetrievalConfig(
            bm25_weight=self.config.get("bm25_weight", 0.5),
            dense_weight=self.config.get("dense_weight", 0.5),
            top_k=self.config.get("top_k", 8),
        )

    def run(self, question: Question) -> RunResult:
        t0 = time.perf_counter()
        cost = 0.0
        calls = 0

        # 1. Retrieval (deterministic, no LLM call).
        retrieved = retrieve(question.question, list(question.docs), self._retrieval_cfg)

        # 2. Analyzer × K (sequential; mocked tests are deterministic).
        claims = []
        for d in retrieved:
            c, c_cost, c_calls = analyze_doc(self.llm, question.question, d)
            claims.append(c)
            cost += c_cost
            calls += c_calls

        # 3. Reliability (two-pass).
        from .reliability import compute_reliability

        reliability, _ = compute_reliability(retrieved, claims)

        # 4. Mediator (deterministic vote → LLM fallback). Uses a separate
        # client when provided (gpt-4o is preferred for conflict resolution).
        answer, used_llm, m_cost, m_calls = mediate(
            self.mediator_llm, question.question, claims, reliability
        )
        cost += m_cost
        calls += m_calls

        return RunResult(
            question_id=question.question_id,
            final_answer=answer,
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
        return base
