"""Pydantic contracts shared across every pipeline.

These types are the *only* stable contract between the versioned
pipelines and the measurement stand. Any architecture version
(v1, v2, ...) must return a ``RunResult`` that ``eval.metrics`` can
score without knowing the internals of the graph.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# ---------- RAMDocs inputs ----------


class DocEvalMeta(BaseModel):
    """Per-document ground truth from RAMDocs (``eval_metadata[i]``).

    Every document is labelled with a type (correct / misinfo / noise)
    and, for ``correct`` documents, the canonical ``answer``. Used by
    the metrics layer; deliberately **not visible to pipelines** — they
    only see ``Question.docs``.
    """

    model_config = ConfigDict(extra="forbid")

    doc_id: str
    type: Literal["correct", "misinfo", "noise"]
    answer: str | None = None


class RAMDoc(BaseModel):
    """A document in the RAMDocs pool — raw text plus an id."""

    model_config = ConfigDict(extra="forbid")

    doc_id: str
    text: str


class Question(BaseModel):
    """One RAMDocs question with its document pool and gold annotations."""

    model_config = ConfigDict(extra="forbid")

    question_id: str
    question: str
    category: Literal["pure_correct", "has_misinfo", "has_noise", "mixed_conflict"]
    disambig_entity: list[str] = Field(default_factory=list)
    gold_answers: list[str]
    wrong_answers: list[str] = Field(default_factory=list)
    docs: list[RAMDoc]
    eval_metadata: list[DocEvalMeta]


# ---------- Pipeline artefacts ----------


class RetrievedDoc(BaseModel):
    """A document picked by the retriever with an opaque score."""

    model_config = ConfigDict(extra="forbid")

    doc_id: str
    text: str
    score: float = Field(ge=0.0, le=1.0)


class Claim(BaseModel):
    """A claim extracted by the analyzer from one document."""

    model_config = ConfigDict(extra="forbid")

    doc_id: str
    text: str = Field(description="Short candidate answer as the agent understood it.")
    stance: Literal["supports", "contradicts", "no_answer"]
    confidence: float = Field(ge=0.0, le=1.0)
    supporting_quote: str | None = None
    entity: str | None = Field(
        default=None,
        description="Canonical entity, if the pipeline supports entity-first (v2+).",
    )


class AnswerVariant(BaseModel):
    """One possible final answer (used by multi-answer pipelines)."""

    model_config = ConfigDict(extra="forbid")

    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    supporting_doc_ids: list[str] = Field(default_factory=list)
    entity: str | None = None


class FinalAnswer(BaseModel):
    """The final pipeline output for one question.

    ``variants`` is a list and may be empty when ``abstained=True``.
    Single-answer pipelines populate exactly one element.
    """

    model_config = ConfigDict(extra="forbid")

    variants: list[AnswerVariant] = Field(default_factory=list)
    rejected_doc_ids: list[str] = Field(default_factory=list)
    abstained: bool = False
    explanation: str = ""

    @property
    def primary_answer(self) -> str | None:
        """Convenience accessor used by single-answer EM metrics."""
        return self.variants[0].answer if self.variants else None

    @property
    def all_answers(self) -> list[str]:
        return [v.answer for v in self.variants]

    @property
    def all_supporting_doc_ids(self) -> list[str]:
        out: list[str] = []
        for v in self.variants:
            out.extend(v.supporting_doc_ids)
        return out


# ---------- Traces and results ----------


class TraceEvent(BaseModel):
    """One entry in the JSONL trace. ``data`` is free-form."""

    model_config = ConfigDict(extra="forbid")

    ts: datetime
    node: str = Field(description="Graph node name (retriever, analyzer, mediator, ...).")
    event: str = Field(description="Event type: start | output | llm_call | error.")
    data: dict = Field(default_factory=dict)


class RunResult(BaseModel):
    """Pipeline result for one question."""

    model_config = ConfigDict(extra="forbid")

    question_id: str
    final_answer: FinalAnswer
    cost_usd: float = 0.0
    latency_s: float = 0.0
    llm_calls: int = 0
    error: str | None = None


class PipelineRun(BaseModel):
    """A complete run of one pipeline version across the dataset."""

    model_config = ConfigDict(extra="forbid")

    pipeline_name: str
    pipeline_version: str
    git_sha: str | None = None
    started_at: datetime
    finished_at: datetime | None = None
    config: dict = Field(default_factory=dict)
    results: list[RunResult] = Field(default_factory=list)


class RunMetrics(BaseModel):
    """Aggregated metrics over a ``PipelineRun``."""

    model_config = ConfigDict(extra="forbid")

    n_questions: int

    # Answer quality
    em_any_gold: float
    em_substring: float
    recall_all_gold: float
    precision_answers: float
    f1_multi_answer: float
    abstention_rate: float

    # Source-handling quality
    misinfo_rejection: float
    noise_rejection: float
    correct_citation: float
    citation_faithfulness: float
    coverage: float

    # Operational
    total_cost_usd: float
    avg_cost_per_question_usd: float
    avg_llm_calls_per_question: float
    avg_latency_s: float

    # Extra — per-category breakdown
    by_category: dict[str, dict[str, float]] = Field(default_factory=dict)
