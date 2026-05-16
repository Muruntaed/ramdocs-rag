"""FastAPI facade for the ramdocs-rag evaluation UI.

Endpoints:
    GET  /api/versions   — available pipeline versions (public whitelist).
    GET  /api/datasets   — available datasets (currently RAMDocs only).
    GET  /api/questions  — items from the active dataset (id, question, category).
    POST /api/answer     — run one (version, question_id) and return final answer + trace.

The backend deliberately does not vendor ramdocs-rag; it is a runtime dependency
installed via `pip install -e <path>` (see deploy/install.sh).
"""

from __future__ import annotations

import contextvars
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ramdocs_rag.core.llm import OpenAIClient
from ramdocs_rag.core.types import FinalAnswer, Question
from ramdocs_rag.pipelines.v1_0_madam_lite import V1MadamLite
from ramdocs_rag.pipelines.v2_0_entity_first import V2EntityFirst
from ramdocs_rag.pipelines.v3_3_analyzer_tuned import V33AnalyzerTuned
from ramdocs_rag.pipelines.v4_1_promptfix import V41PromptFix

# ─── Public pipeline versions ────────────────────────────────────────────────

# Public whitelist: one entry per major version, pointing at its latest
# minor iteration. Internal milestones (v3.0/3.1/3.2, v4.0) are intentionally
# omitted — see runs/_journal.md for the full development log.
VERSIONS: dict[str, dict[str, str]] = {
    "v1.0_madam_lite": {
        "label": "v1.0 — MADAM-lite",
        "description": (
            "BM25 + dense retrieval → per-document analyzer → 5-factor reliability "
            "score → deterministic majority vote with LLM-mediator fallback. Baseline."
        ),
    },
    "v2.0_entity_first": {
        "label": "v2.0 — Entity-first decomposition",
        "description": (
            "v1 + canonical entity per document + bare-answer enforcement in the "
            "analyzer + multi-answer aggregation via entity grouping."
        ),
    },
    "v3.3_analyzer_tuned": {
        "label": "v3 — Skeptic",
        "description": (
            "v2 + Skeptic verification agent (entity grounding, citation faithfulness, "
            "counter-evidence, depth-asymmetry) + tuned analyzer prompt for "
            "multi-candidate disambiguation. Prior recommended default; stays the "
            "cheap option when budget matters."
        ),
    },
    "v4.1_promptfix": {
        "label": "v4 — Evidence Evaluator",
        "description": (
            "v3 + per-document Evidence Evaluator agent (structured DocTrust report — "
            "trust_score + closed-enum red_flags) and a trust-weighted reliability "
            "formula (W_TRUST = 0.35). Analyzer prompt rewritten with a top-anchored "
            "HARD GROUNDING RULE, GOOD/BAD contrasts, place-disambiguator and a "
            "role-question + category-verbatim-scope patch. Current recommended "
            "default — Pareto improvement over v3 on every quality anchor."
        ),
    },
}

# ─── Datasets (built-in default + user-loaded by URL) ────────────────────────

# Default location: data/ramdocs_subset.json at the repository root.
# Override via RAMDOCS_SUBSET_PATH env var when running outside the repo.
_DEFAULT_SUBSET = Path(__file__).resolve().parents[4] / "data" / "ramdocs_subset.json"
_SUBSET_PATH = Path(os.environ.get("RAMDOCS_SUBSET_PATH", _DEFAULT_SUBSET))
_MAX_DATASET_BYTES = 5 * 1024 * 1024  # 5 MB ceiling for ad-hoc URL fetches


def _parse_items(payload: Any) -> list[Question]:
    items = payload if isinstance(payload, list) else payload.get("questions", [])
    return [Question.model_validate(it) for it in items]


def _load_default() -> list[Question]:
    return _parse_items(json.loads(_SUBSET_PATH.read_text(encoding="utf-8")))


class Dataset(BaseModel):
    id: str
    label: str
    source: str
    questions: list[Question]


DATASETS: dict[str, Dataset] = {
    "ramdocs": Dataset(
        id="ramdocs",
        label="RAMDocs (12-question subset)",
        source="",
        questions=_load_default(),
    ),
}


def _fetch_dataset(url: str) -> list[Question]:
    """Download a JSON dataset by URL and parse it into Question[] (RAMDocs schema)."""
    req = Request(url, headers={"User-Agent": "ramdocs-rag-demo/0.1"})
    with urlopen(req, timeout=30) as resp:  # noqa: S310 — explicit user-provided URL
        data = resp.read(_MAX_DATASET_BYTES + 1)
    if len(data) > _MAX_DATASET_BYTES:
        raise ValueError(f"dataset exceeds {_MAX_DATASET_BYTES // 1024 // 1024} MB limit")
    payload = json.loads(data.decode("utf-8"))
    questions = _parse_items(payload)
    if not questions:
        raise ValueError("dataset is empty")
    return questions


# ─── LLM tracing (DI, no monkey-patching) ────────────────────────────────────

_trace_buf: contextvars.ContextVar[list[dict] | None] = contextvars.ContextVar(
    "trace_buf", default=None
)


def _node_from_schema(schema_name: str) -> str:
    s = schema_name.lower()
    if "analyzer" in s or "claim" in s:
        return "analyzer"
    if "skeptic" in s or "verify" in s or "verdict" in s:
        return "skeptic"
    if "mediator" in s or "final" in s or "aggregat" in s:
        return "mediator"
    return "llm"


def _preview(s: str, n: int = 280) -> str:
    return s if len(s) <= n else s[:n] + "…"


class TracingClient(OpenAIClient):
    """OpenAIClient subclass that records every call into the current trace buffer."""

    def complete_json(self, *, system, user, schema, schema_name, temperature=0.0):  # type: ignore[override]
        t0 = time.time()
        try:
            result = super().complete_json(
                system=system,
                user=user,
                schema=schema,
                schema_name=schema_name,
                temperature=temperature,
            )
        except Exception as exc:
            buf = _trace_buf.get()
            if buf is not None:
                buf.append(
                    {
                        "node": _node_from_schema(schema_name),
                        "schema_name": schema_name,
                        "model": self.model,
                        "latency_s": round(time.time() - t0, 3),
                        "tokens_in": 0,
                        "tokens_out": 0,
                        "cost_usd": 0.0,
                        "input_preview": _preview(user),
                        "output_preview": "",
                        "output": {},
                        "error": str(exc),
                    }
                )
            raise
        buf = _trace_buf.get()
        if buf is not None:
            buf.append(
                {
                    "node": _node_from_schema(schema_name),
                    "schema_name": schema_name,
                    "model": result.model,
                    "latency_s": round(time.time() - t0, 3),
                    "tokens_in": result.tokens_in,
                    "tokens_out": result.tokens_out,
                    "cost_usd": round(result.cost_usd, 6),
                    "input_preview": _preview(user),
                    "output_preview": _preview(result.raw_text),
                    "output": result.parsed,
                }
            )
        return result


def _build(version_id: str):
    analyzer = TracingClient(model=os.environ.get("OPENAI_MODEL_ANALYZER", "gpt-4o-mini"))
    mediator = TracingClient(model=os.environ.get("OPENAI_MODEL_MEDIATOR", "gpt-4o"))
    if version_id == "v1.0_madam_lite":
        return V1MadamLite(llm=analyzer, mediator_llm=mediator)
    if version_id == "v2.0_entity_first":
        return V2EntityFirst(llm=analyzer, mediator_llm=mediator)
    if version_id == "v3.3_analyzer_tuned":
        skeptic = TracingClient(model=os.environ.get("OPENAI_MODEL_MEDIATOR", "gpt-4o"))
        return V33AnalyzerTuned(llm=analyzer, mediator_llm=mediator, skeptic_llm=skeptic)
    if version_id == "v4.1_promptfix":
        # Evaluator deliberately uses the cheap analyzer-class model (gpt-4o-mini):
        # its job is local per-document quality rating, not cross-corpus reasoning.
        # Skeptic stays on gpt-4o like in v3 — it's the verification agent.
        evaluator = TracingClient(model=os.environ.get("OPENAI_MODEL_ANALYZER", "gpt-4o-mini"))
        skeptic = TracingClient(model=os.environ.get("OPENAI_MODEL_MEDIATOR", "gpt-4o"))
        return V41PromptFix(
            llm=analyzer,
            evaluator_llm=evaluator,
            mediator_llm=mediator,
            skeptic_llm=skeptic,
        )
    raise HTTPException(status_code=404, detail=f"Unknown version: {version_id}")


# ─── HTTP API ────────────────────────────────────────────────────────────────

app = FastAPI(title="ramdocs-rag-demo", version="0.1.0", docs_url="/api/docs")


class AnswerRequest(BaseModel):
    version: str
    question_id: str
    dataset: str = "ramdocs"
    question_text_override: str | None = Field(
        default=None,
        description="If set, replaces the question text but keeps the original document pool.",
    )


class DatasetLoadRequest(BaseModel):
    url: str
    label: str | None = None


class TraceStep(BaseModel):
    node: str
    schema_name: str
    model: str
    latency_s: float
    tokens_in: int
    tokens_out: int
    cost_usd: float
    input_preview: str
    output_preview: str
    output: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class AnswerResponse(BaseModel):
    question_id: str
    question: str
    category: str
    final_answer: FinalAnswer
    latency_s: float
    total_cost_usd: float
    llm_calls: int
    retrieval: dict[str, Any]
    trace: list[TraceStep]


@app.get("/api/versions")
def get_versions() -> list[dict[str, str]]:
    return [{"id": k, **v} for k, v in VERSIONS.items()]


@app.get("/api/datasets")
def get_datasets() -> list[dict[str, Any]]:
    return [
        {
            "id": d.id,
            "label": d.label,
            "source": d.source,
            "n_questions": len(d.questions),
            "default": d.id == "ramdocs",
        }
        for d in DATASETS.values()
    ]


@app.post("/api/datasets/load")
def load_dataset(req: DatasetLoadRequest) -> dict[str, Any]:
    """Fetch a RAMDocs-format JSON dataset by URL and register it for subsequent queries."""
    try:
        questions = _fetch_dataset(req.url)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to load dataset: {exc}") from exc
    ds_id = "url_" + hashlib.sha1(req.url.encode("utf-8")).hexdigest()[:10]
    DATASETS[ds_id] = Dataset(
        id=ds_id,
        label=req.label or req.url,
        source=req.url,
        questions=questions,
    )
    return {"id": ds_id, "label": DATASETS[ds_id].label, "n_questions": len(questions)}


@app.get("/api/questions")
def get_questions(dataset: str = "ramdocs") -> list[dict[str, Any]]:
    ds = DATASETS.get(dataset)
    if ds is None:
        raise HTTPException(status_code=404, detail=f"Unknown dataset: {dataset}")
    return [
        {
            "id": q.question_id,
            "question": q.question,
            "category": q.category,
            "n_docs": len(q.docs),
        }
        for q in ds.questions
    ]


@app.post("/api/answer", response_model=AnswerResponse)
def post_answer(req: AnswerRequest) -> AnswerResponse:
    ds = DATASETS.get(req.dataset)
    if ds is None:
        raise HTTPException(status_code=400, detail=f"Unknown dataset: {req.dataset}")
    q_by_id = {q.question_id: q for q in ds.questions}
    q = q_by_id.get(req.question_id)
    if q is None:
        raise HTTPException(status_code=404, detail=f"Unknown question_id: {req.question_id}")
    if req.version not in VERSIONS:
        raise HTTPException(status_code=404, detail=f"Unknown version: {req.version}")

    # Custom question text overrides the original wording but keeps the doc pool.
    if req.question_text_override and req.question_text_override.strip() != q.question:
        q = q.model_copy(update={"question": req.question_text_override.strip()})

    pipeline = _build(req.version)

    buf: list[dict] = []
    token = _trace_buf.set(buf)
    t0 = time.time()
    try:
        result = pipeline.run(q)
    finally:
        _trace_buf.reset(token)
    elapsed = time.time() - t0

    return AnswerResponse(
        question_id=q.question_id,
        question=q.question,
        category=q.category,
        final_answer=result.final_answer,
        latency_s=round(elapsed, 2),
        total_cost_usd=round(sum(s.get("cost_usd", 0.0) for s in buf), 6),
        llm_calls=len(buf),
        retrieval={"pool_size": len(q.docs), "doc_ids": [d.doc_id for d in q.docs]},
        trace=[TraceStep(**s) for s in buf],
    )


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "datasets": {k: len(v.questions) for k, v in DATASETS.items()},
        "versions": list(VERSIONS),
    }
