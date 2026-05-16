"""Runner: execute a pipeline over the dataset, compute metrics, dump artefacts.

Usage::

    python -m ramdocs_rag.eval.runner --pipeline v4_1_promptfix

Artefacts are written to ``runs/<pipeline>/<sha>_<ts>/``:
- ``config.yaml`` — snapshot of ``pipeline.describe()``
- ``trace.jsonl`` — one JSON line per question (full ``RunResult``)
- ``metrics.json`` — aggregated ``RunMetrics``

Pipelines register themselves via ``register(...)`` inside
``ramdocs_rag.pipelines.registry``, which the runner imports lazily.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from ..core.dataset import load_subset
from ..core.types import PipelineRun, Question, RunResult
from ..pipelines.base import Pipeline
from .metrics import compute_metrics

# Pipeline registry: name -> factory(no args) -> Pipeline.
_REGISTRY: dict[str, callable] = {}


def register(name: str, factory) -> None:
    """Register a pipeline factory. Called from ``pipelines.registry``."""
    _REGISTRY[name] = factory


def list_pipelines() -> list[str]:
    # Lazy-load the registry module so that tests which do not touch the
    # real OpenAI SDK do not fail at import time.
    _ensure_registry_loaded()
    return sorted(_REGISTRY)


def _ensure_registry_loaded() -> None:
    # ``registry`` may pull optional deps (OpenAIClient, etc.) — keep the
    # import suppressed so that tests which do not touch the real OpenAI
    # SDK can still import the runner.
    with contextlib.suppress(ImportError):
        from ..pipelines import registry  # noqa: F401


def _git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def _run_dir(pipeline_name: str, git_sha: str | None, tag: str = "") -> Path:
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    sha = git_sha or "nosha"
    suffix = f"_{tag}" if tag else ""
    base = Path(__file__).resolve().parents[3] / "runs" / pipeline_name / f"{sha}_{ts}{suffix}"
    base.mkdir(parents=True, exist_ok=True)
    return base


def run_pipeline(
    pipeline: Pipeline, questions: list[Question]
) -> tuple[PipelineRun, list[tuple[Question, RunResult]]]:
    """Run the pipeline over a list of questions. All exceptions are caught here."""
    started = datetime.now(UTC)
    results: list[RunResult] = []
    pairs: list[tuple[Question, RunResult]] = []
    for q in questions:
        t0 = time.perf_counter()
        try:
            res = pipeline.run(q)
        except Exception as e:  # noqa: BLE001 — the runner deliberately swallows every exception
            res = RunResult(
                question_id=q.question_id,
                final_answer=__import__(
                    "ramdocs_rag.core.types", fromlist=["FinalAnswer"]
                ).FinalAnswer(abstained=True, explanation=f"runner error: {e!r}"),
                error=f"{type(e).__name__}: {e}",
            )
        res.latency_s = res.latency_s or (time.perf_counter() - t0)
        results.append(res)
        pairs.append((q, res))
    finished = datetime.now(UTC)

    run = PipelineRun(
        pipeline_name=pipeline.name,
        pipeline_version=pipeline.version,
        git_sha=_git_sha(),
        started_at=started,
        finished_at=finished,
        config=pipeline.describe(),
        results=results,
    )
    return run, pairs


def _dump_artifacts(
    run: PipelineRun,
    pairs: list[tuple[Question, RunResult]],
    *,
    tag: str = "",
) -> Path:
    out = _run_dir(run.pipeline_name, run.git_sha, tag=tag)

    (out / "config.yaml").write_text(
        json.dumps(run.config, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    with (out / "trace.jsonl").open("w", encoding="utf-8") as fh:
        for r in run.results:
            fh.write(json.dumps(r.model_dump(mode="json"), ensure_ascii=False, default=str) + "\n")

    metrics = compute_metrics(pairs)
    (out / "metrics.json").write_text(metrics.model_dump_json(indent=2), encoding="utf-8")
    return out


def _load_env() -> None:
    """Load a project ``.env`` if present. Idempotent."""
    try:
        from dotenv import load_dotenv

        # .env lives at the repository root — three levels above runner.py.
        env_path = Path(__file__).resolve().parents[3] / ".env"
        if env_path.exists():
            load_dotenv(env_path, override=False)
    except Exception:  # noqa: BLE001
        pass


def _flush_langfuse() -> None:
    """Force-flush the Langfuse trace buffer before exit."""
    try:
        from langfuse import get_client

        get_client().flush()
    except Exception:  # noqa: BLE001 — Langfuse is optional
        pass


def _select_questions(
    all_qs: list[Question], question_id: str | None, limit: int | None
) -> list[Question]:
    if question_id is not None:
        match = [q for q in all_qs if q.question_id == question_id]
        if not match:
            raise SystemExit(f"question_id {question_id!r} not found in subset")
        return match
    if limit is not None and limit > 0:
        return all_qs[:limit]
    return all_qs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a registered RAMDocs RAG pipeline.")
    parser.add_argument("--pipeline", help="registered pipeline name")
    parser.add_argument("--list", action="store_true", help="list available pipelines")
    parser.add_argument(
        "--question-id",
        dest="question_id",
        help="run only one question (smoke test before a freeze)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="run only the first N questions (debugging)",
    )
    parser.add_argument(
        "--tag",
        default="",
        help="suffix appended to the run directory name (e.g. 'smoke', 'freeze1', 'freeze2')",
    )
    args = parser.parse_args(argv)

    if args.list:
        names = list_pipelines()
        print("\n".join(names) if names else "<no pipelines registered yet>")
        return 0

    if not args.pipeline:
        parser.error("--pipeline is required (or use --list)")

    # Load .env BEFORE registry (factories read models / keys from env).
    _load_env()
    _ensure_registry_loaded()

    if args.pipeline not in _REGISTRY:
        print(f"unknown pipeline: {args.pipeline}", file=sys.stderr)
        print(f"available: {list_pipelines() or '<none registered yet>'}", file=sys.stderr)
        return 2

    pipeline = _REGISTRY[args.pipeline]()
    all_qs = list(load_subset())
    questions = _select_questions(all_qs, args.question_id, args.limit)
    print(
        f"running {pipeline.name} on {len(questions)}/{len(all_qs)} questions "
        f"(question_id={args.question_id!r}, limit={args.limit})"
    )

    try:
        run, pairs = run_pipeline(pipeline, questions)
    finally:
        _flush_langfuse()

    out = _dump_artifacts(run, pairs, tag=args.tag)
    print(f"run dumped to: {out}")

    # Short stdout summary — convenient for a human-readable smoke check.
    metrics = compute_metrics(pairs)
    print("\n=== summary ===")
    print(f"  EM-any-gold        : {metrics.em_any_gold:.2f}")
    print(f"  F1-multi-answer    : {metrics.f1_multi_answer:.2f}")
    print(f"  Recall-all-gold    : {metrics.recall_all_gold:.2f}")
    print(f"  Misinfo-rejection  : {metrics.misinfo_rejection:.2f}")
    print(f"  Citation-faith     : {metrics.citation_faithfulness:.2f}")
    print(f"  Abstention rate    : {metrics.abstention_rate:.2f}")
    print(f"  $/Q (avg)          : {metrics.avg_cost_per_question_usd:.4f}")
    print(f"  LLM calls/Q (avg)  : {metrics.avg_llm_calls_per_question:.1f}")
    print(f"  Latency s (avg)    : {metrics.avg_latency_s:.2f}")
    return 0


if __name__ == "__main__":
    # Avoid the double-import trap: under ``python -m ramdocs_rag.eval.runner``
    # this file is imported both as ``__main__`` *and* as
    # ``ramdocs_rag.eval.runner``. The registry registers into the canonical
    # instance, while main() would otherwise read from ``__main__``.
    # Delegate to the canonical entry point.
    from ramdocs_rag.eval.runner import main as _canonical_main

    raise SystemExit(_canonical_main())
