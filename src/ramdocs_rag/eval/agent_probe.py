"""Run a SINGLE agent on a SINGLE (question, doc) for fast prompt iteration.

Existing tooling either runs a full pipeline (~30 s, 13 LLM calls,
multi-agent interactions) or a unit test against MockLLM (proves wiring
but not real-model behaviour). When iterating on an analyzer/evaluator
prompt to fix a specific hallucination case, we need the middle ground:
*real* model, *one* call, isolated I/O.

Usage:

    # Real OpenAI on gpt-4o-mini analyzer (~$0.0001 / call):
    python -m ramdocs_rag.eval.agent_probe \\
        --pipeline v4.1_promptfix --agent analyzer \\
        --question q306_ea196cdd --doc d5

    # Dry-run (no API key needed) — prints the rendered system + user
    # prompts so you can audit a prompt change before paying for a call:
    python -m ramdocs_rag.eval.agent_probe \\
        --pipeline v4.1_promptfix --agent analyzer \\
        --question q306_ea196cdd --doc d5 --dry-run

The ``--doc`` flag accepts a substring; the resolver picks the unique
match within the chosen question. So ``--doc d5`` resolves to e.g.
``q306_ea196cdd_d5_f9a1afd9``.

Scope (initial cut): supports analyzer + evaluator on any v4-family
pipeline. Mediator/Skeptic need synthesised pool/draft context — added
in a follow-up if needed.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from dataclasses import dataclass
from typing import Any

from ramdocs_rag.core.dataset import load_subset
from ramdocs_rag.core.llm import LLMCallResult, LLMClient, OpenAIClient
from ramdocs_rag.core.types import Question, RetrievedDoc


# ---------- dry-run mock that prints what the real client would send ----------


@dataclass
class _DryRunLLM:
    """LLM stub that records the prompt without making a network call.

    Returns a minimal JSON shape matching the agent's schema so the
    surrounding agent code (Pydantic parsing of the LLMCallResult) does
    not blow up. We don't care about the values — only the prompts.
    """

    captured: list[tuple[str, str, str]] | None = None
    model: str = "dry-run"

    def __post_init__(self) -> None:
        self.captured = []

    def complete_json(
        self,
        *,
        system: str,
        user: str,
        schema: dict,
        schema_name: str,
        temperature: float = 0.0,
    ) -> LLMCallResult:
        assert self.captured is not None
        self.captured.append((system, user, schema_name))
        # Build a schema-shaped placeholder so the caller's Pydantic
        # validator passes. We synthesize values from the schema's
        # required+properties metadata.
        parsed = _placeholder_for_schema(schema)
        return LLMCallResult(
            parsed=parsed,
            raw_text=json.dumps(parsed),
            cost_usd=0.0,
            tokens_in=len(system) + len(user),
            tokens_out=0,
            model=self.model,
        )


def _placeholder_for_schema(schema: dict) -> dict:
    """Produce a minimal valid object for a strict JSON schema.

    Only handles the shapes we actually use (object / string / number /
    boolean / array / enum). Good enough for the analyzer + evaluator
    schemas; will need extension if probing a richer agent.
    """
    if schema.get("type") != "object":
        return {}
    out: dict[str, Any] = {}
    props = schema.get("properties", {})
    for key in schema.get("required", []):
        out[key] = _placeholder_for_type(props.get(key, {}))
    return out


def _placeholder_for_type(prop: dict) -> Any:
    t = prop.get("type")
    if "enum" in prop:
        return prop["enum"][0]
    if t == "string":
        return ""
    if t == "number":
        return prop.get("minimum", 0.0)
    if t == "integer":
        return 0
    if t == "boolean":
        return False
    if t == "array":
        return []
    if t == "object":
        return _placeholder_for_schema(prop)
    return None


# ---------- agent dispatch ----------


_AGENT_REGISTRY: dict[str, str] = {
    "analyzer": "analyze_doc",
    "evaluator": "evaluate_doc",
}


def _load_agent(pipeline: str, agent: str):
    """Return the agent callable from the given pipeline's agents module.

    Raises a clear error if the agent isn't supported by that pipeline
    (e.g. evaluator on v1/v2/v3 — none of them have it).
    """
    if agent not in _AGENT_REGISTRY:
        raise SystemExit(
            f"Unknown --agent {agent!r}. Supported: {sorted(_AGENT_REGISTRY)}"
        )
    module_path = f"ramdocs_rag.pipelines.{pipeline.replace('.', '_')}.agents"
    try:
        mod = importlib.import_module(module_path)
    except ModuleNotFoundError as e:
        raise SystemExit(f"No such pipeline: {pipeline} ({e})") from e
    fn_name = _AGENT_REGISTRY[agent]
    fn = getattr(mod, fn_name, None)
    if fn is None:
        raise SystemExit(
            f"Pipeline {pipeline} does not implement {agent} "
            f"(no {fn_name} in {module_path})"
        )
    return fn


def _resolve_question_and_doc(
    question_id: str, doc_id_part: str
) -> tuple[Question, RetrievedDoc]:
    subset = load_subset()
    q = next((x for x in subset if x.question_id == question_id), None)
    if q is None:
        raise SystemExit(
            f"Question {question_id!r} not in subset. "
            f"First few: {[x.question_id for x in subset[:3]]}…"
        )
    matches = [d for d in q.docs if doc_id_part in d.doc_id]
    if not matches:
        raise SystemExit(
            f"No doc matched {doc_id_part!r} in {question_id}. "
            f"Available: {[d.doc_id[-12:] for d in q.docs]}"
        )
    if len(matches) > 1:
        raise SystemExit(
            f"Doc selector {doc_id_part!r} ambiguous, matched: "
            f"{[d.doc_id for d in matches]}"
        )
    doc = matches[0]
    return q, RetrievedDoc(doc_id=doc.doc_id, text=doc.text, score=1.0)


# ---------- main ----------


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--pipeline", required=True, help="e.g. v4.1_promptfix")
    p.add_argument(
        "--agent",
        required=True,
        choices=sorted(_AGENT_REGISTRY),
        help="which agent to probe",
    )
    p.add_argument("--question", required=True, help="question_id (e.g. q306_ea196cdd)")
    p.add_argument(
        "--doc",
        required=True,
        help="doc_id substring (e.g. 'd5' or full doc_id) within the question",
    )
    p.add_argument("--model", default="gpt-4o-mini", help="OpenAI model")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't call OpenAI; print the rendered system + user prompts and exit.",
    )
    return p


def _print_dry_run(client: _DryRunLLM) -> None:
    assert client.captured is not None
    if not client.captured:
        print("[dry-run] agent made zero LLM calls — nothing to inspect", file=sys.stderr)
        return
    for i, (system, user, schema_name) in enumerate(client.captured):
        print(f"=== call #{i + 1}  schema_name={schema_name} ===")
        print("--- SYSTEM ---")
        print(system)
        print("--- USER ---")
        print(user)
        print()


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    question, doc = _resolve_question_and_doc(args.question, args.doc)
    agent_fn = _load_agent(args.pipeline, args.agent)

    if args.dry_run:
        client: LLMClient = _DryRunLLM()  # type: ignore[assignment]
    else:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise SystemExit(
                "OPENAI_API_KEY not set. Either export it or pass --dry-run."
            )
        client = OpenAIClient(model=args.model, api_key=api_key)

    print(f"[probe] pipeline={args.pipeline} agent={args.agent}")
    print(f"[probe] question_id={question.question_id}")
    print(f"[probe] question: {question.question}")
    print(f"[probe] doc_id={doc.doc_id}")
    print(f"[probe] doc preview: {doc.text[:160]!r}…")
    print()

    result, cost, calls = agent_fn(client, question.question, doc)

    if args.dry_run:
        _print_dry_run(client)  # type: ignore[arg-type]

    print("=== AGENT OUTPUT ===")
    payload = (
        result.model_dump() if hasattr(result, "model_dump") else result.__dict__
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print()
    print(f"[probe] cost_usd={cost:.6f}  llm_calls={calls}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
