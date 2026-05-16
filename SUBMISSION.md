# Submission — AI-Engineer / Modern RAG

Repository: <https://github.com/Muruntaed/ramdocs-rag>

## TL;DR

A multi-agent RAG prototype over the RAMDocs dataset, evaluated on a
12-question subset with eleven deterministic metrics (no LLM-judge).

Four pipeline versions are shipped frozen so the chosen default is
grounded in measured deltas rather than asserted. The **v4 · Evidence
Evaluator** pipeline is the recommended default; **v3 · Skeptic** stays
the cheap fallback when budget matters.

Final freeze metrics, averaged over two runs per version:

| Pipeline | F1-multi | Misinfo-rej | EM-any-gold | Recall@gold | $/Q | s/Q |
|---|---|---|---|---|---|---|
| v1.0 · MADAM-lite | 0.665 | 0.917 | 0.333 | 0.590 | $0.004 | 10.1 |
| v2.0 · entity-first | 0.667 | 0.792 | 0.833 | 0.667 | $0.002 | 10.7 |
| v3 · Skeptic | 0.675 | 0.875 | 0.792 | 0.646 | $0.010 | 13.5 |
| **v4 · Evidence Evaluator** | **0.818** | **0.917** | **0.875** | **0.847** | $0.022 | 24.0 |

Comparison anchors are F1-multi (correctness under ambiguity) and
Misinfo-rejection (robustness against the seeded misinformation
documents).

## Mapping to the brief

### Part A — Architecture & design

| Brief item | Where it lives |
|---|---|
| Agent definitions (≥3, with I/O) | `docs/architecture.md` (Retriever / Analyzer × K / Evaluator × K / Mediator / Skeptic); per-version SVG diagrams in `docs/journal/versions/`. |
| Inter-agent protocol | Strict-JSON Pydantic messages over a single `LLMClient.complete_json()` Protocol; deterministic Python control flow, no LangGraph. See `src/ramdocs_rag/core/types.py` and `core/llm.py`. |
| Conflict detection & resolution | Entity grouping (deterministic substring-merge) → reliability scoring → intra-group winner-ratio ≥ 1.5 → deterministic vote or LLM mediator → Skeptic four-check verification (entity grounding · citation faithfulness · counter-evidence · depth-asymmetry). See `docs/architecture.md` and v4.1 `pipeline.py`. |
| Metadata & version handling | RAMDocs carries no timestamps or source-authority metadata, so v1–v3 keep recency / authority as zero-weight placeholder slots. v4 replaces those slots with a per-document `trust_score` emitted by the Evidence Evaluator agent (internal_consistency · encyclopedic_quality · specificity · relevance), wired into the reliability formula with `W_TRUST = 0.35`. This is the project's answer to the metadata-priority requirement. |
| Trade-offs (latency / accuracy / cost / scale) | Per-version $/Q and s/Q above; full discussion in `docs/journal/index.html`. v3 vs v4 is an explicit Pareto choice — v4 is +0.143 F1 / +0.201 Recall at roughly 2× cost. |

### Part B — Prototype implementation

| Brief item | Where it lives |
|---|---|
| Indexing / retrieval over RAMDocs | `src/ramdocs_rag/core/retrieval.py` — BM25 + dense MiniLM, weighted fusion, top_k = 8. 12-question subset at `data/ramdocs_subset.json`. |
| Per-document analysis agent | `analyze_doc()` in every `pipelines/v*/agents.py`; strict-JSON `Claim` output. |
| Mediator + reconciliation strategy | Three reconciliation paths actually implemented: deterministic weighted vote, LLM mediator (fallback when intra-group ratio < 1.5), and Skeptic verification (v3+). See `resolve_entity_group()` in v4.1 `agents.py`. |
| Final answer + supporting evidence + explanation | `FinalAnswer` (`core/types.py`) carries `variants[*].supporting_doc_ids`, `rejected_doc_ids`, and a free-form `explanation`. |
| Traceability / logging | JSONL trace per run at `runs/<pipeline>/<sha>_<ts>/trace.jsonl` + `config.yaml` + `metrics.json`. Frozen baselines committed under `runs/_baseline/`. The demo UI at `demo/` shows the full per-call trace (system / user / parsed JSON) for any version and question. |

## How to run

```bash
make install
make test                                # ~8s, no API spend (MockLLM)
make bench PIPELINE=v4_1_promptfix       # full freeze run, ~$0.27, 12 questions
make compare VERSIONS="v3_3_analyzer_tuned v4_1_promptfix"
```

Quick read without running anything: open
`docs/journal/index.html` locally, or browse it via GitHub Pages once
enabled on the repository.

## Scope note (why four pipelines, not one)

The brief asks for one prototype; this repo ships four versioned
implementations measured on identical metrics on the same data subset.
The intent is to let the chosen default fall out of measured deltas
rather than be asserted on first attempt — every architectural decision
between v1 → v2 → v3 → v4 has a recorded hypothesis, freeze runs and
metric delta in the journal.

The whole project is ~3k lines of Python, all four versions included,
and the harness (measurement stand) is shared across them. Code
duplication between minor versions inside the same major (e.g.
`v3_0_skeptic/agents.py` ≡ `v3_3_analyzer_tuned/agents.py`) is
intentional — the freeze-immutability rule says published baselines
must remain bit-for-bit reproducible, so the runtime code is copied
into each version package and only the prompts differ.

## Self-critique — what I would do with more time

Code-side, the four pipelines could share their runtime through an
import-from-canonical-version pattern (e.g. `v3.3/agents.py` imports
from `v3.0/agents.py` and overrides nothing), reducing the
copy-and-edit cost without breaking freeze-immutability. The current
duplication is deliberate but ugly; a versioned-import scheme would be
cleaner.

The Analyzer and Evidence Evaluator run sequentially per document
inside the pipeline loop. With the structure already in place (both
are pure per-document agents), an `asyncio.gather` wrapper around the
K parallel calls would cut latency by roughly the K factor — currently
~24 s/Q at K=8 would drop to ~6 s/Q under the same cost.

The 12-question subset is a deliberate trade-off for cost
(~$0.10–$0.27 per freeze run) and reviewer-time, but it caps confidence
on per-category metrics (3 questions per category). The harness is
designed to scale to the full 500-question RAMDocs test split — a
single `make bench` with the full set would tighten every confidence
interval, at the cost of ~$10 per pipeline freeze.

No LLM-judge is used for any metric (RAMDocs ships per-doc gold
annotations sufficient for all eleven), but for open-ended generation
tasks beyond RAMDocs the same harness would need a pairwise-comparison
judge with reference-distribution calibration. That layer is not built.

CI is not wired. A two-job GitHub Action (`ruff check` + `pytest
tests/unit tests/integration`) would close the loop and produce the
green badge on the README — half an hour of YAML.

The Skeptic still misses one specific misinformation pattern (q131
class — `pure misinfo` inside `has_misinfo`), holding has_misinfo
F1 at 0.667 across v3 and v4. The pattern is identified in the
journal but not fixed; the next minor would target it.

## Pointers for the reviewer

- **Start here:** `docs/journal/index.html` — the research report with
  per-version hypotheses, freeze metrics and trade-offs.
- **Then:** `docs/architecture.md` — code-side architecture map.
- **Then:** `src/ramdocs_rag/pipelines/v4_1_promptfix/pipeline.py` —
  the recommended-default pipeline class (~200 lines).
- **Then:** `src/ramdocs_rag/core/types.py` — Pydantic contracts
  shared across versions.
- **Demo UI:** `demo/README.md` — FastAPI + static frontend that lets
  you pick a version + question and inspect every LLM call.
