# RAMDocs Multi-Agent RAG — Versioned Prototype

A multi-agent RAG prototype over the
[`HanNight/RAMDocs`](https://huggingface.co/datasets/HanNight/RAMDocs)
dataset. The brief asks for one solution; this repository delivers
several — versioned pipelines compared on identical metrics on the
same data subset — so that the chosen architecture is grounded in
evidence rather than asserted.

> **Status:** four major versions published frozen. **v4 · Evidence
> Evaluator** is the current recommended default; v3 · Skeptic stays the
> cheaper option when budget matters. Full version history, hypotheses,
> freeze metrics and trade-offs: see [`docs/journal/index.html`](docs/journal/index.html).

## Why three layers

The repository is organised so that *measurement* never moves while
*architecture* iterates:

| Layer | Folder | Property |
|---|---|---|
| Measurement stand | `eval/`, `tests/` | Written once, never edited. 11 deterministic metrics from the per-doc gold annotations — no LLM-judge. |
| Versioned pipelines | `pipelines/vN_M_*/` | Interchangeable implementations. `vN` = major (graph / agent roles changed), `vN.M` = minor (same graph, different prompts or hyperparameters). |
| Immutable run artefacts | `runs/` | Each freeze run dumps `config.yaml` + `trace.jsonl` + `metrics.json`. `_baseline/<v>.json` records the averaged-over-two-runs frozen metric set per published version. |

## What's in each version

The high-level summary lives in the HTML journal
([`docs/journal/index.html`](docs/journal/index.html)). One sentence per
major:

| Version | Status | Idea |
|---|---|---|
| **v1.0 · MADAM-lite** | FROZEN | BM25 + dense → analyzer × K → 5-factor reliability → deterministic vote with LLM-mediator fallback. Single-answer baseline. |
| **v2.0 · Entity-first** | FROZEN | Multi-answer as a first-class citizen — the analyzer emits an entity per document, the mediator groups by entity, the answer is a `list[AnswerVariant]`. EM-any-gold jumps from 0.33 to 0.83. |
| **v3 · Skeptic** | FROZEN | A four-check verification agent (entity grounding · citation faithfulness · counter-evidence · depth-asymmetry) after the mediator, plus a pipeline-level abstention fallback. Misinfo-rejection recovered to 0.875. |
| **v4 · Evidence Evaluator** | **FROZEN · recommended** | Per-document Evidence Evaluator (`DocTrust` report — trust_score + closed-enum red_flags), trust-weighted reliability (W_TRUST = 0.35), and an analyzer prompt rewritten with a top-anchored HARD GROUNDING RULE. Pareto improvement over v3 on every quality anchor. |

Internal minor iterations (v3.0/3.1/3.2, v4.0) live in the codebase for
archaeology but are intentionally omitted from the public journal; the
final minor of each major is the published baseline. The full development
log including hypotheses, what worked and what didn't is in
[`runs/_journal.md`](runs/_journal.md).

## Layout

```
ramdocs_rag/
├── data/ramdocs_subset.json            12-question slice of RAMDocs
├── src/ramdocs_rag/
│   ├── core/                            shared primitives (types, LLM client, dataset loader, trace, safety)
│   ├── eval/                            measurement stand: metrics, runner, comparison, agent_probe
│   └── pipelines/                       versioned architectures (v1.0 … v4.1)
├── tests/
│   ├── unit/                            fast, no API (Pydantic models, metrics, MockLLM, safety layer)
│   ├── integration/                     full pipeline with MockLLM (no real API)
│   └── e2e/                             real OpenAI smoke (opt-in via @pytest.mark.e2e)
├── runs/
│   ├── _baseline/<v>.json               frozen baselines (committed)
│   ├── _reports/<ts>.md                 committed summary tables
│   ├── _journal.md                      free-form research log
│   ├── _WIP.md                          current iteration state (cross-session continuity)
│   └── <pipeline>/<sha>_<ts>/           local artefacts of a run (gitignored)
├── docs/journal/                        HTML research report (index + per-version pages)
├── demo/                                interactive evaluation UI (FastAPI + static frontend)
├── pyproject.toml
└── Makefile
```

## Quick start

```bash
make install              # editable install + dev extras (pytest, ruff)
make test                 # unit + integration with MockLLM, ~8 s, no API spend
make test-e2e             # real OpenAI smoke (requires OPENAI_API_KEY in ../.env)
make bench PIPELINE=v4_1_promptfix         # full freeze run on 12 questions
make compare VERSIONS="v3_3_analyzer_tuned v4_1_promptfix"
```

Direct module invocations:

```bash
python -m ramdocs_rag.eval.runner --list
python -m ramdocs_rag.eval.runner --pipeline v4_1_promptfix
python -m ramdocs_rag.eval.compare v3_3_analyzer_tuned v4_1_promptfix
```

## Metrics

All metrics are **deterministic** — computed from the per-document
`eval_metadata` gold annotations in RAMDocs, no LLM-judge. The eleven
metrics are defined in `src/ramdocs_rag/eval/metrics.py`. Two are the
declared comparison anchors:

- **F1-multi-answer** — correctness under ambiguity (most RAMDocs
  questions have multiple valid gold answers).
- **Misinfo-rejection** — robustness against the misinformation
  documents seeded into the retrieved pool.

## Freeze discipline

Each published version is frozen with **two consecutive runs**. If they
disagree by more than ±1 answer out of 12 the version is not baselined
and the issue is investigated (fragile prompt or non-deterministic code).
The two run directories plus the averaged metrics live under
`runs/_baseline/<version>.json`.

## Interactive demo

A FastAPI + static-frontend UI lets evaluators pick a version, pick a
question, run it and inspect the full LLM trace (analyzer × K → mediator
→ skeptic, with input / output preview and parsed JSON per call). See
[`demo/README.md`](demo/README.md) for layout and deployment notes.

## Adding a new version

When iterating to the next architecture (or a minor prompt tune):

1. Create `src/ramdocs_rag/pipelines/vN_M_<short_name>/` mirroring the
   structure of an existing version (`pipeline.py` + supporting modules
   + `prompts/` + `__init__.py` exporting the pipeline class).
2. Register it in `pipelines/registry.py` so the runner can find it.
3. Add an integration test in `tests/integration/test_vN_pipeline.py`
   using `MockLLM` — no real API spend at this stage.
4. Run two freeze runs (`make bench PIPELINE=...`); confirm
   freeze-to-freeze stability (≤ ±1 answer out of 12 on the anchors).
5. Add an entry to `runs/_journal.md` (what / hypothesis / fact /
   surprise) and dump the averaged baseline to
   `runs/_baseline/vN_M_<short_name>.json`.
6. If the version becomes a published major (replaces v3 → v4), add a
   page under `docs/journal/versions/` and a summary row in
   `docs/journal/index.html`.

## Conventions

- All documentation, code comments and docstrings are in English.
- Pipelines never edit shared `core/` types; if a version needs a new
  field, it goes through the standard Pydantic schema evolution path.
- Run artefacts in `runs/<pipeline>/<sha>_<ts>/` are immutable —
  finding a bug means a new minor (`vN.M+1`), not a re-edit of an
  existing run.
- LLM-judge is **not** used for any metric — per-doc gold annotations
  in RAMDocs are sufficient for all eleven metrics.

### Why some pipeline modules are byte-identical across minor versions

`agents.py`, `grouping.py` and `reliability.py` are intentionally copied
between minor versions of the same major (e.g. `v3.0_skeptic/agents.py`
== `v3.3_analyzer_tuned/agents.py`). This is a **freeze-immutability**
trade-off: each published baseline must remain runnable byte-for-byte at
any time, so we copy the runtime code into the version package and
iterate only on the prompts inside `prompts/*.txt`. The dual cost is a
larger codebase; the gain is that `runs/_baseline/v3.0_skeptic.json` can
be reproduced years from now without untangling a shared dependency that
has since drifted. If a *new* major version (vN+1) needs the same logic
plus changes, the runtime code branches there — never in place.

## Further reading

- [`docs/journal/index.html`](docs/journal/index.html) — HTML research
  report for evaluators. GitHub renders this file as raw HTML when
  viewed in the repository UI; for the rendered view either clone the
  repo and open `docs/journal/index.html` locally, or enable GitHub
  Pages on the repo (`Settings → Pages → Source: main / /docs`) to get a
  proper public URL.
- [`docs/architecture.md`](docs/architecture.md) — code-side architecture overview
