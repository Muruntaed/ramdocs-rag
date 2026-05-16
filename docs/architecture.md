# Architecture overview

This document is the code-side map of the system. The narrative version
with hypotheses, freeze metrics and trade-offs lives in the
[HTML research journal](journal/index.html); this file is for engineers
opening the codebase.

## Pipeline shape (v4, current default)

```
Question + RAMDocs candidate pool (3–8 documents per question)
        │
        ▼
   Retriever  ───────────  BM25 + dense (MiniLM), top_k = 8, weights 0.5 / 0.5
        │
        ├─► Analyzer 1 ┐
        ├─► Analyzer 2 ├─► claim + entity per document   (gpt-4o-mini, K parallel calls)
        ├─► Analyzer K ┘
        │
        ├─► Evaluator 1 ┐
        ├─► Evaluator 2 ├─► DocTrust + red_flags         (gpt-4o-mini, K parallel calls)  [v4 only]
        ├─► Evaluator K ┘
        │
        ▼
   Group by entity (deterministic, no LLM — canonicalize + substring-merge)
        │
        ▼
   Reliability scoring  ──────  0.40·retrieval + 0.25·confidence + 0.35·trust − 0.10·minority   [v4 W_TRUST = 0.35]
        │
        ▼
   for each entity group above the 0.30 weight threshold:
       intra-group winner ratio ≥ 1.5?
              ├─ yes ─► Deterministic vote (no LLM)
              └─ no  ─► LLM Mediator       (gpt-4o)
        │
        ▼
   FinalAnswer (draft) — list[AnswerVariant], one per entity group
        │
        ▼
   Skeptic verification agent  (gpt-4o)
        - entity grounding
        - citation faithfulness
        - counter-evidence
        - depth-asymmetry
        - trust + red_flags as advisory input         [v4 only]
        - keep / reject each variant
        - pipeline-level fallback restores draft if Skeptic rejects everything
        │
        ▼
   FinalAnswer (verified)
```

For per-version SVG diagrams, see the corresponding pages under
`docs/journal/versions/`.

## What changed from version to version

| Version | New structural piece | What stayed |
|---|---|---|
| v1.0 | Retriever, Analyzer × K, deterministic vote, LLM mediator fallback, 5-factor reliability | — |
| v2.0 | Entity grouping (substring-merge canonicalization), multi-answer output (`list[AnswerVariant]`) | retriever, analyzer, mediator |
| v3 | Skeptic verification agent (four checks), pipeline-level abstention fallback | retriever, analyzer, entity grouping, mediator |
| v4 | **Evidence Evaluator agent** (per-doc DocTrust + red_flags), **trust-weighted reliability** (W_TRUST = 0.35), shared safety layer (auto-attached to every prompt), agent_probe CLI for single-agent debugging | retriever, entity grouping, mediator, Skeptic four-check body |

## Code layout

```
src/ramdocs_rag/
├── core/                                shared, version-agnostic primitives
│   ├── types.py                         Question, RAMDoc, Claim, AnswerVariant, FinalAnswer,
│   │                                    RunResult, RunMetrics — strict Pydantic models
│   ├── dataset.py                       load_subset(), load_by_id()
│   ├── llm.py                           LLMClient (Protocol), MockLLM, OpenAIClient,
│   │                                    pricing, Langfuse drop-in via env
│   ├── retrieval.py                     BM25 + dense MiniLM retriever, weighted fusion
│   ├── safety.py                        shared safety block auto-attached to every prompt
│   │                                    (grounding + prompt-leak refusal + output-schema lock)
│   └── trace.py                         JSONL TraceWriter
│
├── eval/                                measurement stand — never edited per version
│   ├── metrics.py                       11 deterministic metrics (EM-any-gold, EM-substring,
│   │                                    F1-multi, recall-all-gold, precision_answers,
│   │                                    misinfo / noise rejection, citation_faithfulness,
│   │                                    correct_citation, coverage, abstention_rate)
│   ├── runner.py                        pipeline registry + run + dump artefacts to runs/
│   ├── compare.py                       markdown comparison table across versions
│   └── agent_probe.py                   single-agent CLI: one (question, doc) pair for ~$0.0005
│
└── pipelines/                           versioned implementations
    ├── base.py                          class Pipeline(ABC) — the contract
    ├── registry.py                      side-effect module: imports and registers every version
    └── vN_M_*/
        ├── pipeline.py                  the main class (e.g. V41PromptFix)
        ├── agents.py                    analyze_doc(), evaluate_doc(), mediate(),
        │                                run_skeptic() — all through LLMClient (DI)
        ├── grouping.py                  entity grouping logic
        ├── reliability.py               per-version reliability formula
        ├── conflict.py                  v1.0 only — cosine claim clustering
        └── prompts/
            ├── analyzer.txt
            ├── evaluator.txt            v4 only
            ├── mediator.txt
            └── skeptic.txt              v3+
```

## Contracts

### `Pipeline` (base class)

```python
class Pipeline(ABC):
    name: str          # registry key, e.g. "v4.1_promptfix"
    version: str       # semver-ish, e.g. "4.1.0"

    def __init__(self, llm: LLMClient, config: dict | None = None) -> None: ...

    @abstractmethod
    def run(self, question: Question) -> RunResult:
        """Execute the full pipeline on one question and return final answer + per-step trace."""

    def describe(self) -> dict:
        """Snapshot of the configuration (retrieval params, models, thresholds) — dumped to config.yaml."""
```

Every version implements `run()`. Common shape:

1. Retrieval over `question.docs` (already attached to the `Question`)
2. Per-document analyzer calls in parallel (and Evaluator in v4)
3. Entity grouping (deterministic since v2)
4. Reliability scoring
5. Intra-group resolution: deterministic vote or LLM mediator
6. (v3+) Skeptic verification with fallback
7. Build `FinalAnswer` and assemble `RunResult` (answer + metrics + trace)

### `LLMClient` (Protocol)

```python
class LLMClient(Protocol):
    model: str

    def complete_json(
        self,
        *,
        system: str,
        user: str,
        schema: dict,
        schema_name: str,
        temperature: float = 0.0,
    ) -> LLMCallResult:
        """Strict-JSON LLM call. Returns parsed result + raw text + token / cost accounting."""
```

Three concrete implementations:

- `OpenAIClient` — real API calls, auto-switches to `langfuse.openai`
  when `LANGFUSE_PUBLIC_KEY` is set.
- `MockLLM` — script-driven, used in unit and integration tests.
- `TracingClient` (in `demo/backend`) — `OpenAIClient` subclass that
  captures every call into a context-var buffer for the UI.

### Why no LLM-judge

RAMDocs ships per-document gold annotations: every retrieved document
is labelled `correct` / `misinfo` / `noise` with the canonical answer
for `correct` items. All eleven metrics can be computed deterministically
from these labels — no need for a second LLM to judge correctness, which
would introduce its own noise and cost.

## Testing pyramid

| Layer | What it covers | Speed | Cost |
|---|---|---|---|
| `tests/unit/` | Pure-Python primitives — metrics math, Pydantic models, MockLLM, safety layer, conflict / reliability per version | < 8 s | $0 |
| `tests/integration/` | Full pipeline graph with `MockLLM` — every version end-to-end on a few hand-crafted scenarios | < 5 s | $0 |
| `tests/e2e/` | Real OpenAI smoke, opt-in via `@pytest.mark.e2e` | ~30 s / question | ~$0.005–0.02 |
| `make bench` | Full freeze run on the 12-question subset, used to publish baselines | ~5 min | ~$0.10–0.27 / run |

`agent_probe` (under `eval/agent_probe.py`) is the iteration-loop tool:
one analyzer / evaluator call on one `(question, doc)` pair at
~$0.0005, with a `--dry-run` mode for prompt audit without spend.

## Adding a new version

See the **Adding a new version** section in the top-level
[`../README.md`](../README.md) — keeping the procedure in one place.

## Where things are NOT

- No `LangGraph` — we deliberately use plain Python control flow for
  legibility and easy mocking. The earlier LangGraph prototype is
  preserved under `Egzakta/_legacy/ramdocs_rag_v0/` for reference only.
- No `git` repository in `ramdocs_rag/` yet; `.gitignore` is prepared
  for when we initialise.
- No CI/CD in this repo. The `demo/.github/workflows/deploy.yml` is the
  colleague's pull-trigger for the hosted demo, not this codebase.
- No Docker. The deploy runs uvicorn under systemd on the target host;
  installation is `pip install -e .` into a venv.
