# ramdocs-rag-demo

Interactive evaluation UI for the
[`ramdocs-rag`](../README.md) multi-agent RAG prototype.

Pick a pipeline version, pick a question from the RAMDocs subset
(or load your own JSON dataset by URL), edit the question text if you
want, run it, and inspect the answer plus the full LLM trace
(retrieval → analyzer × N → mediator → skeptic) with input / output
preview and parsed JSON for each call.

The demo is a thin layer on top of the main package — the backend
imports `ramdocs_rag` as an editable install, no code is vendored.

## Relationship to the upstream project

This `demo/` directory lives **inside** the `ramdocs_rag/` project as a
subfolder. The upstream package supplies:

- the pipeline registry (v1.0 → v4) imported by `backend/main.py`;
- `core/llm.py` with the `LLMClient` protocol and the Langfuse drop-in;
- `core/types.py` (`Question`, `FinalAnswer`, ...);
- `data/ramdocs_subset.json` — the 12-question slice loaded by default.

`deploy/install.sh` resolves both `PROJECT_ROOT` (this `demo/` folder)
and `RAMDOCS_RAG_PATH` (the parent `ramdocs_rag/`) relative to the
script, so the same `install.sh` works both for local development and
on the deploy host where the demo lives under `/opt/ramdocs-rag-demo`.

## Pipeline versions exposed

The public whitelist points each major version at its **latest** minor
iteration:

| ID (API) | Public label | Notes |
|---|---|---|
| `v1.0_madam_lite` | v1.0 — MADAM-lite | Single-answer baseline |
| `v2.0_entity_first` | v2.0 — Entity-first decomposition | Multi-answer via entity grouping |
| `v3.3_analyzer_tuned` | v3 — Skeptic | Adds the four-check verifier; cheap default |
| `v4.1_promptfix` | v4 — Evidence Evaluator | Per-doc DocTrust + trust-weighted reliability; recommended default |

Internal iterations (v3.0 / 3.1 / 3.2 / 4.0) are intentionally omitted
from the API surface; see `../runs/_journal.md` and the HTML research
journal under `../docs/journal/` for the full development log.

## Layout

```
backend/   FastAPI facade over ramdocs-rag (versions, questions, run)
frontend/  Static HTML + JS, served by Caddy under /demo_1
deploy/    install.sh, systemd unit, Caddy snippet
```

## Local run

```bash
# install (creates .venv, pip-installs the backend + editable ramdocs-rag)
deploy/install.sh

# run
.venv/bin/uvicorn demo_api.main:app --host 127.0.0.1 --port 8773
# open frontend/index.html, or serve it via any static server
```

## API surface

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/versions` | Public pipeline whitelist with labels + descriptions |
| `GET` | `/api/datasets` | Available datasets (RAMDocs default + user-loaded by URL) |
| `POST` | `/api/datasets/load` | Fetch a RAMDocs-format JSON dataset by URL |
| `GET` | `/api/questions` | Items from the active dataset |
| `POST` | `/api/answer` | Run one `(version, question_id)` and return answer + trace |
| `GET` | `/api/health` | Liveness + currently registered datasets / versions |

LLM call accounting (per step latency, tokens in / out, cost, parsed
JSON output) is captured by a `TracingClient` subclass of the project's
`OpenAIClient` and returned inside the `AnswerResponse.trace` list.

## Deploy

`amorson.me/demo_1/` is served by Caddy in front of a uvicorn process
managed by systemd. The Caddy block is in `deploy/caddy.snippet`, the
unit file in `deploy/ramdocs-rag-demo.service`. A push to `main` triggers
a GitHub Action that SSH-invokes a forced-command pull script on the
server, which then:

1. `git pull` in `/opt/ramdocs-rag-demo`;
2. `pip install -e ./backend` (no-op if unchanged);
3. `rsync frontend/ → /var/www/amorson.me/demo_1/`;
4. `systemctl restart ramdocs-rag-demo`.

The demo authenticates evaluators via basic auth; the password hash is
sourced from `/etc/caddy/Caddyfile.env` on the host. `OPENAI_API_KEY`
and the optional Langfuse keys are read from
`/etc/ramdocs-rag-demo/.env` by the systemd unit.

## Environment

See `.env.example`. The only required variable is `OPENAI_API_KEY`;
Langfuse keys enable transparent LLM tracing through `langfuse.openai`
without code changes (`core/llm.py:_ensure_client` switches the client
at first use).
