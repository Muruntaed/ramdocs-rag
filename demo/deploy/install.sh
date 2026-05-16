#!/usr/bin/env bash
# Bootstrap: create venv, install backend + ramdocs-rag (editable).
# Idempotent — safe to re-run.

set -euo pipefail

# Demo lives as a subfolder of ramdocs_rag/. Defaults are anchored on this
# script's location and are overridable via env (e.g. on the deploy server
# where the demo lives at /opt/ramdocs-rag-demo and ramdocs_rag is checked
# out separately):
#   $SCRIPT_DIR        = <ramdocs_rag>/demo/deploy
#   $SCRIPT_DIR/..     = <ramdocs_rag>/demo            ← PROJECT_ROOT
#   $SCRIPT_DIR/../..  = <ramdocs_rag>                 ← RAMDOCS_RAG_PATH
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
RAMDOCS_RAG_PATH="${RAMDOCS_RAG_PATH:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
PY="${PYTHON:-python3.13}"

cd "$PROJECT_ROOT"

if [ ! -d .venv ]; then
  "$PY" -m venv .venv
fi

.venv/bin/pip install -U pip wheel
.venv/bin/pip install -e ./backend
.venv/bin/pip install -e "$RAMDOCS_RAG_PATH"

echo "✓ ramdocs-rag-demo installed at $PROJECT_ROOT (python: $(.venv/bin/python -V))"
