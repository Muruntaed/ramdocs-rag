"""Loader for the 12-question RAMDocs slice in ``data/ramdocs_subset.json``."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

from .types import Question

DEFAULT_SUBSET_PATH = Path(__file__).resolve().parents[3] / "data" / "ramdocs_subset.json"


def _resolve_path(path: str | os.PathLike | None) -> Path:
    if path is not None:
        return Path(path)
    env = os.environ.get("RAMDOCS_SUBSET_PATH")
    return Path(env) if env else DEFAULT_SUBSET_PATH


@lru_cache(maxsize=4)
def load_subset(path: str | None = None) -> tuple[Question, ...]:
    """Parse the JSON slice into an immutable, cached tuple of ``Question``."""
    resolved = _resolve_path(path)
    raw = json.loads(resolved.read_text(encoding="utf-8"))
    return tuple(Question.model_validate(item) for item in raw)


def load_by_id(question_id: str, path: str | None = None) -> Question:
    """Convenience accessor for tests and single-question debugging."""
    for q in load_subset(path):
        if q.question_id == question_id:
            return q
    raise KeyError(f"question {question_id!r} not found in subset")
