"""JSONL trace logger. One run of one question → one file."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class TraceWriter:
    """JSONL logger that can be used as a context manager or a plain object."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")

    def log(self, *, node: str, event: str, data: dict[str, Any] | None = None) -> None:
        record = {
            "ts": datetime.now(UTC).isoformat(),
            "node": node,
            "event": event,
            "data": data or {},
        }
        self._fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> TraceWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
