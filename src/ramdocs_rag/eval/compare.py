"""Summary table across several pipeline versions.

Reads the most recent ``runs/<pipeline>/<latest>/metrics.json`` for each
version listed on the command line.

Usage::

    python -m ramdocs_rag.eval.compare v3_3_analyzer_tuned v4_1_promptfix
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_RUNS_ROOT = Path(__file__).resolve().parents[3] / "runs"

# Metrics displayed in the summary table (column order matters).
_DISPLAY_COLS: list[tuple[str, str, str]] = [
    # (json key, header, fmt)
    ("em_any_gold", "EM", "{:.2f}"),
    ("f1_multi_answer", "F1-multi", "{:.2f}"),
    ("recall_all_gold", "Recall@gold", "{:.2f}"),
    ("misinfo_rejection", "Misinfo-rej", "{:.2f}"),
    ("noise_rejection", "Noise-rej", "{:.2f}"),
    ("citation_faithfulness", "Cite-faith", "{:.2f}"),
    ("abstention_rate", "Abstain", "{:.2f}"),
    ("avg_cost_per_question_usd", "$/Q", "{:.4f}"),
    ("avg_llm_calls_per_question", "LLM/Q", "{:.1f}"),
    ("avg_latency_s", "Lat,s", "{:.2f}"),
]


def _latest_metrics(pipeline_name: str) -> dict | None:
    pdir = _RUNS_ROOT / pipeline_name
    if not pdir.exists():
        return None
    candidates = sorted(
        (p for p in pdir.iterdir() if (p / "metrics.json").exists()),
        key=lambda p: p.name,
    )
    if not candidates:
        return None
    return json.loads((candidates[-1] / "metrics.json").read_text(encoding="utf-8"))


def render_markdown(pipelines: list[str]) -> str:
    rows: list[list[str]] = []
    headers = ["Pipeline"] + [h for _, h, _ in _DISPLAY_COLS]
    rows.append(headers)
    rows.append(["---"] * len(headers))
    for name in pipelines:
        metrics = _latest_metrics(name)
        if metrics is None:
            rows.append([name] + ["—"] * len(_DISPLAY_COLS))
            continue
        row = [name]
        for key, _, fmt in _DISPLAY_COLS:
            val = metrics.get(key)
            row.append(fmt.format(val) if isinstance(val, (int, float)) else "—")
        rows.append(row)
    return "\n".join("| " + " | ".join(r) + " |" for r in rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare metrics of frozen pipeline runs.")
    parser.add_argument("pipelines", nargs="+", help="pipeline names to compare")
    args = parser.parse_args(argv)

    md = render_markdown(args.pipelines)
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
