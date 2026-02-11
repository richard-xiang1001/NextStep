#!/usr/bin/env python3
"""Summarize baseline metrics from a run root into CSV/JSON."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


FIELDS = [
    "run_dir",
    "method",
    "split",
    "num_trajectories",
    "success_rate",
    "avg_tokens",
    "avg_llm_calls",
    "avg_step_count",
    "parse_fail_rate",
    "fallback_rate",
    "repair_trigger_rate",
    "budget_cut_rate",
]


def _load_rows(root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for metrics_file in sorted(root.glob("*/metrics.json")):
        with metrics_file.open("r", encoding="utf-8") as f:
            metrics = json.load(f)
        row = {k: metrics.get(k) for k in FIELDS if k != "run_dir"}
        row["run_dir"] = metrics_file.parent.name
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_json(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize metrics.json files under one run root.")
    parser.add_argument("--root", required=True, help="Run root directory (contains method subdirs).")
    parser.add_argument("--out_csv", default=None, help="CSV output path. Default: <root>/summary.csv")
    parser.add_argument("--out_json", default=None, help="JSON output path. Default: <root>/summary_merged.json")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(f"Run root not found: {root}")

    rows = _load_rows(root)
    if not rows:
        raise RuntimeError(f"No metrics.json found under: {root}")

    out_csv = Path(args.out_csv) if args.out_csv else root / "summary.csv"
    out_json = Path(args.out_json) if args.out_json else root / "summary_merged.json"
    _write_csv(out_csv, rows)
    _write_json(out_json, rows)

    print(f"Wrote {len(rows)} rows")
    print(f"CSV: {out_csv}")
    print(f"JSON: {out_json}")
    print("\nmethod\tn\tok%\ttok\tcall\tstep")
    for row in rows:
        sr = float(row.get("success_rate") or 0.0) * 100.0
        print(
            f"{row.get('method')}\t{row.get('num_trajectories')}\t{sr:.2f}\t"
            f"{float(row.get('avg_tokens') or 0.0):.1f}\t"
            f"{float(row.get('avg_llm_calls') or 0.0):.2f}\t"
            f"{float(row.get('avg_step_count') or 0.0):.2f}"
        )


if __name__ == "__main__":
    main()
