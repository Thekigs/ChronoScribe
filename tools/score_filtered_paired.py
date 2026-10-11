#!/usr/bin/env python3
"""Paired forecast comparison after an auditable source/protocol filter."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


METRICS = ("MAE", "RMSE", "sMAPE", "MASE", "WAPE", "Direction_Accuracy")


def read(path: Path, sources: set[str] | None, require_executable: bool = False) -> dict[tuple[str, str], dict]:
    rows = {}
    for line in (path / "all_results.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if sources is not None and row.get("controller_source") not in sources:
            continue
        if require_executable:
            d = row.get("controller_output")
            if not isinstance(d, dict) or type(d.get("use_chronos2")) is not bool:
                continue
            if d.get("use_chronos2") is True and not all(type(d.get(k)) is int and d[k] > 0 for k in ("prediction_length", "context_length")):
                continue
            if d.get("use_chronos2") is False and not (isinstance(d.get("direct_reply"), str) and d["direct_reply"].strip()):
                continue
        if not row.get("forecast_metrics") or row.get("error"):
            continue
        rows[(str(row.get("dimension")), str(row.get("id")))] = row
    return rows


def paired(left: dict, right: dict, metric: str) -> dict:
    values = []
    for key in sorted(set(left) & set(right)):
        a = left[key].get("forecast_metrics", {}).get(metric)
        b = right[key].get("forecast_metrics", {}).get(metric)
        if a is None or b is None:
            continue
        if math.isfinite(float(a)) and math.isfinite(float(b)):
            values.append(float(b) - float(a))
    if not values:
        return {"n": 0}
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(20261001)
    boot = v[rng.integers(0, len(v), size=(10000, len(v)))].mean(axis=1)
    return {
        "n": int(len(v)),
        "mean_right_minus_left": float(v.mean()),
        "median_right_minus_left": float(np.median(v)),
        "ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
        "right_better_rate": float(np.mean(v < 0)) if metric != "Direction_Accuracy" else float(np.mean(v > 0)),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--left", required=True, type=Path)
    p.add_argument("--right", required=True, type=Path)
    p.add_argument("--left-sources", default="")
    p.add_argument("--require-executable", action="store_true", help="Exclude empty/invalid cached directives.")
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    sources = {x.strip() for x in args.left_sources.split(",") if x.strip()} or None
    left = read(args.left, sources, args.require_executable)
    right = read(args.right, None)
    common = sorted(set(left) & set(right))
    summary = {
        "left": str(args.left.resolve()),
        "right": str(args.right.resolve()),
        "left_controller_sources": sorted(sources) if sources else "all",
        "coverage": {"left_valid_rows": len(left), "right_valid_rows": len(right), "common_valid_rows": len(common)},
        "left_context_lengths": sorted({r.get("c2_params", {}).get("ctx_len") for r in left.values() if r.get("c2_params", {}).get("ctx_len") is not None}),
        "paired": {metric: paired(left, right, metric) for metric in METRICS},
        "interpretation": "right_minus_left; error metrics lower is better, Direction_Accuracy higher is better; bootstrap is sample-level and exploratory.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
