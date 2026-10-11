#!/usr/bin/env python3
"""Paired raw-metric comparison for canonical forecast result folders."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


METRICS = ["MAE", "RMSE", "sMAPE", "MASE", "WAPE", "Direction_Accuracy"]
HIGHER_IS_BETTER = {"Direction_Accuracy"}


def load_records(root: Path) -> dict[tuple[str, str], dict]:
    rows = {}
    for path in sorted(root.glob("*/results.json")):
        dimension = path.parent.name
        for row in json.loads(path.read_text(encoding="utf-8")):
            if not row.get("error") and isinstance(row.get("forecast_metrics"), dict):
                rows[(dimension, str(row["id"]))] = row
    return rows


def ci_mean(values: np.ndarray, rng: np.random.Generator, n_resamples: int = 10000):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"n": 0, "mean": None, "median": None, "ci95": None, "win_rate": None}
    draws = rng.choice(values, size=(n_resamples, len(values)), replace=True).mean(axis=1)
    return {
        "n": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
    }


def compare(baseline: dict, candidate: dict, seed: int):
    common = sorted(set(baseline) & set(candidate))
    rng = np.random.default_rng(seed)
    report = {"n_common": len(common), "metrics": {}, "by_dimension": {}}
    for metric in METRICS:
        base_values, cand_values, delta = [], [], []
        for key in common:
            b = baseline[key].get("forecast_metrics", {}).get(metric)
            c = candidate[key].get("forecast_metrics", {}).get(metric)
            if isinstance(b, (int, float)) and isinstance(c, (int, float)) and np.isfinite(b) and np.isfinite(c):
                base_values.append(float(b)); cand_values.append(float(c)); delta.append(float(c) - float(b))
        stat = ci_mean(np.asarray(delta), rng)
        stat["baseline_mean"] = float(np.mean(base_values)) if base_values else None
        stat["candidate_mean"] = float(np.mean(cand_values)) if cand_values else None
        stat["candidate_better_rate"] = float(np.mean(np.asarray(delta) > 0)) if metric in HIGHER_IS_BETTER else (float(np.mean(np.asarray(delta) < 0)) if delta else None)
        report["metrics"][metric] = stat

    dimensions = sorted({key[0] for key in common})
    for dim in dimensions:
        dim_report = {"n_common": sum(1 for key in common if key[0] == dim), "metrics": {}}
        dim_keys = [key for key in common if key[0] == dim]
        for metric in METRICS:
            delta = []
            base_values, cand_values = [], []
            for key in dim_keys:
                b = baseline[key].get("forecast_metrics", {}).get(metric)
                c = candidate[key].get("forecast_metrics", {}).get(metric)
                if isinstance(b, (int, float)) and isinstance(c, (int, float)) and np.isfinite(b) and np.isfinite(c):
                    base_values.append(float(b)); cand_values.append(float(c)); delta.append(float(c) - float(b))
            stat = ci_mean(np.asarray(delta), rng, n_resamples=5000)
            stat["baseline_mean"] = float(np.mean(base_values)) if base_values else None
            stat["candidate_mean"] = float(np.mean(cand_values)) if cand_values else None
            stat["candidate_better_rate"] = float(np.mean(np.asarray(delta) > 0)) if metric in HIGHER_IS_BETTER else (float(np.mean(np.asarray(delta) < 0)) if delta else None)
            dim_report["metrics"][metric] = stat
        report["by_dimension"][dim] = dim_report
    return report


def main():
    parser = argparse.ArgumentParser(description="Compare paired forecast result folders.")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidates", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    baseline_root = Path(args.baseline)
    baseline = load_records(baseline_root)
    output = {"baseline": str(baseline_root.resolve()), "candidates": {}, "seed": args.seed}
    for candidate in args.candidates:
        candidate_root = Path(candidate)
        output["candidates"][candidate] = compare(baseline, load_records(candidate_root), args.seed)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
