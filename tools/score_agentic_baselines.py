#!/usr/bin/env python3
"""Score paired forecast metrics for agentic baseline result folders.

This scorer intentionally reports raw forecast metrics and paired differences.
It does not synthesize TLS for methods that do not share the full response
protocol.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.evaluation_manifest import load_manifest_samples


METRICS = ("MAE", "RMSE", "sMAPE", "MASE", "WAPE", "Direction_Accuracy")


def read_rows(root: Path) -> dict[tuple[str, str], dict[str, Any]]:
    stream = root / "all_results.jsonl"
    if not stream.exists():
        raise FileNotFoundError(stream)
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for line in stream.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = (str(row.get("dimension")), str(row.get("id")))
        if key in rows:
            raise ValueError(f"duplicate result key: {key}")
        rows[key] = row
    return rows


def mean_metric(rows: dict[tuple[str, str], dict[str, Any]], metric: str) -> float | None:
    values = [(row.get("forecast_metrics") or {}).get(metric) for row in rows.values()]
    values = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.mean(values)) if values else None


def paired_delta(baseline: dict[tuple[str, str], dict[str, Any]], candidate: dict[tuple[str, str], dict[str, Any]], metric: str) -> dict[str, Any]:
    common = sorted(set(baseline) & set(candidate))
    deltas = []
    left_values = []
    right_values = []
    for key in common:
        left = (baseline[key].get("forecast_metrics") or {}).get(metric)
        right = (candidate[key].get("forecast_metrics") or {}).get(metric)
        if left is None or right is None:
            continue
        if math.isfinite(float(left)) and math.isfinite(float(right)):
            deltas.append(float(right) - float(left))
            left_values.append(float(left))
            right_values.append(float(right))
    if not deltas:
        return {"n": 0, "mean_candidate_minus_baseline": None}
    values = np.asarray(deltas, dtype=float)
    # Paired bootstrap CI, deterministic seed for release reproducibility.
    rng = np.random.default_rng(20261001)
    samples = values[rng.integers(0, len(values), size=(5000, len(values)))].mean(axis=1)
    return {
        "n": int(len(values)),
        "mean_baseline_on_paired_samples": float(np.mean(left_values)),
        "mean_candidate_on_paired_samples": float(np.mean(right_values)),
        "mean_candidate_minus_baseline": float(values.mean()),
        "median_candidate_minus_baseline": float(np.median(values)),
        "ci95": [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", type=Path, default=Path("benchmark/evaluation_manifest.json"))
    parser.add_argument("--benchmark-dir", type=Path, default=Path("benchmark"))
    args = parser.parse_args()
    baseline = read_rows(args.baseline)
    manifest_plan = load_manifest_samples(args.benchmark_dir, args.manifest)
    manifest_keys = {
        (dimension, str(sample["id"]))
        for dimension, samples in manifest_plan.items()
        for sample in samples
    }
    report = {
        "baseline": str(args.baseline.resolve()),
        "candidate_reports": [],
        "metrics": list(METRICS),
        "interpretation": "Lower is better for error metrics; higher is better for Direction_Accuracy.",
        "manifest": str(args.manifest.resolve()),
        "invalid_baseline_keys": sorted([list(key) for key in set(baseline) - manifest_keys]),
    }
    for path in args.candidate:
        candidate = read_rows(path)
        paired = {m: paired_delta(baseline, candidate, m) for m in METRICS}
        dimensions = sorted({key[0] for key in set(baseline) & set(candidate)})
        report["candidate_reports"].append({
            "candidate": str(path.resolve()),
            "coverage": {
                "baseline_rows": len(baseline), "candidate_rows": len(candidate),
                "common_rows": len(set(baseline) & set(candidate)),
            },
            "invalid_candidate_keys": sorted([list(key) for key in set(candidate) - manifest_keys]),
            # Headline means must use exactly the same valid pairs. Separate
            # all-available means are retained only as a coverage diagnostic.
            "means": {
                "baseline": {m: paired[m].get("mean_baseline_on_paired_samples") for m in METRICS},
                "candidate": {m: paired[m].get("mean_candidate_on_paired_samples") for m in METRICS},
            },
            "all_available_means_not_comparable": {
                "baseline": {m: mean_metric(baseline, m) for m in METRICS},
                "candidate": {m: mean_metric(candidate, m) for m in METRICS},
            },
            "paired": paired,
            "by_dimension": {
                dimension: {
                    m: paired_delta(
                        {k: v for k, v in baseline.items() if k[0] == dimension},
                        {k: v for k, v in candidate.items() if k[0] == dimension}, m
                    ) for m in METRICS
                } for dimension in dimensions
            },
            "statistical_scope": "Exploratory sample-level paired bootstrap, 5000 resamples; not dataset-cluster significance or multiple-comparison corrected. Raw error means across heterogeneous units are diagnostic only.",
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
