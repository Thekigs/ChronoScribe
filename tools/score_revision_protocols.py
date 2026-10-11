#!/usr/bin/env python3
"""Audit bounded-revision results under explicit intent-to-treat protocols.

The original paired report includes no-op fallbacks, which is the correct
deployment-level (intent-to-treat) view but hides how often the controller
actually emitted a valid revision object. This report keeps that view and adds
valid-directive and exact-target-alignment strata without silently changing
the headline result.
"""

from __future__ import annotations

import json
import math
import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / "results/agentic/revision_chronos2_fixed1024_v2_full/all_results.jsonl"
BASELINE = ROOT / "results/e2e_full_simple_fixed1024_mps/all_results.jsonl"
OUTPUT = ROOT / "scores/agentic/revision_protocol_strata_20261005.json"
METRICS = ("MAE", "RMSE", "sMAPE", "MASE", "Direction_Accuracy")


def load(path: Path) -> dict[tuple[str, str], dict]:
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = (str(row.get("dimension")), str(row.get("id")))
        if key in rows:
            raise ValueError(f"duplicate key in {path}: {key}")
        rows[key] = row
    return rows


def finite(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def ci(values: list[float], seed: int = 20261005) -> list[float] | None:
    if not values:
        return None
    x = np.asarray(values, dtype=float)
    if len(x) == 1:
        return [float(x[0]), float(x[0])]
    rng = np.random.default_rng(seed)
    samples = x[rng.integers(0, len(x), size=(5000, len(x)))].mean(axis=1)
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def score(base: dict, candidate: dict, keys: list[tuple[str, str]]) -> dict:
    out = {"n": len(keys), "metrics": {}}
    for metric in METRICS:
        delta = []
        for key in keys:
            b = base[key].get("forecast_metrics", {}).get(metric)
            c = candidate[key].get("forecast_metrics", {}).get(metric)
            if finite(b) and finite(c):
                delta.append(float(c) - float(b))
        out["metrics"][metric] = {
            "n": len(delta),
            "mean_candidate_minus_baseline": float(np.mean(delta)) if delta else None,
            "median_candidate_minus_baseline": float(np.median(delta)) if delta else None,
            "ci95_sample_bootstrap": ci(delta),
            "candidate_better_rate": (
                float(np.mean(np.asarray(delta) < 0)) if delta and metric != "Direction_Accuracy"
                else float(np.mean(np.asarray(delta) > 0)) if delta else None
            ),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, default=CANDIDATE)
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    candidate_path = args.candidate if args.candidate.is_absolute() else ROOT / args.candidate
    baseline_path = args.baseline if args.baseline.is_absolute() else ROOT / args.baseline
    output_path = args.output if args.output.is_absolute() else ROOT / args.output
    base = load(baseline_path)
    candidate = load(candidate_path)
    common = sorted(set(base) & set(candidate))
    forecast = [k for k in common if candidate[k].get("forecast_metrics")]
    valid = [k for k in forecast if candidate[k].get("revision_audit", {}).get("valid") is True]
    exact = [k for k in forecast if candidate[k].get("context_alignment", {}).get("context_aligned") == "gt_window"]
    valid_exact = [k for k in valid if k in exact]
    applied = [k for k in valid_exact if candidate[k].get("revision_audit", {}).get("applied") is True]

    raw_equal = []
    raw_max_abs = []
    for key in forecast:
        raw = np.asarray(candidate[key].get("raw_predicted_values", []), dtype=float)
        simple = np.asarray(base[key].get("predicted_values", []), dtype=float)
        if len(raw) == len(simple) and len(raw):
            diff = float(np.max(np.abs(raw - simple)))
            raw_max_abs.append(diff)
            raw_equal.append(diff <= 1e-5)

    out = {
        "protocol": "bounded_revision_protocol_strata_v1",
        "candidate": str(candidate_path),
        "baseline": str(baseline_path),
        "future_values_used_for_execution": False,
        "coverage": {
            "candidate_rows": len(candidate),
            "baseline_rows": len(base),
            "common_rows": len(common),
            "forecast_rows": len(forecast),
            "valid_directive_rows": len(valid),
            "exact_target_alignment_rows": len(exact),
            "valid_exact_rows": len(valid_exact),
            "applied_valid_exact_rows": len(applied),
        },
        "raw_trajectory_equivalence": {
            "compared_rows": len(raw_max_abs),
            "equal_within_1e-5_rows": int(sum(raw_equal)),
            "max_abs_difference": max(raw_max_abs) if raw_max_abs else None,
        },
        "strata": {
            "deployment_intent_to_treat_all_forecast_rows": score(base, candidate, forecast),
            "valid_directive_all_alignment": score(base, candidate, valid),
            "valid_directive_exact_target_alignment": score(base, candidate, valid_exact),
            "applied_revision_exact_target_alignment": score(base, candidate, applied),
        },
        "interpretation": (
            "The all-forecast stratum is the deployment intent-to-treat result and includes no-op fallbacks. "
            "The valid-directive strata are diagnostic and do not replace the intent-to-treat estimate. "
            "Exact-target alignment excludes tail_before_gt rows."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output_path), "coverage": out["coverage"]}, indent=2))


if __name__ == "__main__":
    main()
