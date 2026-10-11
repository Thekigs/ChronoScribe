#!/usr/bin/env python3
"""Sensitivity analysis for TLS reference bounds and aggregation weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


LOWER_BOUNDS = {
    "sMAPE": 200.0, "MAPE": 200.0, "WAPE": 200.0, "Relative_Bias": 200.0,
    "MASE": 5.0, "OWA": 5.0, "HD_Ratio": 5.0, "OOD_Ratio": 5.0,
    "Latency_Budget_Ratio": 5.0, "RMSE": 10.0, "MAE": 10.0,
    "MedAE": 10.0, "MSE": 100.0, "latency_s": 30.0, "latency_ms": 30000.0,
    "total_tokens": 12000.0, "output_tokens": 12000.0, "input_tokens": 12000.0,
    "gpu_memory_mb": 16000.0, "Interval_Width": 5.0, "Calibration_Error": 5.0,
    "Error_vs_Horizon_Slope": 5.0, "Error_Increase": 5.0, "Recovery_Time": 5.0,
    "CRPS_approx": 1.0,
}
UPPER_BOUNDS = {
    "tokens_per_second": 50.0,
}
UPPER_METRICS = {
    "Coverage", "Stability_Score", "Direction_Accuracy", "Direction_Accuracy_window",
    "Peak_Recall", "Peak_Precision", "json_valid", "pv_is_list", "pv_all_numeric",
    "field_completeness", "valid_tool_ratio", "workflow_deps_ratio", "classification_accuracy",
    "Attribution_Accuracy", "Variable_Ranking_Accuracy", "Missing_Robustness", "Robustness_Ratio",
    "FewShot_Slope", "cause_multi_label_precision", "cause_multi_label_recall",
}


def score_metric(name: str, value: float, scale: float, fallback: float | None) -> float | None:
    if name in LOWER_BOUNDS:
        bound = LOWER_BOUNDS[name] * scale
        return max(0.0, min(100.0, 100.0 - float(value) / bound * 100.0))
    if name in UPPER_BOUNDS:
        bound = UPPER_BOUNDS[name] * scale
        return max(0.0, min(100.0, float(value) / bound * 100.0))
    if name in UPPER_METRICS or name.startswith("acc_") or name.startswith("approx_"):
        return max(0.0, min(100.0, float(value) * 100.0))
    if name.startswith("QuantileLoss_"):
        return max(0.0, min(100.0, 100.0 - float(value) / scale * 100.0))
    return fallback


def dimension_score(summary: dict, scale: float, scheme: str) -> float | None:
    metric_groups = {}
    for group in ("primary_metrics", "related_metrics"):
        for name, stat in summary.get(group, {}).items():
            if isinstance(stat, dict) and isinstance(stat.get("mean"), (int, float)):
                metric_groups.setdefault(group if group != "primary_metrics" else "forecast", {})[name] = score_metric(
                    name, stat["mean"], scale, stat.get("score_0_100"))
    # The source scorer uses all metric details by group. Reconstruct the
    # forecast group with its documented 80/20 core-vs-related treatment.
    forecast_values = []
    for group in ("primary_metrics", "related_metrics"):
        for name, stat in summary.get(group, {}).items():
            if isinstance(stat, dict) and isinstance(stat.get("mean"), (int, float)):
                forecast_values.append(score_metric(name, stat["mean"], scale, stat.get("score_0_100")))
    groups = dict(summary.get("group_scores", {}))
    if forecast_values:
        core, rest = [], []
        for group in ("primary_metrics", "related_metrics"):
            for name, stat in summary.get(group, {}).items():
                if isinstance(stat, dict) and isinstance(stat.get("mean"), (int, float)):
                    val = score_metric(name, stat["mean"], scale, stat.get("score_0_100"))
                    (core if name in {"sMAPE", "WAPE"} else rest).append(val)
        core = [v for v in core if v is not None]
        rest = [v for v in rest if v is not None]
        forecast_values = [v for v in forecast_values if v is not None]
        if core and rest:
            groups["forecast"] = sum(core) / len(core) * 0.8 + sum(rest) / len(rest) * 0.2
        elif forecast_values:
            groups["forecast"] = sum(forecast_values) / len(forecast_values)
    active = {k: v for k, v in groups.items() if isinstance(v, (int, float))}
    if not active:
        return None
    if scheme == "uniform_active":
        weights = {k: 1.0 for k in active}
    elif scheme == "forecast_heavy":
        weights = dict(summary.get("weights", {})); weights["forecast"] = weights.get("forecast", 0.0) * 1.5
    else:
        weights = dict(summary.get("weights", {}))
    weights = {k: float(v) for k, v in weights.items() if k in active and v > 0}
    total = sum(weights.values())
    return sum(active[k] * weights[k] for k in weights) / total if total else sum(active.values()) / len(active)


def tls(dimensions: dict, outer: dict, scale: float, scheme: str) -> float:
    components = {}
    for summary in dimensions.values():
        comp = summary.get("component")
        score = dimension_score(summary, scale, scheme)
        if comp and score is not None:
            components.setdefault(comp, []).append(score)
    means = {k: sum(v) / len(v) for k, v in components.items() if v}
    active_weights = {k: w for k, w in outer.items() if k in means}
    denom = sum(active_weights.values()) or 1.0
    return sum(means[k] * w for k, w in active_weights.items()) / denom


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", nargs=2, metavar=("NAME", "FINAL_SCORES"), required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    models = {name: json.loads(Path(path).read_text()) for name, path in args.model}
    outer_schemes = {
        "default": {"F": .60, "R": .15, "I": .15, "U": .05, "E": .05},
        "forecast_heavy": {"F": .75, "R": .10, "I": .05, "U": .05, "E": .05},
        "balanced": {"F": .40, "R": .20, "I": .20, "U": .10, "E": .10},
    }
    rows = []
    for scale in (0.5, 1.0, 2.0):
        for intra in ("default", "uniform_active", "forecast_heavy"):
            for outer_name, outer in outer_schemes.items():
                scores = {name: round(tls(doc.get("dimensions", {}), outer, scale, intra), 3) for name, doc in models.items()}
                ranking = sorted(scores, key=scores.get, reverse=True)
                rows.append({"reference_bound_scale": scale, "intra_dimension": intra,
                             "outer_tls_weights": outer_name, "scores": scores, "ranking": ranking})
    output = {"models": list(models), "rows": rows, "interpretation": {
        "reference_bound_scale": "0.5=tighter, 1.0=original, 2.0=looser",
        "ranking_stability": "Compare ranking across rows; TLS is renormalized over components present in each score file.",
    }}
    Path(args.output).write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(f"Wrote {args.output} ({len(rows)} sensitivity settings)")


if __name__ == "__main__":
    main()
