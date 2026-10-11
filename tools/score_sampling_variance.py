#!/usr/bin/env python3
"""Summarize run-level variance from the stochastic controller probe."""

from __future__ import annotations

import json
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results/sampling_variance_probe_20261005/all_results.jsonl"
OUTPUT = ROOT / "scores/sampling_variance_probe_20261005.json"
METRICS = ("MAE", "RMSE", "sMAPE", "MASE", "Direction_Accuracy")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["dimension"], row["id"])].append(row)
    per_sample = []
    for key, values in sorted(grouped.items()):
        valid = [v for v in values if v.get("directive_valid")]
        item = {
            "dimension": key[0],
            "id": key[1],
            "repeats": len(values),
            "valid_repeats": len(valid),
            "directive_unique": len({json.dumps(v["directive"], sort_keys=True) for v in valid}),
            "context_lengths": sorted({v["directive"]["context_length"] for v in valid if isinstance(v.get("directive"), dict) and "context_length" in v["directive"]}),
            "metrics": {},
        }
        for metric in METRICS:
            metric_values = [v["forecast_metrics"][metric] for v in valid if isinstance(v.get("forecast_metrics"), dict) and v["forecast_metrics"].get(metric) is not None]
            item["metrics"][metric] = {
                "n": len(metric_values),
                "mean": float(np.mean(metric_values)) if metric_values else None,
                "std": float(np.std(metric_values, ddof=1)) if len(metric_values) > 1 else None,
                "min": float(np.min(metric_values)) if metric_values else None,
                "max": float(np.max(metric_values)) if metric_values else None,
            }
        per_sample.append(item)

    summary = {
        "samples": len(per_sample),
        "repeats_per_sample": sorted({x["repeats"] for x in per_sample}),
        "valid_directive_rate": float(sum(x["valid_repeats"] for x in per_sample) / max(len(rows), 1)),
        "samples_with_directive_variation": sum(x["directive_unique"] > 1 for x in per_sample),
        "samples_with_context_length_variation": sum(len(x["context_lengths"]) > 1 for x in per_sample),
        "metric_run_level": {},
    }
    for metric in METRICS:
        stds = [x["metrics"][metric]["std"] for x in per_sample if x["metrics"][metric]["std"] is not None]
        means = [x["metrics"][metric]["mean"] for x in per_sample if x["metrics"][metric]["mean"] is not None]
        summary["metric_run_level"][metric] = {
            "samples_with_metric_variance": len(stds),
            "mean_per_sample_std": float(np.mean(stds)) if stds else None,
            "max_per_sample_std": float(np.max(stds)) if stds else None,
            "mean_of_sample_means": float(np.mean(means)) if means else None,
        }
    out = {
        "protocol": "stochastic_controller_and_specialist_variance_probe_v1",
        "input": str(args.input),
        "temperature": rows[0].get("temperature") if rows else None,
        "top_p": rows[0].get("top_p") if rows else None,
        "controller_receives_future_values": False,
        "summary": summary,
        "per_sample": per_sample,
        "interpretation": "This estimates variance under stochastic controller decoding; it is separate from the greedy production result and does not replace it.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
