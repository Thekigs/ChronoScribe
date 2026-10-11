#!/usr/bin/env python3
"""Chronos-only ablation on the paired ChronoScribe evaluation manifest.

This baseline deliberately omits the language component. It uses the same
raw-series resolution and ground-truth-aligned context construction as the
full framework so that the only intended difference is semantic control and
evidence-conditioned response generation.
"""

import argparse
import json
import os
import sys
import time

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH = os.path.join(BASE, "benchmark")
OUT = os.path.join(BASE, "results", "ablate_c2only")
CHRONOS2_DIR = os.path.join(BASE, "models", "chronos-2")
sys.path.insert(0, BASE)

parser = argparse.ArgumentParser(description="Run the Chronos-only paired ablation.")
parser.add_argument("--dims", default="", help="Optional comma-separated manifest dimensions.")
parser.add_argument(
    "--manifest",
    default=os.path.join(BENCH, "evaluation_manifest.json"),
    help="Fixed evaluation manifest shared by every compared system.",
)
args = parser.parse_args()

# Keep command help available before users install the model runtime.
import numpy as np
import torch

from chronos import Chronos2Pipeline
from scoring.detailed_scoring import compute_all_forecast_metrics
from utils.benchmark_data import build_context, load_raw
from utils.evaluation_manifest import load_manifest_samples


def quantile_predictions(pipeline, context: np.ndarray, horizon: int) -> tuple[list[float], list[float], list[float]]:
    """Run Chronos-2 once and return P10, P50, and P90 on a shared horizon."""
    tensor = torch.tensor(context, dtype=torch.float32).reshape(1, 1, -1)
    quantiles, medians = pipeline.predict_quantiles(tensor, prediction_length=horizon)
    if quantiles and hasattr(quantiles[0], "shape") and len(quantiles[0].shape) == 3:
        forecast = quantiles[0]
        count = forecast.shape[2]
        return (
            forecast[0, :, 0].tolist(),
            forecast[0, :, count // 2].tolist(),
            forecast[0, :, -1].tolist(),
        )

    median = medians[0].squeeze().tolist()
    if isinstance(median, float):
        median = [median]
    return median, median, median


def fit_horizon(values: list[float], horizon: int) -> list[float]:
    """Keep output lengths comparable without silently dropping a failed forecast."""
    if len(values) >= horizon:
        return values[:horizon]
    if values:
        return values + [values[-1]] * (horizon - len(values))
    return [0.0] * horizon


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    dims = [item.strip() for item in args.dims.split(",") if item.strip()] or None
    test_plan = load_manifest_samples(BENCH, args.manifest, dims)

    print("Loading Chronos-2 for paired ablation...")
    pipeline = Chronos2Pipeline.from_pretrained(CHRONOS2_DIR)
    print(f"Evaluating {sum(len(samples) for samples in test_plan.values())} fixed samples.")

    all_results: list[dict] = []
    for dimension, samples in sorted(test_plan.items()):
        dimension_results: list[dict] = []
        for sample in samples:
            started_at = time.time()
            ground_truth = sample.get("ground_truth", {})
            target_values = ground_truth.get("values", [])
            horizon = len(target_values) if isinstance(target_values, list) and target_values else int(sample.get("horizon", 24))
            result = {
                "id": sample["id"],
                "dimension": dimension,
                "category": sample.get("category", ""),
                "dataset": sample.get("dataset", "?"),
                "domain": sample.get("domain", "?"),
                "pipeline": "chronos_only",
            }

            try:
                data, source = load_raw(sample)
                result["data_source"] = source
                if data is None or len(data) <= 10:
                    raise RuntimeError("no_usable_history")

                # Fixed C=512 is the Chronos-only policy; alignment is shared with
                # the full runner so it forecasts the same target window.
                context, alignment = build_context(
                    data,
                    sample,
                    target_values,
                    qwen_ctx_len=min(512, len(data)),
                )
                if len(context) <= 10:
                    raise RuntimeError("context_too_short")
                result["context_alignment"] = alignment
                result["c2_params"] = {"pred_len": horizon, "ctx_len": len(context), "policy": "fixed_512"}

                p10, p50, p90 = quantile_predictions(pipeline, context, horizon)
                result["predicted_p10"] = fit_horizon(p10, horizon)
                result["predicted_p50"] = fit_horizon(p50, horizon)
                result["predicted_p90"] = fit_horizon(p90, horizon)
                result["predicted_values"] = result["predicted_p50"]

                if target_values:
                    result["forecast_metrics"] = compute_all_forecast_metrics(
                        np.array(target_values),
                        np.array(result["predicted_values"]),
                        naive_forecast=ground_truth.get("naive_forecast"),
                    )
                result["error"] = None
            except Exception as error:
                result["error"] = str(error)[:200]
            finally:
                # End-to-end per-sample latency, excluding one-time model loading.
                result["latency_s"] = round(time.time() - started_at, 3)

            dimension_results.append(result)
            all_results.append(result)

        output_dir = os.path.join(OUT, dimension)
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "results.json"), "w", encoding="utf-8") as handle:
            json.dump(dimension_results, handle, ensure_ascii=False, indent=2)

    with open(os.path.join(OUT, "all_results.jsonl"), "w", encoding="utf-8") as handle:
        for result in all_results:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")

    succeeded = sum(result.get("error") is None for result in all_results)
    print(f"Chronos-only ablation complete: {succeeded}/{len(all_results)} successful samples.")


if __name__ == "__main__":
    main()
