#!/usr/bin/env python3
"""Run the official TimeSeriesScientist workflow on matched manifest samples.

This adapter does not reimplement TSci.  It only creates a temporary CSV whose
first ``context_length`` rows are the manifest history and whose following
rows are the held-out target, then invokes the official graph unchanged.
The target is used only after the official workflow returns, for scoring.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]


def load_env_key() -> None:
    """Map the project's local API_KEY convention without printing the key."""
    if os.environ.get("OPENAI_API_KEY"):
        return
    env_path = BASE / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("API_KEY="):
            value = line.split("=", 1)[1].strip().strip("'\"")
            if value:
                os.environ["OPENAI_API_KEY"] = value
            return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimension", default="D1_short_term_forecasting")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--official-repo",
        default="",
        help="Path to a separately checked-out TimeSeriesScientist repository.",
    )
    parser.add_argument("--model", default="gpt-4o")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.official_repo:
        raise SystemExit(
            "--official-repo is required: provide the path to a separately checked-out "
            "TimeSeriesScientist repository."
        )
    load_env_key()
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is unavailable; official TSci cannot run.")

    sys.path.insert(0, str(BASE))
    from utils.evaluation_manifest import load_manifest_samples
    from utils.benchmark_data import build_context, load_raw
    from scoring.detailed_scoring import compute_all_forecast_metrics

    plan = load_manifest_samples(
        BASE / "benchmark", BASE / "benchmark/evaluation_manifest.json", [args.dimension]
    )
    samples = plan.get(args.dimension, [])
    if not 0 <= args.sample_index < len(samples):
        raise SystemExit(f"sample-index {args.sample_index} is outside {len(samples)} samples")
    sample = samples[args.sample_index]
    target = np.asarray(sample["ground_truth"]["values"], dtype=float)
    loaded = load_raw(sample)
    if loaded is None or loaded[0] is None:
        raise SystemExit("manifest sample has no usable source series")
    source, source_label = loaded
    context, alignment = build_context(source, sample, target.tolist(), qwen_ctx_len=sample.get("context_length", 512))
    if len(context) != int(sample.get("context_length", 512)):
        raise SystemExit(f"matched context length is {len(context)}, expected {sample.get('context_length')}")

    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "matched_input.csv"
    values = np.concatenate([np.asarray(context, dtype=float), target])
    dates = pd.date_range("2000-01-01", periods=len(values), freq="h")
    pd.DataFrame({"date": dates, "OT": values}).to_csv(csv_path, index=False)

    # Import the official repository after its root is placed first on sys.path.
    repo = Path(args.official_repo).resolve()
    sys.path.insert(0, str(repo / "time_series_agent"))
    from graph.agent_graph import TimeSeriesAgentGraph
    from config.default_config import DEFAULT_CONFIG

    config = copy.deepcopy(DEFAULT_CONFIG)
    config.update({
        "data_path": str(csv_path),
        "output_dir": str(output / "official_workflow"),
        "num_slices": 1,
        "input_length": len(context),
        "horizon": len(target),
        "date_column": "date",
        "value_column": "OT",
        "debug": False,
        "verbose": False,
    })
    graph = TimeSeriesAgentGraph(config=config, model=args.model, debug=False)
    result = graph.run()

    aggregated = result.get("aggregated_results", {}) if isinstance(result, dict) else {}
    ensemble = aggregated.get("ensemble_predictions", {}) if isinstance(aggregated, dict) else {}
    prediction = ensemble.get("predictions") if isinstance(ensemble, dict) else None
    if prediction is None and isinstance(ensemble, list):
        prediction = ensemble
    if not isinstance(prediction, list) or len(prediction) < len(target):
        raise RuntimeError("official TSci output has no usable ensemble prediction")
    prediction = [float(x) for x in prediction[: len(target)]]
    metrics = compute_all_forecast_metrics(target, np.asarray(prediction, dtype=float).reshape(-1))
    report = {
        "baseline": "TimeSeriesScientist",
        "protocol_status": "official_reproduction",
        "comparability": "matched_single_sample_forecast_only",
        "dimension": args.dimension,
        "sample_id": str(sample["id"]),
        "dataset": sample.get("dataset"),
        "source_label": source_label,
        "alignment": alignment,
        "context_length": len(context),
        "horizon": len(target),
        "model": args.model,
        "forecast_metrics": metrics,
        "predicted_values": prediction,
        "target_used_only_for_scoring": True,
        "official_repo": str(repo),
        "output_path": str(output),
    }
    (output / "matched_result.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "official_raw_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: report[k] for k in ("baseline", "sample_id", "forecast_metrics")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
