#!/usr/bin/env python3
"""Run the auditable bounded-revision mechanism control."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from agent_baselines.contracts import write_records
from agent_baselines.local_models import choose_device
from agent_baselines.strict_revision import StrictRevisionPipeline
from utils.evaluation_manifest import load_manifest_samples

DEFAULT_DIMS = [
    "D1_short_term_forecasting", "D2_long_horizon", "D3_multivariate",
    "D4_zero_shot", "D5_few_shot", "D9_robustness", "D10_uncertainty",
    "D14_solar", "D17_missing_robustness", "D20_horizon_degradation",
    "D22_outlier_robustness",
]


def read_existing(path: Path) -> list[dict]:
    stream = path / "all_results.jsonl"
    if not stream.exists():
        return []
    rows = []
    for line in stream.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--specialist", choices=["chronos2", "timesfm"], default="chronos2")
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="mps")
    parser.add_argument("--context-length", type=int, default=1024)
    parser.add_argument("--controller-max-new-tokens", type=int, default=256)
    parser.add_argument("--manifest", default=str(BASE / "benchmark/evaluation_manifest.json"))
    parser.add_argument("--dims", default=",".join(DEFAULT_DIMS))
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    dims = [x.strip() for x in args.dims.split(",") if x.strip()]
    plan = load_manifest_samples(BASE / "benchmark", args.manifest, dims)
    output = Path(args.output_dir)
    existing = read_existing(output) if args.resume else []
    # Checkpoint files are merged by canonical sample key. This removes an old
    # failed row when its retried version succeeds instead of creating duplicate
    # IDs in the final result stream.
    existing_by_key = {
        (str(row.get("dimension")), str(row.get("id"))): row for row in existing
    }
    # Failed rows are deliberately retried after protocol/code fixes; only
    # rows with a complete forecast are treated as resumable checkpoints.
    done = {
        (str(row.get("dimension")), str(row.get("id")))
        for row in existing_by_key.values()
        if (not row.get("error") and (row.get("forecast_metrics") or row.get("skip_reason")))
    }
    rows = list(existing_by_key.values())
    device = choose_device(args.device)
    pipeline = StrictRevisionPipeline(args.specialist, device, args.context_length, args.controller_max_new_tokens)
    config = {
        "baseline": "Qwen_LoRA_bounded_revision",
        "specialist": args.specialist,
        "policy": "local_numerical_revision_v2",
        "protocol_status": "mechanism_control_not_official_reproduction",
        "comparability": "forecast_only_no_external_event_context",
        "device": device,
        "context_length": args.context_length,
        "controller_max_new_tokens": args.controller_max_new_tokens,
        "manifest": str(Path(args.manifest).resolve()),
        "dimensions": dims,
        "future_values_used_for_execution": False,
        "resume": args.resume,
    }
    for dimension in dims:
        samples = plan.get(dimension, [])
        for sample in samples[:args.max_samples or None]:
            key = (dimension, str(sample["id"]))
            if key in done:
                continue
            row = pipeline.run_one(sample, dimension)
            existing_by_key[key] = row
            rows = list(existing_by_key.values())
            if not row.get("error") and row.get("forecast_metrics"):
                done.add(key)
        # Persist after every dimension so an interruption never discards all
        # completed samples. The final file has the same canonical schema.
        config["completed_rows"] = len(rows)
        write_records(output, rows, config)
        print(f"completed {dimension}: {len(rows)} rows", flush=True)
    config["completed_rows"] = len(rows)
    config["errors"] = sum(bool(row.get("error")) for row in rows)
    config["forecast_rows"] = sum(bool(row.get("forecast_metrics")) for row in rows)
    write_records(output, rows, config)
    print(f"Wrote {len(rows)} records to {output}")


if __name__ == "__main__":
    main()
