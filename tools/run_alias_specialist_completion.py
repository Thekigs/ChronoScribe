#!/usr/bin/env python3
"""Complete specialist forecasts for controller-alias rows.

The controller decisions have already been generated and normalized.  This
small runner avoids another 9B language-model call and evaluates only the
Chronos-2 specialist on the two rows whose JSON aliases were repaired.
Targets are loaded only for post-hoc scoring.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from chronos import Chronos2Pipeline
from scoring.detailed_scoring import compute_all_forecast_metrics
from utils.benchmark_data import build_context, load_raw
from utils.evaluation_manifest import load_manifest_samples


TARGETS = {
    "D20_horizon_degradation__samples__000112__F101901_hdeg_h96_Exchange_0_h96",
    "D17_missing_robustness__samples__000106__Z200038_missing_ETTh1_MUFL_h24",
}


def load_decisions(path: Path) -> dict[tuple[str, str], dict]:
    decisions = {}
    for result_path in sorted(path.glob("*/results.json")):
        rows = json.loads(result_path.read_text(encoding="utf-8"))
        for row in rows:
            if str(row.get("id")) in TARGETS and isinstance(row.get("step1_params"), dict):
                # Controller-only files do not repeat the dimension field; the
                # parent directory is the canonical dimension identifier.
                dimension = str(row.get("dimension") or result_path.parent.name)
                decisions[(dimension, str(row["id"]))] = row["step1_params"]
    return decisions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--controller-dir",
        type=Path,
        default=ROOT / "results/controller_fresh_full_strict_normalized_20261003",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/alias_specialist_completion_20261005",
    )
    args = parser.parse_args()

    plan = load_manifest_samples(ROOT / "benchmark", ROOT / "benchmark/evaluation_manifest.json")
    decisions = load_decisions(args.controller_dir)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(json.dumps({"device": device, "mps_available": bool(torch.backends.mps.is_available())}))

    print("Loading Chronos-2 specialist...")
    specialist = Chronos2Pipeline.from_pretrained(ROOT / "models/chronos-2")
    rows = []
    started = time.time()
    for dimension, samples in sorted(plan.items()):
        for sample in samples:
            sid = str(sample.get("id"))
            if sid not in TARGETS:
                continue
            key = (dimension, sid)
            directive = decisions.get(key)
            if directive is None:
                raise RuntimeError(f"missing normalized decision for {dimension}/{sid}")
            target_values = sample.get("ground_truth", {}).get("values")
            target = np.asarray(target_values, dtype=float) if isinstance(target_values, list) else None
            horizon = len(target) if target is not None else int(sample.get("horizon", directive["prediction_length"]))
            loaded = load_raw(sample)
            if loaded is None or loaded[0] is None:
                raise RuntimeError(f"missing source series for {sid}")
            data, source = loaded
            context, alignment = build_context(
                data, sample, target.tolist() if target is not None else [], qwen_ctx_len=directive["context_length"]
            )
            tensor = torch.tensor(context, dtype=torch.float32).reshape(1, 1, -1)
            quantiles, median = specialist.predict_quantiles(
                tensor, prediction_length=horizon
            )
            if isinstance(quantiles, list) and quantiles:
                q = quantiles[0]
                p10 = q[0, :, 0].detach().cpu().numpy().tolist()
                p50 = q[0, :, q.shape[2] // 2].detach().cpu().numpy().tolist()
                p90 = q[0, :, -1].detach().cpu().numpy().tolist()
            else:
                p50 = median[0].squeeze().detach().cpu().numpy().tolist()
                p10 = p90 = p50
            metrics = None
            status = "unscorable_no_manifest_target"
            if target is not None:
                metrics = compute_all_forecast_metrics(
                    target, np.asarray(p50, dtype=float), naive_forecast=sample["ground_truth"].get("naive_forecast")
                )
                status = "scored_manifest_target"
            rows.append({
                    "id": sid,
                    "dimension": dimension,
                    "controller_source": "normalized_cached_qwen",
                    "controller_decision": directive,
                    "specialist": "chronos-2",
                    "device": device,
                    "source": source,
                    "context_alignment": alignment,
                    "history_sha256": hashlib.sha256(np.asarray(context, dtype="<f8").tobytes()).hexdigest(),
                    "predicted_values": p50,
                    "predicted_p10": p10,
                    "predicted_p90": p90,
                    "forecast_metrics": metrics,
                    "forecast_evaluation_status": status,
                    "target_used_only_for_scoring": True,
                })
            print(f"completed {dimension}/{sid}: {metrics}")

    if len(rows) != len(TARGETS):
        raise RuntimeError(f"expected {len(TARGETS)} rows, got {len(rows)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "all_results.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    (args.output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "protocol": "normalized_controller_alias_specialist_completion_v1",
                "controller_dir": str(args.controller_dir),
                "device": device,
                "mps_available": bool(torch.backends.mps.is_available()),
                "target_used_only_for_scoring": True,
                "rows": len(rows),
                "elapsed_s": round(time.time() - started, 3),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output_dir), "rows": len(rows), "device": device}))


if __name__ == "__main__":
    main()
