#!/usr/bin/env python3
"""Execute Chronos-2 forecasts from a fresh controller directory.

This is deliberately forecast-only: it consumes only schema-valid controller
decisions that have a matching history hash and never generates a response
with Qwen.  It is used to isolate adaptive routing from response composition.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scoring.detailed_scoring import compute_all_forecast_metrics
from utils.benchmark_data import build_context, load_raw
from utils.evaluation_manifest import load_manifest_samples
from agent_baselines.local_models import LocalSpecialist


def valid(d: object) -> bool:
    return (isinstance(d, dict) and type(d.get("use_chronos2")) is bool
            and d.get("use_chronos2") is True
            and type(d.get("prediction_length")) is int and d["prediction_length"] > 0
            and type(d.get("context_length")) is int and d["context_length"] > 0)


def load_decisions(root: Path) -> dict[tuple[str, str], dict]:
    out: dict[tuple[str, str], dict] = {}
    for path in sorted(root.glob("*/results.json")):
        rows = json.loads(path.read_text(encoding="utf-8"))
        for row in rows:
            d = row.get("step1_params")
            if valid(d) and row.get("directive_valid") is True and row.get("history_sha256"):
                key = (path.parent.name, str(row.get("id")))
                if key in out:
                    raise ValueError(f"duplicate controller identity: {key}")
                out[key] = row
    return out


def fit(values: object, horizon: int) -> list[float]:
    arr = [float(x) for x in values] if isinstance(values, (list, tuple)) else []
    return (arr[:horizon] if len(arr) >= horizon else arr + ([arr[-1]] * (horizon - len(arr)) if arr else [0.0] * horizon))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--controller", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--manifest", default=str(ROOT / "benchmark/evaluation_manifest.json"))
    ap.add_argument("--device", choices=("mps", "cpu", "auto"), default="mps")
    ap.add_argument("--fixed-context-length", type=int,
                    help="Reuse identical fresh decisions but override only numerical context length.")
    args = ap.parse_args()
    if args.fixed_context_length is not None and not 11 <= args.fixed_context_length <= 4096:
        ap.error("fixed context must be between 11 and 4096")

    import torch

    device = args.device
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS required but unavailable")

    dims = ["D1_short_term_forecasting", "D2_long_horizon", "D3_multivariate",
            "D4_zero_shot", "D5_few_shot", "D9_robustness", "D10_uncertainty",
            "D14_solar", "D17_missing_robustness", "D20_horizon_degradation",
            "D22_outlier_robustness"]
    plan = load_manifest_samples(ROOT / "benchmark", args.manifest, dims)
    controller_config = json.loads((args.controller / "run_config.json").read_text())
    manifest_hash = hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest()
    if controller_config.get("status") != "complete":
        raise RuntimeError("Controller run is incomplete; refusing a final forecast run")
    if controller_config.get("manifest_sha256") != manifest_hash:
        raise RuntimeError("Controller manifest hash mismatch")
    decisions = load_decisions(args.controller)
    specialist = LocalSpecialist("chronos2", device)
    rows = []
    for dimension, samples in sorted(plan.items()):
        for sample in samples:
            key = (dimension, str(sample["id"]))
            row = {"id": str(sample["id"]), "dimension": dimension,
                   "specialist": "chronos2", "controller_source": "fresh_manifest_qwen",
                   "controller_root": str(args.controller.resolve()),
                   "future_values_provided_to_controller": False}
            try:
                decision = decisions.get(key)
                if decision is None and args.fixed_context_length is None:
                    raise RuntimeError("missing_fresh_controller_decision")
                loaded = load_raw(sample)
                if loaded is None or loaded[0] is None:
                    raise RuntimeError("no_usable_history")
                data, source = loaded
                target = sample.get("ground_truth", {}).get("values", [])
                horizon = len(target) if isinstance(target, list) and target else int(sample.get("horizon", 24))
                controller_context, _ = build_context(data, sample, target, qwen_ctx_len=min(4096, len(data)))
                requested_context = (args.fixed_context_length if args.fixed_context_length is not None
                                     else int(decision["step1_params"]["context_length"]))
                context, alignment = build_context(data, sample, target, qwen_ctx_len=requested_context)
                controller_hash = hashlib.sha256(np.asarray(controller_context, dtype="<f8").tobytes()).hexdigest()
                if decision is not None and controller_hash != decision.get("history_sha256"):
                    raise RuntimeError("history_hash_mismatch")
                if len(context) > 4096:
                    context = context[-4096:]
                    alignment = dict(alignment, ctx_len=len(context), specialist_cap_applied=True)
                if len(context) <= 10 or not np.isfinite(context).all():
                    raise RuntimeError("invalid_execution_history")
                p10, p50, p90 = specialist.forecast(np.asarray(context, dtype=np.float32), horizon)
                if any(len(v) != horizon or not np.isfinite(v).all() for v in (p10, p50, p90)):
                    raise RuntimeError("invalid_specialist_forecast")
                proposed = decision["step1_params"] if decision is not None else None
                row.update({"data_source": source, "controller_decision": proposed,
                            "controller_output": proposed,
                            "controller_generation_audit": decision.get("generation_audit", {}) if decision else {},
                            "history_sha256": controller_hash, "execution_context_sha256": hashlib.sha256(np.asarray(context, dtype="<f8").tobytes()).hexdigest(),
                            "context_alignment": alignment,
                            "c2_params": {"pred_len": horizon, "ctx_len": len(context)},
                            "proposed_context_length": int(decision["step1_params"]["context_length"]) if decision else None,
                            "requested_context_length": requested_context,
                            "executed_context_length": len(context), "predicted_values": p50,
                            "predicted_p10": p10, "predicted_p50": p50, "predicted_p90": p90,
                            "forecast_metrics": compute_all_forecast_metrics(
                                np.asarray(target, dtype=float), np.asarray(p50, dtype=float),
                                naive_forecast=sample.get("ground_truth", {}).get("naive_forecast")) if target else None})
            except Exception as exc:
                row["error"] = str(exc)[:240]
            rows.append(row)
        print(f"completed {dimension}: {sum(r.get('dimension') == dimension for r in rows)}", flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "all_results.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    config = {"protocol": "fresh_controller_forecast_only_v1", "manifest": str(Path(args.manifest).resolve()),
              "controller": str(args.controller.resolve()), "device": device, "dimensions": dims,
              "manifest_sha256": manifest_hash, "enable_thinking": controller_config.get("enable_thinking"),
              "status": "complete", "quantile_levels": [0.1, 0.5, 0.9],
              "policy": "fixed" if args.fixed_context_length is not None else "adaptive",
              "fixed_context_length": args.fixed_context_length,
              "scope": "numerical_policy_control_with_identical_cached_fresh_controller; not response_generation",
              "future_values_provided_to_controller": False, "rows": len(rows),
              "decision_rows": len(decisions), "forecast_rows": sum(bool(r.get("forecast_metrics")) for r in rows)}
    (args.output / "run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(config, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
