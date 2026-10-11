#!/usr/bin/env python3
"""Run paired forecasting ablations on the shared evaluation manifest.

This runner is intentionally forecasting-only.  It reuses the released
Chronos-2 checkpoint, the shared manifest, raw-series resolver, context
alignment, and forecast metric implementation.  The output is canonical
manifest-keyed JSON, so it can be consumed by ``scoring/score_results.py``
without the legacy-ID ambiguity present in older result folders.

Policies included here are designed to answer the main reviewer question:
how much comes from adaptive context selection rather than merely attaching
Chronos-2 to an LLM?
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from scoring.detailed_scoring import compute_all_forecast_metrics
from utils.benchmark_data import build_context, load_raw
from utils.evaluation_manifest import load_manifest_samples


DEFAULT_DIMS = [
    "D1_short_term_forecasting",
    "D2_long_horizon",
    "D3_multivariate",
    "D4_zero_shot",
    "D5_few_shot",
    "D9_robustness",
    "D10_uncertainty",
    "D14_solar",
    "D17_missing_robustness",
    "D20_horizon_degradation",
    "D22_outlier_robustness",
]
DEFAULT_POLICIES = [
    "fixed_64",
    "fixed_128",
    "fixed_256",
    "fixed_512",
    "fixed_1024",
    "full_history",
    "random_context",
    "heuristic_context",
    "naive_last",
    "seasonal_naive",
]


def _fit_horizon(values: list[float], horizon: int) -> list[float]:
    values = [float(x) for x in values]
    if len(values) >= horizon:
        return values[:horizon]
    if values:
        return values + [values[-1]] * (horizon - len(values))
    return [0.0] * horizon


def _quantiles(pipeline, context: np.ndarray, horizon: int) -> tuple[list[float], list[float], list[float]]:
    import torch

    tensor = torch.tensor(context, dtype=torch.float32).reshape(1, 1, -1)
    quantiles, medians = pipeline.predict_quantiles(tensor, prediction_length=horizon)
    if quantiles and hasattr(quantiles[0], "shape") and len(quantiles[0].shape) == 3:
        forecast = quantiles[0]
        nq = forecast.shape[2]
        return (
            forecast[0, :, 0].detach().cpu().tolist(),
            forecast[0, :, nq // 2].detach().cpu().tolist(),
            forecast[0, :, -1].detach().cpu().tolist(),
        )
    median = medians[0].squeeze().detach().cpu().tolist()
    if isinstance(median, float):
        median = [median]
    return list(median), list(median), list(median)


def _stable_index(sample_id: str, n: int, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{sample_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % n


def _select_heuristic(context: np.ndarray) -> int:
    """A deterministic, history-only context heuristic.

    It intentionally has no access to the target window.  High lag-1
    autocorrelation receives a longer context; a recent volatility shock gets
    a shorter context; otherwise the middle context is used.
    """
    x = np.asarray(context, dtype=float)
    if len(x) < 8:
        return 64
    recent = x[-min(64, len(x)):]
    long = x[-min(512, len(x)):]
    recent_std = float(np.std(recent))
    long_std = float(np.std(long)) + 1e-12
    volatility_ratio = recent_std / long_std
    centered = x - np.mean(x)
    if len(centered) > 2 and np.std(centered[:-1]) > 1e-12 and np.std(centered[1:]) > 1e-12:
        ac1 = float(np.corrcoef(centered[:-1], centered[1:])[0, 1])
    else:
        ac1 = 0.0
    if volatility_ratio >= 1.8:
        return 256
    if ac1 >= 0.85 and len(x) >= 1024:
        return 1024
    if ac1 >= 0.60 and len(x) >= 512:
        return 512
    return 128 if volatility_ratio < 0.75 else 512


def _seasonal_period(context: np.ndarray, horizon: int) -> int | None:
    x = np.asarray(context, dtype=float)
    candidates = [p for p in (12, 24, 48, 72, 96, 168, 336) if p > horizon and p < len(x) // 2]
    if not candidates:
        candidates = [p for p in (12, 24, 48, 72, 96, 168, 336) if p < len(x)]
    if not candidates:
        return None
    scores = []
    for p in candidates:
        a, b = x[p:], x[:-p]
        if len(a) < 4 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
            score = -1.0
        else:
            score = float(np.corrcoef(a, b)[0, 1])
        scores.append((score, p))
    return max(scores)[1]


def _policy_context(policy: str, data: np.ndarray, sample: dict, target: list[float], seed: int, full_cap: int):
    fixed = {"fixed_64": 64, "fixed_128": 128, "fixed_256": 256, "fixed_512": 512, "fixed_1024": 1024}
    if policy in fixed:
        requested = fixed[policy]
    elif policy == "full_history":
        requested = min(len(data), full_cap)
    elif policy == "random_context":
        requested = (64, 128, 256, 512, 1024)[_stable_index(str(sample["id"]), 5, seed)]
    elif policy == "heuristic_context":
        # The heuristic sees only a provisional tail context; it never sees GT.
        # Align first so that a raw source series containing the benchmark
        # target cannot leak the held-out window into the routing heuristic.
        provisional, _ = build_context(data, sample, target, qwen_ctx_len=full_cap)
        requested = _select_heuristic(provisional)
    else:
        requested = min(512, len(data))
    context, metadata = build_context(data, sample, target, qwen_ctx_len=requested)
    metadata = dict(metadata)
    metadata["policy"] = policy
    metadata["requested_policy_context"] = requested
    return np.asarray(context, dtype=np.float32), metadata


def _forecast_one(policy: str, pipeline, data: np.ndarray, sample: dict, seed: int, full_cap: int):
    gt = sample.get("ground_truth", {})
    target = gt.get("values", [])
    horizon = len(target) if isinstance(target, list) and target else int(sample.get("horizon", 24))
    context, alignment = _policy_context(policy, data, sample, target, seed, full_cap)
    if len(context) <= 10:
        raise RuntimeError("context_too_short")

    if policy == "naive_last":
        last = float(context[-1])
        p10 = p50 = p90 = [last] * horizon
        seasonal_lag = None
    elif policy == "seasonal_naive":
        seasonal_lag = _seasonal_period(context, horizon)
        if seasonal_lag is None:
            last = float(context[-1])
            p10 = p50 = p90 = [last] * horizon
        else:
            values = [float(context[-seasonal_lag + (i % seasonal_lag)]) for i in range(horizon)]
            p10 = p50 = p90 = values
    else:
        p10, p50, p90 = _quantiles(pipeline, context, horizon)
        seasonal_lag = None

    p10, p50, p90 = (_fit_horizon(v, horizon) for v in (p10, p50, p90))
    result = {
        "id": sample["id"],
        "dimension": sample.get("dimension", ""),
        "dataset": sample.get("dataset", "?"),
        "domain": sample.get("domain", "?"),
        "pipeline": policy,
        "policy": policy,
        "context_alignment": alignment,
        "c2_params": {"pred_len": horizon, "ctx_len": len(context)},
        "predicted_values": p50,
        "predicted_p10": p10,
        "predicted_p50": p50,
        "predicted_p90": p90,
        "error": None,
    }
    if seasonal_lag is not None:
        result["seasonal_lag"] = seasonal_lag
    if target:
        result["forecast_metrics"] = compute_all_forecast_metrics(
            np.asarray(target, dtype=float), np.asarray(p50, dtype=float),
            naive_forecast=gt.get("naive_forecast"),
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run paired Chronos-2 forecast policies.")
    parser.add_argument("--manifest", default=str(BASE / "benchmark/evaluation_manifest.json"))
    parser.add_argument("--dims", default=",".join(DEFAULT_DIMS))
    parser.add_argument("--policies", default=",".join(DEFAULT_POLICIES))
    parser.add_argument("--output-root", default=str(BASE / "results"))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--full-context-cap", type=int, default=4096)
    args = parser.parse_args()

    dims = [x.strip() for x in args.dims.split(",") if x.strip()]
    policies = [x.strip() for x in args.policies.split(",") if x.strip()]
    allowed = set(DEFAULT_POLICIES)
    unknown = set(policies) - allowed
    if unknown:
        raise SystemExit(f"Unknown policies: {sorted(unknown)}")

    plan = load_manifest_samples(BASE / "benchmark", args.manifest, dims)
    total = sum(len(v) for v in plan.values())
    print(f"Manifest: {total} samples, {len(plan)} dimensions")
    print(f"Policies: {', '.join(policies)}")

    import torch
    from chronos import Chronos2Pipeline

    print("Loading Chronos-2 once for all policies...")
    pipeline = Chronos2Pipeline.from_pretrained(str(BASE / "models/chronos-2"))

    handles = {}
    all_results = {policy: [] for policy in policies}
    for policy in policies:
        out_dir = Path(args.output_root) / f"ablation_{policy}"
        out_dir.mkdir(parents=True, exist_ok=True)
        handles[policy] = out_dir
        (out_dir / "run_config.json").write_text(json.dumps({
            "policy": policy, "manifest": str(Path(args.manifest).resolve()),
            "dims": dims, "seed": args.seed, "full_context_cap": args.full_context_cap,
            "model": str(BASE / "models/chronos-2"),
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    started = time.time()
    for dimension, samples in sorted(plan.items()):
        print(f"\n[{dimension}] {len(samples)} samples")
        per_dim = {policy: [] for policy in policies}
        for index, sample in enumerate(samples, start=1):
            sample = dict(sample)
            sample["dimension"] = dimension
            loaded = load_raw(sample)
            if loaded is None:
                data, source = None, "loader_returned_none"
            else:
                data, source = loaded
            for policy in policies:
                result = {
                    "id": sample["id"], "dimension": dimension,
                    "dataset": sample.get("dataset", "?"), "domain": sample.get("domain", "?"),
                    "pipeline": policy, "policy": policy, "data_source": source,
                }
                try:
                    if data is None or len(data) <= 10:
                        raise RuntimeError("no_usable_history")
                    result.update(_forecast_one(policy, pipeline, np.asarray(data, dtype=float), sample, args.seed, args.full_context_cap))
                except Exception as error:
                    result["error"] = str(error)[:240]
                per_dim[policy].append(result)
                all_results[policy].append(result)
            print(f"  {index:>2}/{len(samples)} {sample['id'][:55]}", flush=True)

        for policy in policies:
            out_dir = handles[policy] / dimension
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "results.json").write_text(json.dumps(per_dim[policy], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    for policy in policies:
        out_dir = handles[policy]
        with (out_dir / "all_results.jsonl").open("w", encoding="utf-8") as handle:
            for row in all_results[policy]:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        ok = [row for row in all_results[policy] if not row.get("error")]
        print(f"{policy}: {len(ok)}/{len(all_results[policy])} succeeded -> {out_dir}")
    print(f"Elapsed: {(time.time() - started) / 60:.1f} min")


if __name__ == "__main__":
    main()
