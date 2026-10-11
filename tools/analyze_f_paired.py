#!/usr/bin/env python3
"""Paired bootstrap for the forecasting (F) dimension scores.

The scorer's F component is dimension-level, so this deliberately resamples
the common forecasting dimensions rather than pretending that heterogeneous
sample-level metrics are already on one common scale.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load(path: str) -> dict[str, float]:
    doc = json.loads(Path(path).read_text())
    values = {}
    for dim, item in doc.get("tls", {}).get("dimension_contributions", {}).items():
        if item.get("component") == "F" and isinstance(item.get("score"), (int, float)):
            values[dim] = float(item["score"])
    return values


def compare(base: dict[str, float], candidate: dict[str, float], seed: int, n: int) -> dict:
    common = sorted(set(base) & set(candidate))
    if not common:
        return {"n_common_dimensions": 0, "dimensions": [], "delta_mean": None, "ci95": None}
    delta = np.asarray([candidate[d] - base[d] for d in common], dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.choice(delta, size=(n, len(delta)), replace=True).mean(axis=1)
    return {
        "n_common_dimensions": len(common),
        "dimensions": common,
        "baseline_mean_F": float(np.mean([base[d] for d in common])),
        "candidate_mean_F": float(np.mean([candidate[d] for d in common])),
        "delta_mean_F": float(np.mean(delta)),
        "median_delta_F": float(np.median(delta)),
        "ci95_delta_F": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
        "candidate_better_dimension_rate": float(np.mean(delta > 0)),
        "per_dimension_delta_F": {d: float(candidate[d] - base[d]) for d in common},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidates", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--resamples", type=int, default=20000)
    args = parser.parse_args()
    base = load(args.baseline)
    out = {"baseline": str(Path(args.baseline).resolve()), "candidates": {},
           "seed": args.seed, "resamples": args.resamples,
           "note": "F is the scorer's dimension-level forecast component; bootstrap resamples common F dimensions."}
    for candidate in args.candidates:
        out["candidates"][candidate] = compare(base, load(candidate), args.seed, args.resamples)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
