#!/usr/bin/env python3
"""Bootstrap summaries for an explicitly selected set of paired score runs.

This script resamples dimension-level composite scores. It is descriptive and
does not replace a paired per-instance significance test for primary forecast
metrics.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def load_composites(scores_dir: Path, model_name: str) -> list[float]:
    """Load available dimension composites from a completed score directory."""
    root = scores_dir / model_name / "per_dimension"
    values = []
    for summary_path in sorted(root.glob("*/summary.json")):
        with summary_path.open(encoding="utf-8") as handle:
            value = json.load(handle).get("composite_score")
        if value is not None:
            values.append(float(value))
    return values


def bootstrap_mean(values: list[float], n_resamples: int, seed: int) -> dict:
    """Return a deterministic bootstrap interval for a score-vector mean."""
    if len(values) < 2:
        raise ValueError("At least two dimension summaries are required.")
    rng = np.random.default_rng(seed)
    array = np.asarray(values, dtype=float)
    draws = rng.choice(array, size=(n_resamples, len(array)), replace=True).mean(axis=1)
    return {
        "mean": round(float(draws.mean()), 3),
        "ci_95": [round(float(np.percentile(draws, 2.5)), 3), round(float(np.percentile(draws, 97.5)), 3)],
        "std": round(float(draws.std()), 3),
        "n_dimensions": len(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap dimension-level score summaries")
    parser.add_argument("--scores-dir", default="scores")
    parser.add_argument("--models", nargs="+", required=True, help="Score-directory names to summarize")
    parser.add_argument("--n-resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    scores_dir = Path(args.scores_dir)
    report = {"method": "dimension_level_bootstrap", "n_resamples": args.n_resamples, "seed": args.seed, "models": {}}
    for name in args.models:
        values = load_composites(scores_dir, name)
        report["models"][name] = bootstrap_mean(values, args.n_resamples, args.seed)

    output = Path(args.output) if args.output else scores_dir / "bootstrap_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
