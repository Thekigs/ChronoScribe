#!/usr/bin/env python3
"""Audit controller-decision coverage under exact and normalized ID matching."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from utils.evaluation_manifest import load_manifest_samples


def variants(value: str) -> set[str]:
    text = str(value)
    vals = {text, text.rsplit("__", 1)[-1]}
    vals.update(re.sub(r"^v\d+_", "", v) for v in list(vals))
    return vals


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--controller-root", default="results/qwenchronos")
    p.add_argument("--output", default="scores/controller_coverage.json")
    args = p.parse_args()
    decisions = {}
    for path in sorted(Path(args.controller_root).glob("*/results.json")):
        try:
            rows = json.loads(path.read_text())
        except Exception:
            continue
        for row in rows:
            if isinstance(row.get("step1_params"), dict):
                decisions[(path.parent.name, str(row.get("id")))] = row["step1_params"]
    dims = ["D1_short_term_forecasting", "D2_long_horizon", "D3_multivariate", "D4_zero_shot", "D5_few_shot", "D9_robustness", "D10_uncertainty", "D14_solar", "D17_missing_robustness", "D20_horizon_degradation", "D22_outlier_robustness"]
    plan = load_manifest_samples(BASE / "benchmark", BASE / "benchmark/evaluation_manifest.json", dims)
    counts = Counter(); unmatched = []
    for dim, samples in sorted(plan.items()):
        for sample in samples:
            sid = str(sample["id"])
            if (dim, sid) in decisions:
                kind = "exact"
            else:
                suffix = [v for (d, old), v in decisions.items() if d == dim and str(old).endswith(f"__{sid}")]
                reverse = [v for (d, old), v in decisions.items() if d == dim and sid.endswith(f"__{old}")]
                normalized = [v for (d, old), v in decisions.items() if d == dim and variants(sid).intersection(variants(old))]
                if len(suffix) == 1: kind = "suffix"
                elif len(reverse) == 1: kind = "reverse_suffix"
                elif len(normalized) == 1: kind = "normalized"
                else:
                    kind = "unmatched"
                    unmatched.append({"dimension": dim, "id": sid})
            counts[kind] += 1
    result = {"controller_root": str(Path(args.controller_root).resolve()), "total_manifest_samples": sum(counts.values()), "counts": dict(counts), "unmatched": unmatched}
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"total": result["total_manifest_samples"], "counts": result["counts"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()

