#!/usr/bin/env python3
"""Score the two completed controller-alias rows against fixed-1024."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ADAPTIVE = ROOT / "results/alias_specialist_completion_20261005/all_results.jsonl"
FIXED = ROOT / "results/ablate_fixed_c1024/all_results.jsonl"
OUTPUT = ROOT / "scores/alias_specialist_completion_20261005.json"
METRICS = ("MAE", "RMSE", "sMAPE", "MASE", "Direction_Accuracy")


def rows(path: Path) -> dict[str, dict]:
    return {str((r := json.loads(line))["id"]): r for line in path.read_text().splitlines() if line.strip()}


def main() -> None:
    adaptive = rows(ADAPTIVE)
    fixed = rows(FIXED)
    scored = []
    for sid, row in adaptive.items():
        if row.get("forecast_metrics") is None or sid not in fixed:
            continue
        a = row["forecast_metrics"]
        f = fixed[sid].get("forecast_metrics") or {}
        scored.append({
            "id": sid,
            "adaptive": {m: a.get(m) for m in METRICS},
            "fixed1024": {m: f.get(m) for m in METRICS},
            "adaptive_minus_fixed1024": {
                m: (a.get(m) - f.get(m)) if a.get(m) is not None and f.get(m) is not None else None
                for m in METRICS
            },
        })
    summary = {}
    for metric in METRICS:
        vals = [r["adaptive_minus_fixed1024"][metric] for r in scored if r["adaptive_minus_fixed1024"][metric] is not None]
        summary[metric] = {"n": len(vals), "mean": sum(vals) / len(vals) if vals else None}
    out = {
        "protocol": "normalized_controller_alias_specialist_completion_v1",
        "adaptive_result": str(ADAPTIVE),
        "fixed_result": str(FIXED),
        "rows_completed": 2,
        "scored_rows": len(scored),
        "unscored_rows": [sid for sid, row in adaptive.items() if row.get("forecast_metrics") is None],
        "summary": summary,
        "paired_rows": scored,
        "interpretation": "D17 has no manifest ground-truth values and is retained as an executable but unscorable specialist completion; D20 is scored post-hoc against fixed-1024.",
    }
    OUTPUT.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(OUTPUT), "scored_rows": len(scored), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()

