#!/usr/bin/env python3
"""Create one reproducible machine-readable summary for all completed ablations."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

MODELS = {
    "context_fixed_64": "scores/context_fixed_64/final_scores.json",
    "context_fixed_128": "scores/context_fixed_128/final_scores.json",
    "context_fixed_256": "scores/context_fixed_256/final_scores.json",
    "context_fixed_512": "scores/context_fixed_512/final_scores.json",
    "context_fixed_1024": "scores/context_fixed_1024/final_scores.json",
    "context_full_history": "scores/context_full_history/final_scores.json",
    "context_random": "scores/context_random_context/final_scores.json",
    "context_heuristic": "scores/context_heuristic_context/final_scores.json",
    "context_naive_last": "scores/context_naive_last/final_scores.json",
    "context_seasonal_naive": "scores/context_seasonal_naive/final_scores.json",
    "qwen_chronos2_full": "scores/qwen_chronos2_live_full/final_scores.json",
    "qwen_timesfm_full": "scores/qwen_timesfm_live_full/final_scores.json",
    "qwen_chronos2_fixed512": "scores/qwen_chronos2_fixed512/final_scores.json",
    "qwen_chronos2_strict": "scores/qwen_chronos2_final/final_scores.json",
    "qwen_timesfm_strict": "scores/qwen_timesfm_matched_norm/final_scores.json",
    "qwen_sundial_full": "scores/qwen_sundial_full/final_scores.json",
    "qwen_aurora_full": "scores/qwen_aurora_full/final_scores.json",
    "qwen_sundial_strict": "scores/qwen_sundial_strict/final_scores.json",
    "qwen_aurora_strict": "scores/qwen_aurora_strict/final_scores.json",
    "qwen_sundial_fixed512_full": "scores/qwen_sundial_fixed512_full_v2/final_scores.json",
    "qwen_sundial_fixed512_strict": "scores/qwen_sundial_fixed512_strict_v2/final_scores.json",
    "qwen_sundial_legacy": "scores/qwen_sundial_legacy_v2/final_scores.json",
    "qwen_aurora_fixed512_full": "scores/qwen_aurora_fixed512_full_v2/final_scores.json",
    "qwen_aurora_fixed512_strict": "scores/qwen_aurora_fixed512_strict_v2/final_scores.json",
    "qwen_aurora_legacy": "scores/qwen_aurora_legacy_v2/final_scores.json",
}


def load(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def summarize(name: str, report: dict) -> dict:
    tls = report.get("tls", {})
    dims = report.get("dimensions", {})
    row = {
        "experiment": name,
        "TLS": tls.get("TLS"),
        "F": tls.get("components", {}).get("F"),
        "U": tls.get("components", {}).get("U"),
        "n_dimensions": report.get("total_dimensions"),
    }
    for dim, summary in dims.items():
        row[f"{dim}.score"] = summary.get("composite_score")
        for metric in ("sMAPE", "MAE", "RMSE", "MASE", "WAPE", "Direction_Accuracy"):
            stat = summary.get("primary_metrics", {}).get(metric) or summary.get("related_metrics", {}).get(metric)
            if isinstance(stat, dict):
                row[f"{dim}.{metric}"] = stat.get("mean")
    return row


def main() -> None:
    rows = []
    missing = []
    for name, rel in MODELS.items():
        path = ROOT / rel
        if not path.exists():
            missing.append(rel)
            continue
        rows.append(summarize(name, load(rel)))
    out_dir = ROOT / "scores" / "consolidated"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "experiment_summary.json").write_text(json.dumps({"rows": rows, "missing": missing}, ensure_ascii=False, indent=2) + "\n")
    fields = sorted({k for row in rows for k in row})
    with (out_dir / "experiment_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    print(f"Wrote {len(rows)} experiment rows to {out_dir}")
    if missing:
        print("Missing:", ", ".join(missing))


if __name__ == "__main__":
    main()
