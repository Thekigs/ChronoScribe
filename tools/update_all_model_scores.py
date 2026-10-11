#!/usr/bin/env python3
"""Refresh the cross-model score table without excluded MOMENT/Moirai runs.

The commercial-model rows are retained from the existing audited table; new
local specialists are read directly from their scorer reports.  Excluded
specialists are filtered by both display name and score directory so they
cannot re-enter through an old cached row.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "scores" / "all_model_scores"
NEW = {
    "qwen_sundial_full": ROOT / "scores/qwen_sundial_full/final_scores.json",
    "qwen_aurora_full": ROOT / "scores/qwen_aurora_full/final_scores.json",
    "qwen_sundial_strict": ROOT / "scores/qwen_sundial_strict/final_scores.json",
    "qwen_aurora_strict": ROOT / "scores/qwen_aurora_strict/final_scores.json",
    "qwen_sundial_fixed512_full": ROOT / "scores/qwen_sundial_fixed512_full_v2/final_scores.json",
    "qwen_sundial_fixed512_strict": ROOT / "scores/qwen_sundial_fixed512_strict_v2/final_scores.json",
    "qwen_sundial_legacy": ROOT / "scores/qwen_sundial_legacy_v2/final_scores.json",
    "qwen_aurora_fixed512_full": ROOT / "scores/qwen_aurora_fixed512_full_v2/final_scores.json",
    "qwen_aurora_fixed512_strict": ROOT / "scores/qwen_aurora_fixed512_strict_v2/final_scores.json",
    "qwen_aurora_legacy": ROOT / "scores/qwen_aurora_legacy_v2/final_scores.json",
}


def report_row(name: str, path: Path) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    tls = report.get("tls", {})
    components = tls.get("components", {})
    row = {
        "name": name,
        "score_dir": name,
        "tier": "extended_boundary" if "aurora" in name.lower() else (
            "primary_specialist" if any(token in name.lower() for token in ("chronos2", "timesfm", "sundial")) else "other"
        ),
        "TLS": tls.get("TLS"),
        "F": components.get("F"),
        "R": components.get("R"),
        "I": components.get("I"),
        "U": components.get("U"),
        "E": components.get("E"),
    }
    for dim, summary in report.get("dimensions", {}).items():
        row[dim] = summary.get("composite_score")
        for metric in ("sMAPE", "MAE", "RMSE", "MASE", "WAPE", "Direction_Accuracy"):
            stat = summary.get("primary_metrics", {}).get(metric) or summary.get("related_metrics", {}).get(metric)
            if isinstance(stat, dict) and stat.get("mean") is not None:
                row[f"{dim}.{metric}"] = stat["mean"]
    return row


def main() -> None:
    source = OUT / "all_model_scores.json"
    rows = json.loads(source.read_text(encoding="utf-8")) if source.exists() else []
    rows = [
        row for row in rows
        if "moirai" not in str(row.get("name", "")).lower()
        and "moment" not in str(row.get("name", "")).lower()
        and "moirai" not in str(row.get("score_dir", "")).lower()
        and "moment" not in str(row.get("score_dir", "")).lower()
        and str(row.get("name")) not in NEW
    ]
    for name, path in NEW.items():
        if path.exists():
            rows.append(report_row(name, path))
    rows.sort(key=lambda row: str(row.get("name", "")))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "all_model_scores.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fields = sorted({key for row in rows for key in row})
    with (OUT / "all_model_scores.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} active model rows; excluded MOMENT/Moirai rows")


if __name__ == "__main__":
    main()
