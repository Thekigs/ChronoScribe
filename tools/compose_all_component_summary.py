#!/usr/bin/env python3
"""Compose an auditable F/R/I/U/E summary for specialist comparisons.

Numerical specialists provide F/U on the 11 forecast dimensions.  The Qwen
semantic baseline provides R/I/E on the 23-dimension run.  We keep these
sources explicit and compute TLS with all five declared component weights.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEMANTIC = ROOT / "scores/qwenchronos/final_scores.json"

CONDITIONS = {
    "Qwen + TimesFM strict matched": "scores/qwen_timesfm_matched_norm/final_scores.json",
    "Qwen + Chronos-2 strict matched": "scores/qwen_chronos2_final/final_scores.json",
    "Qwen + Chronos-2 fixed-512 strict": "scores/qwen_chronos2_fixed512_matched_norm/final_scores.json",
    "Qwen + Chronos-2 full manifest": "scores/qwen_chronos2_live_full/final_scores.json",
    "Qwen + TimesFM full manifest": "scores/qwen_timesfm_live_full/final_scores.json",
    "Qwen + Chronos-2 fixed-512 full": "scores/qwen_chronos2_fixed512/final_scores.json",
    "Qwen + Chronos-2 matched legacy": "scores/qwen_chronos2_matched/final_scores.json",
    "Qwen + TimesFM matched legacy": "scores/qwen_timesfm_matched/final_scores.json",
    "Qwen + Sundial strict matched": "scores/qwen_sundial_strict_v2/final_scores.json",
    "Qwen + Sundial fixed-512 strict": "scores/qwen_sundial_fixed512_strict_v2/final_scores.json",
    "Qwen + Sundial full manifest": "scores/qwen_sundial_full_v2/final_scores.json",
    "Qwen + Sundial fixed-512 full": "scores/qwen_sundial_fixed512_full_v2/final_scores.json",
    "Qwen + Sundial matched legacy": "scores/qwen_sundial_legacy_v2/final_scores.json",
    "Qwen + Aurora strict matched": "scores/qwen_aurora_strict_v2/final_scores.json",
    "Qwen + Aurora fixed-512 strict": "scores/qwen_aurora_fixed512_strict_v2/final_scores.json",
    "Qwen + Aurora full manifest": "scores/qwen_aurora_full_v2/final_scores.json",
    "Qwen + Aurora fixed-512 full": "scores/qwen_aurora_fixed512_full_v2/final_scores.json",
    "Qwen + Aurora matched legacy": "scores/qwen_aurora_legacy_v2/final_scores.json",
}


def main() -> None:
    semantic = json.loads(SEMANTIC.read_text(encoding="utf-8"))["tls"]["components"]
    fixed = {key: float(semantic[key]) for key in ("R", "I", "E")}
    rows = []
    for name, rel in CONDITIONS.items():
        path = ROOT / rel
        if not path.exists():
            continue
        report = json.loads(path.read_text(encoding="utf-8"))
        comp = report.get("tls", {}).get("components", {})
        row = {
            "pipeline": name,
            "source_score": rel,
            "source_TLS_FU": report.get("tls", {}).get("TLS"),
            "F": comp.get("F"),
            "R": fixed["R"],
            "I": fixed["I"],
            "U": comp.get("U"),
            "E": fixed["E"],
            "semantic_side_source": str(SEMANTIC),
        }
        row["TLS_all_five"] = round(sum(row[c] * w for c, w in {
            "F": 0.60, "R": 0.15, "I": 0.15, "U": 0.05, "E": 0.05
        }.items()), 1)
        rows.append(row)
    out = ROOT / "scores/consolidated/all_component_summary"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps({
        "semantic_components_fixed": fixed,
        "semantic_source": str(SEMANTIC),
        "note": "F/U are from each specialist run; R/I/E are held to the same Qwen3.5+LoRA semantic baseline because historical and current manifests have different IDs.",
        "rows": rows,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fields = ["pipeline", "TLS_all_five", "source_TLS_FU", "F", "R", "I", "U", "E", "source_score", "semantic_side_source"]
    with (out / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    for row in rows:
        print(f"{row['pipeline']} | TLS5={row['TLS_all_five']} F={row['F']} R={row['R']} I={row['I']} U={row['U']} E={row['E']} | FU-source={row['source_TLS_FU']}")
    print(f"Wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
