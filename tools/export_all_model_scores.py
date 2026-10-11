#!/usr/bin/env python3
"""Export every scored model into one machine-readable comparison table."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "scores" / "all_model_scores"


def dimension_number(name: str) -> int | None:
    if not name.startswith("D"):
        return None
    try:
        return int(name.split("_", 1)[0][1:])
    except (ValueError, IndexError):
        return None


def main() -> None:
    rows = []
    for path in sorted((ROOT / "scores").glob("*/final_scores.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        model = report.get("model") or path.parent.name
        tls = report.get("tls", {})
        components = tls.get("components", {})
        row = {
            "model": model,
            "score_dir": str(path.parent.relative_to(ROOT)),
            "total_dimensions": report.get("total_dimensions"),
            "TLS": tls.get("TLS"),
            "F": components.get("F"),
            "R": components.get("R"),
            "I": components.get("I"),
            "U": components.get("U"),
            "E": components.get("E"),
        }
        contributions = tls.get("dimension_contributions", {})
        for dim, value in contributions.items():
            row[f"{dim}.score"] = value.get("score") if isinstance(value, dict) else value
        rows.append(row)

    rows.sort(key=lambda r: (-float(r["TLS"] or -1), str(r["model"])))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "full_model_scores.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    fields = sorted({key for row in rows for key in row})
    preferred = ["model", "score_dir", "total_dimensions", "TLS", "F", "R", "I", "U", "E"]
    fields = preferred + [key for key in fields if key not in preferred]
    with (OUT / "full_model_scores.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} scored runs to {OUT}")


if __name__ == "__main__":
    main()
