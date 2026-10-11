#!/usr/bin/env python3
"""Split specialist comparisons into primary and extended tables.

Primary: Chronos-2, TimesFM, and Sundial.
Extended: Aurora as an additional strong-model/boundary-case baseline.
Raw score reports remain untouched.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scores/consolidated/all_component_summary/summary.json"
OUT = ROOT / "scores/consolidated/specialist_tables"


def write_table(name: str, rows: list[dict], tier: str) -> None:
    target = OUT / name
    target.mkdir(parents=True, exist_ok=True)
    payload = {
        "tier": tier,
        "rows": rows,
        "note": (
            "Primary table contains Chronos-2, TimesFM, and Sundial. "
            "Aurora is retained as an extended strong-model/boundary-case baseline."
        ),
    }
    (target / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fields = ["pipeline", "TLS_all_five", "source_TLS_FU", "F", "R", "I", "U", "E", "source_score", "semantic_side_source"]
    with (target / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    rows = source["rows"]
    primary = [row for row in rows if "Aurora" not in row["pipeline"]]
    extended = [row for row in rows if "Aurora" in row["pipeline"]]
    write_table("primary", primary, "primary")
    write_table("extended", extended, "extended")
    print(f"Primary rows: {len(primary)}")
    print(f"Extended rows: {len(extended)}")


if __name__ == "__main__":
    main()
