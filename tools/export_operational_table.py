#!/usr/bin/env python3
"""Export the verified forecast-only result into a compact primary-table CSV."""

from __future__ import annotations

import csv
import json
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scores/operational_forecasting_verified_20261003.json"
OUTPUT = ROOT / "scores/operational_forecasting_verified_20261003.csv"
METRICS = ("MAE", "RMSE", "sMAPE", "MASE_history_seasonal", "Direction_Accuracy")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=SOURCE, help="Verified operational JSON file.")
    parser.add_argument("--output", type=Path, default=OUTPUT, help="Destination CSV file.")
    args = parser.parse_args()

    if not args.input.exists():
        parser.error(
            f"input file does not exist: {args.input}. "
            "Run tools/score_operational_forecasts.py first."
        )
    data = json.loads(args.input.read_text(encoding="utf-8"))
    rows = []
    for name, summary in data["all_model_matched"].items():
        row = {"model": name, "common_rows": data["all_model_common_rows"]}
        for metric in METRICS:
            value = summary.get(metric, {})
            row[f"{metric}_n"] = value.get("n")
            row[f"{metric}_mean"] = value.get("mean")
            row[f"{metric}_median"] = value.get("median")
        rows.append(row)
    if not rows:
        parser.error(f"no model rows found in {args.input}")
    fields = list(rows[0])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(args.output)


if __name__ == "__main__":
    main()
