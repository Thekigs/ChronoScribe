#!/usr/bin/env python3
"""Merge historical and newly generated Qwen controller decisions by manifest key."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_root(root: Path) -> dict[tuple[str, str], dict]:
    rows = {}
    for path in root.glob("*/results.json"):
        try:
            values = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        dim = path.parent.name
        for row in values:
            if not isinstance(row, dict) or not row.get("step1_params"):
                continue
            rows[(dim, str(row.get("id")))] = row
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--historical-root", required=True, type=Path)
    parser.add_argument("--new-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    merged = read_root(args.historical_root)
    old_count = len(merged)
    merged.update(read_root(args.new_root))
    args.output_root.mkdir(parents=True, exist_ok=True)
    by_dim: dict[str, list[dict]] = {}
    for (dimension, _), row in sorted(merged.items()):
        by_dim.setdefault(dimension, []).append(row)
    for dimension, rows in by_dim.items():
        target = args.output_root / dimension
        target.mkdir(parents=True, exist_ok=True)
        (target / "results.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    config = {
        "historical_root": str(args.historical_root.resolve()),
        "new_root": str(args.new_root.resolve()),
        "output_root": str(args.output_root.resolve()),
        "historical_valid_decisions": old_count,
        "merged_valid_decisions": len(merged),
        "new_or_replaced_decisions": len(merged) - old_count,
    }
    (args.output_root / "run_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(config, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
