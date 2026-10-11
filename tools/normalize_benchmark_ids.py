#!/usr/bin/env python3
"""Assign stable, globally unique IDs to every benchmark example.

Older benchmark construction scripts reused local IDs across augmented
examples. This utility keeps each example's content intact while encoding its
dimension, source file, and source row in a public, reproducible identifier.
Run it once when preparing a fresh benchmark release.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def normalized_stem(path: Path) -> str:
    return re.sub(r"^(?:v|V)\d+_", "", path.stem)


def normalized_legacy_id(value: str) -> str:
    return re.sub(r"^(?:v|V)\d+_", "", value)


def normalize(benchmark_dir: Path) -> tuple[int, int]:
    updated = 0
    total = 0
    for dimension_dir in sorted(path for path in benchmark_dir.iterdir() if path.is_dir()):
        for jsonl_path in sorted(dimension_dir.glob("*.jsonl")):
            rewritten: list[str] = []
            for row_number, line in enumerate(jsonl_path.read_text(encoding="utf-8").splitlines(), start=1):
                if not line.strip():
                    continue
                sample = json.loads(line)
                legacy_id = str(sample.get("id", "")).strip()
                if not legacy_id:
                    raise ValueError(f"Missing id in {jsonl_path}:{row_number}")
                if legacy_id.startswith(f"{dimension_dir.name}__"):
                    raise ValueError(f"Already-normalized id in {jsonl_path}:{row_number}")
                sample["id"] = (
                    f"{dimension_dir.name}__{normalized_stem(jsonl_path)}__"
                    f"{row_number:06d}__{normalized_legacy_id(legacy_id)}"
                )
                rewritten.append(json.dumps(sample, ensure_ascii=False, separators=(",", ":")))
                updated += 1
                total += 1
            jsonl_path.write_text("\n".join(rewritten) + "\n", encoding="utf-8")
    return updated, total


def check(benchmark_dir: Path) -> int:
    """Validate canonical IDs without rewriting any benchmark records."""
    total = 0
    seen: set[str] = set()
    for dimension_dir in sorted(path for path in benchmark_dir.iterdir() if path.is_dir()):
        for jsonl_path in sorted(dimension_dir.glob("*.jsonl")):
            for row_number, line in enumerate(jsonl_path.read_text(encoding="utf-8").splitlines(), start=1):
                if not line.strip():
                    continue
                sample = json.loads(line)
                sample_id = str(sample.get("id", ""))
                expected_prefix = f"{dimension_dir.name}__{normalized_stem(jsonl_path)}__"
                if not sample_id.startswith(expected_prefix):
                    raise ValueError(
                        f"Non-canonical id in {jsonl_path}:{row_number}: {sample_id}"
                    )
                if sample_id in seen:
                    raise ValueError(f"Duplicate canonical id: {sample_id}")
                seen.add(sample_id)
                total += 1
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize benchmark sample IDs.")
    parser.add_argument("--benchmark-dir", default="benchmark")
    parser.add_argument("--check", action="store_true", help="Validate IDs without rewriting files.")
    args = parser.parse_args()
    if args.check:
        total = check(Path(args.benchmark_dir))
        print(f"Validated {total} canonical benchmark sample IDs.")
        return
    updated, total = normalize(Path(args.benchmark_dir))
    print(f"Normalized {updated}/{total} benchmark sample IDs.")


if __name__ == "__main__":
    main()
