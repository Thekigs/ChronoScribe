#!/usr/bin/env python3
"""Validate public ChronoScribe release contracts without running models."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.evaluation_manifest import load_benchmark, load_manifest_samples
from tools.normalize_benchmark_ids import check


BENCHMARK = ROOT / "benchmark"
MANIFEST = BENCHMARK / "evaluation_manifest.json"
META = BENCHMARK / "benchmark_meta.json"


def main() -> None:
    """Fail fast when public metadata and paired-evaluation contracts diverge."""
    canonical_count = check(BENCHMARK)
    samples_by_dimension = load_benchmark(BENCHMARK)
    resolved = load_manifest_samples(BENCHMARK, MANIFEST)
    metadata = json.loads(META.read_text(encoding="utf-8"))

    if metadata.get("total_samples") != canonical_count:
        raise ValueError("benchmark_meta.json total_samples does not match benchmark files")
    if metadata.get("num_dimensions") != len(samples_by_dimension):
        raise ValueError("benchmark_meta.json num_dimensions does not match benchmark files")
    manifest_total = sum(len(samples) for samples in resolved.values())
    if metadata.get("paired_evaluation", {}).get("selected_samples") != manifest_total:
        raise ValueError("benchmark_meta.json paired sample count does not match manifest")

    markdown = []
    missing_companions = []
    for path in ROOT.rglob("*.md"):
        if ".git" in path.parts or path.name.endswith(".zh-CN.md"):
            continue
        markdown.append(path)
        companion = path.with_name(path.stem + ".zh-CN.md")
        if not companion.exists():
            missing_companions.append(str(path.relative_to(ROOT)))
    if missing_companions:
        raise ValueError("missing Chinese Markdown companions: " + ", ".join(missing_companions))

    blocked = [ROOT / ".env", ROOT / "models" / "checkpoint-7500" / "args.json"]
    present = [str(path.relative_to(ROOT)) for path in blocked if path.exists()]
    print(f"Validated {canonical_count} suite records, {manifest_total} paired records, {len(samples_by_dimension)} dimensions, and {len(markdown)} English Markdown files with Chinese companions.")
    if present:
        print("Local ignored artifacts present (do not include them in an archive): " + ", ".join(present))
    else:
        print("No checked local credential or serialized-training artifacts are present.")


if __name__ == "__main__":
    main()
