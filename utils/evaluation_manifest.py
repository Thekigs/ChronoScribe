"""Deterministic evaluation-manifest support.

The public benchmark contains the full evaluation suite. A manifest fixes a
smaller, auditable subset for a particular experiment so that every system is
evaluated on exactly the same sample IDs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL in {path}:{line_number}") from error
            if not row.get("id"):
                raise ValueError(f"Missing sample id in {path}:{line_number}")
            rows.append(row)
    return rows


def load_benchmark(benchmark_dir: str | Path) -> dict[str, list[dict]]:
    """Load all benchmark samples in a stable order and reject duplicate IDs."""
    root = Path(benchmark_dir)
    by_dimension: dict[str, list[dict]] = defaultdict(list)
    seen_ids: set[str] = set()

    for dimension_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        for jsonl_path in sorted(dimension_dir.glob("*.jsonl")):
            for sample in _read_jsonl(jsonl_path):
                sample_id = str(sample["id"])
                if sample_id in seen_ids:
                    raise ValueError(f"Duplicate benchmark sample id: {sample_id}")
                seen_ids.add(sample_id)
                by_dimension[dimension_dir.name].append(sample)

    return {dimension: sorted(samples, key=lambda sample: str(sample["id"]))
            for dimension, samples in sorted(by_dimension.items())}


def benchmark_digest(samples_by_dimension: dict[str, list[dict]]) -> str:
    """Return a content-independent fingerprint of dimension/sample membership."""
    payload = [
        (dimension, [str(sample["id"]) for sample in samples])
        for dimension, samples in sorted(samples_by_dimension.items())
    ]
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def create_manifest(
    benchmark_dir: str | Path,
    output_path: str | Path,
    samples_per_dimension: int = 20,
    seed: int = 2026,
    dimensions: Iterable[str] | None = None,
) -> dict:
    """Create a deterministic manifest without modifying benchmark examples."""
    if samples_per_dimension <= 0:
        raise ValueError("samples_per_dimension must be positive")

    samples_by_dimension = load_benchmark(benchmark_dir)
    requested = set(dimensions) if dimensions else set(samples_by_dimension)
    unknown = requested - set(samples_by_dimension)
    if unknown:
        raise ValueError(f"Unknown dimensions: {sorted(unknown)}")

    rng = random.Random(seed)
    selected: dict[str, list[str]] = {}
    for dimension in sorted(requested):
        candidates = samples_by_dimension[dimension]
        count = min(samples_per_dimension, len(candidates))
        selected[dimension] = sorted(
            str(sample["id"]) for sample in rng.sample(candidates, count)
        )

    manifest = {
        "description": "Fixed sample IDs for paired ChronoScribe evaluation.",
        "selection_seed": seed,
        "samples_per_dimension": samples_per_dimension,
        "benchmark_digest": benchmark_digest(samples_by_dimension),
        "dimensions": selected,
        "total_samples": sum(len(ids) for ids in selected.values()),
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return manifest


def load_manifest_samples(
    benchmark_dir: str | Path,
    manifest_path: str | Path,
    dimensions: Iterable[str] | None = None,
) -> dict[str, list[dict]]:
    """Resolve manifest IDs to benchmark samples and validate the membership."""
    with Path(manifest_path).open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    samples_by_dimension = load_benchmark(benchmark_dir)
    if manifest.get("benchmark_digest") != benchmark_digest(samples_by_dimension):
        raise ValueError(
            "The benchmark membership differs from the manifest. Create a new "
            "manifest deliberately instead of silently changing the evaluation set."
        )

    requested = set(dimensions) if dimensions else set(manifest.get("dimensions", {}))
    available = manifest.get("dimensions", {})
    unknown = requested - set(available)
    if unknown:
        raise ValueError(f"Dimensions not present in manifest: {sorted(unknown)}")

    resolved: dict[str, list[dict]] = {}
    for dimension in sorted(requested):
        index = {str(sample["id"]): sample for sample in samples_by_dimension.get(dimension, [])}
        ids = available[dimension]
        missing = [sample_id for sample_id in ids if sample_id not in index]
        if missing:
            raise ValueError(f"Manifest IDs missing from {dimension}: {missing[:3]}")
        resolved[dimension] = [index[sample_id] for sample_id in ids]
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a paired evaluation manifest.")
    parser.add_argument("--benchmark-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples-per-dim", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--dims", default="", help="Comma-separated optional dimension names.")
    args = parser.parse_args()
    dimensions = [item.strip() for item in args.dims.split(",") if item.strip()] or None
    manifest = create_manifest(
        args.benchmark_dir,
        args.output,
        samples_per_dimension=args.samples_per_dim,
        seed=args.seed,
        dimensions=dimensions,
    )
    print(f"Wrote {manifest['total_samples']} fixed sample IDs to {args.output}")


if __name__ == "__main__":
    main()
