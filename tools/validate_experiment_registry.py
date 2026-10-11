#!/usr/bin/env python3
"""Validate the publication-facing experiment evidence registry."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "scores/experiment_registry_20261003.json"


def read_json(rel: str):
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def main() -> None:
    if not REGISTRY.exists():
        raise SystemExit(
            "No local experiment registry found. This source-only release excludes scores/; "
            "run this check only with a separately supplied result bundle."
        )
    reg = json.loads(REGISTRY.read_text(encoding="utf-8"))
    manifest = ROOT / reg["manifest"]["path"]
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    if digest != reg["manifest"]["sha256"]:
        raise SystemExit(f"manifest hash mismatch: {digest} != {reg['manifest']['sha256']}")

    required = list(reg["authoritative"].values())
    for rel in required:
        if rel.startswith("scores/") and not (ROOT / rel).exists():
            raise SystemExit(f"missing authoritative evidence: {rel}")

    operational = read_json(reg["authoritative"]["forecast_only_models"])
    expected = reg["coverage"]
    checks = {
        "verified_target_rows": operational["verified_target_rows"],
        "all_model_common_rows": operational["all_model_common_rows"],
    }
    for key, actual in checks.items():
        if actual != expected[key]:
            raise SystemExit(f"coverage mismatch for {key}: {actual} != {expected[key]}")

    print(json.dumps({
        "status": "ok",
        "manifest_sha256": digest,
        "authoritative_files": len(required),
        "verified_target_rows": checks["verified_target_rows"],
        "all_model_common_rows": checks["all_model_common_rows"],
    }, indent=2))


if __name__ == "__main__":
    main()

