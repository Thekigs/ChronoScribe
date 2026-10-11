#!/usr/bin/env python3
"""Create a compact audit for adaptive-vs-fixed paired evidence.

The summary distinguishes a complete same-manifest paired score from the
separate provenance audit.  It never upgrades legacy cached decisions to fresh
controller evidence.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", required=True, type=Path)
    ap.add_argument("--provenance", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    a = ap.parse_args()
    score = json.loads(a.score.read_text())
    provenance = json.loads(a.provenance.read_text())
    report = {
        "protocol": "adaptive_vs_fixed_provenance_audit_v1",
        "score": str(a.score.resolve()),
        "provenance": str(a.provenance.resolve()),
        "coverage": score.get("candidate_reports", [{}])[0].get("coverage", {}),
        "paired": score.get("candidate_reports", [{}])[0].get("paired", {}),
        "candidate_controller_provenance": provenance,
        "publication_status": (
            "paired_score_available_but_not_fresh_controller_evidence"
            if provenance.get("fresh_valid_forecast_rows", 0) == 0
            else "fresh_controller_evidence_available"
        ),
        "interpretation": (
            "The paired score is useful for protocol sensitivity, but adaptive "
            "routing is not promoted as a strict same-thinking result until the "
            "fresh controller directory has valid, history-hashed decisions."
        ),
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
