#!/usr/bin/env python3
"""Evaluate the provenance numeric-claim guard on controlled cases."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.numeric_claim_guard import audit_numeric_claims, enforce_numeric_claims


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "scores/numeric_claim_guard_evaluation.json")
    args = parser.parse_args()

    rng = random.Random(2026)
    cases = []
    for i in range(500):
        evidence = f"forecast horizon=24 start={10+i/100:.2f} end={20+i/100:.2f} min=8.00 max=25.00"
        cases.append({"kind": "supported", "response": f"未来24步，起点约{10+i/100:.2f}，终点约{20+i/100:.2f}。", "evidence": evidence})
        fake = round(rng.uniform(-1000, 1000), 3)
        cases.append({"kind": "unsupported", "response": f"预测最高值为{fake}，风险评分为91。", "evidence": evidence})
    counts = {"supported": {"total": 0, "blocked": 0}, "unsupported": {"total": 0, "blocked": 0}}
    masked = 0
    for case in cases:
        audit = audit_numeric_claims(case["response"], case["evidence"])
        counts[case["kind"]]["total"] += 1
        if audit.blocked_count:
            counts[case["kind"]]["blocked"] += 1
        safe, _ = enforce_numeric_claims(case["response"], case["evidence"])
        masked += int("数值已隐藏" in safe)
    result = {"cases": len(cases), "counts": counts, "masked_responses": masked,
              "supported_pass_rate": 1 - counts["supported"]["blocked"] / counts["supported"]["total"],
              "unsupported_detection_rate": counts["unsupported"]["blocked"] / counts["unsupported"]["total"]}
    out = args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
