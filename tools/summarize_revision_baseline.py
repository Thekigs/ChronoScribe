#!/usr/bin/env python3
"""Create a compact, auditable summary for the local bounded-revision run."""

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def finite(x):
    return isinstance(x, (int, float)) and math.isfinite(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    rows = [json.loads(line) for line in Path(args.input).read_text().splitlines() if line.strip()]
    forecast_rows = [r for r in rows if r.get("forecast_metrics") and not r.get("skip_reason") and not r.get("error")]
    skipped = [r for r in rows if r.get("skip_reason")]
    errors = [r for r in rows if r.get("error")]
    valid_directives = [r for r in forecast_rows if r.get("revision_audit", {}).get("valid")]
    applied = [r for r in valid_directives if r.get("revision_audit", {}).get("applied")]
    changed = [r for r in applied if r.get("revision_audit", {}).get("trajectory_changed")]

    def deltas(metric):
        values = []
        for r in forecast_rows:
            raw = r.get("raw_forecast_metrics", {}).get(metric)
            final = r.get("forecast_metrics", {}).get(metric)
            if finite(raw) and finite(final):
                values.append(final - raw)
        if not values:
            return {}
        values_sorted = sorted(values)
        return {
            "n": len(values),
            "mean_final_minus_raw": sum(values) / len(values),
            "median_final_minus_raw": values_sorted[len(values_sorted) // 2],
            "min": min(values),
            "max": max(values),
        }

    ctx = Counter(r.get("context_alignment", {}).get("context_aligned") for r in rows)
    revision_reasons = Counter(r.get("revision_audit", {}).get("reason") for r in rows)
    dim_stats = defaultdict(lambda: {"rows": 0, "forecast_valid": 0, "valid_directives": 0, "applied": 0, "changed": 0})
    for r in rows:
        d = dim_stats[r.get("dimension")]
        d["rows"] += 1
        if r in forecast_rows:
            d["forecast_valid"] += 1
        if r.get("revision_audit", {}).get("valid"):
            d["valid_directives"] += 1
        if r.get("revision_audit", {}).get("applied"):
            d["applied"] += 1
        if r.get("revision_audit", {}).get("trajectory_changed"):
            d["changed"] += 1

    summary = {
        "protocol": "local_numerical_revision_v2",
        "protocol_status": "mechanism_control_not_official_reproduction",
        "input": str(Path(args.input).resolve()),
        "device": sorted(set(r.get("device") for r in rows)),
        "context_length": sorted({r.get("requested_context_length") for r in rows if r.get("requested_context_length") is not None}),
        "future_values_used_for_execution": sorted(set(bool(r.get("ground_truth_provided_to_controller")) for r in rows)),
        "coverage": {
            "rows": len(rows),
            "unique_ids": len({r.get("id") for r in rows}),
            "forecast_valid_rows": len(forecast_rows),
            "skipped_no_forecast_target": len(skipped),
            "errors": len(errors),
            "context_alignment": dict(ctx),
        },
        "controller": {
            "valid_directives": len(valid_directives),
            "valid_rate_over_forecast_rows": len(valid_directives) / len(forecast_rows) if forecast_rows else None,
            "applied": len(applied),
            "trajectory_changed": len(changed),
            "revision_reasons": dict(revision_reasons),
        },
        "raw_vs_final": {metric: deltas(metric) for metric in ["MAE", "RMSE", "sMAPE", "MASE", "Direction_Accuracy"]},
        "by_dimension": dict(dim_stats),
        "errors": [{"id": r.get("id"), "dimension": r.get("dimension"), "error": r.get("error")} for r in errors],
    }
    Path(args.output).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
