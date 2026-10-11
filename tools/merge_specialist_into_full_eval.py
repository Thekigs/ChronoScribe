#!/usr/bin/env python3
"""Inject a specialist-only forecast run into the 23-dimension Qwen eval.

The existing ``results/qwenchronos`` run contains the fixed Qwen semantic
outputs (analysis, reasoning, instruction-following, JSON and timing).  A
specialist run supplies replacement numerical forecasts for the 11 forecast
dimensions.  This creates a full 23-dimension record while keeping the
language side fixed, which is the controlled comparison needed for R/I/E.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


FORECAST_DIMS = {
    "D1_short_term_forecasting", "D2_long_horizon", "D3_multivariate",
    "D4_zero_shot", "D5_few_shot", "D9_robustness", "D10_uncertainty",
    "D14_solar", "D17_missing_robustness", "D20_horizon_degradation",
    "D22_outlier_robustness",
}


def load(root: Path) -> dict[tuple[str, str], dict]:
    rows = {}
    for path in sorted(root.glob("*/results.json")):
        dim = path.parent.name
        for row in json.loads(path.read_text(encoding="utf-8")):
            rows[(dim, str(row["id"]))] = row
    return rows


def match_specialist(spec: dict[tuple[str, str], dict], dim: str, sid: str) -> dict | None:
    exact = spec.get((dim, sid))
    if exact is not None:
        return exact
    # The full Qwen run predates the canonical manifest IDs.  Match only an
    # unambiguous stable task tail; this is the same suffix policy used by the
    # specialist runner and avoids mixing similarly named samples.
    def forms(value: str) -> set[str]:
        text = str(value)
        tail = text.rsplit("__", 1)[-1]
        values = {text, tail}
        for item in list(values):
            values.add(re.sub(r"^v\d+_", "", item))
        return values

    sid_forms = forms(sid)
    candidates = []
    for (s_dim, s_id), row in spec.items():
        if s_dim != dim:
            continue
        s_forms = forms(s_id)
        if sid_forms.intersection(s_forms) or any(
            a.endswith(b) or b.endswith(a) for a in sid_forms for b in s_forms
        ):
            candidates.append(row)
    return candidates[0] if len(candidates) == 1 else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, help="23-dimension Qwen result root")
    ap.add_argument("--specialist", required=True, help="11-dimension specialist result root")
    ap.add_argument("--output", required=True)
    ap.add_argument("--name", required=True)
    args = ap.parse_args()
    base = load(Path(args.baseline))
    spec = load(Path(args.specialist))
    out = {}
    replaced = 0
    failed = 0
    for key, original in base.items():
        row = dict(original)
        dim, sid = key
        row["pipeline"] = args.name
        candidate = match_specialist(spec, dim, sid) if dim in FORECAST_DIMS else None
        if dim in FORECAST_DIMS and candidate is not None:
            if candidate.get("error"):
                row["error"] = candidate["error"]
                failed += 1
            else:
                # Preserve Qwen language-side fields from the baseline while
                # replacing all numerical-specialist fields.
                for field in (
                    "data_source", "context_alignment", "c2_params",
                    "predicted_values", "predicted_p10", "predicted_p50",
                    "predicted_p90", "forecast_metrics", "uncertainty_available",
                    "controller_output", "controller_source", "specialist",
                ):
                    if field in candidate:
                        row[field] = candidate[field]
                # The specialist runner reuses controller decisions, so add its
                # numerical latency to the fixed Qwen baseline latency rather
                # than replacing the latter and overstating efficiency.
                row["latency_s"] = round(float(original.get("latency_s", 0.0)) + float(candidate.get("latency_s", 0.0)), 3)
                row["error"] = None
                replaced += 1
        out[key] = row

    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    by_dim = {}
    for (dim, _), row in out.items():
        by_dim.setdefault(dim, []).append(row)
    for dim, rows in by_dim.items():
        d = root / dim
        d.mkdir(parents=True, exist_ok=True)
        d.joinpath("results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    root.joinpath("merge_config.json").write_text(json.dumps({
        "pipeline": args.name,
        "baseline": str(Path(args.baseline).resolve()),
        "specialist": str(Path(args.specialist).resolve()),
        "forecast_dimensions_replaced": sorted(FORECAST_DIMS),
        "replaced_rows": replaced,
        "failed_specialist_rows": failed,
        "semantic_side_fixed": True,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with root.joinpath("all_results.jsonl").open("w", encoding="utf-8") as handle:
        for row in out.values():
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Merged {args.name}: {replaced} specialist rows, {failed} specialist failures, {len(out)} total rows")


if __name__ == "__main__":
    main()
