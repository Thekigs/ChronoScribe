#!/usr/bin/env python3
"""Score an input-matched adaptive run without fallback contamination.

The controller directory is the source of truth for whether a decision was
actually produced.  A forecast row is eligible only when its same
(dimension,id) has a schema-valid fresh decision with a history hash.  This
prevents an empty cached record from being mislabeled as an adaptive policy.
"""
from __future__ import annotations

import argparse, json, math
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np

METRICS = ("MAE", "RMSE", "sMAPE", "MASE", "Direction_Accuracy")

def valid(d):
    if not isinstance(d, dict) or type(d.get("use_chronos2")) is not bool:
        return False
    if d["use_chronos2"] is False:
        return isinstance(d.get("direct_reply"), str) and bool(d["direct_reply"].strip())
    return all(type(d.get(k)) is int and d[k] > 0 for k in ("prediction_length", "context_length"))

def load_jsonl(path):
    return [json.loads(x) for x in (path / "all_results.jsonl").read_text().splitlines() if x.strip()]

def load_controller(root):
    out = {}
    for p in root.glob("*/results.json"):
        for r in json.loads(p.read_text()):
            k = (str(p.parent.name), str(r.get("id")))
            if valid(r.get("step1_params")) and r.get("history_sha256") and r.get("directive_valid") is True:
                out[k] = r
    return out

def boot(v, seed=20261001):
    v = np.asarray(v, dtype=float)
    rng = np.random.default_rng(seed)
    b = v[rng.integers(0, len(v), (10000, len(v)))].mean(1)
    return [float(np.quantile(b, .025)), float(np.quantile(b, .975))]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--controller", required=True, type=Path)
    ap.add_argument("--adaptive", required=True, type=Path)
    ap.add_argument("--fixed", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--require-no-clamp", action="store_true", help="Keep only rows whose requested and executed context lengths are identical.")
    a = ap.parse_args()
    ctl = load_controller(a.controller)
    adaptive = {(str(r.get("dimension")), str(r.get("id"))): r for r in load_jsonl(a.adaptive)}
    fixed = {(str(r.get("dimension")), str(r.get("id"))): r for r in load_jsonl(a.fixed)}
    eligible = {}
    clamped = 0
    for k in set(adaptive) & set(ctl):
        r = adaptive[k]
        if not r.get("forecast_metrics") or r.get("error"): continue
        requested = ctl[k]["step1_params"].get("context_length")
        executed = r.get("c2_params", {}).get("ctx_len")
        if requested != executed:
            clamped += 1
            if a.require_no_clamp: continue
        eligible[k] = r
    common = sorted(set(eligible) & set(fixed))
    result = {
        "controller_root": str(a.controller.resolve()),
        "adaptive_root": str(a.adaptive.resolve()),
        "fixed_root": str(a.fixed.resolve()),
        "controller_valid_decisions": len(ctl),
        "adaptive_forecast_rows": len(adaptive),
        "fixed_forecast_rows": len(fixed),
        "fixed_policy": next((json.loads((a.fixed / "run_config.json").read_text()).get("fixed_context_length")
                              or json.loads((a.fixed / "run_config.json").read_text()).get("context_length")
                              for _ in [0] if (a.fixed / "run_config.json").exists()), "unknown"),
        "strict_common_rows": len(common),
        "clamped_rows_excluded_or_available": clamped,
        "context_lengths": Counter(int(adaptive[k].get("c2_params", {}).get("ctx_len")) for k in common),
        "dimensions": Counter(k[0] for k in common),
        "paired": {},
        "interpretation": "Only fresh schema-valid controller decisions with a recorded history hash are included; no fallback rows are eligible. Delta is adaptive minus the named fixed policy; lower is better for error metrics, higher for Direction_Accuracy. Requested context lengths are compared with executed lengths; optional no-clamp mode excludes any runner truncation.",
    }
    for m in METRICS:
        vals=[]
        for k in common:
            x=eligible[k].get("forecast_metrics",{}).get(m); y=fixed[k].get("forecast_metrics",{}).get(m)
            if x is None or y is None or not math.isfinite(float(x)) or not math.isfinite(float(y)): continue
            vals.append(float(x)-float(y))
        if vals:
            result["paired"][m]={"n":len(vals),"mean_adaptive_minus_fixed":float(np.mean(vals)),"median":float(np.median(vals)),"ci95":boot(vals),"adaptive_better_rate":float(np.mean(np.asarray(vals)<0 if m != "Direction_Accuracy" else np.asarray(vals)>0))}
        else: result["paired"][m]={"n":0}
    result["by_dimension"] = {}
    for dim in sorted({k[0] for k in common}):
        sub = [k for k in common if k[0] == dim]
        result["by_dimension"][dim] = {}
        for m in METRICS:
            vals=[]
            for k in sub:
                x=eligible[k].get("forecast_metrics",{}).get(m); y=fixed[k].get("forecast_metrics",{}).get(m)
                if x is not None and y is not None and math.isfinite(float(x)) and math.isfinite(float(y)): vals.append(float(x)-float(y))
            result["by_dimension"][dim][m] = {"n":len(vals), "mean_adaptive_minus_fixed":float(np.mean(vals)) if vals else None, "ci95":boot(vals) if vals else None}
    a.output.parent.mkdir(parents=True, exist_ok=True); a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__ == "__main__": main()
