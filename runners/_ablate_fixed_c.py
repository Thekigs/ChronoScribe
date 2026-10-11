#!/usr/bin/env python3
"""Fixed-C ablation sweep — run Chronos2 with fixed context lengths to compare against adaptive selection."""
import os, sys, json, time, argparse, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

parser = argparse.ArgumentParser()
parser.add_argument("--context-length", type=int, required=True, help="Fixed context length (64,128,256,512,1024)")
parser.add_argument("--samples-per-dim", type=int, default=20)
args = parser.parse_args()

C = args.context_length
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "results", f"ablate_fixed_c{C}")
os.makedirs(OUT, exist_ok=True)

from chronos import Chronos2Pipeline
from scoring.detailed_scoring import compute_all_forecast_metrics
from utils.benchmark_data import load_raw, build_context
from utils.evaluation_manifest import load_manifest_samples

print(f"Fixed-C={C} ablation")
pipeline = Chronos2Pipeline.from_pretrained(os.path.join(BASE, "models", "chronos-2"))

test_plan = load_manifest_samples(os.path.join(BASE, "benchmark"), os.path.join(BASE, "benchmark", "evaluation_manifest.json"))

all_results = []
for dim, samples in sorted(test_plan.items()):
    dr = []
    for sample in samples:
        gt = sample.get("ground_truth", {})
        gv = gt.get("values", [])
        gth = len(gv) if isinstance(gv, list) else 0
        r = {"id": sample["id"], "dimension": dim, "pipeline": f"fixed_c{C}"}
        try:
            data, src = load_raw(sample)
            if data is None or len(data) <= 10: raise RuntimeError("no_data")
            ctx, _ = build_context(data, sample, gv, qwen_ctx_len=min(C, len(data)))
            if len(ctx) <= 10: raise RuntimeError("context_too_short")
            pl = gth if gth > 0 else 24
            ct = torch.tensor(ctx, dtype=torch.float32).reshape(1, 1, -1)
            ql, _ = pipeline.predict_quantiles(ct, prediction_length=pl)
            q = ql[0]
            preds = q[0, :, q.shape[2]//2].tolist() if hasattr(q, 'shape') and len(q.shape) == 3 else [float(ql[0].squeeze())]
            if gth > 0:
                if len(preds) > gth: preds = preds[:gth]
                elif len(preds) < gth: preds = preds + [preds[-1]]*(gth-len(preds)) if preds else [0]*gth
            r["predicted_values"] = preds
            if gv and preds:
                r["forecast_metrics"] = compute_all_forecast_metrics(np.array(gv), np.array(preds), naive_forecast=gt.get("naive_forecast"))
            r["error"] = None
        except Exception as e:
            r["error"] = str(e)[:200]
        dr.append(r)
    os.makedirs(os.path.join(OUT, dim), exist_ok=True)
    with open(os.path.join(OUT, dim, "results.json"), "w") as f: json.dump(dr, f, ensure_ascii=False, indent=2)
    all_results.extend(dr)
    print(f"{dim}: {sum(1 for r in dr if not r.get('error'))}/{len(dr)}")

with open(os.path.join(OUT, "all_results.jsonl"), "w") as f:
    for r in all_results: f.write(json.dumps(r, ensure_ascii=False)+"\n")
print(f"Done: {sum(1 for r in all_results if not r.get('error'))}/{len(all_results)}")
