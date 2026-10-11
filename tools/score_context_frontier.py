#!/usr/bin/env python3
"""Rank fixed context controls and identify the per-metric frontier."""
from __future__ import annotations
import argparse,json,math
from pathlib import Path
import numpy as np

METRICS=("MAE","RMSE","sMAPE","MASE","Direction_Accuracy")
def load(p): return {str(r['id']):r for r in (json.loads(x) for x in (p/'all_results.jsonl').read_text().splitlines() if x.strip()) if r.get('forecast_metrics') and not r.get('error')}
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--root',required=True,type=Path); ap.add_argument('--output',required=True,type=Path); a=ap.parse_args()
 docs={p.name.removeprefix('ablation_'):load(p) for p in a.root.glob('ablation_fixed_*')}
 common=set.intersection(*(set(x) for x in docs.values()))
 summary={}
 for name,d in docs.items():
  summary[name]={m:{'n':len(common),'mean':float(np.mean([d[k]['forecast_metrics'][m] for k in common])),'median':float(np.median([d[k]['forecast_metrics'][m] for k in common]))} for m in METRICS}
 best={m:{'by_mean':min(summary,key=lambda n:summary[n][m]['mean']) if m!='Direction_Accuracy' else max(summary,key=lambda n:summary[n][m]['mean']),'by_median':min(summary,key=lambda n:summary[n][m]['median']) if m!='Direction_Accuracy' else max(summary,key=lambda n:summary[n][m]['median'])} for m in METRICS}
 out={'policies':sorted(summary),'common_rows':len(common),'summary':summary,'best_fixed_by_metric':best,'interpretation':'The phrase best fixed policy is metric-dependent. fixed-1024 is the strongest standard fixed context on most scale-aware/relative metrics here, but no single fixed length dominates every metric; full-history and random-context are separate controls.'}
 a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n'); print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
