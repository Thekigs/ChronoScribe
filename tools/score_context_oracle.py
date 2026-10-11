#!/usr/bin/env python3
"""Compute an explicitly retrospective per-sample fixed-context oracle.

The oracle uses held-out targets only after forecasts exist, so it is an upper
bound, never a deployable policy. It quantifies the headroom left after the
fixed-context controls and prevents calling a metric-specific winner a single
"best fixed" policy.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np

METRICS=('MAE','RMSE','sMAPE','MASE','Direction_Accuracy')
def load(p):
    return {str(r['id']):r for r in (json.loads(x) for x in (Path(p)/'all_results.jsonl').read_text().splitlines() if x.strip()) if r.get('forecast_metrics') and not r.get('error')}
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',required=True); ap.add_argument('--output',required=True,type=Path); a=ap.parse_args()
    docs={Path(p).name.removeprefix('ablate_fixed_'):load(p) for p in sorted(Path(a.root).glob('ablate_fixed_*'))}
    common=set.intersection(*(set(d) for d in docs.values()))
    out={'protocol':'retrospective_fixed_context_oracle_v1','policies':sorted(docs),'common_rows':len(common),
         'warning':'This oracle selects the best fixed context after seeing the held-out target; it is not a deployable policy and must not be called adaptive routing.'}
    out['oracle']={}; out['best_policy_by_metric']={}
    for m in METRICS:
        lower=m!='Direction_Accuracy'
        eligible={k:[n for n in docs if isinstance(docs[n][k].get('forecast_metrics',{}).get(m),(int,float))] for k in common}
        eligible={k:v for k,v in eligible.items() if v}
        chosen={k:(min(eligible[k],key=lambda n:docs[n][k]['forecast_metrics'][m]) if lower else max(eligible[k],key=lambda n:docs[n][k]['forecast_metrics'][m])) for k in eligible}
        vals=[docs[chosen[k]][k]['forecast_metrics'][m] for k in chosen]
        out['oracle'][m]={'n':len(vals),'mean':float(np.mean(vals)),'median':float(np.median(vals)),'policy_counts':{n:sum(v==n for v in chosen.values()) for n in docs}}
        means={n:float(np.mean([docs[n][k]['forecast_metrics'][m] for k in common if isinstance(docs[n][k].get('forecast_metrics',{}).get(m),(int,float))])) for n in docs}
        out['best_policy_by_metric'][m]=min(means,key=means.get) if lower else max(means,key=means.get)
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n'); print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
