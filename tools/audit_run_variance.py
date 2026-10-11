#!/usr/bin/env python3
"""Measure reproducibility from completed repeated runs.

This is deliberately descriptive: it reports exact-output agreement and
metric spread on overlapping identities, without pretending two deterministic
runs are a confidence interval for a stochastic deployment.
"""
from __future__ import annotations
import argparse,json,os,math
from pathlib import Path
import numpy as np

def rows(root):
    root=Path(root); out={}
    f=root/'all_results.jsonl'
    if f.exists():
        for line in f.read_text().splitlines():
            if line.strip():
                r=json.loads(line); out[str(r.get('id'))]=r
    else:
        for f in root.glob('*/results.json'):
            for r in json.loads(f.read_text()): out[str(r.get('id'))]=r
    return out

def forecast_pair(a,b):
    pairs=[]; maxdiff=[]
    for k in sorted(set(a)&set(b)):
        x=a[k].get('predicted_values'); y=b[k].get('predicted_values')
        if isinstance(x,list) and isinstance(y,list) and len(x)==len(y):
            d=float(np.max(np.abs(np.asarray(x,dtype=float)-np.asarray(y,dtype=float))))
            maxdiff.append(d)
            if a[k].get('forecast_metrics') and b[k].get('forecast_metrics'):
                pairs.append({m:float(b[k]['forecast_metrics'].get(m))-float(a[k]['forecast_metrics'].get(m))
                              for m in ('MAE','RMSE','sMAPE','MASE','Direction_Accuracy')
                              if a[k]['forecast_metrics'].get(m) is not None and b[k]['forecast_metrics'].get(m) is not None})
    return {'prediction_pairs':len(maxdiff),'nonidentical_predictions':sum(x>1e-8 for x in maxdiff),
            'max_abs_prediction_difference':max(maxdiff or [0.0]),'metric_pairs':len(pairs),
            'metric_delta_mean':{m:float(np.mean([p[m] for p in pairs if m in p])) if any(m in p for p in pairs) else None
                                 for m in ('MAE','RMSE','sMAPE','MASE','Direction_Accuracy')}}

def controller_pair(a,b):
    common=sorted(set(a)&set(b)); valid=[]; invalid=0; diff=[]
    for k in common:
        va=a[k].get('directive_valid') is True; vb=b[k].get('directive_valid') is True
        if va and vb:
            valid.append(k)
            if a[k].get('step1_params')!=b[k].get('step1_params'): diff.append(k)
        else: invalid+=1
    return {'common_rows':len(common),'both_valid':len(valid),'either_invalid':invalid,
            'identical_valid_directives':len(valid)-len(diff),'different_valid_directives':len(diff),
            'different_examples':diff[:10]}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--pair',action='append',nargs=3,metavar=('NAME','RUN_A','RUN_B'),required=True); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args()
    out={'protocol':'completed_run_overlap_exact_agreement_v1','pairs':{}}
    for name,x,y in a.pair:
        A=rows(x);B=rows(y)
        first=next(iter(A.values()),{})
        if 'step1_params' in first or not (A and 'predicted_values' in first): report=controller_pair(A,B)
        else: report=forecast_pair(A,B)
        report.update({'run_a':str(Path(x).resolve()),'run_b':str(Path(y).resolve())})
        out['pairs'][name]=report
    out['interpretation']='Exact agreement on deterministic greedy inference estimates implementation reproducibility, not uncertainty from stochastic sampling, hardware nondeterminism, or future model versions.'
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n'); print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
