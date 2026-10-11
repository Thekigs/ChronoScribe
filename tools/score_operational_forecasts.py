#!/usr/bin/env python3
"""Pure forecasting evaluation, separate from legacy F/TLS.

No target-dependent sample filtering, no text/latency/JSON score, no fabricated
targets, no horizon padding. Standard MASE denominators use aligned HISTORY,
not the future window. All comparative averages use common sample identities.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.benchmark_data import load_raw, locate_gt_window
from utils.evaluation_manifest import load_manifest_samples
from scoring.score_results import normalize_result_record

SEASON = {"h": 24, "10t": 144, "10min": 144, "15t": 96,
          "15min": 96, "d": 7, "w": 52, "min": 60}

def wilson(k, n):
    if not n: return None
    z = 1.959963984540054
    p = k/n; den = 1+z*z/n
    center = (p+z*z/(2*n))/den
    radius = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [center-radius, center+radius]

def metrics(y, p, h, season):
    err = np.abs(y-p)
    mae = float(err.mean())
    scale = float(np.std(h))
    diff = float(np.mean(np.abs(np.diff(h)))) if len(h)>1 else 0
    sd = float(np.mean(np.abs(h[season:]-h[:-season]))) if season and len(h)>season else 0
    wy = float(np.sum(np.abs(y)))
    return {"MAE":mae, "RMSE":float(np.sqrt(np.mean((y-p)**2))),
            "sMAPE":float(np.mean(200*err/(np.abs(y)+np.abs(p)+1e-10))),
            "WAPE":float(err.sum()/wy*100) if wy>1e-10 else None,
            "nMAE_history_std":mae/scale if scale>1e-10 else None,
            "nRMSE_history_std":float(np.sqrt(np.mean((y-p)**2)))/scale if scale>1e-10 else None,
            "MASE_history_lag1":mae/diff if diff>1e-10 else None,
            "MASE_history_seasonal":mae/sd if sd>1e-10 else None,
            "Direction_Accuracy":float(np.mean(np.sign(np.diff(y))==np.sign(np.diff(p)))) if len(y)>1 else None}

def summarize(rows):
    out={}
    for metric in sorted(set().union(*(r.keys() for r in rows))) if rows else []:
        vals=[r[metric] for r in rows if isinstance(r.get(metric),(float,int)) and math.isfinite(r[metric])]
        out[metric]={"n":len(vals), "mean":float(np.mean(vals)) if vals else None,
                     "median":float(np.median(vals)) if vals else None}
    return out

def compare(base, candidate, keys, metadata, rng, nboot):
    out={"n_common_rows":len(keys), "metrics":{}, "by_dimension":{}}
    for m in next(iter(base.values())) if base else []:
        pairs=[(base[k][m],candidate[k][m],k) for k in keys
               if base[k].get(m) is not None and candidate[k].get(m) is not None]
        if not pairs: continue
        b,c=np.array([(x,y) for x,y,_ in pairs]).T; delta=c-b
        draws=delta[rng.integers(0,len(delta),(nboot,len(delta)))].mean(1)
        dims=sorted({metadata[k]["dimension"] for _,_,k in pairs})
        dimdelta=np.array([np.mean([y-x for x,y,k in pairs if metadata[k]["dimension"]==d]) for d in dims])
        ddraw=dimdelta[rng.integers(0,len(dims),(nboot,len(dims)))].mean(1)
        out["metrics"][m]={"n":len(delta),"baseline_mean":float(b.mean()),"candidate_mean":float(c.mean()),
                           "candidate_minus_baseline":float(delta.mean()),
                           "sample_bootstrap_ci95":np.quantile(draws,[.025,.975]).tolist(),
                           "equal_dimension_delta":float(dimdelta.mean()),
                           "dimension_bootstrap_ci95":np.quantile(ddraw,[.025,.975]).tolist(),
                           "candidate_win_rate":float(np.mean(delta>0 if m=="Direction_Accuracy" else delta<0))}
    for d in sorted({metadata[k]["dimension"] for k in keys}):
        sub=[k for k in keys if metadata[k]["dimension"]==d]
        out["by_dimension"][d]={"n":len(sub),"baseline":summarize([base[k] for k in sub]),
                                "candidate":summarize([candidate[k] for k in sub])}
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",action="append",nargs=2,required=True,metavar=("NAME","RESULT_DIR"))
    ap.add_argument("--baseline",required=True)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--controller",type=Path)
    ap.add_argument("--strict-model",action="append",default=[])
    ap.add_argument("--resamples",type=int,default=10000)
    ap.add_argument("--no-naive-controls", action="store_true",
                    help="Do not append Persistence/SeasonalNaive to a policy-only frontier.")
    ap.add_argument("--require-no-clamp", action="store_true",
                    help="For strict-model rows, require proposed and executed context lengths to match.")
    args=ap.parse_args()
    plan=load_manifest_samples(ROOT/"benchmark",ROOT/"benchmark/evaluation_manifest.json")
    samples={(d,str(s["id"])):s for d,ss in plan.items() for s in ss}
    metadata={};targets={}; audit=Counter(); excluded=[]
    for key,s in samples.items():
        gt=s.get("ground_truth",{}); y=gt.get("values")
        if not isinstance(y,list) or not y: audit["no_manifest_target"]+=1;continue
        loaded=load_raw(s)
        if loaded is None or loaded[0] is None: audit["no_source_history"]+=1;continue
        data,source=loaded
        st=locate_gt_window(data,y,gt.get("naive_forecast"))
        if st is None or st<2:
            audit["unverified_target_alignment"]+=1
            excluded.append({"dimension":key[0],"id":key[1],"reason":"unverified_target_alignment","source":source})
            continue
        # Common historical scale: the same past-only 4096 observations for every policy.
        hist=np.asarray(data[max(0,st-4096):st],dtype=float)
        if not np.isfinite(hist).all() or not np.isfinite(y).all(): audit["nonfinite_target_or_history"]+=1;continue
        season=SEASON.get(str(s.get("freq","")).lower())
        metadata[key]={"dimension":key[0],"id":key[1],"dataset":s.get("dataset"),
                       "domain":s.get("domain"),"frequency":s.get("freq"),"horizon":len(y),
                       "seasonal_lag":season,"history_length":len(hist),"source":source,
                       "history_sha256":hashlib.sha256(hist.astype('<f8').tobytes()).hexdigest()}
        targets[key]=(np.asarray(y,dtype=float),hist,season)
    allowed=set()
    if args.controller:
        for p in args.controller.glob('*/results.json'):
            for r in json.loads(p.read_text()):
                if r.get('directive_valid') is True and r.get('history_sha256'):
                    allowed.add((p.parent.name,str(r['id'])))
    docs={}; exclusions={}
    for name,path in args.model:
        rows=[json.loads(x) for x in (Path(path)/'all_results.jsonl').read_text().splitlines() if x.strip()]
        doc={};why=Counter()
        for r in rows:
            k=(str(r.get('dimension')),str(r.get('id')))
            if k not in targets: why['outside_verified_target_set']+=1;continue
            if r.get('error'): why['runner_error']+=1;continue
            if name in args.strict_model and k not in allowed: why['not_fresh_schema_valid_controller']+=1;continue
            if name in args.strict_model and args.require_no_clamp:
                requested=r.get('requested_context_length')
                executed=r.get('executed_context_length')
                if requested is None or executed is None or requested!=executed:
                    why['controller_context_clamped']+=1;continue
            _,_,pred,_=normalize_result_record(r)
            y,h,season=targets[k]
            if not isinstance(pred,list) or len(pred)!=len(y): why['exact_horizon_failure']+=1;continue
            pr=np.asarray(pred,dtype=float)
            if pr.ndim!=1 or not np.isfinite(pr).all(): why['invalid_forecast']+=1;continue
            doc[k]=metrics(y,pr,h,season)
        docs[name]=doc; exclusions[name]=dict(why)
    for name in ([] if args.no_naive_controls else ['Persistence','SeasonalNaive']):
        docs[name]={}
        for k,(y,h,season) in targets.items():
            if name=='Persistence': pred=np.full(len(y),h[-1])
            elif season and len(h)>=season: pred=np.resize(h[-season:],len(y))
            else: continue
            docs[name][k]=metrics(y,pred,h,season)
    common=set.intersection(*(set(d) for d in docs.values()))
    rng=np.random.default_rng(20261002)
    out={'protocol':'pure_forecasting_verified_history_v1','manifest_sha256':hashlib.sha256((ROOT/'benchmark/evaluation_manifest.json').read_bytes()).hexdigest(),
         'notes':['Not legacy F/TLS; no text, JSON, latency or clipping weights.',
                  'All-model table uses identical common identities; pairwise reports have their own explicit coverage.',
                  'Targets must be found in the resolved source; tail_before_gt approximations excluded.',
                  'MASE denominators use pre-origin history, never target differences; undefined scales remain null.',
                  'Seasonal lags are declared frequency heuristics, not validation-tuned periodicities.',
                  'No target-scale outlier removal. WAPE undefined on zero target mass.',
                  'Bootstrap is sample/dimension uncertainty, not repeated-run uncertainty.'],
         'seasonal_lags':SEASON,'target_audit':dict(audit),'verified_target_rows':len(targets),
         'no_clamp_required_for_strict_models':args.require_no_clamp,
         'strict_models':args.strict_model,
         'excluded_alignment_rows':excluded,'model_exclusions':exclusions,'all_model_common_rows':len(common),
         'all_model_matched':{name:summarize([d[k] for k in sorted(common)]) for name,d in docs.items()},'paired':{}}
    base=docs[args.baseline]
    for name,d in docs.items():
        if name!=args.baseline:
            out['paired'][name]=compare(base,d,sorted(set(base)&set(d)),metadata,rng,args.resamples)
    out['per_sample_metadata']=[metadata[k] for k in sorted(common)]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'verified_target_rows':len(targets),'all_model_common_rows':len(common),
                      'pairwise_rows':{n:d['n_common_rows'] for n,d in out['paired'].items()}},indent=2))

if __name__=='__main__': main()
