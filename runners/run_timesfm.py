#!/usr/bin/env python3
"""TimesFM specialist baseline on the ChronoScribe paired manifest.

This runner evaluates a forecasting specialist without the language controller.
Its manifest selection must match the full system and every other baseline.
"""
import argparse
import os,sys,json,time,random,re
from collections import defaultdict

BASE=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH=os.path.join(BASE,'benchmark')
OUT=os.path.join(BASE,'results','timesfm')
sys.path.insert(0,BASE)

parser=argparse.ArgumentParser()
parser.add_argument('--dims',type=str,default='')
parser.add_argument('--manifest',type=str,default=os.path.join(BENCH,'evaluation_manifest.json'),
    help='Fixed evaluation manifest shared by every compared system.')
args=parser.parse_args()

# Do not import specialist dependencies or create output directories for help.
import numpy as np
import pandas as pd
import torch
os.makedirs(OUT,exist_ok=True)


print("Loading TimesFM 2.5 (200M params)...")
import timesfm
model = timesfm.TimesFM_2p5_200M_torch.from_pretrained('google/timesfm-2.5-200m-pytorch')
cfg = timesfm.ForecastConfig(max_context=1024, max_horizon=720, normalize_inputs=True, per_core_batch_size=1)
model.compile(cfg)
print("Ready\n")

# ═══ Data loading (same as c2only) ═══
SKIP_COLS={"date","OT","timestamp","timestamp_(min)","time","datetime"}
DATASET_PATHS={"ETTh1":"data/ETT-small/ETTh1.csv","ETTh2":"data/ETT-small/ETTh2.csv",
    "ETTm1":"data/ETT-small/ETTm1.csv","ETTm2":"data/ETT-small/ETTm2.csv",
    "weather":"data/weather/weather.csv","traffic":"data/traffic/traffic.csv",
    "electricity":"data/electricity/electricity.csv",
    "exchange_rate":"data/exchange_rate/exchange_rate.csv",
    "national_illness":"data/illness/national_illness.csv"}

def load_data(sample):
    ds=str(sample.get("dataset",""));series=str(sample.get("series",""));sid=str(sample.get("id",""))
    if ds in DATASET_PATHS:
        fp=DATASET_PATHS[ds];fpath=os.path.join(BASE,'data',fp) if not fp.startswith('data/') else os.path.join(BASE,fp)
        if os.path.exists(fpath) and fp.endswith('.csv'):
            df=pd.read_csv(fpath)
            dcols=[c for c in df.columns if str(c) not in SKIP_COLS]
            if dcols:
                for c in dcols:
                    if str(c)==str(series):
                        v=df[c].dropna().values.astype(np.float64)
                        if len(v)>50:return v
                nums=re.findall(r'\d+',series+sid)
                for n in nums:
                    idx=int(n)
                    if 0<=idx<len(dcols):
                        v=df[dcols[idx]].dropna().values.astype(np.float64)
                        if len(v)>50:return v
    solarp=os.path.join(BASE,'data','solar','solar_AL.txt')
    if ds.lower()=="solar" and os.path.exists(solarp):
        sdf=pd.read_csv(solarp,header=None)
        nums=re.findall(r'\d+',series+sid)
        for n in nums:
            ci=int(n)
            if 0<=ci<sdf.shape[1]:
                v=sdf.iloc[:,ci].dropna().values.astype(np.float64)
                if len(v)>50:return v
    for fp in ["data/ETT-small/ETTh1.csv","data/ETT-small/ETTh2.csv","data/ETT-small/ETTm1.csv",
               "data/ETT-small/ETTm2.csv","data/traffic/traffic.csv","data/weather/weather.csv",
               "data/electricity/electricity.csv","data/exchange_rate/exchange_rate.csv"]:
        fpath=os.path.join(BASE,fp)
        if not os.path.exists(fpath):continue
        try:
            df=pd.read_csv(fpath)
            dcols=[c for c in df.columns if str(c) not in SKIP_COLS]
            for c in dcols:
                if str(c)==str(series):
                    v=df[c].dropna().values.astype(np.float64)
                    if len(v)>50:return v
        except Exception:pass
    return None

# Resolve only the fixed sample IDs. Do not randomly resample per baseline.
from utils.evaluation_manifest import load_manifest_samples

print("Loading fixed evaluation manifest...")
dims_filter=[d.strip() for d in args.dims.split(',') if d.strip()] if args.dims else None
test_plan=load_manifest_samples(BENCH,args.manifest,dims_filter)

all_results=[];t0=time.time()
for dim_name,samples in sorted(test_plan.items()):
    dr=[];dd=os.path.join(OUT,dim_name);os.makedirs(dd,exist_ok=True)
    for i,sample in enumerate(samples):
        gt=sample.get("ground_truth",{})
        r={"id":sample["id"],"dimension":dim_name,"category":sample.get("category",""),
           "dataset":sample.get("dataset","?"),"domain":sample.get("domain","?"),"pipeline":"timesfm"}
        try:
            data=load_data(sample)
            gv=gt.get("values",[]);gth=len(gv) if isinstance(gv,list) else 0
            if data is None or len(data)<50:r["error"]="no_data";dr.append(r);continue
            pl=gth if gth>0 else 24
            # Use last 512 points as context
            ctx_len=min(512,len(data)-pl)
            ctx=data[-ctx_len-pl:-pl] if len(data)>ctx_len+pl else data[-ctx_len:]
            if len(ctx)<10:r["error"]="context too short";dr.append(r);continue

            t1=time.time()
            # TimesFM needs exact context length if normalize_inputs=True; pad to max_context
            ctx_arr = ctx.astype(np.float32)
            if len(ctx_arr) < 1024:
                ctx_arr = np.pad(ctx_arr, (1024 - len(ctx_arr), 0), mode='edge')
            point_forecast, quantile_forecast = model.forecast(horizon=pl, inputs=[ctx_arr])
            preds=point_forecast[0].tolist()  # (pl,)

            if gth>0:
                if len(preds)>gth:preds=preds[:gth]
                elif len(preds)<gth:preds=preds+[preds[-1]]*(gth-len(preds)) if preds else [0]*gth
            r["predicted_values"]=preds
            if gv and preds:
                from scoring.detailed_scoring import compute_all_forecast_metrics
                r["forecast_metrics"]=compute_all_forecast_metrics(np.array(gv),np.array(preds),naive_forecast=gt.get("naive_forecast"))
            r["latency_s"]=round(time.time()-t1,3);r["error"]=None
            fm=r.get("forecast_metrics",{})
            smape=f" sMAPE={fm.get('sMAPE',0):.0f}%" if 'sMAPE' in fm else ""
            print(f"  [{i+1}/{len(samples)}] ✅ {sample['id'][:35]}... {r['latency_s']:.1f}s{smape}")
        except Exception as e:
            r["error"]=str(e)[:200]
            print(f"  [{i+1}/{len(samples)}] ❌ {str(e)[:80]}")
        dr.append(r)
    with open(os.path.join(dd,"results.json"),'w') as f:json.dump(dr,f,ensure_ascii=False,indent=2)
    all_results.extend(dr)
    okn=sum(1 for r in dr if r.get("error") is None)
    print(f"  ✅ {dim_name}: {okn}/{len(dr)} | {time.time()-t0:.0f}s")

elapsed=time.time()-t0
ok=[r for r in all_results if r.get("error") is None]
print(f"\nTimesFM: {len(ok)}/{len(all_results)} OK | {elapsed/60:.1f}min")
with open(os.path.join(OUT,"all_results.jsonl"),'w') as f:
    for r in all_results:f.write(json.dumps(r,ensure_ascii=False)+'\n')
print(f"✅ {OUT}")
