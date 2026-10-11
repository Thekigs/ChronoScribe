#!/usr/bin/env python3
"""Paired ablation: LoRA-adapted language model without the specialist.

It retains the reference manifest and controller representation while removing
specialist numerical forecasting. This measures a changed system, not an
additive contribution of Chronos-2 to a composite score.
"""
import os,sys,json,time,re,random
from collections import defaultdict,Counter

BASE=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH=os.path.join(BASE,'benchmark')
OUT=os.path.join(BASE,'results','ablate_noc2')
sys.path.insert(0,BASE)
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dims', type=str, default='')
parser.add_argument('--manifest', type=str,
    default=os.path.join(BENCH, 'evaluation_manifest.json'),
    help='Fixed evaluation manifest shared by every compared system.')
args = parser.parse_args()

# Keep --help usable before optional model dependencies and output paths exist.
import numpy as np
import pandas as pd
import torch
os.makedirs(OUT,exist_ok=True)

MODEL_DIR=os.path.join(BASE,'models','Qwen3.5-9B')
ADAPTER_DIR=os.path.join(BASE,'models','checkpoint-7500')
CHRONOS2_DIR=os.path.join(BASE,'models','chronos-2')

print("="*60)
print("Ablation: NO Chronos2 (Pure LLM)")
print("="*60)

from transformers import AutoModelForCausalLM,AutoTokenizer
from peft import PeftModel

print("[1/2] Qwen3.5+LoRA...")
qb=AutoModelForCausalLM.from_pretrained(MODEL_DIR,torch_dtype=torch.float16,device_map='cpu')
if torch.cuda.is_available(): DEVICE='cuda'
elif hasattr(torch.backends,'mps') and torch.backends.mps.is_available(): DEVICE='mps'
else: DEVICE='cpu'
print(f"  Device: {DEVICE}")
qwen=PeftModel.from_pretrained(qb,ADAPTER_DIR).to(DEVICE);qwen.eval()
tok=AutoTokenizer.from_pretrained(MODEL_DIR)
if tok.pad_token is None:tok.pad_token=tok.eos_token
print(f"  Qwen: {sum(p.numel() for p in qwen.parameters())/1e9:.2f}B")

print("[2/2] No Chronos2 — pure LLM mode")
c2=None  # No Chronos2 for this ablation
print("  Ready (pure LLM)\n")

def qgen(msgs,mt=384):
    text=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
    inputs=tok(text,return_tensors='pt').to(DEVICE)
    with torch.no_grad():
        out=qwen.generate(**inputs,max_new_tokens=mt,do_sample=False,pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][inputs['input_ids'].shape[1]:],skip_special_tokens=True)

def extract_json(text):
    if not text:return None
    try:return json.loads(text)
    except Exception:pass
    for m in re.findall(r'```(?:json)?\s*\n?(.*?)\n?```',text,re.DOTALL):
        try:return json.loads(m)
        except Exception:pass
    for i,c in enumerate(text):
        if c=='{':
            d=1;s=i
            for j in range(i+1,len(text)):
                if text[j]=='{':d+=1
                elif text[j]=='}':
                    d-=1
                    if d==0:
                        try:return json.loads(text[s:j+1])
                        except Exception:pass;break
    return None

# ═══ FINAL LOADER ═══
RAW={};RAW_SRC={}
SOLAR_DF=pd.read_csv(os.path.join(BASE,'data','solar','solar_AL.txt'),header=None) if os.path.exists(os.path.join(BASE,'data','solar','solar_AL.txt')) else None
TOUR=json.load(open(os.path.join(BASE,'data','tourism','tourism_data.json'))) if os.path.exists(os.path.join(BASE,'data','tourism','tourism_data.json')) else None

DATASET_PATHS={"ETTh1":"data/ETT-small/ETTh1.csv","ETTh2":"data/ETT-small/ETTh2.csv",
    "ETTm1":"data/ETT-small/ETTm1.csv","ETTm2":"data/ETT-small/ETTm2.csv",
    "weather":"data/weather/weather.csv","traffic":"data/traffic/traffic.csv",
    "Weather":"data/weather/weather.csv","Traffic":"data/traffic/traffic.csv",
    "electricity":"data/electricity/electricity.csv","Electricity":"data/electricity/electricity.csv",
    "exchange_rate":"data/exchange_rate/exchange_rate.csv","exchange":"data/exchange_rate/exchange_rate.csv",
    "Exchange":"data/exchange_rate/exchange_rate.csv","national_illness":"data/illness/national_illness.csv",
    "Illness":"data/illness/national_illness.csv",
    "MSL":"MSL/","SMAP":"SMAP/","SMD":"SMD/","PSM":"PSM/","NAB":"nab/",
    "solar":"solar/","Solar":"solar/","nab":"nab/","SWaT":"SWaT/",
    "M4":"m4/","NAB":"nab/","YahooS5":"yahoo_s5/",
}

DOMAIN_FILES={
    "energy":["data/electricity/electricity.csv","data/ETT-small/ETTh1.csv","data/ETT-small/ETTh2.csv","data/ETT-small/ETTm1.csv","data/ETT-small/ETTm2.csv"],
    "transport":["data/traffic/traffic.csv"],"weather":["data/weather/weather.csv"],"climate":["data/weather/weather.csv"],
    "finance":["data/exchange_rate/exchange_rate.csv"],"health":["data/illness/national_illness.csv"],
    "business":["tourism/"],"cloud_monitoring":["nab/"],"aerospace":["MSL/"],
    "spacecraft":["SMAP/","MSL/"],"cloud":["SMD/","PSM/","nab/"],
    "general":["m4/"],"utility":["data/electricity/electricity.csv"],
}

SKIP_COLS={"date","OT","timestamp","timestamp_(min)","time","datetime"}

def cache_raw(key,v,src):
    RAW[key]=v;RAW_SRC[key]=src
    return v,src

def data_columns(df):
    return [c for c in df.columns if str(c) not in SKIP_COLS and str(c).lower() not in SKIP_COLS]

def normalize_series_name(ds,series):
    name=str(series)
    prefixes=[str(ds)+"_",str(ds).lower()+"_","ETTh1_","ETTh2_","ETTm1_","ETTm2_",
              "Weather_","weather_","Electricity_","electricity_","Traffic_","traffic_",
              "Exchange_","exchange_"]
    changed=True
    while changed:
        changed=False
        for pfx in prefixes:
            if name.startswith(pfx):
                name=name[len(pfx):];changed=True
    name=re.sub(r'_(?:H|h)\d+$','',name)
    name=re.sub(r'_h\d+$','',name)
    name=re.sub(r'_(?:noise|missing|spike|noise_burst|level_shift|trend_break).*$', '', name)
    if name.startswith("multi_"):name=name[len("multi_"):]
    return name

def resolve_csv_column(df,ds,series,sid=""):
    dcols=data_columns(df)
    if not dcols:return None,None
    col_name=normalize_series_name(ds,series)
    probes=[col_name,str(series),str(ds),str(sid)]
    for p in probes:
        if p in dcols:return p,"exact"
    for c in dcols:
        if str(c)==col_name:return c,"exact"
    for c in dcols:
        if col_name and (col_name==str(c) or col_name in str(c) or str(c) in col_name):
            return c,"name"
    nums=re.findall(r'\d+',str(series))
    for n in nums:
        idx=int(n)
        if 0<=idx<len(dcols):return dcols[idx],"index"
    return None,None

def load_solar(sample,key):
    if SOLAR_DF is None:return None,None
    series=str(sample.get("series",""));sid=str(sample.get("id",""))
    nums=re.findall(r'(?:plant|p)(\d+)',series+sid)
    if not nums:nums=re.findall(r'\d+',series)
    for n in nums:
        ci=int(n)
        if 0<=ci<SOLAR_DF.shape[1]:
            v=SOLAR_DF.iloc[:,ci].dropna().values.astype(np.float64)
            if len(v)>50:return cache_raw(key,v,f"solar:col{ci}")
    v=SOLAR_DF.iloc[:,0].dropna().values.astype(np.float64)
    if len(v)>50:return cache_raw(key,v,"solar:col0:fallback")
    return None,None

def load_tourism(sample,key):
    if not TOUR:return None,None
    series=str(sample.get("series",""));sid=str(sample.get("id",""))
    text=series+" "+sid
    for s in TOUR:
        if s.get('id') and (s['id'] in text or text.strip()==s['id']):
            return cache_raw(key,np.array(s['values'],dtype=np.float64),f"tourism:{s['id']}")
    nums=re.findall(r'\d+',text)
    for n in nums:
        tid=f"tourism_{int(n)}"
        for s in TOUR:
            if s.get('id')==tid:
                return cache_raw(key,np.array(s['values'],dtype=np.float64),f"tourism:{tid}")
    return None,None

def load_m4(sample,key):
    series=str(sample.get("series",""));sid=str(sample.get("id",""));ds=str(sample.get("dataset",""))
    text=" ".join([series,sid,ds])
    cats=['Yearly','Quarterly','Monthly','Weekly','Daily','Hourly']
    cat_hints=[c for c in cats if c.lower() in text.lower()]
    id_match=re.search(r'\b([YQMWDH]\d+)\b',text)
    exact_id=id_match.group(1) if id_match else None
    search_cats=cat_hints+([] if cat_hints else cats)
    for m4c in search_cats:
        mfp=os.path.join(BASE,'data','m4',f'{m4c}-train.csv')
        if not os.path.exists(mfp):continue
        df=pd.read_csv(mfp)
        if exact_id is not None:
            hit=df[df.iloc[:,0].astype(str)==exact_id]
            if len(hit)>0:
                row=hit.iloc[0]
                v=pd.to_numeric(pd.Series(row.values[1:]),errors='coerce').dropna().values.astype(np.float64)
                if len(v)>10:return cache_raw(key,v,f"m4:{m4c}:{exact_id}")
    return None,None

def load_m5(sample,key):
    """Load M5 retail competition data. ID format: v8_m5_st_HOBBIES_1_014_CA_1_evaluation_h28"""
    sid=str(sample.get("id",""));series=str(sample.get("series",""))
    # Extract M5 item_id from ID: everything between "m5_" and the trailing "_h\d+"
    item_id=None
    for pattern in [r'm5_st_(.+?)_h\d+', r'm5_(.+?)_h\d+', r'M5_(.+?)_h\d+']:
        m=re.search(pattern,sid)
        if m:item_id=m.group(1);break
    if not item_id:
        # Try from series
        m=re.search(r'HOBBIES_\d+_\d+_\w+_\d+',series+sid)
        if m:item_id=m.group(0)
    if not item_id:return None,None
    m5_dir=os.path.join(BASE,'data','m5-forecasting-accuracy')
    eval_path=os.path.join(m5_dir,'sales_train_evaluation.csv')
    if not os.path.exists(eval_path):return None,None
    try:
        # Only load CSV once
        m5_cache=getattr(load_m5,'_df',None)
        if m5_cache is None:
            m5_cache=pd.read_csv(eval_path)
            load_m5._df=m5_cache
        df=m5_cache
        # Match by 'id' column
        hit=df[df['id'].astype(str)==item_id]
        if len(hit)==0:
            # Try partial match
            hit=df[df['id'].astype(str).str.contains(item_id[:20],na=False)]
        if len(hit)>0:
            row=hit.iloc[0]
            # Values are in columns d_1 through d_1941
            d_cols=[c for c in df.columns if c.startswith('d_') and c[2:].isdigit()]
            vals=pd.to_numeric(row[d_cols],errors='coerce').dropna().values.astype(np.float64)
            if len(vals)>50:return cache_raw(key,vals,f"m5:{item_id}")
    except Exception as e:
        pass
    return None,None

def load_nab(sample,key):
    """Load NAB (Numenta Anomaly Benchmark) data.
    ID format: v7_outlier_level_shift_NAB_realAWSCloudwatch_ec2_cpu_utilization_asg_misconfiguration
    → subdir: realAWSCloudwatch, file prefix: ec2_cpu_utilization
    """
    sid=str(sample.get("id",""));series=str(sample.get("series",""))
    # Extract NAB category and file prefix from ID
    nab_cats=['realAWSCloudwatch','realKnownCause','realTraffic','realTweets',
              'artificialWithAnomaly','artificialNoAnomaly','realAdExchange','real']
    nab_cat=None;file_prefix=None
    for cat in nab_cats:
        if cat in sid:
            nab_cat=cat
            # Everything after the category is the file prefix until the next underscore + anomaly type
            idx=sid.find(cat)+len(cat)+1
            remainder=sid[idx:] if idx<len(sid) else ""
            # Remove perturbation suffixes like _level_shift, _spike, _noise_burst, _trend_break
            for suffix in ['_level_shift','_spike','_noise_burst','_trend_break']:
                # These appear BEFORE the NAB category, not after
                pass
            # The file prefix is the remainder, but may include perturbation info
            # Strip known perturbation suffixes from the end
            file_prefix=remainder
            for pert in ['_level_shift','_spike','_noise_burst','_trend_break','_s0','_s1']:
                if file_prefix.endswith(pert):
                    file_prefix=file_prefix[:-len(pert)]
            break
    if not nab_cat or not file_prefix:
        return None,None
    nab_dir=os.path.join(BASE,'data','nab',nab_cat)
    if not os.path.exists(nab_dir):return None,None
    # Find matching CSV by prefix (filename may have hash suffix)
    # Try progressively shorter prefixes
    matches=[]
    for fn in sorted(os.listdir(nab_dir)):
        if fn.endswith('.csv'):
            for trim_len in range(len(file_prefix), max(len(file_prefix)//2, 5), -1):
                if fn.startswith(file_prefix[:trim_len]):
                    matches.append(fn);break
    for fn in matches[:1]:
        try:
            df=pd.read_csv(os.path.join(nab_dir,fn))
            if 'value' in df.columns:
                v=df['value'].dropna().values.astype(np.float64)
                if len(v)>50:return cache_raw(key,v,f"nab:{nab_cat}:{fn}")
            elif len(df.columns)>=2:
                # Try second column as values
                v=pd.to_numeric(df.iloc[:,1],errors='coerce').dropna().values.astype(np.float64)
                if len(v)>50:return cache_raw(key,v,f"nab:{nab_cat}:{fn}")
        except Exception:pass
    return None,None

def load_uea_ucr(sample,key):
    """Load UEA/UCR/MONSTER classification time series data.
    ID format: v8_UCR_Wafer_0_d0 → dataset=UCR_Wafer, dim=0
    ID format: v8_MONSTER_MonsterTraffic_4_d0 → dataset=MONSTER_MonsterTraffic, dim=4
    """
    ds_lower=""
    sid=str(sample.get("id",""));ds=str(sample.get("dataset",""))
    if not ds:return None,None
    # Map dataset name to data directory
    ds_lower=ds.lower()
    base_map={"ucr_":"data/ucr_classification/","monster_":"data/monster_classification/"}
    data_dir=None;ds_name=None
    for prefix, bp in base_map.items():
        if ds_lower.startswith(prefix):
            ds_name=ds[len(prefix):]  # e.g., "Wafer", "MonsterTraffic"
            data_dir=os.path.join(BASE,bp,ds_name)
            break
    if not data_dir or not os.path.exists(data_dir):
        return None,None
    # Find .ts files
    ts_files=sorted([f for f in os.listdir(data_dir) if f.endswith('.ts')])
    if not ts_files:
        # Check MONSTER subdirectories
        for sub in os.listdir(data_dir):
            subp=os.path.join(data_dir,sub)
            if os.path.isdir(subp):
                ts_files=sorted([os.path.join(sub,f) for f in os.listdir(subp) if f.endswith('.ts')])
                if ts_files:break
    if not ts_files:return None,None
    # Parse the .ts file — prefer TEST, fallback to TRAIN
    test_files=[f for f in ts_files if 'TEST' in f.upper()]
    train_files=[f for f in ts_files if 'TRAIN' in f.upper()]
    target_file=(test_files or train_files or ts_files)[0]
    # Handle nested paths
    if not os.path.sep in str(target_file):
        target_file=os.path.join(data_dir,str(target_file))
    try:
        from utils.converter_utils import parse_ts_file
        parsed=parse_ts_file(str(target_file))
        if not parsed:return None,None
        # Extract dimension index from ID: e.g., v8_UCR_Wafer_0_d0 → dim=0
        dim_match=re.search(r'_(\d+)_d\d+',sid) or re.search(r'_(\d+)$',sid)
        dim_idx=int(dim_match.group(1)) if dim_match else 0
        # Each parsed entry: (label, [dim1_array, dim2_array, ...])
        # We want a specific dimension across all samples → concatenate
        # Clamp dim_idx to actual available dimensions
        dim_counts=[len(d[1]) for _,d in parsed if len(d)>1 and len(d[1])>0]
        max_dims=min(dim_counts) if dim_counts else 1
        if dim_idx>=max_dims:dim_idx=0
        valid_dims=[]
        for label,dimensions in parsed:
            if len(dimensions)>dim_idx and len(dimensions[dim_idx])>0:
                valid_dims.append(dimensions[dim_idx])
        if valid_dims:
            # Concatenate all samples for this dimension
            all_vals=np.concatenate(valid_dims).astype(np.float64)
            if len(all_vals)>20:
                return cache_raw(key,all_vals,f"uea:{ds}:dim{dim_idx}")
    except Exception as e:
        pass
    return None,None

def locate_gt_window(data,gt_values,naive_forecast=None):
    if not isinstance(gt_values,list) or len(gt_values)==0:return None
    v=np.asarray(data,dtype=np.float64);g=np.asarray(gt_values,dtype=np.float64)
    if len(v)<len(g):return None
    tol=1e-2+1e-5*max(1.0,float(np.nanmax(np.abs(g))))
    first=np.where(np.abs(v[:len(v)-len(g)+1]-g[0])<=tol)[0]
    candidates=[]
    for st in first:
        seg=v[st:st+len(g)]
        if len(seg)==len(g) and np.allclose(seg,g,atol=tol,rtol=1e-5):
            candidates.append(int(st))
    if not candidates:return None
    if naive_forecast is not None:
        nf=float(naive_forecast)
        nf_hits=[st for st in candidates if st>0 and abs(float(v[st-1])-nf)<=max(tol,abs(nf)*1e-5+1e-4)]
        if nf_hits:return nf_hits[-1]
    return candidates[-1]

def build_context(data,sample,gt_values,qwen_ctx_len=None):
    if qwen_ctx_len is not None:
        try:hint_cl=int(qwen_ctx_len)
        except Exception:hint_cl=sample.get("context_length",512)
    else:
        hint_cl=sample.get("context_length",512)
    try:hint_cl=int(hint_cl)
    except Exception:hint_cl=512
    hint_cl=max(1,min(hint_cl,len(data)))
    gt=sample.get("ground_truth",{})
    gth=len(gt_values) if isinstance(gt_values,list) else 0
    st=locate_gt_window(data,gt_values,gt.get("naive_forecast")) if gth>0 else None
    if st is not None and st>0:
        ctx=data[max(0,st-hint_cl):st]
        return ctx,{"ctx_len":len(ctx),"requested_ctx_len":hint_cl,"gt_start_index":st,"context_aligned":"gt_window"}
    if gth>0 and len(data)>gth:
        end=len(data)-gth
        ctx=data[max(0,end-hint_cl):end]
        return ctx,{"ctx_len":len(ctx),"requested_ctx_len":hint_cl,"gt_start_index":end,"context_aligned":"tail_before_gt"}
    ctx=data[-hint_cl:]
    return ctx,{"ctx_len":len(ctx),"requested_ctx_len":hint_cl,"gt_start_index":None,"context_aligned":"tail_no_gt"}

def load_raw(sample):
    ds_lower=""
    ds=str(sample.get("dataset",""))
    ds_lower=ds.lower()
    series_raw=sample.get("series")
    need_extract = series_raw is None or str(series_raw).strip()=="" or str(series_raw)=="None"
    if need_extract or (series_raw and not any(kw in str(series_raw) for kw in ['HUFL','HULL','MUFL','MULL','LUFL','LULL','T (','Tpot','Tdew','rh','VP','sh','H2O','rho','wv','wd','rain','OT','p (','AGE','ILI','NUM'])):
        # Series name doesn't look like a real column — try to extract from ID
        sid_raw=str(sample.get("id",""))
        extracted_from_id = None  # track what we find in the ID
        # Build known dataset patterns (case-insensitive)
        known_patterns = [
            ("ETTh1", None), ("ETTh2", None), ("ETTm1", None), ("ETTm2", None),
            ("Weather", "weather"), ("weather", None),
            ("Electricity", "electricity"), ("electricity", None),
            ("Traffic", "traffic"), ("traffic", None),
            ("Exchange", "exchange_rate"), ("exchange_rate", None),
            ("Illness", "national_illness"), ("national_illness", None),
            ("Solar", "solar"), ("solar", None),
        ]
        for pattern, alt in known_patterns:
            if pattern in sid_raw or (alt and alt in sid_raw):
                match_str = pattern if pattern in sid_raw else alt
                parts = sid_raw.split(match_str + "_", 1) if match_str else ["", sid_raw]
                if len(parts) > 1 and parts[1]:
                    remainder = parts[1]
                    # Try known column suffixes with parentheses
                    for suffix in ['(degC)', '(K)', '(mbar)', '(g/kg)', '(mmol/mol)',
                                   '(g/m**3)', '(m/s)', '(deg)', '(mm)', '(%)']:
                        if suffix in remainder:
                            idx = remainder.find(suffix)
                            candidate = remainder[:idx+len(suffix)].strip()
                            if candidate and candidate not in ('h0','h24','h48','h96','h12','h336','h720'):
                                extracted_from_id = candidate
                                break
                    if extracted_from_id is None:
                        # For exchange_rate / numeric datasets: extract first number
                        if ds_lower in ("exchange_rate", "exchange", "traffic", "electricity"):
                            nums = re.findall(r'\d+', remainder)
                            if nums:
                                extracted_from_id = nums[0]  # first number = column index
                        if extracted_from_id is None:
                            # For weather/illness: series name with possible spaces
                            end_match = re.search(r'[_\s](?:p\d+|h\d+|h0|s\d+|nl\d+)$', remainder)
                            if end_match:
                                candidate = remainder[:end_match.start()].strip()
                            else:
                                candidate = re.split(r'_(?:h\d+|p\d+|s\d+)', remainder)[0].strip()
                            if candidate and candidate not in ('0','1','2','3','4','5','6','7','8','9',
                                'eff','eff2','eff_short','eff_medium','eff_long','eff_xlong',
                                'e_short','missing'):
                                extracted_from_id = candidate
                if extracted_from_id is not None:
                    break
        # Use extracted value: either when series was originally None, or when
        # the existing series name is clearly synthetic (e.g., "Electricity_eff_short")
        if extracted_from_id is not None:
            if need_extract or series_raw is None or str(series_raw) == "None" or str(series_raw) == "":
                series_raw = extracted_from_id
            elif not any(kw in str(series_raw) for kw in ['HUFL','HULL','MUFL','MULL','LUFL','LULL',
                  'T (','Tpot','Tdew','rh','VP','sh','H2O','rho','wv','wd','rain','OT','p (',
                  'AGE','ILI','NUM']):
                # Existing series looks synthetic → use extracted value
                series_raw = extracted_from_id
        if series_raw is None or str(series_raw)=="None":
            series_raw = ""
    series=str(series_raw)
    domain=str(sample.get("domain",""));sid=str(sample.get("id",""))
    gt=sample.get("ground_truth",{});gv=gt.get("values",[])
    key=f"{ds}|{series}"
    if key in RAW:return RAW[key],RAW_SRC.get(key,"cache")

    if ds in ("Solar","solar") or series.lower().startswith(("solar","wind","hydro","grid")):
        v,src=load_solar(sample,key) if ("solar" in ds.lower() or "solar" in series.lower()) else (None,None)
        if v is not None:return v,src

    if ds.lower() in ("tourism","ood_tourism") or "tourism" in series.lower() or "tourism" in sid.lower():
        v,src=load_tourism(sample,key)
        if v is not None:return v,src

    if ds=="M4" or ds.startswith("M4") or "M4" in series or "M4" in sid:
        v,src=load_m4(sample,key)
        if v is not None:return v,src

    if "M5" in ds or "m5" in ds_lower or "m5" in sid.lower():
        v,src=load_m5(sample,key)
        if v is not None:return v,src

    if "NAB" in ds or "nab" in ds_lower or "nab" in sid.lower():
        v,src=load_nab(sample,key)
        if v is not None:return v,src

    if ds_lower.startswith(("ucr_","monster_","uea_")):
        v,src=load_uea_ucr(sample,key)
        if v is not None:return v,src

    # Resolve a direct dataset path
    if ds in DATASET_PATHS:
        fp=DATASET_PATHS[ds];fpath=os.path.join(BASE,'data',fp) if not fp.startswith('data/') else os.path.join(BASE,fp)
        if fp.endswith('.csv') and os.path.exists(fpath):
            df=pd.read_csv(fpath);col,_=resolve_csv_column(df,ds,series,sid)
            if col:
                v=df[col].dropna().values.astype(np.float64)
                if len(v)>50:return cache_raw(key,v,f"csv:{ds}:{col}")
        elif fp.endswith('/'):
            pfx=fp.replace('/','')
            for spl in ["train","test"]:
                npf=os.path.join(BASE,'data',pfx,f"{pfx}_{spl}.npy")
                if os.path.exists(npf):
                    d=np.load(npf)
                    nums=re.findall(r'\d+',series+sid)
                    for n in nums:
                        fi=int(n)
                        if 0<=fi<d.shape[1]:v=d[:,fi].astype(np.float64);return cache_raw(key,v,f"csv:{pfx}:f{fi}")
                    if d.shape[1]>0:v=d[:,0].astype(np.float64);return cache_raw(key,v,f"csv:{pfx}:f0")

    # 2) Domain + GT range
    candidates=DOMAIN_FILES.get(domain,["data/ETT-small/ETTh1.csv","data/traffic/traffic.csv","data/weather/weather.csv","data/electricity/electricity.csv","data/exchange_rate/exchange_rate.csv"])
    best_match=None;best_score=-1

    for fp in candidates:
        fpath=os.path.join(BASE,'data',fp) if not fp.startswith('data/') else os.path.join(BASE,fp)
        if not os.path.exists(fpath):continue
        if fp=="tourism/":
            if TOUR:
                for s in TOUR:
                        if s['id'] in series or series in s['id'] or s['id'] in sid:
                            v=np.array(s['values'],dtype=np.float64);return cache_raw(key,v,f"tourism:{s['id']}")
                continue
        if fp=="m4/":
            nums=re.findall(r'\d+',series+sid)
            for m4c in ['Yearly','Quarterly','Monthly','Weekly','Daily','Hourly']:
                mfp=os.path.join(BASE,'data','m4',f'{m4c}-train.csv')
                if os.path.exists(mfp):
                    df=pd.read_csv(mfp)
                    for n in nums:
                        for _,row in df.iterrows():
                            if str(row.values[0])==n:
                                v=pd.to_numeric(pd.Series(row.values[1:]),errors='coerce').dropna().values.astype(np.float64)
                                if len(v)>10:return cache_raw(key,v,f"m4:{m4c}:{n}")
        if fp=="nab/":
            nd=os.path.join(BASE,'data','nab')
            if os.path.exists(nd):
                for cd in os.listdir(nd):
                    for fn in os.listdir(os.path.join(nd,cd)):
                        if fn.endswith('.csv'):
                            try:
                                df=pd.read_csv(os.path.join(nd,cd,fn))
                                if 'value' in df.columns:
                                    v=df['value'].dropna().values.astype(np.float64)
                                    if len(v)>100:return cache_raw(key,v,"nab")
                            except Exception:pass
        if not fp.endswith('.csv'):continue

        try:
            df=pd.read_csv(fpath);dcols=data_columns(df)
            if not dcols:continue
            col,how=resolve_csv_column(df,ds,series,sid)
            if col is not None:
                v=df[col].dropna().values.astype(np.float64)
                if len(v)<50:continue
                best_match=(v,f"dom:{fp.split('/')[-1]}:{col}:{how}",0);best_score=1
        except Exception:pass

    if best_match:
        v,src,ci=best_match
        return cache_raw(key,v,src)

    return None,"not_found:no_exact_match"

# Shared protocol keeps benchmark membership and data alignment consistent.
from utils.benchmark_data import build_context as shared_build_context, load_raw as shared_load_raw
from utils.controller_inputs import history_summary, request_text
from utils.evaluation_manifest import load_manifest_samples

build_context = shared_build_context
load_raw = shared_load_raw

# ═══ LOAD FIXED EVALUATION SET ═══

print("Loading fixed evaluation manifest...")
dims_filter = [d.strip() for d in args.dims.split(',') if d.strip()] if args.dims else None
test_plan = load_manifest_samples(BENCH, args.manifest, dims_filter)
for dim, samples in sorted(test_plan.items()):
    print(f"  {dim}: {len(samples)} fixed samples")
print(f"\nTotal: {sum(len(v) for v in test_plan.values())} samples across {len(test_plan)} dimensions")

# ═══ RUN ═══
T1SYS=("只outputJSON。判断是否需要time series forecasting：\n"
    "有historical data+预测需求 → {\"use_chronos2\":true,\"prediction_length\":int,\"context_length\":int}\n"
    "普通对话/无数据 → {\"use_chronos2\":false,\"direct_reply\":\"简短直接回复\"}\n"
    "禁止outputJSON之外的任何Text、思考过程或解释。")
T2SYS=("你是time series forecasting分析专家。将forecast results转化为natural language解读。JSON:{\"trend\":\"...\",\"analysis\":\"...\"}")

print(f"\n{'='*60}")
print("Starting evaluation")
print(f"{'='*60}")

all_results=[];t0=time.time();issues=[]

for dim_name,samples in sorted(test_plan.items()):
    dt0=time.time();dr=[];di=[]
    dd=os.path.join(OUT,dim_name);os.makedirs(dd,exist_ok=True)
    print(f"\n[{dim_name}] {len(samples)} samples")

    for i,sample in enumerate(samples):
        t1=time.time()
        gt=sample.get("ground_truth",{});cat=sample.get("category","")
        r={"id":sample["id"],"dimension":dim_name,"category":cat,
           "dataset":sample.get("dataset","?"),"domain":sample.get("domain","?")}
        si=[]

        try:
            data,dsrc=load_raw(sample)
            has_data=data is not None and len(data)>10
            gv=gt.get("values",[]);has_gv=isinstance(gv,list) and len(gv)>0
            gth=len(gv) if has_gv else 0
            r["has_data"]=has_data;r["data_source"]=dsrc;r["gt_horizon"]=gth

            if not has_data:
                msgs=sample.get("prompt",[])
                raw_out=qgen(msgs,384)
                parsed=extract_json(raw_out)
                r["raw_output"]=raw_out;r["parsed_output"]=parsed;r["pipeline"]="ablate_noc2"
                if has_gv:si.append(f"NO_DATA: GT有{gth}vals src={dsrc}");di.append(si[-1])
            else:
                r["pipeline"]="ablate_noc2"
                hint_h=gth if gth>0 else 24
                # Use the same task and history representation as the full system;
                # only the numerical specialist is removed in this ablation.
                ctx_prelim,_=build_context(data,sample,gv,qwen_ctx_len=min(4096,len(data)))
                if len(ctx_prelim)<=10:
                    raise RuntimeError(f"context too short after alignment: {len(ctx_prelim)}")
                r["context_alignment"]=_
                task_text=request_text(sample)
                summary=history_summary(ctx_prelim)
                s1r=qgen([{"role":"system","content":T1SYS},
                    {"role":"user","content":(
                        f"user request:\n{task_text}\n\n历史摘要:\n{summary}\n\n"
                        f"本评测要求预测 {hint_h} 步。选择execution mode和历史context length。JSON。"
                    )}],256)
                s1p=extract_json(s1r) or {}
                r["step1_params"]=s1p

                # LLM-based Intent Routing: Qwen decides if Chronos2 is needed
                # Ablation: skip Chronos2 entirely
                r["pipeline"]="ablate_noc2"
                r["raw_output"]=s1r
                r["parsed_output"]=s1p
                msgs=sample.get("prompt",[])
                # Use original benchmark prompt to get predicted_values
                raw_out = qgen(msgs, 2048)
                parsed = extract_json(raw_out)
                r["raw_output"] = raw_out
                r["parsed_output"] = parsed
                if isinstance(parsed, dict):
                    pred = parsed.get("predicted_values", parsed.get("forecast", parsed.get("predictions")))
                    if isinstance(pred, list):
                        pv = [float(v) for v in pred if isinstance(v, (int, float))]
                        if gth > 0:
                            if len(pv) > gth: pv = pv[:gth]
                            elif len(pv) < gth: pv = pv + [pv[-1]] * (gth - len(pv)) if pv else [0] * gth
                        r["predicted_values"] = pv
                        if gv and pv:
                            from scoring.detailed_scoring import compute_all_forecast_metrics
                            r["forecast_metrics"] = compute_all_forecast_metrics(
                                np.array(gv), np.array(pv), naive_forecast=gt.get("naive_forecast"))
                    analysis = parsed.get("analysis", parsed.get("explanation", ""))
                    if analysis: r["analysis"] = str(analysis)
                r["latency_s"]=round(time.time()-t1,3); r["error"]=None
                dr.append(r); continue

                pl=s1p.get("prediction_length",hint_h)
                if not isinstance(pl,(int,float)) or pl<=0:pl=hint_h
                pl=min(int(pl),720)

                if gth>0 and pl!=gth:
                    r["horizon_mismatch"]=f"Q={pl},GT={gth}"
                    if abs(pl-gth)/gth>0.5:si.append(f"HORIZON: Qwen={pl} GT={gth}");di.append(si[-1])
                    pl=gth;r["horizon_corrected"]=True

                # Step2: use Qwen's context_length, with reasonable default (512 not 4096)
                qwen_cl=s1p.get("context_length") or min(512,len(data))
                ctx,ctx_meta2=build_context(data,sample,gv,qwen_ctx_len=qwen_cl)
                if len(ctx)<=10:ctx=ctx_prelim  # fallback
                cl=len(ctx);r["c2_params"]={"pred_len":pl,"ctx_len":cl,"qwen_requested_cl":qwen_cl}
                ct=torch.tensor(ctx,dtype=torch.float32).reshape(1,1,-1)
                # Preserve forecast quantiles
                qlist,medlist=c2.predict_quantiles(ct,prediction_length=pl)
                if isinstance(qlist,list) and len(qlist)>0:
                    q=qlist[0]
                    if hasattr(q,'shape') and len(q.shape)==3:
                        nq=q.shape[2]
                        preds_p10=q[0,:,0].tolist()
                        preds_p50=q[0,:,nq//2].tolist()
                        preds_p90=q[0,:,-1].tolist()
                        preds=preds_p50
                    else:
                        preds=medlist[0].squeeze().tolist()
                        preds_p10=preds_p90=preds
                else:
                    preds=medlist[0].squeeze().tolist()
                    preds_p10=preds_p90=preds
                if isinstance(preds,float):preds=[preds]

                if gth>0:
                    if len(preds)>gth:preds=preds[:gth];r["truncated"]=True
                    elif len(preds)<gth:
                        pad=gth-len(preds);preds=preds+[preds[-1]]*pad if preds else [0]*pad
                        r["padded"]=True
                r["predicted_values"]=preds
                r["predicted_p10"]=preds_p10
                r["predicted_p50"]=preds_p50
                r["predicted_p90"]=preds_p90
                # GPU memory
                try:
                    import torch as _t
                    if _t.cuda.is_available():
                        r["gpu_memory_mb"]={"cuda_allocated_mb":round(_t.cuda.memory_allocated()/1024/1024,1)}
                    elif _t.backends.mps.is_available():
                        r["gpu_memory_mb"]={"mps_allocated_mb":round(_t.mps.current_allocated_memory()/1024/1024,1)}
                except Exception:pass

                if len(preds)>0:
                    rep=(f"预测:{len(preds)}步 起:{preds[0]:.6g} 止:{preds[-1]:.6g} "
                         f"低:{min(preds):.6g} 高:{max(preds):.6g}")
                else:rep="预测Failed"
                r["analysis"]=qgen([{"role":"system","content":T2SYS},{"role":"user","content":rep}],384)

                if gv and preds:
                    from scoring.detailed_scoring import compute_all_forecast_metrics
                    assert len(preds)==len(gv),f"LEN:{len(preds)}vs{len(gv)}"
                    r["forecast_metrics"]=compute_all_forecast_metrics(
                        np.array(gv),np.array(preds),naive_forecast=gt.get("naive_forecast"))
                elif not gv and preds:
                    r["forecast_evaluation_status"]="unscorable_no_manifest_target"

            from scoring.detailed_scoring import (compute_all_text_metrics,compute_json_metrics,
                compute_reasoning_metrics)
            if "reference_analysis" in gt and r.get("analysis"):
                r["text_metrics"]=compute_all_text_metrics(str(gt["reference_analysis"]),str(r["analysis"]))
            r["json_metrics"]=compute_json_metrics(r.get("raw_output",""),r.get("parsed_output"))
            if has_gv and not has_data:
                r["reasoning_metrics"]=compute_reasoning_metrics(r.get("parsed_output",{}),gt)

            r["latency_s"]=round(time.time()-t1,3);r["error"]=None
            c2t="📊" if has_data else "🧠";it=f" ⚠️{len(si)}" if si else ""
            fm=r.get("forecast_metrics",{})
            smape=f" sMAPE={fm.get('sMAPE',0):.0f}%" if 'sMAPE' in fm else ""
            print(f"  [{i+1}/{len(samples)}] {c2t}{it} {sample['id'][:35]}... ✅ {r['latency_s']:.1f}s{smape} {dsrc[:25]}")
        except Exception as e:
            r["error"]=str(e)[:200];r["latency_s"]=round(time.time()-t1,3)
            si.append(f"❌ {str(e)[:80]}");di.append(si[-1])
            print(f"  [{i+1}/{len(samples)}] ❌ {str(e)[:80]}")

        if si:
            for iss in si:print(f"     {iss}")
        dr.append(r)

    with open(os.path.join(dd,"results.json"),'w') as f:json.dump(dr,f,ensure_ascii=False,indent=2)
    all_results.extend(dr)
    okn=sum(1 for r in dr if r.get("error") is None)
    c2n=sum(1 for r in dr if r.get("pipeline")=="ablate_noc2")
    print(f"  ✅ {dim_name}: {okn}/{len(dr)} | {c2n}xC2 | {len(di)} issues | {time.time()-dt0:.0f}s")
    issues.extend(di)

elapsed=time.time()-t0
ok=[r for r in all_results if r.get("error") is None]
fail=[r for r in all_results if r.get("error") is not None]
c2s=[r for r in ok if r.get("pipeline")=="ablate_noc2"]
fc=[r for r in ok if 'forecast_metrics' in r]

print(f"\n{'='*60}")
print("Evaluation complete")
print(f"{'='*60}")
print(f"  sample:{len(all_results)} | Succeeded:{len(ok)} | Failed:{len(fail)} | {len(ok)/max(len(all_results),1)*100:.0f}%")
print(f"  📊C2:{len(c2s)} | 预测:{len(fc)}条 | 耗时:{elapsed/60:.1f}min | 问题:{len(issues)}")

print(f"\n{'─'*60}")
for dim in sorted(set(r['dimension'] for r in all_results)):
    dr=[r for r in all_results if r['dimension']==dim]
    okn=sum(1 for r in dr if r.get('error') is None)
    c2n=sum(1 for r in dr if r.get('pipeline')=='ablate_noc2')
    hm=sum(1 for r in dr if 'horizon_mismatch' in r)
    al=np.mean([r['latency_s'] for r in dr if r.get('error') is None])
    fc_dr=[r for r in dr if 'forecast_metrics' in r]
    smapes=[r['forecast_metrics']['sMAPE'] for r in fc_dr if r['forecast_metrics'].get('sMAPE') is not None]
    smape_str=f" sMAPE={np.mean(smapes):.0f}%" if smapes else ""
    print(f"  {'✅' if okn==len(dr) else '⚠️'} {dim:33s} {okn}/{len(dr)} | {c2n}xC2 | {hm}mis | {al:.0f}s{smape_str}")

if fc:
    print(f"\n📊 预测metric ({len(fc)}条):")
    for mk in ['sMAPE','MASE','MAE']:
        vals=[r['forecast_metrics'][mk] for r in fc if r['forecast_metrics'].get(mk) is not None]
        if vals:print(f"  {mk}: {np.mean(vals):.4f} ± {np.std(vals):.4f} (n={len(vals)})")

if issues:
    cats=Counter()
    for iss in issues:
        if "horizon" in iss.lower():cats["horizon"]+=1
        elif "NO_DATA" in iss:cats["nodata"]+=1
        elif "❌" in iss:cats["error"]+=1
    print(f"\n⚠️  Issues: {dict(cats)}")

with open(os.path.join(OUT,"all_results.jsonl"),'w') as f:
    for r in all_results:f.write(json.dumps(r,ensure_ascii=False)+'\n')
print(f"\n✅ {OUT}")
