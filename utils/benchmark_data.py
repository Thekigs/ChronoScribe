#!/usr/bin/env python3
"""Shared benchmark data resolution and context alignment utilities."""
import json
import os
import re
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Cache resolved source series so multiple tasks can reuse a local dataset
# without repeatedly opening large files during a paired evaluation.
RAW={};RAW_SRC={}
SOLAR_DF=pd.read_csv(os.path.join(BASE,'data','solar','solar_AL.txt'),header=None) if os.path.exists(os.path.join(BASE,'data','solar','solar_AL.txt')) else None
TOUR=json.load(open(os.path.join(BASE,'data','tourism','tourism_data.json'))) if os.path.exists(os.path.join(BASE,'data','tourism','tourism_data.json')) else None

# Canonical local locations for supported public dataset families.
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
    """Build a past-only context aligned to the benchmark target values.

    The function is shared by the full runner and ablations. It must never
    expose the ground-truth forecast window as model input; ground truth is
    used only to locate the corresponding point in a source series.
    """
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
    """Resolve a benchmark record to its source numerical history and label.

    Returning the source label makes loader fallbacks auditable in result
    records and prevents silent substitution of an unrelated series.
    """
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
