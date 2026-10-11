#!/usr/bin/env python3
"""Audit adaptive-controller provenance before paired scoring.

This intentionally does not score anything.  It separates schema-valid fresh
decisions from legacy cached directives and empty records that the runner
would otherwise turn into its default policy.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from collections import Counter

def valid(v):
    if not isinstance(v, dict) or type(v.get("use_chronos2")) is not bool: return False
    if v["use_chronos2"] is False: return isinstance(v.get("direct_reply"), str) and bool(v["direct_reply"].strip())
    return all(type(v.get(k)) is int and v[k] > 0 for k in ("prediction_length", "context_length"))

def main():
    p=argparse.ArgumentParser(); p.add_argument("--results",required=True,type=Path); p.add_argument("--output",required=True,type=Path)
    p.add_argument("--assume-strict-cache",action="store_true",help="Treat cached_qwen rows as fresh only when the caller has separately verified strict cache matching.")
    a=p.parse_args()
    rows=[json.loads(x) for x in (a.results/"all_results.jsonl").read_text().splitlines() if x.strip()]
    cats=Counter(); by_source=Counter(); by_dim=Counter(); eligible=[]
    for r in rows:
        d=r.get("controller_output")
        if d is None:
            d=r.get("controller_raw_output")
            if isinstance(d,str):
                try: d=json.loads(d)
                except Exception: d=None
        schema=valid(d); source=r.get("controller_source")
        if not schema: cat="empty_or_invalid"
        elif (r.get("history_sha256") and r.get("controller_source") in ("fresh_manifest_qwen","live")) or (a.assume_strict_cache and source=="cached_qwen"): cat="fresh_provenance_valid"
        else: cat="legacy_or_unverifiable_valid"
        cats[cat]+=1; by_source[str(source)]+=1
        if cat == "fresh_provenance_valid" and r.get("forecast_metrics") and not r.get("error"):
            eligible.append(r); by_dim[str(r.get("dimension"))]+=1
    out={"results":str(a.results.resolve()),"rows":len(rows),"categories":cats,"controller_sources":by_source,"fresh_valid_forecast_rows":len(eligible),"fresh_valid_forecast_by_dimension":by_dim,
         "publication_status":("strict_cache_verified_by_runner_and_forecast_rows_scored" if a.assume_strict_cache else "not_strict_until_fresh_valid_forecast_rows_are_scored_against_the_same_fixed_manifest")}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n"); print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
