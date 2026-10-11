#!/usr/bin/env python3
"""Audit the controlled simple LLM+TSFM comparison.

This is a provenance/equivalence check, not a new ranking metric: it records
whether the fixed-policy and parser-policy runs use the same TSFM, context,
guard, and numerical trajectory on common manifest rows.
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np

def rows(p):
    return {(str(r.get('dimension')),str(r.get('id'))):r for r in (json.loads(x) for x in (p/'all_results.jsonl').read_text().splitlines() if x.strip())}
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--simple',required=True,type=Path); ap.add_argument('--parser',required=True,type=Path); ap.add_argument('--output',required=True,type=Path); a=ap.parse_args()
    s,p=rows(a.simple),rows(a.parser); common=sorted(set(s)&set(p)); diffs=[]; fields=('tsfm','fixed_context_length','numeric_claim_guard')
    for k in common:
        x,y=s[k],p[k]
        if x.get('error') or y.get('error'): continue
        xv=np.asarray(x.get('predicted_values',[]),dtype=float); yv=np.asarray(y.get('predicted_values',[]),dtype=float)
        if len(xv)!=len(yv): diffs.append({'key':k,'reason':'length_mismatch'}); continue
        d=float(np.max(np.abs(xv-yv))) if len(xv) else float('nan')
        if not math.isfinite(d) or d>1e-6: diffs.append({'key':k,'reason':'trajectory_difference','max_abs_diff':d})
    out={'simple':str(a.simple.resolve()),'parser':str(a.parser.resolve()),'common_rows':len(common),'trajectory_identical_rows':len(common)-len(diffs),'trajectory_differences':diffs[:20],'trajectory_difference_count':len(diffs),'simple_language_interface':sorted({str(r.get('language_interface')) for r in s.values()}),'parser_language_interface':sorted({str(r.get('language_interface')) for r in p.values()}),'shared_execution_fields':{f:len({(s[k].get(f),p[k].get(f)) for k in common if k in s and k in p}) for f in fields},'interpretation':'The fixed-policy run is a numerical/control baseline. Identical trajectories isolate the effect of the controller/parser and response interface; they do not prove adaptive routing is superior.'}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n"); print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
