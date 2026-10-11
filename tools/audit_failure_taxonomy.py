#!/usr/bin/env python3
"""Summarize strict-run failures and provenance outcomes by dimension."""
from __future__ import annotations
import argparse,json
from collections import Counter,defaultdict
from pathlib import Path

def bucket(row):
    if row.get('error'):
        e=str(row['error'])
        if 'missing_controller' in e: return 'controller_missing_match'
        if 'no_usable_history' in e: return 'no_usable_history'
        return 'runner_error'
    if row.get('forecast_evaluation_status')=='unscorable_no_manifest_target': return 'no_manifest_target'
    if not row.get('forecast_metrics'): return 'no_forecast_metrics'
    if row.get('horizon_mismatch') or row.get('horizon_corrected'): return 'horizon_issue'
    return 'success_scored'

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--results',type=Path,required=True); ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args(); rows=[json.loads(x) for x in (a.results/'all_results.jsonl').read_text().splitlines() if x.strip()]
    out={'results':str(a.results.resolve()),'rows':len(rows),'overall':dict(Counter(bucket(r) for r in rows)),'by_dimension':{},'by_context':{}}
    dims=defaultdict(list); ctx=defaultdict(list)
    for r in rows:
        dims[str(r.get('dimension'))].append(r); ctx[str((r.get('c2_params') or {}).get('ctx_len','none'))].append(r)
    for k,rr in sorted(dims.items()):
        out['by_dimension'][k]={'n':len(rr),'outcomes':dict(Counter(bucket(r) for r in rr)),
            'controller_sources':dict(Counter(r.get('controller_source','none') for r in rr)),
            'guard_blocked_rows':sum((r.get('numeric_claim_audit') or {}).get('blocked_count',0)>0 for r in rr),
            'forecast_metric_rows':sum(bool(r.get('forecast_metrics')) for r in rr)}
    for k,rr in sorted(ctx.items(),key=lambda kv:(kv[0]=='none',kv[0])):
        out['by_context'][k]={'n':len(rr),'outcomes':dict(Counter(bucket(r) for r in rr)),
                              'guard_blocked_rows':sum((r.get('numeric_claim_audit') or {}).get('blocked_count',0)>0 for r in rr)}
    out['interpretation']='This is an audit of execution/provenance outcomes, not an accuracy score. Missing controller matches and unavailable manifest targets remain excluded from forecasting comparisons.'
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n'); print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
