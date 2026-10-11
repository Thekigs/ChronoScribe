#!/usr/bin/env python3
"""Build an explicit fixed-1024-on-controller-failure deployment sensitivity.

No inference or invented predictions: every trajectory comes from an already
executed, same-manifest adaptive or fixed-policy row. This is a declared
deployment sensitivity, not the strict adaptive subset and not evidence that
invalid Qwen outputs were valid routing decisions.
"""
from __future__ import annotations
import argparse, json, hashlib
from collections import Counter
from pathlib import Path

def read(root: Path) -> dict:
    rows = {}
    for line in (root / 'all_results.jsonl').read_text().splitlines():
        if not line.strip(): continue
        row = json.loads(line)
        key = (row['dimension'], row['id'])
        if key in rows: raise ValueError(f'duplicate identity: {key}')
        rows[key] = row
    return rows

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--adaptive', required=True, type=Path)
    ap.add_argument('--fixed', required=True, type=Path)
    ap.add_argument('--output', required=True, type=Path)
    a = ap.parse_args()
    adaptive, fixed = read(a.adaptive), read(a.fixed)
    if set(adaptive) != set(fixed): raise ValueError('manifest identities must match exactly')
    configs = [json.loads((p/'run_config.json').read_text()) for p in (a.adaptive,a.fixed)]
    if configs[0].get('manifest_sha256') != configs[1].get('manifest_sha256'):
        raise ValueError('manifest hashes differ')
    if configs[1].get('fixed_context_length') != 1024:
        raise ValueError('fallback must be the declared fixed-1024 control')
    records=[]
    for key in sorted(adaptive):
        raw = adaptive[key]
        failed = raw.get('error') == 'missing_fresh_controller_decision'
        row = dict(fixed[key] if failed else raw)
        row['controller_source'] = 'explicit_fixed1024_fallback' if failed else raw['controller_source']
        row['policy_source'] = 'fixed1024_on_invalid_controller' if failed else 'fresh_adaptive_with_execution_cap'
        row['adaptive_original_error'] = raw.get('error')
        row['deployment_sensitivity'] = True
        row['source_result'] = str((a.fixed if failed else a.adaptive).resolve())
        records.append(row)
    a.output.mkdir(parents=True,exist_ok=True)
    stream = a.output/'all_results.jsonl'
    stream.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
    report={'protocol':'explicit_fixed1024_fallback_deployment_sensitivity_v1',
            'status':'complete','manifest_sha256':configs[0]['manifest_sha256'],
            'adaptive_root':str(a.adaptive.resolve()),'fixed_root':str(a.fixed.resolve()),
            'source_sha256':{str(p/'all_results.jsonl'):hashlib.sha256((p/'all_results.jsonl').read_bytes()).hexdigest() for p in (a.adaptive,a.fixed)},
            'rows':len(records),'policy_counts':dict(Counter(r['policy_source'] for r in records)),
            'forecast_rows':sum(bool(r.get('forecast_metrics')) for r in records),
            'errors':sum(bool(r.get('error')) for r in records),
            'warning':'Explicit deployment sensitivity; fallback rows are not valid Qwen decisions. No new inference or full response generation was performed.'}
    (a.output/'run_config.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
