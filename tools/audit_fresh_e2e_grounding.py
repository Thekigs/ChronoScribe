#!/usr/bin/env python3
"""Audit raw versus guarded numerical response claims on one run."""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.numeric_claim_guard import audit_numeric_claims, audit_typed_numeric_claims

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--results',type=Path,required=True); ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args(); rows=[json.loads(x) for x in (a.results/'all_results.jsonl').read_text().splitlines() if x.strip()]
    out={'results':str(a.results.resolve()),'rows':len(rows),'valid_rows':0,'raw_claims':0,'raw_unsupported':0,
         'guarded_unsupported':0,'typed_unsupported':0,'blocked_rows':0,'post_guard_unresolved':0,'examples':[]}
    for r in rows:
        if r.get('error'): continue
        out['valid_rows']+=1
        raw=str(r.get('analysis_raw','')); guarded=str(r.get('analysis',''))
        evidence=' '.join(f'{k}:{v}' for k,v in (r.get('typed_numeric_evidence') or {}).items())
        raw_a=audit_numeric_claims(raw,evidence); guarded_a=audit_numeric_claims(guarded,evidence)
        typed=audit_typed_numeric_claims(guarded,r.get('typed_numeric_evidence') or {})
        out['raw_claims']+=len(raw_a.supported)+len(raw_a.unsupported)
        out['raw_unsupported']+=len(raw_a.unsupported)
        out['guarded_unsupported']+=len(guarded_a.unsupported)
        out['typed_unsupported']+=len(typed)
        if (r.get('numeric_claim_audit') or {}).get('blocked_count',0): out['blocked_rows']+=1
        unresolved=len(guarded_a.unsupported)+len(typed)
        out['post_guard_unresolved']+=unresolved
        if unresolved and len(out['examples'])<10:
            out['examples'].append({'id':r.get('id'),'guarded':guarded,'unresolved':unresolved})
    out['raw_unsupported_rate']=out['raw_unsupported']/max(out['raw_claims'],1)
    out['post_guard_unresolved_rate']=out['post_guard_unresolved']/max(out['raw_claims'],1)
    out['interpretation']='A zero post_guard_unresolved count means every extracted numeric token in the guarded response is supported by the typed evidence under the same tolerance; this does not prove nonnumeric semantic claims are grounded.'
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n'); print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
