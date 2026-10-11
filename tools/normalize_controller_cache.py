#!/usr/bin/env python3
"""Create an auditable controller-cache normalization view.

Only complete JSON wrappers/aliases are normalized. No missing value is
filled, and every changed row is recorded in the audit file.
"""
from __future__ import annotations
import argparse,json,shutil
from pathlib import Path

def norm(v):
    if not isinstance(v,dict): return v,False
    c=v.get('json') if isinstance(v.get('json'),dict) else v
    changed=isinstance(v.get('json'),dict)
    for alias in ('use_chrono2','use_chronos'):
        if 'use_chronos2' not in c and alias in c:
            c=dict(c); c['use_chronos2']=c.pop(alias); changed=True; break
    return c,changed
def valid(v):
    if not isinstance(v,dict) or type(v.get('use_chronos2')) is not bool:return False
    if not v['use_chronos2']:return isinstance(v.get('direct_reply'),str) and bool(v['direct_reply'].strip())
    return all(type(v.get(k)) is int and v[k]>0 for k in ('prediction_length','context_length'))
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--source',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args()
    changed=[]; before=after=0
    for f in a.source.glob('*/results.json'):
        rows=json.loads(f.read_text()); new=[]
        for r in rows:
            before+=int(valid(r.get('step1_params')))
            d,c=norm(r.get('step1_params'))
            if c: changed.append({'dimension':f.parent.name,'id':r.get('id'),'old':r.get('step1_params'),'new':d})
            q=dict(r); q['step1_params']=d; q['directive_normalization_applied']=c
            after+=int(valid(d)); new.append(q)
        dest=a.output/f.name if False else a.output/f.parent.name/'results.json'
        dest.parent.mkdir(parents=True,exist_ok=True); dest.write_text(json.dumps(new,ensure_ascii=False,indent=2)+'\n')
    a.output.mkdir(parents=True,exist_ok=True)
    cfg={'source':str(a.source.resolve()),'output':str(a.output.resolve()),'normalization':'json_unwrap_or_use_chrono2_alias_only','before_valid':before,'after_valid':after,'changed_rows':changed,'future_values_provided_to_controller':False}
    (a.output/'normalization_audit.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2)+'\n'); print(json.dumps({'before_valid':before,'after_valid':after,'changed_rows':len(changed)},ensure_ascii=False))
if __name__=='__main__':main()
