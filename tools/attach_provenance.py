#!/usr/bin/env python3
"""Write immutable provenance metadata for a paired result directory."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
def main():
    p=argparse.ArgumentParser(); p.add_argument('--result',required=True,type=Path); p.add_argument('--controller',required=True,type=Path); p.add_argument('--manifest',required=True,type=Path); a=p.parse_args()
    cfg=json.loads((a.controller/'run_config.json').read_text())
    rows=sum(len(json.loads(x.read_text())) for x in a.controller.glob('*/results.json'))
    valid=sum(sum(bool(r.get('directive_valid')) for r in json.loads(x.read_text())) for x in a.controller.glob('*/results.json'))
    out={'result_dir':str(a.result.resolve()),'controller_dir':str(a.controller.resolve()),'manifest':str(a.manifest.resolve()),'manifest_sha256':sha(a.manifest),'controller_run_config_sha256':sha(a.controller/'run_config.json'),'controller_rows':rows,'controller_valid_decisions':valid,'future_values_provided_to_controller':cfg.get('future_values_provided_to_controller'),'protocol':'fresh_manifest_qwen_controller_then_chronos2_strict_match','fallback_allowed':False}
    (a.result/'provenance.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n'); print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
