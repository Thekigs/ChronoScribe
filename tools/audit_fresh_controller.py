#!/usr/bin/env python3
"""Summarize schema, horizon, retry, and execution-bound audits."""
from __future__ import annotations
import argparse,json,os,glob
from pathlib import Path
from collections import Counter
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--root',required=True,type=Path); ap.add_argument('--output',required=True,type=Path); a=ap.parse_args(); rows=[]
 for p in a.root.glob('*/results.json'):
  dim=p.parent.name
  for r in json.loads(p.read_text()): r['_dim']=dim; rows.append(r)
 valid=[r for r in rows if r.get('directive_valid') is True]
 out={'root':str(a.root.resolve()),'rows':len(rows),'valid':len(valid),'invalid':len(rows)-len(valid),'retry_rows':sum(bool(r.get('retry_used')) for r in rows),'budget_exhausted_rows':sum(bool(r.get('generation_audit',{}).get('budget_exhausted')) for r in rows),'eos_rows':sum(bool(r.get('generation_audit',{}).get('ended_with_eos')) for r in rows),'context_lengths_requested':Counter(r.get('step1_params',{}).get('context_length') for r in valid),'contexts_over_chronos2_limit':sum(int(r.get('step1_params',{}).get('context_length',0))>4096 for r in valid),'prediction_length_missing_or_nonpositive':sum(not isinstance(r.get('step1_params',{}).get('prediction_length'),int) or r['step1_params']['prediction_length']<=0 for r in valid),'future_values_provided_to_controller':json.loads((a.root/'run_config.json').read_text()).get('future_values_provided_to_controller'),'status':'valid_decisions_require_no_clamp_filter_for_strict_score'}
 a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n'); print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
