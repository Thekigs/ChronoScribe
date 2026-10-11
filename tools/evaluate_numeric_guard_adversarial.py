#!/usr/bin/env python3
"""Adversarial audit for the numeric provenance guard.

The ordinary smoke test only uses random unsupported values.  These cases
also test swapped fields, percent/unit confusion, and reuse of an evidence
number in an unsupported semantic claim.
"""
from __future__ import annotations
import json
import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.numeric_claim_guard import enforce_numeric_claims

CASES=[
 {"name":"random_novel","response":"最高值为91，风险评分为87。","evidence":"起点=10.00 终点=20.00 最低=8.00 最高=25.00"},
 {"name":"swapped_supported_tokens","response":"起点为25.00，终点为10.00。","evidence":"起点=10.00 终点=25.00"},
 {"name":"same_number_wrong_semantics","response":"风险评分为25.00。","evidence":"最高值=25.00"},
 {"name":"percent_unit_confusion","response":"增长率为25%。","evidence":"最高值=25.00"},
 {"name":"supported_rounding","response":"终点约20.01。","evidence":"终点=20.00"},
]
def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--output', type=Path, default=ROOT/'scores/numeric_claim_guard_adversarial.json')
 args=parser.parse_args()
 out=[]
 for c in CASES:
  typed={"start":10.0,"end":25.0,"max":25.0} if c['name'] in {"swapped_supported_tokens","same_number_wrong_semantics","percent_unit_confusion"} else None
  safe,a=enforce_numeric_claims(c['response'],c['evidence'],typed_evidence=typed)
  out.append({**c,'blocked_count':a.blocked_count,'supported':list(a.supported),'unsupported':list(a.unsupported),'guarded_response':safe})
 result={'cases':out,'novel_token_detection_rate':sum(x['name']=='random_novel' and x['blocked_count']>0 for x in out)/1,'limitation':'The typed layer catches known field swaps and unit-like labels. Claims in arbitrary prose without a recognized field label still receive conservative token-level checking only.'}
 args.output.parent.mkdir(parents=True, exist_ok=True)
 args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
