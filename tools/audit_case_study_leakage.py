#!/usr/bin/env python3
"""Audit whether a case-study evidence card contains future-dependent fields."""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]


def main():
 parser = argparse.ArgumentParser(description=__doc__)
 parser.add_argument(
  "--source",
  type=Path,
  default=BASE / "docs/AuthorKit27/Figures/source/fig_case_study.py",
  help="Path to the separately supplied case-study source file.",
 )
 parser.add_argument(
  "--output",
  type=Path,
  default=BASE / "scores/case_study_leakage_audit.json",
 )
 args = parser.parse_args()
 p = args.source
 if not p.exists():
  out = {
   "source": str(p),
   "evidence_fields": [],
   "future_dependent_fields_detected": [],
   "status": "source_unavailable_in_source_only_release",
   "note": "The case-study source is excluded from this source-only release. Supply it with --source to run the audit.",
  }
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
  print(json.dumps(out, ensure_ascii=False, indent=2))
  return
 text = p.read_text(encoding="utf-8")
 future_fields=[]
 m=re.search(r'EVIDENCE\s*=\s*\[(.*?)\]\n\n# ⑤',text,re.S)
 if m:
  future_fields=re.findall(r'\("([^"]+)"\s*,',m.group(1))
 forbidden={'MAE','sMAPE(%)','MASE','Dir_Acc(%)','RMSE','MAPE','WAPE','OWA'}
 found=sorted(set(future_fields)&forbidden)
 out={'source':str(p.resolve()),'evidence_fields':future_fields,'future_dependent_fields_detected':found,'status':'leakage_present_requires_case_study_revision' if found else 'no_known_future_metric_fields','note':'Forecast-derived endpoint/trend fields are permissible only if computed from forecast; ground-truth metrics must not be injected before response generation.'}
 args.output.parent.mkdir(parents=True, exist_ok=True)
 args.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
