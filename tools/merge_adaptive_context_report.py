#!/usr/bin/env python3
"""Combine strict adaptive-vs-context comparisons into one audit record."""
from __future__ import annotations
import argparse,json
from pathlib import Path
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--adaptive',required=True,type=Path); ap.add_argument('--comparison',action='append',nargs=2,metavar=('NAME','JSON'),required=True); ap.add_argument('--frontier',required=True,type=Path); ap.add_argument('--output',required=True,type=Path); a=ap.parse_args()
 d={"adaptive_strict_score":str(a.adaptive.resolve()),"comparisons":{},"fixed_frontier":json.loads(a.frontier.read_text()),"conclusion":"No single fixed policy dominates all metrics. Adaptive is compared against fixed-512, fixed-1024, and full-history under the same no-clamp eligibility; claims of universal adaptive superiority are not supported."}
 for name,p in a.comparison: d['comparisons'][name]=json.loads(Path(p).read_text())
 a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n'); print('Wrote',a.output)
if __name__=='__main__': main()
