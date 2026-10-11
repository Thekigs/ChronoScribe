#!/usr/bin/env python3
"""Bundle the fresh thinking adaptive comparison against fixed controls."""
from __future__ import annotations
import argparse, json
from pathlib import Path

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adaptive", required=True, type=Path)
    ap.add_argument("--comparison", action="append", nargs=2, metavar=("NAME", "JSON"), required=True)
    ap.add_argument("--output", required=True, type=Path)
    a = ap.parse_args()
    comparisons = {name: json.loads(Path(path).read_text()) for name, path in a.comparison}
    out = {
        "protocol": "fresh_thinking_adaptive_fixed_frontier_v1",
        "adaptive": str(a.adaptive.resolve()),
        "comparisons": comparisons,
        "conclusion": "Under identical fresh-controller eligibility and no-clamp filtering, adaptive routing does not show a universal advantage over fixed-512, fixed-1024, or full-history controls. Fixed-policy choice remains metric- and dimension-dependent.",
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"comparisons": list(comparisons), "output": str(a.output)}, ensure_ascii=False, indent=2))

if __name__ == "__main__": main()
