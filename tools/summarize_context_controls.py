#!/usr/bin/env python3
"""Create a compact auditable table for the five fixed-context controls."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--scores-root', default='scores')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(args.scores_root)
    rows = []
    for context in (64, 128, 256, 512, 1024):
        path = root / f'context_fixed_{context}' / 'final_scores.json'
        doc = json.loads(path.read_text())
        dims = doc.get('dimensions', {})
        raw = {}
        for dim, item in dims.items():
            for group in ('primary_metrics', 'related_metrics'):
                for metric, stat in item.get(group, {}).items():
                    if isinstance(stat, dict) and isinstance(stat.get('mean'), (int, float)):
                        raw.setdefault(metric, []).append(float(stat['mean']))
        rows.append({
            'context_length': context,
            'tls': doc.get('tls', {}).get('TLS'),
            'F': doc.get('tls', {}).get('components', {}).get('F'),
            'U': doc.get('tls', {}).get('components', {}).get('U'),
            'forecast_dimensions': len(dims),
            'successful_samples': sum(int(v.get('n_success', 0)) for v in dims.values()),
            'raw_metric_means_across_dimension_summaries': {
                metric: sum(values) / len(values) for metric, values in raw.items()
                if values and metric in {'sMAPE', 'MASE', 'RMSE', 'MAE', 'WAPE', 'Direction_Accuracy'}
            },
            'source': str(path.resolve()),
        })
    out = {'controls': rows, 'best_fixed_by_F': max(rows, key=lambda x: x['F']),
           'note': 'Raw means are descriptive averages over per-dimension summaries; use per-sample files for statistical tests.'}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
