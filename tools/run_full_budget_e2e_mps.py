#!/usr/bin/env python3
"""Serialize full-budget MPS evaluations and preserve generation diagnostics."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

BASE = Path(__file__).resolve().parents[1]
RUNS = [
    ('simple_fixed512', 'simple_fixed', 512),
    ('simple_fixed1024', 'simple_fixed', 1024),
    ('adaptive', 'adaptive', None),
]
STATE = BASE / 'results/e2e_full_budget_mps_status.json'


def save(state):
    state['updated_at'] = datetime.now(timezone.utc).isoformat()
    temporary = STATE.with_suffix('.tmp')
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(STATE)


def command(argv, logfile):
    print('Running:', ' '.join(map(str, argv)), flush=True)
    with logfile.open('w') as handle:
        subprocess.run(list(map(str, argv)), cwd=BASE, stdout=handle,
                       stderr=subprocess.STDOUT, check=True)


def audit(root):
    rows = [json.loads(line) for line in (root / 'all_results.jsonl').read_text().splitlines()]
    if len(rows) != 460 or any(r.get('device') != 'mps' or
        r.get('generation_budget') != {'controller': 256, 'response': 384} for r in rows):
        raise RuntimeError(f'Incomplete or incompatible full-budget run: {root}')
    live = [r for r in rows if r.get('controller_source') == 'live_qwen']
    cache_miss = [r for r in rows if r.get('controller_source') == 'cache_miss_fallback']
    calls = [c for r in rows for c in r.get('generation_audit', [])]
    return {
        'rows': len(rows), 'errors': sum(r.get('error') is not None for r in rows),
        'forecast_rows': sum('forecast_metrics' in r for r in rows),
        'live_controller_rows': len(live),
        'live_controller_parse_failures': sum(not r.get('controller_parse_ok') for r in live),
        'cache_miss_fallback_rows': len(cache_miss),
        'generation_calls': len(calls),
        'budget_exhausted_calls': sum(c['budget_exhausted'] for c in calls),
        'budget_exhausted_sample_ids': [r['id'] for r in rows
            if any(c['budget_exhausted'] for c in r.get('generation_audit', []))],
        'interpretation': 'Full configured budgets do not imply all outputs finish. '
            'Controller parse failures and cache-miss fallbacks are not valid adaptive decisions.',
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wait-fixed512-pid', type=int)
    args = parser.parse_args()
    STATE.parent.mkdir(parents=True, exist_ok=True)
    logs = BASE / 'results/e2e_full_budget_mps_logs'
    logs.mkdir(exist_ok=True)
    state = {'status': 'running', 'pid': os.getpid(), 'runs': {},
             'controller_budget': 256, 'response_budget': 384, 'controller_cache': False,
             'source_sha256': {name: hashlib.sha256((BASE / name).read_bytes()).hexdigest()
                for name in ['runners/run_chronoscribe.py', 'benchmark/evaluation_manifest.json']}}
    save(state)
    try:
        if args.wait_fixed512_pid:
            state['phase'] = 'waiting_for_existing_fixed512'
            state['existing_fixed512_pid'] = args.wait_fixed512_pid
            save(state)
            while True:
                try:
                    os.kill(args.wait_fixed512_pid, 0)
                except ProcessLookupError:
                    break
                time.sleep(20)
        for name, policy, context in RUNS:
            root = BASE / f'results/e2e_full_{name}_mps'
            score = BASE / f'scores/e2e_full_{name}_mps'
            state['phase'] = name
            save(state)
            if (root / 'all_results.jsonl').exists() and (score / 'final_scores.json').exists():
                report = audit(root)
                state['runs'][name] = {**report, 'scored': True, 'reused_existing': True}
                save(state)
                continue
            if not (name == 'simple_fixed512' and args.wait_fixed512_pid):
                if (root / 'all_results.jsonl').exists():
                    raise RuntimeError(f'Refusing to overwrite completed run {root}')
                argv = [sys.executable, '-u', 'runners/run_chronoscribe.py',
                        '--policy', policy, '--output-dir', root, '--require-mps',
                        '--controller-max-new-tokens', '256', '--response-max-new-tokens', '384']
                if context:
                    argv.extend(['--fixed-context-length', str(context)])
                if policy == 'adaptive':
                    # The live controller was audited separately and found to
                    # emit malformed natural-language output at full budget.
                    # Reuse the prior valid Qwen decisions so this end-to-end
                    # run measures the full response budget without silently
                    # treating fallback as adaptive routing.
                    argv.extend(['--reuse-controller-dir', 'results/qwenchronos'])
                    state['controller_cache'] = True
                    save(state)
                command(argv, logs / f'{name}.log')
            report = audit(root)
            state['runs'][name] = report
            save(state)
            score.mkdir(parents=True, exist_ok=True)
            (score / 'generation_audit.json').write_text(json.dumps(report, indent=2) + '\n')
            command([sys.executable, 'scoring/score_results.py', '--results-dir', root,
                     '--benchmark-dir', 'benchmark', '--output-dir', score,
                     '--model-name', f'qwen_e2e_full_{name}_mps'], logs / f'{name}_score.log')
            state['runs'][name]['scored'] = True
            save(state)
        state['phase'] = 'paired_forecast_comparison'
        save(state)
        command([sys.executable, 'tools/analyze_paired_forecasts.py',
                 '--baseline', BASE / 'results/e2e_full_adaptive_mps', '--candidates',
                 BASE / 'results/e2e_full_simple_fixed512_mps',
                 BASE / 'results/e2e_full_simple_fixed1024_mps',
                 '--output', BASE / 'scores/paired_full_e2e_adaptive_vs_fixed.json'],
                logs / 'paired_forecasts.log')
        state['status'] = 'complete'
        save(state)
    except Exception as exc:
        state['status'] = 'failed'
        state['error'] = repr(exc)
        save(state)
        raise


if __name__ == '__main__':
    main()
