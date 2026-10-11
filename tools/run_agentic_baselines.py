#!/usr/bin/env python3
"""Run controlled or official LLM-assisted forecasting baselines.

Examples (execution intentionally omitted from repository checks):

  python tools/run_agentic_baselines.py --baseline simple_fusion \
    --specialist chronos2 --device mps --context-length 512 \
    --output-dir results/agentic/simple_fusion_c2_fixed512

  python tools/run_agentic_baselines.py --baseline revision_control \
    --specialist timesfm --device mps --context-length 1024 \
    --output-dir results/agentic/revision_timesfm_fixed1024

The third-party methods use ``--baseline external`` and a command containing
``{request}`` and ``{response}`` placeholders.  Their output is explicitly
labelled official_reproduction or controlled_adaptation in run_config.json.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from agent_baselines.contracts import write_records
from agent_baselines.external_adapter import ExternalAgentAdapter
from agent_baselines.local_models import choose_device
from agent_baselines.local_pipelines import ControlledPipeline
from utils.evaluation_manifest import load_manifest_samples


DEFAULT_DIMS = [
    "D1_short_term_forecasting", "D2_long_horizon", "D3_multivariate",
    "D4_zero_shot", "D5_few_shot", "D9_robustness", "D10_uncertainty",
    "D14_solar", "D17_missing_robustness", "D20_horizon_degradation",
    "D22_outlier_robustness",
]


def read_existing(output_dir: Path) -> dict[tuple[str, str], dict]:
    """Read a resumable checkpoint without accepting duplicate sample keys."""
    stream = output_dir / "all_results.jsonl"
    if not stream.exists():
        return {}
    rows = {}
    for line in stream.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = (str(row.get("dimension")), str(row.get("id")))
        rows[key] = row
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", choices=["simple_fusion", "revision_control", "external"], required=True)
    parser.add_argument("--specialist", choices=["chronos2", "timesfm"], default="chronos2")
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--controller-max-new-tokens", type=int, default=256)
    parser.add_argument("--manifest", default=str(BASE / "benchmark/evaluation_manifest.json"))
    parser.add_argument("--dims", default=",".join(DEFAULT_DIMS))
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume", action="store_true",
                        help="Resume complete rows from output-dir/all_results.jsonl.")
    parser.add_argument("--checkpoint-every", type=int, default=10,
                        help="Persist after this many newly completed rows (default: 10).")
    parser.add_argument("--external-name", default="external_agent")
    parser.add_argument("--external-status", choices=["official_reproduction", "controlled_adaptation", "qualitative_only"], default="controlled_adaptation")
    parser.add_argument("--external-command", default="", help="Shell-like argv with {request} and {response} placeholders")
    args = parser.parse_args()

    dims = [item.strip() for item in args.dims.split(",") if item.strip()]
    plan = load_manifest_samples(BASE / "benchmark", args.manifest, dims)
    output_dir = Path(args.output_dir)
    existing = read_existing(output_dir) if args.resume else {}
    records_by_key = dict(existing)
    done = {
        key for key, row in existing.items()
        if row.get("forecast_metrics") or row.get("skip_reason")
    }
    newly_completed = 0

    def checkpoint(config: dict) -> None:
        checkpoint_config = dict(config)
        checkpoint_config["sample_count"] = len(records_by_key)
        checkpoint_config["completed_rows"] = len(records_by_key)
        checkpoint_config["thinking_enabled"] = True
        write_records(output_dir, list(records_by_key.values()), checkpoint_config)

    if args.baseline == "external":
        if not args.external_command:
            raise SystemExit("--external-command is required for --baseline external")
        adapter = ExternalAgentAdapter(
            args.external_name, shlex.split(args.external_command), args.external_status
        )
        work_dir = Path(args.output_dir) / "external_io"
        for dimension, samples in sorted(plan.items()):
            for sample in samples[:args.max_samples or None]:
                key = (dimension, str(sample["id"]))
                if key in done:
                    continue
                row = adapter.run_one(sample, dimension, work_dir)
                records_by_key[key] = row
                newly_completed += 1
                print(f"completed {newly_completed} / {sum(len(v[:args.max_samples or None]) for v in plan.values())}: {dimension}/{sample['id']}", flush=True)
                if newly_completed % max(1, args.checkpoint_every) == 0:
                    checkpoint(vars(args) | {"execution": "external_command", "checkpoint": True})
        config = vars(args) | {"protocol_status": args.external_status, "execution": "external_command"}
    else:
        device = choose_device(args.device)
        pipeline = ControlledPipeline(
            args.baseline, args.specialist, device, args.context_length,
            controller_tokens=args.controller_max_new_tokens,
        )
        for dimension, samples in sorted(plan.items()):
            for sample in samples[:args.max_samples or None]:
                key = (dimension, str(sample["id"]))
                if key in done:
                    continue
                row = pipeline.run_one(sample, dimension)
                records_by_key[key] = row
                newly_completed += 1
                print(f"completed {newly_completed} / {sum(len(v[:args.max_samples or None]) for v in plan.values())}: {dimension}/{sample['id']}", flush=True)
                if newly_completed % max(1, args.checkpoint_every) == 0:
                    checkpoint(vars(args) | {"device": device, "execution": "local_qwen_tsfm", "checkpoint": True,
                                             "thinking_enabled": True})
        config = vars(args) | {"device": device, "execution": "local_qwen_tsfm"}
    config.update({
        "manifest": str(Path(args.manifest).resolve()),
        "dimensions": dims,
        "sample_count": len(records_by_key),
        "future_values_used_for_execution": False,
        "thinking_enabled": True,
        "resume": args.resume,
        "checkpoint_every": args.checkpoint_every,
    })
    write_records(output_dir, list(records_by_key.values()), config)
    print(f"Wrote {len(records_by_key)} records to {output_dir}")


if __name__ == "__main__":
    main()
