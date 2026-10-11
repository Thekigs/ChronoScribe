#!/usr/bin/env python3
"""Small diagnostic probe, never a publication benchmark result."""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent_baselines.local_models import QwenController, LocalSpecialist, choose_device, device_memory
from agent_baselines.local_pipelines import _forecast_prompt
from agent_baselines.contracts import extract_json
from utils.benchmark_data import load_raw, build_context
from utils.controller_inputs import history_summary
from utils.evaluation_manifest import load_manifest_samples


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--modes", default="legacy,corrected")
    args = ap.parse_args()
    modes = args.modes.split(",")
    if set(modes) - {"legacy", "corrected"}:
        raise ValueError("unknown probe mode")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    def log(row):
        with (out / "events.jsonl").open("a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(json.dumps(row, ensure_ascii=False), flush=True)
    device = choose_device("mps")
    plan = load_manifest_samples(ROOT / "benchmark", ROOT / "benchmark/evaluation_manifest.json",
                                 ["D1_short_term_forecasting"])
    sample = plan["D1_short_term_forecasting"][0]
    target = sample["ground_truth"]["values"]
    t = time.perf_counter()
    data, source = load_raw(sample)
    ctx, alignment = build_context(data, sample, target, qwen_ctx_len=1024)
    log({"stage": "data", "elapsed_s": time.perf_counter()-t, "id": sample["id"],
         "source": source, "alignment": alignment, "protocol_status": "diagnostic_only"})
    t = time.perf_counter()
    controller = QwenController(device, args.max_new_tokens)
    log({"stage": "load_qwen", "elapsed_s": time.perf_counter()-t, "memory": device_memory(device)})
    t = time.perf_counter()
    specialist = LocalSpecialist("chronos2", device)
    log({"stage": "load_specialist", "elapsed_s": time.perf_counter()-t, "memory": device_memory(device)})
    messages = _forecast_prompt(sample, history_summary(ctx), len(target), "chronos2")
    for mode in modes:
        controller.enable_thinking = mode == "legacy"
        controller.release_mps_cache = mode != "legacy"
        log({"stage": "start_generation", "mode": mode, "memory": device_memory(device)})
        text, audit = controller.generate(messages)
        log({"stage": "generation", "mode": mode, "audit": audit,
             "raw_output": text, "parsed": extract_json(text)})
        t = time.perf_counter()
        _, values, _ = specialist.forecast(ctx, len(target))
        log({"stage": "specialist", "mode": mode, "elapsed_s": time.perf_counter()-t,
             "memory": device_memory(device), "horizon": len(values), "prediction_preview": values[:5]})
        del values
        import torch
        torch.mps.empty_cache()
        log({"stage": "cache_released", "mode": mode, "memory": device_memory(device)})
    log({"stage": "complete", "device": device})


if __name__ == "__main__":
    main()
