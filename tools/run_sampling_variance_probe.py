#!/usr/bin/env python3
"""Estimate controller and forecast variance under stochastic decoding.

This is a diagnostic probe, not a replacement for the greedy production
protocol.  The controller sees the request, horizon and past-only history
summary; manifest targets are used only after generation for scoring.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.benchmark_data import build_context, load_raw
from utils.controller_inputs import history_summary, request_text
from utils.evaluation_manifest import load_manifest_samples


def extract_json(text: str) -> dict:
    if not text:
        return {}
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else {}
    except Exception:
        pass
    for block in re.findall(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL):
        try:
            value = json.loads(block)
            if isinstance(value, dict):
                return value
        except Exception:
            continue
    start = text.find("{")
    while start >= 0:
        depth = 0
        for end in range(start, len(text)):
            if text[end] == "{":
                depth += 1
            elif text[end] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        value = json.loads(text[start : end + 1])
                        if isinstance(value, dict):
                            return value
                    except Exception:
                        break
        start = text.find("{", start + 1)
    return {}


def normalize(value: dict) -> dict:
    if not isinstance(value, dict):
        return {}
    candidate = value.get("json") if isinstance(value.get("json"), dict) else value
    for alias in ("use_chrono2", "use_chronos"):
        if "use_chronos2" not in candidate and alias in candidate:
            candidate = dict(candidate)
            candidate["use_chronos2"] = candidate.pop(alias)
            break
    return candidate


def valid(value: dict) -> bool:
    return (
        isinstance(value, dict)
        and type(value.get("use_chronos2")) is bool
        and value.get("use_chronos2") is True
        and type(value.get("prediction_length")) is int
        and value["prediction_length"] > 0
        and type(value.get("context_length")) is int
        and value["context_length"] > 0
    )


def greedy_decisions(root: Path) -> dict[tuple[str, str], dict]:
    decisions = {}
    for path in sorted(root.glob("*/results.json")):
        try:
            rows = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for row in rows:
            directive = normalize(row.get("step1_params"))
            if valid(directive):
                decisions[(path.parent.name, str(row.get("id")))] = directive
    return decisions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-count", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    thinking_group = parser.add_mutually_exclusive_group()
    thinking_group.add_argument("--enable-thinking", dest="enable_thinking", action="store_true",
                                help="Use the formal Qwen thinking template for stochastic diagnostics.")
    thinking_group.add_argument("--no-thinking", dest="enable_thinking", action="store_false",
                                help="Use the compact non-thinking template for a separate diagnostic.")
    parser.set_defaults(enable_thinking=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/sampling_variance_probe_20261005")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    import torch
    from chronos import Chronos2Pipeline
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from scoring.detailed_scoring import compute_all_forecast_metrics

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    if device != "mps":
        raise RuntimeError(f"MPS is required for this probe; selected device={device}")

    dims = ["D1_short_term_forecasting", "D2_long_horizon", "D3_multivariate", "D4_zero_shot", "D5_few_shot", "D9_robustness", "D10_uncertainty", "D14_solar", "D20_horizon_degradation", "D22_outlier_robustness"]
    plan = load_manifest_samples(ROOT / "benchmark", ROOT / "benchmark/evaluation_manifest.json", dims)
    selected = []
    for dimension in dims:
        for sample in plan.get(dimension, []):
            values = sample.get("ground_truth", {}).get("values")
            loaded = load_raw(sample)
            if isinstance(values, list) and values and loaded is not None and loaded[0] is not None:
                selected.append((dimension, sample))
            if len(selected) >= args.sample_count:
                break
        if len(selected) >= args.sample_count:
            break
    if len(selected) < args.sample_count:
        raise RuntimeError(f"only {len(selected)} eligible samples found")

    base_model = AutoModelForCausalLM.from_pretrained(str(ROOT / "models/Qwen3.5-9B"), torch_dtype=torch.float16, device_map="cpu")
    qwen = PeftModel.from_pretrained(base_model, str(ROOT / "models/checkpoint-7500")).to(device)
    qwen.eval()
    tokenizer = AutoTokenizer.from_pretrained(str(ROOT / "models/Qwen3.5-9B"))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    specialist = Chronos2Pipeline.from_pretrained(ROOT / "models/chronos-2")
    greedy = greedy_decisions(ROOT / "results/controller_fresh_full_strict_normalized_20261003")
    system = (
        "你是 ChronoScribe 的 controller，不是 forecaster。只输出一个 JSON 对象，"
        "格式严格为 {\"use_chronos2\":true,\"prediction_length\":整数,\"context_length\":整数}。"
        "不要输出 predicted_values、forecast、analysis、解释或思考过程。"
    )

    records = []
    if args.resume and (args.output_dir / "all_results.jsonl").exists():
        records = [json.loads(line) for line in (args.output_dir / "all_results.jsonl").read_text().splitlines() if line.strip()]
    completed_ids = {
        (r["dimension"], r["id"])
        for r in records
        if sum(1 for x in records if x["dimension"] == r["dimension"] and x["id"] == r["id"]) >= args.repeats
    }
    started = time.time()
    for dimension, sample in selected:
        sid = str(sample["id"])
        if args.resume and (dimension, sid) in completed_ids:
            print(f"resuming: skip completed {dimension}/{sid}")
            continue
        target = np.asarray(sample["ground_truth"]["values"], dtype=float)
        data, source = load_raw(sample)
        context, alignment = build_context(data, sample, target.tolist(), qwen_ctx_len=min(4096, len(data)))
        prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": (
                    f"user request:\n{request_text(sample)}\n\n"
                    f"历史摘要:\n{history_summary(context)}\n\n"
                    f"本评测要求另一个模型预测 {len(target)} 步。现在只输出三个字段的 JSON。"
                )},
            ],
            tokenize=False,
            add_generation_prompt=True,
                    enable_thinking=args.enable_thinking,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        for repeat in range(args.repeats):
            seed = 20261005 + repeat * 1009
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            error = None
            raw = ""
            directive = {}
            try:
                with torch.no_grad():
                    output = qwen.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=True,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                generated = output[0][inputs["input_ids"].shape[1] :]
                raw = tokenizer.decode(generated, skip_special_tokens=True)
                directive = normalize(extract_json(raw))
                del output, generated
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            record = {
                "dimension": dimension,
                "id": sid,
                "repeat": repeat,
                "seed": seed,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "source": source,
                "history_sha256": hashlib.sha256(np.asarray(context, dtype="<f8").tobytes()).hexdigest(),
                "context_alignment": alignment,
                "controller_receives_future_values": False,
                "raw_output": raw,
                "directive": directive,
                "directive_valid": valid(directive),
            }
            if error:
                record["error"] = error
            if valid(directive):
                try:
                    forecast_len = len(target)
                    tensor = torch.tensor(context[-directive["context_length"] :], dtype=torch.float32).reshape(1, 1, -1)
                    quantiles, median = specialist.predict_quantiles(tensor, prediction_length=forecast_len)
                    if isinstance(quantiles, list) and quantiles:
                        q = quantiles[0]
                        prediction = q[0, :, q.shape[2] // 2].detach().cpu().numpy().astype(float)
                    else:
                        prediction = median[0].squeeze().detach().cpu().numpy().astype(float)
                    record["forecast_metrics"] = compute_all_forecast_metrics(
                        target, prediction, naive_forecast=sample["ground_truth"].get("naive_forecast")
                    )
                    del tensor, quantiles, median, prediction
                except Exception as exc:
                    record["forecast_error"] = f"{type(exc).__name__}: {exc}"
            records.append(record)
            # MPS retains allocator blocks across repeated generations; release
            # them between repeats so the diagnostic cannot be killed by cache
            # growth while the formal greedy run remains unchanged.
            if device == "mps":
                try:
                    torch.mps.empty_cache()
                except Exception:
                    pass
            args.output_dir.mkdir(parents=True, exist_ok=True)
            (args.output_dir / "all_results.jsonl").write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8"
            )
        print(f"completed {dimension}/{sid}")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "all_results.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "all_results.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8"
    )
    config = {
        "protocol": "stochastic_controller_and_specialist_variance_probe_v1",
        "device": device,
        "mps_available": bool(torch.backends.mps.is_available()),
        "sample_count": len(selected),
        "repeats": args.repeats,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "enable_thinking": args.enable_thinking,
        "controller_receives_future_values": False,
        "greedy_reference_dir": str(ROOT / "results/controller_fresh_full_strict_normalized_20261003"),
        "elapsed_s": round(time.time() - started, 3),
    }
    (args.output_dir / "run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")

    valid_rows = [r for r in records if r["directive_valid"]]
    summary = {
        "records": len(records),
        "valid_directives": len(valid_rows),
        "valid_rate": len(valid_rows) / len(records),
        "unique_directives": len({json.dumps(r["directive"], sort_keys=True) for r in valid_rows}),
        "context_lengths": sorted({r["directive"]["context_length"] for r in valid_rows}),
        "forecast_metric_means": {},
        "forecast_metric_stds": {},
    }
    for metric in ("MAE", "RMSE", "sMAPE", "MASE", "Direction_Accuracy"):
        values = [r["forecast_metrics"][metric] for r in valid_rows if r.get("forecast_metrics", {}).get(metric) is not None]
        summary["forecast_metric_means"][metric] = float(np.mean(values)) if values else None
        summary["forecast_metric_stds"][metric] = float(np.std(values, ddof=1)) if len(values) > 1 else None
    (args.output_dir / "variance_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
