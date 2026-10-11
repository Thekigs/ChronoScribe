#!/usr/bin/env python3
"""Batch-generate Qwen3.5+LoRA controller directives on MPS.

The specialist runner can reuse the resulting directory together with the
historical controller directory.  Batch generation avoids paying the full
9B-model launch overhead separately for Chronos-2 and TimesFM.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import hashlib
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(BASE))

from utils.benchmark_data import build_context, load_raw
from utils.controller_inputs import history_summary, request_text
from utils.evaluation_manifest import load_manifest_samples


def extract_json(text: str) -> dict:
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
            pass
    start = text.find("{")
    while start >= 0:
        depth = 0
        for end in range(start, len(text)):
            if text[end] == "{": depth += 1
            elif text[end] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        value = json.loads(text[start:end + 1])
                        if isinstance(value, dict): return value
                    except Exception:
                        break
        start = text.find("{", start + 1)
    return {}


def historical_ids(root: Path) -> dict[tuple[str, str], dict]:
    found = {}
    for path in sorted(root.glob("*/results.json")):
        try:
            rows = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for row in rows:
            if valid_directive(row.get("step1_params")):
                found[(path.parent.name, str(row.get("id")))] = row["step1_params"]
    return found


def has_match(dimension: str, sample_id: str, decisions: dict) -> bool:
    if (dimension, sample_id) in decisions:
        return True
    for (dim, old_id) in decisions:
        if dim != dimension:
            continue
        if str(old_id).endswith(f"__{sample_id}") or sample_id.endswith(f"__{old_id}"):
            return True
    return False


def valid_directive(value):
    if not isinstance(value, dict) or type(value.get("use_chronos2")) is not bool:
        return False
    if not value["use_chronos2"]:
        return isinstance(value.get("direct_reply"), str)
    return all(type(value.get(k)) is int and value[k] > 0
               for k in ("prediction_length", "context_length"))


def normalize_directive(value):
    """Normalize harmless controller JSON wrappers/typos without inventing values.

    The normalization is auditable: it only unwraps a single ``json`` object
    or renames the observed ``use_chrono2`` alias. It never fills missing
    fields, changes integers, or turns free-form text into a decision.
    """
    if not isinstance(value, dict):
        return value
    candidate = value.get("json") if isinstance(value.get("json"), dict) else value
    for alias in ("use_chrono2", "use_chronos"):
        if "use_chronos2" not in candidate and alias in candidate:
            candidate = dict(candidate)
            candidate["use_chronos2"] = candidate.pop(alias)
            break
    return candidate


def checkpoint(root, rows_by_dim, config):
    root.mkdir(parents=True, exist_ok=True)
    for dimension, rows in rows_by_dim.items():
        directory = root / dimension
        directory.mkdir(parents=True, exist_ok=True)
        temp = directory / "results.json.tmp"
        temp.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temp.replace(directory / "results.json")
    config["completed_rows"] = sum(len(rows) for rows in rows_by_dim.values())
    config["valid_decisions"] = sum(bool(r.get("directive_valid")) for rows in rows_by_dim.values() for r in rows)
    (root / "run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Batch-generate live Qwen controller directives.")
    parser.add_argument("--manifest", default=str(BASE / "benchmark/evaluation_manifest.json"))
    parser.add_argument("--dims", default="D1_short_term_forecasting,D2_long_horizon,D3_multivariate,D4_zero_shot,D5_few_shot,D9_robustness,D10_uncertainty,D14_solar,D17_missing_robustness,D20_horizon_degradation,D22_outlier_robustness")
    parser.add_argument("--historical-root", default=str(BASE / "results/qwenchronos"))
    parser.add_argument("--output-dir", default=str(BASE / "results/live_controller_directives"))
    parser.add_argument("--batch-size", type=int, default=4)
    # The fine-tuned controller occasionally emits a short reasoning preamble
    # before the constrained JSON. 64 tokens truncates that response before
    # the directive appears; keep the batch path aligned with the runner's
    # 256-token live generation budget.
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-samples", type=int, default=0, help="Global smoke cap; zero means all planned samples.")
    parser.add_argument("--ignore-historical", action="store_true", help="Fresh manifest decisions for strict input-matched evaluation.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--require-mps", action="store_true")
    retry_group = parser.add_mutually_exclusive_group()
    retry_group.add_argument("--retry-invalid", dest="retry_invalid", action="store_true",
                             help="Retry invalid/truncated outputs with a compact prompt.")
    retry_group.add_argument("--no-retry-invalid", dest="retry_invalid", action="store_false",
                             help="Do not issue a second controller call; keep invalid outputs auditable.")
    parser.set_defaults(retry_invalid=False)
    args = parser.parse_args()

    dims = [x.strip() for x in args.dims.split(",") if x.strip()]
    plan = load_manifest_samples(BASE / "benchmark", args.manifest, dims)
    old = {} if args.ignore_historical else historical_ids(Path(args.historical_root))
    out_root = Path(args.output_dir)
    rows_by_dim = {}
    if args.resume:
        for f in out_root.glob("*/results.json"):
            rows_by_dim[f.parent.name] = json.loads(f.read_text())
    completed = {(d, r["id"]) for d, rows in rows_by_dim.items() for r in rows}
    todo = []
    for dimension, samples in sorted(plan.items()):
        for sample in samples:
            if not has_match(dimension, sample["id"], old) and (dimension, sample["id"]) not in completed:
                todo.append((dimension, sample))
    if args.max_samples:
        todo = todo[:args.max_samples]
    print(f"Samples requiring live Qwen directives: {len(todo)}", flush=True)
    if not todo:
        return

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    if args.require_mps and device != "mps":
        raise RuntimeError("MPS required; refusing CPU fallback")
    print(f"Loading Qwen3.5+LoRA on {device}...", flush=True)
    base_model = AutoModelForCausalLM.from_pretrained(str(BASE / "models/Qwen3.5-9B"), torch_dtype=torch.float16, device_map="cpu")
    qwen = PeftModel.from_pretrained(base_model, str(BASE / "models/checkpoint-7500")).to(device)
    qwen.eval()
    tokenizer = AutoTokenizer.from_pretrained(str(BASE / "models/Qwen3.5-9B"))
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    system = ("你是 ChronoScribe 的 controller，不是 forecaster。你绝对不能生成 predicted_values、"
              "forecast、analysis 或回答用户；只为后面的时序模型选择执行参数。只输出一个 JSON 对象，"
              "格式严格为 {\"use_chronos2\":true,\"prediction_length\":整数,\"context_length\":整数}。"
              "不要输出 Markdown、解释、思考过程或任何其他字段。历史数值仅用于判断上下文长度，不能据此生成预测序列。")
    config = {"manifest": str(Path(args.manifest).resolve()), "batch_size": args.batch_size,
              "max_new_tokens": args.max_new_tokens, "device": device,
              "historical_root": None if args.ignore_historical else str(Path(args.historical_root).resolve()),
              "status": "running", "future_values_provided_to_controller": False,
              "manifest_sha256": hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest()}
    checkpoint(out_root, rows_by_dim, config)
    started = time.time()
    for offset in range(0, len(todo), args.batch_size):
        batch = todo[offset:offset + args.batch_size]
        prompts, valid = [], []
        for dimension, sample in batch:
            gt = sample.get("ground_truth", {}).get("values", [])
            try:
                loaded = load_raw(sample)
                data, source = loaded if loaded is not None else (None, "loader_returned_none")
                if data is None or len(data) <= 10: raise RuntimeError("no_usable_history")
                horizon = len(gt) if isinstance(gt, list) and gt else int(sample.get("horizon", 24))
                context, alignment = build_context(data, sample, gt, qwen_ctx_len=min(4096, len(data)))
                prompts.append(tokenizer.apply_chat_template([
                    {"role": "system", "content": system},
                    {"role": "user", "content": (
                        f"user request:\n{request_text(sample)}\n\n历史摘要:\n{history_summary(context)}\n\n"
                        f"本评测要求另一个模型预测 {horizon} 步。你只选择参数，不执行预测。"
                        "现在只输出上述三个字段的 JSON。"
                    )},
                ], tokenize=False, add_generation_prompt=True,
                    # Keep the controller protocol aligned with the formal
                    # Qwen3.5 evaluation: thinking is enabled for every live
                    # adaptive decision and is recorded in run_config.
                    enable_thinking=True))
                valid.append((dimension, sample, source, alignment,
                              hashlib.sha256(np.asarray(context, dtype="<f8").tobytes()).hexdigest(),
                              np.asarray(context, dtype=np.float32)))
            except Exception as exc:
                rows_by_dim.setdefault(dimension, []).append({"id": sample["id"], "step1_params": {}, "controller_source": "live_error", "error": str(exc)[:240]})
        if prompts:
            print(f"Generating batch {offset + 1}..{offset + len(batch)} / {len(todo)}", flush=True)
            batch_started = time.perf_counter()
            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
            with torch.no_grad():
                generated = qwen.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
            width = inputs["input_ids"].shape[1]
            eos = qwen.generation_config.eos_token_id or tokenizer.eos_token_id
            eos = eos if isinstance(eos, list) else [eos]
            for idx, (dimension, sample, source, alignment, history_hash, context) in enumerate(valid):
                tokens = generated[idx, width:].tolist()
                eos_index = next((i for i, t in enumerate(tokens) if t in eos), None)
                tokens = tokens[:eos_index + 1] if eos_index is not None else tokens
                text = tokenizer.decode(tokens, skip_special_tokens=True)
                directive = normalize_directive(extract_json(text))
                retry_used = False
                retry_audit = None
                initial_audit = {"input_tokens": int(inputs["attention_mask"][idx].sum().item()),
                                 "generated_tokens": len(tokens), "ended_with_eos": eos_index is not None,
                                 "budget_exhausted": eos_index is None and len(tokens) >= args.max_new_tokens}
                if args.retry_invalid and not (valid_directive(directive) and eos_index is not None):
                    compact_messages = [
                        {"role": "system", "content":
                         "你只负责给时序模型选参数，不负责预测。只输出一个JSON对象，不能有"
                         "predicted_values或任何解释。字段必须为use_chronos2、prediction_length、context_length；"
                         "有历史序列和预测任务时use_chronos2=true。"},
                        {"role": "user", "content": (
                            f"dataset={sample.get('dataset')}; series={sample.get('series')}; "
                            f"horizon={len(sample.get('ground_truth', {}).get('values', [])) or sample.get('horizon', 24)}\n"
                            f"history_summary={history_summary(context, max_points=32)}\nJSON:"
                        )},
                    ]
                    compact_prompt = tokenizer.apply_chat_template(
                        compact_messages, tokenize=False, add_generation_prompt=True,
                        enable_thinking=True)
                    retry_inputs = tokenizer(compact_prompt, return_tensors="pt").to(device)
                    retry_started = time.perf_counter()
                    with torch.no_grad():
                        retry_output = qwen.generate(**retry_inputs, max_new_tokens=args.max_new_tokens,
                                                      do_sample=False, pad_token_id=tokenizer.eos_token_id)
                    retry_tokens = retry_output[0][retry_inputs["input_ids"].shape[1]:].tolist()
                    retry_eos = next((i for i, token in enumerate(retry_tokens) if token in eos), None)
                    retry_tokens = retry_tokens[:retry_eos + 1] if retry_eos is not None else retry_tokens
                    retry_text = tokenizer.decode(retry_tokens, skip_special_tokens=True)
                    retry_directive = normalize_directive(extract_json(retry_text))
                    retry_used = True
                    retry_audit = {"input_tokens": int(retry_inputs["attention_mask"].sum().item()),
                                   "generated_tokens": len(retry_tokens), "ended_with_eos": retry_eos is not None,
                                   "budget_exhausted": retry_eos is None and len(retry_tokens) >= args.max_new_tokens,
                                   "latency_s": time.perf_counter() - retry_started}
                    if valid_directive(retry_directive) and retry_eos is not None:
                        text, directive, tokens, eos_index = retry_text, retry_directive, retry_tokens, retry_eos
                selected_audit = {
                    "input_tokens": int(retry_audit["input_tokens"]) if retry_used and retry_audit else int(inputs["attention_mask"][idx].sum().item()),
                    "generated_tokens": len(tokens), "ended_with_eos": eos_index is not None,
                    "budget_exhausted": eos_index is None and len(tokens) >= args.max_new_tokens,
                    "selected_from_retry": bool(retry_used and retry_audit and valid_directive(directive) and eos_index is not None),
                }
                rows_by_dim.setdefault(dimension, []).append({
                    "id": sample["id"], "step1_params": directive,
                    "controller_source": "fresh_manifest_qwen", "raw_output": text,
                    "data_source": source, "context_alignment": alignment, "history_sha256": history_hash,
                    "directive_valid": valid_directive(directive) and eos_index is not None,
                    "retry_used": retry_used,
                    "initial_generation_audit": initial_audit,
                    "retry_audit": retry_audit,
                    "generation_audit": {**selected_audit, "max_new_tokens": args.max_new_tokens,
                                         "batch_latency_s": time.perf_counter() - batch_started,
                                         "enable_thinking": True},
                })
        config["enable_thinking"] = True
        checkpoint(out_root, rows_by_dim, config)
        print(f"{min(offset + args.batch_size, len(todo))}/{len(todo)} directives; elapsed={(time.time()-started)/60:.1f} min", flush=True)

    config["status"] = "complete"
    config["enable_thinking"] = True
    checkpoint(out_root, rows_by_dim, config)
    print(f"Wrote live directives to {out_root}")


if __name__ == "__main__":
    main()
