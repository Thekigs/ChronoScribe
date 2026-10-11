#!/usr/bin/env python3
"""Paired Qwen3.5+LoRA controller with interchangeable TSFM specialists.

This runner is the primary cross-specialist control requested by the review:
the Qwen controller, prompt contract, task horizon, raw-series resolver, and
context alignment stay fixed while only the numerical specialist changes.
It intentionally writes forecast-focused records; full natural-language/TLS
evaluation remains a separate experiment because standalone specialists do
not share the same response interface.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))


def valid_controller_directive(value):
    """Accept only an executable, schema-complete controller decision.

    Empty dictionaries are parse failures, not historical decisions.  Keeping
    this check at the cache boundary prevents a missing/invalid Qwen output
    from silently turning into the default 512-point policy.
    """
    if not isinstance(value, dict) or type(value.get("use_chronos2")) is not bool:
        return False
    if value["use_chronos2"] is False:
        return isinstance(value.get("direct_reply"), str) and bool(value["direct_reply"].strip())
    return all(type(value.get(k)) is int and value[k] > 0
               for k in ("prediction_length", "context_length"))

from scoring.detailed_scoring import compute_all_forecast_metrics
from utils.benchmark_data import build_context, load_raw
from utils.controller_inputs import history_summary, request_text
from utils.evaluation_manifest import load_manifest_samples


DEFAULT_DIMS = [
    "D1_short_term_forecasting", "D2_long_horizon", "D3_multivariate",
    "D4_zero_shot", "D5_few_shot", "D9_robustness", "D10_uncertainty",
    "D14_solar", "D17_missing_robustness", "D20_horizon_degradation",
    "D22_outlier_robustness",
]


def extract_json(text: str | None) -> dict:
    if not text:
        return {}
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else {}
    except Exception:
        pass
    for match in re.findall(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL):
        try:
            value = json.loads(match)
            if isinstance(value, dict):
                return value
        except Exception:
            continue
    for start, char in enumerate(text):
        if char != "{":
            continue
        depth = 0
        for end in range(start, len(text)):
            if text[end] == "{":
                depth += 1
            elif text[end] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        value = json.loads(text[start:end + 1])
                        if isinstance(value, dict):
                            return value
                    except Exception:
                        break
    return {}


def id_variants(value: str) -> set[str]:
    """Return stable task-id forms across versioned and canonical manifests."""
    text = str(value)
    variants = {text, text.rsplit("__", 1)[-1]}
    for item in list(variants):
        variants.add(re.sub(r"^v\d+_", "", item))
    return variants


def fit_horizon(values, horizon: int) -> list[float]:
    values = [float(x) for x in list(values)]
    if len(values) >= horizon:
        return values[:horizon]
    return values + ([values[-1]] * (horizon - len(values)) if values else [0.0] * horizon)


def quantile_forecast(model, context: np.ndarray, horizon: int):
    import torch

    tensor = torch.tensor(context, dtype=torch.float32).reshape(1, 1, -1)
    quantiles, medians = model.predict_quantiles(tensor, prediction_length=horizon)
    if quantiles and hasattr(quantiles[0], "shape") and len(quantiles[0].shape) == 3:
        q = quantiles[0]
        nq = q.shape[2]
        return q[0, :, 0].detach().cpu().tolist(), q[0, :, nq // 2].detach().cpu().tolist(), q[0, :, -1].detach().cpu().tolist()
    med = medians[0].squeeze().detach().cpu().tolist()
    if isinstance(med, float):
        med = [med]
    return med, med, med


def point_forecast(model_name: str, model, context: np.ndarray, horizon: int):
    import torch

    if model_name == "chronos2":
        return quantile_forecast(model, context, horizon)
    if model_name == "timesfm":
        # TimesFM 2.5 is compiled for max_context=1024.  Left padding is the
        # documented runner behavior and does not expose target values.
        x = np.asarray(context, dtype=np.float32)
        if len(x) < 1024:
            x = np.pad(x, (1024 - len(x), 0), mode="edge")
        point, _ = model.forecast(horizon=horizon, inputs=[x])
        med = point[0].tolist()
        return med, med, med
    if model_name == "moirai":
        x = np.asarray(context, dtype=np.float32)
        need = 512 + 720
        x = x[-need:]
        if len(x) < need:
            x = np.pad(x, (need - len(x), 0), mode="edge")
        xt = torch.tensor(x, dtype=torch.float32).reshape(1, -1, 1)
        obs = torch.ones(1, len(x), 1, dtype=torch.bool)
        pad = torch.zeros(1, len(x), dtype=torch.bool)
        with torch.no_grad():
            out = model(past_target=xt, past_observed_target=obs, past_is_pad=pad, num_samples=1)
        med = out[0, 0, :horizon].detach().cpu().numpy().tolist()
        return med, med, med
    if model_name == "moment":
        x = np.asarray(context, dtype=np.float32)[-512:]
        if len(x) < 512:
            x = np.pad(x, (512 - len(x), 0), mode="edge")
        with torch.no_grad():
            out = model(x_enc=torch.tensor(x).reshape(1, 1, -1))
        med = out.forecast[0, 0, :horizon].detach().cpu().numpy().tolist()
        return med, med, med
    if model_name in {"sundial", "aurora"}:
        # Both flow-matching specialists return [batch, samples, horizon].
        # Keep the full sample set for p10/p50/p90 so the uncertainty path is
        # directly comparable to Chronos-2 rather than collapsing to a point
        # forecast before scoring.
        model_device = next(model.parameters()).device
        input_tensor = torch.tensor(context, dtype=torch.float32, device=model_device).reshape(1, -1)
        with torch.no_grad():
            if model_name == "sundial":
                samples = model.generate(
                    input_tensor,
                    max_new_tokens=horizon, num_samples=20,
                )
            else:
                samples = model.generate(
                    inputs=input_tensor,
                    max_output_length=horizon, num_samples=20,
                    inference_token_len=48,
                )
        samples = samples.detach().float().cpu().numpy()
        if samples.ndim == 2:
            samples = samples[:, None, :]
        if samples.shape[-1] < horizon:
            pad = np.repeat(samples[..., -1:], horizon - samples.shape[-1], axis=-1)
            samples = np.concatenate([samples, pad], axis=-1)
        samples = samples[..., :horizon]
        return (
            np.quantile(samples, 0.10, axis=1)[0].tolist(),
            np.quantile(samples, 0.50, axis=1)[0].tolist(),
            np.quantile(samples, 0.90, axis=1)[0].tolist(),
        )
    raise ValueError(f"unknown specialist: {model_name}")


def load_specialist(name: str):
    if name == "chronos2":
        from chronos import Chronos2Pipeline
        return Chronos2Pipeline.from_pretrained(str(BASE / "models/chronos-2"))
    if name == "timesfm":
        import timesfm
        model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
        model.compile(timesfm.ForecastConfig(max_context=1024, max_horizon=720, normalize_inputs=True, per_core_batch_size=1))
        return model
    if name == "moirai":
        from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
        return MoiraiForecast(
            module=MoiraiModule.from_pretrained("Salesforce/moirai-1.0-R-small"),
            prediction_length=720, context_length=512, patch_size="auto",
            num_samples=1, target_dim=1, feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )
    if name == "moment":
        from momentfm import MOMENTPipeline
        model = MOMENTPipeline.from_pretrained(
            "AutonLab/MOMENT-1-large",
            model_kwargs={"task_name": "forecasting", "forecast_horizon": 720,
                          "head_dropout": 0.1, "weight_decay": 0.0,
                          "finetuning_mode": "linear-probing"},
        )
        model.init()
        model.eval()
        return model
    if name == "sundial":
        import torch
        from transformers import AutoModelForCausalLM
        return AutoModelForCausalLM.from_pretrained(
            str(BASE / "models/sundial-base-128m"),
            trust_remote_code=True, local_files_only=True,
            torch_dtype=torch.float32,
        )
    if name == "aurora":
        # Aurora was released against Transformers 4.40 but its model code is
        # otherwise compatible with our current runtime.  Transformers 5
        # renamed this bookkeeping attribute; providing the empty mapping keeps
        # local weight loading deterministic without changing model weights.
        from models.aurora.modeling_aurora import AuroraForPrediction
        AuroraForPrediction.all_tied_weights_keys = {}
        return AuroraForPrediction.from_pretrained(
            str(BASE / "models/aurora"), local_files_only=True,
            trust_remote_code=True,
        )
    raise ValueError(f"unknown specialist: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3.5+LoRA controller with interchangeable TSFM.")
    parser.add_argument("--specialist", required=True, choices=["chronos2", "timesfm", "moirai", "moment", "sundial", "aurora"])
    parser.add_argument("--dims", default=",".join(DEFAULT_DIMS))
    parser.add_argument("--manifest", default=str(BASE / "benchmark/evaluation_manifest.json"))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-samples", type=int, default=0, help="Debug limit per dimension; 0 means all.")
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto",
                        help="Execution device for the numerical specialist (auto chooses MPS when available).")
    parser.add_argument(
        "--controller-results",
        default=str(BASE / "results/qwenchronos"),
        help="Existing Qwen runner output(s) containing step1_params, comma-separated. Used by default to avoid re-generating the same controller decisions.",
    )
    parser.add_argument(
        "--live-controller",
        action="store_true",
        help="Run Qwen3.5+LoRA live instead of reusing controller-results.",
    )
    parser.add_argument(
        "--hybrid-controller",
        action="store_true",
        help="Reuse matching historical decisions and run live Qwen only for unmatched manifest samples.",
    )
    parser.add_argument(
        "--require-controller-match",
        action="store_true",
        help="When reusing historical decisions, skip samples without an exact/suffix match instead of using the default directive.",
    )
    parser.add_argument(
        "--force-context-length",
        type=int,
        default=0,
        help="Override the controller's requested history context length while keeping the same controller decisions (e.g. 512 for a fixed-context control).",
    )
    args = parser.parse_args()

    dims = [x.strip() for x in args.dims.split(",") if x.strip()]
    output_dir = Path(args.output_dir or (BASE / "results" / f"qwen_{args.specialist}"))
    output_dir.mkdir(parents=True, exist_ok=True)
    plan = load_manifest_samples(BASE / "benchmark", args.manifest, dims)

    import torch
    if args.device == "cpu":
        device = "cpu"
    elif args.device == "mps":
        device = "mps"
    else:
        device = "cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")

    # Reuse the exact controller decisions from the released/reference run by
    # default.  This preserves the same Qwen+LoRA decision while allowing the
    # numerical specialist to change.  The live path is kept for GPU reruns.
    controller_map = {}
    if not args.live_controller or args.hybrid_controller:
        controller_roots = [Path(item.strip()) for item in str(args.controller_results).split(",") if item.strip()]
        for controller_root in controller_roots:
          for path in sorted(controller_root.glob("*/results.json")):
              try:
                  records = json.loads(path.read_text(encoding="utf-8"))
              except Exception:
                  continue
              dimension = path.parent.name
              for record in records:
                  # New full-pipeline records call this field step1_params;
                  # older specialist runs persisted the same directive as
                  # controller_output.  Accept both so legacy comparisons use
                  # the historical controller decisions rather than silently
                  # degrading to fallback_default.
                  params = record.get("step1_params")
                  if not isinstance(params, dict):
                      params = record.get("controller_output")
                  if valid_controller_directive(params):
                      controller_map[(dimension, str(record.get("id")))] = params
        print(f"Loaded {len(controller_map)} historical Qwen controller decisions from {args.controller_results}")

    qwen = tokenizer = None
    if args.live_controller or args.hybrid_controller:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print(f"Loading Qwen3.5-9B+LoRA on {device}...")
        base_model = AutoModelForCausalLM.from_pretrained(str(BASE / "models/Qwen3.5-9B"), torch_dtype=torch.float16, device_map="cpu")
        qwen = PeftModel.from_pretrained(base_model, str(BASE / "models/checkpoint-7500")).to(device)
        qwen.eval()
        tokenizer = AutoTokenizer.from_pretrained(str(BASE / "models/Qwen3.5-9B"))
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    def qgen(messages, max_tokens=256):
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output = qwen.generate(**inputs, max_new_tokens=max_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        return tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    def reused_directive(dimension, sample_id):
        exact = controller_map.get((dimension, str(sample_id)))
        if exact is not None:
            return exact, "historical_exact"
        suffix = f"__{sample_id}"
        matches = [value for (dim, old_id), value in controller_map.items()
                   if dim == dimension and str(old_id).endswith(suffix)]
        if len(matches) == 1:
            return matches[0], "historical_suffix"
        # Historical result folders sometimes store a local ID that is itself
        # the suffix of the canonical manifest ID.
        matches = [value for (dim, old_id), value in controller_map.items()
                   if dim == dimension and str(sample_id).endswith(f"__{old_id}")]
        if len(matches) == 1:
            return matches[0], "historical_reverse_suffix"
        # Older released runs use IDs such as ``v8_<task>`` while the current
        # manifest uses ``Dk__...__<task>``. Match the stable task tail after
        # removing the version prefix, but only accept an unambiguous match.
        current_forms = id_variants(sample_id)
        matches = [value for (dim, old_id), value in controller_map.items()
                   if dim == dimension and current_forms.intersection(id_variants(old_id))]
        if len(matches) == 1:
            return matches[0], "historical_normalized"
        return {"use_chronos2": True, "prediction_length": 0, "context_length": 512}, "fallback_default"

    print(f"Loading specialist: {args.specialist}...")
    specialist = load_specialist(args.specialist)
    if args.specialist in {"sundial", "aurora"}:
        specialist = specialist.to(device)
        specialist.eval()
    t0 = time.time()
    all_results = []
    system_prompt = (
        "只输出JSON。判断是否需要time series forecasting：有historical data+预测需求 → "
        "{\"use_chronos2\":true,\"prediction_length\":int,\"context_length\":int}。"
        "禁止输出JSON之外的任何Text。"
    )

    config = {"specialist": args.specialist, "dims": dims, "manifest": str(Path(args.manifest).resolve()),
              "model": str(BASE / "models/Qwen3.5-9B"), "adapter": str(BASE / "models/checkpoint-7500"),
              "device": device,
              "controller": ("qwen3.5-9b+checkpoint-7500" if args.live_controller else
                              ("historical_plus_live_unmatched" if args.hybrid_controller else "reused_step1_params")),
              "controller_results": str(Path(args.controller_results).resolve()),
              "controller_decisions_loaded": len(controller_map)}
    (output_dir / "run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    for dimension, samples in sorted(plan.items()):
        rows = []
        selected = samples[:args.max_samples] if args.max_samples else samples
        print(f"\n[{dimension}] {len(selected)} samples")
        for i, sample in enumerate(selected, start=1):
            gt = sample.get("ground_truth", {})
            target = gt.get("values", [])
            horizon = len(target) if isinstance(target, list) and target else int(sample.get("horizon", 24))
            row = {"id": sample["id"], "dimension": dimension, "dataset": sample.get("dataset", "?"),
                   "domain": sample.get("domain", "?"), "pipeline": f"qwen+{args.specialist}",
                   "specialist": args.specialist}
            started = time.time()
            try:
                loaded = load_raw(sample)
                if loaded is None:
                    data, source = None, None
                else:
                    data, source = loaded
                row["data_source"] = source
                if data is None or len(data) <= 10:
                    raise RuntimeError("no_usable_history")
                source_name = None
                if args.live_controller:
                    preliminary, _ = build_context(data, sample, target, qwen_ctx_len=min(4096, len(data)))
                    directive_text = qgen([{"role": "system", "content": system_prompt}, {"role": "user", "content": (
                        f"user request:\n{request_text(sample)}\n\n历史摘要:\n{history_summary(preliminary)}\n\n"
                        f"本评测要求预测 {horizon} 步。选择 execution mode 和 history context length。JSON。"
                    )}])
                    directive = extract_json(directive_text)
                    row["controller_source"] = "live" if args.live_controller else "live_unmatched"
                else:
                    directive, source_name = reused_directive(dimension, sample["id"])
                    if args.hybrid_controller and source_name == "fallback_default":
                        preliminary, _ = build_context(data, sample, target, qwen_ctx_len=min(4096, len(data)))
                        directive_text = qgen([{"role": "system", "content": system_prompt}, {"role": "user", "content": (
                            f"user request:\n{request_text(sample)}\n\n历史摘要:\n{history_summary(preliminary)}\n\n"
                            f"本评测要求预测 {horizon} 步。选择 execution mode 和 history context length。JSON。"
                        )}])
                        directive = extract_json(directive_text)
                        row["controller_source"] = "live_unmatched"
                    else:
                        row["controller_source"] = source_name
                        if args.require_controller_match and source_name == "fallback_default":
                            raise RuntimeError("missing_controller_match")
                if args.force_context_length > 0:
                    directive = dict(directive or {})
                    directive["context_length"] = int(args.force_context_length)
                row["controller_output"] = directive
                proposed_horizon = directive.get("prediction_length", horizon)
                if not isinstance(proposed_horizon, (int, float)) or proposed_horizon <= 0:
                    proposed_horizon = horizon
                if int(proposed_horizon) != horizon:
                    row["horizon_mismatch"] = {"proposed": int(proposed_horizon), "task": horizon}
                context_limit = {"chronos2": 4096, "timesfm": 1024, "moirai": 512, "moment": 512,
                                 "sundial": 2880, "aurora": 2048}[args.specialist]
                requested_context = directive.get("context_length", 512)
                if not isinstance(requested_context, (int, float)) or requested_context <= 0:
                    requested_context = 512
                requested_context = min(int(requested_context), context_limit)
                context, alignment = build_context(data, sample, target, qwen_ctx_len=requested_context)
                p10, p50, p90 = point_forecast(args.specialist, specialist, np.asarray(context, dtype=np.float32), horizon)
                row["context_alignment"] = alignment
                row["c2_params"] = {"pred_len": horizon, "ctx_len": len(context), "requested_ctx_len": requested_context}
                row["predicted_values"] = fit_horizon(p50, horizon)
                if args.specialist in {"chronos2", "sundial", "aurora"}:
                    row["predicted_p10"] = fit_horizon(p10, horizon)
                    row["predicted_p50"] = fit_horizon(p50, horizon)
                    row["predicted_p90"] = fit_horizon(p90, horizon)
                    row["uncertainty_available"] = True
                else:
                    row["uncertainty_available"] = False
                if target:
                    row["forecast_metrics"] = compute_all_forecast_metrics(
                        np.asarray(target, dtype=float), np.asarray(row["predicted_values"], dtype=float),
                        naive_forecast=gt.get("naive_forecast"),
                    )
                row["error"] = None
            except Exception as exc:
                row["error"] = str(exc)[:240]
            row["latency_s"] = round(time.time() - started, 3)
            rows.append(row)
            all_results.append(row)
            print(f"  {i:>2}/{len(selected)} {sample['id'][:48]} {'OK' if not row.get('error') else row['error'][:45]}", flush=True)
        dim_dir = output_dir / dimension
        dim_dir.mkdir(parents=True, exist_ok=True)
        (dim_dir / "results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    with (output_dir / "all_results.jsonl").open("w", encoding="utf-8") as handle:
        for row in all_results:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    ok = sum(1 for row in all_results if not row.get("error"))
    print(f"{args.specialist}: {ok}/{len(all_results)} succeeded; elapsed={(time.time()-t0)/60:.1f} min")
    print(f"Results: {output_dir}")


if __name__ == "__main__":
    main()
