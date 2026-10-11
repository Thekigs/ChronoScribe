"""Forecast-only numerical-revision control, NOT a ChronoSteer reproduction.

Fixed bounds and a history-only scale are declared before evaluating targets.
The raw TSFM trajectory and the revised trajectory share the very same call.
"""
from __future__ import annotations

import hashlib
import time
from typing import Any

import numpy as np

from .contracts import extract_json
from .local_models import LocalSpecialist, QwenController
from scoring.detailed_scoring import compute_all_forecast_metrics
from utils.benchmark_data import build_context, load_raw
from utils.controller_inputs import history_summary

PROTOCOL = "local_numerical_revision_v2"
BOUNDS = {"scale": (0.5, 1.5), "offset": (-2.0, 2.0), "trend_delta": (-0.25, 0.25)}


def history_coordinates(context) -> tuple[float, float]:
    values = np.asarray(context, dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("nonfinite_or_empty_history")
    center = float(values.mean())
    # Constant histories remain stable: avoid arbitrarily giving them a unit
    # correction scale. The epsilon is proportional to the history magnitude.
    scale = max(float(values.std()), np.finfo(float).eps * max(1.0, abs(center)))
    return center, scale


def revise(raw, directive: dict[str, Any], center: float, unit: float):
    """Strict boolean/numeric schema; invalid directive is an audited no-op.

    y'_t = mu + scale*(y_t-mu) + sigma*(offset + trend_delta*t/H).
    Positive scale preserves quantile ordering. Trend is a TOTAL horizon
    adjustment, not an adjustment multiplied without bound by the horizon.
    """
    values = np.asarray(raw, dtype=float)
    def invalid(reason):
        return values.tolist(), {"valid": False, "applied": False, "reason": reason,
                                  "fallback": "unchanged_raw_forecast", "clipped": False}
    if not isinstance(directive, dict) or type(directive.get("apply_revision")) is not bool:
        return invalid("missing_or_non_boolean_apply_revision")
    if not isinstance(directive.get("reason"), str) or not directive["reason"].strip():
        return invalid("missing_reason")
    params = {}
    for name, bounds in BOUNDS.items():
        number = directive.get(name)
        if isinstance(number, bool) or not isinstance(number, (int, float)) or not np.isfinite(number):
            return invalid(f"missing_or_nonfinite_{name}")
        params[name] = float(np.clip(number, *bounds))
    applied = directive["apply_revision"]
    revised = values.copy()
    if applied:
        t = np.arange(1, len(values) + 1) / max(1, len(values))
        revised = center + params["scale"] * (values - center) + unit * (params["offset"] + params["trend_delta"] * t)
    audit = {"valid": True, "applied": applied, "params": params,
             "reason": directive["reason"], "fallback": None,
             "clipped": any(params[k] != directive[k] for k in params),
             "trajectory_changed": not np.array_equal(revised, values),
             "max_abs_change": float(np.max(np.abs(revised-values))) if len(values) else 0.0}
    return revised.tolist(), audit


def metrics(target, predicted, history, naive):
    report = compute_all_forecast_metrics(target, predicted, naive_forecast=naive)
    # Preserve the legacy project's target-denominator MASE for backward
    # compatibility, explicitly labelled. Standard lag-1 MASE uses HISTORY.
    report["MASE_legacy_target_denominator"] = report.get("MASE")
    denom = float(np.mean(np.abs(np.diff(np.asarray(history, dtype=float)))))
    report["MASE_history_lag1"] = float(report["MAE"] / denom) if denom > 1e-10 else None
    report["MASE_history_denominator"] = denom
    return report


def prompt(sample, context, forecast, horizon):
    # Never accept user prompt arrays as history: those may describe a
    # different context length than the fixed-policy experiment.
    return [{"role": "system", "content":
             '你是时序预测修正控制器。仅输出一个JSON对象，不输出思维过程。'
             '只可根据已观测历史和TSFM原始预测决定是否修正，不存在未来真实值或额外事件信息。'
             '没有足够证据时保持原预测。输出所有字段：'
             '{"apply_revision":false,"scale":1.0,"offset":0.0,"trend_delta":0.0,"reason":"历史证据理由"}。'
             '定义：新预测=历史均值+scale*(原预测-历史均值)+'
             '历史标准差*(offset+trend_delta*t/H)。t=1..H。'
             'scale范围[0.5,1.5]，offset范围[-2,2]，trend_delta范围[-0.25,0.25]；'
             'trend_delta代表整个预测区间末端的调整，不是每一步的增量。'},
            {"role": "user", "content":
             f"dataset={sample.get('dataset')}; series={sample.get('series')}; horizon={horizon}\n"
             f"实际历史：{history_summary(context)}\n"
             f"原始预测摘要：min={min(forecast):.7g}; max={max(forecast):.7g}; "
             f"mean={float(np.mean(forecast)):.7g}; 前32步=[" +
             ",".join(f"{x:.7g}" for x in forecast[:32]) +
             "]; 末8步=[" + ",".join(f"{x:.7g}" for x in forecast[-8:]) +
             "]\n请输出修正JSON。"}]


class StrictRevisionPipeline:
    def __init__(self, specialist, device, context_length, tokens=256):
        self.specialist_name = specialist
        self.device = device
        self.context_length = int(context_length)
        self.controller = QwenController(device, max_new_tokens=tokens)
        self.specialist = LocalSpecialist(specialist, device)

    def run_one(self, sample, dimension):
        started = time.perf_counter()
        row = {"id": str(sample["id"]), "dimension": dimension,
               "dataset": str(sample.get("dataset", "?")), "series": sample.get("series"),
               "baseline": "Qwen_LoRA_bounded_revision+" + self.specialist_name,
               "protocol": PROTOCOL, "protocol_status": "mechanism_control_not_official_reproduction",
               "comparability": "forecast_only_no_external_event_context",
               "device": self.device, "numerical_source": self.specialist_name,
               "tsfm_calls": 0, "qwen_calls": 0}
        try:
            gt = sample.get("ground_truth", {})
            target = gt.get("values", [])
            if not isinstance(target, list) or not target:
                row["skip_reason"] = "no_forecast_target_in_manifest"
                row["comparability"] = "not_applicable_no_forecast_target"
                row["latency_s"] = time.perf_counter() - started
                return row
            # Some robustness rows encode the perturbation horizon in the
            # manifest name while retaining a different evaluation target
            # length. The shared runner uses the actual target length; match
            # that rule instead of rejecting otherwise valid rows.
            horizon = len(target) if isinstance(target, list) and target else int(sample["horizon"])
            if horizon <= 0:
                raise ValueError("target_horizon_missing")
            data, source = load_raw(sample)
            if data is None or len(data) <= 10:
                raise ValueError("no_usable_history")
            context, alignment = build_context(data, sample, target, qwen_ctx_len=self.context_length)
            # Match the shared evaluation loader: exact target-window matches
            # are preferred, while synthetic/cross-domain rows may correctly
            # use the past-only tail before the benchmark target. The latter
            # never exposes target values to the model and is not an error.
            if alignment.get("context_aligned") == "tail_no_gt":
                raise ValueError("unverified_target_alignment")
            center, unit = history_coordinates(context)
            hash_value = hashlib.sha256(np.asarray(context, dtype="<f8").tobytes()).hexdigest()
            row.update({"context_alignment": alignment, "history_sha256": hash_value,
                        "history_center": center, "history_scale": unit,
                        "requested_context_length": self.context_length, "horizon": horizon,
                        "data_source": source, "ground_truth_provided_to_controller": False,
                        "target_used_for_alignment_and_scoring_only": True})
            forecast_started = time.perf_counter()
            row["tsfm_calls"] += 1
            p10, raw, p90 = self.specialist.forecast(np.asarray(context, dtype=np.float32), horizon)
            if len(raw) != horizon or not np.isfinite(raw).all():
                raise ValueError("invalid_raw_forecast")
            row["tsfm_latency_s"] = time.perf_counter() - forecast_started
            row["raw_predicted_values"] = raw
            row["raw_forecast_metrics"] = metrics(target, raw, context, gt.get("naive_forecast"))
            row["qwen_calls"] += 1
            output, generation = self.controller.generate(prompt(sample, context, raw, horizon))
            directive = extract_json(output)
            revised, audit = revise(raw, directive, center, unit)
            if generation["budget_exhausted"]:
                revised, audit = revise(raw, {}, center, unit)
                audit["reason"] = "generation_budget_exhausted"
            row.update({"controller_raw_output": output, "directive": directive,
                        "generation_audit": generation, "revision_audit": audit,
                        "input_tokens": generation["input_tokens"],
                        "output_tokens": generation["generated_tokens"],
                        "predicted_values": revised,
                        "forecast_metrics": metrics(target, revised, context, gt.get("naive_forecast"))})
            if self.specialist_name == "chronos2":
                row["raw_predicted_p10"] = p10
                row["raw_predicted_p90"] = p90
                row["predicted_p10"], _ = revise(p10, directive if audit["valid"] else {}, center, unit)
                row["predicted_p90"], _ = revise(p90, directive if audit["valid"] else {}, center, unit)
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        row["latency_s"] = time.perf_counter() - started
        return row
