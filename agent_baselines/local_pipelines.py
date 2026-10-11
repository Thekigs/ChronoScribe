"""Controlled local LLM+TSFM pipelines used to isolate orchestration effects."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from scoring.detailed_scoring import compute_all_forecast_metrics
from utils.benchmark_data import build_context, load_raw
from utils.controller_inputs import history_summary, request_text

from .contracts import BaselineRecord, extract_json, fit_horizon
from .local_models import LocalSpecialist, QwenController


def _forecast_prompt(sample: dict[str, Any], summary: str, horizon: int, specialist: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "只输出JSON，不输出解释。你是一个forecasting controller。"
                "不要访问未来值，不要修改任何预测数值。"
                "输出 {\"context_length\":整数,\"prediction_length\":整数,"
                "\"use_specialist\":true}。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"请求：{request_text(sample)}\n历史摘要：{summary}\n"
                f"specialist={specialist}; 预测步数={horizon}。"
            ),
        },
    ]


def _revision_prompt(sample: dict[str, Any], summary: str, raw_forecast: list[float], horizon: int) -> list[dict[str, str]]:
    preview = ",".join(f"{x:.6g}" for x in raw_forecast[:32])
    return [
        {
            "role": "system",
            "content": (
                "只输出JSON。你是一个受控的 forecast revision controller。"
                "基于历史摘要和TSFM原始预测，给出是否修正的结构化指令。"
                "禁止使用未来真实值。revision 只能使用 scale、offset、trend_delta，"
                "并给出每个字段的数值理由。输出字段："
                "{\"apply_revision\":false,\"scale\":1.0,\"offset\":0.0,"
                "\"trend_delta\":0.0,\"reason\":\"...\"}。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"请求：{request_text(sample)}\n历史摘要：{summary}\n"
                f"原始预测前32步：[{preview}]\n预测步数={horizon}。"
            ),
        },
    ]


def _apply_revision(raw: list[float], directive: dict[str, Any]) -> tuple[list[float], dict[str, Any]]:
    """Apply only bounded, auditable revision operations."""
    try:
        apply = bool(directive.get("apply_revision", False))
        scale = float(directive.get("scale", 1.0))
        offset = float(directive.get("offset", 0.0))
        trend_delta = float(directive.get("trend_delta", 0.0))
    except (TypeError, ValueError):
        return raw, {"valid": False, "reason": "non_numeric_revision"}
    bounded = {
        "scale": float(np.clip(scale, 0.5, 1.5)),
        "offset": float(np.clip(offset, -2.0, 2.0)),
        "trend_delta": float(np.clip(trend_delta, -0.25, 0.25)),
    }
    if not apply:
        return raw, {"valid": True, "applied": False, **bounded}
    revised = [
        float(value * bounded["scale"] + bounded["offset"] + bounded["trend_delta"] * (index + 1))
        for index, value in enumerate(raw)
    ]
    return revised, {"valid": True, "applied": True, **bounded}


class ControlledPipeline:
    """Run one of the two internally controlled fusion baselines.

    ``simple_fusion`` freezes a fixed context and lets Qwen produce only the
    final response metadata. ``revision_control`` exposes the extra operation
    that a ChronoSteer-like workflow would add, while keeping the same TSFM,
    Qwen, manifest, and compute budget.
    """

    def __init__(self, baseline: str, specialist: str, device: str, context_length: int,
                 controller_tokens: int = 256):
        if baseline not in {"simple_fusion", "revision_control"}:
            raise ValueError(f"Unsupported controlled baseline: {baseline}")
        self.baseline = baseline
        self.specialist_name = specialist
        self.context_length = int(context_length)
        self.controller = QwenController(device, max_new_tokens=controller_tokens)
        self.specialist = LocalSpecialist(specialist, device)

    def run_one(self, sample: dict[str, Any], dimension: str) -> dict[str, Any]:
        started = time.time()
        gt = sample.get("ground_truth", {})
        target = gt.get("values", [])
        horizon = len(target) if isinstance(target, list) and target else int(sample.get("horizon", 24))
        record = BaselineRecord(
            id=str(sample["id"]), dimension=dimension,
            baseline=f"{self.baseline}+{self.specialist_name}",
            protocol_status="controlled_adaptation",
            comparability="forecast_only",
            dataset=str(sample.get("dataset", "?")), domain=str(sample.get("domain", "?")),
            numerical_source=self.specialist_name,
        )
        try:
            loaded = load_raw(sample)
            if loaded is None:
                raise RuntimeError("no_usable_history")
            data, source = loaded
            if data is None or len(data) <= 10:
                raise RuntimeError("history_too_short")
            context, alignment = build_context(data, sample, target, qwen_ctx_len=self.context_length)
            summary = history_summary(context)
            directive_text, controller_audit = self.controller.generate(
                _forecast_prompt(sample, summary, horizon, self.specialist_name)
            )
            directive = extract_json(directive_text)
            p10, p50, p90 = self.specialist.forecast(np.asarray(context, dtype=np.float32), horizon)
            revision = None
            if self.baseline == "revision_control":
                revision_text, revision_audit = self.controller.generate(
                    _revision_prompt(sample, summary, p50, horizon)
                )
                revision_directive = extract_json(revision_text)
                p10, revision = _apply_revision(p10, revision_directive)
                p50, _ = _apply_revision(p50, revision_directive)
                p90, _ = _apply_revision(p90, revision_directive)
                p10, p50, p90 = (fit_horizon(values, horizon) for values in (p10, p50, p90))
                revision["raw_output"] = revision_text
                revision["generation_audit"] = revision_audit
            record.predicted_values = fit_horizon(p50, horizon)
            record.predicted_p10 = fit_horizon(p10, horizon)
            record.predicted_p50 = record.predicted_values
            record.predicted_p90 = fit_horizon(p90, horizon)
            if target:
                record.forecast_metrics = compute_all_forecast_metrics(
                    np.asarray(target, dtype=float), np.asarray(record.predicted_values, dtype=float),
                    naive_forecast=gt.get("naive_forecast"),
                )
            record.controller_output = directive
            record.controller_source = "live_qwen"
            record.revision_output = revision
            record.context_alignment = alignment
            record.audit = {
                "data_source": source,
                "controller_raw_output": directive_text,
                "controller_generation": controller_audit,
                "revision_changes_trajectory": self.baseline == "revision_control",
                "future_values_provided_to_controller": False,
                "target_used_only_for_scoring": True,
            }
        except Exception as exc:
            record.error = str(exc)[:240]
        record.latency_s = round(time.time() - started, 3)
        return record.to_dict()
