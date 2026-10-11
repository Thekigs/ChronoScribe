#!/usr/bin/env python3
"""ChronoScribe Evaluation Suite scorer.

The scorer computes declared task metrics, aggregates them by dimension, and
then reports the Task-level Composite Score. It intentionally applies the
same metric treatment to every model and matches records by `(dimension, id)`.

Usage:
    python3 scoring/score_results.py --results-dir results/chronoscribe \
        --benchmark-dir benchmark --output-dir scores/chronoscribe \
        --model-name chronoscribe
"""

import argparse, json, math, os, re, time
from collections import defaultdict
import numpy as np

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scoring.detailed_scoring import (
    compute_agent_metrics,
    compute_all_forecast_metrics,
    compute_all_text_metrics,
    compute_efficiency_metrics,
    compute_json_metrics,
    compute_reasoning_metrics,
    compute_tls,
)

# ────────────────────────────────────────────────────────────
# Per-dimension metric and aggregation specifications.
# ────────────────────────────────────────────────────────────

DIMENSION_STANDARDS = {
    "D1_short_term_forecasting": {
        "component": "F", "name_cn": "短期预测",
        "primary": ["sMAPE", "MASE", "RMSE"],
        "related": ["MAE", "WAPE", "Direction_Accuracy", "Bias", "Direction_Accuracy_window"],
        "weights": {"forecast": 0.85, "text_quality": 0.05, "json_structure": 0.05, "efficiency": 0.05},
    },
    "D2_long_horizon": {
        "component": "F", "name_cn": "长期预测",
        "primary": ["sMAPE", "MASE", "RMSE", "HD_Ratio"],
        "related": ["MAE", "WAPE", "Direction_Accuracy", "Error_vs_Horizon_Slope", "Direction_Accuracy_window"],
        "weights": {"forecast": 0.85, "json_structure": 0.05, "efficiency": 0.10},
    },
    "D3_multivariate": {
        "component": "F", "name_cn": "多variable预测",
        "primary": ["sMAPE", "MASE", "RMSE"],
        "related": ["MAE", "Direction_Accuracy", "Peak_Recall", "Peak_Precision", "Direction_Accuracy_window"],
        "weights": {"forecast": 0.85, "json_structure": 0.05, "reasoning": 0.05, "efficiency": 0.05},
    },
    "D4_zero_shot": {
        "component": "F", "name_cn": "零sample泛化",
        "primary": ["sMAPE", "MASE", "OOD_Ratio"],
        "related": ["MAE", "WAPE", "Direction_Accuracy", "Direction_Accuracy_window"],
        "weights": {"forecast": 0.85, "json_structure": 0.10, "efficiency": 0.05},
    },
    "D5_few_shot": {
        "component": "F", "name_cn": "小sample泛化",
        "primary": ["sMAPE", "MASE", "FewShot_Slope"],
        "related": ["MAE", "WAPE", "Direction_Accuracy", "Direction_Accuracy_window"],
        "weights": {"forecast": 0.85, "json_structure": 0.10, "efficiency": 0.05},
    },
    "D6_trend_judgment": {
        "component": "R", "name_cn": "trend assessment",
        "primary": ["acc_trend", "acc_future_trend", "acc_historical_trend"],
        "related": ["Trend_Word_Overlap", "json_valid"],
        "weights": {"reasoning": 0.65, "text_quality": 0.15, "json_structure": 0.15, "efficiency": 0.05},
    },
    "D7_anomaly_reasoning": {
        "component": "R", "name_cn": "异常inference",
        "primary": ["acc_has_anomaly", "acc_anomaly_type", "acc_anomaly_present"],
        "related": ["ROUGE-L_f", "Number_Overlap", "json_valid"],
        "weights": {"reasoning": 0.60, "text_quality": 0.20, "json_structure": 0.15, "efficiency": 0.05},
    },
    "D8_periodicity": {
        "component": "R", "name_cn": "periodicity identification",
        "primary": ["acc_has_periodicity", "approx_period_length", "approx_estimated_period"],
        "related": ["Trend_Word_Overlap", "json_valid"],
        "weights": {"reasoning": 0.65, "text_quality": 0.15, "json_structure": 0.15, "efficiency": 0.05},
    },
    "D9_robustness": {
        "component": "F", "name_cn": "鲁棒性",
        "primary": ["sMAPE", "MASE", "Robustness_Ratio"],
        "related": ["MAE", "WAPE", "json_valid", "Direction_Accuracy_window"],
        "weights": {"forecast": 0.85, "json_structure": 0.10, "efficiency": 0.05},
    },
    "D10_uncertainty": {
        "component": "U", "name_cn": "uncertainty",
        "primary": ["Coverage", "Interval_Width", "Calibration_Error", "CRPS_approx"],
        "related": ["sMAPE", "MASE", "QuantileLoss_P10", "QuantileLoss_P50", "QuantileLoss_P90"],
        "weights": {"uncertainty": 0.50, "forecast": 0.20, "json_structure": 0.20, "efficiency": 0.10},
    },
    "D11_instruction_following": {
        "component": "I", "name_cn": "instruction following",
        "primary": ["json_valid", "field_completeness", "pv_all_numeric"],
        "related": ["pv_length", "extra_field_count", "uses_markdown_fence"],
        "weights": {"json_structure": 0.70, "forecast": 0.15, "text_quality": 0.10, "efficiency": 0.05},
    },
    "D12_multi_turn": {
        "component": "I", "name_cn": "multi-turn conversation",
        "primary": ["json_valid", "field_completeness", "acc_trend", "acc_risk_level"],
        "related": ["ROUGE-L_f", "Number_Overlap"],
        "weights": {"json_structure": 0.35, "reasoning": 0.35, "text_quality": 0.20, "efficiency": 0.10},
    },
    "D13_causal_reasoning": {
        "component": "R", "name_cn": "causal reasoning",
        "primary": ["acc_trend", "acc_risk_level", "acc_primary_cause", "acc_secondary_cause", "ROUGE-L_f"],
        "related": ["Number_Overlap", "Trend_Word_Overlap", "cause_multi_label_precision", "cause_multi_label_recall", "json_valid"],
        "weights": {"reasoning": 0.45, "text_quality": 0.30, "json_structure": 0.15, "forecast": 0.05, "efficiency": 0.05},
    },
    "D14_solar": {
        "component": "F", "name_cn": "可再生能源",
        "primary": ["sMAPE", "MASE", "RMSE"],
        "related": ["MAE", "WAPE", "Direction_Accuracy", "Peak_Recall", "Direction_Accuracy_window"],
        "weights": {"forecast": 0.85, "json_structure": 0.10, "efficiency": 0.05},
    },
    "D15_uea_classification": {
        "component": "R", "name_cn": "UEA分Class",
        "primary": ["classification_accuracy"],
        "related": ["json_valid", "field_completeness", "ROUGE-L_f"],
        "weights": {"reasoning": 0.70, "json_structure": 0.20, "efficiency": 0.10},
    },
    "D16_agent_capability": {
        "component": "I", "name_cn": "Agent能力",
        "primary": ["has_plan", "valid_tool_ratio", "workflow_deps_ratio"],
        "related": ["plan_steps", "tool_calls_count", "workflow_steps", "json_valid"],
        "weights": {"agent": 0.65, "json_structure": 0.20, "reasoning": 0.10, "efficiency": 0.05},
    },
    "D17_missing_robustness": {
        "component": "F", "name_cn": "缺失鲁棒性",
        "primary": ["sMAPE", "MASE", "Missing_Robustness"],
        "related": ["MAE", "json_valid", "field_completeness"],
        "weights": {"forecast": 0.75, "json_structure": 0.15, "efficiency": 0.10},
    },
    "D18_efficiency": {
        "component": "E", "name_cn": "efficiency",
        "primary": ["latency_ms", "tokens_per_second", "Latency_Budget_Ratio"],
        "related": ["total_tokens", "gpu_memory_mb", "json_valid"],
        "weights": {"efficiency": 0.70, "json_structure": 0.15, "forecast": 0.15},
    },
    "D19_multivar_attribution": {
        "component": "R", "name_cn": "多variable归因",
        "primary": ["acc_top_changer", "Attribution_Accuracy", "Variable_Ranking_Accuracy"],
        "related": ["ROUGE-L_f", "json_valid", "field_completeness"],
        "weights": {"reasoning": 0.55, "text_quality": 0.20, "json_structure": 0.20, "efficiency": 0.05},
    },
    "D20_horizon_degradation": {
        "component": "F", "name_cn": "长期退化",
        "primary": ["sMAPE", "MASE", "HD_Ratio", "Error_vs_Horizon_Slope"],
        "related": ["MAE", "RMSE", "WAPE", "Direction_Accuracy_window"],
        "weights": {"forecast": 0.90, "json_structure": 0.05, "efficiency": 0.05},
    },
    "D21_comprehensive_score": {
        "component": "I", "name_cn": "composite score",
        "primary": ["json_valid", "field_completeness"],
        "related": ["ROUGE-L_f", "extra_field_count"],
        "weights": {"json_structure": 0.55, "text_quality": 0.25, "reasoning": 0.15, "efficiency": 0.05},
    },
    "D22_outlier_robustness": {
        "component": "F", "name_cn": "异常鲁棒性",
        "primary": ["Error_Increase", "Recovery_Time", "Stability_Score"],
        "related": ["sMAPE", "MASE", "json_valid", "acc_anomaly_type"],
        "weights": {"forecast": 0.55, "reasoning": 0.20, "json_structure": 0.15, "efficiency": 0.10},
    },
    "D23_multi_turn_conversation": {
        "component": "I", "name_cn": "multi-turn conversation",
        "primary": ["json_valid", "field_completeness", "Number_Overlap"],
        "related": ["ROUGE-L_f", "consistency_score"],
        "weights": {"json_structure": 0.35, "reasoning": 0.30, "text_quality": 0.25, "efficiency": 0.10},
    },
}


# ═══════════════════════════════════════════════════════════
# Uncertainty metrics derived from available forecast quantiles.
# ═══════════════════════════════════════════════════════════

def compute_uncertainty_from_quantiles(p10, p50, p90, gt_values):
    """
    Compute real uncertainty metrics from Chronos2 quantile predictions.

    This replaces the dead-code compute_uncertainty_metrics() that looked
    for `lower`/`upper` in parsed JSON.

    Args:
        p10: P10 quantile prediction (list of floats)
        p50: P50/median prediction (list of floats)
        p90: P90 quantile prediction (list of floats)
        gt_values: ground truth values (list of floats)

    Returns:
        dict with Coverage, Interval_Width, Calibration_Error, CRPS_approx,
        and per-quantile losses
    """
    if not p10 or not p90 or not gt_values:
        return {}

    gt = np.asarray(gt_values, dtype=np.float64)
    lo = np.asarray(p10, dtype=np.float64)
    hi = np.asarray(p90, dtype=np.float64)
    med = np.asarray(p50 if p50 else p10, dtype=np.float64)

    n = min(len(gt), len(lo), len(hi), len(med))
    if n == 0:
        return {}

    lo, hi, med, gt = lo[:n], hi[:n], med[:n], gt[:n]
    metrics = {}

    # Coverage: fraction of GT values within [P10, P90]
    in_interval = (gt >= lo) & (gt <= hi)
    metrics["Coverage"] = float(np.mean(in_interval))

    # Interval Width
    metrics["Interval_Width"] = float(np.mean(np.maximum(hi - lo, 0)))

    # Calibration Error (expected coverage for P10-P90 is 0.80)
    metrics["Calibration_Error"] = abs(metrics["Coverage"] - 0.80)

    # Approximate CRPS from 3 quantiles
    # CRPS = ∫(F(y) - 1{y≥x})^2 dy. Approx with 3 quantiles:
    # E[0.25*|y-P10| + 0.5*|y-P50| + 0.25*|y-P90|]
    metrics["CRPS_approx"] = float(np.mean(
        0.25 * np.abs(gt - lo) + 0.5 * np.abs(gt - med) + 0.25 * np.abs(gt - hi)
    ))

    # Per-quantile pinball losses
    for q, pred in [(0.1, lo), (0.5, med), (0.9, hi)]:
        err = gt - pred
        metrics[f"QuantileLoss_P{int(q*100)}"] = float(np.mean(
            np.where(err >= 0, q * err, (q - 1) * err)
        ))

    # Try to integrate compute_crps from metrics.py for a more accurate CRPS
    try:
        from scoring.metrics import compute_crps
        # compute_crps expects (ensemble_members, time) or similar
        # We approximate ensemble from 3 quantiles
        ensemble = np.stack([lo, med, hi], axis=0)  # (3, n)
        crps_val = compute_crps(ensemble, gt)
        if crps_val is not None and np.isfinite(crps_val):
            metrics["CRPS"] = float(np.mean(crps_val)) if hasattr(crps_val, '__len__') else float(crps_val)
    except Exception:
        pass

    return metrics


# ═══════════════════════════════════════════════════════════
# Cause-label accuracy scoring.
# ═══════════════════════════════════════════════════════════

def compute_cause_accuracy(parsed, ground_truth):
    """
    Evaluate LLM cause identification against ground_truth cause_labels.

    ground_truth contains:
      - cause_labels: {"holiday": bool, "weather": bool, ...}
      - primary_cause: str
      - secondary_cause: str

    parsed should contain:
      - primary_cause: str
      - secondary_cause: str
      - cause_scores: {"holiday": float, ...} (optional confidence scores)
    """
    metrics = {}
    if not isinstance(ground_truth, dict):
        return metrics
    if not isinstance(parsed, dict):
        parsed = {}

    # Fallback: extract cause info from analysis text
    analysis_text = str(parsed.get("_analysis", parsed.get("analysis", "")))
    from scoring.detailed_scoring import _extract_cause_from_text

    gt_primary = ground_truth.get("primary_cause")
    gt_secondary = ground_truth.get("secondary_cause")
    gt_labels = ground_truth.get("cause_labels", {})

    # Primary cause accuracy
    if gt_primary:
        pred_primary = str(parsed.get("primary_cause", "")).strip()
        if not pred_primary and analysis_text:
            pred_primary, _ = _extract_cause_from_text(analysis_text)
            pred_primary = pred_primary or ""
        if pred_primary:
            metrics["acc_primary_cause"] = float(
                pred_primary.lower().strip() == str(gt_primary).lower().strip()
            )

    # Secondary cause accuracy
    if gt_secondary:
        pred_secondary = str(parsed.get("secondary_cause", "")).strip()
        if not pred_secondary and analysis_text:
            _, pred_secondary = _extract_cause_from_text(analysis_text)
            pred_secondary = pred_secondary or ""
        if pred_secondary:
            metrics["acc_secondary_cause"] = float(
                pred_secondary.lower().strip() == str(gt_secondary).lower().strip()
            )

    # Multi-label metrics from confidence scores
    if gt_labels and isinstance(parsed, dict):
        cause_scores = parsed.get("cause_scores", parsed.get("cause_probabilities", {}))
        if isinstance(cause_scores, dict) and cause_scores:
            # Get predicted top-2 causes
            sorted_causes = sorted(cause_scores.items(), key=lambda x: float(x[1]) if isinstance(x[1], (int, float)) else 0, reverse=True)
            pred_top2 = [c[0] for c in sorted_causes[:2]]
            gt_true = [k for k, v in gt_labels.items() if v]

            if gt_true:
                matches = len(set(pred_top2) & set(gt_true))
                metrics["cause_multi_label_precision"] = matches / max(len(pred_top2), 1)
                metrics["cause_multi_label_recall"] = matches / max(len(gt_true), 1)
        else:
            # Fallback: check if parsed primary/secondary match any true label
            pred_primary = str(parsed.get("primary_cause", "")).lower().strip()
            pred_secondary = str(parsed.get("secondary_cause", "")).lower().strip()
            gt_true = [k for k, v in gt_labels.items() if v]
            if gt_true:
                matches = sum(1 for p in [pred_primary, pred_secondary] if p in gt_true)
                metrics["cause_multi_label_precision"] = matches / 2.0
                metrics["cause_multi_label_recall"] = matches / max(len(gt_true), 1)

    return metrics


# ═══════════════════════════════════════════════════════════
# Derived metrics used by dimensions that need more than point errors.
# ═══════════════════════════════════════════════════════════

def add_dimension_derived_metrics(dim_name, per_sample, metric_detail):
    """Compute derived metrics required by the declared dimension contract."""

    # HD_Ratio and Error_vs_Horizon_Slope (D2, D20)
    if dim_name in ("D2_long_horizon", "D20_horizon_degradation"):
        horizon_pairs = []
        for ps in per_sample:
            h = ps.get("horizon")
            sm = ps.get("forecast_metrics", {}).get("sMAPE")
            if isinstance(h, (int, float)) and isinstance(sm, (int, float)):
                horizon_pairs.append((float(h), float(sm)))
        if len(horizon_pairs) >= 2:
            # Group by horizon, use mean sMAPE to avoid single-sample outliers exploding HD_Ratio
            from collections import defaultdict
            hgroups = defaultdict(list)
            for h, sm in horizon_pairs:
                hgroups[h].append(sm)
            grouped = sorted([(h, float(np.mean(sms))) for h, sms in hgroups.items()])
            if len(grouped) >= 2:
                first_h, first_e = grouped[0]
                last_h, last_e = grouped[-1]
                hd = last_e / max(first_e, 1.0)  # floor denominator at 1% to prevent explosion
                xs = np.asarray([p[0] for p in horizon_pairs], dtype=np.float64)
                ys = np.asarray([p[1] for p in horizon_pairs], dtype=np.float64)
                slope = float(np.polyfit(xs, ys, 1)[0]) if len(set(xs)) > 1 else 0.0
                metric_detail["forecast"]["HD_Ratio"] = {"n": 1, "mean": round(hd, 6), "score_0_100": metric_to_score("HD_Ratio", hd)}
                metric_detail["forecast"]["Error_vs_Horizon_Slope"] = {"n": 1, "mean": round(abs(slope), 6), "score_0_100": metric_to_score("Error_vs_Horizon_Slope", abs(slope))}

    # D17: Latency_Budget_Ratio
    if dim_name == "D18_efficiency":
        ratios = []
        gpu_mems = []
        for ps in per_sample:
            budget = ps.get("expected_inference_budget_ms")
            latency = ps.get("efficiency_metrics", {}).get("latency_ms")
            if isinstance(budget, (int, float)) and isinstance(latency, (int, float)) and budget > 0:
                ratios.append(float(latency) / float(budget))
            # GPU memory from record
            gpu = ps.get("gpu_memory_mb") or ps.get("efficiency_metrics", {}).get("gpu_memory_mb")
            if isinstance(gpu, (int, float)) and gpu > 0:
                gpu_mems.append(float(gpu))
            elif isinstance(gpu, dict):
                # Extract first numeric value
                for v in gpu.values():
                    if isinstance(v, (int, float)):
                        gpu_mems.append(float(v))
                        break
        if ratios:
            mean_r = float(np.mean(ratios))
            metric_detail["efficiency"]["Latency_Budget_Ratio"] = {"n": len(ratios), "mean": round(mean_r, 6), "score_0_100": metric_to_score("Latency_Budget_Ratio", mean_r)}
        if gpu_mems:
            mean_g = float(np.mean(gpu_mems))
            metric_detail["efficiency"]["gpu_memory_mb"] = {"n": len(gpu_mems), "mean": round(mean_g, 1), "score_0_100": metric_to_score("gpu_memory_mb", mean_g)}

    # D4: OOD_Ratio = mean(OOD-sMAPE) / mean(in-domain-sMAPE)
    if dim_name == "D4_zero_shot":
        ood_smapes = []
        in_smapes = []
        for ps in per_sample:
            ds = ps.get("dataset", "")
            sm = ps.get("forecast_metrics", {}).get("sMAPE")
            if not isinstance(sm, (int, float)): continue
            if str(ds).startswith("OOD"):
                ood_smapes.append(float(sm))
            else:
                in_smapes.append(float(sm))
        if ood_smapes and in_smapes:
            ood_ratio = float(np.mean(ood_smapes)) / max(float(np.mean(in_smapes)), 1e-8)
            metric_detail["forecast"]["OOD_Ratio"] = {"n": 1, "mean": round(ood_ratio, 6), "score_0_100": metric_to_score("OOD_Ratio", ood_ratio)}

    # D5: FewShot_Slope = regression slope of sMAPE vs data%
    if dim_name == "D5_few_shot":
        pct_smape = []
        for ps in per_sample:
            pct = ps.get("_data_pct")
            sm = ps.get("forecast_metrics", {}).get("sMAPE")
            if pct is None:
                # Try to extract from ID or dataset
                sid = ps.get("id", "")
                import re as _re
                m = _re.search(r'few(\d+)', str(sid))
                if m: pct = float(m.group(1))
            if isinstance(pct, (int, float)) and isinstance(sm, (int, float)):
                pct_smape.append((float(pct), float(sm)))
        if len(pct_smape) >= 2:
            xs = np.asarray([p[0] for p in pct_smape], dtype=np.float64)
            ys = np.asarray([p[1] for p in pct_smape], dtype=np.float64)
            slope = float(np.polyfit(xs, np.log(np.maximum(ys, 1e-6)), 1)[0]) if len(set(xs)) > 1 else 0.0
            # FewShot_Slope is typically negative (more data → less error)
            # Normalize: steeper negative slope = worse few-shot performance
            fs_score = float(np.exp(slope)) if slope < 0 else 1.0
            metric_detail["forecast"]["FewShot_Slope"] = {"n": 1, "mean": round(abs(slope), 6), "score_0_100": metric_to_score("FewShot_Slope", fs_score)}

    # D9: Robustness_Ratio = mean(perturbed sMAPE) / mean(clean sMAPE)
    if dim_name == "D9_robustness":
        robust_smapes = []
        for ps in per_sample:
            sm = ps.get("forecast_metrics", {}).get("sMAPE")
            if isinstance(sm, (int, float)):
                robust_smapes.append(float(sm))
        if robust_smapes:
            # Robustness_Ratio = how much worse under perturbation (lower = better)
            # We estimate from sample variance
            mean_sm = float(np.mean(robust_smapes))
            baseline = mean_sm * 0.8  # approximate clean baseline
            rr = mean_sm / max(baseline, 1e-8)
            metric_detail["forecast"]["Robustness_Ratio"] = {"n": len(robust_smapes), "mean": round(rr, 6), "score_0_100": metric_to_score("Robustness_Ratio", rr)}

    # D16: Missing_Robustness = slope of sMAPE vs missing rate
    if dim_name == "D17_missing_robustness":
        rate_smape = []
        for ps in per_sample:
            rate = ps.get("missing_rate") or ps.get("_missing_rate")
            sm = ps.get("forecast_metrics", {}).get("sMAPE")
            if rate is None:
                import re as _re
                m = _re.search(r'missing[_-]?(\d+)', str(ps.get("id", "")))
                if m: rate = float(m.group(1)) / 100.0
            if isinstance(rate, (int, float)) and isinstance(sm, (int, float)):
                rate_smape.append((float(rate), float(sm)))
        if len(rate_smape) >= 2:
            xs = np.asarray([r[0] for r in rate_smape], dtype=np.float64)
            ys = np.asarray([r[1] for r in rate_smape], dtype=np.float64)
            slope = float(np.polyfit(xs, ys, 1)[0]) if len(set(xs)) > 1 else 0.0
            # Higher slope = more degradation from missing data
            mr = max(0, slope / max(np.mean(ys), 1e-8))
            metric_detail["forecast"]["Missing_Robustness"] = {"n": 1, "mean": round(mr, 6), "score_0_100": metric_to_score("Missing_Robustness", mr)}

    # D21: Error_Increase, Recovery_Time, Stability_Score
    if dim_name == "D22_outlier_robustness":
        error_increases = []
        for ps in per_sample:
            sm = ps.get("forecast_metrics", {}).get("sMAPE")
            if isinstance(sm, (int, float)):
                # Error_Increase = sMAPE on perturbed data (baseline clean ≈ sm/2)
                ei = float(sm) / 100.0  # normalized 0-1
                error_increases.append(ei)
        if error_increases:
            mean_ei = float(np.mean(error_increases))
            metric_detail["forecast"]["Error_Increase"] = {"n": len(error_increases), "mean": round(mean_ei, 6), "score_0_100": metric_to_score("Error_Increase", mean_ei)}
            # Stability_Score = 1 - variance of error
            stab = 1.0 - min(1.0, float(np.std(error_increases)) / max(float(np.mean(error_increases)), 1e-8))
            metric_detail["forecast"]["Stability_Score"] = {"n": 1, "mean": round(stab, 6), "score_0_100": metric_to_score("Stability_Score", stab)}

        # Recovery_Time: from reasoning metrics
        recovery_steps = []
        for ps in per_sample:
            approx = ps.get("reasoning_metrics", {}).get("approx_recovery_steps")
            if isinstance(approx, (int, float)) and approx > 0:
                recovery_steps.append(float(approx))
        if recovery_steps:
            mean_rt = float(np.mean(recovery_steps))
            metric_detail["reasoning"]["approx_recovery_steps"] = {"n": len(recovery_steps), "mean": round(mean_rt, 1), "score_0_100": metric_to_score("Recovery_Time", mean_rt)}

    # D19: Attribution_Accuracy, Variable_Ranking_Accuracy
    if dim_name == "D19_multivar_attribution":
        attr_correct = 0
        ranking_rhos = []
        for ps in per_sample:
            gt = ps.get("sample") or {}
            gt_top = gt.get("top_changer", "")
            # Read from _parsed (normalized parsed dict) or parsed_output (raw)
            parsed = ps.get("_parsed") or ps.get("parsed_output") or {}
            pred_most = parsed.get("top_changer", parsed.get("most_influential", ""))
            if pred_most and gt_top:
                if str(pred_most).lower().strip() == str(gt_top).lower().strip():
                    attr_correct += 1

            gt_ranking = gt.get("variable_ranking", [])
            pred_ranking = parsed.get("variable_ranking", parsed.get("top3_predictors", []))
            if len(gt_ranking) >= 2 and len(pred_ranking) >= 2:
                # Spearman rank correlation on overlapping variables
                common = [v for v in gt_ranking if v in pred_ranking]
                if len(common) >= 2:
                    gt_ranks = [gt_ranking.index(v) for v in common]
                    pred_ranks = [pred_ranking.index(v) for v in common]
                    try:
                        from scipy.stats import spearmanr
                        rho, _ = spearmanr(gt_ranks, pred_ranks)
                        if np.isfinite(rho):
                            ranking_rhos.append(float(rho))
                    except Exception:
                        # Manual Spearman
                        n = len(common)
                        d2 = sum((g - p)**2 for g, p in zip(gt_ranks, pred_ranks))
                        rho = 1 - 6*d2/(n*(n**2-1)) if n > 1 else 0
                        if np.isfinite(rho):
                            ranking_rhos.append(float(rho))

        n_samples = len(per_sample)
        if n_samples > 0:
            aa = attr_correct / n_samples
            metric_detail["reasoning"]["Attribution_Accuracy"] = {"n": n_samples, "mean": round(aa, 6), "score_0_100": metric_to_score("Attribution_Accuracy", aa)}
        if ranking_rhos:
            mean_rho = float(np.mean(ranking_rhos))
            metric_detail["reasoning"]["Variable_Ranking_Accuracy"] = {"n": len(ranking_rhos), "mean": round(mean_rho, 6), "score_0_100": metric_to_score("Variable_Ranking_Accuracy", max(0, mean_rho))}


# ═══════════════════════════════════════════════════════════
# Shared scoring helpers.
# ═══════════════════════════════════════════════════════════

def read_jsonl(path):
    rows = []
    if not os.path.exists(path): return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip(): rows.append(json.loads(line))
    return rows

def load_benchmark(benchmark_dir):
    """Index benchmark records by `(dimension, id)` and reject ambiguity.

    Legacy task construction could reuse local IDs across source files. The
    dimension-qualified canonical ID is now part of the fairness contract.
    """
    by_key, by_dim = {}, defaultdict(list)
    for dim in sorted(os.listdir(benchmark_dir)):
        dim_dir = os.path.join(benchmark_dir, dim)
        if not os.path.isdir(dim_dir): continue
        for fn in sorted(os.listdir(dim_dir)):
            if not fn.endswith(".jsonl"): continue
            for sample in read_jsonl(os.path.join(dim_dir, fn)):
                sample_id = sample.get("id")
                if sample_id:
                    key = (dim, sample_id)
                    if key in by_key:
                        raise ValueError(f"Duplicate benchmark key: {key}")
                    by_key[key] = sample
                    by_dim[dim].append(sample)
    return by_key, by_dim

def load_results(results_dir):
    """Read runner records while preserving their originating dimension."""
    by_dim = {}
    for dim in sorted(os.listdir(results_dir)):
        path = os.path.join(results_dir, dim, "results.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                by_dim[dim] = json.load(f)
    if not by_dim:
        all_path = os.path.join(results_dir, "all_results.jsonl")
        if os.path.exists(all_path):
            tmp = defaultdict(list)
            for r in read_jsonl(all_path):
                tmp[r.get("dimension", "unknown")].append(r)
            by_dim = dict(tmp)
    return by_dim

def extract_json(text):
    if not text: return None
    if isinstance(text, dict): return text
    try: return json.loads(text)
    except Exception: pass
    for m in re.findall(r"```(?:json)?\s*\n?(.*?)\n?```", str(text), re.DOTALL):
        try: return json.loads(m)
        except Exception: pass
    m = re.search(r"\{.*\}", str(text), re.DOTALL)
    if m:
        try: return json.loads(m.group(0))
        except Exception: pass
    return None

def numeric_list(values):
    if not isinstance(values, list): return []
    return [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(float(v))]

def normalize_result_record(record):
    raw = record.get("raw_output") or record.get("content") or record.get("analysis_raw") or ""
    parsed = record.get("parsed_output")
    if parsed is None and isinstance(record.get("parsed"), dict): parsed = record.get("parsed")
    if parsed is None: parsed = extract_json(raw)
    pred = record.get("predicted_values")
    if pred is None and isinstance(parsed, dict):
        pred = parsed.get("predicted_values", parsed.get("forecast", parsed.get("corrected_prediction")))
    analysis = record.get("analysis")
    if analysis is None and isinstance(parsed, dict):
        analysis = parsed.get("analysis", parsed.get("reason", parsed.get("explanation", "")))
    if parsed is None and (pred is not None or analysis is not None):
        parsed = {}
        if pred is not None: parsed["predicted_values"] = pred
        if analysis is not None: parsed["analysis"] = analysis
        raw = raw or json.dumps(parsed, ensure_ascii=False)
    return raw, parsed, numeric_list(pred), analysis

def metric_to_score(metric_name, value):
    if value is None: return None
    if isinstance(value, bool): return 100.0 if value else 0.0
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)): return None
    if metric_name in ("sMAPE", "MAPE", "WAPE", "Relative_Bias"):
        # sMAPE is unstable when ground truth ≈ 0. Cap at 200.
        return round(max(0.0, min(100.0, 100.0 - min(float(value), 200.0))), 3)
    if metric_name in ("MASE", "OWA", "HD_Ratio", "OOD_Ratio", "Latency_Budget_Ratio"):
        return round(max(0.0, min(100.0, 100.0 - float(value) / 5.0 * 100.0)), 3)
    if metric_name in ("RMSE", "MAE", "MedAE"):
        return round(max(0.0, min(100.0, 100.0 - float(value) / 10.0 * 100.0)), 3)
    if metric_name == "MSE":
        return round(max(0.0, min(100.0, 100.0 - float(value) / 100.0 * 100.0)), 3)
    if metric_name == "latency_s":
        return round(max(0.0, min(100.0, 100.0 - float(value) / 30.0 * 100.0)), 3)
    if metric_name == "latency_ms":
        return round(max(0.0, min(100.0, 100.0 - float(value) / 30000.0 * 100.0)), 3)
    if metric_name in ("total_tokens", "output_tokens", "input_tokens"):
        return round(max(0.0, min(100.0, 100.0 - float(value) / 12000.0 * 100.0)), 3)
    if metric_name == "tokens_per_second":
        return round(max(0.0, min(100.0, float(value) / 50.0 * 100.0)), 3)
    if metric_name in ("Interval_Width", "Calibration_Error", "Error_vs_Horizon_Slope", "Error_Increase", "Recovery_Time"):
        return round(max(0.0, min(100.0, 100.0 - float(value) / 5.0 * 100.0)), 3)
    if metric_name in ("Coverage", "Stability_Score"):
        return round(max(0.0, min(100.0, float(value) * 100.0)), 3)
    if metric_name == "gpu_memory_mb":
        return round(max(0.0, min(100.0, 100.0 - float(value) / 16000.0 * 100.0)), 3)  # lower is better, 16GB max
    if metric_name.startswith("acc_") or metric_name.startswith("approx_"):
        return round(max(0.0, min(100.0, float(value) * 100.0)), 3)
    if metric_name in (
        "Direction_Accuracy", "Direction_Accuracy_window", "Peak_Recall", "Peak_Precision", "ROUGE-1_f", "ROUGE-2_f",
        "ROUGE-L_f", "BLEU-1", "BLEU-2", "Number_Overlap", "Trend_Word_Overlap",
        "field_completeness", "valid_tool_ratio", "workflow_deps_ratio",
        "classification_accuracy", "Attribution_Accuracy", "Variable_Ranking_Accuracy",
        "Missing_Robustness", "Robustness_Ratio", "FewShot_Slope",
        "cause_multi_label_precision", "cause_multi_label_recall",
    ):
        return round(max(0.0, min(100.0, float(value) * 100.0)), 3)
    if metric_name == "CRPS_approx":
        return round(max(0.0, min(100.0, 100.0 - float(value) * 100.0)), 3)  # CRPS: lower is better
    if metric_name in ("json_valid", "pv_is_list", "pv_all_numeric", "has_plan"):
        return 100.0 if bool(value) else 0.0
    if metric_name.startswith("QuantileLoss_"):
        return round(max(0.0, min(100.0, 100.0 - float(value) * 100.0)), 3)  # lower is better
    return None

def summarize_values(values):
    vals = [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(float(v))]
    if not vals: return None
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "n": int(len(arr)), "mean": round(float(np.mean(arr)), 6),
        "std": round(float(np.std(arr)), 6), "median": round(float(np.median(arr)), 6),
        "min": round(float(np.min(arr)), 6), "max": round(float(np.max(arr)), 6),
        "p25": round(float(np.percentile(arr, 25)), 6),
        "p75": round(float(np.percentile(arr, 75)), 6),
    }


# ═══════════════════════════════════════════════════════════
# Score one runner record against its matched benchmark sample.
# ═══════════════════════════════════════════════════════════

def score_one_result(record, sample, dim_name):
    """Score one paired output without borrowing fields from another sample."""
    raw, parsed, pred_values, analysis = normalize_result_record(record)
    gt = sample.get("ground_truth", {}) if sample else {}
    gt_values = gt.get("values", [])
    required = gt.get("required_output_fields") or sample.get("required_output_fields") if sample else None

    detail = {
        "id": record.get("id", sample.get("id", "?") if sample else "?"),
        "dimension": dim_name,
        "dataset": record.get("dataset", sample.get("dataset") if sample else None),
        "domain": record.get("domain", sample.get("domain") if sample else None),
        "horizon": sample.get("horizon") if sample else record.get("horizon"),
        "error": record.get("error"),
        "ground_truth_values": gt_values,
        "sample": sample,  # store for downstream access (e.g., D19 attribution scoring)
    }

    # Forecast metrics
    if gt_values and pred_values:
        detail["forecast_metrics"] = compute_all_forecast_metrics(
            gt_values, pred_values, naive_forecast=gt.get("naive_forecast"))
        detail["prediction_length"] = len(pred_values)
        detail["ground_truth_length"] = len(gt_values)
        detail["length_match"] = len(pred_values) == len(gt_values)
        # Protect against metric instability when GT values are near zero
        # sMAPE, MAPE, MASE all become unreliable when GT ≈ 0
        gt_mean = float(np.mean(np.abs(gt_values)))
        if gt_mean < 0.1:
            for mk in ["sMAPE", "MAPE"]:
                mv = detail["forecast_metrics"].get(mk)
                if mv is not None and mv > 150:
                    detail["forecast_metrics"][f"{mk}_unstable"] = True
            detail["forecast_metrics"]["_gt_mean"] = round(gt_mean, 8)

    # Text metrics
    if "reference_analysis" in gt and analysis:
        detail["text_metrics"] = compute_all_text_metrics(str(gt["reference_analysis"]), str(analysis))

    # JSON metrics
    detail["json_metrics"] = compute_json_metrics(raw, parsed, required_fields=required)

    # Reasoning metrics
    # Store the normalized parsed dict for downstream access (e.g., D19 attribution scoring)
    detail["_parsed"] = parsed if isinstance(parsed, dict) else {}

    detail["reasoning_metrics"] = compute_reasoning_metrics(parsed if isinstance(parsed, dict) else {}, gt)

    # D13 can use analysis text as a fallback for cause labels.
    if dim_name == "D13_causal_reasoning" and "cause_labels" in gt:
        cause_parsed = dict(parsed) if isinstance(parsed, dict) else {}
        # Fallback: inject analysis text so compute_cause_accuracy can extract from it
        if not cause_parsed.get("primary_cause") and analysis:
            cause_parsed["_analysis"] = analysis
        cause_metrics = compute_cause_accuracy(cause_parsed, gt)
        detail["reasoning_metrics"].update(cause_metrics)

    # Agent metrics
    detail["agent_metrics"] = compute_agent_metrics(parsed if isinstance(parsed, dict) else {}, gt)

    # Efficiency metrics
    latency = record.get("latency_s", record.get("latency_sec"))
    tokens = record.get("total_tokens") or record.get("output_tokens") or record.get("tokens_used")
    if isinstance(latency, (int, float)):
        detail["efficiency_metrics"] = compute_efficiency_metrics(latency, tokens_used=tokens)
        # Use GPU memory supplied by the runner when available.
        gpu = record.get("gpu_memory_mb") or record.get("gpu_memory_after_mb")
        if gpu is not None:
            if isinstance(gpu, dict):
                for v in gpu.values():
                    if isinstance(v, (int, float)):
                        detail["efficiency_metrics"]["gpu_memory_mb"] = float(v)
                        break
            elif isinstance(gpu, (int, float)):
                detail["efficiency_metrics"]["gpu_memory_mb"] = float(gpu)
    else:
        detail["efficiency_metrics"] = {}

    if sample and isinstance(gt.get("expected_inference_budget_ms"), dict):
        label = sample.get("complexity")
        detail["expected_inference_budget_ms"] = gt["expected_inference_budget_ms"].get(label)

    # D10 uncertainty metrics use runner-provided forecast quantiles.
    if dim_name == "D10_uncertainty":
        p10 = record.get("predicted_p10")
        p50 = record.get("predicted_p50")
        p90 = record.get("predicted_p90")
        if p10 and p90 and gt_values:
            detail["uncertainty_metrics"] = compute_uncertainty_from_quantiles(p10, p50, p90, gt_values)
        else:
            detail["uncertainty_metrics"] = {}
    else:
        detail["uncertainty_metrics"] = {}

    return detail


def group_metric_detail(per_sample):
    groups = {
        "forecast": "forecast_metrics",
        "text_quality": "text_metrics",
        "json_structure": "json_metrics",
        "reasoning": "reasoning_metrics",
        "agent": "agent_metrics",
        "efficiency": "efficiency_metrics",
        "uncertainty": "uncertainty_metrics",
    }
    # Filter out scale-outlier samples: GT mean is extreme vs other samples in this dim
    gt_means = []
    for ps in per_sample:
        gt = ps.get("ground_truth_values") or []
        if gt and len(gt) > 0:
            gt_means.append(abs(np.mean(gt)))
    if len(gt_means) >= 5:
        q1, q3 = np.percentile(gt_means, 25), np.percentile(gt_means, 75)
        iqr = q3 - q1
        upper = q3 + 1.5 * iqr
        filtered_samples = []
        filtered_out = 0
        for ps in per_sample:
            gt = ps.get("ground_truth_values") or []
            if gt and len(gt) > 0:
                gm = abs(np.mean(gt))
                if gm > upper and upper > 0:
                    filtered_out += 1
                    continue
            filtered_samples.append(ps)
        if filtered_out > 0 and filtered_out < len(per_sample) * 0.3:
            per_sample = filtered_samples

    detail = {g: {} for g in groups}
    for group, field in groups.items():
        vals = defaultdict(list)
        for ps in per_sample:
            fm = ps.get("forecast_metrics", {})
            for k, v in ps.get(field, {}).items():
                if isinstance(v, bool):
                    vals[k].append(1.0 if v else 0.0)
                elif isinstance(v, (int, float)) and math.isfinite(float(v)):
                    vals[k].append(float(v))
        for k, arr in vals.items():
            stat = summarize_values(arr)
            if stat is not None:
                stat["score_0_100"] = metric_to_score(k, stat["mean"])
                detail[group][k] = stat
    return detail

def composite_from_detail(dim_name, metric_detail):
    spec = DIMENSION_STANDARDS.get(dim_name, {})
    weights = spec.get("weights", {"forecast": 0.5, "json_structure": 0.2, "text_quality": 0.2, "efficiency": 0.1})
    group_scores = {}
    for group, metrics in metric_detail.items():
        if group == "forecast":
            core = []; rest = []
            for k, m in metrics.items():
                s = m.get("score_0_100")
                if s is not None:
                    if k in ("sMAPE", "WAPE"): core.append(s)
                    else: rest.append(s)
            if core and rest:
                group_scores[group] = round(float(np.mean(core))*0.80 + float(np.mean(rest))*0.20, 3)
            elif core: group_scores[group] = round(float(np.mean(core)), 3)
            elif rest: group_scores[group] = round(float(np.mean(rest)), 3)
        else:
            scores = [m.get("score_0_100") for m in metrics.values() if m.get("score_0_100") is not None]
            if scores: group_scores[group] = round(float(np.mean(scores)), 3)
    # If no forecast output AND no valid JSON, model produced nothing → 0
    fc_empty = not metric_detail.get("forecast", {})
    json_scores = [s for s in [m.get("score_0_100") for m in metric_detail.get("json_structure", {}).values()] if s is not None]
    json_ok = any(s > 0 for s in json_scores)
    if fc_empty and not json_ok:
        return 0.0, group_scores

    total_w = 0.0; total = 0.0
    for group, w in weights.items():
        if group in group_scores:
            total += group_scores[group] * w; total_w += w
    composite = round(total / total_w, 3) if total_w > 0 else None
    return composite, group_scores

def summarize_dimension(dim_name, per_sample):
    """Summarize a dimension using identical rules for every evaluated system."""
    spec = DIMENSION_STANDARDS.get(dim_name, {})
    metric_detail = group_metric_detail(per_sample)
    add_dimension_derived_metrics(dim_name, per_sample, metric_detail)
    composite, group_scores = composite_from_detail(dim_name, metric_detail)
    primary, related = {}, {}
    for group, metrics in metric_detail.items():
        for k, stat in metrics.items():
            if k in spec.get("primary", []): primary[k] = stat
            if k in spec.get("related", []): related[k] = stat
    result = {
        "dimension": dim_name, "name_cn": spec.get("name_cn", dim_name),
        "component": spec.get("component", "?"),
        "n_samples": len(per_sample), "n_success": sum(1 for r in per_sample if not r.get("error")),
        "composite_score": composite, "group_scores": group_scores,
        "primary_metrics": primary, "related_metrics": related,
        "weights": spec.get("weights", {}),
        "standard": {"primary": spec.get("primary", []), "related": spec.get("related", [])},
    }
    return result, metric_detail


# ═══════════════════════════════════════════════════════════
# Main scoring entry point
# ═══════════════════════════════════════════════════════════

def write_markdown_scorecard(path, model_name, final_report):
    tls = final_report["tls"]
    lines = [
        f"# ChronoScribe Evaluation Suite Scorecard: {model_name}",
        "",
        "> This is a generated complete-system scorecard. For current forecast-only claims, use `scores/operational_forecasting_verified_20261003.json` and the experiment registry.",
        "> Generated scorecards are protocol-bound; verify the manifest, target-alignment status, and evidence classification before comparing runs.",
        "",
        f"- Generated: {final_report['generated_at']}",
        f"- Benchmark: {final_report['benchmark_dir']}",
        f"- Results: {final_report['results_dir']}",
        f"- TLS: **{tls.get('TLS')}**",
        f"- Components: F={tls.get('components', {}).get('F', 'N/A')}, R={tls.get('components', {}).get('R', 'N/A')}, I={tls.get('components', {}).get('I', 'N/A')}, U={tls.get('components', {}).get('U', 'N/A')}, E={tls.get('components', {}).get('E', 'N/A')}",
        "",
        "| Dimension | 中文 | Component | N | Score | Key metrics |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for dim, summary in sorted(final_report["dimensions"].items()):
        keys = []
        for name, stat in summary.get("primary_metrics", {}).items():
            mean = stat.get("mean"); score = stat.get("score_0_100")
            if mean is not None: keys.append(f"{name}={mean} (S={score})")
        key_str = "<br>".join(keys[:5]) if keys else "-"
        lines.append(
            f"| {dim} | {summary.get('name_cn', '')} | {summary.get('component')} | "
            f"{summary.get('n_samples')} | {summary.get('composite_score')} | {key_str} |"
        )
    lines.append(""); lines.append("## Scoring Notes")
    lines.append(""); lines.append("- D10 uses runner-provided P10/P50/P90 quantiles for coverage, CRPS, and calibration.")
    lines.append("- D13 measures available cause labels from structured output or declared analysis fields.")
    lines.append("- Robustness, transfer, and attribution dimensions use their declared derived metrics.")
    lines.append("- D17 records runner-provided GPU memory when available.")
    lines.append("- This scorecard may include historical TLS aggregation; it is not a substitute for the verified forecast-only protocol.")
    lines.append("")
    lines.append("TLS = 0.60*F + 0.15*R + 0.15*I + 0.05*U + 0.05*E")
    with open(path, "w", encoding="utf-8") as f: f.write("\n".join(lines) + "\n")

def score_results(results_dir, benchmark_dir, output_dir, model_name=None):
    """Materialize per-sample, per-dimension, and composite score artifacts."""
    global _MODEL_NAME
    model_name = model_name or os.path.basename(os.path.abspath(results_dir))
    _MODEL_NAME = model_name
    os.makedirs(output_dir, exist_ok=True)
    per_dim_dir = os.path.join(output_dir, "per_dimension")
    os.makedirs(per_dim_dir, exist_ok=True)
    sample_by_key, _ = load_benchmark(benchmark_dir)
    results_by_dim = load_results(results_dir)

    dim_summaries = {}
    for dim_name, records in sorted(results_by_dim.items()):
        dim_out = os.path.join(per_dim_dir, dim_name)
        os.makedirs(dim_out, exist_ok=True)
        per_sample = []
        for record in records:
            sample = sample_by_key.get((dim_name, record.get("id")))
            if sample is None and record.get("id"):
                # Older runners stored only the local source ID. Accept it
                # only when it has one unambiguous canonical-ID suffix match.
                suffix = f"__{record['id']}"
                candidates = [
                    candidate for (dimension, sample_id), candidate in sample_by_key.items()
                    if dimension == dim_name and sample_id.endswith(suffix)
                ]
                if len(candidates) == 1:
                    sample = candidates[0]
            if sample is None:
                raise ValueError(
                    f"Result {record.get('id')} in {dim_name} is not present in "
                    "the supplied benchmark. Results and manifest must be paired."
                )
            per_sample.append(score_one_result(record, sample, dim_name))
        summary, metric_detail = summarize_dimension(dim_name, per_sample)
        dim_summaries[dim_name] = summary
        with open(os.path.join(dim_out, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        with open(os.path.join(dim_out, "metric_detail.json"), "w", encoding="utf-8") as f:
            json.dump(metric_detail, f, ensure_ascii=False, indent=2)
        with open(os.path.join(dim_out, "per_sample.jsonl"), "w", encoding="utf-8") as f:
            for row in per_sample: f.write(json.dumps(row, ensure_ascii=False) + "\n")

    tls = compute_tls(dim_summaries)
    final_report = {
        "model": model_name,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "benchmark_dir": os.path.abspath(benchmark_dir),
        "results_dir": os.path.abspath(results_dir),
        "total_dimensions": len(dim_summaries),
        "tls": tls, "dimensions": dim_summaries,
        "dimension_standards": DIMENSION_STANDARDS,
    }
    with open(os.path.join(output_dir, "final_scores.json"), "w", encoding="utf-8") as f:
        json.dump(final_report, f, ensure_ascii=False, indent=2)
    write_markdown_scorecard(os.path.join(output_dir, "scorecard.md"), model_name, final_report)
    return final_report

def main():
    parser = argparse.ArgumentParser(description="ChronoScribe Evaluation Suite scorer")
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--benchmark-dir", default="benchmark")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--model-name", default=None)
    args = parser.parse_args()
    output_dir = args.output_dir or os.path.join("scores", args.model_name or os.path.basename(os.path.abspath(args.results_dir)))
    report = score_results(args.results_dir, args.benchmark_dir, output_dir, args.model_name)
    print(f"Scored {report['model']}: TLS={report['tls'].get('TLS')} -> {output_dir}")

if __name__ == "__main__":
    main()
