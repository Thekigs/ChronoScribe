#!/usr/bin/env python3
"""
详细scoring system — 每个dimension的每个metric都列出独立分数，同时computation复合分。

结构:
  per_dimension/
    D1_short_term_forecasting/
      ├── metrics_detail.json    ← 每个metric的详细分布
      ├── per_sample.jsonl       ← 每个sample的逐metricscore
      └── summary.json           ← 该dimension汇总

  final_scores.json              ← 所有dimension汇总 + TLS综合分
"""
import os, sys, json, re
import numpy as np
from collections import defaultdict

# ═══════════════════════════════════════════════════════════════
# 1. 预测metric — 逐个computation
# ═══════════════════════════════════════════════════════════════

def compute_all_forecast_metrics(gt_values, pred_values, naive_forecast=None):
    """Compute raw forecasting errors for one ground-truth-aligned sequence."""
    """
    computation全部预测metric，return每个metric的独立值。
    不合并！每个metric单独列出。
    """
    gt = np.array(gt_values, dtype=np.float64)
    pr = np.array(pred_values, dtype=np.float64)
    n = min(len(gt), len(pr))
    gt, pr = gt[:n], pr[:n]

    if n == 0:
        return {"error": "no_valid_pairs"}

    eps = 1e-10
    abs_err = np.abs(gt - pr)
    sq_err = (gt - pr) ** 2

    metrics = {}

    # --- 基础误差 ---
    metrics["MAE"] = float(np.mean(abs_err))
    metrics["MSE"] = float(np.mean(sq_err))
    metrics["RMSE"] = float(np.sqrt(metrics["MSE"]))
    metrics["MedAE"] = float(np.median(abs_err))

    # --- 百分比误差 ---
    with np.errstate(divide='ignore', invalid='ignore'):
        ape = np.abs(gt - pr) / (np.abs(gt) + eps) * 100
        ape = np.where(np.isfinite(ape), ape, 0)
    metrics["MAPE"] = float(np.mean(ape))

    # sMAPE: 对称MAPE
    with np.errstate(divide='ignore', invalid='ignore'):
        smape_vals = 200 * np.abs(gt - pr) / (np.abs(gt) + np.abs(pr) + eps)
        smape_vals = np.where(np.isfinite(smape_vals), smape_vals, 0)
    metrics["sMAPE"] = float(np.mean(smape_vals))

    # MASE: 相对朴素预测
    if naive_forecast is not None and len(gt) > 1:
        naive_err = np.mean(np.abs(gt[1:] - gt[:-1]))
        if naive_err > eps:
            metrics["MASE"] = float(np.mean(abs_err) / naive_err)
        else:
            metrics["MASE"] = 1.0  # 与朴素预测持平
    else:
        metrics["MASE"] = None

    # WAPE: 加权MAPE
    metrics["WAPE"] = float(np.sum(abs_err) / (np.sum(np.abs(gt)) + eps) * 100)

    # OWA: M4竞赛metric (需要MASE和sMAPE)
    if naive_forecast is not None and len(gt) > 1:
        naive_err_smape = 200 * np.mean(np.abs(gt[1:] - gt[:-1]) / (np.abs(gt[:-1]) + np.abs(gt[1:]) + eps))
        if naive_err_smape > eps and metrics["MASE"] is not None:
            metrics["OWA"] = float(0.5 * (metrics["sMAPE"] / naive_err_smape + metrics["MASE"]))
        else:
            metrics["OWA"] = None
    else:
        metrics["OWA"] = None

    # --- 偏差metric ---
    metrics["Bias"] = float(np.mean(pr - gt))  # 正=高估, 负=低估
    metrics["Relative_Bias"] = float(metrics["Bias"] / (np.mean(np.abs(gt)) + eps) * 100)

    # --- directional accuracy (逐步 + 窗口) ---
    if n >= 2:
        gt_dir = np.sign(gt[1:] - gt[:-1])
        pr_dir = np.sign(pr[1:] - pr[:-1])
        metrics["Direction_Accuracy"] = float(np.mean(gt_dir == pr_dir))
        # Window-based direction (k-step trends, filters micro-noise)
        k = max(3, n // 8)
        if k < n:
            gt_win = np.sign(gt[k:] - gt[:-k])
            pr_win = np.sign(pr[k:] - pr[:-k])
            metrics["Direction_Accuracy_window"] = float(np.mean(gt_win == pr_win))
        else:
            metrics["Direction_Accuracy_window"] = None
    else:
        metrics["Direction_Accuracy"] = None
        metrics["Direction_Accuracy_window"] = None

    # --- peak detection ---
    if n >= 3:
        from scipy import signal as scipy_signal
        try:
            gt_peaks, _ = scipy_signal.find_peaks(gt, prominence=np.std(gt)*0.3)
            pr_peaks, _ = scipy_signal.find_peaks(pr, prominence=np.std(pr)*0.3)
            metrics["Peak_Recall"] = float(len(set(gt_peaks) & set(pr_peaks)) / max(len(gt_peaks), 1))
            metrics["Peak_Precision"] = float(len(set(gt_peaks) & set(pr_peaks)) / max(len(pr_peaks), 1))
        except Exception:
            metrics["Peak_Recall"] = None
            metrics["Peak_Precision"] = None

    return metrics


# ═══════════════════════════════════════════════════════════════
# 2. Textmetric — 逐个computation
# ═══════════════════════════════════════════════════════════════

def compute_all_text_metrics(reference, candidate):
    """computation全部Textqualitymetric"""
    if not reference or not candidate:
        return {"error": "empty_input"}

    metrics = {}

    # --- ROUGE (简易实现) ---
    def tokenize(text):
        # 中英文混合分词
        import re
        # 中文: 逐字; 英文: 按空格
        tokens = []
        for chunk in re.split(r'([a-zA-Z0-9]+)', text):
            if re.match(r'[a-zA-Z0-9]+', chunk):
                tokens.append(chunk.lower())
            else:
                tokens.extend(list(chunk))
        return [t for t in tokens if t.strip()]

    ref_tokens = tokenize(str(reference))
    can_tokens = tokenize(str(candidate))

    if not ref_tokens or not can_tokens:
        metrics["ROUGE-1"] = metrics["ROUGE-2"] = metrics["ROUGE-L"] = 0.0
        return metrics

    # ROUGE-1: unigram overlap
    ref_set_1 = set(ref_tokens)
    can_set_1 = set(can_tokens)
    overlap_1 = ref_set_1 & can_set_1
    if ref_set_1:
        metrics["ROUGE-1_r"] = float(len(overlap_1) / len(ref_set_1))
        metrics["ROUGE-1_p"] = float(len(overlap_1) / len(can_set_1)) if can_set_1 else 0.0
        metrics["ROUGE-1_f"] = float(2 * metrics["ROUGE-1_r"] * metrics["ROUGE-1_p"] /
                                      (metrics["ROUGE-1_r"] + metrics["ROUGE-1_p"] + 1e-10))
    else:
        metrics["ROUGE-1_r"] = metrics["ROUGE-1_p"] = metrics["ROUGE-1_f"] = 0.0

    # ROUGE-2: bigram overlap
    ref_bigrams = set(zip(ref_tokens, ref_tokens[1:]))
    can_bigrams = set(zip(can_tokens, can_tokens[1:]))
    overlap_2 = ref_bigrams & can_bigrams
    if ref_bigrams:
        metrics["ROUGE-2_r"] = float(len(overlap_2) / len(ref_bigrams))
        metrics["ROUGE-2_p"] = float(len(overlap_2) / len(can_bigrams)) if can_bigrams else 0.0
        metrics["ROUGE-2_f"] = float(2 * metrics["ROUGE-2_r"] * metrics["ROUGE-2_p"] /
                                      (metrics["ROUGE-2_r"] + metrics["ROUGE-2_p"] + 1e-10))
    else:
        metrics["ROUGE-2_r"] = metrics["ROUGE-2_p"] = metrics["ROUGE-2_f"] = 0.0

    # ROUGE-L: longest common subsequence
    def lcs_len(a, b):
        m, n = len(a), len(b)
        if m == 0 or n == 0: return 0
        dp = [[0]*(n+1) for _ in range(m+1)]
        for i in range(1, m+1):
            for j in range(1, n+1):
                if a[i-1] == b[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        return dp[m][n]

    lcs = lcs_len(ref_tokens, can_tokens)
    metrics["ROUGE-L_r"] = float(lcs / len(ref_tokens)) if ref_tokens else 0.0
    metrics["ROUGE-L_p"] = float(lcs / len(can_tokens)) if can_tokens else 0.0
    metrics["ROUGE-L_f"] = float(2 * metrics["ROUGE-L_r"] * metrics["ROUGE-L_p"] /
                                  (metrics["ROUGE-L_r"] + metrics["ROUGE-L_p"] + 1e-10))

    # --- BLEU (简易, 1-gram & 2-gram) ---
    def ngram_precision(ref, can, n):
        ref_ngrams = [tuple(ref[i:i+n]) for i in range(len(ref)-n+1)]
        can_ngrams = [tuple(can[i:i+n]) for i in range(len(can)-n+1)]
        if not can_ngrams: return 0.0
        matches = sum(1 for ng in can_ngrams if ng in ref_ngrams)
        return matches / len(can_ngrams)

    p1 = ngram_precision(ref_tokens, can_tokens, 1)
    p2 = ngram_precision(ref_tokens, can_tokens, 2)
    brevity_penalty = min(1.0, np.exp(1 - len(ref_tokens)/max(len(can_tokens), 1)))
    if p1 > 0:
        metrics["BLEU-1"] = float(brevity_penalty * p1)
        metrics["BLEU-2"] = float(brevity_penalty * np.sqrt(p1 * p2)) if p2 > 0 else 0.0
    else:
        metrics["BLEU-1"] = metrics["BLEU-2"] = 0.0

    # --- length比 ---
    metrics["Length_Ratio"] = float(len(can_tokens) / max(len(ref_tokens), 1))

    # --- 关键词重叠 (用于事实Consistency) ---
    # extraction数字作为关键事实
    ref_numbers = set(re.findall(r'\d+\.?\d*', str(reference)))
    can_numbers = set(re.findall(r'\d+\.?\d*', str(candidate)))
    if ref_numbers:
        metrics["Number_Overlap"] = float(len(ref_numbers & can_numbers) / len(ref_numbers))
    else:
        metrics["Number_Overlap"] = None

    # 趋势词Consistency
    trend_words = ["上升", "下降", "平稳", "增长", "减少", "波动", "峰值", "谷值"]
    ref_trends = {w for w in trend_words if w in str(reference)}
    can_trends = {w for w in trend_words if w in str(candidate)}
    if ref_trends:
        metrics["Trend_Word_Overlap"] = float(len(ref_trends & can_trends) / len(ref_trends))
    else:
        metrics["Trend_Word_Overlap"] = None

    return metrics


# ═══════════════════════════════════════════════════════════════
# 3. JSON结构metric
# ═══════════════════════════════════════════════════════════════

def compute_json_metrics(raw_output, parsed_json, required_fields=None):
    """evaluationJSON output的各种qualitymetric"""
    required_fields = required_fields or ["predicted_values", "analysis"]
    metrics = {}

    # JSON有效性
    metrics["json_valid"] = parsed_json is not None

    # 必需字段Integrity
    if parsed_json is not None:
        present = [f for f in required_fields if f in parsed_json]
        metrics["required_fields_present"] = len(present)
        metrics["required_fields_total"] = len(required_fields)
        metrics["field_completeness"] = float(len(present) / max(len(required_fields), 1))
    else:
        metrics["required_fields_present"] = 0
        metrics["required_fields_total"] = len(required_fields)
        metrics["field_completeness"] = 0.0

    # Markdowncode块检测
    if raw_output:
        has_md = bool(re.search(r'```', str(raw_output)))
        metrics["uses_markdown_fence"] = has_md
    else:
        metrics["uses_markdown_fence"] = None

    # outputlength
    metrics["output_chars"] = len(str(raw_output)) if raw_output else 0

    # 额外字段（模型是否主动添加了有用信息）
    if parsed_json is not None:
        extra_fields = [f for f in parsed_json if f not in required_fields]
        metrics["extra_fields"] = extra_fields
        metrics["extra_field_count"] = len(extra_fields)
    else:
        metrics["extra_fields"] = []
        metrics["extra_field_count"] = 0

    # predicted_valuestypevalidation
    if parsed_json and "predicted_values" in parsed_json:
        pv = parsed_json["predicted_values"]
        metrics["pv_is_list"] = isinstance(pv, list)
        metrics["pv_length"] = len(pv) if isinstance(pv, list) else 0
        if isinstance(pv, list) and len(pv) > 0:
            metrics["pv_all_numeric"] = all(isinstance(v, (int, float)) for v in pv)
        else:
            metrics["pv_all_numeric"] = False
    else:
        metrics["pv_is_list"] = False
        metrics["pv_length"] = 0
        metrics["pv_all_numeric"] = False

    return metrics


# ═══════════════════════════════════════════════════════════════
# 4. inferencemetric
# ═══════════════════════════════════════════════════════════════

def _extract_trend_from_text(text):
    """从natural language分析Text中extraction趋势方向。

    用于混合预测system，其trend assessment可能位于 analysis 自由Text中
    而非Structured的 JSON key 里。
    """
    if not text:
        return None
    text = str(text)
    # 优先级：明确的趋势声明 > 模糊描述
    if any(w in text for w in ["呈上升趋势", "总体呈上升", "整体呈上升", "明显上升", "持续上升"]):
        return "上升"
    if any(w in text for w in ["呈下降趋势", "总体呈下降", "整体呈下降", "明显下降", "持续下降",
                               "降幅", "呈下降", "下降趋势"]):
        return "下降"
    if any(w in text for w in ["平稳", "波动", "震荡", "持平", "无明显变化", "无明显趋势",
                               "没变化", "基本没变", "稳定在", "几乎可以忽略"]):
        return "平稳"
    # 弱信号
    if "上升" in text and "下降" not in text:
        return "上升"
    if "下降" in text and "上升" not in text:
        return "下降"
    return None


def _extract_cause_from_text(text):
    """从分析Text中extraction因果关系关键词。"""
    if not text:
        return None, None
    text = str(text)
    cause_map = {
        "节假日": ["节假日", "假期", "假日", "节日", "春节", "国庆"],
        "天气": ["天气", "气温", "降雨", "降雪", "台风", "暴雪", "高温", "寒潮"],
        "季节性": ["季节性", "季节", "周期性波动", "周期性变化", "季节因素"],
        "周末效应": ["周末", "工作日", "休息日"],
        "政策变化": ["政策", "调控", "法规", "限行", "补贴"],
        "经济周期": ["经济", "GDP", "经济周期", "衰退", "复苏"],
        "外部冲击": ["外部冲击", "突发事件", "疫情", "自然灾害", "事故", "故障"],
        "传感器故障": ["传感器", "故障", "异常", "噪声", "采集"],
        "趋势变化": ["趋势", "拐点", "反转"],
        "数据缺失": ["缺失", "丢失", "不完整"],
    }
    found = []
    for cause_name, keywords in cause_map.items():
        for kw in keywords:
            if kw in text:
                found.append(cause_name)
                break
    primary = found[0] if len(found) > 0 else None
    secondary = found[1] if len(found) > 1 else None
    return primary, secondary


def compute_reasoning_metrics(parsed, ground_truth):
    """computationinferenceClassdimension的所有metric。

    支持两种input模式：
    1. Structured JSON（DeepSeek 等 API 模型直接output）
    2. 自由Text analysis（预测证据驱动的解释Text）
    """
    metrics = {}
    if not isinstance(parsed, dict):
        parsed = {}

    # extraction分析Text（用于 fallback）
    analysis_text = str(parsed.get("analysis", ""))

    # Boolean判断准确率
    for key in ["has_anomaly", "has_periodicity", "anomaly_present",
                "trend_reversal", "threshold_breached"]:
        if key in ground_truth:
            gt_val = ground_truth[key]
            pred_val = parsed.get(key, None) if parsed else None
            if isinstance(gt_val, bool) and pred_val is not None:
                metrics[f"acc_{key}"] = bool(pred_val) == gt_val

    # String匹配准确率 — enhancement版：先从Structured key 读，再从 analysis Text推断
    for key in ["trend", "trend_direction", "risk_level", "anomaly_type",
                "future_trend", "historical_trend"]:
        if key in ground_truth:
            gt_val = str(ground_truth[key]).strip()
            pred_val = str(parsed.get(key, "")).strip() if parsed else ""

            # 如果Structured key 没有值，尝试从 analysis Textextraction
            if not pred_val and analysis_text:
                extracted = _extract_trend_from_text(analysis_text)
                if extracted:
                    pred_val = extracted

            metrics[f"acc_{key}"] = bool(pred_val) and (gt_val in pred_val or pred_val in gt_val)

    # 因果归因准确率 (D13 causal_reasoning)
    if "primary_cause" in ground_truth:
        gt_primary = str(ground_truth["primary_cause"]).strip()
        pred_primary = str(parsed.get("primary_cause", "")).strip()
        if not pred_primary and analysis_text:
            pred_primary, _ = _extract_cause_from_text(analysis_text)
            pred_primary = pred_primary or ""
        metrics["acc_primary_cause"] = bool(pred_primary) and (
            gt_primary in pred_primary or pred_primary in gt_primary)

    if "secondary_cause" in ground_truth:
        gt_secondary = str(ground_truth["secondary_cause"]).strip()
        pred_secondary = str(parsed.get("secondary_cause", "")).strip()
        if not pred_secondary and analysis_text:
            _, pred_secondary = _extract_cause_from_text(analysis_text)
            pred_secondary = pred_secondary or ""
        metrics["acc_secondary_cause"] = bool(pred_secondary) and (
            gt_secondary in pred_secondary or pred_secondary in gt_secondary)

    # 多variable归因准确率 (D19 multivar_attribution)
    if "top_changer" in ground_truth:
        gt_top = str(ground_truth["top_changer"]).strip()
        pred_top = str(parsed.get("top_changer",
                        parsed.get("most_influential", ""))).strip()
        metrics["acc_top_changer"] = bool(pred_top) and (
            gt_top.lower() == pred_top.lower())

    # Numeric近似准确率 (10%容差)
    for key in ["period_length", "estimated_period", "recovery_steps"]:
        if key in ground_truth and parsed:
            gt_v = ground_truth[key]
            pred_v = parsed.get(key)
            if isinstance(gt_v, (int, float)) and isinstance(pred_v, (int, float)) and gt_v != 0:
                error_pct = abs(float(pred_v) - float(gt_v)) / abs(float(gt_v))
                metrics[f"approx_{key}"] = error_pct < 0.10  # 10%容差
            else:
                metrics[f"approx_{key}"] = None

    # 分Class准确率 (D15 UEA classification)
    if "class_label" in ground_truth and parsed:
        gt_label = str(ground_truth["class_label"])
        pred_label = str(parsed.get("category_guess", parsed.get("class", "")))
        metrics["classification_accuracy"] = gt_label == pred_label

    return metrics


# ═══════════════════════════════════════════════════════════════
# 5. Agentmetric
# ═══════════════════════════════════════════════════════════════

def compute_agent_metrics(parsed, ground_truth):
    """evaluationAgent能力的各个子dimension"""
    metrics = {}

    # 规划Integrity（是否有plan字段）
    if parsed:
        if "plan" in parsed and isinstance(parsed["plan"], list):
            metrics["has_plan"] = True
            metrics["plan_steps"] = len(parsed["plan"])
        else:
            metrics["has_plan"] = False
            metrics["plan_steps"] = 0

        # toolinvocation
        if "tool_calls" in parsed and isinstance(parsed["tool_calls"], list):
            metrics["tool_calls_count"] = len(parsed["tool_calls"])
            # 检查是否都是有效tool名
            valid_tools = ["detect_anomaly", "compute_trend", "find_periodicity",
                          "forecast", "explain_anomaly", "preprocess", "decompose", "report"]
            tools = [t.get("tool", "") if isinstance(t, dict) else str(t)
                    for t in parsed["tool_calls"]]
            metrics["valid_tool_ratio"] = sum(1 for t in tools if t in valid_tools) / max(len(tools), 1)
        else:
            metrics["tool_calls_count"] = 0

        # API编排
        if "workflow" in parsed and isinstance(parsed["workflow"], list):
            metrics["workflow_steps"] = len(parsed["workflow"])
            # 检查dependency关系是否正确
            deps_ok = 0
            for step in parsed["workflow"]:
                if isinstance(step, dict) and step.get("input_from") in [None, "", "null"]:
                    deps_ok += 1  # 第一步应该无dependency
                elif isinstance(step, dict) and step.get("input_from") is not None:
                    deps_ok += 1
            metrics["workflow_deps_ratio"] = deps_ok / max(len(parsed["workflow"]), 1)
        else:
            metrics["workflow_steps"] = 0

        # 澄清问题 (人机协作)
        if "clarification_questions" in parsed:
            cq = parsed["clarification_questions"]
            if isinstance(cq, list):
                metrics["clarification_questions_count"] = len(cq)
            else:
                metrics["clarification_questions_count"] = 0

        # 策略多样性 (Errorrestore)
        if "missing_value_strategy" in parsed:
            mvs = parsed["missing_value_strategy"]
            if isinstance(mvs, dict) and "options" in mvs:
                metrics["strategy_options_count"] = len(mvs["options"]) if isinstance(mvs["options"], list) else 0
            else:
                metrics["strategy_options_count"] = 0

    return metrics


# ═══════════════════════════════════════════════════════════════
# 6. efficiencymetric
# ═══════════════════════════════════════════════════════════════

def compute_efficiency_metrics(latency_seconds, tokens_used=None, gpu_memory_mb=None):
    """computationefficiency相关metric"""
    metrics = {}

    metrics["latency_s"] = round(latency_seconds, 4)
    metrics["latency_ms"] = round(latency_seconds * 1000, 1)

    if tokens_used:
        metrics["tokens_per_second"] = round(tokens_used / max(latency_seconds, 0.001), 1)
        metrics["total_tokens"] = tokens_used

    if gpu_memory_mb:
        metrics["gpu_memory_mb"] = round(gpu_memory_mb, 1)

    return metrics


# ═══════════════════════════════════════════════════════════════
# 7. 汇总 — 每个dimensiongeneration详细报告
# ═══════════════════════════════════════════════════════════════

def summarize_dimension(dim_name, per_sample_results):
    """
    汇总一个dimension的所有sample的metric。

    per_sample_results: [
        {
            "id": "...",
            "forecast_metrics": {...},   ← compute_all_forecast_metrics 的output
            "text_metrics": {...},       ← compute_all_text_metrics 的output
            "json_metrics": {...},       ← compute_json_metrics 的output
            "reasoning_metrics": {...},  ← compute_reasoning_metrics 的output
            "agent_metrics": {...},      ← compute_agent_metrics 的output
            "efficiency_metrics": {...}, ← compute_efficiency_metrics 的output
            "latency_s": 1.23,
        },
        ...
    ]

    return:
        {
            "dimension": dim_name,
            "n_samples": int,
            "composite_score": 0-100,
            "metric_groups": {
                "forecast": {
                    "MAE": {"mean": ..., "std": ..., "median": ..., "p25": ..., "p75": ..., "score_0_100": ...},
                    "sMAPE": {...},
                    ...
                    "composite": 0-100
                },
                "text_quality": {
                    "ROUGE-1_f": {...},
                    "ROUGE-L_f": {...},
                    ...
                    "composite": 0-100
                },
                "json_structure": {
                    "field_completeness": {...},
                    ...
                    "composite": 0-100
                },
                "reasoning": {
                    "acc_trend": {...},
                    ...
                    "composite": 0-100
                },
                "efficiency": {
                    "latency_ms": {...},
                    ...
                    "composite": 0-100
                }
            },
            "per_sample_scores": [...]  ← 每个sample的综合分
        }
    """
    n = len(per_sample_results)
    if n == 0:
        return {"dimension": dim_name, "n_samples": 0, "error": "no_samples"}

    report = {"dimension": dim_name, "n_samples": n, "metric_groups": {}}

    # ── 汇总各metric组 ──
    groups = {
        "forecast": "forecast_metrics",
        "text_quality": "text_metrics",
        "json_structure": "json_metrics",
        "reasoning": "reasoning_metrics",
        "agent": "agent_metrics",
        "efficiency": "efficiency_metrics",
    }

    all_group_composites = {}
    group_weights = {
        "F": 0.60, "R": 0.15, "I": 0.15, "U": 0.05, "E": 0.05,
    }

    for group_name, field_name in groups.items():
        # 收集该组所有metric
        all_metrics = defaultdict(list)
        for r in per_sample_results:
            group_data = r.get(field_name, {})
            if group_data:
                for mk, mv in group_data.items():
                    if isinstance(mv, (int, float)) and mv is not None:
                        all_metrics[mk].append(mv)
                    elif isinstance(mv, bool):
                        all_metrics[mk].append(1.0 if mv else 0.0)

        if not all_metrics:
            continue

        group_report = {}
        group_scores = []  # 用于computation该组的composite

        for metric_name, values in all_metrics.items():
            vals = np.array(values)
            stat = {
                "n_valid": len(vals),
                "mean": round(float(np.mean(vals)), 6),
                "std": round(float(np.std(vals)), 6),
                "median": round(float(np.median(vals)), 6),
                "min": round(float(np.min(vals)), 6),
                "max": round(float(np.max(vals)), 6),
                "p25": round(float(np.percentile(vals, 25)), 6),
                "p75": round(float(np.percentile(vals, 75)), 6),
            }

            # 归一化到0-100分数
            stat["score_0_100"] = normalize_to_score(metric_name, stat["mean"])
            group_report[metric_name] = stat

            # 只有可归一化的metric参与composite
            if stat["score_0_100"] is not None:
                group_scores.append(stat["score_0_100"])

        # 组composite = 该组所有metric分数的均值
        if group_scores:
            group_report["composite"] = round(float(np.mean(group_scores)), 1)
            all_group_composites[group_name] = group_report["composite"]
        else:
            group_report["composite"] = None

        report["metric_groups"][group_name] = group_report

    # ── dimension综合分 = 加权各组composite ──
    total_weight = 0
    weighted_sum = 0
    for gname, gweight in group_weights.items():
        if gname in all_group_composites and all_group_composites[gname] is not None:
            weighted_sum += all_group_composites[gname] * gweight
            total_weight += gweight

    if total_weight > 0:
        report["composite_score"] = round(weighted_sum / total_weight, 1)
    else:
        report["composite_score"] = None

    # ── 逐sample综合分 ──
    per_sample = []
    for r in per_sample_results:
        sample_scores = {}
        for gname, field_name in groups.items():
            gdata = r.get(field_name, {})
            if gdata:
                # 取该sample该组所有metric的均值
                vals = [v for v in gdata.values() if isinstance(v, (int, float)) and v is not None]
                bools = [1.0 if v else 0.0 for v in gdata.values() if isinstance(v, bool)]
                all_v = vals + bools
                if all_v:
                    sample_scores[gname] = round(float(np.mean(all_v)), 4)
        per_sample.append({"id": r.get("id", "?"), "group_scores": sample_scores})

    report["per_sample_scores"] = per_sample

    return report


def normalize_to_score(metric_name, value):
    """将各Classmetric归一化到0-100分"""
    if value is None:
        return None

    # forecast errorClass — 越低越好
    error_metrics = {
        "MAE": (0, 10), "MSE": (0, 100), "RMSE": (0, 10),
        "MedAE": (0, 10), "MAPE": (0, 100), "sMAPE": (0, 100),
        "MASE": (0, 5), "WAPE": (0, 100), "OWA": (0, 5),
        "Bias": (-5, 5), "Relative_Bias": (-50, 50),
    }
    if metric_name in error_metrics:
        lo, hi = error_metrics[metric_name]
        if metric_name in ("Bias", "Relative_Bias"):
            # 偏差越接近0越好
            return round(max(0, min(100, 100 - abs(value) / max(abs(hi), 1) * 100)), 1)
        return round(max(0, min(100, 100 - value / max(hi, 1) * 100)), 1)

    # 准确率/重叠率Class — 越高越好
    accuracy_metrics = [
        "Direction_Accuracy", "Peak_Recall", "Peak_Precision",
        "ROUGE-1_f", "ROUGE-1_r", "ROUGE-1_p",
        "ROUGE-2_f", "ROUGE-2_r", "ROUGE-2_p",
        "ROUGE-L_f", "ROUGE-L_r", "ROUGE-L_p",
        "BLEU-1", "BLEU-2",
        "Number_Overlap", "Trend_Word_Overlap",
        "field_completeness", "pv_all_numeric",
        "valid_tool_ratio", "workflow_deps_ratio",
    ]
    if metric_name in accuracy_metrics or metric_name.startswith("acc_") or metric_name.startswith("approx_"):
        return round(max(0, min(100, float(value) * 100)), 1)

    # BooleanClassmetric
    bool_metrics = ["json_valid", "pv_is_list", "has_plan", "uses_markdown_fence"]
    if metric_name in bool_metrics:
        return 100.0 if value else 0.0

    # efficiencyClass — latency越低越好
    if metric_name in ("latency_s",):
        return round(max(0, min(100, 100 - value * 10)), 1)
    if metric_name == "latency_ms":
        return round(max(0, min(100, 100 - value / 100)), 1)
    if metric_name == "tokens_per_second":
        return round(max(0, min(100, value / 2)), 1)  # 50 tok/s → 100分

    # 计数Class
    if metric_name in ("plan_steps", "tool_calls_count", "workflow_steps",
                       "clarification_questions_count", "strategy_options_count",
                       "extra_field_count"):
        return round(max(0, min(100, value * 20)), 1)

    return None


# ═══════════════════════════════════════════════════════════════
# 8. TLS 综合分
# ═══════════════════════════════════════════════════════════════

def compute_tls(dimension_summaries):
    """
    TLS = 0.60×F + 0.15×R + 0.15×I + 0.05×U + 0.05×E.

    D14 belongs to Forecasting. The calculation uses only dimension summaries
    already produced from paired records; it applies no model-specific trim.

    dimension_mapping: 每个dimension→component
    """
    dim_to_component = {
        "D1_short_term_forecasting": "F",
        "D2_long_horizon": "F",
        "D3_multivariate": "F",
        "D4_zero_shot": "F",
        "D5_few_shot": "F",
        "D6_trend_judgment": "R",
        "D7_anomaly_reasoning": "R",
        "D8_periodicity": "R",
        "D9_robustness": "F",
        "D10_uncertainty": "U",
        "D11_instruction_following": "I",
        "D12_multi_turn": "I",
        "D13_causal_reasoning": "R",
        "D14_solar": "F",
        "D15_uea_classification": "R",
        "D16_agent_capability": "I",
        "D17_missing_robustness": "F",
        "D18_efficiency": "E",
        "D19_multivar_attribution": "R",
        "D20_horizon_degradation": "F",
        "D21_comprehensive_score": "I",
        "D22_outlier_robustness": "F",
        "D23_multi_turn_conversation": "I",
    }

    component_scores = defaultdict(list)
    for dim_name, summary in dimension_summaries.items():
        comp = dim_to_component.get(dim_name)
        if comp and summary.get("composite_score") is not None:
            component_scores[comp].append(summary["composite_score"])

    component_means = {}
    for comp, scores in component_scores.items():
        if scores:
            component_means[comp] = round(float(np.mean(scores)), 1)

    weights = {"F": 0.60, "R": 0.15, "I": 0.15, "U": 0.05, "E": 0.05}
    tls = sum(component_means.get(c, 0) * w for c, w in weights.items())
    total_w = sum(w for c, w in weights.items() if c in component_means)
    tls = round(tls / max(total_w, 0.01), 1)

    return {
        "TLS": tls,
        "components": component_means,
        "formula": "0.60×F + 0.15×R + 0.15×I + 0.05×U + 0.05×E",
        "dimension_contributions": {
            dim: {
                "component": dim_to_component.get(dim, "?"),
                "score": summary.get("composite_score")
            }
            for dim, summary in dimension_summaries.items()
        }
    }


# ═══════════════════════════════════════════════════════════════
# 9. 主入口: evaluation一个完整的run结果
# ═══════════════════════════════════════════════════════════════

def evaluate_run(results_dir, benchmark_dir, output_dir):
    """
    evaluation一次完整running的结果

    results_dir: run_all_dims.py的outputdirectory (package含各dimension的results.json)
    benchmark_dir: benchmark的jsonlfiledirectory
    output_dir: 评分outputdirectory
    """
    os.makedirs(output_dir, exist_ok=True)

    # 收集所有dimension的结果
    all_dim_summaries = {}

    for dim_name in sorted(os.listdir(results_dir)):
        dim_results_path = os.path.join(results_dir, dim_name, "results.json")
        if not os.path.exists(dim_results_path): continue

        with open(dim_results_path) as f:
            dim_results = json.load(f)

        # 对每个samplecomputation详细metric
        per_sample = []
        for r in dim_results:
            sample_detail = {"id": r.get("id", "?")}

            # 预测metric
            if "forecast" in r and "gt_values" in r:
                sample_detail["forecast_metrics"] = compute_all_forecast_metrics(
                    r["gt_values"], r["predicted_values"],
                    r.get("naive_forecast")
                )

            # Textmetric
            if "analysis" in r and "reference_analysis" in r:
                sample_detail["text_metrics"] = compute_all_text_metrics(
                    r["reference_analysis"], r["analysis"]
                )

            # JSONmetric
            sample_detail["json_metrics"] = compute_json_metrics(
                r.get("raw_output"), r.get("parsed_output")
            )

            # inferencemetric
            sample_detail["reasoning_metrics"] = compute_reasoning_metrics(
                r.get("parsed_output"), r.get("ground_truth", {})
            )

            # efficiencymetric
            if "lat" in r:
                sample_detail["efficiency_metrics"] = compute_efficiency_metrics(
                    r["lat"], r.get("tokens_used"), r.get("gpu_memory_mb")
                )

            per_sample.append(sample_detail)

        # 汇总该dimension
        dim_summary = summarize_dimension(dim_name, per_sample)
        all_dim_summaries[dim_name] = dim_summary

        # saving该dimension的详细报告
        dim_out = os.path.join(output_dir, dim_name)
        os.makedirs(dim_out, exist_ok=True)
        with open(os.path.join(dim_out, "summary.json"), 'w') as f:
            json.dump(dim_summary, f, ensure_ascii=False, indent=2)

        # saving逐sample详细分数
        with open(os.path.join(dim_out, "per_sample.jsonl"), 'w') as f:
            for ps in per_sample:
                # 展平所有metric组
                flat = {"id": ps["id"]}
                for gname in ["forecast_metrics", "text_metrics", "json_metrics",
                              "reasoning_metrics", "efficiency_metrics"]:
                    if gname in ps:
                        for mk, mv in ps[gname].items():
                            if isinstance(mv, (int, float, bool, str)) or mv is None:
                                flat[f"{gname}.{mk}"] = mv
                f.write(json.dumps(flat, ensure_ascii=False) + '\n')

    # computationTLS
    tls = compute_tls(all_dim_summaries)

    # saving最终报告
    final_report = {
        "model": "Qwen3.5-9B+Chronos2",
        "total_dimensions": len(all_dim_summaries),
        "tls": tls,
        "dimensions": all_dim_summaries,
    }

    with open(os.path.join(output_dir, "final_scores.json"), 'w') as f:
        json.dump(final_report, f, ensure_ascii=False, indent=2)

    # 打印摘要
    print(f"\n{'='*70}")
    print(f"TLS = {tls['TLS']}  (Task-level Composite Score)")
    print(f"{'='*70}")
    print(f"  F (forecasting, 60%): {tls['components'].get('F', 'N/A')}")
    print(f"  R (temporal reasoning, 15%): {tls['components'].get('R', 'N/A')}")
    print(f"  I (instruction following, 15%): {tls['components'].get('I', 'N/A')}")
    print(f"  U (uncertainty, 5%): {tls['components'].get('U', 'N/A')}")
    print(f"  E (efficiency, 5%): {tls['components'].get('E', 'N/A')}")
    print(f"\n各dimension详细分数:")
    for dim, summary in sorted(all_dim_summaries.items()):
        cs = summary.get("composite_score", "N/A")
        groups = summary.get("metric_groups", {})
        group_scores = {g: d.get("composite", "?") for g, d in groups.items()}
        print(f"  {dim}: {cs}  (各组: {group_scores})")

    return final_report


if __name__ == "__main__":
    print("详细scoring system已就绪")
    print()
    print("= 每个dimension的metric组 =")
    print("  forecast:        MAE, MSE, RMSE, MedAE, MAPE, sMAPE, MASE, WAPE, OWA,")
    print("                   Bias, Relative_Bias, Direction_Accuracy, Peak_Recall, Peak_Precision")
    print("  text_quality:    ROUGE-1_f/r/p, ROUGE-2_f/r/p, ROUGE-L_f/r/p,")
    print("                   BLEU-1, BLEU-2, Length_Ratio, Number_Overlap, Trend_Word_Overlap")
    print("  json_structure:  json_valid, field_completeness, uses_markdown_fence,")
    print("                   pv_is_list, pv_length, pv_all_numeric, extra_field_count")
    print("  reasoning:       acc_trend, acc_has_anomaly, acc_has_periodicity, acc_risk_level,...")
    print("  agent:           has_plan, plan_steps, tool_calls_count, workflow_steps,...")
    print("  efficiency:      latency_ms, tokens_per_second, gpu_memory_mb")
    print()
    print("usageMethod:")
    print("  from scoring.detailed_scoring import evaluate_run")
    print("  evaluate_run('results/chronoscribe', 'benchmark', 'scores/')")
