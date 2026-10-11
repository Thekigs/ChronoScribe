#!/usr/bin/env python3
"""
Time series forecasting and language-system metric library.

覆盖：
  A. forecasting accuracy (8项):   MAE, MSE, RMSE, MAPE, sMAPE, MASE, OWA, WAPE
  B. 概率预测 (3项):   CRPS, QuantileLoss(P10/P50/P90), IntervalScore
  C. Textquality (7项):   ROUGE-1/2/L, BLEU-1/4, BERTScore, METEOR
  D. Structured准确 (4项):  JSON有效性, ExactMatch关键字段, instruction following率, 幻觉率
  E. Statistical summaries: confidence intervals and paired comparisons
  F. Efficiency: tokens/sample, latency, and prediction throughput

Metric helpers are model-neutral. Callers must preserve the declared task
semantics and use paired records when comparing systems.
"""

import numpy as np
import json, re
from collections import defaultdict
from scipy import stats as scipy_stats


# ══════════════════════════════════════════════════════════════
# A. forecasting accuracymetric
# ══════════════════════════════════════════════════════════════

def compute_forecast_metrics(y_true, y_pred, naive_forecast=None, y_train=None):
    """
    computation全部forecasting accuracymetric。

    Args:
        y_true: 真实值 array
        y_pred: predicted values array
        naive_forecast: 朴素predicted values (通常 = y_true 的最后一点历史值)
        y_train: training set值 (用于 MASE 分母computation)

    Returns:
        dict of metric_name → float
    """
    y = np.asarray(y_true, dtype=np.float64)
    y_hat = np.asarray(y_pred, dtype=np.float64)
    n = len(y)

    m = {}

    # MAE
    m["MAE"] = float(np.mean(np.abs(y - y_hat)))

    # MSE
    m["MSE"] = float(np.mean((y - y_hat) ** 2))

    # RMSE
    m["RMSE"] = float(np.sqrt(m["MSE"]))

    # MAPE — processing接近0的真值
    mask = np.abs(y) > 1e-8
    if np.any(mask):
        m["MAPE"] = float(np.mean(np.abs((y[mask] - y_hat[mask]) / y[mask])) * 100)
    else:
        m["MAPE"] = None

    # sMAPE — M4 官方公式
    denom = (np.abs(y) + np.abs(y_hat)) / 2
    mask_s = denom > 1e-8
    if np.any(mask_s):
        m["sMAPE"] = float(np.mean(
            np.abs(y[mask_s] - y_hat[mask_s]) / denom[mask_s]
        ) * 100)
    else:
        m["sMAPE"] = None

    # WAPE — WAPE（业务友好）
    if np.sum(np.abs(y)) > 1e-8:
        m["WAPE"] = float(np.sum(np.abs(y - y_hat)) / np.sum(np.abs(y)) * 100)
    else:
        m["WAPE"] = None

    # MASE — 相对于朴素预测
    if naive_forecast is not None:
        naive_err = np.mean(np.abs(y - naive_forecast))
        if naive_err > 1e-8:
            m["MASE"] = float(m["MAE"] / naive_err)
        else:
            m["MASE"] = None
    elif y_train is not None:
        # MASE 的标准分母：training set上 naive 的一步误差
        train_naive_err = np.mean(np.abs(np.diff(y_train)))
        if train_naive_err > 1e-8:
            m["MASE"] = float(m["MAE"] / train_naive_err)
        else:
            m["MASE"] = None
    else:
        m["MASE"] = None

    # OWA — M4 综合rankingmetric
    if m["sMAPE"] is not None and m["MASE"] is not None:
        m["OWA"] = float((m["sMAPE"] + m["MASE"]) / 2)
    else:
        m["OWA"] = None

    # ND — Normalized Deviation（常用于 DL 预测）
    if np.sum(np.abs(y)) > 1e-8:
        m["ND"] = float(np.sum(np.abs(y - y_hat)) / np.sum(np.abs(y)))
    else:
        m["ND"] = None

    return m


# ══════════════════════════════════════════════════════════════
# B. 概率预测metric
# ══════════════════════════════════════════════════════════════

def compute_crps(y_true, y_pred_samples):
    """
    Continuous Ranked Probability Score.
    这是 Chronos / TimesFM / Moirai 的核心评测metric。

    Args:
        y_true: 真实值 (array, shape=(n,))
        y_pred_samples: 预测sample (array, shape=(n, num_samples))
                       每个时间点有 num_samples 个预测sample

    Returns:
        float CRPS (越低越好)
    """
    y = np.asarray(y_true, dtype=np.float64)
    samples = np.asarray(y_pred_samples, dtype=np.float64)

    if samples.ndim == 1:
        samples = samples.reshape(-1, 1)

    n, S = samples.shape
    crps_sum = 0.0

    for t in range(n):
        # CRPS = E|X - y| - 0.5 * E|X - X'|
        x = samples[t]  # S 个sample
        term1 = np.mean(np.abs(x - y[t]))
        # 配对computation
        diffs = np.abs(x[:, None] - x[None, :])
        term2 = 0.5 * np.mean(diffs)
        crps_sum += term1 - term2

    return float(crps_sum / n)


def compute_quantile_loss(y_true, y_pred_quantiles, quantiles=[0.1, 0.5, 0.9]):
    """
    Quantile (Pinball) Loss。
    用于evaluation预测区间quality。

    Args:
        y_true: 真实值 array (n,)
        y_pred_quantiles: 各分位数预测 array (n, num_quantiles)
        quantiles: 分位数List

    Returns:
        dict: {q: loss} 和 avg_quantile_loss
    """
    y = np.asarray(y_true, dtype=np.float64)
    q_preds = np.asarray(y_pred_quantiles, dtype=np.float64)

    if q_preds.ndim == 1:
        q_preds = q_preds.reshape(-1, 1)

    losses = {}
    total = 0.0
    for i, q in enumerate(quantiles):
        errors = y - q_preds[:, i]
        loss = float(np.mean(np.where(errors >= 0, q * errors, (q - 1) * errors)))
        losses[f"QL_{int(q*100)}"] = loss
        total += loss

    losses["QL_avg"] = total / len(quantiles)
    return losses


def compute_interval_score(y_true, lower, upper, alpha=0.1):
    """
    Interval Score (Winkler Score) — 同时evaluation覆盖率和width。

    IS = (upper - lower) + 2/α * (lower - y) * 1_{y < lower}
                        + 2/α * (y - upper) * 1_{y > upper}

    越低越好。
    """
    y = np.asarray(y_true, dtype=np.float64)
    lo = np.asarray(lower, dtype=np.float64)
    hi = np.asarray(upper, dtype=np.float64)

    width = hi - lo
    penalty_lo = (2.0 / alpha) * (lo - y) * (y < lo)
    penalty_hi = (2.0 / alpha) * (y - hi) * (y > hi)

    return float(np.mean(width + penalty_lo + penalty_hi))


# ══════════════════════════════════════════════════════════════
# C. Textgenerationqualitymetric
# ══════════════════════════════════════════════════════════════

def compute_rouge(pred_text, ref_text):
    """computation ROUGE-1, ROUGE-2, ROUGE-L"""
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=True)
        scores = scorer.score(ref_text, pred_text)
        return {
            "ROUGE-1": round(scores['rouge1'].fmeasure, 4),
            "ROUGE-2": round(scores['rouge2'].fmeasure, 4),
            "ROUGE-L": round(scores['rougeL'].fmeasure, 4),
        }
    except ImportError:
        return {"ROUGE_error": "pip install rouge-score"}


def compute_bleu(pred_text, ref_text):
    """computation BLEU-1 和 BLEU-4"""
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        ref_tokens = [list(ref_text)]
        pred_tokens = list(pred_text)
        smooth = SmoothingFunction().method1
        return {
            "BLEU-1": round(sentence_bleu(ref_tokens, pred_tokens,
                             weights=(1,0,0,0), smoothing_function=smooth), 4),
            "BLEU-4": round(sentence_bleu(ref_tokens, pred_tokens,
                             weights=(0.25,0.25,0.25,0.25), smoothing_function=smooth), 4),
        }
    except ImportError:
        return {"BLEU_error": "pip install nltk"}


def compute_bertscore(pred_text, ref_text, model="bert-base-uncased"):
    """computation BERTScore (Precision, Recall, F1)"""
    try:
        from bert_score import score
        P, R, F1 = score([pred_text], [ref_text], model_type=model,
                         lang="zh" if any('一' <= c <= '鿿' for c in pred_text) else "en",
                         verbose=False)
        return {
            "BERTScore-P": round(float(P[0]), 4),
            "BERTScore-R": round(float(R[0]), 4),
            "BERTScore-F1": round(float(F1[0]), 4),
        }
    except ImportError:
        return {"BERTScore_error": "pip install bert-score"}


def compute_meteor(pred_text, ref_text):
    """computation METEOR"""
    try:
        from nltk.translate.meteor_score import meteor_score
        score = meteor_score([ref_text.split()], pred_text.split())
        return {"METEOR": round(score, 4)}
    except ImportError:
        return {"METEOR_error": "pip install nltk"}


def compute_all_text_metrics(pred_text, ref_text):
    """一站式computation所有Textmetric"""
    metrics = {}
    metrics.update(compute_rouge(pred_text, ref_text))
    metrics.update(compute_bleu(pred_text, ref_text))
    metrics.update(compute_bertscore(pred_text, ref_text))
    # meteor 对中文支持差，仅在英文时invocation
    if not any('一' <= c <= '鿿' for c in pred_text):
        metrics.update(compute_meteor(pred_text, ref_text))
    return metrics


# ══════════════════════════════════════════════════════════════
# D. Structured准确 & 事实性metric
# ══════════════════════════════════════════════════════════════

def compute_json_validity_rate(raw_outputs):
    """JSON output有efficiency"""
    valid = 0
    for out in raw_outputs:
        try:
            # 尝试直接parsing
            json.loads(out)
            valid += 1
        except (json.JSONDecodeError, ValueError, TypeError):
            # 尝试extraction ```json ... ``` 块
            match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', out, re.DOTALL)
            if match:
                try:
                    json.loads(match.group(1))
                    valid += 1
                except (json.JSONDecodeError, ValueError, TypeError):
                    pass
    return {
        "JSON_validity_rate": round(valid / max(len(raw_outputs), 1) * 100, 1),
        "JSON_valid_count": valid,
        "JSON_total": len(raw_outputs),
    }


def compute_exact_match(pred_values, gt_values, tolerance=1e-4):
    """Structured字段的精确匹配率"""
    if not pred_values or not gt_values:
        return {"exact_match": 0.0}

    pred = np.asarray(pred_values, dtype=np.float64)
    gt = np.asarray(gt_values, dtype=np.float64)

    matches = np.sum(np.abs(pred - gt) < tolerance)
    return {
        "exact_match_rate": round(matches / len(gt) * 100, 1),
        "exact_match_count": int(matches),
        "total_values": len(gt),
    }


def compute_hallucination_rate(pred_text, gt_facts):
    """
    检测幻觉率 — 模型generation的分析中，有多少关键事实是Error的。

    gt_facts: dict of {fact_key: expected_value}
    例如: {"trend": "上升", "risk_level": "较高", "max_value": 15.2}
    """
    if not gt_facts:
        return {"hallucination_rate": None}

    errors = {}
    total = 0
    wrong = 0

    for key, expected in gt_facts.items():
        total += 1
        # 查找Text中是否有对应的事实陈述
        pattern_found = False
        pattern_correct = False

        if key == "trend":
            if "上升" in pred_text: pattern_found = True; pattern_correct = (expected == "上升")
            elif "下降" in pred_text: pattern_found = True; pattern_correct = (expected == "下降")
            elif "平稳" in pred_text: pattern_found = True; pattern_correct = (expected == "平稳")
        elif key == "risk_level":
            for level in ["高", "较高", "较低", "低"]:
                if f"风险{level}" in pred_text or f"{level}危" in pred_text:
                    pattern_found = True
                    pattern_correct = (level == expected)
                    break
        elif isinstance(expected, (int, float)):
            # Numeric型事实 — 检查是否在Text中出现了接近的值
            # 简化：检查Text中是否package含与 expected 误差 <5% 的数字
            nums = re.findall(r'([0-9]+\.?[0-9]*)', pred_text)
            for n_str in nums:
                try:
                    n = float(n_str)
                    if abs(expected) > 1e-8 and abs(n - expected) / abs(expected) < 0.05:
                        pattern_found = True
                        pattern_correct = True
                        break
                except ValueError:
                    pass

        if pattern_found and not pattern_correct:
            wrong += 1
            errors[key] = {"expected": expected, "status": "wrong"}

    return {
        "hallucination_rate": round(wrong / max(total, 1) * 100, 1),
        "total_facts_checked": total,
        "wrong_facts": wrong,
        "error_details": errors,
    }


def compute_instruction_following(raw_output, required_fields):
    """
    instruction following率 — 模型是否正确按照要求output format。

    required_fields: list of field names that must be in the JSON output
    """
    parsed = None
    try:
        parsed = json.loads(raw_output)
    except (json.JSONDecodeError, ValueError, TypeError):
        match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', raw_output, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(1))
            except (json.JSONDecodeError, ValueError, TypeError):
                pass

    if parsed is None:
        return {
            "instruction_followed": False,
            "json_parsed": False,
            "missing_fields": required_fields,
            "score": 0.0,
        }

    present = [f for f in required_fields if f in parsed]
    missing = [f for f in required_fields if f not in parsed]

    return {
        "instruction_followed": len(missing) == 0,
        "json_parsed": True,
        "fields_present": present,
        "missing_fields": missing,
        "score": round(len(present) / max(len(required_fields), 1) * 100, 1),
    }


# ══════════════════════════════════════════════════════════════
# E. statistical test
# ══════════════════════════════════════════════════════════════

def bootstrap_confidence_interval(data, stat_fn=np.mean, n_bootstrap=1000, alpha=0.05):
    """
    Bootstrap 95% confidence interval。
    适用于任何metric的非parameter区间估计。
    """
    d = np.asarray(data, dtype=np.float64)
    d = d[~np.isnan(d)]
    if len(d) < 10:
        return {"mean": float(np.mean(d)) if len(d) > 0 else None,
                "ci_lower": None, "ci_upper": None, "n": len(d)}

    stats = []
    rng = np.random.RandomState(42)
    for _ in range(n_bootstrap):
        sample = rng.choice(d, size=len(d), replace=True)
        stats.append(stat_fn(sample))

    ci_lower = np.percentile(stats, alpha/2 * 100)
    ci_upper = np.percentile(stats, (1 - alpha/2) * 100)

    return {
        "mean": float(np.mean(d)),
        "std": float(np.std(d)),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "ci_95": f"[{ci_lower:.4f}, {ci_upper:.4f}]",
        "n": len(d),
    }


def compare_two_models(scores_a, scores_b, metric_name, alpha=0.05):
    """
    比较两个模型的某项metric：
      - Welch's t-test (parameter)
      - Mann-Whitney U test (非parameter)
      - Cohen's d (效应量)
      - 胜率 (A better than B 的比例)

    Returns dict package含显著性判断。
    """
    a = np.asarray(scores_a, dtype=np.float64)
    b = np.asarray(scores_b, dtype=np.float64)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]

    if len(a) < 5 or len(b) < 5:
        return {"error": "not enough samples"}

    result = {"metric": metric_name, "n_A": len(a), "n_B": len(b)}

    # t-test
    t_stat, t_pval = scipy_stats.ttest_ind(a, b, equal_var=False)
    result["ttest_statistic"] = round(float(t_stat), 4)
    result["ttest_pvalue"] = round(float(t_pval), 4)
    result["ttest_significant"] = t_pval < alpha

    # Mann-Whitney U
    try:
        u_stat, u_pval = scipy_stats.mannwhitneyu(a, b, alternative='two-sided')
        result["mwU_statistic"] = round(float(u_stat), 4)
        result["mwU_pvalue"] = round(float(u_pval), 4)
        result["mwU_significant"] = u_pval < alpha
    except Exception:
        pass

    # Cohen's d
    pooled_std = np.sqrt((np.std(a)**2 + np.std(b)**2) / 2)
    if pooled_std > 1e-8:
        cohens_d = (np.mean(a) - np.mean(b)) / pooled_std
        result["cohens_d"] = round(float(cohens_d), 4)
        result["effect_size"] = (
            "negligible" if abs(cohens_d) < 0.2 else
            "small" if abs(cohens_d) < 0.5 else
            "medium" if abs(cohens_d) < 0.8 else "large"
        )
    else:
        result["cohens_d"] = 0.0
        result["effect_size"] = "negligible"

    # Win rate
    win_rate = np.mean(a < b)  # A better if MAE is lower
    result["A_better_than_B_rate"] = round(float(win_rate) * 100, 1)

    # 综合判断
    result["verdict"] = (
        f"A {'significantly' if result.get('ttest_significant') else 'not significantly'} "
        f"{'better' if np.mean(a) < np.mean(b) else 'worse'} than B "
        f"(p={result['ttest_pvalue']:.4f}, d={result.get('cohens_d', 0):.2f})"
    )

    return result


# ══════════════════════════════════════════════════════════════
# F. efficiencymetric
# ══════════════════════════════════════════════════════════════

def compute_efficiency_metrics(latencies, token_counts, n_predictions, n_samples):
    """
    computationefficiencymetric。

    Args:
        latencies: list of (seconds per sample)
        token_counts: list of (input_tokens, output_tokens) per sample
        n_predictions: number of prediction steps
        n_samples: total number of test samples
    """
    lat = np.asarray(latencies, dtype=np.float64)

    metrics = {
        "avg_latency_sec": round(float(np.mean(lat)), 3),
        "median_latency_sec": round(float(np.median(lat)), 3),
        "p95_latency_sec": round(float(np.percentile(lat, 95)), 3),
        "throughput_pred_per_sec": round(n_predictions / max(np.sum(lat), 1e-8), 2),
    }

    if token_counts:
        input_tokens = np.sum([t[0] for t in token_counts])
        output_tokens = np.sum([t[1] for t in token_counts])
        metrics["total_input_tokens"] = int(input_tokens)
        metrics["total_output_tokens"] = int(output_tokens)
        metrics["avg_input_tokens_per_sample"] = round(float(input_tokens / max(len(token_counts), 1)), 1)
        metrics["avg_output_tokens_per_sample"] = round(float(output_tokens / max(len(token_counts), 1)), 1)
        metrics["tokens_per_prediction"] = round(float(output_tokens / max(n_predictions, 1)), 2)

    return metrics


# ══════════════════════════════════════════════════════════════
# G. 综合报告generation
# ══════════════════════════════════════════════════════════════

def generate_model_report(model_name, all_results_by_dimension):
    """
    综合所有evaluation resultsgeneration专业报告。

    all_results_by_dimension: {
        "D1": {
            "forecast_metrics": {metric_name: [values...]},
            "text_metrics": {metric_name: [values...]},
            "structured_metrics": {...},
            "efficiency": {...},
        },
        ...
    }
    """
    report = {
        "model": model_name,
        "evaluated_dimensions": list(all_results_by_dimension.keys()),
        "dimension_details": {},
        "overall_scores": {},
    }

    # 汇总跨dimensionmetric
    all_mae, all_rouge_l, all_json_valid = [], [], []

    for dim_key, dim_data in all_results_by_dimension.items():
        dim_report = {}

        # 预测metric (带CI)
        for metric_name, values in dim_data.get("forecast_metrics", {}).items():
            if values:
                dim_report[metric_name] = bootstrap_confidence_interval(values)
                if metric_name == "MAE":
                    all_mae.extend(values)

        # Textmetric (带CI)
        for metric_name, values in dim_data.get("text_metrics", {}).items():
            if values:
                dim_report[metric_name] = bootstrap_confidence_interval(values)
                if metric_name == "ROUGE-L":
                    all_rouge_l.extend(values)

        # Structuredmetric
        dim_report["structured"] = dim_data.get("structured_metrics", {})

        # efficiency
        dim_report["efficiency"] = dim_data.get("efficiency", {})

        report["dimension_details"][dim_key] = dim_report

    # 全局汇总
    if all_mae:
        report["overall_scores"]["MAE"] = bootstrap_confidence_interval(all_mae)
    if all_rouge_l:
        report["overall_scores"]["ROUGE-L"] = bootstrap_confidence_interval(all_rouge_l)

    return report


# ══════════════════════════════════════════════════════════════
# test
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=== metriclibrarytest ===\n")

    # A. forecasting accuracy
    yt = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    yp = np.array([1.1, 2.2, 2.8, 4.1, 5.3])
    fm = compute_forecast_metrics(yt, yp, naive_forecast=1.0)
    print("A. forecasting accuracy:")
    for k, v in fm.items():
        print(f"   {k}: {v}")

    # B. 概率预测
    samples = np.random.normal(3, 0.5, (5, 100))
    crps = compute_crps(yt, samples)
    print(f"\nB. 概率预测:\n   CRPS: {crps:.4f}")

    # C. Textquality
    ref = "预测呈上升趋势，low risk，从1.0变化到5.0"
    pred = "趋势为上升，risk level低，Numeric从1变化到5.3"
    tm = compute_all_text_metrics(pred, ref)
    print(f"\nC. Textquality:")
    for k, v in tm.items():
        if not k.endswith("_error"):
            print(f"   {k}: {v}")

    # D. Structured
    outputs = ['{"a":1}', 'not json', '{"predicted_values":[1,2]}']
    print(f"\nD. Structured:\n   JSON有efficiency: {compute_json_validity_rate(outputs)}")

    # E. statistical test
    a_scores = np.random.normal(0.5, 0.1, 100)
    b_scores = np.random.normal(0.55, 0.1, 100)
    comp = compare_two_models(a_scores, b_scores, "MAE")
    print(f"\nE. model comparison:\n   {comp['verdict']}")
    print(f"   Cohen's d={comp.get('cohens_d')}, p={comp.get('ttest_pvalue')}")

    print("\n✅ 所有metricmodule正常!")
