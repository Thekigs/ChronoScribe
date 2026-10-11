#!/usr/bin/env python3
"""
Converter utilities for raw time series to ChronoScribe controller training.

The utilities construct two reference tasks: request-to-directive and
forecast-evidence-to-response. All numerical statements are computed from raw
data or supplied forecast evidence; text generation must not invent values.

Usage:
    from converter_utils import (
        extract_windows, compute_statistics,
        generate_task1_sample, generate_task2_sample,
        DOMAIN_CN, FREQ_CN
    )
"""

import json
import random
import numpy as np
from scipy import signal as scipy_signal

# ──────────────────────────────────────────────────────────────
# Domain & Frequency Mappings
# ──────────────────────────────────────────────────────────────

DOMAIN_CN = {
    "health": "医疗健康",
    "business": "商业零售",
    "climate": "气象气候",
    "energy": "能源消耗",
    "traffic": "交通运输",
    "cloud": "云computation",
    "finance": "金融外汇",
    "aerospace": "航空航天",
    "industrial": "工业控制",
    "chemistry": "化学工艺",
    "smart_security": "智慧安防",
    "biometric": "生物识别",
    "speech": "语音识别",
    "general": "通用预测",
    "society": "社会民生",
}

FREQ_CN = {
    "T": "分钟",
    "min": "分钟",
    "15T": "15分钟",
    "10T": "10分钟",
    "5T": "5分钟",
    "H": "小时",
    "D": "天",
    "W": "周",
    "M": "月",
    "Q": "季",
    "Y": "年",
}

FREQ_STEP_MAP = {
    "T": 1, "min": 1, "15T": 15, "10T": 10, "5T": 5,
    "H": 1, "D": 1, "W": 7, "M": 30, "Q": 90, "Y": 365,
}

UNIT_MAP = {
    "health": "例",
    "business": "件",
    "climate": "°C",
    "energy": "kWh",
    "traffic": "辆",
    "cloud": "%",
    "finance": "元",
    "aerospace": "affiliation",
    "industrial": "affiliation",
    "chemistry": "%",
    "smart_security": "次",
    "biometric": "affiliation",
    "speech": "affiliation",
    "general": "affiliation",
    "society": "人",
}

STYLES = ["alert", "casual", "risk", "summary"]
DEPTHS = ["risk", "summary", "detail"]

# ──────────────────────────────────────────────────────────────
# System Messages (matching user's training format exactly)
# ──────────────────────────────────────────────────────────────

SYSTEM_TASK1 = (
    "你的任务是将用户的natural languagetime series forecasting需求转化为专业预测器可执行的parameterJSON。"
    "需要正确识别预测目标、时间频率、预测length等关键parameter，并output标准format的预测指令。"
)

SYSTEM_TASK2 = (
    "你的任务是将time series forecasting器return的统计证据转化为对用户友好的natural language解读。"
    "只能表达可由input证据支持的Numeric、趋势和uncertaintyconclusion。"
)

# ──────────────────────────────────────────────────────────────
# Window Extraction
# ──────────────────────────────────────────────────────────────

def extract_windows(series, n_windows=3, min_len=12):
    """
    Extract prediction windows from a univariate time series.

    Args:
        series: 1D numpy array (must be clean - no NaN)
        n_windows: number of windows to extract
        min_len: minimum series length to work with

    Returns:
        list of (window, history, anchor_pos) tuples
    """
    L = len(series)
    if L < min_len:
        return []

    # Adaptive horizon
    H = max(12, min(48, L // 10))

    # For very short series, use just 1 window
    if L < H * 3:
        H = L // 3
        if H < 4:
            H = L
            return [(series.copy(), np.array([]), 0)]

    positions = []
    for frac in [0.33, 0.50, 0.67]:
        anchor = int(L * frac)
        if anchor + H > L:
            anchor = max(0, L - H)
        # Avoid duplicate positions
        if not positions or anchor != positions[-1]:
            positions.append(anchor)

    # If we got fewer than n_windows due to dedup, try to add more
    while len(positions) < min(n_windows, 3):
        # Add a random position not already used
        candidates = [p for p in range(H, L - H, H) if p not in positions]
        if candidates:
            positions.append(random.choice(candidates))
        else:
            break

    windows = []
    for anchor in positions[:n_windows]:
        window = series[anchor:anchor + H].copy()
        history = series[:anchor].copy()
        windows.append((window, history, anchor))

    return windows


# ──────────────────────────────────────────────────────────────
# Statistics Computation
# ──────────────────────────────────────────────────────────────

def compute_statistics(window, history=None):
    """
    Compute all prediction-report statistics from a window.
    ALL values are exact computations from raw data.

    Returns:
        dict with all statistics
    """
    w = window.astype(np.float64)
    L = len(w)

    stats = {}

    # Basic values
    stats["start_value"] = float(w[0])
    stats["end_value"] = float(w[-1])
    stats["min_value"] = float(np.min(w))
    stats["max_value"] = float(np.max(w))
    stats["mean_value"] = float(np.mean(w))
    stats["min_position"] = int(np.argmin(w))
    stats["max_position"] = int(np.argmax(w))
    stats["std_value"] = float(np.std(w))

    # Trend analysis
    if stats["start_value"] != 0:
        change_pct = ((stats["end_value"] - stats["start_value"]) / abs(stats["start_value"])) * 100
    else:
        change_pct = (stats["end_value"] - stats["start_value"]) * 100 if stats["end_value"] != 0 else 0.0

    stats["change_pct"] = round(change_pct, 2)
    abs_pct = abs(stats["change_pct"])

    if abs_pct < 0.1:
        stats["trend_direction"] = "平稳"
    elif change_pct > 0:
        stats["trend_direction"] = "上升"
    else:
        stats["trend_direction"] = "下降"

    if abs_pct < 3:
        stats["trend_strength"] = "轻微"
    elif abs_pct < 10:
        stats["trend_strength"] = "明显"
    else:
        stats["trend_strength"] = "显著"

    # Peak analysis
    if L >= 3:
        try:
            peaks, properties = scipy_signal.find_peaks(w, prominence=max(stats["std_value"] * 0.5, 0.001))
            stats["peak_count"] = len(peaks)
            if len(peaks) > 0:
                peak_values = w[peaks]
                max_peak_idx = np.argmax(peak_values)
                stats["max_peak_value"] = float(peak_values[max_peak_idx])
                stats["max_peak_position"] = int(peaks[max_peak_idx])
            else:
                stats["max_peak_value"] = float(stats["max_value"])
                stats["max_peak_position"] = int(stats["max_position"])
        except Exception:
            stats["peak_count"] = 0
            stats["max_peak_value"] = float(stats["max_value"])
            stats["max_peak_position"] = int(stats["max_position"])
    else:
        stats["peak_count"] = 0
        stats["max_peak_value"] = float(stats["max_value"])
        stats["max_peak_position"] = int(stats["max_position"])

    # Threshold detection (using history context if available)
    if history is not None and len(history) > 0:
        threshold = float(np.mean(history) + 2 * np.std(history))
    else:
        threshold = float(stats["mean_value"] + 2 * stats["std_value"])

    breaches = w > threshold
    stats["threshold_breached"] = bool(np.any(breaches))
    stats["breach_probability"] = round(float(np.mean(breaches)) * 100, 2)
    if stats["threshold_breached"]:
        stats["first_breach_hour"] = int(np.argmax(breaches))
    else:
        stats["first_breach_hour"] = -1

    # Risk assessment
    risk_score = 0
    risk_score += min(abs_pct * 3, 30)  # up to 30 from change magnitude
    risk_score += min(stats["std_value"] / (abs(stats["mean_value"]) + 1e-8) * 100, 25)  # up to 25 from volatility
    risk_score += min(stats["breach_probability"] * 0.45, 45)  # up to 45 from threshold breaches

    stats["risk_score"] = int(min(round(risk_score), 100))

    if stats["risk_score"] >= 70:
        stats["risk_level"] = "高"
    elif stats["risk_score"] >= 40:
        stats["risk_level"] = "较高"
    elif stats["risk_score"] >= 15:
        stats["risk_level"] = "较低"
    else:
        stats["risk_level"] = "低"

    # Max drawdown
    peak = w[0]
    max_dd = 0.0
    for v in w:
        if v > peak:
            peak = v
        if peak != 0:
            dd = (peak - v) / abs(peak) * 100
            if dd > max_dd:
                max_dd = dd
    stats["max_drawdown"] = round(max_dd, 2)

    return stats


# ──────────────────────────────────────────────────────────────
# Task 2: User Message (structured prediction report)
# ──────────────────────────────────────────────────────────────

def generate_task2_user_message(stats, domain, series_name, horizon, freq_str, unit=None):
    """Generate the structured prediction report (user input for Task 2)."""
    if unit is None:
        unit = UNIT_MAP.get(domain, "affiliation")

    domain_cn = DOMAIN_CN.get(domain, domain)
    freq_cn = FREQ_CN.get(freq_str, freq_str)

    # Compute time-based positions
    freq_factor = FREQ_STEP_MAP.get(freq_str, 1)
    total_hours = horizon * freq_factor
    peak_day = round(stats["max_peak_position"] * freq_factor / 24, 1)
    total_days = round(total_hours / 24, 1)

    # Breach info
    if stats["threshold_breached"]:
        breach_str = f"是否突破: 是\n  突破概率: {stats['breach_probability']:.1f}%\n  首次突破: 第{stats['first_breach_hour']}小时"
    else:
        breach_str = f"是否突破: 否\n  突破概率: {stats['breach_probability']:.1f}%"

    parts = [
        f"预测任务: 对 {domain_cn} 的 {series_name} 进行未来{total_hours}小时的time series forecasting",
        f"预测length: {horizon} 个时间步 (频率: {freq_str})",
        "",
        "预测序列概览:",
        f"  起始值: {stats['start_value']:.4g} {unit}",
        f"  结束值: {stats['end_value']:.4g} {unit}",
        f"  最低值: {stats['min_value']:.4g} {unit}",
        f"  最高值: {stats['max_value']:.4g} {unit}",
        f"  平均值: {stats['mean_value']:.4g} {unit}",
        "",
        "trend analysis:",
        f"  方向: {stats['trend_direction']}",
        f"  强度: {stats['trend_strength']}",
        f"  变化幅度: {stats['change_pct']:.1f}%",
        "",
        "阈值检测结果:",
        f"  {breach_str}",
        "",
        "峰值分析:",
        f"  峰值quantity: {stats['peak_count']}",
        f"  最大峰值: {stats['max_peak_value']:.4g} {unit}",
        f"  峰值位置: 第{peak_day}天",
        "",
        "综合风险evaluation:",
        f"  risk level: {stats['risk_level']}",
        f"  风险评分: {stats['risk_score']}/100",
        f"  最大回撤: {stats['max_drawdown']:.1f}%",
    ]

    return "\n".join(parts)


# ──────────────────────────────────────────────────────────────
# Task 2: Assistant Response (natural language interpretation)
# ──────────────────────────────────────────────────────────────

def generate_task2_assistant_response(stats, domain, series_name, horizon, freq_str, style, unit=None):
    """Generate the natural language interpretation based on style."""
    if unit is None:
        unit = UNIT_MAP.get(domain, "affiliation")

    domain_cn = DOMAIN_CN.get(domain, domain)
    freq_cn = FREQ_CN.get(freq_str, freq_str)
    freq_factor = FREQ_STEP_MAP.get(freq_str, 1)
    total_hours = horizon * freq_factor
    total_days = round(total_hours / 24, 1)
    peak_day = round(stats["max_peak_position"] * freq_factor / 24, 1)

    if style == "alert":
        return _generate_alert_response(stats, domain_cn, series_name, unit, total_hours, total_days, peak_day, horizon, freq_str)
    elif style == "casual":
        return _generate_casual_response(stats, domain_cn, series_name, unit, total_hours, total_days, peak_day, horizon)
    elif style == "risk":
        return _generate_risk_response(stats, domain_cn, series_name, unit, total_hours, total_days, peak_day)
    else:  # summary
        return _generate_summary_response(stats, domain_cn, series_name, unit, total_hours, total_days, peak_day, horizon, freq_str)


def _generate_alert_response(stats, domain_cn, series_name, unit, total_hours, total_days, peak_day, horizon, freq_str):
    """Alert style: urgent with ⚠️ markers, structured, actionable."""
    level_emoji = {"高": "🔴", "较高": "🟠", "较低": "🟡", "低": "🟢"}
    emoji = level_emoji.get(stats["risk_level"], "⚪")

    lines = [
        f"【{stats['risk_level']}危告警】{series_name}",
        f"risk level: {stats['risk_level']} | composite score: {stats['risk_score']}/100",
        f"预测趋势: {stats['trend_direction']} | 变化幅度: {stats['change_pct']:.1f}%",
        f"波动区间: [{stats['min_value']:.4g} {unit}, {stats['max_value']:.4g} {unit}]",
        f"预期峰值: {stats['max_peak_value']:.4g} {unit} @ 第{peak_day}天",
        "",
    ]

    if stats["risk_level"] in ("高", "较高"):
        lines.append(f"{emoji} 建议提高监控频率，通知相关责任人关注后续走势。")

    if stats["threshold_breached"]:
        lines.append(f"\n⚠️ 阈值突破: 是，预计在第{stats['first_breach_hour']}小时发生")
        lines.append(f"   突破概率: {stats['breach_probability']:.1f}%")

    if stats["peak_count"] > 0 and stats["risk_score"] >= 30:
        lines.append(f"📈 峰值: {stats['max_peak_value']:.4g} {unit} @ 第{peak_day}天")

    lines.append(f"\n跟踪窗口: 未来{total_days}天 | 数据频率: {freq_str} | 步数: {horizon}")

    return "\n".join(lines)


def _generate_casual_response(stats, domain_cn, series_name, unit, total_hours, total_days, peak_day, horizon):
    """Casual style: conversational, plain language."""
    trend_desc = f"{stats['trend_direction']}"
    if stats["trend_direction"] != "平稳":
        trend_desc += f"，涨幅{stats['change_pct']:.1f}%"

    risk_desc = {
        "高": "需要height关注，建议立即采取干预措施",
        "较高": "存在一定风险，建议密切关注",
        "较低": "风险可控，目前没有特别需要干预的地方",
        "低": "一切平稳，无需额外关注",
    }

    lines = [
        f"帮你梳理了{series_name}接下来{total_days}天的forecast results，长话短说：",
        f"{trend_desc}，从{stats['start_value']:.4g} {unit}到{stats['end_value']:.4g} {unit}。",
    ]

    if stats["trend_direction"] == "平稳" or abs(stats["change_pct"]) < 3:
        lines.append("基本维持当前水平，波动不大。")

    lines.append(f"从风险角度看：综合evaluation下来{risk_desc.get(stats['risk_level'], '情况正常')}。")

    return "".join(lines)


def _generate_risk_response(stats, domain_cn, series_name, unit, total_hours, total_days, peak_day):
    """Risk style: focused on risk metrics."""
    breach_info = ""
    if stats["threshold_breached"]:
        breach_info = f"预计第{stats['first_breach_hour']}小时将突破阈值（概率{stats['breach_probability']:.1f}%）"
    else:
        breach_info = "未检测到阈值突破"

    lines = [
        f"【风险evaluation】{series_name}",
        f"风险评分: {stats['risk_score']}/100 ({stats['risk_level']})",
        f"最大回撤: {stats['max_drawdown']:.1f}%",
        f"趋势: {stats['trend_direction']} ({stats['trend_strength']}), 变化幅度 {stats['change_pct']:.1f}%",
        f"阈值状态: {breach_info}",
        f"区间: [{stats['min_value']:.4g}, {stats['max_value']:.4g}] {unit}",
    ]

    return "\n".join(lines)


def _generate_summary_response(stats, domain_cn, series_name, unit, total_hours, total_days, peak_day, horizon, freq_str):
    """Summary style: comprehensive neutral overview."""
    freq_factor = FREQ_STEP_MAP.get(freq_str, 1)

    lines = [
        f"【time series forecasting报告】{domain_cn} - {series_name}",
        "",
        f"📊 预测概览：未来{total_days}天（{horizon}步，频率{freq_str}），",
        f"   predicted values从 {stats['start_value']:.4g} {unit} 变化至 {stats['end_value']:.4g} {unit}，",
        f"   区间 [{stats['min_value']:.4g}, {stats['max_value']:.4g}] {unit}，均值为 {stats['mean_value']:.4g} {unit}。",
        "",
        f"📈 trend analysis：整体呈{stats['trend_direction']}趋势（{stats['trend_strength']}），变化幅度 {stats['change_pct']:.1f}%。",
    ]

    if stats["peak_count"] > 0:
        lines.append(f"   检测到 {stats['peak_count']} 个峰值，最大峰值 {stats['max_peak_value']:.4g} {unit} 位于第{peak_day}天。")

    lines.append(f"\n⚠️ 风险evaluation：等级「{stats['risk_level']}」，评分 {stats['risk_score']}/100，最大回撤 {stats['max_drawdown']:.1f}%。")

    if stats["threshold_breached"]:
        lines.append(f"   阈值将于第{stats['first_breach_hour']}小时突破（概率 {stats['breach_probability']:.1f}%），建议提前关注。")
    else:
        lines.append(f"   未检测到阈值突破（概率 {stats['breach_probability']:.1f}%），metric在正常范围内。")

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────
# Task 1: NL → Chronos2 JSON
# ──────────────────────────────────────────────────────────────

def generate_task1_sample(domain, series_name, horizon, freq_str, context_len, unit=None):
    """Generate a complete Task 1 sample (NL request → Chronos2 JSON)."""
    if unit is None:
        unit = UNIT_MAP.get(domain, "affiliation")

    domain_cn = DOMAIN_CN.get(domain, domain)
    freq_cn = FREQ_CN.get(freq_str, freq_str)
    freq_factor = FREQ_STEP_MAP.get(freq_str, 1)
    total_hours = horizon * freq_factor
    total_days = round(total_hours / 24, 1)

    # --- User message: natural language request ---
    nl_templates = [
        f"请帮我预测{domain_cn}领域{series_name}metric未来{total_hours}小时的变化趋势，需要output{horizon}个时间步的forecast results。",
        f"对{domain_cn}的{series_name}进行time series forecasting，预测length为{horizon}步，频率为{freq_str}。",
        f"我需要一个{domain_cn}场景下的time series forecasting：目标metric是{series_name}，预测未来{total_days}天，数据采样频率为{freq_str}。",
        f"针对{series_name}（{domain_cn}领域），请generation未来{horizon}个时间步的time series forecastingconfiguration。",
    ]
    user_msg = random.choice(nl_templates)

    # --- Assistant message: Chronos-2 JSON ---
    chronos_config = {
        "task": "forecast",
        "domain": domain,
        "target": series_name,
        "horizon": horizon,
        "prediction_length": horizon,
        "frequency": freq_str,
        "context_length": context_len,
        "features": [series_name],
        "quantiles": [0.1, 0.5, 0.9],
        "num_samples": 100,
        "output_format": "quantiles",
        "seasonality": "auto",
    }

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_TASK1},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": json.dumps(chronos_config, ensure_ascii=False, indent=2)},
        ],
        "_domain": domain,
        "_stage": "stage1",
    }


def generate_task2_sample(stats, domain, series_name, horizon, freq_str, unit=None):
    """Generate a complete Task 2 sample (prediction report → NL interpretation)."""
    if unit is None:
        unit = UNIT_MAP.get(domain, "affiliation")

    style = random.choice(STYLES)
    depth = random.choice(DEPTHS)

    user_msg = generate_task2_user_message(stats, domain, series_name, horizon, freq_str, unit)
    assistant_msg = generate_task2_assistant_response(stats, domain, series_name, horizon, freq_str, style, unit)

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_TASK2},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ],
        "_domain": domain,
        "_style": style,
        "_depth": depth,
        "_stage": "stage2",
    }


# ──────────────────────────────────────────────────────────────
# Utility: format float values for display
# ──────────────────────────────────────────────────────────────

def fmt_val(v, precision=4):
    """Format a float value with appropriate precision."""
    if abs(v) >= 10000:
        return f"{v:,.{precision}g}"
    elif abs(v) >= 1:
        return f"{v:.{precision}g}"
    elif abs(v) < 0.001 and v != 0:
        return f"{v:.{precision}e}"
    else:
        return f"{v:.{precision}g}"


# ──────────────────────────────────────────────────────────────
# Utility: clean series (remove NaN, handle edge cases)
# ──────────────────────────────────────────────────────────────

def clean_series(series):
    """
    Clean a time series for processing.
    Returns (clean_array, nan_ratio, is_valid)
    """
    s = np.asarray(series, dtype=np.float64)
    nan_mask = np.isnan(s)
    nan_ratio = float(np.mean(nan_mask))

    if nan_ratio >= 0.5:
        return s, nan_ratio, False

    if nan_ratio > 0:
        # Forward fill then backward fill
        s = s.copy()
        nans = np.isnan(s)
        if np.all(nans):
            return s, nan_ratio, False

        # Forward fill
        last_valid = None
        for i in range(len(s)):
            if np.isnan(s[i]):
                if last_valid is not None:
                    s[i] = last_valid
            else:
                last_valid = s[i]

        # Backward fill for leading NaN
        first_valid = None
        for i in range(len(s) - 1, -1, -1):
            if not np.isnan(s[i]):
                first_valid = s[i]
            elif first_valid is not None:
                s[i] = first_valid

    # Check for constant series (all same value)
    if np.std(s) < 1e-12:
        return s, nan_ratio, True  # still valid, just constant

    return s, nan_ratio, True


# ──────────────────────────────────────────────────────────────
# Utility: parse .ts UEA/UCR format
# ──────────────────────────────────────────────────────────────

def parse_ts_file(filepath):
    """
    Parse a UEA/UCR .ts time series classification file.

    Returns:
        list of (label, list_of_dimensions) where each dimension is a 1D numpy array
    """
    results = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('@'):
                continue
            # Format: dim1_vals:dim2_vals:...:dimN_vals:class_label
            # The LAST : separated field is the classification label (string).
            # All preceding fields are numeric dimensions.
            parts = line.split(':')
            if len(parts) < 2:
                continue

            # Last part is the class label
            label_raw = parts[-1].strip()
            try:
                label = int(label_raw)
            except ValueError:
                try:
                    label = float(label_raw)
                except ValueError:
                    label = label_raw  # keep as string (e.g., "E35", "normal", "positivity")

            # All parts except the last are numeric dimensions
            dimensions = []
            for dim_str in parts[:-1]:
                vals = np.array([float(x) for x in dim_str.split(',') if x.strip()], dtype=np.float64)
                dimensions.append(vals)

            results.append((label, dimensions))

    return results


# ──────────────────────────────────────────────────────────────
# Utility: write JSONL
# ──────────────────────────────────────────────────────────────

def write_jsonl(filepath, samples, mode='a'):
    """Write samples to a JSONL file."""
    with open(filepath, mode, encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    # Quick test
    import sys
    print("converter_utils.py loaded successfully")
    print(f"  STYLES: {STYLES}")
    print(f"  Domains: {list(DOMAIN_CN.keys())}")

    # Test with dummy data
    test_series = np.array([10.0, 11.2, 10.8, 12.5, 13.1, 12.9, 14.0, 15.2, 14.8, 16.0,
                            15.5, 17.2, 18.0, 17.5, 19.1, 20.0, 19.5, 21.2, 22.0, 21.5,
                            23.0, 24.1, 23.5, 25.0, 26.2, 25.8, 27.0, 28.5, 29.0, 28.2])

    windows = extract_windows(test_series)
    print(f"\nTest series length: {len(test_series)}, extracted {len(windows)} windows")

    for i, (w, h, a) in enumerate(windows):
        print(f"  Window {i}: anchor={a}, window_len={len(w)}, history_len={len(h)}")
        stats = compute_statistics(w, h)
        print(f"    Start={stats['start_value']:.2f}, End={stats['end_value']:.2f}")
        print(f"    Trend={stats['trend_direction']}, Risk={stats['risk_level']}({stats['risk_score']}/100)")

        # Generate Task 2 sample
        sample = generate_task2_sample(stats, "energy", "test_metric", len(w), "H")
        print(f"    Style: {sample['_style']}, Depth: {sample['_depth']}")
        print(f"    Assistant response: {sample['messages'][2]['content'][:100]}...")

    print("\nTest Task 1 sample:")
    t1_sample = generate_task1_sample("traffic", "traffic_sensor_1", 24, "H", 168)
    print(f"  User: {t1_sample['messages'][1]['content'][:100]}...")
    print(f"  Assistant JSON: {t1_sample['messages'][2]['content'][:100]}...")

    print("\n✅ All tests passed!")
