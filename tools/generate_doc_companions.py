#!/usr/bin/env python3
"""Create Chinese companions for local markdown artifacts.

Core project documentation has hand-written translations. This utility fills
the remaining generated scorecards and upstream model/data cards without
altering their numerical content or upstream English source files.
"""
from __future__ import annotations

from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
COMMON = {
    "ChronoScribe Evaluation Suite Scorecard": "ChronoScribe 评测套件 Scorecard",
    "Generated": "生成时间",
    "Benchmark": "Benchmark",
    "Results": "结果目录",
    "Components": "组件",
    "Dimension": "维度",
    "Component": "组件",
    "Key metrics": "关键指标",
    "Scoring Notes": "评分说明",
    "uncertainty": "不确定性",
    "可再生能源": "可再生能源",
    "short-term forecasting": "短期预测",
    "long horizon": "长期预测",
    "long-term forecasting": "长期预测",
    "multivariate": "多变量",
    "zero-shot": "零样本",
    "few-shot": "少样本",
    "多variable预测": "多变量预测",
    "零sample泛化": "零样本泛化",
    "小sample泛化": "小样本泛化",
    "robustness": "鲁棒性",
    "outlier robustness": "异常值鲁棒性",
    "missing robustness": "缺失鲁棒性",
    "Score": "分数",
    "分数card": "评分卡",
    "TLS": "TLS",
    "Scorecard": "评分卡",
    "- D10 uses runner-provided P10/P50/P90 quantiles for coverage, CRPS, and calibration.": "- D10 使用 runner 提供的 P10/P50/P90 分位数计算覆盖率、CRPS 和校准误差。",
    "- D13 measures available cause labels from structured output or declared analysis fields.": "- D13 根据结构化输出或声明的分析字段测量可用的原因标签。",
    "- Robustness, transfer, and attribution dimensions use their declared derived metrics.": "- 鲁棒性、迁移和归因维度使用各自声明的派生指标。",
    "- D17 records runner-provided GPU memory when available.": "- D17 在可用时记录 runner 提供的 GPU 显存。",
    "> This generated scorecard is protocol-bound. For current forecast-only claims, use `scores/operational_forecasting_verified_20261003.json` and `scores/experiment_registry_20261003.json`.": "> 本评分卡由程序生成并绑定到特定协议。当前纯预测结论请使用 `scores/operational_forecasting_verified_20261003.json` 和 `scores/experiment_registry_20261003.json`。",
}


def translate_scorecard(text: str, path: Path) -> str:
    # Keep protocol identifiers and inline paths byte-for-byte identical so
    # that the Chinese companion remains searchable against the registry.
    text = text.replace(
        "> This generated scorecard is protocol-bound. For current forecast-only claims, use",
        "> 本评分卡由程序生成并绑定到特定协议。当前纯预测结论请使用",
    )
    protected: list[str] = []

    def protect(match: re.Match[str]) -> str:
        protected.append(match.group(0))
        return f"\x00{len(protected) - 1}\x00"

    out = re.sub(r"`[^`]+`|\bD\d+_[A-Za-z0-9_]+\b", protect, text)
    for src, dst in COMMON.items():
        out = out.replace(src, dst)
    out = re.sub(r"\x00(\d+)\x00", lambda m: protected[int(m.group(1))], out)
    out = out.replace("` and `", "` 和 `").replace("`.", "`。")
    header = (
        "\n> 中文伴随版：本文件保留对应英文 scorecard 的数值、样本数和协议字段。"
        "由于 `scores/` 同时包含历史运行和当前运行，论文主表必须以"
        " `scores/experiment_registry_20261003.json` 为准。\n"
    )
    return "# 中文版\n" + header + "\n" + out


def model_note(path: Path) -> str:
    name = path.parent.name
    mapping = {
        "Qwen3.5-9B": "Qwen3.5-9B 语言控制器",
        "chronos-2": "Chronos-2 时序基础模型",
        "sundial-base-128m": "Sundial 时序基础模型",
        "aurora": "Aurora 多模态时序基础模型",
        "checkpoint-7500": "ChronoScribe LoRA adapter",
    }
    title = mapping.get(name, name)
    return f"""# {title}（中文本地说明）

本目录保存 `{name}` 的本地模型卡或 adapter 说明。上游模型卡的英文内容保留在同目录的 `README.md` 中；本文件补充它在 ChronoScribe 项目中的实际角色和当前协议边界。

## 在 ChronoScribe 中的角色

- Qwen3.5-9B：负责请求解释、结构化 directive 和证据条件回答。
- Chronos-2：参考冻结时序模型，负责产生预测轨迹和分位数。
- Sundial：主比较中的 specialist baseline，通过统一 adapter 接入。
- Aurora：额外的强模型边界案例；当前结果显示它在不同指标间存在明显取舍，不能单独作为总体优势结论。
- checkpoint-7500：本地 LoRA adapter。它不包含公开训练语料，完整训练复现需要单独提供数据 provenance、split、LoRA 参数和随机种子。

## 评测约束

模型只能接收目标之前的历史窗口。所有 paired 运行使用 `benchmark/evaluation_manifest.json`，目标预测长度由 benchmark 固定。替换 specialist 时必须保留数值来源、预测长度、输出格式和 evidence contract。

## 结果说明

当前主预测协议见 [`benchmark/evaluation_manifest.json`](../../benchmark/evaluation_manifest.json) 和 [`docs/SCORING.md`](../../docs/SCORING.md)。运行生成的 scorecard 只用于追溯，不能脱离运行协议和样本覆盖范围进行论文比较。
"""


def dataset_note(path: Path) -> str:
    return """# GIFT-Eval（中文本地说明）

该目录保存上游 GIFT-Eval 数据卡，英文原始说明见 `README.md`。GIFT-Eval 是用于 zero-shot 时间序列预测的公开 benchmark，覆盖多个数据集、领域、频率和预测长度。

## 在本项目中的用途

Chronos-2、Sundial 等时序基础模型的预训练或外部模型卡可能引用 GIFT-Eval，但 ChronoScribe 当前论文主表使用的是 `benchmark/evaluation_manifest.json` 中的 23 维 paired suite。两者不能直接当作同一评测协议。

## 使用规则

数据的许可、下载和再分发条件以 Salesforce GIFT-Eval 上游仓库和数据卡为准。不要把上游数据许可证误认为 ChronoScribe 项目许可证。
"""


def generic_note(path: Path, text: str) -> str:
    return "# 中文版\n\n这是对应英文文件的中文伴随说明。原始内容和数值保留在英文文件中。\n\n" + text


def main() -> None:
    created = []
    for path in sorted(ROOT.rglob("*.md")):
        if path.name.endswith(".zh-CN.md"):
            continue
        if ".git" in path.parts:
            continue
        target = path.with_name(path.stem + ".zh-CN.md")
        if target.exists():
            continue
        if path.parts and path.parts[0] == "scores":
            content = translate_scorecard(path.read_text(encoding="utf-8"), path)
        elif path.parts and path.parts[0] == "models":
            content = model_note(path)
        elif path.parts[:2] == ("data", "GIFT-Eval"):
            content = dataset_note(path)
        elif path.parts[:2] == ("training", "README.md"):
            continue
        else:
            content = generic_note(path, path.read_text(encoding="utf-8"))
        target.write_text(content.rstrip() + "\n", encoding="utf-8")
        created.append(str(target.relative_to(ROOT)))
    print(f"created {len(created)} Chinese companions")
    for item in created:
        print(item)


if __name__ == "__main__":
    main()
