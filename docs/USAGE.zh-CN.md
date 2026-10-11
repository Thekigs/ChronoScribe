# 使用与运行流程

## 先选择运行目标

| 目标 | 从哪里开始 | 外部输入 |
| --- | --- | --- |
| 检查干净源码克隆 | `README.zh-CN.md` 中的校验命令 | 无 |
| 运行完整 ChronoScribe 系统 | [配对 benchmark](#paired-benchmark) | Qwen3.5-9B、参考 LoRA adapter、Chronos-2 和允许使用的数据集 |
| 对比不同数值时序模型 | [时序模型对比](#时序模型对比) | 所选模型的 checkpoint 或 package |
| 复现外部 agent 对比 | [AGENTIC_BASELINE_PROTOCOL.zh-CN.md](AGENTIC_BASELINE_PROTOCOL.zh-CN.md) | 外部 runner、其规定的输入和另行提供的结果包 |

下面的命令都从仓库根目录执行。只有源码校验流程可以在尚未准备模型和数据时运行。

## 安装

```bash
conda env create -f configs/environment.yml
conda activate ChronoScribe
pip install -r requirements/requirements.txt
pip install -e .
python scripts/download_models.py
python scripts/download_data.py
```

支持 Python 3.10 或 3.11。参考环境包含 Qwen/LoRA 控制器、Chronos-2、TimesFM、Sundial、Aurora adapter、评分和审计工具。下载器不会下载参考 LoRA adapter，也不会绕过数据注册要求。干净源码包可以运行验证检查，但完整 adaptive benchmark 仍需要 `REPRODUCIBILITY_STATUS.zh-CN.md` 中列出的外部输入。MOMENT 的官方依赖与参考 Torch/Transformers 版本冲突，必须在单独环境中使用 `requirements/moment.txt`。存档的 Moirai runner 也必须在单独环境中使用 `requirements/moirai.txt`，因为 Uni2TS 2.0.0 固定 Datasets 2.17.x，而主环境使用已修复路径校验问题的 Datasets 5.0.1。

Qwen3.5-9B、Chronos-2 和 LoRA 文件应放在 `models/` 下，并被版本控制忽略。下载器会报告需要注册或单独接受条款的数据源，不会绕过这些要求。

## Paired benchmark

所有可比系统使用固定 manifest：

```bash
python runners/run_chronoscribe.py --manifest benchmark/evaluation_manifest.json
python runners/run_chronoscribe.py --policy simple_fixed --fixed-context-length 1024 \
  --manifest benchmark/evaluation_manifest.json \
  --output-dir results/simple_fixed1024
python runners/ablate_nolora.py --manifest benchmark/evaluation_manifest.json
python runners/ablate_noc2.py --manifest benchmark/evaluation_manifest.json
python runners/ablate_c2only.py --manifest benchmark/evaluation_manifest.json
```

可用 `--dims` 和 `--max-samples` 运行 smoke test。smoke test 仍属于同一协议的子集，不能作为完整结果表述。

## 时序模型对比

`runners/run_qwen_specialist.py` 固定控制器决策、manifest、历史解析、预测长度和输出格式，只更换时序模型：

```bash
python runners/run_qwen_specialist.py --specialist chronos2 --device mps \
  --manifest benchmark/evaluation_manifest.json --output-dir results/chronos2_run
python runners/run_qwen_specialist.py --specialist timesfm --device mps \
  --manifest benchmark/evaluation_manifest.json --output-dir results/timesfm_run
python runners/run_qwen_specialist.py --specialist sundial --device mps \
  --manifest benchmark/evaluation_manifest.json --output-dir results/sundial_run
python runners/run_qwen_specialist.py --specialist aurora --device cpu \
  --manifest benchmark/evaluation_manifest.json --output-dir results/aurora_run
```

这些运行只产生预测记录，应使用纯预测评分器评分。除非时序模型也实现相同的语言回答协议，否则不能直接并入 TLS 表。

## 纯预测评分

```bash
python tools/score_operational_forecasts.py \
  --baseline FreshAdaptive \
  --model FreshAdaptive results/e2e_fresh_adaptive_strict_forecastdims_20261002 \
  --model ParserFixed1024 results/p0_parser_fixed1024_mps_v2 \
  --model Chronos2Fixed512 results/qwen_chronos2_fixed512 \
  --model TimesFM results/qwen_timesfm_matched \
  --model Sundial results/qwen_sundial_strict_v2 \
  --model Aurora results/qwen_aurora_strict_v2 \
  --output scores/operational_forecasting_verified_20261003.json
python tools/export_operational_table.py
```

导出步骤读取前一个评分器写入的 JSON 文件，只有评分器成功完成后才能执行。

该协议只使用源数据中可定位的留出目标、基于历史的尺度、准确预测长度和共同 ID，并报告 MAE、RMSE、sMAPE、历史季节 MASE、Direction Accuracy、中位数、覆盖率和 paired bootstrap 区间。

## 外部及受控 agent baseline

运行融合或 revision 实验前先阅读 `docs/AGENTIC_BASELINE_PROTOCOL.md`。`agent_baselines/` 提供 `simple_fusion`、受限 `revision_control` 和外部 JSON adapter。本地 ChronoSteer-style 结果是受控适配实验，不是官方 ChronoSteer 复现。

## 完整运行评分

```bash
python scoring/score_results.py \
  --results-dir results/chronoscribe \
  --benchmark-dir benchmark \
  --output-dir scores/chronoscribe \
  --model-name chronoscribe
```

完整评分器会生成逐样本指标、维度摘要、`final_scores.json` 和 `scorecard.md`，并按 `(dimension, id)` 检查 benchmark 配对。

## 交互模式和本地 API

```bash
python runners/chat.py --temperature 0.5 --max-tokens 1024
python runners/server.py --port 8000
```

交互路径可以回答普通语言请求；有可用数值历史时，控制器可通过 `use_chronos2` 选择时序模型路径。本地服务提供 OpenAI-compatible chat-completions 接口，模型 ID 为 `chronoscribe`；默认只监听 `127.0.0.1`，并关闭流式响应和跨域访问。只有在使用可信浏览器客户端时才传入 `--allow-origin https://your-trusted-origin.example`。

## 重新运行前的验证

```bash
python tools/normalize_benchmark_ids.py --benchmark-dir benchmark --check
python tools/validate_release.py
python tools/evaluate_numeric_guard_adversarial.py
python -m unittest tests.test_training
```
