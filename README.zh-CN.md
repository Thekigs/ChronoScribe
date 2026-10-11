# ChronoScribe（中文说明）

ChronoScribe 是一个以证据约束的时间序列预测框架，将语言交互和数值预测分开处理。语言控制器解释请求并生成类型化执行指令；时序基础模型产生数值预测轨迹；系统从轨迹中确定性地构造证据；语言模型只依据这些证据组织最终回答。

~~~text
自然语言请求
        |
类型化指令：路径、预测长度、历史窗口
        |
时序基础模型
        |
确定性预测证据
        |
基于预测证据的回答
~~~

## 先从这里开始

先运行下面三条命令。它们可以在干净克隆中执行，不需要模型权重或原始数据：

~~~bash
python tools/normalize_benchmark_ids.py --benchmark-dir benchmark --check
python tools/validate_release.py
python -m unittest discover -v
~~~

干净克隆应报告 8,120 条记录、460 条 paired 记录、23 个维度，并通过两个测试。

三条命令都通过，说明源码包自身的结构和评测契约一致。若要运行依赖模型的
冒烟测试，请先安装参考环境、下载允许使用的模型和数据，再执行
[docs/USAGE.zh-CN.md](docs/USAGE.zh-CN.md) 中的冒烟命令。若要复现论文中的结果表，
先阅读 [docs/REPRODUCIBILITY_STATUS.zh-CN.md](docs/REPRODUCIBILITY_STATUS.zh-CN.md)：私有
LoRA adapter、部分需要注册的数据集、专用模型 checkpoint 和外部结果包都不在本仓库中。

当前参考配置使用 LoRA 微调后的 Qwen3.5-9B 控制器和 Chronos-2。TimesFM 与 Sundial 是时序模型基线，Aurora 是额外的边界案例时序模型。控制器、时序模型、证据和评分契约见 [docs/MODEL.zh-CN.md](docs/MODEL.zh-CN.md)。

## 本次发布包含什么

本源码发布包包含 benchmark 记录、固定评测 manifest、参考 runner、评分代码、审计工具、训练数据构造器、模型集成代码、配置文件、中英文文档和发布校验脚本。

本发布包明确不包含模型权重、私有 LoRA adapter、需要单独下载或注册的原始数据、生成结果、scorecard、本地密钥和论文投稿文件。data/ 与 models/ 下提交的 README 说明了这些外部输入和许可证边界。

## 结论边界

目前有充分证据支持的机制结论是：在时序模型和目标预测长度保持一致时，语言控制器生成的历史上下文配置会影响预测质量。benchmark 为所有系统提供并固定目标预测长度，控制器只重复该长度以满足 schema；控制器不能看到留出的未来值，也不会生成或修改数值预测轨迹。

当前证据不支持“adaptive routing 优于所有固定策略和所有时序模型”的普遍说法。严格 paired 分析显示它相对 fixed-512 有明显的总体优势，但相对 fixed-1024 只有较小的 MAE/RMSE 优势，其他指标和 full-history 对比并不稳定。每次比较都应同时报告完整指标向量、覆盖样本数和协议标签。

## 项目的核心价值

ChronoScribe 围绕五点展开：

1. **有证据约束的语言控制。** 语言模型负责解释请求、选择执行路径和历史长度；
   数值预测始终只由时序模型产生。
2. **明确的类型化执行契约。** 控制器输出经过 schema 检查的指令，显式包含路径、
   预测长度和历史窗口；所有系统都在同一 manifest 目标窗口上评测。
3. **基于证据的回答生成。** runner 从时序模型轨迹中确定性地构造证据，
   numeric-claim guard 会阻断没有证据支持的数值声明。
4. **受控的对比协议。** 配对 ID、源数据目标、仅历史尺度和独立的纯预测评分，
   可以区分上下文路由带来的变化、时序模型差异和回答格式差异。
5. **清晰的训练数据边界。** 公开的 SFT 构造器生成成对的指令/证据记录，保证未来值
   不进入控制器输入，并默认拒绝评测源数据泄漏。

## 结果概览

下面列出历史 23 维完整系统评测中的全部模型和组件消融结果。所有行使用同一套 F/TLS 评分协议，
最后一列是 ChronoScribe 相对该行的分数优势。

| 系统 / 模型 | F | TLS | ChronoScribe 优势 |
| --- | ---: | ---: | ---: |
| **ChronoScribe（Qwen3.5-9B + Chronos-2）** | **73.7** | **65.545** | — |
| fixed-1024 控制器 | 59.9 | 60.090 | **F +13.8 / TLS +5.455** |
| fixed-512 控制器 | 59.5 | 59.640 | **F +14.2 / TLS +5.905** |
| fixed-256 控制器 | 59.3 | 59.640 | **F +14.4 / TLS +5.905** |
| fixed-128 控制器 | 58.4 | 59.095 | **F +15.3 / TLS +6.450** |
| fixed-64 控制器 | 56.3 | 57.820 | **F +17.4 / TLS +7.725** |
| ChronoScribe（无 LoRA） | 56.5 | 51.650 | **F +17.2 / TLS +13.895** |
| 仅 Chronos-2 消融 | 37.6 | 35.335 | **F +36.1 / TLS +30.210** |
| 移除 Chronos-2 消融 | 23.9 | 25.470 | **F +49.8 / TLS +40.075** |
| Gemini 3.1 Flash | 51.8 | 54.995 | **F +21.9 / TLS +10.550** |
| MiniMax-M3 | 51.8 | 50.020 | **F +21.9 / TLS +15.525** |
| GPT-5.5 | 51.5 | 51.690 | **F +22.2 / TLS +13.855** |
| Grok-4.3 | 49.4 | 50.260 | **F +24.3 / TLS +15.285** |
| GPT-5.6-sol | 49.1 | 50.920 | **F +24.6 / TLS +14.625** |
| Claude Fable 5 | 48.4 | 48.900 | **F +25.3 / TLS +16.645** |
| GLM-5.2 | 47.9 | 49.275 | **F +25.8 / TLS +16.270** |
| Claude Sonnet 5 | 47.6 | 50.040 | **F +26.1 / TLS +15.505** |
| GPT-5.4 | 46.5 | 50.255 | **F +27.2 / TLS +15.290** |
| DeepSeek-V4 Flash | 46.0 | 50.335 | **F +27.7 / TLS +15.210** |
| Gemini 3.5 Flash | 46.0 | 48.000 | **F +27.7 / TLS +17.545** |
| TimesFM | 44.3 | 49.085 | **F +29.4 / TLS +16.460** |
| Qwen3.7 Max | 43.6 | 46.900 | **F +30.1 / TLS +18.645** |
| Gemini 3.8 Flash | 43.6 | 45.900 | **F +30.1 / TLS +19.645** |
| Kimi K3 | 42.7 | 45.000 | **F +31.0 / TLS +20.545** |
| Claude Opus 4.8 | 41.1 | 46.185 | **F +32.6 / TLS +19.360** |
| Grok-4.6 | 40.8 | 45.400 | **F +32.9 / TLS +20.145** |
| DeepSeek-V4 Pro | 40.0 | 47.315 | **F +33.7 / TLS +18.230** |

`F` 是预测组件分数；`TLS` 是完整工作流分数，同时计入推理、指令遵循、不确定性和效率。
这些数值来自历史完整系统快照。仅 specialist 的 11 维评测单独保留在
[docs/SCORING.zh-CN.md](docs/SCORING.zh-CN.md) 中，不与本表混合排名。

## 评测快照

- 完整 benchmark：23 个维度，共 8,120 条源记录。
- 固定的 paired manifest：[benchmark/evaluation_manifest.json](benchmark/evaluation_manifest.json)，共 460 条记录。
- manifest 文件 SHA-256：`25eb2519ed22cc61697fe2faab4337e85cdc3070f23d66952b075f2c7d650bea`。
- benchmark 成员摘要：`a3ab95f613cfd2cc99f63465cd7a2b4af31d200a941456259eccbfb27847cca5`。
- verified forecast-only 协议使用源数据中可定位的目标、仅历史数据计算的尺度、原始预测指标和共有 ID 配对比较。

分数定义和结果来源详见 [docs/SCORING.zh-CN.md](docs/SCORING.zh-CN.md) 与 [docs/REPRODUCIBILITY_STATUS.zh-CN.md](docs/REPRODUCIBILITY_STATUS.zh-CN.md)。

## 安装

支持 Python 3.10 或 3.11。参考 Conda 文件固定 Python 3.10；使用 Python 3.11 时，需要确认目标平台提供兼容的 wheel，并继续使用同一组依赖版本。

~~~bash
conda env create -f environment.yml
conda activate ChronoScribe
pip install -r requirements.txt
pip install -e .
~~~

可编辑安装会让 Python 工具直接发现可复用源码包。模型和 specialist 依赖仍保留在独立的 requirements 文件中，因为部分 baseline 环境有意使用互不兼容的依赖版本。

主环境覆盖 Qwen/LoRA 控制器、Chronos-2、TimesFM、评分、审计和本地 API。MOMENT 与存档的 Moirai/Uni2TS 使用不兼容的依赖版本，必须分别安装：

~~~bash
pip install -r requirements-moment.txt
pip install -r requirements-moirai.txt
~~~

参考环境可根据 runner 配置使用 CUDA、Apple MPS 或 CPU。只安装当前复现实验所需的环境。

## 外部输入

| 输入 | 位置或命令 | 用途 |
| --- | --- | --- |
| Qwen3.5-9B | `python scripts/download_models.py --qwen-only` | 语言控制器 |
| Chronos-2 | `python scripts/download_models.py --c2-only` | 参考数值时序模型 |
| 参考 LoRA adapter | `models/checkpoint-7500/` | 完整自适应控制器 |
| 原始及需注册的数据集 | `python scripts/download_data.py`，以及[复现状态说明](docs/REPRODUCIBILITY_STATUS.zh-CN.md)中的注册步骤 | 基于数据集的实验 |
| TimesFM、Sundial、Aurora checkpoint | [docs/USAGE.zh-CN.md](docs/USAGE.zh-CN.md)说明的上游 package/checkpoint | 时序模型基线 |
| MOMENT 和 Moirai 依赖 | `requirements-moment.txt`、`requirements-moirai.txt` | 隔离的存档基线 |

模型下载器固定文档参考配置使用的 Qwen 和 Chronos-2 snapshot。它不会下载私有 LoRA adapter，也不会绕过上游注册和许可证要求。数据下载器遇到自动来源失败或输出不完整时会以非零状态退出。

## 运行配对 benchmark

所有可比较系统必须使用同一个 manifest：

~~~bash
python runners/run_chronoscribe.py --manifest benchmark/evaluation_manifest.json
python runners/ablate_nolora.py --manifest benchmark/evaluation_manifest.json
python runners/ablate_noc2.py --manifest benchmark/evaluation_manifest.json
python runners/ablate_c2only.py --manifest benchmark/evaluation_manifest.json
~~~

使用少量维度进行冒烟运行：

~~~bash
python runners/run_chronoscribe.py \
  --manifest benchmark/evaluation_manifest.json \
  --dims D1_short_term_forecasting,D10_uncertainty
~~~

运行完成后评分：

~~~bash
python scoring/score_results.py \
  --results-dir results/chronoscribe \
  --benchmark-dir benchmark \
  --output-dir scores/chronoscribe \
  --model-name chronoscribe
~~~

如果只比较预测质量，不纳入文本、延迟和 TLS 权重，先阅读
[docs/USAGE.zh-CN.md](docs/USAGE.zh-CN.md) 中的完整命令：

~~~bash
python tools/score_operational_forecasts.py --help
~~~

评分器成功生成
`scores/operational_forecasting_verified_20261003.json` 后，再导出紧凑 CSV 表：

~~~bash
python tools/export_operational_table.py
~~~

该评分器会排除无法在源数据历史中精确对齐的目标，不会把尾部近似或依赖目标的数据字段当作真实未来目标。

## 本地 API

启动本地 OpenAI-compatible 服务：

~~~bash
python runners/server.py --port 8000
~~~

默认监听地址为 `127.0.0.1`。除非通过 `--allow-origin` 明确提供精确 origin，否则 CORS 关闭。请求字段、错误行为和示例 payload 见 [docs/USAGE.zh-CN.md](docs/USAGE.zh-CN.md)。

## 项目结构

~~~text
ChronoScribe/
├── benchmark/         # 23 个维度、源记录、元数据和固定 manifest
├── configs/            # 脱敏后的控制器和 baseline 配置
├── data/               # README 和本地上游数据，下载内容被忽略
├── docs/               # 框架、benchmark、评分、使用和证据文档
├── models/             # 模型集成代码、模型卡，权重被忽略
├── runners/            # 主 runner、API、specialist 和消融 runner
├── scripts/            # 数据集和模型下载入口
├── scoring/            # 指标计算和 paired 聚合
├── tests/              # 离线回归测试
├── tools/              # 审计、规范化、评分、导出和验证
├── training/           # provenance 隔离的 SFT 构造器和测试
├── utils/              # manifest、数据、证据和 guard 契约
└── agent_baselines/    # 受控 revision 和外部 adapter 契约
~~~

仓库通过 `pyproject.toml` 提供 Python 包元数据。安装当前工作流所需的 requirements 后，再执行 `pip install -e .`。现有 `runners/` 和 `tools/` 路径仍是命令行入口。


## 文档和证据政策

| 先看什么 | 这份文档解决什么问题 |
| --- | --- |
| [使用和运行流程](docs/USAGE.zh-CN.md) | 安装、冒烟测试、benchmark、评分和本地 API |
| [复现状态](docs/REPRODUCIBILITY_STATUS.zh-CN.md) | 干净克隆能运行什么，以及哪些模型、数据或结果文件必须另行提供 |
| [模型与结论边界](docs/MODEL.zh-CN.md) | 控制器与时序模型的契约，以及当前协议支持哪些结论 |
| [Benchmark 与任务目录](docs/BENCHMARK.zh-CN.md) | manifest 成员、评测维度、数据来源和目标泄漏规则 |
| [评分与指标政策](docs/SCORING.zh-CN.md) | 完整系统 TLS 和 verified forecast-only 指标如何计算 |
| [复现和发布规则](docs/REPRODUCIBILITY.zh-CN.md) | 公开结果包需要披露什么，以及哪些内容可以发布 |
| [外部及受控 agent baseline](docs/AGENTIC_BASELINE_PROTOCOL.zh-CN.md) | 如何区分本地控制实验和外部 agent 对比 |

`benchmark/`、`data/`、`models/` 和 `training/` 下的 README 解释各自目录中的文件，
并说明使用这些文件时需要满足的条件。

每个英文 Markdown 文档都有同名 .zh-CN.md 中文伴随版。另行提供结果包时，论文中引用生成结果前应先核对其中的注册表和协议标签。

## 许可证和引用

项目代码使用 MIT License。第三方模型、数据集和上游代码仍受各自许可证和条款约束，详见 [docs/legal/THIRD_PARTY_NOTICES.zh-CN.md](docs/legal/THIRD_PARTY_NOTICES.zh-CN.md)。

学术使用请引用 [CITATION.cff](CITATION.cff)。不要把 .env、私有 adapter、本地 checkpoint、受限制的原始数据、生成的 score 目录或论文投稿文件作为源码发布。
