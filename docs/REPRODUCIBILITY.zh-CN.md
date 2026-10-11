# 复现与发布指南

本文档说明本地项目可以复现哪些内容、哪些输入仍需外部提供，并防止历史 score 目录被误认为当前实验。

## 项目边界

项目包含 benchmark 记录、规范化 ID、固定的 paired manifest、运行脚本、时序模型适配器、组件消融、评分代码和审计工具。不包含私有模型 checkpoint、API 密钥、原始 LoRA 训练记录或受限制的上游数据。训练数据构造器是一种可复现的数据构造方法，不是原始私有语料。

## 固定评测契约

- 完整套件：23 个维度、8,120 条记录；
- paired manifest：460 条，每个维度 20 条，seed 为 `2026`；
- 配对键：`(dimension, id)`；
- manifest：`benchmark/evaluation_manifest.json`；
- 当前 manifest SHA-256：`25eb2519ed22cc61697fe2faab4337e85cdc3070f23d66952b075f2c7d650bea`；
- 当前源数据验证目标池：212 条；
- 当前所有模型共有预测交集：85 条。

每次运行都必须记录 manifest digest、模型版本、设备、框架版本、生成参数、controller 来源、目标对齐状态和失败处理策略。规范化 ID 或固定 manifest 之前生成的结果都属于历史结果，除非重新配对并审计。

## 目标与泄漏规则

benchmark 提供目标预测长度。执行过程中，控制器和时序模型只接收历史数据；留出的目标只在预测完成后用于评分。无法在解析后的源序列中定位的目标必须从 verified forecast 表排除。任何 runner 都不能把 `tail_before_gt`、从数据中提取的未来字段或观测尾部当作论文目标。

numeric-claim guard 检查预测相关数字是否能由类型化证据支持；case-study audit 单独检查未来指标是否进入回答生成输入。

## 必须披露的内容

使用这些结果的论文或发布包必须说明数据来源和许可证、完整套件与 paired 数量、manifest digest、数据适配来源、prompt 和输出契约、时序模型 checkpoint 与分位数设置、硬件/软件/设备、生成预算、运行次数、延迟边界、失败策略、原始指标、聚合规则和置信区间方法。

这是经验型系统研究。除非加入正式理论命题和证明，否则复现清单中的 theoretical-contribution 分支应标记为 `no`。

## 验证命令

```bash
python tools/normalize_benchmark_ids.py --benchmark-dir benchmark --check
python tools/validate_release.py
python tools/evaluate_numeric_guard_adversarial.py
python -m unittest tests.test_training
```

这些公开发布命令不会下载模型或运行昂贵推理，只检查规范化 ID、manifest 成员、套件元数据、中文 Markdown 伴随文件以及不能进入公开压缩包的本地文件。实验注册表验证器单独运行，因为源码包不包含结果注册表和 score 文件。

## 公开发布规则

- `.env`、本地 checkpoint、私有 adapter、受限制原始数据、`results/` 和生成的 `scores/` 不应进入源码包；若有意发布结果，必须同时提供其 provenance。
- 不要发布包含私有路径的序列化 trainer state。
- 不要用项目许可证重新授权第三方模型或数据集。
- 发布结果时应同时包含模型卡、数据集引用、manifest digest、实验注册表及其配套证据文件。
