# 公开源码包的复现状态

本仓库是源码发布包，不是自包含的结果归档。干净克隆可以验证 benchmark 契约并运行单元测试，但在提供外部模型、adapter 和允许使用的数据集之前，不能复现论文中的完整数值。

`benchmark/` 下的 JSONL 是派生评测记录，不是原始数据集；但它们仍可能受到上游再分发条款约束。发布此目录前请阅读 `benchmark/README.md` 并逐项核对来源条款；技术校验通过不等于获得了许可证许可。

## 干净克隆即可运行的检查

安装 Python 3.10 或 3.11 以及 `requirements/requirements.txt` 后，下面的检查不需要模型权重或原始数据：

```bash
python tools/normalize_benchmark_ids.py --benchmark-dir benchmark --check
python tools/validate_release.py
python tools/evaluate_numeric_guard_adversarial.py
python -m unittest tests.test_training
```

当前发布包会验证 8,120 条 benchmark 记录、460 条 paired manifest 记录、23 个维度和中文文档伴随文件。

## 必须单独提供的输入

- Qwen3.5-9B 基础模型和 Chronos-2 specialist 可以通过 `scripts/download_models.py` 下载。
- `scripts/download_models.py` 将 Qwen3.5-9B 固定为 `c202236235762e1c871ad0ccb60c8ee5ba337b9a`，将 Chronos-2 固定为 `29ec3766d36d6f73f0696f85560a422f50e8498c`；与本文档对应的参考实验不要替换这些 revision。
- 参考 LoRA adapter 不在本发布包中，且没有提供公开下载地址。运行 adaptive controller 前，需要把兼容 adapter 放到 `models/checkpoint-7500/`。没有它，无法复现论文中的完整 ChronoScribe controller 决策。
- `scripts/download_data.py` 会获取允许下载的公开来源，但 SWaT 和 Yahoo S5 需要手动注册。数据集可用性和再分发条款可能由提供方调整。
- TimesFM、Sundial 和 Aurora 需要各自的上游 checkpoint 或 package。源码包只包含 Aurora/Sundial 集成代码，不包含权重。存档的 Moirai runner 使用 `requirements/moirai.txt`，因为 Uni2TS 2.0.0 固定 Datasets 2.17.x；不能把它装进主环境的 Datasets 5.0.1 环境。
- 要得到论文中的精确数字，还需要相同的 manifest digest、模型版本、prompt/configuration、框架版本、设备、精度、生成参数和失败策略。

## 复现等级

| 等级 | 结果 | 所需输入 |
|---|---|---|
| 契约与元数据检查 | 可由本仓库复现 | 仅 Python 环境 |
| 数值 guard 与训练构造器测试 | 可由本仓库复现 | 仅 Python 环境 |
| 小规模 specialist 冒烟测试 | 完成安装后可复现 | 允许使用的数据和 Chronos-2 权重 |
| 完整 fixed 或 adaptive benchmark | 有条件复现 | 全部可解析数据、Qwen 基础模型、兼容 LoRA adapter、Chronos-2 和硬件 |
| 精确论文结果表 | 仅凭源码无法保证 | 以上全部输入，加原始结果包和固定版本 |

脚本会在本地创建 `results/` 和 `scores/`。这两个目录被有意排除，因此历史证据文档里出现的结果文件不会存在于干净克隆中。只有在另行提供 `scores/experiment_registry_20261003.json` 及其权威文件的结果包时，才应运行 `tools/validate_experiment_registry.py`。
