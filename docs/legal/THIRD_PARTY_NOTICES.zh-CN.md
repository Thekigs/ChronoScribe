# 第三方模型与数据集

ChronoScribe 项目代码采用 `LICENSE` 中的 MIT 许可证。模型权重和数据集不随本源码包发布。请从上游提供方获取，并遵守各自的使用、许可和再分发条款。

| 组件 | 上游条款 | 本源码包的处理 |
|---|---|---|
| Qwen3.5-9B | 参见 Qwen 上游模型卡和许可证 | 通过 `scripts/download_models.py` 在本地下载，不提交权重 |
| Chronos-2 | 参见 Amazon/Chronos 上游模型卡和许可证 | 通过 `scripts/download_models.py` 在本地下载，不提交权重 |
| Sundial | Apache-2.0 模型卡 | 只保留本地集成代码，权重从上游获取 |
| Aurora | MIT 模型卡 | 只保留本地集成代码，权重从上游获取 |
| 基准数据集 | 不同数据集和提供方的条款不同 | 排除 `data/` 原始数据；使用 `scripts/download_data.py` 并遵守注册和再分发要求 |

SWaT、Yahoo S5 等数据源需要注册，项目不会代为再分发。`benchmark/` 下的 JSONL 是派生评测材料，可能包含上游数据集的目标值、统计值、标签或异常元数据。发布前请阅读 `benchmark/README.md`；即使没有原始数据，重新分发受影响的记录前仍必须核对对应上游条款。
