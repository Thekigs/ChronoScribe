# 本地数据文件

`data/` 保存数据解析器使用的本地上游数据和文件。由于不同上游数据集的许可证和再分发条件不同，这些文件不进入公开源码包。

benchmark 记录和固定的 paired 样本选择位于 `benchmark/`；`data/` 只是原始/本地数据层。纯预测评分只有在 manifest target 能够在解析后的源序列中定位时才接受该样本。数据来源信息和许可证规则见 `docs/BENCHMARK.zh-CN.md` 与 `docs/REPRODUCIBILITY.zh-CN.md`。
