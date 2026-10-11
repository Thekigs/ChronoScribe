# 本地模型文件

本公开源码包只包含轻量级模型集成代码和配置。模型权重、缓存、Qwen 基础模型、Chronos-2 文件和私有 LoRA adapter 都通过 `.gitignore` 排除。

| 目录 | 在 ChronoScribe 中的角色 | 获取运行文件的方式 |
|---|---|---|
| `Qwen3.5-9B/` | 语言控制器和回答模型 | 运行 `python scripts/download_models.py --qwen-only` |
| `checkpoint-7500/` | 参考 LoRA adapter | 另行获得兼容 adapter 后放入此处；本发布包不包含它 |
| `chronos-2/` | 参考数值 specialist | 运行 `python scripts/download_models.py --c2-only` |
| `sundial-base-128m/` | specialist baseline | 从上游获取 `thuml/sundial-base-128m` 并放入此处 |
| `aurora/` | 额外边界 specialist | 从 [Aurora 上游发布](https://github.com/decisionintelligence/Aurora) 获取 checkpoint 并放入此处 |

为了明确模型版本，runner 对 Chronos-2、Sundial 和 Aurora 使用本地路径。TimesFM、Moirai 和 MOMENT 直接加载上游 package/checkpoint。完整的源码检查与 benchmark 复现边界见 [`REPRODUCIBILITY_STATUS.zh-CN.md`](../docs/REPRODUCIBILITY_STATUS.zh-CN.md)。

`scripts/download_models.py` 将 Qwen3.5-9B 固定到 revision
`c202236235762e1c871ad0ccb60c8ee5ba337b9a`，将 Chronos-2 固定到 revision
`29ec3766d36d6f73f0696f85560a422f50e8498c`。每个模型下载后都会在旁边写入
`chronoscribe_download.json`，记录这些标识，避免后续运行静默接受不同快照。
