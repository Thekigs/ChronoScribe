# 外部 agent 对比策略
> **源码发布说明：**本源码包不包含 `scores/` 和 `results/`。文档中的这些路径指向另行提供的实验结果包，不是干净克隆的必需输入。参见 [`REPRODUCIBILITY_STATUS.zh-CN.md`](REPRODUCIBILITY_STATUS.zh-CN.md)。


外部 agent 的比较分成两层证据，避免把不同输入协议误写成 paired benchmark 结果。

| 层级 | 方法 | 输入协议 | 论文中的用途 |
|---|---|---|---|
| 主 paired 对照 | ChronoSteer-style bounded revision | 当前 manifest、相同 Qwen controller、Chronos-2；没有匹配事件文本；revision 受到严格边界约束并审计 | 与简单 Qwen+TSFM 管线比较机制；可以放在主预测表旁讨论，但不能称为官方 ChronoSteer |
| 官方外部参考 | 官方 ChronoSteer checkpoint 推理 | 官方 checkpoint、Chronos-Bolt、BGE-M3 和包含预计算 context embedding 的 MTSFBench-300 `rev-test-reply.json` | 单独的参考表；证明已运行已发布 checkpoint 的推理，但不能建立与 ChronoScribe 的跨 benchmark 排名 |
| 协议不匹配 | TimeSeriesScientist | 外部 LLM/API 和图结构 agent 契约 | 在获得授权的共同输入协议前只做相关工作和协议讨论 |
| 协议不匹配 | Last-mile forecasting agent | 当前 manifest 没有的业务/事件上下文及检索记忆输入 | 只作为定性边界案例，不制造数值排名 |

## 为什么必须分开

ChronoSteer 官方数据包含预测前文本上下文，并且使用不同 benchmark。已发布的测试文件已经提供 context embedding，因此本次 checkpoint 复现没有重新调用原始外部 LLM/API 来生成 revision 指令。把这些上下文删除，或者为当前 manifest 人工编造上下文，都不再是官方 reproduction。反过来，直接把 MTSFBench-300 的官方数字与 ChronoScribe 的 manifest 数字比较，会混淆数据集、horizon、尺度处理、上下文和指标单位。

因此严谨结论分为两部分：

1. 在当前 manifest 上，允许 Qwen controller 发出受限的 ChronoSteer-style revision，并没有相对简单 fixed Qwen+TSFM 管线产生稳定改善（见 `scores/agentic/paired_strict_revision_vs_simple_fixed1024.json`）。
2. 已发布的 ChronoSteer checkpoint 已在其官方 MTSFBench-300 协议上成功运行（使用官方提供的预计算 context embedding；见 `scores/agentic/chronosteer_official_reproduction_20261005.json`）。

这两项是互补证据，不应合并成一个 leaderboard。

## 可直接放入论文的表述

> 我们在两个明确分开的协议下评估 LLM 辅助的 forecasting agent。首先，同一 manifest 下的 ChronoSteer-style bounded-revision 对照使用与 ChronoScribe 相同的 Qwen 接口和 Chronos-2 specialist；它是机制对照，不是官方 reproduction。其次，我们在作者发布的 MTSFBench-300 `rev-test-reply` 协议上运行已发布的 ChronoSteer checkpoint。由于该协议的 benchmark 和预测前事件上下文与我们的 manifest 不同，后者作为 external reference 单独报告，不作为 paired rank。TimeSeriesScientist 和 Last-mile 因需要当前 benchmark 不具备的外部 API 或业务/事件上下文，保留为受协议限制的相关 baseline。

对应证据和执行环境记录在 `scores/agentic/external_protocol_audit.json` 与 `scores/agentic/chronosteer_official_reproduction_20261005.json`。
