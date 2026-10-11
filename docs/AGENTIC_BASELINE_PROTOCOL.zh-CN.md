# LLM 辅助预测 baseline 协议
> **源码发布说明：**本源码包不包含 `scores/` 和 `results/`。文档中的这些路径指向另行提供的实验结果包，不是干净克隆的必需输入。参见 [`REPRODUCIBILITY_STATUS.zh-CN.md`](REPRODUCIBILITY_STATUS.zh-CN.md)。


本文档覆盖本地融合/修正控制，以及评审意见中提到的 ChronoSteer、TimeSeriesScientist 和 last-mile forecasting agent。它与完整系统 TLS runner 分开，因为纯预测 agent 不一定实现相同的回答协议。

## 统一条件

- 使用 `benchmark/evaluation_manifest.json` 并报告 digest；
- 使用 `utils.benchmark_data.load_raw` 和 `build_context` 解析历史；
- controller、事件、路由和 revision 输入都不能包含留出目标；
- 使用相同预测长度、样本 ID、预测指标和失败统计；
- 先报告 raw forecast metrics，再讨论 F/R/I/U/E/TLS；
- 记录设备、模型调用、token 预算、延迟、revision 操作和数值来源；
- 纯预测结论使用 `tools/score_operational_forecasts.py`，它会排除未验证目标和伪目标。

## baseline 标签

`official_reproduction` 表示按论文/官方代码的输入协议执行了引用实现和 checkpoint。`controlled_adaptation` 表示在统一 manifest 和 backbone 下测试相同机制，但不声称官方复现。`qualitative_only` 表示没有共同的输入/任务协议，不分配数值排名。

## 已完成的本地控制

```text
simple_fusion      固定上下文 → TSFM → Qwen 回答
revision_control   固定上下文 → TSFM → Qwen revision → 受限预测
```

受限 revision control 用于隔离 ChronoSteer-style 系统的核心操作，使用相同 manifest、Chronos-2、Qwen LoRA 控制器和 scale/offset/trend 边界。它只改变了两条预测轨迹，没有带来稳定的总体改善，因此被记录为机制对照：

- `results/agentic/revision_chronos2_fixed1024_v2_full/all_results.jsonl`
- `scores/agentic/paired_strict_revision_vs_simple_fixed1024.json`
- `scores/agentic/chronosteer_style_comparison.json`
- `results/agentic/simple_fusion_chronos2_fixed1024_thinking_full_20261006/all_results.jsonl`
- `scores/agentic/revision_vs_strict_simple_fixed1024_thinking_20261006.json`

这些纯预测结果不能并入六组件 TLS 表。
paired 结果在 `scores/agentic/revision_protocol_strata_20261005.json` 中进一步分层：all-forecast intent-to-treat 保留无效 directive 的 no-op fallback；valid-directive 和 exact-target-alignment 只作为诊断子集，避免形成只挑成功修正样本的结论。

严格 simple-fusion 重跑保持历史 Qwen 的 thinking 设置开启，请求上下文长度为 1024，使用相同的 Chronos-2 specialist。结果包含 220 个唯一 manifest 样本、179 个可评分 forecast 样本，以及 1 条明确记录的 `no_usable_history` loader error。在全部 179 个共同 forecast 行上，revision control 的 raw Chronos-2 轨迹与该 fixed baseline 完全一致（最大绝对差为 0.0）。因此它是当前权威的同语言接口 fixed-policy baseline；旧的 `e2e_full_simple_fixed1024_mps` 目录记录的 context alignment 并非始终为 1024，不再用于这项严格比较。

旧 adaptive 结果现在只作为协议敏感性证据保留：`scores/p1_adaptive_protocol_audit_20261006.json` 给出完整配对覆盖，同时明确标注其中 adaptive controller 行来自历史、不可验证的决策。新的 thinking-enabled fresh run 已完成：`results/controller_fresh_thinking_full_20261006` 包含 220 条 manifest 样本、144 条有效 decision 和 124 条 no-clamp paired forecast。固定窗口的性能前沿见 `scores/p1_fresh_thinking_adaptive_fixed_frontier_20261006.json`；adaptive 相对 fixed-512、fixed-1024 和 full-history 没有普遍优势。通过真实 target 验证的 fixed control 表见 `scores/fresh_thinking_all_fixed_controls_verified_20261006.json`。

## 官方外部方法

ChronoSteer 官方仓库、已发布的 alignment checkpoint、BGE-M3 编码器、Chronos-Bolt backbone 和 MTSFBench-300 的 `rev-test-reply.json` 已在兼容环境中通过官方 `--only_test` 流程完成运行。测试文件已经提供预测前的 context embedding，因此这里是 checkpoint 推理复现，不是重新调用原始外部 LLM/API 生成 revision 指令。官方状态是 `official_checkpoint_reproduction_completed`，汇总结果见 `scores/agentic/chronosteer_official_reproduction_20261005.json`。由于该官方协议使用不同 benchmark 和预测前事件上下文格式，结果作为外部参考单独报告，不并入当前 manifest 的 paired ranking；同 manifest 的本地结果仍标记为 `controlled_adaptation`。

TimeSeriesScientist 需要外部 LLM API 和自己的图结构执行输入/输出协议。目前有单样本匹配 adapter，但在数据传输和 API 协议未形成正式条件前，不把它扩展为多样本官方数值结果。Last-mile forecasting agent 需要当前 manifest 没有的业务或事件上下文，因此只做定性讨论。二者不进入主表，也不与 ChronoScribe 做伪 paired 排名。

详细状态见 `scores/agentic/external_protocol_audit.json`，方法配置见 `configs/official_agent_profiles.json`。

生产控制器使用 greedy decoding。已完成的 MPS 诊断 probe
（`scores/sampling_variance_probe_20261005.json`）在 10 个匹配样本上、
temperature 0.7/top-p 0.9 下各重复 3 次，估计运行级方差和
invalid-directive rate；另有一个开启 thinking 的 4 样本、每样本 3 次重复
pilot（`scores/sampling_variance_probe_thinking_20261006.json`）。汇总状态见
`scores/sampling_variance_protocol_summary_20261006.json`。两者都不替代生产
分数，也不生成随机 TLS 排名。

## 外部 adapter 契约

外部 runner 必须写出含 `predicted_values` 的 JSON，也可以返回分位数、controller 输出、revision 输出和数值来源。adapter 会在发送请求前删除 `ground_truth`，检查预测长度，记录 command/stdout/stderr，并在外部程序返回后计算原始预测指标。

```bash
python tools/run_agentic_baselines.py --baseline external \
  --external-name chronosteer_official \
  --external-status official_reproduction \
  --external-command 'python ./external_runner.py --request {request} --response {response}' \
  --output-dir results/agentic/chronosteer_official
```

## Paired 评分

```bash
python tools/score_agentic_baselines.py \
  --baseline results/e2e_full_simple_fixed1024_mps \
  --candidate results/agentic/revision_chronos2_fixed1024_v2_full \
  --output scores/agentic/paired_revision_vs_simple_fixed1024_new.json
```

评分器报告共同样本数、均值、paired 差值和确定性 bootstrap 区间，不为纯预测外部 agent 生成 TLS 排名。
