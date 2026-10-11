# ChronoScribe 评测套件

ChronoScribe Evaluation Suite 用于评测接收自然语言请求和历史观测的完整系统。它的覆盖范围比纯预测 benchmark 更广，因此完整系统的 TLS 和 verified forecast-only 评分必须分开报告。

## 套件与 paired 协议

完整套件包含 23 个维度、8,120 条记录。`benchmark/evaluation_manifest.json` 每个维度选择 20 条，共 460 条，seed 为 `2026`。当前 manifest digest 为：

`25eb2519ed22cc61697fe2faab4337e85cdc3070f23d66952b075f2c7d650bea`

所有 paired 运行都必须使用该 manifest，并按 `(dimension, id)` 配对。完整套件数量和 paired 数量是不同概念，必须分别报告。manifest 和 canonical ID 迁移前生成的结果属于历史结果，除非重新配对。

## 维度目录

| ID | 组别 | 评测重点 |
| --- | --- | --- |
| D1 | Forecasting | 多领域短期预测 |
| D2 | Forecasting | 长期预测退化 |
| D3 | Forecasting | 多变量预测 |
| D4 | Forecasting | 跨领域 zero-shot 迁移 |
| D5 | Forecasting | few-shot 数据效率 |
| D6 | Temporal reasoning | 趋势判断 |
| D7 | Temporal reasoning | 异常检测与解释 |
| D8 | Temporal reasoning | 周期识别 |
| D9 | Forecasting | 噪声、缺失和分布变化鲁棒性 |
| D10 | Uncertainty | 分位数区间与 calibration |
| D11 | Instruction following | JSON 有效性与字段完整性 |
| D12 | Instruction following | 多轮修正 |
| D13 | Temporal reasoning | 因果归因 |
| D14 | Forecasting | 可再生能源预测 |
| D15 | Temporal reasoning | 时序分类 |
| D16 | Instruction following | 工作流规划和执行输出 |
| D17 | Forecasting | 渐进缺失数据鲁棒性 |
| D18 | Efficiency | 延迟、吞吐、内存和 token 预算 |
| D19 | Temporal reasoning | 多变量贡献归因 |
| D20 | Forecasting | 不同预测长度下的误差 |
| D21 | Instruction following | 端到端预测与解释遵循 |
| D22 | Forecasting | 极端异常值鲁棒性 |
| D23 | Instruction following | 长多轮一致性 |

历史 TLS 将 Forecasting、Temporal reasoning、Instruction following、Uncertainty、Efficiency 分别赋予 60%、15%、15%、5%、5%。这是完整工作流摘要，当前主要预测表不使用 TLS。

## 数据来源和构造

套件使用 ETT、Electricity、Traffic、Weather、Exchange Rate、Illness、M5、可再生能源、NAB、Yahoo S5、UEA/UCR、MONSTER、MSL、SMAP、SMD、PSM 和 SWaT 等来源。数据许可证和访问条款仍以各上游项目为准。benchmark 记录由历史窗口、自然语言请求和 ground truth 或任务标签组成，并覆盖不同预测长度、上下文、输出格式、推理、不确定性和鲁棒性条件。

被测系统不能接收未来观测。纯预测评分还要求 manifest target 能在解析后的原始序列中定位；无法定位时直接排除，不猜测目标位置。

## Manifest 生成

```bash
python -m utils.evaluation_manifest --benchmark-dir benchmark \
  --output benchmark/evaluation_manifest.json --samples-per-dim 20 --seed 2026
```

ID 规范为 `<dimension>__<source-file>__<row-number>__<legacy-id>`。加入维度前缀可以避免旧文件中的 ID 冲突。已发布 manifest 成员发生变化时应建立新协议，不能静默覆盖。

## 评测边界

benchmark 为所有系统提供目标预测长度，并在同一目标窗口上评分。控制器可以配置上下文和路径，但该协议不证明无约束的 horizon planning。无效结构化输出、fallback、目标无法对齐和 controller cache miss 都必须和有效 adaptive decision 分开记录。
