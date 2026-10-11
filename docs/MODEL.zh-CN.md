# ChronoScribe 模型与结论边界

ChronoScribe 是一个系统框架，不是新的单体预测模型。语言模型负责请求解释、执行配置和回答组织；冻结的时序模型始终是数值预测的唯一来源。

## 执行契约

控制器输出经过 schema 检查的指令：

```json
{
  "use_chronos2": true,
  "prediction_length": 96,
  "context_length": 512
}
```

`use_chronos2` 选择参考时序模型路径，`prediction_length` 记录控制器请求的预测长度，`context_length` 选择历史窗口。paired benchmark 为所有系统提供统一的可执行目标长度，因此所有模型都在同一未来窗口上评分；执行时控制器看不到该未来窗口。

该指令是中间表示，不是预测结果。时序模型接收历史和预测长度，返回点预测或分位数轨迹。runner 再从轨迹确定性地构造证据记录，例如起止值、范围、方向和区间信息。回答模型只接收这些证据并组织文字，不能重新生成或替换预测轨迹。

```text
请求 → 类型化指令 → 时序模型预测 → 证据记录 → 回答
```

## 指令字段

| 字段 | 含义 | 评测规则 |
| --- | --- | --- |
| `use_chronos2` | 时序模型路径或语言路径 | 无效的可预测请求会被记录，并按照 runner 声明的策略处理。 |
| `prediction_length` | 控制器请求的预测长度 | paired 评分以 benchmark 的目标长度为准。 |
| `context_length` | 请求使用的历史长度 | 数据解析器只构造历史上下文并记录对齐信息。 |

这些字段名属于当前实现。TimesFM、Sundial、Aurora 等适配器可以使用不同的原生 API，但必须保留路径、历史、预测长度、预测结果和证据的语义。

## 数值来源与 guard

时序模型输出的轨迹是唯一数值来源。证据从该轨迹确定性计算。numeric-claim guard 会将回答中的数字与类型化证据核对，并阻断没有证据支持的数值声明。当前 guard 审计并不能证明所有非数字语义陈述都被证据支持。

## 参考配置

参考控制器是带本地 LoRA adapter 的 Qwen3.5-9B，参考时序模型是 Chronos-2。TimesFM 和 Sundial 是当前的主要时序模型对比，Aurora 是额外的强模型边界案例。`agent_baselines/` 中的 adapter 和受控 revision baseline 不能宣称是外部 agent 的官方复现。

## 微调数据与复现

公开项目包含脱敏配置和 CSV→SFT 构造器，不包含私有训练语料或原始 trainer state。构造器生成指令记录和证据条件回答记录，并保证未来值不进入控制器输入。完整训练复现还需要另行提供数据来源、许可证、样本数、数据划分、prompt、LoRA target modules 和随机种子。

## 可以成立的结论

在时序模型固定时，可以说语言模型生成的上下文配置会改变预测行为。当前严格 paired 结果支持它相对 fixed-512 有优势，相对 fixed-1024 只有有限的总体优势；不能据此宣称它普遍优于 full-history 或所有时序模型。context oracle 使用了目标依赖信息，只能作为回顾性上界，不能作为在线决策策略。
