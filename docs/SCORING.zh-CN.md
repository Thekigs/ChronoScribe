# 评分与报告政策

ChronoScribe 将结果分成两类：包含语言、uncertainty 和 efficiency 任务的完整系统评分，以及在统一目标协议下只比较数值预测的 verified forecast-only 评分。预测结论应以第二类为主要证据。

## 完整系统 TLS

历史 Task-Level Score 定义为：

```text
TLS = 0.60 F + 0.15 R + 0.15 I + 0.05 U + 0.05 E
```

`F`、`R`、`I`、`U`、`E` 是映射到 `[0,100]` 后的维度聚合分数。TLS 是完整工作流的摘要，不是预测准确率。报告 TLS 时必须同时给出原始维度分数、覆盖率和参考边界敏感性分析。旧的 65.5 和 10.6 分叙事属于特定历史协议，不能直接作为当前纯预测结论。

## Verified forecast-only 协议

`tools/score_operational_forecasts.py` 是当前纯预测评分器。它会：

- 按 `(dimension, id)` 对齐记录；
- 只保留能在真实源序列中定位的 manifest target；
- 只用预测起点之前的历史计算尺度；
- 要求预测长度与目标长度完全一致；
- 排除尾部近似目标和伪目标；
- 显式报告所有模型交集和 pairwise 覆盖率；
- 不加入文本、延迟、内存或 TLS 权重。

主指标为 MAE、RMSE、sMAPE、基于历史季节尺度的 MASE 和 Direction Accuracy。WAPE、历史 lag-1 MASE、归一化误差、中位数和逐维度结果作为诊断指标保留。当前全模型交集为 85 条，经过源数据验证的目标池在模型交集过滤前为 212 条。

## 聚合方式

先逐样本计算指标，再汇总均值和中位数。paired 对比只使用相同的 `(dimension, id)`，报告 candidate−baseline 差值以及 sample-level 和 dimension-level bootstrap 区间。MAE、RMSE、sMAPE、MASE、WAPE 越低越好；Direction Accuracy 和区间 coverage 越高越好。

不同领域的原始单位不同，不能直接比较原始误差。因此 CSV 同时导出均值和中位数，论文中还应报告覆盖率和逐维度结果。

在当前 85 条全模型共有样本上，没有任何方法在所有指标上都最好。Persistence 的原始
平均 MAE/RMSE 受大尺度异常值影响而表现很强；Aurora 是较强但不够均衡的学习型时序模型；
TimesFM 和 FreshAdaptive 则在 sMAPE、MASE 和方向准确率之间各有取舍。因此主表应展示完整
指标向量，而不是只挑一个标题数字。

## 不确定性与回答质量

如果时序模型提供 P10/P50/P90，应保留这些分位数并计算 coverage、区间宽度、calibration 和 CRPS 类指标。只有点预测的系统应标记为缺少区间输出，不能推算一个区间后计分。numeric-claim guard 只检查数值声明与类型化证据的一致性，不能证明所有定性推理都正确。

## Efficiency 边界

当硬件、软件、初始化、精度、batch size 和 timeout 固定时，本地完整 runner 的 latency 具有可比性。商业 API 的延迟和 token 统计依赖供应商和网络环境，因此不能把它们与本地模型直接做无条件效率排名。只有测量边界和 provider 条件一致时才报告 API latency。

## 解释规则

TLS 更高不表示系统在所有能力上都更好。不能看到结果后只挑一个有利指标。当前纯预测证据显示 FreshAdaptive、固定 Qwen、Chronos-2、TimesFM、Sundial、Aurora、Persistence 和 SeasonalNaive 之间存在指标取舍，诚实比较必须给出完整五指标向量。
