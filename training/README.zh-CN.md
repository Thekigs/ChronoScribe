# ChronoScribe SFT 数据构造器

`build_chronoscribe_sft.py` 将常规宽表时间序列 CSV 转换为 ChronoScribe 语言控制器使用的成对对话记录。输入可以包含一个时间戳列和一个或多个数值列。

## 记录构造

每个可用历史窗口生成两个共享 `pair_id` 的记录：

1. 指令记录：把请求和确定性的历史摘要映射成 `runners/run_chronoscribe.py` 接受的 JSON，包括 `use_chronos2`、`prediction_length` 和 `context_length`。
2. 回答记录：把请求和由预测结果得到的证据映射成 `trend` 与 `analysis` JSON；控制器不需要输出预测轨迹。

输入 prompt 只包含历史前缀。未来值只用于确认时间窗口，不会序列化到 directive 输入或 target。所需窗口存在缺失值时直接拒绝，不会静默重采样或插值。默认情况下，两个 system prompt 和 compact evidence string 与当前参考 runner 使用的内容完全一致。

## 示例

```bash
python -B training/build_chronoscribe_sft.py \
  --input-csv data/sft/provenance_disjoint_series.csv \
  --output-dir data/sft/provenance_disjoint_series \
  --dataset-name provenance_disjoint_series \
  --domain energy \
  --horizons 24 48 96 \
  --context-candidates 128 256 512 \
  --stride 48
```

命令会写出 `train.jsonl`、`validation.jsonl` 和可审计的 `manifest.json`。数据按 `pair_id` 切分，因此同一窗口的指令记录和回答记录不会被拆到不同集合。

## 证据来源

只有 CSV 时，builder 会从历史前缀构造 seasonal-naive P10/P50/P90 proxy quantiles。这会生成与接口一致的回答记录，但不能复现参考 adapter 的 specialist evidence。要完整复现训练记录，必须为相同的 history/horizon 计算 Chronos-2 分位数，再使用相同的 evidence 和 response contract。

只有在推理 runner 已传入对应 P10/P50/P90 字段后，才使用 `--response-evidence-format full_quantiles`。默认 `current_runner` 使用 `runners/run_chronoscribe.py` 当前的 compact evidence string。

如果 source name 与固定评测套件匹配，命令会拒绝运行，除非显式使用 `--allow-evaluation-source`。该选项只适合 schema 测试；正式报告的 LoRA 训练必须与评测源数据保持 provenance 隔离。

## 验证

```bash
python -B -m unittest tests.test_training
```

测试会构造小型合成 CSV，并检查确定性输出、按 `pair_id` 成对切分、运行中使用的 directive/response 字段名、证据是否对齐，以及 target 中不存在 forecast trajectory。
