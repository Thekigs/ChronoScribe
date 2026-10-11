# 工具目录说明

本目录中的脚本保留历史路径，以保持复现文档中的命令稳定。查找工具时可以参考下表；新增脚本应优先遵循对应分组。

| 分组 | 常见脚本 | 用途 |
| --- | --- | --- |
| `audit_*`、`evaluate_*` | `audit_*`、`evaluate_numeric_guard.py` | 发布、来源和契约检查 |
| `score_*`、`export_*` | `score_operational_forecasts.py`、`export_operational_table.py` | 评分和结果导出 |
| `run_*` | `run_agentic_baselines.py`、`run_controller_forecasts.py` | 实验和 baseline 执行 |
| `normalize_*`、`consolidate_*`、`merge_*` | `normalize_benchmark_ids.py` | 数据和结果准备 |
| `build_*`、`generate_*`、`annotate_*` | `build_adaptive_deployment_control.py` | 构造 artifact 和报告 |

脚本有意保持为独立的命令行入口。只有同时提供兼容 wrapper 或记录命令迁移时，才应将脚本移动到子目录。
