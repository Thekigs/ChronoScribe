# Runner 目录说明

Runner 路径是 README 和 `docs/USAGE.md` 中记录的稳定命令行入口。

- `run_chronoscribe.py`、`ablate_*.py`：参考流程和消融流程。
- `run_*_specialist.py`、`run_timesfm.py`、`run_moment.py`、`run_moirai.py`：specialist 对比。
- `server.py`、`chat.py`、`run_api.py`：本地服务和交互客户端。
- `_ablate_fixed_c.py`：共享的固定上下文实验实现。

模型专用 import 应保留在实际需要它的 runner 中，让离线校验无需模型权重即可运行。
