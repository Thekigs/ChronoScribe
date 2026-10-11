# 依赖环境配置

这些文件集中保存公开源码所需的可复现实验环境，同时隔离互不兼容的
specialist 依赖栈。

- `requirements.txt`：当前 ChronoScribe 主工作流。
- `moment.txt`：独立的 MOMENT 基线环境。
- `moirai.txt`：独立的存档 Moirai 环境。

先根据工作流选择依赖文件，再执行 `pip install -e .` 安装项目源码。
