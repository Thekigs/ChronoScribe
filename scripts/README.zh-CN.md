# 下载脚本

本目录包含两个可选的下载入口，用于获取源码发布包中排除的外部输入：

- `python scripts/download_models.py` 下载固定版本的 Qwen3.5-9B 和 Chronos-2 snapshot。
- `python scripts/download_data.py` 获取允许下载的公开 benchmark 数据源。

两个脚本分别将文件写入仓库的 `models/` 或 `data/` 目录，不会提交下载结果。下载需要注册或受限的数据源前，请先阅读复现和许可证说明。
