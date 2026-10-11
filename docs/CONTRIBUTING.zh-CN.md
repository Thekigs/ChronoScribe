# 参与 ChronoScribe 开发

欢迎改进 ChronoScribe。文档修正、复现问题报告、benchmark 质量改进和范围明确的代码修改都可以提交。

## 创建 Pull Request 前

1. 阅读 `docs/` 中与改动相关的文档，并查看[安全政策](SECURITY.zh-CN.md)。
2. 保持 benchmark 成员、评测 manifest 和数据来源边界明确。不要提交私有数据、凭据、模型权重或生成的评分结果。
3. 安装当前工作流所需的环境，然后执行 `pip install -e .` 安装源码包。
4. 运行 README 中的离线检查：

   ```bash
   python tools/normalize_benchmark_ids.py --benchmark-dir benchmark --check
   python tools/validate_release.py
   python -m unittest discover -v
   ```

## Pull Request

说明对用户或复现流程的影响，列出验证命令，并明确 benchmark 记录、manifest、指标或外部输入是否发生变化。不要把无关的格式化修改和生成文件混入提交。

## 报告安全问题

请遵循 [SECURITY.zh-CN.md](SECURITY.zh-CN.md)，涉及凭据、数据访问或漏洞的问题应通过私密渠道报告。
