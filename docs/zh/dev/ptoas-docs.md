# 文档更新


站点在构建时获取镜像的 `master` 分支。现有 Docs 工作流在 `main` 推送和
手动触发时发布，并通过每日运行获取已镜像的 PTOAS 文档更新。
构建失败时，线上保留最近一次成功发布的版本。

本地构建前，先准备一次文档来源，再使用常规 MkDocs 命令：

```bash
git clone --depth 1 https://github.com/hw-native-sys/PTOAS.git .cache/ptoas-docs
pip install -r docs/requirements.txt
mkdocs build --strict
```

更新已有来源目录时，在构建前执行
`git -C .cache/ptoas-docs pull --ff-only`。

PyPTO 当前生成的操作见 [PTOAS 算子状态矩阵](../../dev/ptoas-op-status.md)。
