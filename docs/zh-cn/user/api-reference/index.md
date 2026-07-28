# `pl.*` API 参考

> **状态：** 草稿骨架 / 占位。本参考将从前端 docstring **自动生成**（Issue #2120），
> 覆盖 `pypto.language.__all__` 中的每个符号（223 个）。请勿在此手写条目 ——
> 它们由文档生成器产出。

## 生成方案

_TODO —— 决定并记录：_

- **生成器：** Sphinx（`autodoc` + `autosummary` + `napoleon`）**或**
  MkDocs-Material（`mkdocstrings[python]`）。二者都消费 Google 风格 docstring
  （由 `.claude/rules/python-style.md` 强制）。
- **CI 门禁：** 以 `--fail-on-warning` / 覆盖率检查构建，使任何新增或重命名的
  公开符号若缺少规范 docstring 即构建失败 —— 保证文档与代码同步。
- **翻译策略：** 决定 EN 权威 + zh-cn 镜像 vs 仅 EN，并记录选择
  （依据 `.claude/rules/documentation.md`）。

## 分类分组（生成后）

生成的参考按类别对 `pl.__all__` 分组：

- 类型与 dtype
- Unified ops
- Tile ops
- 控制流
- `@pl.function` / `@pl.program` / `@pl.jit` / `@pl.inline`
- 内存与数据搬运
- 片上并行（`spmd`、`cluster`、`at`、`spmd_submit`）
- 任务 / manual-scope（`manual_scope`、`scope`、`submit`、`no_dep`、`TaskId`…）
- 跨核 split（`split_aiv`、`split`、`aiv_shard`、`aic_gather`、`tpush*`、
  `tpop*`、`tfree*`、`*_initialize_pipe`）
- Peer buffer（`reserve_buffer`、`import_peer_buffer`）
- 分布式（`pld.*`）

## 散文指南 vs 参考

手写的散文指南（[语言指南](../02-language_guide.md)、
[操作参考](../03-operation_reference.md)）保留为概念/教程材料并交叉链接到此处。
一旦本参考上线，会移除那些容易漂移的手工维护 op 表。
