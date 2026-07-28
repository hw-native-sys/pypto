# 性能调优：依赖与分发控制

> **状态：** 草稿骨架。塑造任务图，使独立工作重叠、条件工作被跳过。这些旋钮控制
> *调度*，而非计算。

## 任务图基础

*TODO：*

- `pl.manual_scope` / `pl.scope` / `ScopeMode` —— 手动任务图作用域。
- `pl.submit(...)` / `pl.spmd_submit(...)` —— 启动一个任务，捕获其
  `Scalar[TASK_ID]`。
- `pl.TaskId` / `pl.TASK_ID` —— 依赖句柄类型。

## 显式依赖

*TODO：*

- `deps=[tid, ...]` —— 显式声明生产者边（与自动依赖取并集）。
- `pl.no_dep(arg)` / `pl.adir.no_dep` —— 让单个参数退出自动依赖推断。

> **不只是性能，还有正确性 —— 丢失的 WAR 依赖。** 在 AUTO 作用域下，运行时目前对
> 循环携带缓冲会遗漏 **read 后 write** 的边，静默污染结果（**issue #2058**）。在
> 自动检测落地前，你必须手动加这条边，让 `reader(N)` 先于 `writer(N+1)` 覆盖缓冲
> 完成：
>
> ```python
> _, tid_read  = pl.submit(self.reader, buf, ...)
> _, tid_write = pl.submit(self.writer, buf, ..., deps=[tid_read])
> ```
>
> 完整说明：[精度 › 丢失的 WAR 依赖](../precision/00-workflow.md#已知陷阱丢失的-war-依赖循环携带缓冲)。

## 锚点 / 屏障任务

*TODO：*

- `dummy_task` —— 无计算的依赖锚点 / 屏障；何时以及为何插入。

## 推测式与条件式分发

*TODO：*

- `allow_early_resolve=True` —— 让调度器在任务完成前预先编排其消费者（推测式提前
  分发）。权衡与正确性条件。
- `predicate=(...)` —— 单比较的**分发谓词**，当 gate 为假时整体跳过该任务（例如
  跳过 row count 为 0 的 expert）。应与 `deps=` 搭配，避免读到 stale 的 gate 值。

## 相互作用

*TODO —— 表格：`deps` vs `no_dep` vs `allow_early_resolve` vs `predicate`，
组合约束、常见陷阱（如 `predicate` 未配 `deps`）。*

## 验证效果

通过 [DFX › 诊断渲染](../dfx/02-diagnostics-render.md)（`deps.json` → HTML）
检查生成的图，然后用[测量收益](05-measuring-impact.md)重新测量。
