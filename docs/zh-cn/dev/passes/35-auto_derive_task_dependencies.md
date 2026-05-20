# AutoDeriveTaskDependencies Pass

## 概览

`AutoDeriveTaskDependencies` 在 `with pl.manual_scope():` 区域内保守推导
task-to-task 依赖边。它运行在
[`DeriveCallDirections`](33-derive_call_directions.md) 之后，读取已经解析的
`Call.attrs["arg_directions"]`，并把编译器推导出的 producer TaskId 边写入
`Call.attrs["compiler_manual_dep_edges"]`。

用户显式写下的 `pl.submit(..., deps=[...])` 边仍保存在
`Call.attrs["manual_dep_edges"]`。两个 attr 有意分开，以便 IR dump 保留来源；
orchestration codegen 在发出 `Arg::set_dependencies(...)` 前合并并去重。

## Pipeline 位置

```text
... -> DeriveCallDirections -> AutoDeriveTaskDependencies -> CollectCommGroups -> Simplify (final)
```

本 pass 只修改 manual runtime scope。Auto scope 继续使用 runtime OverlapMap，
行为不变。

## P0 算法

对每个函数体：

1. 为 tensor Var 构建保守 storage-root 映射。直接别名、loop carry、tuple 元素、
   `tensor.slice`、`tensor.assemble` 和跨函数输出在可追踪时继承同一 root。
2. 从 `pl.submit` tuple 尾部收集静态绑定的 producer TaskId。
3. 按源码顺序扫描每个 `RuntimeScopeStmt(manual=true)`，仅在该 manual scope 内维护
   prior accesses。
4. 对每个带有已解析 `arg_directions` 的非 builtin call，把 tensor 参数分类为
   read、write 或 read-write。P0 将同一 storage root 的访问都视为 may-overlap。
5. 对 RAW、WAR、WAW hazard，从先前 producer TaskId 添加 compiler edge；read-read
   不生成边。用户显式依赖保持权威且不会重复添加。

如果某个 hazard 需要的 prior producer 没有静态绑定 TaskId，pass 会抛出定向
`ValueError`，提示用户把 producer 写成 `out, tid = pl.submit(...)`。

## Properties

| Required | Produced | Invalidated |
| -------- | -------- | ----------- |
| `SplitIncoreOrch`, `CallDirectionsResolved` | `CallDirectionsResolved` | — |

本 pass 保持 `CallDirectionsResolved`：它只改依赖 attr，不改 call 参数或
`arg_directions`。

## API

| C++ | Python | Level |
| --- | ------ | ----- |
| `pass::AutoDeriveTaskDependencies()` | `passes.auto_derive_task_dependencies()` | Program-level |

## 参考

- Source: [src/ir/transforms/auto_derive_task_dependencies_pass.cpp](../../../../src/ir/transforms/auto_derive_task_dependencies_pass.cpp)
- Proposal: [Automatic Task Dependency Derivation](../proposals/auto_task_dependencies.md)
- Lowering: [Orchestration Code Generation](../codegen/01-orchestration_codegen.md#manual-scope-and-taskid-lowering)
