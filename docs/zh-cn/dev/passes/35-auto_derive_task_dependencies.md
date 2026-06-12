# AutoDeriveTaskDependencies Pass

## 概览

`AutoDeriveTaskDependencies` 在显式开启时，为 AUTO runtime scope 保守推导
task-to-task 依赖边。它运行在
[`DeriveCallDirections`](34-derive_call_directions.md) 之后，读取已经解析的
`Call.attrs["arg_directions"]`，并把编译器推导出的 producer TaskId 边写入
`Call.attrs["compiler_manual_dep_edges"]`。

用户显式写下的 `pl.submit(..., deps=[...])` 边仍保存在
`Call.attrs["manual_dep_edges"]`。两个 attr 有意分开，以便 IR dump 保留来源；
orchestration codegen 在发出 `Arg::set_dependencies(...)` 前合并并去重。

## Pipeline 位置

```text
... -> DeriveCallDirections
    -> AutoDeriveTaskDependencies
    -> ExpandManualPhaseFence
    -> CollectCommGroups
    -> Simplify (final)
```

用户手写的 MANUAL 区域会被跳过：显式 `deps=[...]` 是用户完整的调度契约，
本 pass 不会新增 compiler deps，也不会把 scope 降级为 AUTO。AUTO 区域只有在
编译期开关 `analyze_auto_scopes_for_deps` 开启时才会分析。手写的 AUTO `RuntimeScopeStmt`
在输出 IR 中仍保持 `manual=false`。对于默认 `auto_scope=True` 的
orchestration 函数，本 pass 运行在 `MaterializeRuntimeScopes` 之前；当 AUTO
分析开启时，它会把整个函数体当作 analysis-only 的虚拟 AUTO 区域来分析，但不会
插入或移动 scope wrapper。codegen 仍由 `MaterializeRuntimeScopes` 产生的
scope 发出 `PTO2_SCOPE()`，runtime OverlapMap/TensorMap tracking 也继续启用。
静态可表达的编译器推导边会通过 `Arg::set_dependencies(...)` 叠加发出。

## 算法

对每个函数体：

1. 为 tensor Var 构建保守 storage-location 映射。直接别名、loop carry、tuple
   元素、`tensor.assemble` 和跨函数输出在可追踪时继承同一 storage root 和
   region。
2. 通过合并有限分支 root set，为 `IfStmt.return_vars` 保留 storage lineage。
   不同分支 root 会作为候选 root 保留，而不是丢弃。同一 root 的 region 相同则保留；
   region 不同时拓宽为 unknown。
3. 为 loop 和 while 的 return var 合并初始 carried value 与循环体末尾
   `pl.yield_()` 的 lineage，然后把 region 拓宽为 unknown，因为最终来源受控制流影响。
4. 只对裸 tensor 或 packed ND `TensorView` tensor，把常量矩形 `tensor.slice`
   window 记录为相对 storage root 的 region。shape/offset 含符号表达式、strided
   view、非 ND layout、`valid_shape` 或 padding 的 slice 会回退为 unknown region，
   并保守视为重叠。
5. 对带 MemRef 的 shaped value，如果 `MemRef::MayAlias` 判断它们来自同一 base
   且字节范围重叠或包含符号 offset，则视为可能 alias。
6. 从 `pl.submit` tuple 尾部收集静态绑定的 producer TaskId。
7. 按源码顺序扫描每个 `RuntimeScopeStmt`，仅在该 scope 内维护 prior accesses。
   对尚未物化 scope 的默认 `auto_scope=True` orchestration 函数，把整个函数体
   当作虚拟 AUTO 分析区域。对 AUTO scope 来说这只是分析层行为；最终 scope mode
   仍保持 AUTO。
8. 对每个带有已解析 `arg_directions` 的非 builtin call，把 tensor 参数分类为
   read、write 或 read-write。同一 storage root，或 MemRef root 之间可能 alias 的
   访问，会继续进入 region overlap 判断。
9. 对静态证明 disjoint 的 region 跳过依赖边。否则，对 RAW、WAR、WAW hazard 从先前
   producer TaskId 添加 compiler edge；read-read 不生成边。用户显式依赖保持权威且
   不会重复添加。

如果 analyzed AUTO scope 中 dependency-relevant tensor access 无法表达成有界静态
roots 加固定 TaskId deps，pass 会从整个 enclosing 区域中剥离任何部分编译器推导
deps，并保持 AUTO tracking。已实现的 fallback 触发条件包括：

- 必须建边的 hazard 对应的 prior producer 没有静态绑定 TaskId；
- prior producer 位于 loop 内，一个 scalar TaskId 无法代表所有迭代形成的 runtime
  fan-in；
- dynamic gather/scatter 类 tensor value，其访问 region 依赖运行时 index；
- root-set lineage 超过 pass 允许的静态 alternatives 上限；
- 带 read/write direction 的 tensor argument 无法通过当前 lineage analysis 解析
  storage location。

这样整段 AUTO 区域都会回到 runtime OverlapMap/TensorMap tracking，而不会在 scope
边界混用部分 compiler deps 与 runtime 状态。这个 fallback 不适用于用户手写的
MANUAL scope，因为本 pass 不分析 MANUAL scope。

## 默认路径变化

- MANUAL scope 不会被分析。用户手写的 `deps=[...]` 是 `pl.manual_scope()` 内
  唯一的依赖来源，scope 保持 MANUAL。
- AUTO-scope 分析需要显式开启。默认开关值下，AUTO runtime scope mode 和
  TensorMap/OverlapMap tracking 保持不变。
- Dead scalar assignment elimination 在所有构建中都会保留 TaskId tuple-element
  extract。这可能留下以前会被删掉的廉价 scalar TaskId local，方便 dependency
  derivation/codegen 恢复 producer task id。

## Properties

| Required | Produced | Invalidated |
| -------- | -------- | ----------- |
| `SplitIncoreOrch`, `CallDirectionsResolved` | `CallDirectionsResolved` | — |

本 pass 保持 `CallDirectionsResolved`：它只改依赖 attr，不改 call 参数或
`arg_directions`。

## API

- C++: `pass::AutoDeriveTaskDependencies()`
- Python: `passes.auto_derive_task_dependencies()`
- Level: program-level

## 参考

- Source: [pass source][pass-source]
- Proposal: [Automatic Task Dependency Derivation](../proposals/auto_task_dependencies.md)
- Lowering: [Orchestration Code Generation][orchestration-lowering]

[pass-source]: ../../../../src/ir/transforms/auto_derive_task_dependencies_pass.cpp
[orchestration-lowering]: ../codegen/01-orchestration_codegen.md#manual-scope-and-taskid-lowering
