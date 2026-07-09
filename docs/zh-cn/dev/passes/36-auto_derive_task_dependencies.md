# AutoDeriveTaskDependencies Pass

## 概览

`AutoDeriveTaskDependencies` 在显式开启时，为 AUTO runtime scope 保守推导
task-to-task 依赖边。它运行在
[`DeriveCallDirections`](34-derive_call_directions.md) 之后，读取已经解析的
`Call.attrs["arg_directions"]`，并把编译器推导出的 producer TaskId 边写入
`Call.attrs["compiler_manual_dep_edges"]`。当某个 tensor slot 的 runtime 依赖查询已经被
同一个成功分析的 scope 内的用户显式边或编译器边完整覆盖时，本 pass 也可以改写部分
call-site direction：

- 只读 `ArgDirection::Input` -> `ArgDirection::NoDep`；
- 读写 `ArgDirection::InOut` -> `ArgDirection::OutputExisting`。

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

用户手写的 MANUAL 区域会保持原有调度契约：显式 `deps=[...]` 仍是唯一的 task
依赖边，本 pass 不会新增 compiler deps，也不会把 scope 降级为 AUTO，也不会把
call-site direction 自动改成 `NoDep` 或 `OutputExisting`。AUTO 区域只有在
编译期开关 `analyze_auto_scopes_for_deps` 开启时才会分析。手写的 AUTO `RuntimeScopeStmt`
在输出 IR 中仍保持 `manual=false`。对于默认 `auto_scope=True` 的
orchestration 函数，本 pass 运行在 `MaterializeRuntimeScopes` 之前；当 AUTO
分析开启时，它会把整个函数体当作 analysis-only 的虚拟 AUTO 区域来分析，但不会
自行插入或移动 scope wrapper。当分析证明默认模式下的整个函数体或后续 for/if
分区已经被编译器推导出的显式 deps 完整覆盖时，它会记录 compiler-only marker，
由 `MaterializeRuntimeScopes` 消费这些 marker，并为该区域发出编译器拥有的
MANUAL `RuntimeScopeStmt`。否则，`MaterializeRuntimeScopes` 会发出普通 AUTO
wrapper，runtime OverlapMap/TensorMap tracking 也继续启用。静态可表达的编译器
推导边会通过 `Arg::set_dependencies(...)` 叠加发出。

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
6. 从 `pl.submit` tuple 尾部收集静态绑定的 producer TaskId。对于
   `Array[TASK_ID]` 依赖值，只有在 lineage 已知时才会保守展开：直接 scalar
   写入、loop 内动态写入，以及由 per-element `arr[i]` deps 合成的数组，只有在覆盖
   所有静态槽位时才可以覆盖用户手写 hazard；来源不清楚或只覆盖部分槽位的数组不会展开，
   缺失依赖仍会触发 fallback。
7. 按源码顺序扫描每个 `RuntimeScopeStmt`，维护当前可分析区域的 prior accesses。
   一个 scope 如果没有 fallback，退出时会把 access 记录导出到外层区域，这样后续 sibling scope
   或父 scope 的 consumer 可以依赖这个 producer TaskId。对于尚未物化 scope 的默认
   `auto_scope=True` orchestration 函数，把整个函数体当作虚拟 AUTO 分析区域。对 AUTO scope
   来说这只是分析层行为；最终 scope mode 仍保持 AUTO，除非
   `MaterializeRuntimeScopes` 随后消费了某个完整覆盖的默认模式区域上的 compiler
   auto-manual marker。
8. 对每个带有已解析 `arg_directions` 的非 builtin call，把 tensor 参数分类为
   read、write 或 read-write。同一 storage root，或 MemRef root 之间可能 alias 的
   访问，会继续进入 region overlap 判断。
9. 对静态证明 disjoint 的 region 跳过依赖边。否则，对 RAW、WAR、WAW hazard 从先前
   producer TaskId 添加 compiler edge；read-read 不生成边。用户显式依赖保持权威且
   不会重复添加。
10. 对 analyzed AUTO scope 中的只读 `Input` 参数，如果它的精确静态 region（`Full` 或
    `Box`）或已知 root 的 conservative region 的 RAW hazard，已经被同一个成功分析的
    runtime scope 内的用户边或编译器边覆盖，则把 call-site direction 改成 `NoDep`，
    让 codegen 生成 `add_no_dep(...)`。
11. 对 analyzed AUTO scope 中的 `InOut` 参数，如果这个 slot 的所有重叠 RAW、WAR、
    WAW hazard 都已经被用户边或编译器边覆盖，则把 call-site direction 改成
    `OutputExisting`。这样会跳过该 slot 的 TensorMap lookup，但仍然把写结果作为 output
    发布。MANUAL scope、`OutputExisting` slot、slice window 依赖运行时值的
    dynamic-window region、没有完整用户 fan-in 覆盖的 dynamic producer，以及跨 scope
    场景保持原方向。

如果 analyzed AUTO scope 中 dependency-relevant tensor access 无法表达成有界静态
roots 加固定 TaskId deps，pass 会从整个 enclosing 区域中剥离任何部分编译器推导
deps，并保持 AUTO tracking。已实现的 fallback 触发条件包括：

- 必须建边的 hazard 对应的 prior producer 没有静态绑定 TaskId；
- prior producer 位于 loop 内，并且它的 TaskId 不能表示成固定 loop-return 值或静态大小的
  TaskId 数组 fan-in；
- dynamic gather/scatter 类 tensor value，其访问 region 依赖运行时 index；
- root-set lineage 超过 pass 允许的静态 alternatives 上限；
- 带 read/write direction 的 tensor argument 无法通过当前 lineage analysis 解析
  storage location。

这样整段 AUTO 区域都会回到 runtime OverlapMap/TensorMap tracking，而不会在 scope
边界混用部分 compiler deps 与 runtime 状态。用户手写的 MANUAL scope 会被本 pass
直接跳过，所以 AUTO fallback 逻辑不适用于它们。

## 默认路径变化

- MANUAL scope 不会获得编译器推导出的依赖边，也不会应用自动 direction 改写。用户手写的
  `deps=[...]` 是 `pl.manual_scope()` 内唯一的依赖来源，scope 保持 MANUAL。
- AUTO-scope 分析需要显式开启。默认开关值下，AUTO runtime scope mode 和
  TensorMap/OverlapMap tracking 保持不变。
- Dead scalar assignment elimination 在所有构建中都会保留 TaskId tuple-element
  extract。这可能留下以前会被删掉的廉价 scalar TaskId local，方便 dependency
  derivation/codegen 恢复 producer task id。

## Properties

| Required | Produced | Invalidated |
| -------- | -------- | ----------- |
| `SplitIncoreOrch`, `CallDirectionsResolved` | `CallDirectionsResolved` | — |

本 pass 保持 `CallDirectionsResolved`：它不改 call 参数；如果修改 `arg_directions`，
也只在 analyzed AUTO scope 中做 verifier 合法的 direction 降级，并且对应 tensor slot
已经有显式依赖边兜底：对覆盖完整的只读输入做 `Input -> NoDep`，对覆盖完整的读写 slot
做 `InOut -> OutputExisting`。

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
