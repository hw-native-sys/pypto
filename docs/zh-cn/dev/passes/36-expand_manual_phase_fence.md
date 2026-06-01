# ExpandManualPhaseFence Pass

## 概览

`ExpandManualPhaseFence` 会压缩显式 manual-scope deps 携带的、
有收益的完整数组 `TaskId` 依赖。它是一个很窄的 orchestration-only pass：
当 manual-scope consumer fanout 只依赖一个稳定、只读的 `Array[TASK_ID]` 时，本 pass
插入一个 dependency-only 的 `system.task_dummy` barrier，并把覆盖到的 consumer
改写为依赖该 barrier 的 `TaskId`。

依赖形状会从重复的 all-to-all fanout：

```text
tids[N] -> consumers[M]
```

变为显式 phase fence：

```text
tids[N] -> system.task_dummy -> consumers[M]
```

本 pass 不改变 kernel 执行语义。它只改写选中的 consumer call 上的
`Call.attrs["manual_dep_edges"]`，并插入一个带标记的 dummy-task call；后续
codegen 会把该 call lowering 为 `rt_submit_dummy_task(...)`。在 outline 之后，
它会统一覆盖带 `manual_dep_edges` 的 orchestration call，包括
`pl.submit(..., deps=[...])` 和 `pl.at(..., deps=[...])` 这两类形状。

## Pipeline 位置

```text
... -> DeriveCallDirections -> ExpandManualPhaseFence -> CollectCommGroups -> Simplify（最终）
```

`DeriveCallDirections` 必须先运行，使 call 已经带有解析好的 `arg_directions`，
同时 parser / outline 产生的 `manual_dep_edges` 已经可见。`ExpandManualPhaseFence`
运行在最终分布式元数据收集之前，也在 orchestration codegen 观察
`manual_dep_edges` 之前。

## 算法

对每个 orchestration 函数，本 pass 会访问 `RuntimeScopeStmt(manual=true)` 区域，
并分析其中的每个 loop body：

1. **寻找候选数组。** 候选 consumer 必须只有一个 `manual_dep_edges` entry，
   且该 entry 必须是 `Array[TASK_ID]`。
2. **估算收益。** pass 比较直接 fanout（`N * M`）和 barrier 形状（`N + M`）。
   `N -> 1`、`2 -> 2` 等低收益形状继续保持直接依赖。
3. **拒绝不安全形状。** mixed deps、scalar deps、无法解析的数组、当前 loop
   iter-arg 数组、body 内定义的数组、通过同存储 `Array[TASK_ID]` alias 更新的数组、
   非 manual scope、非 orchestration 函数都会跳过。
4. **插入 barrier。** 对有收益且安全的候选，pass 创建一个新的
   `Scalar[TASK_ID]` 变量，并赋值为 `system.task_dummy`；该 call 带有
   `attrs["dummy_task"] = true` 和 `attrs["manual_dep_edges"] = [source_array]`。
5. **改写 consumer。** 覆盖到的 consumer call 会以
   `manual_dep_edges=[barrier_tid]` 重建，其它 call attr 保持不变。

对 sequential loop 和 parallel loop，只要候选被接受，barrier 都插在改写后的 loop
之前。这样稳定的 sequential dependency 只会为整个 loop 提交一次 dummy task，而不是每
iteration 提交一次。已知 trip count 为零的 sequential loop 不会插入 barrier。

safety index 保守处理 nested loop `return_vars_`、nested body update，以及
transitive `Array[TASK_ID]` iter-arg alias class。Nested loop summary 会缓存后合并到父
loop 分析中，因此本 pass 不会为了每个候选依赖数组反复扫描同一个 nested body。
`pl.parallel` 不会削弱 `manual_scope` 中用户显式写出的依赖：如果 body 读取
`deps=[tids]` 同时又更新 `tids[branch]` 或 `tids` 的 alias，本 pass 会保留直接依赖。

## Fallback 边界

除非模式明确、安全且有收益，本 pass 会保留已有的直接 dependency lowering 路径。

会压缩的形状：

- 有正向 estimated edge savings 的完整数组 manual-scope fanout；
- double-buffered phase fence，即 body 读取一个 `Array[TASK_ID]`，写入另一个
  carrier，例如 `tids_next`。

保持直接依赖的形状：

- 标量 TaskId deps；
- mixed scalar + array deps；
- multiple-array deps；
- `prev = tids[i]; deps=[prev]` 这类 partial-slot deps；
- 当前 loop 的 iter-arg 数组；
- 在同一个 loop body 内定义或更新的数组；
- 通过 transitive `Array[TASK_ID]` iter-arg alias 更新的数组；
- nested loop return var 被用作依赖数组的情况；
- 已知 trip count 为零的 loop；
- `N -> 1` 或 `2 -> 2` 这类低收益 fanout；
- 非 manual scope 和非 orchestration 函数。

## 输出不变式

pass 运行后：

- 每个插入的 barrier 都是带 `attrs["dummy_task"] = true` 标记的
  `system.task_dummy` call；
- barrier call 自己仍在 `attrs["manual_dep_edges"]` 中保留原始完整数组依赖；
- 被改写的 consumer 依赖 barrier `TaskId`，而不是原始数组；
- fallback 形状保留原始 `manual_dep_edges`；
- `arg_directions` 仍保持已解析状态，本 pass 不会重新推导它们。

## Pass 属性

| 字段 | 值 |
| ---- | -- |
| `required` | `{NoNestedCalls, NormalizedStmtStructure, CallDirectionsResolved}` |
| `produced` | `{NoNestedCalls, NormalizedStmtStructure, CallDirectionsResolved}` |
| `invalidated` | `{}` |

## 参考

- 实现：[src/ir/transforms/expand_manual_phase_fence_pass.cpp](../../../../src/ir/transforms/expand_manual_phase_fence_pass.cpp)
- 头文件：[include/pypto/ir/transforms/passes.h](../../../../include/pypto/ir/transforms/passes.h)
- Attr key：[include/pypto/ir/expr.h](../../../../include/pypto/ir/expr.h)
- Codegen lowering：[src/codegen/orchestration/orchestration_codegen.cpp](../../../../src/codegen/orchestration/orchestration_codegen.cpp)
- 示例：
  [examples/utils/phase_fence_dep_compression.py](../../../../examples/utils/phase_fence_dep_compression.py)
- 测试：
  [tests/ut/ir/transforms/test_expand_manual_phase_fence.py](../../../../tests/ut/ir/transforms/test_expand_manual_phase_fence.py)，
  [tests/ut/codegen/test_phase_fence_dep_compression.py](../../../../tests/ut/codegen/test_phase_fence_dep_compression.py)
