# MaterializeSemanticAliases Pass

将**语义要求**必须是同一块分配的 buffer 归一到同一个 MemRef —— 通过把每个循环
carried 的 `iter_arg`/`initValue` MemRef 沿 yield/producer 链向下传播来实现。

## 概述

内存规划区分两种 buffer 共享：

- **强制别名（语义要求）：** 循环累加器、或原地算子的结果**必须**落在同一块
  buffer——写"下一个"值必须更新 carried buffer，否则循环无法累加。这是正确性,
  不是优化。
- **机会别名（可选）：** 生命周期不冲突的两块独立 buffer *可以*共享存储以省内存,
  属于优化。

本 pass 只处理**强制别名**。它从 [`MemoryReuse`](33-memory_reuse.md) 中拆出
（原来是那个 pass 的 "Step 0"），以便机会性的生命周期复用可以被独立跳过：

- `MemoryPlanner.DSA_RP` 保留独立分配身份，交给进程内 DSA-RP 求解器放置。
- `MemoryPlanner.PTOAS` 把生命周期复用和地址分配交给 ptoas。

**使用时机**：在 [`InitMemRef`](31-init_memref.md)（创建 MemRef）之后、所选内存
规划器之前运行。它总是运行。`PYPTO` 随后运行
[`MemoryReuse`](33-memory_reuse.md)；`DSA_RP` 在
[`AllocateMemoryAddr`](34-allocate_memory_addr.md) 中消费这些分配身份。

## API

| C++ | Python | 级别 |
| --- | ------ | ---- |
| `pass::MaterializeSemanticAliases()` | `passes.materialize_semantic_aliases()` | 函数级 |

```python
from pypto.pypto_core import passes

program = passes.materialize_semantic_aliases()(program)
```

## 算法

`InitMemRef` 已经让循环 carried 的 `iter_arg` 和 `return_var` 与 `initValue`
（累加器 buffer）共享同一 MemRef，但 yield 值的*生产者* —— 例如计算 `acc_next`
的 `tile.add` —— 仍被分配了自己的新 MemRef。本 pass 补上这个缺口：

1. **自顶向下重定向**（`TopDownRetargeter`）：对每个 `ForStmt`，取每个 `iter_arg`
   的规范 MemRef 作为目标，推送到 yield 值及其 producer 链上（跟随原地
   `output-reuses-input` 算子与 view 输入）。`IfStmt` 的返回值被推送到两个分支的
   yield。
2. **应用重定型**（`RetypeApplier`）：就地改写收集到的变量类型，使生产者直接写入
   carried buffer。

当没有可重定向的内容时（`Compute` 返回空）本 pass 是 no-op，并跳过
`Orchestration` 函数（无 TileType 变量）。

## 与 codegen 的关系

PTO codegen 把解析到*同一* MemRef 身份（`base` + `byte_offset` + `size`）的变量
渲染成同一个 `tile_buf` handle，因此本 pass 之后,循环累加器会发出原地的
`pto.tadd ins(%acc, %t) outs(%acc)`，而不是写到独立的 `%acc_next`。
`memory_planner=DSA_RP` 会把每个所得分配身份变成一个 DSA buffer；
`memory_planner=PTOAS` 则让 codegen 不带物理地址发射该身份，交给 ptoas
`PlanMemory`。
参见 [PTO 代码生成 — 由谁规划内存](../codegen/00-pto_codegen.md)。

## 说明

- view / 部分 view 保留各自的 `byte_offset`/`size` 元数据。在 `DSA_RP` 下，共享
  同一 `base` 的所有成员属于同一个物理分配；规划器整体移动该分配，并在回写时保留
  每个成员的相对偏移。
- 在默认（`PYPTO`）流水线里,本 pass 加上 `MemoryReuse` 组合起来等于原来单个
  `MemoryReuse` pass 的行为。
- `DSA_RP` 与 `PTOAS` 都跳过这里的机会性 MemRef 合并；二者都不能撤销本 pass
  建立的强制别名关系。
