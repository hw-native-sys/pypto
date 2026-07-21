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

本 pass 只处理**强制别名**。它从 [`MemoryReuse`](30-memory_reuse.md) 中拆出
（原来是那个 pass 的 "Step 0"），以便机会性的生命周期复用可以被独立跳过 ——
`MemoryPlanner.PTOAS` 下由 ptoas 接管规划，或 `MemoryPlanner.DSA` 下由独立 solver
联合选择复用与 offset。

**使用时机**：在 [`InitMemRef`](28-init_memref.md)（创建 MemRef）之后、
[`MemoryReuse`](30-memory_reuse.md) 之前运行。它总是运行；只有机会性复用可跳过。

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
3. **协调 external planner 剩余 carry**（`YieldFixupMutator`）：跳过 `MemoryReuse`
   时，为无法安全 retarget 的 producer/view shape 插入显式 move。PTOAS 在此物化
   loop-carry move，并在不带地址的 codegen 中处理 if-phi copy；DSA 会发出显式地址，
   因而必须在 lifetime export 前同时物化 if-phi 与 loop-carry move。

当没有可重定向的内容时（`Compute` 返回空）本 pass 是 no-op，并跳过
`Orchestration` 函数（无 TileType 变量）。

## 与 codegen 的关系

PTO codegen 把解析到*同一* MemRef 身份（`base` + `byte_offset` + `size`）的变量
渲染成同一个 `tile_buf` handle，因此本 pass 之后,循环累加器会发出原地的
`pto.tadd ins(%acc, %t) outs(%acc)`，而不是写到独立的 `%acc_next`。在
`memory_planner=PTOAS`（不烘焙物理 `addr`、跳过 `MemoryReuse`）下,这正是让 ptoas
`PlanMemory` 把累加器保持在一块 buffer、同时自己完成生命周期复用与地址分配的关键。
参见 [PTO 代码生成 — 由谁规划内存](../codegen/00-pto_codegen.md)。

DSA 不能使用 PTOAS 专用的 codegen repair，因为其 level-3 PTO 包含显式地址。因此，
if-phi copy 会保留为普通 IR operation，参与 DSA lifetime analysis，并与 placement
一起接受验证。

## 说明

- view / 部分 view 共享 `base` 但 `byte_offset`/`size` 不同,因此不会被并入强制
  别名 buffer —— 只有完全同一分配的变量才合并。
- 在默认（`PYPTO`）流水线里,本 pass 加上 `MemoryReuse` 组合起来等于原来单个
  `MemoryReuse` pass 的行为（字节级一致）。
