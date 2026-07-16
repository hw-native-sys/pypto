# MaterializePTOTileHandles Pass

在保留逻辑 `TileType` SSA 的同时物化显式 PTO tile-buffer handle。它是值语义 Tile
IR 与目标传递 PTO 目标 IR 之间经过验证的桥接层。

## 概述

该 Pass 位于 codegen 边界，运行在所有逻辑 Tile 优化、内存规划和 orchestration
Pass 之后。它会为每个受支持的 Tile producer：

1. 创建一个支配该 producer 的 `pto.alloc_tile`，结果类型为 `PTOTileBufType`；
2. 把逻辑 Tile operand 映射为有序输入 handle；
3. 把逻辑 Tile 结果映射为唯一输出 handle；
4. 把临时 plan 写入逻辑 call，供下一个 Pass 消费。

此阶段仍保留逻辑 call 和 `TileType` 结果，因此可以在删除任何逻辑 Tile 定义之前
检查过渡状态，并由 `PTOHandlesMaterialized` 验证完整映射。

## 默认流水线位置

该 Pass 默认加入 `Default` 和 `DebugTileOptimization`，紧跟
[`ClassifyIterArgCarry`](42-classify_iter_arg_carry.md)，并位于
[`LowerTileToPTOIR`](44-lower_tile_to_pto_ir.md) 之前。它只修改 InCore 函数。

如果函数包含目标 IR 尚未覆盖的 Tile 操作族，Pass 会在修改前标记
`pto.target_lowering_deferred`，随后整个函数继续走旧 codegen。这个显式的整函数边界
避免产生逻辑/目标混合 IR。

## Allocation operand

内存规划器只影响物理地址 operand：

| Planner | `pto.alloc_tile` operand |
| ------- | ------------------------ |
| `PYPTO` | byte offset（`i64`）、有效行（`index`）、有效列（`index`） |
| `PTOAS` | 有效行（`index`）、有效列（`index`） |

`AllocatedMemoryAddr` 不是无条件 Pass 属性，因为 PTOAS 会有意跳过地址分配。PYPTO
模式下，Pass 会检查每个 Tile 是否带有含常量 byte offset 的 MemRef，否则报错。

## 支持范围

当前实现支持静态物理 rank-2 Tile 和静态/动态 valid extent。已覆盖
load/store、create/full、slice（`pto.subview`）、带 allocation 的 reshape、基本
一元/二元/tile-scalar 运算、move/fillpad 和 row sum；并递归规划 `ForStmt`、
`IfStmt` 中的 Tile carry/phi alias 与必要 move。在 PTOAS 下，可写的 region 局部
producer 会直接改写到共享 phi/carry handle；pass-through 值和只含元数据的 view
仍保留显式 move。Tensor 传输可保留 rank-N 分区元数据。`pto.subview` 会逐维把常量
valid extent 写入结果类型，运行时维度仍保持动态。

Tile 参数/返回值、动态 shape Tensor 参数、`WhileStmt`、跨核 pipe、一般归约和其他
Tile 操作族会整函数 deferred。

## 验证

`PTOHandlesMaterialized` 验证：

- 每个受支持的逻辑 Tile producer 恰好有一个输出 handle；
- 每个逻辑 Tile operand 都有符合顺序的输入 handle；
- 每个 handle 恰好由一个支配其使用的已注册 handle op（`pto.alloc_tile` 或
  `pto.subview`）定义；
- allocation 元数据与所选 planner 一致；
- 结构化 carry/phi alias 指向已定义 handle，region 内使用满足支配关系；
- deferred 函数中不存在插入一半的目标 handle。

该属性已加入默认验证属性集合，正常流水线会在 Pass 边界执行验证。

## API

| C++ | Python |
| --- | ------ |
| `pass::MaterializePTOTileHandles()` | `passes.materialize_pto_tile_handles()` |

## Pass 属性

| - | 属性 |
| - | ---- |
| 前置 | `SSAForm`、`SplitIncoreOrch`、`IncoreTileOps`、`HasMemRefs`、`TileOps2D`、`TileMemoryInferred`、`NormalizedStmtStructure` |
| 产出 | 全部前置属性，以及 `PTOHandlesMaterialized` |
| 失效 | — |

## 另见

- [LowerTileToPTOIR](44-lower_tile_to_pto_ir.md)
- [显式 PTO 目标 IR](../codegen/02-explicit_pto_target_ir.md)
- [PTO codegen](../codegen/00-pto_codegen.md)
