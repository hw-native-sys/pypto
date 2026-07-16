# LowerTileToPTOIR Pass

消费经过验证的逻辑值到 handle plan，并把受支持的逻辑 Tile 范围改写为目标传递
PTO 目标 IR。

## 概述

[`MaterializePTOTileHandles`](43-materialize_pto_tile_handles.md) 已经显式确定
allocation 和 destination。该 Pass 只做机械式改写：

| 逻辑操作 | 目标操作 |
| -------- | -------- |
| `tile.load(tensor, offsets)` | `pto.tload(tensor, offsets, extents, output_handle)` |
| `tile.sqrt(input)` | `pto.tsqrt(input_handle, output_handle)` |
| `tile.add(lhs, rhs)` | `pto.tadd(lhs_handle, rhs_handle, output_handle)` |
| `tile.mul(lhs, rhs)` | `pto.tmul(lhs_handle, rhs_handle, output_handle)` |
| `tile.row_sum(input, tmp)` | `pto.trowsum(input_handle, tmp_handle, output_handle)` |
| `tile.slice(input, ...)` | 删除逻辑结果，保留 Step 3 的 `pto.subview` |
| `tile.store(input, offsets, tensor)` | `pto.tstore(input_handle, offsets, extents, tensor)` |

目标 call 不返回逻辑值，并放在 `EvalStmt` 中。注册的 operand schema 描述读写、
元数据和 destination 位置。虽然目标操作会修改 handle 指向的存储，
`PTOTileBufType` handle 本身仍是 SSA 值。

逻辑 `tile.alloc` 指针 token 会被删除，因为它只用于挂载 Tile MemRef。发出带副作用
的目标 store 后，`tile.store` 的 Tensor 结果会改为普通 Tensor alias。

## 默认流水线位置

这是 `Default` 和 `DebugTileOptimization` 的最后一个 Pass。它运行在所有逻辑 Tile
消费者之后，产出 `PTOIRPrinter` 直接消费的 IR。前一 Pass 标记为
`pto.target_lowering_deferred` 的函数保持不变，并整体继续走旧逻辑 Tile codegen。

## 结果不变量

`PTOBufferized` 属性要求 InCore 目标函数：

- 不含逻辑 `TileType` 参数、赋值、返回值或 call；
- 每个 PTO buffer handle 都由 `pto.alloc_tile` 唯一定义；
- 每次 handle 使用都位于其支配性定义之后；
- allocation 与目标传递 op 使用规定的语句形式；
- 每个目标 op 都符合注册的 operand 和 result 契约；
- 保留结构化 `ForStmt`/`IfStmt` region，同时不让 Tile handle 进入 `scf` SSA
  result/carry。

该属性已加入默认验证属性集合。PTO codegen 检测到目标 IR 后，会把完整 Program
交给机械式 printer。
SPMD identity 的合成参数会显式出现在该目标签名中，但仍是 codegen 私有后缀；
orchestration 的调用参数个数只按函数公开参数计算。

## API

| C++ | Python |
| --- | ------ |
| `pass::LowerTileToPTOIR()` | `passes.lower_tile_to_pto_ir()` |

## Pass 属性

| - | 属性 |
| - | ---- |
| 前置 | `SSAForm`、`SplitIncoreOrch`、`IncoreTileOps`、`HasMemRefs`、`TileOps2D`、`TileMemoryInferred`、`NormalizedStmtStructure`、`PTOHandlesMaterialized` |
| 产出 | `SSAForm`、`SplitIncoreOrch`、`NormalizedStmtStructure`、`PTOBufferized` |
| 失效 | `IncoreTileOps`、`HasMemRefs`、`AllocatedMemoryAddr`、`TileOps2D`、`TileMemoryInferred`、`PTOHandlesMaterialized` |

## 限制

该 Pass 有意不分析 alias、不选择输出 buffer、不推断传输 extent，也不恢复地址；这些
决策必须来自经过验证的 Step 3 plan。因此，扩展操作覆盖时需要同步更新 op schema、
Step 3、Step 4、两个 property verifier、printer 支持和 PTOAS 测试。

## 另见

- [MaterializePTOTileHandles](43-materialize_pto_tile_handles.md)
- [显式 PTO 目标 IR](../codegen/02-explicit_pto_target_ir.md)
- [PTO codegen](../codegen/00-pto_codegen.md)
