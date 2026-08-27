# InsertMxScaleAddr Pass

在每个 MX matmul 消费者之前插入编译器生成的 `tile.tget_scale_addr` 绑定；前提是操作数 memory space 已经全部确定。

## 概述

[`InferTileMemorySpace`](18-infer_tile_memory_space.md) 解析完 `Left` / `LeftScale` / `Right` / `RightScale` 并插入必要的 `tile.move` 之后，本 pass 物化 A5 的 scale 地址绑定：

```text
bound_scale = tile.tget_scale_addr(scale, data)
matmul_mx(..., bound_scale, ...)
```

`tile.tget_scale_addr` 有意不暴露在公共 `pypto.language` API 中。用户只编写高层 `matmul_mx` 算子族；编译器根据 matmul 操作数位置推导左右侧。低层 `ir.op.tile.tget_scale_addr` 仍供编译器构造与 IR 解析使用，且只接受已解析的 `(LeftScale, Left)` 与 `(RightScale, Right)` 配对，data tile 必须为 `FP8E4M3FN`。左侧 FP4 输入会在进入本 pass 前先转换为 FP8。

**流水线位置**：紧接在 [`InferTileMemorySpace`](18-infer_tile_memory_space.md) 之后，[`ResolveBackendOpLayouts`](20-resolve_backend_op_layouts.md) 之前。

**前置属性**：`SSAForm`、`IncoreTileOps`、`SplitIncoreOrch`、`NormalizedStmtStructure`、`TileMemoryInferred`。

**产出属性**：与前置相同（属性保持型重写）。

## API

| C++ | Python | Level |
| --- | ------ | ----- |
| `pass::InsertMxScaleAddr()` | `passes.insert_mx_scale_addr()` | Program-level |

```python
from pypto.pypto_core import passes

after = passes.insert_mx_scale_addr()(passes.infer_tile_memory_space()(program))
```

仅重写 `FunctionType::InCore` 函数。

## 算法

遍历每个 InCore 函数体。对每个 `tile.matmul_mx` / `tile.matmul_mx_acc` / `tile.matmul_mx_bias` 赋值：

1. 按算子配对 data/scale 操作数下标（`matmul_mx` / `_bias` 为 `(0,1)/(2,3)`；`_acc` 为 `(1,2)/(3,4)`）。
2. 要求操作数为 Var-like（`Var` 或 `IterArg`），且 memory space 已构成合法的 LeftScale↔Left 或 RightScale↔Right 配对。
3. 在 matmul 前插入 `tile.tget_scale_addr(scale, data)`，并把 matmul 改写为使用绑定后的 scale SSA。

`NormalizedStmtStructure` 可能 unwrap 单语句 `SeqStmts`，使 `if` / `for` / `while`（或函数）body 成为裸 `AssignStmt`。插入绑定时，pass 会把该 body 包成 `SeqStmts`，处理方式与 `InsertCommFence` 的裸 body 逻辑一致。

### 禁止跨消费者复用

`tget_scale_addr` 会原地修改物理 scale buffer 地址（`dst_addr = src_addr >> SHIFT_MX_ADDR`）。普通 SSA alias、view alias 与 bound result 都可能共享该 buffer，因此 SSA 身份无法证明缓存绑定仍然有效。pass 会为每个 MX matmul 消费者生成新的绑定。

即使 scale 操作数已经是更早一次 `tget_scale_addr` 的 bound result，也必须遵循此规则：该 result 仍与同一个可变 buffer alias，无法证明其地址未被其他 alias 改写。因此重复执行 pass 会保守地再增加一层绑定；标准 lowering pipeline 只运行该 pass 一次。

## 相关

- 算子注册与类型检查：`src/ir/op/tile_ops/matmul_mx.cpp`
- MX 操作数 memory space 求解：[`InferTileMemorySpace`](18-infer_tile_memory_space.md)
- PTOAS 可能把 `tget_scale_addr` 重排到 Mat→Scale `tmov` 之前（`PTOA5NormalizeTMovPass`）
