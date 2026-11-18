# InsertMxScaleAddr Pass

操作数的存储空间全部确定后，在 MX matmul 和 MX GEMV 消费者之前插入编译器生成的 `tile.tget_scale_addr` 绑定。

## 概述

[`InferTileMemorySpace`](22-infer_tile_memory_space.md) 解析完 `Left` / `LeftScale` / `Right` / `RightScale` 并插入必要的 `tile.move` 之后，本 pass 物化 A5 的 scale 地址绑定：

```text
bound_scale = tile.tget_scale_addr(scale, data)
matmul_mx(..., bound_scale, ...)
```

用户调用公开的 `matmul_mx` 或 `gemv_mx` 算子族。编译器根据操作数位置推导左右侧。低层 `ir.op.tile.tget_scale_addr` 供编译器构造与 IR 解析使用，接受已解析的 `(LeftScale, Left)` 和 `(RightScale, Right)` 配对，数据使用FP8E4M3FN、FP8E5M2 或原生打包 FP4，缩放使用 FP8E8M0。

**流水线位置**：紧接在 [`InferTileMemorySpace`](22-infer_tile_memory_space.md) 之后，[`ResolveBackendOpLayouts`](24-resolve_backend_op_layouts.md) 之前。

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

重写所有 InCore 变体函数（`FunctionType::InCore` / `AIC` / `AIV`）。Mixed kernel 与前端直接书写的 cube/vector 函数走 AIC/AIV，也必须插入绑定。

## 算法

遍历每个 InCore 变体函数体，处理 `tile.matmul_mx`、`tile.matmul_mx_acc`、`tile.matmul_mx_bias`、`tile.gemv_mx`、`tile.gemv_mx_acc` 和 `tile.gemv_mx_bias` 赋值。

1. 配对数据和缩放操作数下标。两个算子族的基础和偏置形式均为 `(0,1)/(2,3)`，累加形式为 `(1,2)/(3,4)`。
2. 要求操作数为 Var-like（`Var` 或 `IterArg`），且 memory space 已构成合法的 LeftScale↔Left 或 RightScale↔Right 配对。
3. 只有两个 SSA 操作数均未变化、且中间没有使绑定失效的操作时，才复用该侧最近的绑定。否则在消费者前插入 `tile.tget_scale_addr(scale, data)`，并将消费者改写为使用绑定后的 scale SSA。

`NormalizedStmtStructure` 可能 unwrap 单语句 `SeqStmts`，使 `if` / `for` / `while`（或函数）body 成为裸 `AssignStmt`。插入绑定时，pass 会把该 body 包成 `SeqStmts`，处理方式与 `InsertCommFence` 的裸 body 逻辑一致。

### 保守的绑定复用

`tget_scale_addr` 会原地修改物理 scale buffer 地址（`dst_addr = src_addr >> SHIFT_MX_ADDR`）。在 scale 消费者之后重复相同绑定，会使 PTOAS 尝试跨越该消费者，将绑定前移到最初的 Mat→Scale 填充之前；PTOAS 0.70 会拒绝这种前移。因此，pass 仅在直线代码中分别保留 Left 和 Right 侧最近的 scale/data 绑定。

由于别名（alias）可能共享可变 buffer，仅比较 SSA 身份还不够。不同的配对会替换该侧状态；任何接收或产生 Left/LeftScale 或 Right/RightScale tile 的非 MX 操作都会使对应侧失效。未知调用和控制流边界会清空两侧状态。Mat 加载和 Acc 存储不会使这些绑定失效。状态只占常量空间，遍历仍为线性复杂度。

已有的显式 `tget_scale_addr` 也会使状态失效。因此重复执行 pass 会保守地再增加一层绑定；标准 lowering pipeline 只运行该 pass 一次。

## 相关

- 算子注册与类型检查：`src/ir/op/tile_ops/matmul_mx.cpp`
- MX 操作数 memory space 求解：[`InferTileMemorySpace`](22-infer_tile_memory_space.md)
- PTOAS 可能把 `tget_scale_addr` 重排到 Mat→Scale `tmov` 之前（`PTOA5NormalizeTMovPass`）
