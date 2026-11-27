# InsertMxScaleAddr Pass

在操作数内存空间确定后，物化 A5 MX scale 地址绑定。

## 概述

[`InferTileMemorySpace`](22-infer_tile_memory_space.md) 解析完 `Left` / `LeftScale` / `Right` / `RightScale` 并插入必要的 `tile.move` 之后，本 pass 在写入 scale 数据前为符合条件的 Mat→Scale extract 绑定目标地址。

```text
dst = tile.bind_scale_addr(data, shape, valid_shape, target_memory=LeftScale)
scale = tile.extract_into(dst, scale_panel, row, col)
matmul_mx(..., scale, ...)
```

`tile.bind_scale_addr` 创建 scale tile 句柄，地址从对应的 Left/Right payload
派生。实际搬运由 `tile.extract_into` 完成，其返回的 SSA 值与传入的
目标句柄共享存储。两条 IR 分别发射为 `pto.tget_scale_addr` 和
`pto.textract`。其他生产者仍使用 `tile.tget_scale_addr(scale, data)`。
地址操作接受已解析的 `(LeftScale, Left)` 和 `(RightScale, Right)` 配对，
支持 FP8E4M3FN、FP8E5M2 与打包 FP4 payload。新增操作可通过 IR builder 构造，
用于 pass dump 与 IR 解析。

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

遍历每个 InCore 变体函数体。对每个 `tile.matmul_mx` / `tile.matmul_mx_acc` / `tile.matmul_mx_bias` 赋值：

1. 按算子配对 data/scale 操作数下标（`matmul_mx` / `_bias` 为 `(0,1)/(2,3)`；`_acc` 为 `(1,2)/(3,4)`）。
2. 要求操作数为 Var-like（`Var` 或 `IterArg`），且 memory space 已构成合法的 LeftScale↔Left 或 RightScale↔Right 配对。
3. 如果 scale 来自同一直线区域内的 Mat→Scale extract，且 payload 在
   extract 处已可用，就在原位置改写为绑定和显式写入。同侧 L0
   的中间使用、定义或控制流边界会让候选失效。两侧独立跟踪。
4. 对其他生产者，复用相同 scale/data SSA 配对的最近有效绑定，或
   在 matmul 前插入 `tile.tget_scale_addr(scale, data)`。matmul 消费
   绑定后的 scale SSA。

`NormalizedStmtStructure` 可能 unwrap 单语句 `SeqStmts`，使 `if` / `for` / `while`（或函数）body 成为裸 `AssignStmt`。插入绑定时，pass 会把该 body 包成 `SeqStmts`，处理方式与 `InsertCommFence` 的裸 body 逻辑一致。

### 保守的绑定复用

`tget_scale_addr` 会原地修改物理 scale buffer 地址（`dst_addr = src_addr >> SHIFT_MX_ADDR`）。在 scale 消费者之后重复相同绑定，会使 PTOAS 尝试跨越该消费者，将绑定前移到最初的 Mat→Scale 填充之前；PTOAS 0.70 会拒绝这种前移。因此，pass 仅在直线代码中分别保留 Left 和 Right 侧最近的 scale/data 绑定。

由于别名（alias）可能共享可变 buffer，仅比较 SSA 身份还不够。不同的配对会替换该侧状态；任何接收或产生 Left/LeftScale 或 Right/RightScale tile 的非 MX 操作都会使对应侧失效。未知调用和控制流边界会清空两侧状态。Mat 加载和 Acc 存储不会使这些绑定失效。状态只占常量空间，遍历仍为线性复杂度。

已有的显式 `tget_scale_addr` 也会使状态失效。因此重复执行 pass 会保守地再增加一层绑定；标准 lowering pipeline 只运行该 pass 一次。

## 相关

- 算子注册与类型检查：`src/ir/op/tile_ops/matmul_mx.cpp`
- MX 操作数 memory space 求解：[`InferTileMemorySpace`](22-infer_tile_memory_space.md)
- PTOAS 可能把 `tget_scale_addr` 重排到 Mat→Scale `tmov` 之前（`PTOA5NormalizeTMovPass`）
