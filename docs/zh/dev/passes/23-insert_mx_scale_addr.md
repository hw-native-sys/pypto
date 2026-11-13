# InsertMxScaleAddr Pass

在每个 MX matmul 消费者之前插入编译器生成的 `tile.tget_scale_addr` 绑定；前提是操作数 memory space 已经全部确定。

## 概述

[`InferTileMemorySpace`](22-infer_tile_memory_space.md) 解析完 `Left` / `LeftScale` / `Right` / `RightScale` 并插入必要的 `tile.move` 之后，本 pass 物化 A5 的 scale 地址绑定：

```text
bound_scale = tile.tget_scale_addr(scale, data)
matmul_mx(..., bound_scale, ...)
```

`tile.tget_scale_addr` 有意不暴露在公共 `pypto.language` API 中。用户只编写高层 `matmul_mx` 算子族；编译器根据 matmul 操作数位置推导左右侧。低层 `ir.op.tile.tget_scale_addr` 仍供编译器构造与 IR 解析使用，且只接受已解析的 `(LeftScale, Left)` 与 `(RightScale, Right)` 配对，data tile 为 `FP8E4M3FN` 或受支持的原生 packed FP4。

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
3. 只有两个 SSA 操作数均未变化、且中间没有使绑定失效的操作时，才复用该侧最近的绑定；否则在 matmul 前插入 `tile.tget_scale_addr(scale, data)`。把 matmul 改写为使用绑定后的 scale SSA。

`NormalizedStmtStructure` 可能 unwrap 单语句 `SeqStmts`，使 `if` / `for` / `while`（或函数）body 成为裸 `AssignStmt`。插入绑定时，pass 会把该 body 包成 `SeqStmts`，处理方式与 `InsertCommFence` 的裸 body 逻辑一致。

### 保守的绑定复用

`tget_scale_addr` 会原地修改物理 scale buffer 地址（`dst_addr = src_addr >> SHIFT_MX_ADDR`）。在 scale 消费者之后重复相同绑定，会使 PTOAS 尝试跨越该消费者，将绑定前移到最初的 Mat→Scale 填充之前；PTOAS 0.70 会拒绝这种前移。因此，pass 仅在直线代码中分别保留 Left 和 Right 侧最近的 scale/data 绑定。

由于别名（alias）可能共享可变 buffer，仅比较 SSA 身份还不够。不同的配对会替换该侧状态；任何接收或产生 Left/LeftScale 或 Right/RightScale tile 的非 MX 操作都会使对应侧失效。未知调用和控制流边界会清空两侧状态。Mat 加载和 Acc 存储不会使这些绑定失效。状态只占常量空间，遍历仍为线性复杂度。

已有的显式 `tget_scale_addr` 也会使状态失效。因此重复执行 pass 会保守地再增加一层绑定；标准 lowering pipeline 只运行该 pass 一次。

### 跨控制流区域的驻留 scale

scale 可以在 panel 循环外填充，再由循环内的 MX matmul 消费。仅在消费点绑定地址并不充分：
`tget_scale_addr` 只改变描述符地址，不会搬运此前写入临时分配地址的数据。因此，地址绑定必须先于填充的 `tmov`。

Tile IR 的 scale producer 是函数式 `tile.move`，结果变量在该赋值前尚未定义。因此，本 pass 保留满足 SSA 的
consumer 绑定，由 PTO 代码生成在声明目标句柄之后、发出填充指令之前，建立驻留 producer 的地址绑定。
函数级索引用于确认跨区域的数据／scale 配对唯一；具体数据变量的描述符（descriptor）及其句柄必须已在支配当前填充点的活动区域中定义。
仅与已分配的基础 buffer 共享存储并不足够：reshape 或其他数据视图必须在 scale 填充前定义。
这也包括已生成的数据视图：保留视图自身的声明区域和物理偏移。当 `if` 的两个分支都直接返回同一个支配视图时，
代码生成还会保留该视图已有的有效形状，而不对它执行仅适用于分配句柄的有效形状恢复。packed-FP4 数据描述符按字节载体计数，
因此比较有效形状前，需要将配对 scale 的逻辑有效长度换算为载体单位；普通 packed 驻留 tile 不需要元数据视图。
已有的 consumer 绑定继续保留，以维持后续重新绑定行为。

跨区域配对不唯一，或数据句柄仅在消费区域内部定义时，不能无条件外提绑定。代码生成会诊断这些暂不支持的情况，
避免静默生成写入地址错误的驻留填充。同一区域内的填充继续使用现有 PTOAS 归一化路径。
若元数据别名需要提前建立描述符视图，当前要求 producer 有效形状为静态值，且物理数据存储不使用 packed 格式。
对于从 `if` 返回的 scale，或循环替换携带 scale 后返回的结果，当前尚不能证明唯一的 producer/data 配对。
代码生成会明确拒绝这些结果描述符；应在消费区域内填充 scale。直接自回传的不变循环携带仍受支持。
跨区域消费零拷贝 scale 视图也会被拒绝，包括 `tile.slice` 和 `tile.reinterpret_view`，因为当前提前绑定尚不能保留视图来源；应在消费区域内填充视图后的 scale。

## 相关

- 算子注册与类型检查：`src/ir/op/tile_ops/matmul_mx.cpp`
- MX 操作数 memory space 求解：[`InferTileMemorySpace`](22-infer_tile_memory_space.md)
- PTOAS 可能把 `tget_scale_addr` 重排到 Mat→Scale `tmov` 之前（`PTOA5NormalizeTMovPass`）
