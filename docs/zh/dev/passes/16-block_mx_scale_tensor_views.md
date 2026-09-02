# BlockMxScaleTensorViews Pass

## 概述

`BlockMxScaleTensorViews` 把逻辑 MX scale tensor view 转换为 A5 所要求的
rank-5 SFractal 紧致形式，并改写所有坐标依赖这一物理形式的操作。

这里是一次**物理 view lowering 的迁移**：原先后端中的
`EmitMxPhysicalView` 专用逻辑被迁移到显式 IR pass。此后 tensor type、load
窗口、别名、stride 物化、验证器与通用 codegen 都只看到同一种规范表示；
codegen 不再私下重建另一套 MX view。

本 pass 与 `BlockNzTensorViews` 相互独立。MX 与 NZ 是不同 layout，转换和证明
实现也分别维护。

## 物理形式

DSL 暴露两种逻辑 rank-2 scale layout：

```text
MX_A_ZZ [M, G] -> [1, M/16, G/2, 16, 2]
MX_B_NN [G, N] -> [1, N/16, G/2, 16, 2]
```

末尾 `[16, 2]` 是一个 32 字节 FP8E8M0 scale box。对分块 shape 求行主序
stride 就得到物理 GlobalTensor stride，因此本 pass 之后
`MaterializeTensorStrides` 可以直接使用普通紧致 stride 路径。

## 在流水线中的位置

```text
... -> FlattenTileNdTo2D -> BlockNzTensorViews
    -> BlockMxScaleTensorViews -> LegalizeTileCast -> ...
```

本 pass 在 `FlattenTileNdTo2D` 之后运行，此时 `tile.load` 的结果已经是逻辑
2-D tile；它位于所有要求物理 MX tensor shape 的消费者之前。
`MaterializeTensorStrides` 随后填入 rank-5 行主序 stride。

## 改写内容

对每个 MX_A_ZZ 或 MX_B_NN tensor，本 pass 改写：

- 参数、返回值、tuple、变量、迭代参数、Call 与 Submit 中递归出现的
  `TensorType` 槽位；
- `tile.load` 的 offset 和 shape，使其使用 rank-5 坐标；
- 物理 `valid_shape` 参数，使其保持完整对齐 load box，同时把缩窄的逻辑
  `TileType.valid_shape` 保留为 tile 元数据；
- 两个方向的 FP8E8M0 有 shape `tensor.view` 别名：ND-to-MX 与 MX-to-ND；
- Submit 返回类型，同时保持依赖、关键字参数、属性、core 数、predicate 和同步字段不变。

load 的目标 `TileType` 仍是逻辑 2-D；只有 GM 源分区变为 rank-5。

## Offset 映射与证明

逻辑坐标映射如下：

```text
MX_A_ZZ [m0, g0] -> [0, m0/16, g0/2, 0, 0]
MX_B_NN [g0, n0] -> [0, n0/16, g0/2, 0, 0]
```

常量必须非负且对齐。符号 offset 只有在本 pass 私有的 MX 证明引擎同时证明
整除性与非负性后才会被接受。证明引擎支持：

- scalar SSA 定义；
- 常量、乘法、加法，以及用于整除证明的减法；
- 正的 2 的幂 floor division，包括 `k0 // 32`；
- start 与 step 为常量的循环变量；
- `tile.get_block_idx` 和 `tile.get_block_num` 的非负结果；
- 经每个 Call 与 Submit 调用点传播的 scalar 参数。

callee 参数必须在每一个 caller 中都可证明。调用映射缺失或畸形、递归，以及超出
256 步有界证明预算的表达式都会被保守拒绝。该上界使 pass 保持 O(N)，且证明失败
绝不会退化成“假定已对齐”。

商以 `FloorDiv(offset, divisor)` 发出，保证原定宽表达式先按原语义求值，再做除法。

## 范围与诊断

| 条件 | 结果 |
| ---- | ---- |
| 静态、对齐的 rank-2 MX shape | 转换为规范 rank-5 形式 |
| 对齐且可证明非负的符号 offset | 转换为 rank-5 坐标 |
| 无法证明或为负的 offset | 拒绝 |
| tensor 级部分 `valid_shape` | 拒绝 |
| load 级缩窄 `valid_shape` | 保留为 tile 元数据；物理 box 仍完整 |
| `target_memory != Mat` 或缺失 | 拒绝 |
| MX tensor 被不支持的算子使用或作为 store 目标 | 拒绝 |
| 有 shape 的 FP8E8M0 ND/MX backing alias | 改写 |
| distributed MX tensor | 拒绝 |

pass 完成后会给函数写入 `mx_tensor_views_blocked` 属性。这一来源标记使第二次运行
直接成为 no-op，无需从可能碰巧长得像分块形式的 shape 猜测 pass 是否已经执行。

## 另请参阅

- [BlockNzTensorViews](15-block_nz_tensor_views.md)
- [MaterializeTensorStrides](33-materialize_tensor_strides.md)
- [InsertMxScaleAddr](21-insert_mx_scale_addr.md)
