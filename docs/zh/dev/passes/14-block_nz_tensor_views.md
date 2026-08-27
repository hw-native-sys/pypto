# BlockNzTensorViews Pass

## 概述

`BlockNzTensorViews` 把*逻辑* `pl.NZ` 张量改写成 pto-isa `Layout::NZ`
GlobalTensor 所要求的*分块*形式，并同步改写读取它的 `tile.load`。

`pl.Tensor[[..., R, C], dtype, pl.NZ]` 标注是一条**关于 GM 中现有字节的断言**：
这些字节已经按 PTO 原生 NZ 分形序存放。它**不是**"请帮我转换"的请求。DSL 层保持
逻辑 shape 和逻辑切片不变，本 pass 负责补上后端需要的物理描述。

收益是 matmul B 操作数的 `TLOAD` 从 ND→NZ 变成 NZ→NZ，去掉了每次权重载入时的
在线分形转换。

## 分块形式

记 `c0` 为一条 32 字节 C0 线所含的元素个数（`256 / dtype 位宽`，`INT8` 时为 32），分形行数为 16。pto-isa 对 NZ
缓冲的描述为（`pto/common/pto_tile.hpp` 中 `TileShape2D` / `BaseShape2D` 的
`Layout::NZ` 特化）：

```text
shape   = [..., C/c0, R/16, 16, c0]
strides = [..., C*R,  R*c0, 16*c0, c0, 1]
```

从内往外读：`c0` 个连续元素构成一条 32 字节 C0 线，16 行构成一个 `16 x c0` 分形
（512 字节），`R/16` 个分形沿行轴排列，**最外层**才在列块之间步进。即"列块在外、
行分形在内"——与 tile 侧的 `blayout=col_major, slayout=row_major, fractal=512`
描述的是同一个字节序。

### 为什么 NZ 不需要自己的 stride 规则

对分块 shape 求普通行主序 stride，结果*就是* pto-isa 的 NZ stride：

| 槽位 | 行主序推导 | `BaseShape2D<T, R, C, NZ>` |
| ---- | ---------- | -------------------------- |
| `c0` | `1` | `1` |
| `16` | `c0` | `C0Size` |
| `R/16` | `16*c0` | `FRACTAL_NZ_ROW*C0Size` |
| `C/c0` | `(R/16)*16*c0 = R*c0` | `rows*C0Size` |
| 前导维 | `(C/c0)*R*c0 = C*R` | `cols*rows` |

因此一旦 shape 被分块，NZ 就是行主序家族的普通成员，
`BuildLogicalStridesFromLayout` 通过与 ND 相同的 `BuildRowMajorStrides` 路径处理
它。stride 由 `MaterializeTensorStrides`（pass 31）稍后填充；本 pass 只改写 shape。

这修正了 RFC #1300 中"NZ 没有 logical-stride 表示"的结论——该结论对逻辑 2-D shape
成立，对分块后的 rank-(r+2) shape 不成立。

## 在流水线中的位置

```text
... -> LowerCompositeOps -> FlattenTileNdTo2D -> BlockNzTensorViews -> LegalizeTileCast -> ...
```

三条约束确定了这个位置：

- **在 `ConvertTensorToTileOps` / `LowerCompositeOps` 之后**——阶段 2 要改写的
  `tile.load` 必须已经存在。
- **在 `FlattenTileNdTo2D` 之后**——目标 tile 必须已经是逻辑 2-D 操作数。对仍是
  ND 秩的 tile 做分块，会产生类型标注与参数秩无法同时打印的 `tile.load`，破坏
  printer 往返。
- **在 `MaterializeTensorStrides` 之前**——该 pass 断言每个 NZ view 已分块，然后
  填充其行主序 stride。

`FlattenTileNdTo2D` 对 NZ 源跳过它的 ND2NZ 源窗口塌缩（该塌缩存在是因为 ND→NZ
需要 2-D GlobalTensor，而 NZ→NZ 不需要），所以本 pass 运行时逻辑窗口仍然完整。

## 行为

**阶段 1 —— 对每个 NZ `TensorType` 的 shape 分块。**

```text
# before
w: pl.Tensor[[32, 2048, 4096], pl.INT8, pl.NZ]

# after  (c0 = 32:  4096/32 = 128,  2048/16 = 128)
w: pl.Tensor[[32, 128, 128, 16, 32], pl.INT8, pl.NZ]
```

**阶段 2 —— 改写消费它的 `tile.load`。**

```text
# before （从 [E, N, K] 权重中切出 w[1:2, 256:512, 512:1024]）
wt: pl.Tile[[256, 512], pl.INT8, pl.Mem.Mat] =
    pl.tile.load(w, [1, 256, 512], [1, 256, 512], target_memory=pl.Mem.Mat)

# after  （offsets -> [.., k0/c0, n0/16, 0, 0]；sizes -> 分块形式）
wt: pl.Tile[[256, 512], pl.INT8, pl.Mem.Mat] =
    pl.tile.load(w, [1, 16, 16, 0, 0], [1, 16, 16, 16, 32], target_memory=pl.Mem.Mat)
```

末尾两个 offset 必须是**常量**。里程碑 1 只对 `ConstInt` 做 `k0/c0` / `n0/16` 映射，
因此即使符号表达式可证明对齐也会被拒绝——要映射它需要整除性证明加代数改写
（`nb*256` -> `nb*16`），目前没有实现。

所以**尚不支持循环推导出的切片**：

```python
for nb in pl.spmd(N // N_TILE):
    n0 = nb * N_TILE
    wt = w[n0 : n0 + N_TILE, 0:K_TILE]   # 被拒绝：shape[-2] 上是动态 offset
```

这是本里程碑与催生它的 grouped-matmul 权重路径之间最主要的差距，已记录为 issue #2548。

目标 `TileType` **原样保留**：GM 分区变成 rank-(r+2)，而 tile 保持逻辑 2-D 操作数。
因此该 load 使用显式类型的 `Call` 构造函数重建，而不是 `OpRegistry::Create`——后者
会从分块后的 shapes 参数重新推导出 rank-(r+2) 的 tile。

本 pass 之后不再有逻辑 shape 的 NZ `TensorType` 存活，因此下游（包括 codegen）
都不需要知道 NZ 的特殊性——codegen 会分别从 `TensorType::shape_` 推导
`pto.make_tensor_view`、其 `!pto.tensor_view<>` 类型以及 `pto.partition_view` 的秩，
三者必须一致。

## 生成代码

```mlir
%w_view = pto.make_tensor_view %arg1,
    shape = [%c16, %c16, %c16, %c32], strides = [%c8192, %c512, %c32, %c1]
    {layout = #pto.layout<nz>} : !pto.tensor_view<?x?x?x?xi8>
%w_pview = pto.partition_view %w_view,
    offsets = [%c0, %c0, %c0, %c0], sizes = [%c16, %c16, %c16, %c32]
    : !pto.tensor_view<?x?x?x?xi8> -> !pto.partition_tensor_view<16x16x16x32xi8>
pto.tload ins(%w_pview : !pto.partition_tensor_view<16x16x16x32xi8>)
          outs(%wt : !pto.tile_buf<loc=mat, dtype=i8, rows=256, cols=512,
                                   blayout=col_major, slayout=row_major, fractal=512, ...>)
```

## 范围与拒绝项

里程碑 1 的范围是刻意收窄的。范围之外一律报错并指明修复方式——NZ 张量绝不能被
静默地错误寻址。

| 条件 | 结果 |
| ---- | ---- |
| `shape[-2] % 16 != 0` | 拒绝——不完整的分形没有表示 |
| `shape[-1] % c0 != 0` | 拒绝——不完整的 C0 线没有表示 |
| `shape[-2]` / `shape[-1]` 为动态维 | 拒绝——无法静态证明整除 |
| 切片偏移未对齐到分形边界 | 拒绝——没有分块表示 |
| **末尾切片偏移为动态（非常量）** | **拒绝——只映射 `ConstInt` 偏移；即使符号表达式可证明对齐也一并拒绝（#2548）** |
| rank < 2 | 拒绝 |
| `target_memory != Mat`（或缺省） | 拒绝——NZ→NZ 是 cube 操作数路径 |
| `tile.load` 之外的消费者 | 拒绝——此处 NZ 是只读的 |
| 显式 stride 或部分 `valid_shape` | 拒绝 |
| 分布式张量 | 拒绝——`remote_load` 没有 NZ 分块 |
| 对 NZ 做 `tensor.view` / `tensor.reinterpret_view` | 在算子构造期拒绝 |

sub-byte dtype（INT4 / UINT4 / FP4 / HF4 / BOOL）被拒绝，这是 **PyPTO 里程碑 1 的
范围限制，不是硬件限制**——pto-isa 的 NZ 机制确实处理 FP4（`tload_common.hpp` 中有
显式的 `caps::IsFP4` 分支，并断言 `staticShape[4] == C0_SIZE_BYTE / sizeof(DType)`）。
`c0` 已按位宽推导，因此等 packed-nibble 寻址端到端验证完成后，算术部分即可直接复用。

对齐类诊断是面向用户的（`CHECK_SPAN` → `ValueError`），位于 `BlockNzShape`。
到了下游，*未分块*的 NZ view 属于 pass 顺序不变量（`INTERNAL_CHECK_SPAN`），由
`MaterializeTensorStrides` 中的 `CheckNzViewIsBlocked` 和 `TensorViewCanonical`
验证器强制执行。

## 幂等性

分块**不是**幂等的——对已分块的 shape 再分块是错的——而结构性的
`IsBlockedNzShape` 判断无法区分"已分块的 shape"和"恰好以 `[16, c0]` 结尾的逻辑
shape"。因此本 pass 会在每个改写过的函数上打 `nz_tensor_views_blocked` 标记，
再次进入时直接返回。

## 下游依赖

PTOAS 通过结构推断 `make_tensor_view` 的 layout。分块 NZ 与 ND 在结构上完全相同
（都是行主序），因此 PTOAS 目前会推断出 `nd` 并覆盖显式的 `nz` 标注，报
`layout mismatch: user-specified layout=nz but inferred=nd`。这个失败是安全的而非
静默的：在分块 shape 下，pto-isa 的 ND→NZ `TLOAD` 路径要求
`staticShape[0..2] == 1`，而分块维度违反了它，所以生成的 C++ 会在 `static_assert`
上编译失败，而不是算出错误结果。端到端可用需要等 PTOAS 信任显式标注。

## 相关文档

- [13-flatten_tile_nd_to_2d.md](13-flatten_tile_nd_to_2d.md) —— 对 NZ 源跳过 ND2NZ 窗口塌缩
- [31-materialize_tensor_strides.md](31-materialize_tensor_strides.md) —— 填充分块 NZ stride
- [../ir/02-types.md](../ir/02-types.md) —— `TensorLayout` 与 `TensorView`
