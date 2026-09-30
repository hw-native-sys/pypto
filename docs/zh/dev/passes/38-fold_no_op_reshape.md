# FoldNoOpReshape Pass

将既不改变物理形状也不改变分配的 `tile.reshape` 调用折叠为普通的 Var 到 Var 赋值，
让该 trivial reshape Call 在 PTO codegen 之前从 IR 中消失。

## 概述

`InitMemRef` 与 `MaterializeSemanticAliases` 完成分配身份之后，`tile.reshape` 的
LHS 与 RHS 可能已经指向同一个 `MemRef` 根，并且具有相同的 `TileBufSignature`。
在这种情况下，该 reshape 在
PTO 层面是 no-op —— 按 var 分配的模型已经为 LHS 预声明了与 RHS 相同的 shape、layout、
fractal、valid-shape 与 pad，并共享同一个分配身份。`pto.treshape` 在此无事可做。

历史上 PTO codegen 在发射阶段识别这种情况并通过 peephole 静默丢弃 `pto.treshape`
那一行。这把一个 IR 到 IR 的优化藏在了 codegen 层；该 Pass 把这种优化挪到它该在的
地方，将：

```python
lhs: pl.Tile[..., MemRef(R)] = pl.tile.reshape(rhs, [...])  # rhs 与 lhs 同 MemRef + 同 sig
```

改写为：

```python
lhs: pl.Tile[..., MemRef(R)] = rhs
```

PTO codegen 之后对所有幸存的 `tile.reshape` 都做 1:1 翻译，因为 no-op 的情况已经
在上游被折掉了。

该 Pass 还会折叠**无缓冲区（buffer-less）** tile 上的恒等 reshape —— 即跨核
`tile.tpop_from_aic` / `tile.tpop_from_aiv` 的结果，或由其派生的视图，`InitMemRef`
会让它们保持无 MemRef。由于没有分配可比较，判定依据是两侧 tile 类型结构相等。这一
折叠关乎正确性而不只是精简：幸存的 reshape 会降级为 `pto.treshape` 视图，而该操作
没有 `valid_row` / `valid_col` 操作数，因此其上的符号 `valid_shape` 在运行时从未被
设置，`pto.tstore` 等消费者读到的是未初始化的有效范围。此类 tile 上的恒等 reshape
（例如升秩的 `[16, 128] -> [16, 1, 128]`，`FlattenTileNdTo2D` 会将其折回
`[16, 128]`）是 no-op，折叠后消费者直接读取弹出的 tile，其 `pto.tpop_from_*` 以
操作数形式携带运行时有效范围。真正改变形状的 reshape 仍会降级为 `pto.treshape`；
其上的符号 `valid_shape` 不在本 Pass 的处理范围内。

**前置条件**：

- `IRProperty::SplitIncoreOrch` —— Orchestration 已从 InCore 中拆分
- `IRProperty::IncoreTileOps` —— InCore 函数使用 tile 类型
- `IRProperty::HasMemRefs` —— `MemRef` 槽已由 `InitMemRef` 填充
- `IRProperty::TileOps2D` —— tile op 至多 2D
- `MaterializeSemanticAliases` 必须先完成语义要求的共享。`PYPTO` 与 `DSA_RP`
  在该 Pass 前运行 `AllocateMemoryAddr`；`PTOAS` 会故意跳过地址分配，但同根身份已经
  足够，因为 ptoas 必须把两者放在同一个分配中。
- 仅扫描 InCore 类型函数（`InCore`、`AIC`、`AIV`）；Opaque 与 Orchestration
  函数原样返回。

**使用时机**：在所选规划器的内存阶段之后（PyPTO 负责放置时为
`AllocateMemoryAddr`，否则为已经定型的语义别名），先于
`FuseCreateAssembleToSlice`。

## API

| C++ | Python | 级别 |
| --- | ------ | ---- |
| `pass::FoldNoOpReshape()` | `passes.fold_no_op_reshape()` | Function 级 |

**Python 用法**：

```python
from pypto.pypto_core import passes

fold_pass = passes.fold_no_op_reshape()
program_folded = fold_pass(program)
```

## 算法

对每一个 InCore 类型函数（其它原样返回），`FoldNoOpReshapeMutator` 遍历其函数体。
对于每一条 `value` 是 `tile.reshape` Call 的 `AssignStmt`，首先要求 **LHS 与源都是
tile**：`assign.var.type` 与第一个参数的类型都能成功转为 `TileType`。随后按 MemRef
情况走两条路径之一：

**两侧都有 MemRef** —— 再检查三个条件：

1. **MemRef 均非空**：双方的 `tile_type.memref_` 均已设置且不为空。
2. **同一个 MemRef 根**：`CompareBaseAddress(lhs_memref, rhs_memref)` 为 `kSame`。
3. **签名相同**：`TileBufSignature::FromTileType(lhs) == TileBufSignature::FromTileType(rhs)`。

**两侧都没有 MemRef**（无缓冲区的 tpop 结果或其派生视图）—— 一个条件：

1. **类型结构相等**：`structural_equal(lhs.type, rhs.type)`。这比签名检查更严格，是
   有意为之：`TileBufSignature` 把所有符号有效维度都只记为"dynamic"，因此
   `[vr, 128]` 与 `[vc, 128]` 会被判为相等，而别名会让 LHS 表示源的运行时有效范围。

任一路径的条件满足时，将 `AssignStmt(lhs, Call(tile.reshape, [src, shape]))` 替换为
`AssignStmt(lhs, src)`。Call 被整个丢弃；自该语句起 LHS 成为 RHS 的纯别名，下游使用
看到的 MemRef 与类型与之前完全一致。只有一侧有 MemRef 的 reshape 从不折叠。

该 Pass 不修改任何其它语句形式；任何不满足其路径条件的 reshape 都被保留 ——
这些情况需要真正的 `pto.treshape`。

| 源模式 | 行为 |
| ------ | ---- |
| `lhs = tile.reshape(rhs, shape)` 且同 MemRef、同 `TileBufSignature` | 改写为 `lhs = rhs`；丢弃 Call |
| `lhs = tile.reshape(rhs, shape)` 且 MemRef 根不同 | 不变 |
| `lhs = tile.reshape(rhs, shape)` 且 MemRef 相同但 `TileBufSignature` 不同 | 不变（真正的 reshape） |
| `lhs = tile.reshape(rhs, shape)`，两侧都无 MemRef 且类型结构相等 | 改写为 `lhs = rhs`；丢弃 Call |
| `lhs = tile.reshape(rhs, shape)`，两侧都无 MemRef 但类型不同 | 不变（真正的视图） |
| `lhs = tile.reshape(rhs, shape)` 且仅一侧有 MemRef | 不变 |
| 任何非 `tile.reshape` Call | 不变 |
| Opaque / Orchestration 函数 | 原样返回 |

## 示例

### MemRef 合并后的 trivial reshape

```python
# Pass 之前（双方 TileBufSignature 相同；语义别名及可选的 PyPTO 放置完成后共享 MemRef R）
@pl.function(type=pl.FunctionType.InCore)
def kernel(x, out):
    a: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec, MemRef(R)] = pl.tile.load(x, ...)
    b: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec, MemRef(R)] = pl.tile.reshape(a, [64, 64])
    pl.tile.store(b, [0, 0], out)
```

```python
# FoldNoOpReshape 之后
@pl.function(type=pl.FunctionType.InCore)
def kernel(x, out):
    a: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec, MemRef(R)] = pl.tile.load(x, ...)
    b: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec, MemRef(R)] = a   # Var 到 Var
    pl.tile.store(b, [0, 0], out)
```

PTO codegen 不再为该情况看到 reshape Call。下游 `Simplify` 之类的 Pass 可以进一步
内联这个别名。

### 真正的 reshape 不会被折叠

```python
# 物理形状不同 —— 不应被折叠
a: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec, MemRef(R)] = pl.tile.load(x, ...)
b: pl.Tile[[4096, 1], pl.FP32, pl.Mem.Vec, MemRef(R)] = pl.tile.reshape(a, [4096, 1])
```

`TileBufSignature::FromTileType` 对 `a` 与 `b` 产出的 `rows`/`cols` 不同，
`lhs_sig == rhs_sig` 为假，该 Call 被保留。PTO codegen 会发射真正的 `pto.treshape`。

### 带运行时有效范围的跨核 tile 上的恒等 reshape

```python
# DSL（InCore）：cv 是移到 Vec 的 cube 结果，有效行数为 `vr`
r = pl.reshape(cv, [16, 1, 128])
back = pl.reshape(r, [16, 128])
pl.store(back, [0, 0], out)
```

经过 `FlattenTileNdTo2D` 与 `ExpandMixedKernel` 后，两个 reshape 都是无 MemRef 的
弹出 tile 上的恒等 reshape：

```python
# Pass 之前（AIV）
cv: pl.Tile[[16, 128], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[vr, 128])] = pl.tile.tpop_from_aic(split=0)
r: pl.Tile[[16, 128], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[vr, 128])] = pl.tile.reshape(cv, [16, 128])
back: pl.Tile[[16, 128], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[vr, 128])] = pl.tile.reshape(r, [16, 128])
```

```python
# FoldNoOpReshape 之后
r: pl.Tile[[16, 128], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[vr, 128])] = cv
back: pl.Tile[[16, 128], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[vr, 128])] = r
```

store 现在直接读取弹出的 tile，其 `pto.tpop_from_aic(%vr, %c128)` 设置了运行时有效
范围。若不折叠，每个 reshape 都会变成 `v_row=?, v_col=?` 且有效范围从未被设置的
`pto.treshape`，store 将一行也不写。

## 验证

**测试**：`tests/ut/ir/transforms/test_fold_no_op_reshape.py`

- `test_noop_reshape_is_folded` —— 同 MemRef、同签名的 reshape 被折叠
- `test_genuine_reshape_kept` —— 物理形状变化的 reshape 得以保留
- `test_same_allocation_different_window_is_kept` —— 同一分配基址、不同偏移并非同一个值
- `test_buffer_less_identity_reshape_is_folded` —— 无 MemRef 的 tpop 结果上带符号有效
  形状的恒等 reshape 被折叠
- `test_buffer_less_reshape_with_a_different_valid_extent_is_kept` —— 签名相同但符号
  有效范围不同时不折叠
- `test_buffer_less_shape_change_is_kept` —— 无 MemRef tile 上改变形状的视图得以保留
- `test_pass_runs_without_error_on_simple_kernel` —— 无 reshape kernel 上的冒烟测试

端到端地，`tests/ut/codegen/test_pto_codegen_tpop_view.py::test_identity_reshape_over_dynamic_valid_tpop_reads_the_popped_tile`
在两种内存规划器下检查上例不发射 `pto.treshape`，且 `pto.tstore` 读取弹出的 tile。

历史上 codegen 端那个丢弃 no-op reshape 发射的 peephole 暂时保留作为兜底；待该
Pass 在线上运行一段时间后由后续提交移除。

## Pass Properties

| 属性 | 值 |
| ---- | -- |
| Required | `SplitIncoreOrch`、`IncoreTileOps`、`HasMemRefs`、`TileOps2D` |
| Produced | — |
| Invalidated | — |

该 Pass 保留所有输入属性：仅把一条 `AssignStmt` 的值从 Call 改为 Var，两侧仍是相同
的 `TileType`。SSA 形式、类型检查、MemRef 绑定、tile op 形状约束均不受影响。

## Scope

| 函数类型 | 行为 |
| -------- | ---- |
| InCore（InCore、AIC、AIV） | 扫描；命中的 no-op reshape 被折叠 |
| Orchestration | 原样返回 |
| Opaque | 原样返回 |

任何 InCore 类型函数都不含可折叠的 `tile.reshape` AssignStmt 时，该 Pass 是 no-op。
