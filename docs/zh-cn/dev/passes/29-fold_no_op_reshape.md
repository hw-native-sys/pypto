# FoldNoOpReshape Pass

将既不改变物理形状也不改变分配的 `tile.reshape` 调用折叠为普通的 Var 到 Var 赋值，
让该 trivial reshape Call 在 PTO codegen 之前从 IR 中消失。

## 概述

`LegalizePTOBufferReuse` 运行之后，`tile.reshape` 的 LHS 与 RHS 可能已经指向同一
个 `MemRef` 根，并且具有相同的 `TileBufSignature`。在这种情况下，该 reshape 在
PTO 层面是 no-op —— 按 var 分配的模型已经为 LHS 预声明了与 RHS 相同的 shape、layout、
fractal、valid-shape 与 pad，并共享同一块内存地址。`pto.treshape` 在此无事可做。

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

**前置条件**：

- `IRProperty::SplitIncoreOrch` —— Orchestration 已从 InCore 中拆分
- `IRProperty::IncoreTileOps` —— InCore 函数使用 tile 类型
- `IRProperty::HasMemRefs` —— `MemRef` 槽已由 `InitMemRef` 填充
- `IRProperty::TileOps2D` —— tile op 至多 2D
- 该 Pass 必须在 `LegalizePTOBufferReuse` 之后运行，让 view 合并的决策反映在
  规范 alloc 上 —— 否则即便逻辑上应共享，LHS 与 RHS 也可能尚未指向同一个 `MemRef`。
- 仅扫描 InCore 类型函数（`InCore`、`AIC`、`AIV`）；Opaque 与 Orchestration
  函数原样返回。

**使用时机**：在 `Default` 策略中作为第 29 个 Pass 运行，紧随 `AllocateMemoryAddr`
之后（保证 `MemRef` 合并已经定型），先于 `FuseCreateAssembleToSlice`。

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
对于每一条 `value` 是 `tile.reshape` Call 的 `AssignStmt`，依次检查四个条件：

1. **LHS 与源都是 tile**：`assign.var.type` 与第一个参数的类型都能成功转为 `TileType`。
2. **两侧都有 MemRef**：双方的 `tile_type.memref_` 均已设置且不为空。
3. **同一个 MemRef 根**：`lhs_tile.memref->base.get() == rhs_tile.memref->base.get()`。
4. **签名相同**：`TileBufSignature::FromTileType(lhs) == TileBufSignature::FromTileType(rhs)`。

四者全部满足时，将 `AssignStmt(lhs, Call(tile.reshape, [src, shape]))` 替换为
`AssignStmt(lhs, src)`。Call 被整个丢弃；自该语句起 LHS 成为 RHS 的纯别名，下游使用
看到的 MemRef 与类型与之前完全一致。

该 Pass 不修改任何其它语句形式；任何在以上四点中存在差异的 reshape 都被保留 ——
这些情况需要真正的 `pto.treshape`。

| 源模式 | 行为 |
| ------ | ---- |
| `lhs = tile.reshape(rhs, shape)` 且同 MemRef、同 `TileBufSignature` | 改写为 `lhs = rhs`；丢弃 Call |
| `lhs = tile.reshape(rhs, shape)` 且 MemRef 根不同 | 不变 |
| `lhs = tile.reshape(rhs, shape)` 且 MemRef 相同但 `TileBufSignature` 不同 | 不变（真正的 reshape） |
| 任何非 `tile.reshape` Call | 不变 |
| Opaque / Orchestration 函数 | 原样返回 |

## 示例

### MemRef 合并后的 trivial reshape

```python
# Pass 之前（双方 TileBufSignature 相同；LegalizePTOBufferReuse 之后双方共享 MemRef R）
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

## 验证

**测试**：`tests/ut/ir/transforms/test_fold_no_op_reshape.py`

- `test_genuine_reshape_kept` —— 物理形状变化的 reshape 得以保留
- `test_pass_runs_without_error_on_simple_kernel` —— 无 reshape kernel 上的冒烟测试

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
