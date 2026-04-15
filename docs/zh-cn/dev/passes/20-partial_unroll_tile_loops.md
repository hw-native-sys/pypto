# PartialUnrollTileLoops Pass

在 tile 层级展开 `pl.range(N, unroll=F)` 循环：将循环体复制 `F` 份以启用 ping-pong 缓冲，同时保留外层顺序循环。

## 概述

`pl.unroll(N)` 在 SSA 之前的 slot #1 完整展开循环为 `N` 份副本。用户使用它通常并非需要 `N` 份副本，而是希望获得不同的 tile MemRef —— 否则 `MemoryReuse` 会把生命周期相邻的 tile 合并为同一缓冲区，导致 ping-pong 失效。

`PartialUnrollTileLoops` 提供更精细的开关：在 tile 层级把循环体复制 `F` 份（典型值 2–4），保留外层 `N/F` 次顺序迭代。每个副本获得独立的定义变量（保持 SSA），各自操作独立的 tile，下游 `MemoryReuse` 无法将其合并。

**前置条件**: SSAForm、SplitIncoreOrch、IncoreTileOps、TileOps2D、TileMemoryInferred、NormalizedStmtStructure。

**流水线位置**: 位于 `NormalizeReturnOrder` 之后、`InitMemRef` 之前（slot 20.5）。此时 tile 结构决策已完成；同时早于 `InitMemRef`/`MemoryReuse`，使其看到每个副本独立的 tile 变量。

## API

| C++ | Python | 级别 |
| --- | ------ | ---- |
| `pass::PartialUnrollTileLoops()` | `passes.partial_unroll_tile_loops()` | 函数级 |

```python
from pypto import passes
result = passes.partial_unroll_tile_loops()(program)
```

## DSL 语法

```python
# 每次外层迭代复制循环体 4 次；外层循环 16 次，步长为 4。
for i in pl.range(64, unroll=4):
    tile_x = pl.tile.load(input_a, [i * 128], [128])
    pl.tile.store(tile_x, [i * 128], output)
```

## 行为

对于 `attrs_["unroll_factor"] = F`、迭代次数 `T = (stop - start) / step` 的循环：

- **整除情形**（`T % F == 0`）：一个外层循环 `T/F` 次迭代，循环体为 `F` 份副本组成的 `SeqStmts`；外层循环带 `attrs_["unroll_replicated"] = F` 标记。
- **带余数情形**（`T % F != 0`）：外层复制循环覆盖 `(T // F) * F` 次迭代；剩余 `T % F` 次由原步长的余数循环承担。余数循环不带标记。
- **克隆细节**：每份副本通过 `DeepClone(body, {loop_var → new_var + k * step}, clone_def_vars=true)` 生成。每个副本拥有新鲜的定义变量，既保持 SSA，又给 `MemoryReuse` 提供独立的 tile 身份。

## 约束（首版）

| 约束 | 原因 |
| ---- | ---- |
| `start`、`stop`、`step` 必须为编译期整数常量 | 计算主循环 / 余数循环大小所需 |
| 不允许 `iter_args` / `init_values` | 跨副本的循环携带状态需要 SSA 重命名，首版未实现 |
| `unroll` 与 `chunk` 在 `pl.range` 中互斥 | 二者优化方向不同，组合使用语义模糊且无明显场景 |

## 示例

**变换前**（输入 IR，ForStmt 带 `unroll_factor=4` 属性）:

```python
for i in pl.range(0, 8, 1, attrs={"unroll_factor": 4}):
    tile_x = pl.tile.load(input_a, [i * 128], [128])
    pl.tile.store(tile_x, [i * 128], output)
```

**变换后**:

```python
for i in pl.range(0, 8, 4, attrs={"unroll_replicated": 4}):
    # k=0 副本
    tile_x_0 = pl.tile.load(input_a, [i * 128], [128])
    pl.tile.store(tile_x_0, [i * 128], output)
    # k=1 副本
    tile_x_1 = pl.tile.load(input_a, [(i + 1) * 128], [128])
    pl.tile.store(tile_x_1, [(i + 1) * 128], output)
    # k=2、k=3 副本类推
```

下游的 `ReorderUnrolledIO` Pass 会识别 `unroll_replicated` 标记的循环，将 load 上拉、store 下沉，让各副本的输入 tile 同时活跃，从而 `MemoryReuse` 不能合并它们。

## 相关

- [`ReorderUnrolledIO`](21-reorder_unrolled_io.md) —— 消费 `unroll_replicated` 标记
- [`UnrollLoops`](01-unroll_loops.md) —— slot #1 的全展开 Pass，仍是 `pl.unroll(N)` 的主要降级路径
- RFC #1025 —— 设计文档
