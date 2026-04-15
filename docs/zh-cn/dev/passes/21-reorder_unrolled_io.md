# ReorderUnrolledIO Pass

在每个 `unroll_replicated` 区域内，把 `tile.load` 上拉到顶部、`tile.store` 下沉到底部 —— 受 SSA 依赖图约束 —— 从而启用对称的 ping-pong 缓冲。

## 概述

`PartialUnrollTileLoops` 生成的外层 `ForStmt` 体是 `F` 份克隆体的 `SeqStmts`，自然顺序为 `[load_0, compute_0, store_0, load_1, compute_1, store_1, …]`。这种布局下，相邻克隆的 tile 生命周期不重叠，`MemoryReuse` 会把它们合并为同一缓冲区，ping-pong 失效。

本 Pass 重排每个标记的 `SeqStmts`：

- 每个 `tile.load` 上拉到依赖图允许的最早位置。
- 每个 `tile.store` 下沉到依赖图允许的最晚位置。
- 计算语句留在中间。

只要数据流允许，结果即为 `[loads…, compute…, stores…]`。各克隆的输入 tile 在顶部同时活跃，输出 tile 在底部同时活跃 —— `MemoryReuse` 无法合并它们，每个克隆保留独立的 MemRef，从而 ping-pong 缓冲成为可能。

**前置条件**: SSAForm、SplitIncoreOrch、IncoreTileOps、TileOps2D、TileMemoryInferred、NormalizedStmtStructure。

**流水线位置**: 位于 `PartialUnrollTileLoops` 之后、`InitMemRef` 之前（slot 20.6）。在 `InitMemRef` 之前运行可保留 SSAForm，依赖分析正常工作。

## API

| C++ | Python | 级别 |
| --- | ------ | ---- |
| `pass::ReorderUnrolledIO()` | `passes.reorder_unrolled_io()` | 程序级 |

```python
from pypto import passes
result = passes.reorder_unrolled_io()(program)
```

## 算法

优先级感知的稳定拓扑排序。把标记 `SeqStmts` 内的每条语句分类：

| 类别 | 优先级 | 示例 |
| ---- | ------ | ---- |
| `Load` | 0（最先发射） | `AssignStmt(tile, Call("tile.load", …))` |
| `Compute` | 1 | 区域内其余语句 |
| `Store` | 2（最后发射） | `AssignStmt(_, Call("tile.store", …))` 或 `EvalStmt(Call("tile.store", …))` |

每一步：

- 在所有前驱已发射的 `ready` 语句中：
  - 若存在非 store 就绪 → 发射 `(category, original_index)` 最小者。Load（cat 0）优先于 Compute（cat 1）。
  - 否则 → 发射索引最小的 store。

Store 被推迟，因为只有当没有 load 或 compute 可发射时才会被选中 —— 配合 load 优先规则，只要数据流允许，结果即为 `[loads…, compute…, stores…]`。

示例 —— 输入 `[load_0, compute_0, store_0, load_1, compute_1, store_1]`，每个克隆的 compute 读其 load、每个 store 读其 compute：

```text
ready={load_0, load_1}        非 store 就绪 → 发射 load_0
ready={load_1, compute_0}     非 store 就绪 → 发射 load_1   (cat 0 < cat 1)
ready={compute_0, compute_1}  非 store 就绪 → 发射 compute_0
ready={compute_1, store_0}    非 store 就绪 → 发射 compute_1
ready={store_0, store_1}      仅 store 就绪 → 发射 store_0
ready={store_1}               仅 store 就绪 → 发射 store_1
```

输出: `[load_0, load_1, compute_0, compute_1, store_0, store_1]`。

## 正确性

重排是对 SSA def-use 依赖图的拓扑排序，因此保留所有数据流。可靠性依赖于 `stmt_dependency_analysis.h` 中的两个工具：

1. `CheckInOutUseDiscipline(region, program)` —— 若某个用户函数调用以 `InOut`/`Out` 方式传入变量，且后续语句读取该变量，则中止编译。该规约（RFC #1026）保证物理内存变更对应 SSA 版本变化，因此 SSA def-use 即捕获所有真实依赖。
2. `BuildStmtDependencyGraph(region, program)` —— 在规约成立时，构造区域顶层语句的可靠 def-use DAG。

## 约束

| 约束 | 原因 |
| ---- | ---- |
| 仅在带 `attrs_["unroll_replicated"]` 的 `ForStmt` 内部生效 | 否则会无意中重排无关 SeqStmts |
| 区域必须满足 InOut-use 规约 | 数据流分析的可靠性前提（RFC #1026） |
| 依赖图存在环时中止 | SSA 区域不应出现环；以 `INTERNAL_CHECK` 抛出 |

## 示例

**变换前**（来自 `PartialUnrollTileLoops` 的输入）:

```python
for i in pl.range(0, 8, 4, attrs={"unroll_replicated": 4}):
    tile_x_0 = pl.tile.load(input_a, [i * 128], [128])
    tile_y_0 = pl.tile.add(tile_x_0, 1.0)
    pl.tile.store(tile_y_0, [i * 128], output)
    tile_x_1 = pl.tile.load(input_a, [(i + 1) * 128], [128])
    tile_y_1 = pl.tile.add(tile_x_1, 1.0)
    pl.tile.store(tile_y_1, [(i + 1) * 128], output)
    # ... k=2、k=3 ...
```

**变换后**:

```python
for i in pl.range(0, 8, 4, attrs={"unroll_replicated": 4}):
    tile_x_0 = pl.tile.load(input_a, [i * 128], [128])
    tile_x_1 = pl.tile.load(input_a, [(i + 1) * 128], [128])
    tile_x_2 = pl.tile.load(input_a, [(i + 2) * 128], [128])
    tile_x_3 = pl.tile.load(input_a, [(i + 3) * 128], [128])
    tile_y_0 = pl.tile.add(tile_x_0, 1.0)
    tile_y_1 = pl.tile.add(tile_x_1, 1.0)
    tile_y_2 = pl.tile.add(tile_x_2, 1.0)
    tile_y_3 = pl.tile.add(tile_x_3, 1.0)
    pl.tile.store(tile_y_0, [i * 128], output)
    pl.tile.store(tile_y_1, [(i + 1) * 128], output)
    pl.tile.store(tile_y_2, [(i + 2) * 128], output)
    pl.tile.store(tile_y_3, [(i + 3) * 128], output)
```

到最后一个 load 为止，四个 `tile_x_k` 同时活跃；到第一个 store 之前，四个 `tile_y_k` 同时活跃。下一个 Pass `MemoryReuse` 无法合并它们 —— 每个都拥有独立的 MemRef。

## 相关

- [`PartialUnrollTileLoops`](20-partial_unroll_tile_loops.md) —— 生成本 Pass 消费的 `unroll_replicated` 标记
- [`MemoryReuse`](16-memory_reuse.md) —— 在本 Pass 之后运行；受益于同时活跃的 tile
- RFC #1025 —— 设计文档
- RFC #1026 / PR #1029 —— InOut-use 规约 + 依赖分析工具
