# CanonicalizeIOOrder Pass

仅限于 **`ForKind::Pipeline` 循环体内部** 的 `SeqStmts`，把标量计算与 `tile.load` / `tile.read` 上拉到顶部、`tile.store` / `tile.write` 下沉到底部 —— 受 SSA 依赖图约束 —— 从而规范化 IO 顺序。对于 `LowerPipelineLoops` 产生的复制区域，该规范化直接启用对称的 ping-pong 缓冲。非流水线循环则保持不变。

## 概述

`LowerPipelineLoops` 生成的外层 `ForStmt`（kind=Pipeline 标记）体是 `F` 份克隆体的 `SeqStmts`，自然顺序为 `[scalar_0, load_0, compute_0, store_0, scalar_1, load_1, compute_1, store_1, …]`（每个克隆的地址运算先于其 load）。这种布局下，相邻克隆的 tile 生命周期不重叠，`MemoryReuse` 会把它们合并为同一缓冲区，ping-pong 失效。

本 Pass 仅对 **`ForKind::Pipeline` 循环体内部** 的 `SeqStmts`（包括该流水线作用域内嵌套的 `IfStmt` 分支 body 等）做重排：

- 每个标量计算（典型为地址运算）上拉到依赖图允许的最早位置，从而解锁后续 load。
- 每个 `tile.load` / `tile.read` 上拉到依赖图允许的最早位置。
- tile 计算语句留在中间。
- 每个 `tile.store` / `tile.write` 下沉到依赖图允许的最晚位置。

只要数据流允许，结果即为 `[scalars…, loads…, tile compute…, stores…]`。对于复制区域，各克隆的输入 tile 在顶部同时活跃，输出 tile 在底部同时活跃 —— `MemoryReuse` 无法合并它们，每个克隆保留独立的 MemRef，从而 ping-pong 缓冲成为可能。

上拉标量计算正是 load 聚集的关键：若不区分类别，每个克隆的地址运算 assign 会被归为普通 compute、按原始位置排序，从而在兄弟 load 之间穿插，把 load 钉在原始克隆里。把标量计算作为最高优先级类别后，所有兄弟克隆的地址运算先发射，所有依赖的 load 同时就绪，load 自然聚集。

**前置条件**: SSAForm、SplitIncoreOrch、IncoreTileOps、TileOps2D、TileMemoryInferred、NormalizedStmtStructure。

**流水线位置**: 位于 `LowerPipelineLoops` 之后、`InitMemRef` 之前（slot 20.6）。在 `InitMemRef` 之前运行可保留 SSAForm，依赖分析正常工作。该 Pass 在退出时会把外层流水线循环的 `kind_` 从 `ForKind::Pipeline` 降级为 `ForKind::Sequential`，并清除残留的 `pipeline_stages` attr —— `ForKind::Pipeline` 是一个过渡标记，不得穿过本 Pass。

## API

| C++ | Python | 级别 |
| --- | ------ | ---- |
| `pass::CanonicalizeIOOrder()` | `passes.canonicalize_io_order()` | 程序级 |

```python
from pypto import passes
result = passes.canonicalize_io_order()(program)
```

## 算法

对所有两条及以上语句的 `SeqStmts` 做优先级感知的稳定拓扑排序。每条语句分类：

| 类别 | 优先级 | 示例 |
| ---- | ------ | ---- |
| `ScalarCompute` | 0（最先发射） | LHS 为 `ScalarType` 的 `AssignStmt`（如 `off = i * 64`） |
| `Load` | 1 | `AssignStmt(_, Call("tile.load", …))` 或 `AssignStmt(_, Call("tile.read", …))` |
| `TileCompute` | 2 | 区域内其余语句（tile.move、tile.matmul、yield_ 等） |
| `Store` | 3（最后发射） | `AssignStmt(_, Call("tile.store", …))` / `EvalStmt(Call("tile.store", …))` / `tile.write` 变体 |

`tile.read` 虽然产出标量，但仍归为 `Load` —— 它是针对 tile 的 I/O，与 `tile.load` 同属 load 层。LHS 类型检查仅在 RHS 不是已识别的 I/O op 时生效。

每一步在 `ready`（所有前驱已发射）的语句中，发射 `(category, original_index)` 最小者。Store 因 `Store = 3` 是最大类别而自然排在最后 —— 只有当没有其他可发射时才会被选中。

示例 —— 输入 `[scalar_0, load_0, compute_0, store_0, scalar_1, load_1, compute_1, store_1]`，每个克隆的 load 读其 scalar、每个 compute 读其 load、每个 store 读其 scalar 与 compute：

```text
ready={scalar_0, scalar_1}              发射 scalar_0    (cat 0, idx 0)
ready={load_0, scalar_1}                发射 scalar_1    (cat 0 < cat 1)
ready={load_0, load_1}                  发射 load_0      (cat 1, idx 1 < 5)
ready={load_1, compute_0}               发射 load_1      (cat 1 < cat 2)
ready={compute_0, compute_1}            发射 compute_0
ready={compute_1, store_0}              发射 compute_1   (cat 2 < cat 3)
ready={store_0, store_1}                发射 store_0
ready={store_1}                         发射 store_1
```

输出: `[scalar_0, scalar_1, load_0, load_1, compute_0, compute_1, store_0, store_1]`。

## 正确性

重排是对 SSA def-use 依赖图的拓扑排序，因此保留所有数据流。可靠性依赖于 `stmt_dependency_analysis.h` 中的两个工具：

1. `CollectInOutUseDisciplineDiagnostics(region, program)` —— 报告任何以 `InOut`/`Out` 传入变量而后续语句仍读取该变量的用户函数调用。自 PR #1039 起该规约已是结构化 IR 不变式（RFC #1026）：所有合法 IR 的每个函数都满足它。变量作用域不跨函数边界，故本 Pass 在函数级别运行该检查一次（而非在每个 `SeqStmts` 上）；若某函数报告违规，则整个函数跳过重排（即使在 `VerificationLevel.NONE` 下也保证可靠）。
2. `BuildStmtDependencyGraph(region, program)` —— 在规约成立时，构造区域顶层语句的可靠 def-use DAG。由于已在函数级别完成规约检查，调用时对 `program` 传入 `nullptr`。

## 约束

| 约束 | 原因 |
| ---- | ---- |
| 函数必须满足 InOut-use 规约 | 数据流分析的可靠性前提（自 PR #1039 起为结构化不变式）；函数级检查未通过时跳过重排 |
| 依赖图存在环时中止 | SSA 区域不应出现环；以 `INTERNAL_CHECK` 抛出 |

## 示例

**变换前**（来自 `LowerPipelineLoops` 的输入 —— 注意每个克隆都有标量地址运算 assign，外层循环 kind=Pipeline 标记位、属性下调为 stage=1）:

```python
for i in pl.pipeline(0, 8, 4, stage=1):  # kind=Pipeline 标记；属性=1
    off_0: pl.Scalar[pl.INDEX] = i * 128
    tile_x_0 = pl.tile.load(input_a, [off_0], [128])
    tile_y_0 = pl.tile.add(tile_x_0, 1.0)
    pl.tile.store(tile_y_0, [off_0], output)
    off_1: pl.Scalar[pl.INDEX] = (i + 1) * 128
    tile_x_1 = pl.tile.load(input_a, [off_1], [128])
    tile_y_1 = pl.tile.add(tile_x_1, 1.0)
    pl.tile.store(tile_y_1, [off_1], output)
    # ... k=2、k=3 ...
```

**变换后**:

```python
for i in pl.range(0, 8, 4):
    off_0: pl.Scalar[pl.INDEX] = i * 128
    off_1: pl.Scalar[pl.INDEX] = (i + 1) * 128
    off_2: pl.Scalar[pl.INDEX] = (i + 2) * 128
    off_3: pl.Scalar[pl.INDEX] = (i + 3) * 128
    tile_x_0 = pl.tile.load(input_a, [off_0], [128])
    tile_x_1 = pl.tile.load(input_a, [off_1], [128])
    tile_x_2 = pl.tile.load(input_a, [off_2], [128])
    tile_x_3 = pl.tile.load(input_a, [off_3], [128])
    tile_y_0 = pl.tile.add(tile_x_0, 1.0)
    tile_y_1 = pl.tile.add(tile_x_1, 1.0)
    tile_y_2 = pl.tile.add(tile_x_2, 1.0)
    tile_y_3 = pl.tile.add(tile_x_3, 1.0)
    pl.tile.store(tile_y_0, [off_0], output)
    pl.tile.store(tile_y_1, [off_1], output)
    pl.tile.store(tile_y_2, [off_2], output)
    pl.tile.store(tile_y_3, [off_3], output)
```

四个 `off_k` 先上拉以解锁 load。到最后一个 load 为止，四个 `tile_x_k` 同时活跃；到第一个 store 之前，四个 `tile_y_k` 同时活跃。下一个 Pass `MemoryReuse` 无法合并它们 —— 每个都拥有独立的 MemRef。

## 相关

- [`LowerPipelineLoops`](23-lower_pipeline_loops.md) —— 上游复制区域生成者；保留 `ForKind::Pipeline` 标记供本 Pass 识别
- [`MemoryReuse`](26-memory_reuse.md) —— 在本 Pass 之后运行；受益于复制区域中同时活跃的 tile
- RFC #1026 / PR #1029 —— InOut-use 规约 + 依赖分析工具
