# CanonicalizeIOOrder Pass

仅限于 **`ForKind::Pipeline` 循环体内部** 的 `SeqStmts`，沿**同核硬件单元阶段阶梯**（标量 → load → compute → store）重排语句 —— 受 SSA 依赖图约束。把各克隆的 load 聚到前面，使两个 stage 的预取先于 compute 发射，于是 MTE load 引擎能跑在 Vector/Cube compute 引擎之前（双缓冲重叠）；compute/store 这一档再按 pipeline **stage** 排序，使每个 stage 算完立刻 store。注意：stage 之间的缓冲**分离**不再是这种聚集的副作用 —— 它现在是 [`MemoryReuse`](30-memory_reuse.md) 的显式约束（`pipeline_membership`）；本 pass 只塑造**调度顺序**。跨核（cube/vector）流水线由 [`SkewCrossCorePipeline`](24-skew_cross_core_pipeline.md) 在上游软流水，到达本 pass 时已是 `ForKind::Sequential`，故此处不含跨核处理。非流水线循环则保持不变。

## 概述

`LowerPipelineLoops` 生成的外层 `ForStmt`（kind=Pipeline 标记）体是 `F` 份克隆体的 `SeqStmts`，自然顺序为 `[scalar_0, load_0, compute_0, store_0, scalar_1, load_1, compute_1, store_1, …]`（每个克隆的地址运算先于其 load）。这种布局下，每个克隆的 load 排在上一个克隆的 store 之后，MTE load 引擎无法跑在 compute 引擎之前 —— 没有预取重叠。

本 Pass 仅对 **`ForKind::Pipeline` 循环体内部** 的 `SeqStmts`（包括该流水线作用域内嵌套的 `IfStmt` 分支 body 等）做重排：

- 每个标量计算（典型为地址运算）上拉到依赖图允许的最早位置，从而解锁后续 load。
- 每个 `tile.load` / `tile.read` 上拉到依赖图允许的最早位置。
- tile 计算语句留在中间。
- 每个 `tile.store` / `tile.write` 下沉到依赖图允许的最晚位置。

只要数据流允许，结果即为 `[scalars…, loads…, 每个 stage 的 (compute, store)…]` —— 例如 `stage=2` 的循环体发射为 `load load compute_s0 store_s0 compute_s1 store_s1`。聚集 load 带来预取重叠（MTE 引擎跑在前面）；compute/store 按 stage 排序意味着每个 stage 算完立刻 store，在下一个 stage 之前释放该缓冲，既降低片上压力，也减轻跨迭代的 load↔store 耦合。stage 之间的缓冲分离由 `MemoryReuse` 单独强制（见 [30-memory_reuse.md](30-memory_reuse.md)）。

上拉标量计算正是 load 聚集的关键：若不区分类别，每个克隆的地址运算 assign 会被归为普通 compute、按原始位置排序，从而在兄弟 load 之间穿插，把 load 钉在原始克隆里。把标量计算作为最高优先级类别后，所有兄弟克隆的地址运算先发射，所有依赖的 load 同时就绪，load 自然聚集。

### 跨核（AIC↔AIV）—— 由上游处理

跨核（cube/vector）pipeline 循环由 [`SkewCrossCorePipeline`](24-skew_cross_core_pipeline.md) 软流水：它在 `LowerPipelineLoops` *之前*运行，把每个跨核循环改写为 `ForKind::Sequential`。因此它们永远不会以 `ForKind::Pipeline` body 进入本 pass，`CanonicalizeIOOrder` 也**不含任何跨核处理** —— 这里 `tpush`/`tpop` 只是普通 tile 计算，不会被重排进任何跨核阶段。本 pass 只对剩余的同核 pipeline 循环（GM→L1、L1→L0、嵌套 matmul）聚集**同核**阶段（标量 → load → compute → store）以实现 ping-pong。

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

| 类别 | 优先级 | 硬件单元 | 示例 |
| ---- | ------ | -------- | ---- |
| `ScalarCompute` | 0（最先发射） | 标量 | LHS 为 `ScalarType` 的 `AssignStmt`（如 `off = i * 64`） |
| `Load` | 1 | MTE 入口（GM→L1/L0） | `tile.load` / `tile.read` / L1→L0 的 `tile.extract` |
| `TileCompute` | 2 | CUBE/Vec 计算 | 其余一切（matmul 循环、elementwise、`tile.move`、`tpush`/`tpop` —— 见下注） |
| `Store` | 3（最后发射） | MTE 出口（L1/L0→GM） | `tile.store` / `tile.write`（AssignStmt 或 EvalStmt） |

`tile.read` 虽然产出标量，但仍归为 `Load` —— 它是针对 tile 的 I/O，与 `tile.load` 同属 load 层。LHS 类型检查仅在 RHS 不是已识别的 I/O op 时生效。

跨核 `tpush`/`tpop` 不带任何特殊类别 —— 它们落入 `TileCompute`，在兄弟语句间保持程序顺序（跨核软流水由上游的 [`SkewCrossCorePipeline`](24-skew_cross_core_pipeline.md) 完成；见上文「跨核（AIC↔AIV）」）。

每一步在 `ready`（所有前驱已发射）的语句中，发射 `(tier, stage, sub, original_index)` 最小者 —— `tier` 为标量 compute=0、load=1、tile-compute/store=2；`stage` 取语句的 `pipeline_membership`（这样 tile 定义与消费它的 store 共享同一 stage）；`sub` 为 compute=0、store=1。于是 load（tier 1）聚集在所有 compute/store 之前，而 compute/store 这一档里每个 stage 的 compute 先于其 store、再到下一个 stage。非流水线区域无 membership（`stage` 为空），tier/sub 排序退化为原先的 标量 → load → compute → store 阶梯。

示例 —— 输入 `[scalar_0, load_0, compute_0, store_0, scalar_1, load_1, compute_1, store_1]`，每个克隆的 load 读其 scalar、每个 compute 读其 load、每个 store 读其 scalar 与 compute：

```text
ready={scalar_0, scalar_1}              发射 scalar_0    (cat 0, idx 0)
ready={load_0, scalar_1}                发射 scalar_1    (cat 0 < cat 1)
ready={load_0, load_1}                  发射 load_0      (cat 1, idx 1 < 5)
ready={load_1, compute_0}               发射 load_1      (cat 1 < cat 2)
ready={compute_0, compute_1}            发射 compute_0
ready={compute_1, store_0}              发射 compute_1   (cat 2 < cat 6)
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
    pl.tile.store(tile_y_0, [off_0], output)
    tile_y_1 = pl.tile.add(tile_x_1, 1.0)
    pl.tile.store(tile_y_1, [off_1], output)
    tile_y_2 = pl.tile.add(tile_x_2, 1.0)
    pl.tile.store(tile_y_2, [off_2], output)
    tile_y_3 = pl.tile.add(tile_x_3, 1.0)
    pl.tile.store(tile_y_3, [off_3], output)
```

四个 `off_k` 先上拉以解锁 load，load 随之聚集（预取重叠 —— MTE 引擎跑在 compute 之前）。compute/store 这一档按 stage 排序，每个 `tile_y_k` 算完立刻 store，在下一个 stage 之前释放该输出缓冲。缓冲的 stage **分离**（每个克隆保留独立 MemRef）由 `MemoryReuse` 经 `pipeline_membership` 强制，而非由本排序保证。

## 相关

- [`LowerPipelineLoops`](25-lower_pipeline_loops.md) —— 上游复制区域生成者；保留 `ForKind::Pipeline` 标记供本 Pass 识别
- [`MaterializeTensorStrides`](27-materialize_tensor_strides.md) —— 接入默认流水线后紧随本 Pass 运行；在 `InitMemRef` 消费前补全隐式 `TensorView` stride
- [`MemoryReuse`](30-memory_reuse.md) —— 在本 Pass 之后运行；经 `pipeline_membership` 显式强制 stage 缓冲分离（本 pass 只塑造调度顺序）
- RFC #1026 / PR #1029 —— InOut-use 规约 + 依赖分析工具
