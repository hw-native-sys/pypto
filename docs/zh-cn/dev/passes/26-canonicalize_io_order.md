# CanonicalizeIOOrder Pass

仅限于 **`ForKind::Pipeline` 循环体内部** 的 `SeqStmts`，沿**硬件单元阶段阶梯**重排语句 —— 受 SSA 依赖图约束 —— 使每个阶段在 `LowerPipelineLoops` 产生的各克隆间聚集。聚集让相邻迭代的 tile 同时活跃，正是 ping-pong（双缓冲）得以成立的前提。该阶梯既覆盖核内 MTE/计算阶段（标量 → load → compute → store），也覆盖跨核 AIC↔AIV 往返（跨核 push → pop → 消费侧计算），从而让融合的 cube/vector 流水线跨 `tpush`/`tpop` 边界软流水（issue #1610）。非流水线循环则保持不变。

## 概述

`LowerPipelineLoops` 生成的外层 `ForStmt`（kind=Pipeline 标记）体是 `F` 份克隆体的 `SeqStmts`，自然顺序为 `[scalar_0, load_0, compute_0, store_0, scalar_1, load_1, compute_1, store_1, …]`（每个克隆的地址运算先于其 load）。这种布局下，相邻克隆的 tile 生命周期不重叠，`MemoryReuse` 会把它们合并为同一缓冲区，ping-pong 失效。

本 Pass 仅对 **`ForKind::Pipeline` 循环体内部** 的 `SeqStmts`（包括该流水线作用域内嵌套的 `IfStmt` 分支 body 等）做重排：

- 每个标量计算（典型为地址运算）上拉到依赖图允许的最早位置，从而解锁后续 load。
- 每个 `tile.load` / `tile.read` 上拉到依赖图允许的最早位置。
- tile 计算语句留在中间。
- 每个 `tile.store` / `tile.write` 下沉到依赖图允许的最晚位置。

只要数据流允许，结果即为 `[scalars…, loads…, tile compute…, stores…]`。对于复制区域，各克隆的输入 tile 在顶部同时活跃，输出 tile 在底部同时活跃 —— `MemoryReuse` 无法合并它们，每个克隆保留独立的 MemRef，从而 ping-pong 缓冲成为可能。

上拉标量计算正是 load 聚集的关键：若不区分类别，每个克隆的地址运算 assign 会被归为普通 compute、按原始位置排序，从而在兄弟 load 之间穿插，把 load 钉在原始克隆里。把标量计算作为最高优先级类别后，所有兄弟克隆的地址运算先发射，所有依赖的 load 同时就绪，load 自然聚集。

### 跨核流水（AIC↔AIV）

同样的聚集可推广到融合的 cube/vector kernel：其 `pl.pipeline` 循环已被 `ExpandMixedKernel` 拆为带跨核 `tpush`/`tpop` 搬移的逐引擎 AIC/AIV body。flash-attention 的一个 AIC 克隆是 `QK matmul → tpush_to_aiv → tpop_from_aiv → SV matmul`。为跨核 push/pop 设独立阶段，并区分生产者（`QK`）与消费者（`SV`）计算后，`F` 份克隆体从逐克隆串行

```text
QK0, tpush0, tpop0, SV0,   QK1, tpush1, tpop1, SV1
```

重排为按阶段聚集

```text
QK0, QK1,   tpush0, tpush1,   tpop0, tpop1,   SV0, SV1
```

于是 `raw_scores0`/`raw_scores1`（QK 与 tpush 之间）同时活跃 ⇒ 两块 score 缓冲；两个 pop 结果（tpop 与 SV 之间）同时活跃 ⇒ 两块结果缓冲 —— 两侧都能 ping-pong。

**为何关键在于排序而非指令重叠。** 昇腾是事件/依赖驱动执行，而非按指令顺序：发射 `QK0 QK1 tpush0 tpush1` 会把四个任务一起下发，`tpush0` 在 `QK0` 完成后即可执行 —— 聚集*不会*延迟消费侧。重排的目的在于**绕过 `MemoryReuse`**。把同一阶段的生产者与消费者交错（`QK0, tpush0, QK1, tpush1`）会让 `raw_scores0` 在 `tpush0` 处死亡、早于 `raw_scores1` 在 `QK1` 处诞生；生命周期不重叠使 `MemoryReuse` 把二者合并为一块缓冲，进而注入一条*伪* WAR 依赖（`QK1` 必须等 `tpush0`），把流水线串行化。聚集保持生命周期重叠，迫使分配独立 MemRef。

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
| `TileCompute` | 2 | CUBE/Vec（生产者） | 跨核往返*之前*的计算（如 QK matmul 循环、tile.move） |
| `CrossCorePush` | 3 | 跨核出口 | `EvalStmt(Call("tile.tpush_to_aiv" / "tile.tpush_to_aic", …))` |
| `CrossCorePop` | 4 | 跨核入口 | `AssignStmt(_, Call("tile.tpop_from_aiv" / "tile.tpop_from_aic", …))` |
| `ConsumerCompute` | 5 | CUBE/Vec（消费者） | 跨核 pop *下游*的计算（如 SV matmul 循环、`tfree`、`set_validshape`） |
| `Store` | 6（最后发射） | MTE 出口（L1/L0→GM） | `tile.store` / `tile.write`（AssignStmt 或 EvalStmt） |

`tile.read` 虽然产出标量，但仍归为 `Load` —— 它是针对 tile 的 I/O，与 `tile.load` 同属 load 层。LHS 类型检查仅在 RHS 不是已识别的 I/O op 时生效。

**生产者计算 vs 消费者计算。** `tpop` 没有 SSA 参数（它按 `id`/`split` 从 GM 环形缓冲弹出），故依赖图本身无法判断消费其结果的计算属于往返之后的阶段。本 Pass 沿 SSA 边正向传播“位于跨核 pop 下游”这一标记，并把此类 `TileCompute` 降为 `ConsumerCompute`。若某 `tile.create` 的使用者*全部*属于消费侧（如夹在 `tpush` 与其 `tpop` 之间的 SV 累加器初始化），同样归为 `ConsumerCompute`，使其紧挨消费者、而非被上拉进生产者簇（否则会拉长其 Acc 缓冲生命周期、加剧 L0C 压力）。

**为何 `tpush` 不像 `Store` 那样下沉。** 二者都是出口，但 `tpush` 必须在其生产者允许的最早时刻发射、好让对端核尽快开工；它排在生产者 `TileCompute` *之后*（使兄弟生产者先聚集）、却在 pop *之前* —— 不像 GM store 那样被推到最底部。

每一步在 `ready`（所有前驱已发射）的语句中，发射 `(category, original_index)` 最小者。Store 因 `Store` 是最大类别而自然排在最后 —— 只有当没有其他可发射时才会被选中。（下方示例不含跨核 op，故只有 scalar/load/compute/store 阶段参与；跨核 tpush/tpop 示例见上文「跨核流水」。）

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

- [`LowerPipelineLoops`](25-lower_pipeline_loops.md) —— 上游复制区域生成者；保留 `ForKind::Pipeline` 标记供本 Pass 识别
- [`MaterializeTensorStrides`](27-materialize_tensor_strides.md) —— 接入默认流水线后紧随本 Pass 运行；在 `InitMemRef` 消费前补全隐式 `TensorView` stride
- [`MemoryReuse`](29-memory_reuse.md) —— 在本 Pass 之后运行；受益于复制区域中同时活跃的 tile
- RFC #1026 / PR #1029 —— InOut-use 规约 + 依赖分析工具
