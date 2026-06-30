# SkewCrossCorePipeline Pass

对 cube/vector 混合（跨核）的 `pl.pipeline` 循环做软流水（skew），使两个核相互重叠，替代旧的「unroll + IO 聚类」跨核处理方式。紧接在 [`LowerPipelineLoops`](25-lower_pipeline_loops.md) 之前运行。

## 概述

在 A2/A3 上，cube+vector 融合 kernel（例如 flash-decode `qk_pv`）通过 GM 往返：cube（AIC）用 `tile.tpush_to_aiv` 把 scores 发给 vector（AIV），再用 `tile.tpop_from_aiv` 取回 softmax 结果；vector 侧对称地使用 `tile.tpop_from_aic` / `tile.tpush_to_aic`。若直接顺序执行，两个核会互相等待而 stall。

旧方案对这些循环做 unroll（`pl.pipeline(stage=F)`）并由 `CanonicalizeIOOrder` 聚类跨核算子 —— 这会产生**背靠背的 `tpop`**，把消费者串行化。`SkewCrossCorePipeline` 改为对循环做软流水：

- **单次往返、生产者角色** —— 恰好一个 `tpush` 和一个 `tpop`，且 `tpush` 的反向切片不通过 SSA 边喂给 body（cube：`QK → tpush`，`tpop → SV`）。两半仅通过有序的跨核 FIFO 关联，于是让生产者**提前 `D = max(2, stage-1)` 个迭代**运行（跨核默认 depth-2）：`produce(start … start+(D-1)·step)` 序言（prologue）、一个 `ForKind::Sequential` 稳态循环（其循环变量 `k` 作为每组的首个 produce 索引，把该组的 `D` 个 produce `produce(k+i·step)` 与滞后的 `D` 个 consume `consume(k-(D-i)·step)` 配对，`k` 以 `D·step` 为步长取值 `[start+D·step, start+trip·step)`）、以及 `consume(最后 D 个)` 尾声（epilogue）。cube 发出第 k 组的 `D` 个 `QK` 时，vector 正在跑第 k-D 组的 `D` 个 softmax。详见 [skew 深度](#skew-深度stage)。
- **消费者角色，或多次往返** —— lead 算子通过 SSA 喂给 body（vector：弹出的 scores 喂给 softmax），或某个 FIFO 方向上有多于一条消息。此时**降级为普通的 `ForKind::Sequential` 循环**（body 不变）。这消除了 unroll 的背靠背 `tpop`，同时保持 FIFO 的有序性；跨核重叠则来自**对端**核的生产者 skew —— 它提前一步把每个 tile 放入 FIFO，使本核有序的 `tpop` 不再 block。

每个**非跨核**的 pipeline 循环（同核 GM→L1、L1→L0、嵌套 matmul stage 循环 —— 没有 `tpush`/`tpop`）保持不变，交给 `LowerPipelineLoops` 复制。

输出为 `ForKind::Sequential` 且不带 `pipeline_stages` 标记，因此 `LowerPipelineLoops`（触发条件 `kind == Pipeline`）会跳过它，`CanonicalizeIOOrder`（作用域限定在 pipeline body）也不会再去重排已手工排好序的 skew。

**Requires**: SSAForm, SplitIncoreOrch, IncoreTileOps, TileOps2D, TileMemoryInferred, NormalizedStmtStructure.

**流水线位置**: 在 [`NormalizeReturnOrder`](23-normalize_return_order.md) 之后，紧接在 [`LowerPipelineLoops`](25-lower_pipeline_loops.md) 之前。两者之间没有其他 pass，因此跨核循环会先被 skew（→ Sequential），unroll pass 没有机会复制它。

## API

```python
from pypto import passes
p = passes.skew_cross_core_pipeline()
```

## 行为

对 `ForStmt(kind=Pipeline, attrs={"pipeline_stages": F})` 且 `F > 1`：

1. **非跨核** —— body 不含 `tpush`/`tpop` 对 → 保持为 `ForKind::Pipeline`，交给 `LowerPipelineLoops` 复制。
2. **跨核** —— body 同时含 `tpush` 和 `tpop`。找 lead（程序顺序第一个跨核算子），反向切片成生产者半，并分类：
   - 一个 `tpush` + 一个 `tpop`、`carried`（被 body 使用的 lead 定义变量）为空、且静态可 skew → **生产者 skew**（prologue / Sequential 稳态循环 / epilogue）。
   - 否则 —— `carried` 非空（消费者角色）、多于一个 `tpush`/`tpop`（多次往返；只 skew 一条消息会打乱有序 FIFO —— verifier 抓不到的静默数据错误）、动态边界、或 trip < 2 → **`DemoteToSequential`**。

无论哪种，跨核循环**总是**以 `ForKind::Sequential`（无 `pipeline_stages` 标记）离开本 pass，因此永远不会以 Pipeline body 的形式到达 `LowerPipelineLoops` 或 `CanonicalizeIOOrder`。

### skew 深度（stage）

跨核生产者 skew **至少需要 depth 2**：两个流水 stage（例如 cube 的两个 QK matmul）必须落在不同的 L1/L0 buffer 上，否则 `MemoryReuse` 会把它们合并到一块、把 cube 串行化。因此**请求**的深度为

```text
D = max(2, stage - 1)
```

—— 默认 depth-2，只有当 `stage - 1` 超过 2（即 `stage >= 4`）时才取标准的 `stage - 1`。每个稳态迭代发 **`D` 个 produce、再 `D` 个 consume**（稳态循环按 `D` 展开）：

| `stage` | 深度 `D` | 稳态体 |
| ------- | -------- | ------ |
| 2、3 | 2 | `produce(k); produce(k+step); consume(k-2·step); consume(k-step)` |
| 4 | 3 | `produce(k … k+2·step); consume(k-3·step … k-step)` |

**实际**深度需满足 `trip % D == 0` 且 `trip >= 2·D`；请求的 `D` 不满足时取**最大可行的 `D' <= D`**——对不整除的 trip（如奇数）一路回退到 `1`（即经典的提前一个迭代的 skew）。每个迭代的 `D` 个 produce tile 和 `D` 个 consume tile 各自独立，使两个流水 stage 不落在同一块 buffer 上——depth-1 时一个 cube QK/SV 对共用一块 Mat buffer，正是它把两个 matmul 串行化的。

`iter_args`（例如 flash-attention 的 `mi/li/oi` 累加器）会穿过生产者 skew 的 prologue → 稳态 → epilogue；生产者半对 iter_arg 透明。

## 约束

- skew 仅支持静态边界（`start`、`stop`、`step` 为编译期常量）。动态边界的跨核循环回退到 unroll 路径。
- 生产者 skew 的稳态区**保留为循环**（不完全展开），使 `AllocateMemoryAddr` 分配的 matmul `Acc` 双缓冲仍有循环可交替。
- 有意**不**做消费者侧预取：它会破坏 codegen 的 `tpop → tfree` FIFO 槽位追踪（以 SSA 变量身份为键，无法跨 iter_arg），且提前整整一个迭代发出阻塞式 `tpop` 只会 stall。

## 局限

- **多次往返循环目前不做 skew（TODO）。** 某个 FIFO 方向上有多于一条消息的循环
  （例如 `C→V→C→V` = `tpush, tpop, tpush, tpop`）当前**降级为 `ForKind::Sequential`**，
  而非软流水。只 skew lead 那一条消息会打乱有序的跨核 FIFO，静默地把错误的 tile
  喂给对端核（property verifier 不建模 FIFO 顺序），因此保守地降级是正确的，但损失了
  重叠收益。未来应将整个 FIFO 组一起 skew（让每条消息都提前一个往返），使多次往返的
  生产者也能重叠。

## 相关

- [`LowerPipelineLoops`](25-lower_pipeline_loops.md) —— 复制其余（同核）pipeline 循环以实现 ping-pong。
- [`CanonicalizeIOOrder`](26-canonicalize_io_order.md) —— 在 pipeline body 内聚类同核 IO（跨核循环到这里已是 Sequential，不再进入此 pass）。
- [`SplitVectorKernel`](21-split_vector_kernel.md) —— `UP_DOWN` vector 切分，与 skew 正交且可组合。
