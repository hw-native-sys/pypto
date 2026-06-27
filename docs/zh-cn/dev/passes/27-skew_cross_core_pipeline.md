# SkewCrossCorePipeline Pass

对 cube/vector 混合（跨核）的 `pl.pipeline` 循环做软流水（skew），使两个核相互重叠，替代旧的「unroll + IO 聚类」跨核处理方式。紧接在 [`LowerPipelineLoops`](28-lower_pipeline_loops.md) 之前运行。

## 概述

在 A2/A3 上，cube+vector 融合 kernel（例如 flash-decode `qk_pv`）通过 GM 往返：cube（AIC）用 `tile.tpush_to_aiv` 把 scores 发给 vector（AIV），再用 `tile.tpop_from_aiv` 取回 softmax 结果；vector 侧对称地使用 `tile.tpop_from_aic` / `tile.tpush_to_aic`。若直接顺序执行，两个核会互相等待而 stall。

旧方案对这些循环做 unroll（`pl.pipeline(stage=F)`）并由 `CanonicalizeIOOrder` 聚类跨核算子 —— 这会产生**背靠背的 `tpop`**，把消费者串行化。`SkewCrossCorePipeline` 改为对循环做软流水：

- **单次往返、生产者角色** —— 恰好一个 `tpush` 和一个 `tpop`，且 `tpush` 的反向切片不通过 SSA 边喂给 body（cube：`QK → tpush`，`tpop → SV`）。两半仅通过有序的跨核 FIFO 关联，于是让生产者**提前一个迭代**运行：`produce(start)` 序言（prologue）、一个 `ForKind::Sequential` 稳态循环（其循环变量 `k` 索引 produce，把 `produce(k)` 与滞后一步的 `consume(k-step)` 配对，`k` 取值 `[start+step, start+trip*step)`）、以及 `consume(last)` 尾声（epilogue）。cube 发出第 k 次迭代的 `QK` 时，vector 正在跑第 k-step 次迭代的 softmax。
- **消费者角色，或多次往返** —— lead 算子通过 SSA 喂给 body（vector：弹出的 scores 喂给 softmax），或某个 FIFO 方向上有多于一条消息。此时**降级为普通的 `ForKind::Sequential` 循环**（body 不变）。这消除了 unroll 的背靠背 `tpop`，同时保持 FIFO 的有序性；跨核重叠则来自**对端**核的生产者 skew —— 它提前一步把每个 tile 放入 FIFO，使本核有序的 `tpop` 不再 block。

每个**非跨核**的 pipeline 循环（同核 GM→L1、L1→L0、嵌套 matmul stage 循环 —— 没有 `tpush`/`tpop`）保持不变，交给 `LowerPipelineLoops` 复制。

输出为 `ForKind::Sequential` 且不带 `pipeline_stages` 标记，因此 `LowerPipelineLoops`（触发条件 `kind == Pipeline`）会跳过它，`CanonicalizeIOOrder`（作用域限定在 pipeline body）也不会再去重排已手工排好序的 skew。

**Requires**: SSAForm, SplitIncoreOrch, IncoreTileOps, TileOps2D, TileMemoryInferred, NormalizedStmtStructure.

**流水线位置**: 在 [`NormalizeReturnOrder`](26-normalize_return_order.md) 之后，紧接在 [`LowerPipelineLoops`](28-lower_pipeline_loops.md) 之前。两者之间没有其他 pass，因此跨核循环会先被 skew（→ Sequential），unroll pass 没有机会复制它。

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

- [`LowerPipelineLoops`](28-lower_pipeline_loops.md) —— 复制其余（同核）pipeline 循环以实现 ping-pong。
- [`CanonicalizeIOOrder`](29-canonicalize_io_order.md) —— 在 pipeline body 内聚类同核 IO（跨核循环到这里已是 Sequential，不再进入此 pass）。
- [`SplitVectorKernel`](24-split_vector_kernel.md) —— `UP_DOWN` vector 切分，与 skew 正交且可组合。
