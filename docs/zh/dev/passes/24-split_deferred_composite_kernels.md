# SplitDeferredCompositeKernels Pass

将带 `defer=True` 的 mesh 组合集体通信内核拆成 push / wait / epilogue 三个任务，
并改写 Orchestration 侧的 Submit，使对外可见的 TaskId 指向 epilogue。

## 概述

`OutlineIncoreScopes` 会在含 `pld.tensor.*(defer=True)` 的任务级
`pl.at(CORE_GROUP)` 上盖章 `deferred_completion_waiter`。
`LowerCompositeOps` 随后把该函数体展开为 push+notify、`pld.system.defer_wait`
与自清理 epilogue（仍在同一函数内）。

`ExpandMixedKernel` 要求 deferred-waiter 函数体只能做条件注册。
若 `defer_wait` 旁仍留有 push/epilogue，会违反该契约。本 Pass 紧接
[`LowerAutoVectorSplit`](23-lower_auto_vector_split.md) 之后、
[`ExpandMixedKernel`](25-expand_mixed_kernel.md) 之前运行，并：

1. 找到仍含 deferred 组合 lowering（push + `defer_wait` + epilogue）且盖了
   `deferred_completion_waiter` 的 InCore 函数；
2. 克隆为 `_push` / `_wait` / `_epi` 三个兄弟函数；
3. 将原函数名的 Orchestration `Submit`/`Call` 改写成三次 **Submit**：
   push → wait（`deps=[push]`）→ epi（`deps=[wait]`），并把原 TaskId 绑定到
   **epilogue**，使消费者的 `deps=[tid]` 等到完整完成（含 signal 清零）。
   纯 `Call` 站点会提升为 `Submit` 以便串联 TaskId 依赖（无 `as tid` 时
   `OutlineIncoreScopes` 已为 deferred waiter 合成 TaskId）。

## 流水线位置

Default 策略：`lower_auto_vector_split` 之后、`expand_mixed_kernel` 之前
（`python/pypto/ir/pass_manager.py`）。

## 相关

- [`LowerCompositeOps`](13-lower_composite_ops.md)
- [`OutlineIncoreScopes`](09-outline_incore_scopes.md)
- [`ExpandMixedKernel`](25-expand_mixed_kernel.md)
- 校验属性 `DeferredCompositePlacementValid`（`99-verifier.md`）
