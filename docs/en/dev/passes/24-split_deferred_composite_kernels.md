# SplitDeferredCompositeKernels Pass

Splits outlined `defer=True` mesh composite kernels into push / wait / epilogue
tasks and rewrites Orchestration submits so the public TaskId is the epilogue.

## Overview

`OutlineIncoreScopes` stamps `deferred_completion_waiter` on a task-level
`pl.at(CORE_GROUP)` whose body contains `pld.tensor.*(defer=True)`.
`LowerCompositeOps` then expands that body into push+notify, `pld.system.defer_wait`,
and the self-clearing epilogue **in one function**.

`ExpandMixedKernel` requires a deferred-waiter body to be registration-only.
Leaving push/epilogue beside `defer_wait` violates that contract. This pass runs
**immediately after** [`LowerAutoVectorSplit`](23-lower_auto_vector_split.md) and
**before** [`ExpandMixedKernel`](25-expand_mixed_kernel.md) and:

1. Finds InCore functions stamped `deferred_completion_waiter` that still contain
   a deferred composite lowering (push region + `defer_wait` + epilogue).
2. Clones the body into three sibling functions: `_push`, `_wait`, `_epi`.
3. Rewrites Orchestration `Submit`/`Call` sites that launched the original name
   into three **submits**: push → wait (`deps=[push]`) → epi (`deps=[wait]`),
   binding the original TaskId Var to the **epilogue** so consumers' `deps=[tid]`
   wait for full completion (including signal clear). Plain `Call` sites are
   promoted to `Submit` so the chain can carry TaskId deps (`OutlineIncoreScopes`
   already forces a synthetic TaskId for deferred waiters without `as tid`).

## Pipeline position

Default strategy: after `lower_auto_vector_split`, before `expand_mixed_kernel`
(`python/pypto/ir/pass_manager.py`).

## Related

- [`LowerCompositeOps`](13-lower_composite_ops.md) — emits fused or deferred barrier
- [`OutlineIncoreScopes`](09-outline_incore_scopes.md) — stamps `deferred_completion_waiter`
- [`ExpandMixedKernel`](25-expand_mixed_kernel.md) — deferred-waiter body contract
- Verifier property `DeferredCompositePlacementValid` (`99-verifier.md`) — rejects
  illegal `defer=True` placements at pipeline input
