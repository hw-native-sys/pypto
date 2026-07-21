# MaterializeSemanticAliases Pass

Forces buffers that the program *semantics require* to be the same allocation to
share one MemRef, by propagating each loop-carried `iter_arg`/`initValue` MemRef
down the yield/producer chain.

## Overview

Memory planning distinguishes two kinds of buffer sharing:

- **Must-alias (semantics-required):** a loop-carried accumulator, or an in-place
  op result, *has* to live in one buffer — writing the "next" value must update
  the carried buffer, or the loop does not accumulate. This is correctness, not
  optimization.
- **May-alias (opportunistic):** two independent buffers with non-overlapping
  lifetimes *may* share storage to save memory. This is optimization.

This pass handles only the **must-alias** case. It was split out of
[`MemoryReuse`](30-memory_reuse.md) (it is that pass's former "Step 0") so that
the opportunistic lifetime coalescing can be skipped independently — either
when ptoas owns planning under `MemoryPlanner.PTOAS`, or when the standalone
solver jointly chooses reuse and offsets under `MemoryPlanner.DSA`.

**When to use**: Run after [`InitMemRef`](28-init_memref.md) (which creates the
MemRefs) and before [`MemoryReuse`](30-memory_reuse.md). It always runs; only the
opportunistic reuse is skippable.

## API

| C++ | Python | Level |
| --- | ------ | ----- |
| `pass::MaterializeSemanticAliases()` | `passes.materialize_semantic_aliases()` | Function-level |

```python
from pypto.pypto_core import passes

program = passes.materialize_semantic_aliases()(program)
```

## Algorithm

`InitMemRef` already gives the loop-carried `iter_arg` and `return_var` the same
MemRef as the `initValue` (the accumulator buffer), but the *producer* of the
yielded value — e.g. the `tile.add` that computes `acc_next` — is still assigned
its own fresh MemRef. This pass closes that gap:

1. **Top-down retarget** (`TopDownRetargeter`): for each `ForStmt`, take each
   `iter_arg`'s canonical MemRef as the target and push it onto the yielded value
   and its producer chain (following in-place `output-reuses-input` ops and
   view inputs). `IfStmt` return values are retargeted into both branch yields.
2. **Apply retype** (`RetypeApplier`): rewrite the collected variable types in
   place so the producer writes directly into the carried buffer.
3. **Reconcile remaining external-planner carries** (`YieldFixupMutator`): when
   `MemoryReuse` is skipped, insert explicit moves for producer/view shapes that
   cannot be retargeted safely. PTOAS materializes loop-carry moves here and
   handles if-phi copies in its addr-less codegen. DSA emits explicit addresses,
   so it materializes both if-phi and loop-carry moves before lifetime export.

The pass is a no-op when there is nothing to retarget (`Compute` returns no
rewrites), and skips `Orchestration` functions (no TileType variables).

## Relationship to codegen

PTO codegen renders variables that resolve to the *same* MemRef identity
(`base` + `byte_offset` + `size`) as a single `tile_buf` handle, so after this
pass a loop-carried accumulator emits an in-place `pto.tadd ins(%acc, %t)
outs(%acc)` rather than writing to a distinct `%acc_next` buffer. Under
`memory_planner=PTOAS` (no physical `addr` baked, `MemoryReuse` skipped) this is
what lets ptoas `PlanMemory` keep the accumulator in one buffer while still
doing the lifetime reuse and address assignment itself. See
[PTO Codegen — Who plans memory](../codegen/00-pto_codegen.md).

DSA cannot use the PTOAS-only codegen repair because its level-3 PTO contains
explicit addresses. Its if-phi copies therefore remain ordinary IR operations,
participate in DSA lifetime analysis, and are validated with the placement.

## Notes

- Views/partial-views share a `base` but differ in `byte_offset`/`size`, so they
  are never merged into a must-alias buffer — only exact same-allocation vars are.
- In the default (`PYPTO`) pipeline this pass plus `MemoryReuse` compose to the
  behavior of the former single `MemoryReuse` pass (byte-identical output).
