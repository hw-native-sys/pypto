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
[`MemoryReuse`](31-memory_reuse.md) (it is that pass's former "Step 0") so that
the opportunistic lifetime coalescing can be skipped independently — e.g. when
ptoas owns lifetime reuse under `compile(memory_planner=MemoryPlanner.PTOAS)`.

**When to use**: Run after [`InitMemRef`](28-init_memref.md) (which creates the
MemRefs) and before
[`MaterializeInplaceAliases`](30-materialize_inplace_aliases.md). It always
runs; only the planner-specific may-alias stages are conditional.

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

## Notes

- Views/partial-views share a `base` but differ in `byte_offset`/`size`, so they
  are never merged into a must-alias buffer — only exact same-allocation vars are.
- In the default (`PYPTO`) pipeline this pass plus `MemoryReuse` compose to the
  behavior of the former single `MemoryReuse` pass (byte-identical output).
- Under `PTOAS`, `MaterializeInplaceAliases` follows this pass and encodes only
  legal `dst == dead-src` operation boundaries before PTOAS performs packing.
