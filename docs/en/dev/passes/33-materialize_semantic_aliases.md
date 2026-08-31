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
[`MemoryReuse`](34-memory_reuse.md) (it is that pass's former "Step 0") so that
the opportunistic lifetime coalescing can be skipped independently:

- `MemoryPlanner.DSA_RP` keeps independent allocation identities for the
  in-process DSA-RP solver.
- `MemoryPlanner.PTOAS` leaves lifetime reuse and address assignment to ptoas.

**When to use**: Run after [`InitMemRef`](32-init_memref.md) (which creates the
MemRefs) and before the selected memory planner. It always runs. `PYPTO` follows
it with [`MemoryReuse`](34-memory_reuse.md); `DSA_RP` consumes its allocation
identities in [`AllocateMemoryAddr`](35-allocate_memory_addr.md).

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
   view inputs). `IfStmt` return values are retargeted into both branch yields,
   then the collected type rewrites are applied.
2. **Normalize peeled accumulator phis**: visit nested `IfStmt` nodes in
   post-order and recognize both direct in-place accumulator producers and
   branch-local loops carried by an accumulator seeded outside that branch.
   When exactly one branch is the accumulator continuation, retarget the other
   branch's local seed, the phi result, aliases, and nested loop carry onto the
   reused input's canonical `Acc` allocation. Both the accumulator loop and the
   sibling seed must be local to their respective branches, and the target must
   be dead in the remainder of the seed branch. Whether the continuation is a
   direct `tile.matmul_acc` or a branch-local loop, its reused input and every
   bare/metadata alias must have no independent post-`if` read; otherwise the
   sibling branch would clobber an observable value on the path where the
   continuation does not execute.
3. **Normalize semantic identity chains**
   (`NormalizeIdentityCopyBuffersMutator`): make bare SSA copies share their
   source allocation and make every registered in-place result share its reused
   input allocation. This closes lowering-created type drift before any memory
   planner observes lifetimes or PTOAS emits tile handles.

The pass is a no-op when there is nothing to retarget (`Compute` returns no
rewrites), and skips `Orchestration` functions (no TileType variables).

## Relationship to codegen

PTO codegen renders variables that resolve to the *same* physical MemRef window
(`base` + `byte_offset` + `size` + pipeline-slot metadata) as a single
`tile_buf` handle, so after this
pass a loop-carried accumulator emits an in-place `pto.tadd ins(%acc, %t)
outs(%acc)` rather than writing to a distinct `%acc_next` buffer. Under
`memory_planner=DSA_RP`, each resulting allocation identity becomes one DSA
buffer; under `memory_planner=PTOAS`, codegen emits that identity without a
physical address for ptoas `PlanMemory`. See
[PTO Codegen — Who plans memory](../codegen/00-pto_codegen.md).

## Notes

- Views/partial-views keep their distinct `byte_offset`/`size` metadata. Under
  `DSA_RP`, all members that share one `base` belong to one physical allocation;
  placement moves that allocation as a unit and writeback preserves each
  member's relative offset. Sharing only the `base` is not enough to establish a
  must-alias relation: disjoint byte windows and different pipeline slots remain
  distinct until the producer is safely retargeted to the exact canonical
  window.
- In the default (`PYPTO`) pipeline this pass plus `MemoryReuse` compose to the
  behavior of the former single `MemoryReuse` pass.
- `DSA_RP` and `PTOAS` both skip opportunistic MemRef coalescing here; neither
  may undo a must-alias relation established by this pass.
- Accumulator-phi normalization runs for every memory planner before lifetime
  planning. The legacy `PYPTO` path repeats it after opportunistic reuse because
  reuse can introduce a fresh carry/phi mismatch.
- The preferred spelling for new matmul accumulators is a single
  `tile.matmul_acc(..., init_cond=...)`. Peeled `matmul`/`matmul_acc` branches
  remain supported for existing hand-written kernels and are normalized by this
  pass.
