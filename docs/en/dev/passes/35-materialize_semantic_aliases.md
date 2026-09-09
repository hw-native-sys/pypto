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
[`MemoryReuse`](36-memory_reuse.md) (it is that pass's former "Step 0") so that
the opportunistic lifetime coalescing can be skipped independently:

- `MemoryPlanner.DSA_RP` keeps independent allocation identities for the
  in-process DSA-RP solver.
- `MemoryPlanner.PTOAS` leaves lifetime reuse and address assignment to ptoas.

**When to use**: Run after [`InitMemRef`](34-init_memref.md) (which creates the
MemRefs) and before the selected memory planner. It always runs. `PYPTO` follows
it with [`MemoryReuse`](36-memory_reuse.md); `DSA_RP` consumes its allocation
identities in [`AllocateMemoryAddr`](37-allocate_memory_addr.md).

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

### Staged Buffer IR pipeline

`PassContext(enable_buffer_ir=True)` enables the storage-legalization portion of
the Buffer IR migration. The temporary development option defaults to false;
it does not by itself promise that every Tile operation or control-flow form
can be lowered to Buffer IR. The C++ accessor is `GetEnableBufferIR()` and the
Python accessor is `get_enable_buffer_ir()`. Compilation, IR dumping and
profiling preserve the active option, and JIT cache keys distinguish its value
for every memory planner.

With this option, `MaterializeSemanticAliases` also establishes explicit branch
destinations before `PYPTO`, `DSA_RP`, or `PTOAS` performs memory planning:

1. If both arms already yield the same physical window, retain that window.
2. Otherwise allocate an independent canonical destination for the result. An
   input yielded from outside a branch keeps its original storage, including
   when it remains live after the `IfStmt`.
3. A direct branch-local producer with unaliased, unpinned output storage may
   write the new destination when its registered operation contract allows it.
   Views and operations with a required input/output alias retain their storage.
4. Insert explicit `tile.move` operations in each remaining arm before its
   yield. Remove allocations made unused by producer retargeting. Branch copies
   and the existing For-carry fixups run before all three planners; `PYPTO`
   reconciles any new mismatch after reuse, retaining the declared phi target.

For example, `if flag: yield a; else: yield b` with `a` and `b` still live after
the branch receives a separate result allocation and one copy in each arm.
Two independent branch-local elementwise producers can instead write that same
result allocation directly, without a copy. PTOAS therefore receives explicit
branch transfers and does not need codegen to select destinations or add them.

The added branch analysis uses fixed IR walks and indexed lookups, with
O(N log N) work and O(N) storage. No persistent alias table is attached to IR.
Accumulator branches still use the existing guarded coalescing; a remaining
divergent `Acc` branch is rejected because Acc-to-Acc copying is unsupported.
Loop-input isolation, parallel transfers, While carries, and post-reuse storage
verification are described below. These establish canonical storage boundaries;
complete initialization and asynchronous lifetime verification remain separate.

### Default pipeline

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

The staged pipeline runs `VerifyTileStorage` after shared and post-reuse storage
reconciliation, before address placement. See the [storage property contracts](99-verifier.md#tile-storage-properties)
for symbolic closure and the separate allocated-address overlap check.

### Explicit loop and branch transfers

With `enable_buffer_ir=True`, shared legalization first isolates loop inputs
that remain independently observable. A read through the original input or a
metadata alias can require an entry copy; the copy runs before a `ForStmt` or
`WhileStmt`, preserving zero-iteration behavior. Overlapping initial carry
windows receive independent storage. Metadata-only views do not read data.
For nested loops, earlier data observations are conservatively included in the
indexed liveness decision to protect the next enclosing-loop iteration. A
top-level loop does not need isolation solely because its input was read earlier.

For and While initializers, iter_args, results, and result views are aligned
before producers are retargeted. Branch and loop yields use the same parallel
transfer scheduler: every source that may overlap a destination is snapshotted
before any destination write. This handles swaps, cycles, fanout, and partial
source overlap with O(N log N) indexed queries; scratch allocations are explicit
before placement. Post-reuse reconciliation repeats the same scheduling, and
address placement cannot add new scratch or transfers.

Storage requiring a same-space copy that the target cannot implement is rejected
with an early diagnostic; in particular, a live Acc input cannot be preserved
by an Acc-to-Acc move. The existing guarded accumulator producer coalescing
remains available for compatible carries. Ambiguous view addresses and
simultaneously overlapping destination windows must be resolved before final
Buffer lowering. The default legacy path retains its previous behavior.
