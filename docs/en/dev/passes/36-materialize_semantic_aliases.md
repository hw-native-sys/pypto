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
[`MemoryReuse`](37-memory_reuse.md) (it is that pass's former "Step 0") so that
the opportunistic lifetime coalescing can be skipped independently:

- `MemoryPlanner.DSA_RP` keeps independent allocation identities for the
  in-process DSA-RP solver.
- `MemoryPlanner.PTOAS` leaves lifetime reuse and address assignment to ptoas.

**When to use**: Run after [`InitMemRef`](35-init_memref.md) (which creates the
MemRefs) and before the selected memory planner. It always runs. `PYPTO` follows
it with [`MemoryReuse`](37-memory_reuse.md); `DSA_RP` consumes its allocation
identities in [`AllocateMemoryAddr`](38-allocate_memory_addr.md).

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
   then the collected type rewrites are applied. The traversal descends through
   `WhileStmt` bodies, so an enclosing scalar while does not hide nested For
   carries from the existing retargeting and liveness checks. Before changing a
   producer's destination, check independent reads of the target's physical
   window that can recur through an enclosing loop, including reads outside the
   inner For's lexical lifetime. This includes the same For's backedge when a
   body read names the original pre-loop initializer; reads through the advancing
   carry itself are excluded by its loop-header definition. Binding a loop initializer to a carry is not
   itself an independent data observation. For an actually observed For/While
   carry, record its initializer's payload read at the binding header. A worklist
   propagates these entry observations through nested carry bindings, preserving
   intermediate versions across intervening loops. Body reads retain the carry's
   current definition: its own backedge advances the value rather than rereading
   the original seed. Header reads exclude their own loop from recurrence, while
   enclosing loops still protect repeated initialization. This distinction also
   preserves accumulator-phi coalescing through metadata-view yields.
   Consuming a possibly zero-trip For/While result also observes its initializer
   at entry, even when the body never reads the carry. Constant For bounds that
   prove at least one iteration exclude this zero-trip observation; unknown
   bounds remain conservative. Result observations participate in the same
   worklist, including results used to initialize another observed loop carry.
   Bare SSA assignments, zero-copy view sources, and bare-variable yields are
   forwarding bindings, not payload reads. Both alias and physical recurrence
   indexes and Buffer IR loop-input isolation share the activated payload
   observations; ordinary access indexes still
   retain all forwarding uses. Only when an alias, branch/loop result, or later
   iteration actually consumes the value does the worklist activate its source
   observations. Constant For bounds disable body-result edges for zero trips
   and backedge edges for at most one trip. Ordinary body observations are also
   excluded for proven zero-trip For loops and constant-false While conditions.
   A For with at most one iteration contributes no recurrence interval, though
   enclosing repeating loops still protect its actual reads. Registered
   write-only operands do not observe old payloads; reads, read-modify-writes,
   computed arguments and reference attributes retain their observations.
   Unknown bounds and conditions remain conservative. This also preserves
   consumed forwarding results of nonempty loops, without making unused Acc
   carries falsely block retargeting and require an unsupported Acc-to-Acc copy.
   Buffer IR also avoids private loop-input storage for proven zero-trip loops
   and when the complete body only
   yields every carry unchanged: this binding cannot overwrite its initializer.
   Zero-trip body canonicalization applies only to Tile carry boundaries;
   Tensor/DistributedTensor-only loops retain their descriptor writebacks and
   body structure, handled by tensor boundary normalization.
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
   continuation does not execute. This post-order traversal also descends through
   `WhileStmt` bodies. For/While carry arguments and results belong to their
   initializer's alias family, so reading an initializer cannot bypass this
   check through a carry name. Reads before the `if` also block coalescing when
   an enclosing loop can repeat them after the branch. The recurrence index
   considers loops entered after the read handle's definition; an accumulator
   recreated inside the loop can still be coalesced safely. These queries use
   precomputed alias and loop-interval indexes, retaining O(N log N) analysis.
3. **Normalize semantic identity chains**
   (`NormalizeIdentityCopyBuffersMutator`): maintain every required alias relation
   after retargeting, rather than repairing one producer in isolation. Bare SSA
   aliases follow their source, registered `output_reuses_input` results follow
   the specified writeback input, and registered first-input buffer-aliasing
   operations follow their current first input. A single forward walk substitutes
   each repaired result into downstream uses, so a writeback result followed by
   `transpose_view`, `reshape`, or `slice` stays on the correct storage throughout
   the chain; no fixed-point iteration is needed. This closes lowering-created
   type drift before any memory planner observes lifetimes or PTOAS emits tile
   handles.

These rules cover shaped results with MemRefs, including tile and tensor
writebacks. Bare SSA renames follow the complete source type without copying
data; call-result retyping preserves result shape, dtype, view/distribution
metadata. An explicit `output_reuses_input` contract takes precedence over
first-input inheritance. The first-input buffer-alias predicate requires both
`OutputMemoryInheritsInput()` and `IsInplaceSafe()`; inheriting the memory space
alone does not establish a buffer alias.
Writeback results inherit the current writeback input's exact MemRef window.
First-input aliases retain their result extent and derive their window from the
current input allocation and byte offset plus the operation's relative view
offset, applied once; they also inherit the input's pipeline slot identity and,
for tile results, memory space. Recomputing the relative offset from the
operation, rather than adding it to a stale absolute offset, makes normalization
idempotent and handles changes within the same allocation as well as changes of
allocation. Buffer aliasing is a storage contract: `tile.assemble` also follows
its first input even though it writes data. Real non-in-place `tile.transpose` and independent
`tile.extract` results keep their own storage.

In both legacy Tile and Buffer IR modes, `TensorType`/`DistributedTensorType`
For/While arguments and returns are mutable views of the current initializer.
Normalization aligns their storage while preserving each boundary result's
shape, view metadata, and distributed-window identity, including on zero-trip
loops. Independently produced yielded tensors keep their own storage. Tile
carries retain their separate canonical-buffer and yield-fixup rules, including
explicit transfers when a yielded tile needs to reach the carry buffer.

The pre-yield-fixup form of normalization seeds unresolved `IfStmt` results and
propagates that status through bare aliases, writeback results, and first-input
aliases, including through tensor loop arguments and returns. None of the
dependent calls is anchored to that provisional window.
Once the control-flow boundary is resolved, the following normalization walk
repairs the whole chain. Buffer IR also uses this deferred form before indexing
old-value observations and after loop-entry isolation, while branch storage is
still provisional; normalization after boundary resolution needs no deferred
seed. See [MemoryReuse step 6](37-memory_reuse.md#algorithm) for the normalization
walks around yield fixup.

The pass skips orchestration-like functions, including graph bodies. In other
functions, alias normalization still runs when `Compute` returns no retargeting
rewrites; it can repair existing type drift independently.

## Relationship to codegen

### Staged Buffer IR pipeline

`PassContext(enable_buffer_ir=True)` enables the storage-legalization portion of
the Buffer IR migration. The option defaults to true. `SelectBufferIRFunctions` records whole-kernel
support before this pass; only selected functions receive Buffer-specific
rewrites. Unsupported functions keep the Tile storage path. The C++ accessor is `GetEnableBufferIR()` and the
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

Branch planning propagates required aliases in the same forward walk between
successive phi storage choices, so later phis see current windows rather than
abandoned ones. For/While carries follow their current initializers before the
body is visited, and loop results follow those canonical carries afterward,
including the zero-iteration path. Handle counts follow planned allocation
changes and rollback, keeping optional producer retargeting conservative.

Trailing yields inside transparent `SplitAivScopeStmt` and `RuntimeScopeStmt`
wrappers are supported. Transfers stay inside the same scope immediately before
the yield; yields belonging to nested control-flow regions are left unchanged.

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
- On the legacy `PTOAS` path, if both branches yield the same tile byte window
  (allocation, offset, size, slot count, and slot index), and both yields and the
  result have the same known memory space, the merged result retains that window
  without a copy. Semantic aliases follow the repaired result while preserving
  their type/view metadata. Registered views rebase their relative byte offset
  onto the repaired source, preserving their own extent and inheriting its
  pipeline slot identity;
  subsequent whole-window writeback results follow that view. Reusing an existing
  mixed-type codegen handle additionally requires matching its declared SSA type,
  both yields' window and current binding, and scope dominance. If an earlier
  merge selected a new dominating handle, later merges use that common yielded
  binding even when the identity map still names a rejected sibling handle.
  Distinct branch windows keep their existing handling.
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
A read can recur across iterations only in an enclosing loop entered after the
read handle's logical definition. Indexed recurrence intervals protect such
reads, including those before an inner loop or in a sibling branch. A seed
recreated inside each outer iteration does not need isolation merely because it
was read earlier in that iteration; bare aliases and metadata views inherit the
seed's definition. Reads outside a repeating region cannot recur within it.
Forward observations still use both branch orders to exclude mutually exclusive
sibling arms. The analysis remains O(N log N) without ancestor walks.

For and While initializers, iter_args, results, and result views are aligned
before producers are retargeted. Branch and loop yields use the same parallel
transfer scheduler: every source that may overlap a destination is snapshotted
before any destination write. This handles swaps, cycles, fanout, and partial
source overlap with O(N log N) indexed queries; scratch allocations are explicit
before placement. Post-reuse reconciliation repeats the same scheduling, and
address placement cannot add new scratch or transfers.

Storage requiring a same-space copy that the target cannot implement is rejected
before any entry copy, yield transfer, or snapshot is synthesized. Same-space
Mat, Left, Right, and Acc transfers receive an early diagnostic; a live Acc input cannot be preserved
by an Acc-to-Acc move. The existing guarded accumulator producer coalescing
remains available for compatible carries. Ambiguous view addresses and
simultaneously overlapping destination windows must be resolved before final
Buffer lowering. Unselected kernels and explicit Off mode retain their previous Tile behavior.

The default Buffer rollout selects complete supported kernels before storage
legalization; see [whole-kernel selection](55-lower_tile_to_buffer.md#default-whole-kernel-selection).
