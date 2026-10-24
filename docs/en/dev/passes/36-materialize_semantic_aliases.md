# MaterializeSemanticAliases Pass

Establishes semantic storage relations and one destination plan for every tile-valued
`IfStmt` result before any memory planner runs.

## Overview

Loop carries and registered in-place results require shared storage. An if result
instead selects one of two incoming SSA values; that selection does not make the
two sources mandatory aliases. This pass resolves those relations and materializes
the required edge transfers. Opportunistic lifetime reuse remains the planner's job.

Run after [`InitMemRef`](35-init_memref.md). All three planners consume the same
normalized control-flow storage:

- `PYPTO` follows with [`MemoryReuse`](37-memory_reuse.md).
- `DSA_RP` places the resulting allocation identities in
  [`AllocateMemoryAddr`](38-allocate_memory_addr.md).
- `PTOAS` leaves lifetime reuse and address assignment to ptoas.

The pass skips orchestration-like functions, including graph bodies.

## API

| C++ | Python | Level |
| --- | ------ | ----- |
| `pass::MaterializeSemanticAliases()` | `passes.materialize_semantic_aliases()` | Function-level |

```python
from pypto.pypto_core import passes

program = passes.materialize_semantic_aliases()(program)
```

## Algorithm

1. **Collect mandatory storage constraints.** For/While carries use their
   initializer's storage, including on zero-trip loops. Preserve registered
   writeback contracts and collect compatible accumulator continuation targets
   before selecting if destinations. Guarded coalescing can redirect a local
   `matmul` seed onto a continuing `matmul_acc` buffer; an Acc-to-Acc copy is
   never the default way to realize that relation. All planners use common loop
   entry isolation and `RetypeApplier`'s scoped For/While binding rewrites. `In` parameters protect
   the caller's old value; `InOut`/`Out` parameters require isolation when an
   independent old-value observation inside the function proves it necessary.
   Isolation first indexes actual native, registered and workspace writes plus
   possible yield writebacks. Independently mutable carry slots with overlapping
   windows require isolation; otherwise an old-value observation requires a copy
   only when the iteration may write the window. A non-overlapping read-only
   For/While pass-through, including SSA aliases and same-source if results,
   retains its original window without introducing a Mat/Acc copy.
   Old-read, recurrence and readonly-caller queries retain actual and projected
   byte windows and pipeline slots. A read from disjoint bytes or a different
   proven slot does not require entry isolation. Old handles are activated by
   definition age, and reachable reads must satisfy both branch-order bounds
   within the same indexed subset; unrelated reads from mutually exclusive
   paths cannot combine their positions to manufacture an observation.
   Isolation and if-result mutability use the same frozen native/registered-write
   and possible-backedge-transfer proof. Loop entry alone is not a data write.
   Later writes follow mandatory storage aliases through readonly loop returns;
   an actual isolation copy cuts that chain. Logical value versions still remain
   distinct for old-value reads and lifetime analysis.
   Reinserting a complete updated carry result into the same source slice
   does not read that slot's old bytes only when its valid shape covers the
   complete physical shape. Vec insertion otherwise writes only valid elements;
   retargeting a full-box producer could corrupt the preserved invalid region.
   A byte/slot/layout/coordinate proof excludes only that producing slot from
   the target's preserved-outside read. Partial-valid results and independent
   old-row or whole-target observations retain their ordinary protection.
2. **Record each if result and its incoming edges.** A pass-local `IfInfo` records
   the owning `IfStmt`, result slot, result SSA variable, and the then/else source
   variables. Tile results require both legal yields. `IterArg` sources retain
   their own identity instead of being replaced by their initializer. An outer
   edge can reference an inner result without flattening the inner alternatives.
   Result identity never comes from a variable name or a MemRef address.
3. **Plan destinations once.** `InitMemRef` gives an unbound result its own
   unplaced request, sized from the result's physical layout and alignment.
   Inferred result types retain shape, dtype and view metadata without inheriting
   a source MemRef; explicit result bindings remain constraints. For each if,
   reserve all slots' required targets and legal common incoming windows first.
   Only then consider optional incoming-storage borrowing or independent requests
   for the remaining slots. Yields pass values simultaneously, so slot order
   cannot displace another slot's common window. Required loop/Acc targets
   reference their actual SSA storage owner and resolve its current window when
   consumed. They retain byte offsets, extents and pipeline slots as an outer
   result moves; explicit fixed bindings remain fixed. The owner metadata follows
   the same transaction rollback and bounded replay as its producer rewrites.
   Borrowing must preserve old-value observations, loop recurrence, operator
   no-alias rules, pipeline separation and protected storage. The other arm may
   write the borrowed destination directly or use a supported edge transfer;
   producer retargeting is not a prerequisite for borrowing.
   Identical incoming windows can also retain a native mutable continuation,
   provided independently observed old values and read-only caller inputs remain
   protected. Being writable alone does not require a new destination. Optional
   duplicate-result sharing keeps its stricter read-only rule.
   Pipeline membership prevents borrowing or moving a different allocation;
   it does not prohibit keeping the exact common byte/slot window. Workspace
   and load/tpop hardware hazards still forbid creating a mutable continuation.
   Recheck the same common-window proof after earlier slots retarget their
   producers, because a later result's two inputs can become common then.
   Publish each settled live destination to incremental byte/slot indexes before
   selecting the next result. A mutable candidate cannot overlap any live peer;
   a readonly candidate cannot overlap a mutable peer. The same proof applies
   to preferred, equivalent and borrowed windows, in either result-slot order.
   Incompatible mandatory destinations are rejected instead of silently sharing.
   Future-clobber checks use both sibling orders, each with its own last read
   and proven terminal-writeback boundary. Only possible clobber in both orders
   rejects sharing.
4. **Apply the plan.** Planning uses a frozen IR snapshot and pending rewrites;
   apply the chosen result and producer destinations together, then close their
   required aliases. Eligible branch-local producer families can write the
   destination directly. Nested destinations may be forwarded only with a proof
   of complete equal-extent windows in the same memory space. Distinct result
   slots remain distinct logical values even when their sources or windows match.
   Full/short descriptors retain their own metadata; an edge requirement never
   changes another source value's shape or valid extent.
5. **Realize and validate if-edge transfers.** Each arm supplies its result slot
   directly or through a legal transfer into the selected destination. Parallel
   yields preserve all source values before any conflicting write, using
   explicit scratch when needed. Equivalent source/destination windows and
   descriptors may share a transfer; conflicting values targeting the same
   constrained bytes are rejected. Loop carries retain independent target slots.
   A required result target can be filled by an edge copy after the branch's old
   reads finish, provided no old target observation remains after the merge.
   Otherwise keep the result independent until the enclosing boundary. Memory
   spaces without a legal same-space transfer still require direct producer retargeting.
6. **Finish if planning before lifetime packing.** Close aliases, validate the
   materializable incoming edges and remove unused requests and allocations.
   Discard the planning snapshot.
   Later packing may relocate allocation groups and close their aliases, but
   must not infer new result classes or choose another arm as the destination.
   Legacy `PYPTO` retains loop-copy scheduling after packing and before address
   assignment, using the already established carry targets.

An independent request does not imply an extra physical buffer: common windows
can be retained, producers can write the chosen destination directly, and unused
requests disappear before capacity calculation. A separate destination is required
when independently live values or hardware constraints prevent sharing.

Destination selection uses three fixed walks over the result slots. The snapshot
and indexed control-flow analyses use map/index lookups, with O(N log N) work
and storage. The immutable storage-access index queries byte windows and
pipeline slots; its time envelopes conservatively report possible overlap.
Old-value queries first activate only handles defined before the if, then check
reads after it in both sibling orders. An enclosing mutually exclusive arm is
therefore excluded; a loop backedge retains a virtual read for external values
needed on the next iteration. Read windows stay distinct even if an unused
whole-allocation handle covers them both. Already relocated values use a
monotone byte/slot observation index, with fixed-width sparse range trees,
so incremental planning does not collapse independent windows or scan earlier
results. No persistent planning table is attached to the public IR.

## Alias and view contracts

Bare SSA aliases follow their source. Registered `output_reuses_input` results
follow the specified writeback operand. Registered first-input buffer aliases
follow that input's allocation, offset and pipeline slot, adding the operation's
relative view offset once while retaining their own extent. Memory-space
inheritance alone is not a buffer-alias contract: the predicate also requires
`IsInplaceSafe()`. Independent `tile.extract` and non-in-place `tile.transpose`
results keep their own storage.

Call-result retyping preserves shape, dtype, view and distribution metadata.
`TensorType`/`DistributedTensorType` loop boundaries follow their initializer's
window without retargeting independently produced yielded tensors. Metadata-only
views do not read their first operand's data, but their other arguments and
registered effects are still visited. Writing aliases such as `tile.assemble`
produce a new logical value version even when storage is shared. Before the
destination plan is resolved, normalization
must not anchor a dependent writeback onto a result's temporary request.

Source windows retain their physical root, pitch and offsets independently of
their logical tile shape and value version. Projected windows follow partial
views, if results and loops without losing byte offsets or pipeline slots.
Known GM writes retain their actual windows, including aliases of typed DDR
Tiles. Unknown GM/scalar storage does not become an opaque on-chip write;
unknown pointer effects remain conservative. Supported narrowed Vec transfers use `tile.extract`
from that root into the selected destination; dense transfers remain ordinary
moves. Runtime slice coordinates are captured once at their definitions.
For/While window provenance follows entry storage through carries and results,
including zero-trip loops. Unproven bindings keep a conservative window marker
without inventing source coordinates.

Transparent `SplitAivScopeStmt` and `RuntimeScopeStmt` wrappers retain transfers
immediately before their own trailing yield. Nested control-flow yields belong
to their own result records.

## Relationship to codegen

### Common planner boundary

Both legacy Tile kernels and selected Buffer kernels pass through the same if
destination planning. `PassContext(enable_buffer_ir=True)` additionally enables
whole-kernel selection and Buffer-specific storage legalization; unsupported
kernels remain on Tile IR. The option is preserved by compilation, IR dumping,
profiling and JIT cache keys.

Every planner receives the common plan and materialized if-edge obligations for
known storage. Legacy opaque tile sources without MemRefs retain their recorded
incoming SSA edges but defer lowering to the existing handle/codegen path; no
source coordinates or copies are invented. A selected Buffer function instead
rejects an input that cannot be materialized. `MemoryReuse` does not repeat
resolved result selection, and codegen does not choose an alternative allocation
for those resolved results. Same-descriptor whole-root copies may disappear
after legal placement makes their windows identical; a retained copy must still
have disjoint windows. Fixed caller-owned and declared storage retain their
address-validation requirements.

### Typed handles

A physical window includes allocation identity, byte offset, extent and pipeline
slot metadata. Sharing only the base does not make sibling windows identical.
PTOAS keeps the same-window alias behavior and the typed, dominating-handle
rules from [#3029](https://github.com/hw-native-sys/pypto/pull/3029): a handle
must match the required SSA descriptor and dominate its use. A branch-local
handle cannot escape merely because its address matches the result's address.
Different logical descriptors may therefore need distinct typed handles for one
physical window.
The result's validity and padding belong to its own handle, even when dynamic
valid extents print as the same type. Initialize its metadata view at the if
boundary without changing the canonical allocation's metadata. Full-span reshape
aliases already carry the same bytes and need no identity transfer; subviews
retain distinct origins.

`DSA_RP` consumes symbolic allocation identities; under `PTOAS`, codegen omits
their physical addresses for ptoas `PlanMemory`. See
[PTO Codegen — Who plans memory](../codegen/00-pto_codegen.md).

## Example

```python
if flag:
    r = pl.yield_(a)
else:
    r = pl.yield_(b)
use(r)
use(a)
```

The later observation of `a` can prevent using its storage as the result's
destination. Choose independent storage or another proven-safe destination;
write it on each incoming edge without overwriting old `a`. If both arms contain
independent local producers, those producers may write the same destination
directly and need no copies.

## Notes

- Loop-entry copies execute before the loop, preserving zero-trip behavior.
  Indexed old-value observations and recurrence checks distinguish an enclosing
  loop's next iteration from a seed recreated in the current iteration.
- Resolve copy legality after mandatory producer/carry constraints. Fresh local
  Acc producers can write a destination directly; reject a remaining unsupported
  Acc-to-Acc transfer at the boundary that actually requires it.
- Prefer `tile.matmul_acc(..., init_cond=...)` for new accumulator code. Existing
  peeled `matmul`/`matmul_acc` forms retain guarded support.
- Address placement cannot add an unplanned scratch allocation or edge transfer.
  Symbolic edge closure and allocated-address overlap are separate checks; see
  [storage property contracts](99-verifier.md#tile-storage-properties).
- The Buffer rollout selects complete supported kernels before lowering; see
  [whole-kernel selection](55-lower_tile_to_buffer.md#default-whole-kernel-selection).
