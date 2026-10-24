# MemoryReuse Pass

Uses dependency analysis to identify memory reuse opportunities and removes redundant alloc operations.

## Overview

This pass analyzes variable lifetimes and dependencies to enable memory sharing. Variables with non-overlapping lifetimes in the same memory space can share MemRef objects, reducing memory footprint.

After applying MemRef sharing, the pass also **removes redundant `tile.alloc` statements** for MemRefs that are no longer referenced by any TileType variable.

**Key insights**:

- Variables that don't overlap in lifetime can reuse memory
- Only variables in the same memory space can share MemRef
- Lifetime is determined by def-use analysis
- After sharing, MemRefs that become unreferenced are cleaned up along with their alloc statements

**When to use**: This is the opportunistic reuse stage for
`MemoryPlanner.PYPTO`. It runs after
[`MaterializeSemanticAliases`](36-materialize_semantic_aliases.md) and before
[`AllocateMemoryAddr`](38-allocate_memory_addr.md). `MemoryPlanner.DSA_RP`
skips it so independent buffers remain visible to the DSA-RP solver;
`MemoryPlanner.PTOAS` skips it because ptoas owns lifetime reuse.
Semantic loop/in-place aliases and materializable if-edge obligations are
already established by `MaterializeSemanticAliases` in all three modes;
legacy opaque inputs retain the existing handle/codegen path.

## Planner boundary

`MemoryReuse` selects shared MemRef identities before address assignment.
`DSA_RP` instead keeps those identities separate and jointly chooses their
addresses under capacity and reuse penalties in `AllocateMemoryAddr`. Running
both would erase alternatives before DSA-RP can evaluate them. Correctness
facts collected for this pass—lifetime interference, semantic no-alias rules,
and target hazards—remain hard constraints in the DSA-RP problem.

## API

| C++ | Python | Level |
| --- | ------ | ----- |
| `pass::MemoryReuse()` | `passes.memory_reuse()` | Function-level |

**Factory function**:

```cpp
Pass MemoryReuse();
```

**Python usage**:

```python
from pypto.pypto_core import passes

reuse_pass = passes.memory_reuse()
program_optimized = reuse_pass(program)
```

## Algorithm

1. **Lifetime Analysis**: Consume the common control-flow storage established by
   `MaterializeSemanticAliases`; do not select if destinations again. Result `r`
   is a logical definition at the merge. Each arm reads its own yield operand on
   its incoming edge; a later read of `r` keeps the result live rather than
   independently extending both sources. Required aliases retain their allocation
   groups and relative windows. Loop carries and values needed again on a
   backedge remain live through the loop boundary.

   The temporary requests created by `InitMemRef` have already been resolved
   through the common per-slot `IfInfo` plan. Mandatory loop/Acc constraints,
   producer retargeting and materializable if-edge transfers precede packing for
   every planner. Legacy opaque sources retain their handle/codegen path. An
   unused result request does not contribute to peak capacity.
2. **Interference Check**: Summarize each allocation group in both then-first
   and else-first lexical order. Each interval starts at the earliest logical
   definition and ends at the latest read across all required aliases, including
   loop recurrence. If the group intervals are disjoint in either order, their
   storage can be shared, subject to the remaining guards. Touching is allowed.
   Every executable path respects both orderings, so this is conservative:
   different reads in nested sibling branches can make both interval hulls
   overlap and prevent otherwise legal reuse.
   The second numbering is always part of the analysis, not a packing retry or
   a capacity-dependent rule; no phi-family exemptions are needed. The added
   numbering and summaries take O(N log N) time and O(N) storage. Each root-pair
   lifetime check is O(1), with no alias-member pair scan. The existing packing
   search remains O(KM²) for M allocation groups and K packing attempts; this
   does not make the entire pass O(N log N).
3. **MemRef Sharing** (global first-fit-decreasing packing, `IdentifyReuseOpportunities`): Within each memory space, intervals are packed **largest-first**; every later interval joins the first existing buffer all of whose members it can share with (non-overlapping lifetime + hazard/no-alias safe — see `can_share`). A buffer's allocated size is fixed by its first (largest) member, so admitting a smaller member afterwards costs nothing — and a *later, larger* interval can now host an *earlier, smaller* one. (The previous definition-order greedy had a one-directional size gate `source.size >= target.size`, so two lifetime-disjoint tiles whose small one was defined first were never coalesced.) The representative each member is rebased onto is the buffer's largest member; its `tile.alloc` dominates the whole function because InitMemRef hoists every alloc to the function-body head, so a representative defined after some of its members is safe. Because the packer no longer runs in program order, every pairwise gate (hazard, no-alias) is checked in both directions.

   Whole-root `tile.move` copies can be coalesced when source and destination
   have the same tile descriptor apart from the MemRef, equal extents and
   matching slot metadata. This includes incoming-edge copies and ordinary
   transfers. If placement makes their windows identical, remove the assignment
   and substitute its SSA source into later uses before codegen; hardware never
   executes an in-place `TMOV` for these coalesced copies. Copies involving fixed
   caller-owned or declared storage are retained so address restrictions remain
   validated.
   A retained copy needs disjoint windows. Subrange packing conservatively
   separates these source/destination pairs to prevent partial overlap. Other
   no-alias, pipeline, target-hazard and declared-allocation constraints still
   apply. Incoming Tile-parameter allocations remain fixed: body-local aliases
   and results cannot move their storage or add new occupants.

   **Capacity-overflow subrange fallback (PYPTO only).** The pass first computes the whole-buffer packing above and keeps it byte-for-byte when its exact allocator footprint fits the backend capacity. Only when that footprint overflows does it retry packing smaller roots into aligned subranges of an existing largest-member arena. For example, after a dead 64 KiB root has reserved `[0, 65536)`, two later 32 KiB roots that are live together may occupy `[0, 32768)` and `[32768, 65536)`. The 64 KiB lifetime must not overlap either smaller root; the two smaller roots must occupy disjoint byte ranges because they interfere. The retry selects the lowest aligned gap not blocked by a member that cannot share bytes with the candidate. It never grows an arena, so two 32 KiB arenas do not become a new 64 KiB arena; the 64 KiB representative must already exist. If subdivision still does not fit, the existing pipeline-depth shed and legacy fallback run unchanged.

   Subdivision moves a complete root allocation; it never packs into holes outside an alias/view's accessed window. The candidate and arena roots must therefore have constant, non-negative allocation offsets. A symbolic offset is supported only on a direct internal `tile.slice`, whose runtime address PTO reconstructs from the translated source tile and the slice operands. If a memory space contains another dynamic-offset member, such as a reshape, reinterpret, or transpose view of that slice, subdivision is disabled for the complete space so compacting unrelated roots cannot turn a legacy overflow into wrong code. Constant member offsets are still checked to be non-negative and in bounds; supported symbolic slices keep the exact relative expression `arena_offset + placement + (member_offset - root_offset)`. A dynamic root offset remains ineligible, and pinned/declared allocations remain closed. Lifetime interference, pipeline separation, target hazards, and op no-alias rules still call `can_share`; if any such rule prevents byte overlap, the two root address intervals must be disjoint. `AllocateMemoryAddr` sizes the physical slot from the arena root's complete allocation capacity; every constant subrange was already checked to stay within that capacity. A pure dynamic `tile.slice` MemRef then retains the legacy bare-base address because PTO lowers its actual window as `pto.subview` of the already placed source tile using the slice operands. Downstream reshape folding compares base plus offset and tile signature, while PTO DPS aliasing also compares the extent. Sibling root subranges therefore cannot be mistaken for the same tile merely because their base pointers match.
4. **Relocate complete storage groups**: Apply the packing offsets to producers,
   if results and required aliases while preserving relative windows and pipeline
   slots. For/While arguments and returns follow their relocated initializer
   storage, including nested and zero-trip loops. This changes physical placement,
   not logical result identity or the incoming source relation.
5. **Close semantic aliases** (`NormalizeIdentityCopyBuffersMutator`): A forward
   walk propagates current storage to downstream uses:

   | Alias category | Required result storage |
   | -------------- | ----------------------- |
   | Bare SSA alias `x = y` | Current source's exact MemRef window |
   | Registered `output_reuses_input` result | Exact window of the registered writeback input |
   | Registered first-input buffer alias | Current input allocation/offset/slot plus the operation-relative offset; retain result extent |
   | Tensor/DistributedTensor For/While boundaries | Current initializer's exact window, including zero-trip loops |

   Call-result retyping preserves shape, dtype, view and distribution metadata.
   An explicit writeback contract takes precedence over first-input inheritance;
   the latter also requires `IsInplaceSafe()`. Recompute view-relative offsets
   from the operation and add them once. Independent `tile.extract` and
   non-in-place `tile.transpose` results retain their own storage. Forward
   substitutions close alias chains without fixed-point iteration.
6. **Validate control-flow storage**: The common plan already selected the
   result classes and materialized known if-edge obligations before packing.
   Check that both if edges and loop bindings still agree with their relocated
   destinations. Do not derive another destination from then/else storage.
   Legacy `PYPTO` still schedules loop copies and scratch after packing, using
   the established carry targets and finishing before address assignment.
   This does not rerun the if-result plan. A retained copy must remain legal
   for its source/destination windows and descriptors. Accumulator constraints
   are settled before packing rather than repaired with an Acc-to-Acc copy.
7. **Remove redundant allocs**: Collect the MemRefs still referenced by tile
   values and remove unused `tile.alloc` statements.

**Reuse conditions**:

- Disjoint conservative intervals in at least one sibling order. In straight-line code,
  `prev.last_use <= curr.def` permits sharing: inputs are consumed before outputs
  are written at the same statement. Checking both orders also permits an
  else-only old-value read and a then-only replacement to share storage when
  their intervals separate in the reverse order.
- Same memory space
- A buffer is sized to its **largest** member; because packing is largest-first, every member admitted later is no larger than the representative, so no explicit byte-size check is needed (and the reuse direction is no longer constrained to "earlier-and-larger only")
- Subrange placement is attempted only after an exact known-capacity overflow. A candidate must fit wholly inside the existing representative's capacity at an address accepted by the backend alignment policy; every simultaneously live or otherwise non-shareable member blocks its complete byte range.
- **No-alias guard** (op-semantic): the op that defines the reusing variable may forbid its output from sharing a buffer with one or more of its input operands, because the hardware reads those inputs *while* writing the output — an in-place write would corrupt the op mid-flight. Three sources feed one per-output forbidden-input set (`ForbidAliasCollector`):
  - `not_inplace_safe()` — the op cannot run with `src == dst`, so its output must not alias **any** input operand.
  - `forbid_output_alias(i)` — the op is in-place-safe w.r.t. its value operands but reads a **specific** operand while writing its output, so the output must not alias that one operand's buffer.
  - **dtype-changing `tile.cast`** — widening casts require disjoint source and destination buffers on every target. A2/A3 narrowing casts also require disjoint buffers: TCVT processes the full-repeat heads of all rows before their tails, so later output rows can overwrite unread source tails ([#2973](https://github.com/hw-native-sys/pypto/issues/2973)). This conservative rule applies regardless of shape or dynamic valid shape; A5 narrowing and same-width casts retain reuse. One shared predicate enforces the rule in both allocation planners (PyPTO and DSA_RP), control-flow retargeting, and PTO codegen validation of explicit physical addresses. PTOAS-managed allocation omits that address check because physical addresses are not assigned yet. TPOP sources and their views use reserved pipe storage without a MemRef; codegen validates explicit source addresses only, while their cast results still receive general-pool allocations.

  MemoryReuse refuses to place the output on a forbidden operand's **physical buffer**, resolving each operand through both reuse-map coalescing *and* VIEW inheritance (a `reshape`/`slice` shares its source's MemRef base) — so a forbidden operand reached indirectly (its owning tile reused onto another buffer, or occupied via a view) is still caught. For a tuple-returning op, the constraint is propagated from the tuple temporary to every tile-valued `TupleGetItem` result.

  Ops that currently declare a no-alias constraint:

  | Op(s) | Constraint | Why the output cannot alias the input |
  | ----- | ---------- | ------------------------------------- |
  | `tile.recip`, `tile.rsqrt` | `not_inplace_safe` | high-precision path reads the input **and** a tmp scratch while writing the output |
  | `tile.row_sum` / `row_max` / `row_min` | `not_inplace_safe` | `TROW*` reads the full input row + tmp scratch while writing the reduced `[M, 1]` output |
  | `tile.mrgsort_format1` | `not_inplace_safe` | merge-sort intrinsic requires `src != dst` |
  | `tile.move` | `not_inplace_safe` | `TMOV` requires distinct source and destination addresses; baked-address PTO codegen rejects a remaining same-address move rather than silently treating it as a no-op |
  | `tile.rem`, `tile.rems`, `tile.fmod`, `tile.fmods` | `not_inplace_safe` | The remainder kernels write intermediate quotient/remainder state into `dst` while still reading the original dividend; aliasing `dst` with that source corrupts the final correction/subtraction. In addition, A2/A3 `tile.rem` / `tile.rems` codegen rejects a scratch `tmp` MemRef that overlaps any live tile source (the dividend and, for `tile.rem`, the divisor), including overlapping subviews; A5 does not consume that scratch operand |
  | `tile.transpose` | `not_inplace_safe` | `pto.ttrans` is not in-place safe: the a2a3 unaligned scalar path writes `dst` directly from `src` (no tmp staging), so `dst == src` corrupts the data mid-write. The output always gets a fresh buffer (also enforced in InitMemRef, which never inherits the input's buffer for it). |
  | `tile.sel` | `forbid_output_alias(0)` (mask), `(3)` (tmp) | `TSEL` reads the predicate mask + tmp scratch while writing `dst` |
  | `tile.sels` | target-aware | `TSELS` keeps `dst` disjoint from the predicate mask and may reuse `src` or `tmp`; A2/A3 writes the scalar into `tmp` and loads it with `set_cmpmask` before writing `dst`, so `tmp` may alias `dst` but must remain disjoint from mask/src; A5 retains an unread `tmp` ABI operand that may alias any operand |
  | `tile.prelu` | target-aware | A2/A3 is `not_inplace_safe` because `TPRELU` reads `src`, `slope`, and `tmp` while writing `dst`; A5 retains the ABI-required `tmp` operand but does not read it, so `dst` may reuse `tmp` but not the active `src`/`slope` inputs |
  | `tile.{row,col}_expand{,_mul,_add,_sub,_div}` | `forbid_output_alias(1)` (broadcast vector) | the row/col vector (arg 1) is re-read for **every** output row/col, so an output aliasing it is overwritten after the first row/col |
  | `tile.cast` (widening on all targets; narrowing on A2/A3) | output and input byte ranges must be disjoint | writes can clobber unread source elements (see above) |

- **Pipeline-stage guard** (capacity-gated, replication path only): `pl.pipeline(stage=F)` replicates a loop body `F` times for ping-pong, and `LowerPipelineLoops` tags each clone's tile-producing `Call` with a `pipeline_membership` `(group, stage)` (see [30-lower_pipeline_loops.md](32-lower_pipeline_loops.md)). This whole guard is about that replication path. Under `memory_planner=PTOAS` an eligible loop is instead claimed earlier by [`LowerPipelineToSlots`](31-lower_pipeline_to_slots.md), which gives each load a `pl.MemRef(..., slots=F)` indexed `iv % F` and leaves one un-replicated body — no clones, no `pipeline_membership`, and MemoryReuse is skipped for that pipeline anyway. Everything that follows describes only the loops that pass declines. The `F` clones run concurrently under the scheduler, so their program-order-disjoint lifetimes are *not* a safe reuse signal — collapsing concurrent clones onto one buffer injects a false write-after-read that serializes the stages (the #1475 cube-matmul-operand collapse). MemoryReuse therefore keeps concurrent clones in **distinct buffers in every memory space** — including the L0 matmul spaces (Left/Right/Acc/Bias), and regardless of whether a tile is a load or a `tile.move` result — up to the **max-affordable double-buffering depth** `F_g = min(depth_g, ⌊C_s / slot_g⌋)`: a clone at stage `k` lands in residue `ordinal(k) mod F_g` (the **dense** stage ordinal, so sparse stage IDs like `{0, 2}` can't collide via raw `2 mod 2 == 0 mod 2`), so concurrent clones never share (full ping-pong when it fits, maximal spread when the space is tight). Whether the separation fits is decided by the **exact per-space allocator footprint** (`SpaceFootprint`, shared with `AllocateMemoryAddr` — parity by construction), not an estimate. When a space still overflows at every group's affordable depth, a **graceful cross-group shed** lowers one group's depth by a residue and re-packs (choosing the group by the `MaxRelief` heuristic — free the most bytes first, ties by lowest group id); if the shed exhausts, the pass re-packs from scratch in a **legacy fallback** (`force_legacy`), so it never overflows where the legacy packing would have fit. A space whose capacity is unknown (no backend configured) uses the legacy predicate, so the capacity-gated path is never worse than legacy. When the gate reduces a group below its requested `stage=` depth (or a space hits the legacy fallback), MemoryReuse emits a diagnostic through the unified channel — a `PH-MR-001` **perf hint** (or a **warning** for the fallback) naming the requested vs achieved depth and the fix (shrink the per-stage tile to `≤ C_s / stage`, or lower `stage=` to what fits) — so a capacity-forced serialization is never silent. The fallback warning is suppressed when one physical buffer alone exceeds the entire memory space: that is an intrinsically impossible allocation, not a reuse/depth degradation, and an operation-specific check or `AllocateMemoryAddr` reports it as a hard error. Once the reuse decision is made, MemoryReuse strips the now-consumed `pipeline_membership` attr so it does not ride downstream into later passes or codegen.
- `DSA_RP` skips `MemoryReuse`; it represents requested pipeline depth as hard
  separations and performs its pipeline-only fallback in `AllocateMemoryAddr`.

**No shape / dtype / TileView compatibility gate**: tiles that share a physical MemRef may carry **different** shapes, dtypes, or `TileView` attributes. PTO codegen binds a per-variable `alloc_tile` to each tile, so each alias declares the shared base with its own static shape / dtype / layout / `valid_shape`. This permits, for example:

- cross-dtype reuse — a BF16 tile reusing a dead FP32 tile's buffer (e.g. across a narrowing `tile.cast` on A5, or after the source lifetime has ended);
- `tile.fillpad` output reusing its input, and two fillpad outputs with different `pad` sharing one buffer;
- N-D tiles with divergent `valid_shape` sharing a buffer (each keeps its own `valid_shape` on its own `alloc_tile`);
- L0 cube-input `Left` / `Right` sub-tiles of differing shape sharing one slot (e.g. fused-attention QK `Right` `[k, SEQ]` reused by PV `Right` `[k', HEAD]`, halving peak L0B — issue #1595).

  Earlier revisions gated reuse on an `AreTileTypesCompatible` shape / dtype / view match (with a narrow L0 byte-reuse exception); that gate has been removed. Correctness for read-while-write ops is now handled precisely by the no-alias guard above rather than by a coarse whole-tile match.

**Alloc cleanup**:

After MemRef sharing, some MemRef objects become unreferenced (their variables now point to a different shared MemRef). The pass traverses the surrounding `SeqStmts` and removes any `tile.alloc` `AssignStmt` whose LHS MemRef pointer is not in the set of still-used MemRefs.

## Declared allocations

Reuse is opportunistic: any two tiles whose lifetimes do not overlap are candidates
for one buffer. That is the right default for capacity, but it is not free — two
tiles sharing a buffer are ordered by a WAR dependency the source never asked for,
and the hardware must serialize work the scheduler could otherwise overlap.

Referencing a declared `pl.MemRef("name")` in a tile annotation lets the author take an
allocation out of the packer's hands. InitMemRef materializes it as a `tile.alloc(..., pinned=True)`
(see [InitMemRef](35-init_memref.md#declared-allocations)), and this pass then treats it as
closed: a pinned interval opens its own slot in the first-fit pack and that slot is
skipped when placing every later candidate. (Isolation is a per-slot flag inside the
packing loop rather than another `can_share` gate — `can_share` is the innermost step
of the quadratic root-pair packing loop that re-runs per shed step, so the check is resolved once per interval
instead of once per pair.) Concretely:

- Tiles the author bound to **different** buffers are never coalesced, however
  disjoint their lifetimes — the point is to keep them independent.
- Tiles the author bound to the **same** buffer already share one base from InitMemRef
  and stay that way.
- Unbound tiles are packed as before, and never pulled into a declared allocation.

The cost is the author's to manage: pinning trades capacity for parallelism, and an
over-pinned kernel surfaces as a hard `AllocateMemoryAddr` overflow rather than being
silently coalesced back.

**Overlap check.** Two tiles independently bound to the **same slot** must not be live at
the same time — that is not reuse, it is the later write destroying data the earlier tile
still needs. The check is per slot, not per allocation: two tiles on *different* slots of a
`pl.MemRef(slots=N)` declaration are meant to be live together (that is the ping-pong), and
only tiles landing on one slot can corrupt each other. A runtime slot index (`l0c[i % 2]`)
has no static slot to attribute a tile to, so the check is skipped there — the rotation is
the author's to get right — while isolation from every other allocation still holds. This pass owns the check because it is where lifetimes are computed
(`AnalyzeAllocationLifetimes`). Declared-slot validation uses conservative intervals;
as in straight-line packing, *touching* is allowed (one tile's last read may be the
statement that produces the next member).
Tiles that land on the allocation by inheritance rather than by binding — views, in-place
results, bare SSA aliases — are excluded: they are the same data as their source, so
overlapping with it is expected.

Because the isolation guarantee lives here, and ptoas replaces this pass wholesale,
A declared allocation under `memory_planner=PTOAS` is **rejected** at InitMemRef rather than
silently honored-but-unenforced: allocating them without isolating them would
hand back exactly the coalescing the author wrote the binding to prevent.
`memory_planner=DSA_RP` also skips this pass, but preserves the contract in its
allocation problem: every declared allocation is hard-separated from every
other allocation in its memory space, and independently bound co-live members
of one declaration are rejected before solving.

## Ascend910 load + tpop_from_aic hazard

On Ascend910 AIV functions with a non-`None` `SplitMode`, a writer that consumes **both** a `tile.load` result (or a legal-view descendant of one) **and** a `tile.tpop_from_aic` value must not place its output in the same physical buffer as that load result. Allowing the writer's output to in-place-reuse the load buffer produces silently wrong results on this hardware.

MemoryReuse owns every buffer-coalescing decision, so it prevents the hazardous sharing from ever forming rather than relying on a later split. When the guard is active, the reuse decision is blocked exactly when:

- the writer's defining op consumes a `tile.tpop_from_aic` value, **and**
- the buffer member it would reuse in place (whose last use is the writer's def statement) is load-derived.

Both classifications are keyed on `Var` identity. An operand takes one extra step: a value can reach the writer through a **loop carry**, and an `IterArg` is never itself an `AssignStmt` def, so `Var` identity can never classify it. Since [`MaterializeSemanticAliases`](36-materialize_semantic_aliases.md) has already fused each carry chain — init value, `IterArg`, yield value — onto one MemRef base, an `IterArg` operand is classified by the taint of that base instead. Reading operands with `AsVarLike` (never `As<Var>`, which does not match `IterArg`'s own `ObjectKind`) is what makes the carry visible in the first place; without it, `down_next = tile.add(down_prev, pipe_carry)` with a carried `tpop` value silently loses the taint and the hazardous in-place reuse is formed.

A carry also breaks program order, so the collector walks the body **twice**. The producer that taints a carry's buffer may stand *after* the use it taints — `w = tile.add(l, carry); p = tile.tpop_from_aic(); yield p` reads a tpop value in `w` from iteration 1 onwards, but a single forward walk classifies `w` before it has seen `p`. A third traversal could add nothing: the base sets are complete after the first walk (a `tile.tpop_from_aic` def is order-independent, and a view shares its source's base), so only the `Var` sets grow in the second, and those propagate in program order within it. Both walks are O(N).

The guard is gated by `BackendHandler::RequiresSplitLoadTpopWorkaround()` (true only for Ascend910) and the function being split-AIV; on every other backend / function kind the inputs are empty and reuse behaviour is unchanged. The writer is still free to reuse any **non**-load buffer — only the load + tpop in-place combination is rejected. (This guard previously lived in a dedicated `LegalizePTOBufferReuse` pass that split the buffer after the fact; it now folds into MemoryReuse.)

## Written-workspace hazard

`set_workspace_arg(i)` declares an operand to be *compiler-supplied scratch: the hardware writes it, but it carries no result the caller reads*; a writing `set_arg_effect(i, ...)` says the write happens. `tile.gather`, `tile.quant_mx` and `tile.sels` all declare one.

Such a buffer is written by the intrinsic **itself**, on whatever pipe its implementation chooses. A2/A3's TSELS opens with `*scalarPtr = scalar` — a one-element store into `tmp` at offset 0 issued on the **scalar** pipe — while every tile operand is read on the vector pipe. pto-isa orders only the op's own RAW (`PtoSetWaitFlag<PIPE_S, PIPE_V>` before `set_cmpmask`); nothing retires an earlier *vector* reader of that buffer first. The write never reaches the generated `.pto` as a write either — the emitter passes operands positionally — so no sync insertion downstream can see the WAR.

MemoryReuse therefore has to prevent the placement:

- a var passed to a written workspace operand must not join a buffer whose other member's lifetime **ends at or before** the workspace's def point.

In straight-line code the guard is **directional**, which keeps the legal opposite case: a later value — typically the TSELS `dst` — may still join the workspace's own buffer, because by then it has retired. `forbid_output_alias(0)` continues to cover `dst` vs `mask` independently.

Inside a loop that direction collapses. The back edge re-writes the workspace from the scalar pipe on iteration *i+1* while iteration *i*'s vector reads of anything sharing that buffer — the `dst`, and whatever consumes it — can still be in flight, so lifetime order within one iteration proves nothing about the next. A workspace enclosed in a `ForStmt` or `WhileStmt` therefore takes an **exclusive** buffer, blocked in both directions. Those two are the only statements that repeat a body by the time MemoryReuse runs: `pl.pipeline` is lowered to a loop by [`LowerPipelineToSlots`](31-lower_pipeline_to_slots.md) / [`LowerPipelineLoops`](32-lower_pipeline_loops.md).

It is keyed on the registry rather than on operator names or a backend flag, and is **not** backend-gated: PyPTO cannot know which pipe a given intrinsic writes a workspace from, so every declared one is isolated. A5's TSELS happens to leave `tmp` untouched (it dups the scalar into a vector register via `vdup`), and paying a small buffer there is the price of not encoding per-intrinsic pipe knowledge in the pass. The same fact is consumed by DSA-RP as a `TargetHazard` separation, so both planners agree.

Without the guard, `pl.tile.select`-built clamps corrupt exactly element 0 of the compare source: the composite's synthesized workspace is born and dies inside one statement, which makes it the prime reuse candidate for the buffer the preceding `tile.cmps` is still reading.

## Example

### MemRef Sharing with Alloc Cleanup

**Before** (after InitMemRef):

```python
# SeqStmts [
mem_vec_0: MemRefType = tile.alloc(Vec, -1, 16384, 0)
mem_vec_1: MemRefType = tile.alloc(Vec, -1, 16384, 1)
mem_vec_2: MemRefType = tile.alloc(Vec, -1, 16384, 2)
tile_a: Tile[[64, 64], FP32, memref=mem_vec_0] = tile.load(...)
tile_b: Tile[[64, 64], FP32, memref=mem_vec_1] = tile.add(tile_a, ...)
# tile_a last use ↑
tile_c: Tile[[64, 64], FP32, memref=mem_vec_2] = tile.load(...)
# ]
```

**After** (tile_c reuses mem_vec_0 from tile_a, alloc for mem_vec_2 removed):

```python
# SeqStmts [
mem_vec_0: MemRefType = tile.alloc(Vec, -1, 16384, 0)
mem_vec_1: MemRefType = tile.alloc(Vec, -1, 16384, 1)
# mem_vec_2 alloc removed — no longer referenced
tile_a: Tile[[64, 64], FP32, memref=mem_vec_0] = tile.load(...)
tile_b: Tile[[64, 64], FP32, memref=mem_vec_1] = tile.add(tile_a, ...)
tile_c: Tile[[64, 64], FP32, memref=mem_vec_0] = tile.load(...)
# tile_c now shares mem_vec_0 with tile_a
# ]
```

### Producer-Consumer Reuse

When a variable's last use is at the same statement that defines a new variable (producer-consumer relationship), the new variable can reuse the old variable's memory because inputs are read before outputs are written:

```python
# Before:
tile_a: Tile[[64, 64], FP32, memref=mem_vec_0] = tile.create(...)
tile_b: Tile[[64, 64], FP32, memref=mem_vec_1] = tile.muls(tile_a, 0.0)
# tile_a.last_use == tile_b.def → reuse allowed

# After:
tile_a: Tile[[64, 64], FP32, memref=mem_vec_0] = tile.create(...)
tile_b: Tile[[64, 64], FP32, memref=mem_vec_0] = tile.muls(tile_a, 0.0)
# tile_b reuses mem_vec_0
```

### Overlapping Lifetimes (No Reuse)

In straight-line code, when a variable is still alive **after** another variable's definition (last_use > def), their lifetimes overlap and they cannot share memory:

```python
# SeqStmts [
mem_vec_0: MemRefType = tile.alloc(Vec, -1, 16384, 0)
mem_vec_1: MemRefType = tile.alloc(Vec, -1, 16384, 1)
tile_a: Tile[[64, 64], FP32, memref=mem_vec_0] = tile.load(...)
tile_b: Tile[[64, 64], FP32, memref=mem_vec_1] = tile.load(...)
# tile_a.last_use > tile_b.def → tile_a still live when tile_b is defined
# ]
```

## Implementation

**Header**: `include/pypto/ir/transforms/passes.h`

```cpp
Pass MemoryReuse();
```

**Implementation**: `src/ir/transforms/memory_reuse_pass.cpp`

- `LifetimeAnalyzer` records logical definitions, incoming-edge uses and loop recurrence, including `IfStmt` results
- `AnalyzeAllocationLifetimes` builds MemRef sharing groups and conservative allocation intervals in both sibling orders
- `IdentifyReuseOpportunities` finds reuse candidates, retries with aligned subrange placement on a known-capacity overflow, and records each member's placement offset
- `ApplyMemRefSharing` updates MemRef pointers via `MemRefSharingMutator`
- `NormalizeIdentityCopyBuffersMutator` maintains bare SSA, registered writeback-result, and first-input buffer aliases after retargeting; `ReanchorAliasingOutput` follows current input windows through downstream call chains (see Algorithm step 5)
- `UsedMemRefCollector` gathers still-referenced MemRef pointers after sharing
- `RemoveUnusedAllocStatements` filters out redundant `tile.alloc` statements from `SeqStmts`

**Python binding**: `python/bindings/modules/passes.cpp`

```cpp
passes.def("memory_reuse", &pass::MemoryReuse, "Memory reuse optimization");
```

**Tests**: `tests/ut/ir/transforms/test_memory_reuse.py`

- Tests non-overlapping lifetime reuse with MemRef sharing
- Tests producer-consumer reuse (last_use == def at same statement)
- Tests overlapping lifetime no-reuse
- Tests memory space separation
- Tests byte-size compatibility
- Tests cross-dtype / cross-`TileView` reuse (now permitted: BF16↔FP32, fillpad output↔input, divergent `valid_shape`)
- Tests the no-alias guard (`TestForbidOutputAlias` + `TestInplaceOps`), one case per constraint above:
  - `tile.recip` / `tile.rsqrt` / `tile.row_sum` — output must not alias input (`not_inplace_safe`)
  - `tile.sel` — output must not alias the mask / tmp (`forbid_output_alias`)
  - `tile.sels` — output never aliases mask; both A2/A3 and A5 permit tmp/output alias, while A2/A3 backend validation still rejects tmp overlap with mask/src
  - `tile.prelu` — A2/A3 output must not alias any input; A5 output may alias only the unused `tmp`
  - `tile.col_expand_mul` — output must not alias the broadcast vector
  - `tile.cast` — widening on all targets and narrowing on A2/A3 must keep source/output disjoint; coverage includes views, dynamic valid shapes, loop carries, both allocation planners, and explicit overlapping physical addresses
  - a forbidden operand reached through a VIEW is still honored (physical-buffer resolution)
- Tests semantic alias chains after writeback retargeting, including view offsets, same-allocation window changes, pipeline slots, unresolved if-phis, and normalization idempotence
- Tests redundant alloc statement removal
- Tests control flow lifetime analysis (nested IfStmt in ForStmt, branch variable sharing)
- Tests conditional reset and gather-chain reuse, with independent observers, loop backedges and post-merge consumers preserved
