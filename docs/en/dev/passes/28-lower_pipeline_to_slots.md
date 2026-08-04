# LowerPipelineToSlots Pass

Multi-buffers a `pl.pipeline(N, stage=F)` loop by rotating one body through the `F` slots of a single allocation, instead of replicating the body `F` times.

## Overview

`pl.pipeline(N, stage=F)` asks for ping-pong buffering. [`LowerPipelineLoops`](28-lower_pipeline_loops.md) delivers it by *replication*: `F` copies of the body, each with fresh def-vars, so each copy's tiles are distinct MemRefs that `MemoryReuse` is forbidden to coalesce. That works, but it costs `F` times the code, a static-or-dynamic remainder dispatch, and the `pipeline_membership` machinery that keeps the copies apart.

This pass expresses the same intent in the form ptoas already understands. `pl.MemRef(name, slots=F)` says *"one allocation, F uniform slots, this use takes slot k"* — see [Slots](../language/00-python_syntax.md#slots) — and PTO codegen lowers exactly that to `pto.alloc_multi_tile` + `pto.multi_tile_get`. So the loop keeps **one** body and each per-stage buffer becomes slot `iv % F` of a synthesized declaration:

```python
# Before (as this pass sees it)
for i in pl.pipeline(64, stage=2):
    x: pl.Tile[[128], pl.FP32, pl.Mem.Vec] = pl.tile.load(a, [i * 128], [128])
    pl.tile.store(x, [i * 128], out)

# After — one body, bounds and step untouched, kind demoted
for i in pl.range(64):
    x: pl.Tile[[128], pl.FP32, pl.MemRef("pipe_x", slots=2)[i % 2], pl.Mem.Vec] = \
        pl.tile.load(a, [i * 128], [128])
    pl.tile.store(x, [i * 128], out)
```

**No new IR op and no new user-facing switch.** The synthesized MemRef is shaped exactly like an author's declaration, so [`InitMemRef`](31-init_memref.md) resolves it through the same path and codegen sees no difference between a rotation the author wrote and one the compiler derived.

Because bounds, step and `iter_args` are untouched, there is no remainder to dispatch — a dynamic trip count needs no special case at all.

**Requires**: SSAForm, SplitIncoreOrch, IncoreTileOps, TileOps2D, TileMemoryInferred, NormalizedStmtStructure.

**Pipeline position**: After [`SkewCrossCorePipeline`](26-skew_cross_core_pipeline.md), immediately before [`LowerPipelineLoops`](28-lower_pipeline_loops.md). Late enough that memory spaces are inferred and the tile structure is final; early enough that `InitMemRef` has not yet handed the tiles compiler-owned MemRefs.

## The two passes are complementary, not alternatives

Both run, in this order. This pass takes the loops it can prove safe and demotes them; every loop it declines keeps `ForKind::Pipeline` and is replicated by `LowerPipelineLoops` exactly as before. Nothing loses its ping-pong because this pass exists — matmul L0 stage loops, nested pipelines and unusual loop shapes all keep the replication path.

This mirrors [`SkewCrossCorePipeline`](26-skew_cross_core_pipeline.md), which handles cross-core pipeline loops the same way and leaves the rest intact.

## Gated on `memory_planner=PTOAS`

Under the default PyPTO planner the pass returns every function untouched, so that path stays byte-identical.

The gate tracks where a region is *emitted*, not where ptoas can use one. PTO codegen's `PlanMultiBufferRegions` bails under the PyPTO planner, so a rotation synthesized there would resolve to a runtime address on an ordinary `alloc_tile` — correct, but with none of the slot analysis this transform exists to trigger.

**ptoas itself is not the limitation.** Given `pto.alloc_multi_tile addr = <constant base>`, ptoas 0.55 derives the same per-slot dynamic-event synchronization at `--pto-level=level3` as it does at level2 — measured on a prefetch loop, the sync-op sequence is identical after normalizing event ids (two primed events, `wait_flag`/`set_flag` keyed on the slot, two drains). ptoas 0.54 does not, which is what the older
[PTOAS#1106](https://github.com/hw-native-sys/PTOAS/issues/1106) note describes.

Widening the gate therefore needs work on the PyPTO side rather than upstream: the address allocator must reserve `slot_count * slot_size` for the region base and codegen must emit that address on the region. That is follow-up work; until it lands, this pass stays scoped to the planner whose codegen path already exists.

## API

| C++ | Python | Level |
| --- | ------ | ----- |
| `pass::LowerPipelineToSlots()` | `passes.lower_pipeline_to_slots()` | Function-level |

```python
from pypto import passes
with passes.PassContext([], memory_planner=passes.MemoryPlanner.PTOAS):
    result = passes.lower_pipeline_to_slots()(program)
```

## Behavior

For a `ForStmt(kind == ForKind::Pipeline, attrs["pipeline_stages"] == F)` with `F > 1` that passes every gate below:

1. Each candidate tile's `TileType` is rebound onto a fresh pinned `MemRef(name, slots=F)` with `slot_index = iv % F`. The definition and all its uses are rewritten in one walk — the IR is in SSA form, so no use precedes its definition.
2. `kind_` becomes `ForKind::Sequential` and `pipeline_stages` is stripped. The two always travel together, so the `PipelineLoopValid` invariant (`kind == Pipeline` iff `pipeline_stages` present) holds at every observable state.
3. Nothing else changes: no statement is added, removed or reordered, and `start` / `stop` / `step` / `iter_args` are left alone.

`F == 1` is a user-written `pl.pipeline(stage=1)` or the marker a previous `LowerPipelineLoops` run left behind. Nothing needs multi-buffering, so the (kind, attr) pair stays whole for `CanonicalizeIOOrder` to scope on.

### Which tiles take a slot

Every top-level `tile.load` result the author has not already bound.

- **Loads only.** A load buffer is what must stay private so iteration `i+1`'s prefetch overlaps iteration `i`'s compute; compute intermediates may coalesce. This is the same distinction [`MemoryReuse`](33-memory_reuse.md) already draws with `pipeline_load_tiles`. Giving *every* tile `F` private copies overflows the on-chip budget on real kernels — a `stage=4` RMSNorm would need `4 x 67 KB > 188 KB` UB. `tile.read` is **not** included: it returns a scalar element, not a tile, so there is no buffer to rotate.
- **Top-level.** Only direct members of the loop body's `SeqStmts`; a load nested in an inner loop or `if` belongs to that region.
- **No loop-invariance filter.** Every unbound top-level load qualifies, including one whose arguments never mention the induction variable. Invariance cannot be read off the induction variable: a load addressed through a loop-carried `IterArg` reads different data each iteration without naming `iv`, and skipping it would strand it with neither a slot nor a replicated copy once a sibling candidate demotes the loop. Slotting a genuinely invariant load costs nothing over the fallback either — `LowerPipelineLoops` replicates its buffer `F` times all the same.
- A tile the author already bound to a declared allocation stays the author's.

### Eligibility

Codegen **refuses** a region it cannot describe rather than degrading it, because falling back to per-slot `alloc_tile`s would let ptoas plan the slots on top of each other. So every gate here mirrors a `PlanMultiBufferRegions` blocker: synthesizing a doubtful region would turn a kernel that compiles today into a compile failure.

| Gate | Why |
| ---- | --- |
| `F` in `[2, 16]` | ptoas' `multi_tile_buf` slot-count bounds |
| Memory space is Vec, Mat or Acc | The spaces ptoas accepts for a slot |
| Static valid shape | A region declares one static extent for all its slots |
| Not carried into a phi | A tile that is yielded, or used as a nested loop's `init_values`, makes that phi share its MemRef. Both reach the phi the same way — one through a `YieldStmt`, the other through `IterArg::initValue_`. Checked through **alias roots**: `InitMemRef` shares one MemRef across a bare `a = b` tile copy and across a view / in-place result, so yielding an alias carries the original's slot just as yielding it directly would |
| Not consumed by a view / in-place op | Such a result *is* its source's buffer, so it would land on the same allocation with a different `tile_buf` type |
| `step == 1` and `start % F == 0` | See below |
| No enclosing pipeline loop was declined | See below |
| Slots fit the memory space | See below |

**The decline is per loop, not per tile.** The four tile gates above (space, static
valid shape, phi, view / in-place) are checked only for a load that actually wants a
slot — top-level and not already bound by the author. If any such load trips one of them,
the **whole loop** is declined, even when its siblings are eligible. Dropping just that one
load would not work: a surviving sibling still demotes the loop to `Sequential`, and the
blocked load would then reach neither these slots nor `LowerPipelineLoops`' replication,
silently losing the per-stage privacy the `pl.pipeline(stage=F)` annotation asked for. Only
an author-bound tile is skipped without affecting the loop, because declining over it would
push its declaration onto the replication path, which rejects it.

**Why the slot index must be literally `iv % F`.** ptoas matches the *affine form* of the slot index to decide which accesses share a slot, and that match is what earns the rotation its per-slot dynamic event ids — handing it a folded byte offset defeats the analysis. A general `((iv - start) / step) % F` would have to be materialized as an intermediate SSA value, risking the loss of exactly the analysis this transform exists to trigger. Loops whose index cannot be written directly are left to replication.

**Why the slots must fit on chip.** The declared slots are **pinned**: `InitMemRef`
sizes the allocation at `F * slot_size` and ptoas may not reuse any of it, so this pass
is directly accountable for those bytes. A loop with many eligible loads otherwise
multiplies its footprint by `F` and ptoas answers a region it cannot place with a hard
`overflow` error — it does *not* degrade. The replication path does: `MemoryReuse`'s
capacity gate lowers the effective double-buffering depth (`F_g = min(depth_g, ⌊C_s /
slot_g⌋)`) and sheds groups until the space fits, so declining hands the loop to a path
that shrinks rather than fails.

The budget is per memory space, summed over the loop's candidates and accumulated across
the whole function — a slotted inner loop's region is co-live with its slotted ancestor's.
It is seeded with the allocations the **author** already declared: those are `is_pinned_`
too, so ptoas cannot reuse them either, and ignoring them would admit a synthesized region
that fits on its own while the pair overflows.
Capacity comes from `Backend::GetMemSize(space)`; a space with unknown capacity (no backend
configured) is left ungated, mirroring `MemoryReuse`. This bounds only what *this pass*
pins: tiles it does not slot are still planned by ptoas with lifetime reuse, which the pass
cannot model.

**Why a declined enclosing loop disqualifies everything below it.** That loop will be replicated, and its `F` clones would each select one slot of the same allocation inside one loop body. ptoas derives the per-slot WAR guard only for the *first* `multi_tile_get` of an iteration, so codegen rejects that shape ([PTOAS#1118](https://github.com/hw-native-sys/PTOAS/issues/1118)).

## Generated PTO IR

```mlir
%pipe_t_mb = pto.alloc_multi_tile valid_row = %c64_index valid_col = %c64_index
           : !pto.multi_tile_buf<!pto.tile_buf<loc=vec, dtype=f32, rows=64, cols=64, ...>, count=2>
scf.for %i = %c0_index to %c4_index step %c1_index {
  %0 = arith.remsi %i, %c2_index : index
  %t = pto.multi_tile_get %pipe_t_mb[%0] : !pto.multi_tile_buf<..., count=2> -> !pto.tile_buf<...>
  pto.tload ins(...) outs(%t : ...)
  ...
}
```

The loop strides by its original step with a single body, and the region carries no `addr` — ptoas `PlanMemory` places it, which is what lets it overlap iteration *i*'s load with iteration *i-1*'s compute.

## Related

- [`LowerPipelineLoops`](28-lower_pipeline_loops.md) — the replication path, which still handles every loop this pass declines
- [`SkewCrossCorePipeline`](26-skew_cross_core_pipeline.md) — same structure for cross-core pipeline loops
- [`InitMemRef`](31-init_memref.md) — resolves the synthesized declaration
- [PTO codegen](../codegen/00-pto_codegen.md) — lowers the slots to a ptoas region
- [Python syntax: Slots](../language/00-python_syntax.md#slots) — the hand-written form of the same declaration
