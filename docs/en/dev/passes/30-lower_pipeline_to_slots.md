# LowerPipelineToSlots Pass

Lowers eligible `pl.pipeline(N, stage=F)` loops to existing slot MemRefs. The default PyPTO path keeps unroll + reorder; `enable_software_pipeline=True` opts into explicit preloading with fixed addresses. The existing PTOAS-planner rotation remains available separately.

## Legacy rotation overview

`pl.pipeline(N, stage=F)` asks for ping-pong buffering. [`LowerPipelineLoops`](31-lower_pipeline_loops.md) delivers it by *replication*: `F` copies of the body, each with fresh def-vars, so each copy's tiles are distinct MemRefs that `MemoryReuse` is forbidden to coalesce. That works, but it costs `F` times the code, a static-or-dynamic remainder dispatch, and the `pipeline_membership` machinery that keeps the copies apart.

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

**The legacy rotation needs no new IR op or additional switch.** The synthesized MemRef is shaped exactly like an author's declaration, so [`InitMemRef`](34-init_memref.md) resolves it through the same path and codegen sees no difference between a rotation the author wrote and one the compiler derived.

Because bounds, step and `iter_args` are untouched, there is no remainder to dispatch — a dynamic trip count needs no special case at all.

**Requires**: SSAForm, SplitIncoreOrch, IncoreTileOps, TileOps2D, TileMemoryInferred, NormalizedStmtStructure.

**Pipeline position**: After [`SkewCrossCorePipeline`](29-skew_cross_core_pipeline.md), immediately before [`LowerPipelineLoops`](31-lower_pipeline_loops.md). Late enough that memory spaces are inferred and the tile structure is final; early enough that `InitMemRef` has not yet handed the tiles compiler-owned MemRefs.

For bounded local nested pipelines, the same planner preserves each child
loop's stage and specializes a shared affine slot stream. See
[nested scope rules and limits](29-skew_cross_core_pipeline.md#nested-local-pipeline-scopes).

### Deterministic nested stream rules

The same scoped producer scheduler handles one-way FIFO loops with or without
child pipelines. Auxiliary GM loads have their declaring stage and S-1 lookahead;
fixed unrolled work introduces no separate hard-coded two-slot policy.

For an eligible nested pipeline, an allocation has one storage coordinate per
enclosing pipeline scope. A root stage A and child stage B therefore reserve A*B
versions; another child stage C reserves A*B*C. Child trip counts do not determine
the number of versions. Root-owned allocations still have A versions.

After InitMemRef, MemoryReuse applies the same operator, alias and shared lifetime proofs at all depths.
A computed store result may overwrite a last-used compatible input. A store
extends that storage's asynchronous lifetime; it does not itself require a
separate output ring. Mandatory views retain the source identity. Results that
cannot safely reuse an input retain ordinary allocations; this pass does not create output rings.

Physical layout stores the root coordinate fastest: for two levels,
`(child_iteration % B) * A + (root_iteration % A)`. Deeper child coordinates use
mixed-radix stage residues. PyPTO reserves the full product region. Runtime or predicated
schedules emit addressed parent buffers and explicit slot subviews, preserving the full physical slot product
without requiring a private PTOAS event-boundary contract.

An unconditional fixed-length child uses the continuous ordinal
`parent * child_trip_count + child_iteration`. Preload `S-1` inputs, then issue
the input `S-1` iterations ahead at each child phase, including across parent
boundaries. Future loads retain their own address and validity predicates and
are guarded by the future parent bound. Lookahead is also bounded by the
minimum distance between uses of a physical slot, so a short child cannot
overwrite a still-live parent context. Allocation depths are never reduced.
Parent computation stays in order.
An enclosing predicate that prevents this proof retains local staging.

InitMemRef materializes storage, MemoryReuse decides physical sharing, and
AllocateMemoryAddr checks final capacity. The scheduling pass has no independent
capacity budget and does not reduce the declared slot product.

## The two passes are complementary, not alternatives

Both run, in this order. This pass takes the loops it can prove safe and demotes them; every loop it declines keeps `ForKind::Pipeline` and is replicated by `LowerPipelineLoops` exactly as before. Nothing loses its ping-pong because this pass exists — matmul L0 stage loops, nested pipelines and unusual loop shapes all keep the replication path.

This mirrors [`SkewCrossCorePipeline`](29-skew_cross_core_pipeline.md), which handles cross-core pipeline loops the same way and leaves the rest intact.

## Opt-in software pipeline with the PyPTO planner

Enable the new schedule per compilation, leaving the DSL unchanged:

```python
import pypto.language as pl
from pypto.runtime import RunConfig

@pl.jit
def vec_add(x: pl.Tensor[[64, 1024], pl.FP32],
            y: pl.Out[pl.Tensor[[64, 1024], pl.FP32]]):
    with pl.at(level=pl.Level.CORE_GROUP):
        for i in pl.pipeline(64, stage=3):
            y[i:i + 1, :] = pl.add(x[i:i + 1, :], 1.0)
    return y

vec_add.compile(config=RunConfig(enable_software_pipeline=True, codegen_only=True))
```

`PassContext([], enable_software_pipeline=True)` and
`ir.compile(..., enable_software_pipeline=True)` select the same option. The
context default is `False`; an omitted compile/RunConfig option inherits the
active context. An explicit compile option alongside an active context is an
error. The effective flag participates in the JIT cache key and survives
profiling and pass-dump contexts. The first implementation requires the PyPTO
memory planner and the existing Tile IR pipeline, not the staged Buffer IR path.

For a single local stream with `S` slots, the prefetch distance is `P = S - 1`:

1. Preload logical iterations `0 .. P-1` into their slots.
2. For `t = 0 .. N-P-1`, load `t+P` into `(t+P) % S`, then compute/store `t`
   using `t % S`.
3. Compute/store the remaining `P` iterations.

The pass uses ordinary statements and loops, existing `MemRef` slot metadata,
and `tile.create` to name previously filled storage. No IR node or operator is
added. The main loop is sequential, so the later unroll and IO-order passes do
not reschedule it. The scalar slot expressions lower locally to `arith.remui`;
general scalar modulo semantics do not change. When a compute result reuses its
input slot, codegen also reuses the dominating tile handle: independently emitted
but equivalent slot remainders can prevent PTOAS from deriving the dynamic event schedule.

PyPTO reserves the complete region and assigns its base address. Static straight-line codegen emits
`pto.alloc_multi_tile addr = ...` plus `pto.multi_tile_get` under PTOAS level3.
Runtime or predicated schedules use `pto.alloc_tile addr = <base>` plus `pto.subview`,
retaining proven static valid shapes and the same rotating storage.
No PTOAS memory planning is involved. Single and nested pipelines use the same
operator contracts and last-use analysis:

- GM loads own `S`-slot input regions.
- An in-place-safe computation may continue in a last-used input slot. Equal-width
  dtype changes (such as INT32 to FP32) and complete one-dimensional reshape
  aliases use `TileBufSignature`'s physical-layout proof. A live alias, a forbidden
  operand, or an incompatible physical layout prevents that reuse. All views of
  a result must fit the same complete slot; a narrower view keeps ordinary storage.
- MemoryReuse decides whether a terminal store source can reuse an input version;
  otherwise it uses ordinary storage. Values with both store and compute consumers
  remain unsupported by scheduling.
- Independent intermediates and reduction scratch keep ordinary allocations.
  A resident full Vec tile can remain outside the loop. Writable arguments must
  be registry-declared workspace backed by an unbound `tile.create`; workspace
  used as loop data is rejected. Legacy reductions use their existing
  `LaneInvariantArg::Scratch` contract. These allocations participate in the capacity check.

For example, an accumulator load can feed cast, ReLU and column multiplication
in the same slot; row-sum output and scratch remain separate; a final multiplication
can consume and overwrite the last-used KV-scale slot. A 128x64 score epilogue
with three slots needs two rings, not three copies of every intermediate.

Mandatory views inherit their source allocation through `InitMemRef`. Codegen
uses static valid extents only when proven, materializes slot views with existing
`pto.treshape`, and shares canonical slot-index SSA values within each straight-line
region. Dynamic or partial valid shapes are never silently replaced by full shapes.

The local path supports full-valid flat 2D Vec tiles, stages 2–4,
nonnegative static starts and positive constant steps. Runtime bounds use
guarded preloads and a normalized loop; overflow-risk paths retain the original
schedule. Bounded nested scopes follow the storage and prefetch rules above.
Unsupported effects, true data recurrences and unproved aliases decline the
whole loop to the established lowering. Static single-level loops shorter than
the preload distance also retain that fallback.

The preceding [joint phase](29-skew_cross_core_pipeline.md#joint-one-way-fifo-pipeline)
handles eligible one-way AIC→AIV loops through standard GM-entry pipe operations.
It retains one outer consumer body with dynamic slot selection. General
feedback and multiple-message schedules are outside this optimization.

**Performance limitation:** stock PTOAS can conservatively serialize dynamic
subviews even when their slots are disjoint. This does not remove required
synchronization, but can prevent MTE2/vector overlap; see
[PTOAS #1587](https://github.com/hw-native-sys/PTOAS/issues/1587).
Enabling the transformation therefore does not promise a speedup.

## Existing PTOAS-planner rotation

With the new option disabled, `memory_planner=PTOAS` retains the original
single-body `iv % F` rotation described above and below. With the default
PyPTO planner and the option disabled, this pass leaves every loop untouched.

## API

| C++ | Python | Level |
| --- | ------ | ----- |
| `pass::LowerPipelineToSlots()` | `passes.lower_pipeline_to_slots()` | Function-level |

```python
from pypto import passes
with passes.PassContext([], memory_planner=passes.MemoryPlanner.PTOAS):
    result = passes.lower_pipeline_to_slots()(program)
```

## Legacy rotation behavior

For a `ForStmt(kind == ForKind::Pipeline, attrs["pipeline_stages"] == F)` with `F > 1` that passes every gate below:

1. Each candidate tile's `TileType` is rebound onto a fresh pinned `MemRef(name, slots=F)` with `slot_index = iv % F`. The definition and all its uses are rewritten in one walk — the IR is in SSA form, so no use precedes its definition.
2. `kind_` becomes `ForKind::Sequential` and `pipeline_stages` is stripped. The two always travel together, so the `PipelineLoopValid` invariant (`kind == Pipeline` iff `pipeline_stages` present) holds at every observable state.
3. Nothing else changes: no statement is added, removed or reordered, and `start` / `stop` / `step` / `iter_args` are left alone.

`F == 1` is a user-written `pl.pipeline(stage=1)` or the marker a previous `LowerPipelineLoops` run left behind. Nothing needs multi-buffering, so the (kind, attr) pair stays whole for `CanonicalizeIOOrder` to scope on.

### Which tiles take a slot

Every top-level `tile.load` result the author has not already bound.

- **Loads only.** A load buffer is what must stay private so iteration `i+1`'s prefetch overlaps iteration `i`'s compute; compute intermediates may coalesce. This is the same distinction [`MemoryReuse`](36-memory_reuse.md) already draws with `pipeline_load_tiles`. Giving *every* tile `F` private copies overflows the on-chip budget on real kernels — a `stage=4` RMSNorm would need `4 x 67 KB > 188 KB` UB. `tile.read` is **not** included: it returns a scalar element, not a tile, so there is no buffer to rotate.
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

**Why a declined enclosing loop disqualifies everything below it.** That loop will be replicated, and its `F` clones would each select one slot of the same allocation inside one loop body — a shape PTO codegen rejects under this planner (see [one slot per iteration](../codegen/00-pto_codegen.md#multi-slot-declarations-become-one-ptoas-region-ptoas-mode)).

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

The legacy loop strides by its original step with a single body, and its region carries no `addr`: PTOAS places it. Retaining the region and slot identity enables per-slot synchronization and overlap.

## Related

- [`LowerPipelineLoops`](31-lower_pipeline_loops.md) — the replication path, which still handles every loop this pass declines
- [`SkewCrossCorePipeline`](29-skew_cross_core_pipeline.md) — same structure for cross-core pipeline loops
- [`InitMemRef`](34-init_memref.md) — resolves the synthesized declaration
- [PTO codegen](../codegen/00-pto_codegen.md) — lowers the slots to a ptoas region
- [Python syntax: Slots](../language/00-python_syntax.md#slots) — the hand-written form of the same declaration

### Runtime-bound schedules

Eligible runtime-bound loops have guarded preloads and one normalized body.
Internal PyPTO scheduling metadata is not emitted as an assembler contract.
Codegen uses addressed parent buffers and explicit slot subviews for runtime or predicated loops, preserving
slot counts, fixed allocation bases and safe static valid shapes. PTOAS inserts
synchronization for these standard operations; no private capability flag is
required. A5 retains the established fallback. Measure achieved overlap separately
from correctness because conservative synchronization can serialize transfers.
