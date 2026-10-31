# LowerPipelineToSlots Pass

Lowers eligible `pl.pipeline(N, stage=F)` loops to existing slot MemRefs. The default PyPTO path keeps unroll + reorder; `enable_software_pipeline=True` opts into explicit preloading with fixed addresses. The existing PTOAS-planner rotation remains available separately.

## Legacy rotation overview

`pl.pipeline(N, stage=F)` asks for ping-pong buffering. [`LowerPipelineLoops`](32-lower_pipeline_loops.md) delivers it by *replication*: `F` copies of the body, each with fresh def-vars, so each copy's tiles are distinct MemRefs that `MemoryReuse` is forbidden to coalesce. That works, but it costs `F` times the code, a static-or-dynamic remainder dispatch, and the `pipeline_membership` machinery that keeps the copies apart.

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

**The legacy rotation needs no new IR op or additional switch.** The synthesized MemRef is shaped exactly like an author's declaration, so [`InitMemRef`](35-init_memref.md) resolves it through the same path and codegen sees no difference between a rotation the author wrote and one the compiler derived.

Because bounds, step and `iter_args` are untouched, there is no remainder to dispatch — a dynamic trip count needs no special case at all.

**Requires**: SSAForm, SplitIncoreOrch, IncoreTileOps, TileOps2D, TileMemoryInferred, NormalizedStmtStructure.

**Pipeline position**: After [`SkewCrossCorePipeline`](30-skew_cross_core_pipeline.md), immediately before [`LowerPipelineLoops`](32-lower_pipeline_loops.md). Late enough that memory spaces are inferred and the tile structure is final; early enough that `InitMemRef` has not yet handed the tiles compiler-owned MemRefs.

## The two passes are complementary, not alternatives

Both run, in this order. This pass takes the loops it can prove safe and demotes them; every loop it declines keeps `ForKind::Pipeline` and is replicated by `LowerPipelineLoops` exactly as before. Nothing loses its ping-pong because this pass exists — matmul L0 stage loops, nested pipelines and unusual loop shapes all keep the replication path.

This mirrors [`SkewCrossCorePipeline`](30-skew_cross_core_pipeline.md), which handles cross-core pipeline loops the same way and leaves the rest intact.

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

`PassContext([], enable_buffer_ir=False, enable_software_pipeline=True)` and
`ir.compile(..., enable_software_pipeline=True)` select the same option. The
context default is `False`; an omitted compile/RunConfig option inherits the
active context. An explicit compile option alongside an active context is an
error. The effective flag participates in the JIT cache key and survives
profiling and pass-dump contexts. The first implementation requires the PyPTO
memory planner and the Tile IR pipeline. An explicit context must disable Buffer
IR; without an active context, the compile/RunConfig opt-in selects Tile IR
automatically. Other compilations retain default-on Buffer IR selection.

The pass records transformed slot allocations with the existing
`software_pipeline_slots` call attribute. Memory reuse preserves this marker,
and codegen reads it from the lowered IR rather than requiring a second option.
This also applies to short pipelines containing only preload and drain phases.

The PTOAS contracts in `tests/ut/codegen/test_software_pipeline_codegen.py` use
the existing CPU compilation-regression step:
`pytest tests/ut/codegen/ --require-ptoas -m ptoas_compile`.
This step uses the pinned assembler without requiring a device runtime, CANN or
an NPU. Ordinary unit runs can skip these contracts when PTOAS is unavailable;
`--require-ptoas` makes a missing or incompatible assembler fail CI.

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

PyPTO reserves the complete region and assigns its base address. At PTOAS level3,
schedules without co-live loop slots can emit `pto.alloc_multi_tile addr = ...`
plus `pto.multi_tile_get`. Co-live prefetch slots, including static loops, and
runtime or predicated schedules use `pto.alloc_tile addr = <base>` plus
`pto.subview`, retaining proven static valid shapes and the same rotating storage.
This avoids the PTOAS multi-tile prefetch synchronization regression (#1519).
No PTOAS memory planning is involved. The local pipeline uses existing
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

Additional isolation of legacy reduction scratch is scoped to loops carrying
`software_pipeline_slots` after transformation. Enabling the option alone
does not change scratch reuse in a declined or otherwise untransformed loop.
The isolation applies to scratch used in the marked steady-state loop. Ordinary
loops and preload or drain statements outside that loop use the existing
allocation constraints.

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
schedule. Nested pipelines, cross-core FIFO operations, unsupported effects,
true data recurrences and unproved aliases decline the
whole loop to the established lowering. Static single-level loops shorter than
the preload distance also retain that fallback.

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

- **Loads only.** A load buffer is what must stay private so iteration `i+1`'s prefetch overlaps iteration `i`'s compute; compute intermediates may coalesce. This is the same distinction [`MemoryReuse`](37-memory_reuse.md) already draws with `pipeline_load_tiles`. Giving *every* tile `F` private copies overflows the on-chip budget on real kernels — a `stage=4` RMSNorm would need `4 x 67 KB > 188 KB` UB. `tile.read` is **not** included: it returns a scalar element, not a tile, so there is no buffer to rotate.
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
| Constant `step > 0`, constant `start >= 0`; for `step > 1`, a constant `stop <= INT64_MAX - (step - 1)` | See below |
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

**Why the slot index is `iv % F`, `(iv - start) % F` or `((iv - start) / step) % F`.**
ptoas' slot accounting reads the value `pto.multi_tile_get` carries and reasons about
`(symbol + offset) mod N`; what it needs is for two accesses to share the *identity* of that
symbol, not for the expression to have a particular shape. So all three qualify.

All three are `((iv - start) / step) % F` with the parts that are identities dropped. Which one is
emitted is decided by the loop shape. Eligibility is not: any constant `step > 0` and constant
`start >= 0` qualifies.

- `step == 1`, `start % F == 0`: the induction variable **is** the rotation counter, so the slot is
  literally `iv % F`. This is the form these loops have always had.
- `step == 1`, `start % F != 0`: the counter starts at `start`, so the slot is `(iv - start) % F`.
  ptoas sees `(symbol + constant) mod N`, which its slot matcher peels.
  `pl.pipeline(1, 5, 1, stage=2)` selects `0, 1, 0, 1`.
- `step > 1`: `iv` takes `start + j*step`, so the slot is `((iv - start) / step) % F`. The division is
  exact by construction, so this is still a remainder over a single SSA value, never a folded byte offset.

A runtime `step` or `start` is still declined: the slot would then depend on a value the analysis
cannot bound. A stepped loop (`step > 1`) is also declined unless `stop` is a constant no larger
than `INT64_MAX - (step - 1)`: the retained `scf.for` computes `iv + step` after its last
iteration, and that update can overshoot `stop` by up to `step - 1`, overflowing the signed index.
`pl.pipeline(INT64_MAX - 4, INT64_MAX, 3, stage=3)` is the shape this closes. `step == 1` cannot
overshoot, so it keeps a runtime `stop`. Note that the counter form loses the `kDisjoint` inference for an
`(iv + c) % N`-style pair, because the constant peel sits above the division and the two sides
end up with different symbols; that degrades to `kUnknown`, which the callers treat
conservatively (the dependency is kept), never to a wrong answer.

A negative `start` is declined in every form: `iv % F` is a signed remainder, so a negative `iv`
selects a negative slot, and `iv - start` is not representable for a sufficiently negative `start`.

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
  %slot = arith.remsi %i, %c2_index : index
  %t = pto.multi_tile_get %pipe_t_mb[%slot] : !pto.multi_tile_buf<..., count=2> -> !pto.tile_buf<...>
  pto.tload ins(...) outs(%t : ...)
  ...
}
```

The operand remains a logical index, without byte scaling; ptoas can still
derive dynamic slot selection and per-slot event ids.

Signed integer and `INDEX` `FloorMod` preserve Python's `%` semantics: a nonzero
remainder has the divisor's sign, which `arith.remsi` alone does not give — it
follows the *dividend's* sign instead. Codegen therefore corrects the remainder,
so that `-97 % 24` produces `23`:

```mlir
%rem = arith.remsi %i, %c2_index : index
%negative = arith.cmpi slt, %rem, %c0_index : index
%corrected = arith.addi %rem, %c2_index : index
%slot = arith.select %negative, %corrected, %rem : index
```

A positive constant divisor needs only the negative-remainder check above; a
negative constant uses a positive-remainder check. For a dynamic divisor,
codegen checks that the remainder is nonzero and its sign differs from the
divisor before adding the divisor. Division by zero and signed minimum modulo
`-1` remain outside the operation's valid inputs.

**The correction is skipped when it cannot change the result**, which is why the
example above shows a bare `remsi`. With a non-negative dividend and a positive
constant divisor, `arith.remsi` already lands in `[0, divisor)`, the two agree,
and the correction would be dead code. Codegen proves the dividend non-negative
syntactically, from the shapes an index expression is built out of: a literal; an
induction variable whose loop starts at a non-negative literal (a device loop is
always ascending, so the variable never goes below its start); that variable
minus its own start; and floor divisions and floor remainders of those by a
positive constant literal. Sums and products are deliberately excluded: being
built from non-negative terms does not make a result non-negative, because the
emitted arithmetic wraps — `i * 4611686018427387904` is negative at `i == 2`.

Every form in that proof must also *execute* at the dtype the proof is made at.
`EmitArithOperand` coerces each operand into its operation's dtype, so an
induction variable reaching a narrower operation is a different value: over an
`INDEX` loop from `2147483648`, the `i` an `INT32` operation sees is
`-2147483648`. That applies to nested operations too, not just to the variable —
an `INT32` `FloorDiv` inside an `INDEX` `FloorMod` yields a negative quotient
that is then widened back, so proving the subtree at the outer width would skip a
correction the result needs. The check is conservative — where it cannot tell,
the correction is emitted.

This matters beyond the three instructions saved. A multi-buffer slot operand has
to stay in the `(symbol + constant) mod N` shape that ptoas' slot matcher peels,
and a `rem`/`cmp`/`add`/`select` chain is not that shape. A rotation derived from
a stepped or unaligned loop — `((iv - start) / step) % count` — is both the case
that needs the bare remainder and the case whose dividend is non-negative by
construction.

Signed integer and `INDEX` `FloorDiv` use `arith.floordivsi` for the matching
floor quotient, preserving `a == (a // b) * b + a % b`, and fall back to
`arith.divsi` under the same proof — ptoas expands `floordivsi` into a truncating
division plus the same conditional correction. Floating-point and unsigned
lowering are unchanged. Floor-division and floor-modulo constant folding leave
out-of-range typed literals for runtime evaluation so they cannot replace their
emitted bit patterns with calculations on the raw stored integers.

The loop strides by its original step with a single body, and the region carries no `addr` — ptoas `PlanMemory` places it, which is what lets it overlap iteration *i*'s load with iteration *i-1*'s compute.

## Related

- [`LowerPipelineLoops`](32-lower_pipeline_loops.md) — the replication path, which still handles every loop this pass declines
- [`SkewCrossCorePipeline`](30-skew_cross_core_pipeline.md) — same structure for cross-core pipeline loops
- [`InitMemRef`](35-init_memref.md) — resolves the synthesized declaration
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
