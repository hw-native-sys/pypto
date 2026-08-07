# AutoTile

`AutoTile` turns one explicitly marked tensor function into one complete
Ascend 910B vector kernel. The user fixes the operation graph; the pass chooses
the core grid, tile or stream shape, reduction phases, on-chip lifetimes, and
stores.

This is deliberately different from graph fusion. `AutoTile` never partitions
the marked function and never falls back to several kernels. A marked function
either has one exact, capacity-safe schedule or compilation fails. Unmarked
functions are unchanged.

## Placement and API

The default strategy runs `AutoTile` immediately after
[`FlattenCallExpr`](06-flatten_call_expr.md) and before hierarchy or InCore
outlining. At that point calls are in three-address form and the complete tensor
DAG is still visible.

Mark a function with `auto_tile`:

```python
import pypto.language as pl


@pl.program
class Program:
    @pl.function(attrs={"auto_tile": True})
    def kernel(
        self, x: pl.Tensor[[128, 8192], pl.FP32]
    ) -> pl.Tensor[[128, 8192], pl.FP32]:
        shifted: pl.Tensor[[128, 8192], pl.FP32] = pl.add(x, 1.0)
        out: pl.Tensor[[128, 8192], pl.FP32] = pl.mul(shifted, 2.0)
        return out
```

The pass is also available directly for custom pipelines. Direct use requires
the same tensor-level preparation prefix as the default strategy:

```python
from pypto import passes

prepared = passes.convert_to_ssa()(program)
prepared = passes.simplify()(prepared)
prepared = passes.normalize_stmt_structure()(prepared)
prepared = passes.flatten_call_expr()(prepared)
result = passes.auto_tile()(prepared)
```

An absent or false marker is a no-op. A successful rewrite consumes the marker,
so running the pass again is also a no-op.

## Admission contract

The initial implementation supports the following closed surface:

- an explicitly configured Ascend 910B backend and a marked tensor-level
  `FunctionType::Opaque` function before scope outlining;
- a straight-line, topologically ordered SSA tensor DAG with one top-level
  return and positive static rank-2 shapes;
- FP32 and FP16 vector computation;
- BF16 tensor storage and native cast-chain endpoints;
- a terminal, unconsumed FP32-to-INT8 cast;
- elementwise arithmetic, scalar forms, `part_*`, row/column broadcasts,
  `exp`, `log`, `abs`, `sqrt`, `rsqrt`, `recip`, and `fmod`;
- `row_sum`, `row_max`, `col_sum`, and `col_max`;
- one reduction plus an elementwise producer or consumer DAG;
- the canonical five-operation row softmax graph;
- multiple returned pointwise values and capacity-fitting materialized
  reduction live-outs; and
- row and column reductions that fit one non-atomic kernel schedule.

Unified binary operations are normalized to explicit row- or column-expand
operations before emission. Ambiguous `[1,1]` tensor broadcasts and a
broadcasted left operand of non-commutative subtraction or division are rejected.
Broadcast division supports FP16 and FP32; its high-precision form is not part
of this contract.

The validated Ascend 910B A2/A3 AutoTile arithmetic surface is FP16/FP32;
PTOAS rejects direct BF16 `TADD` on this target. AutoTile therefore rejects
BF16 arithmetic during admission, before planning or PTOAS compilation. BF16
remains supported as a stored tensor and as the source or destination of a
native cast chain. AutoTile does not implicitly promote BF16 arithmetic through
FP32 because that would be a different algorithm with additional UB storage,
transfers, and modeled cost.

A native cast chain may consume a boundary or full-frame elementwise value. A
cast rooted directly in a reduction result is declined: reduction emission owns
a separately padded result box that the current emitter cannot yet widen to the
cast chain's common physical granule. Applying or broadcasting the reduction
result back into the full iteration frame before casting remains supported.

The pass rejects dynamic or non-rank-2 shapes, control flow, side-effecting
statements, `full`/shape construction, minimum/product/argument reductions,
matmul and other Cube work, mixed kernels, Welford, unsupported dtypes, and any
graph that would require more than one kernel. These are user-facing admission
errors, not requests to another planner.

## Planning model

The planner first constructs a typed vector graph. Tensor nodes record static
shape, dtype, boundary status, and whether the value must survive to a return.
Operation nodes record their primitive and geometry. The original SSA statement
order is the topological order; the pass does not reorder user operations.

Same-shaped values connected by elementwise operations form a physical-shape
class. For a native cast chain, the planner takes the least common multiple of
the per-dtype DMA element granularities and assigns that one element-count
granule to the complete class. Thus every native `TCVT` hop has an identical
physical shape. This does **not** allocate every value as the widest dtype:
each SSA result is charged and later allocated as
`physical_elements * sizeof(its_own_dtype)`. The emitter carries the original
logical extent as `valid_shape`, so padding never changes the program's
logical bounds.

It then enumerates balanced two-dimensional core grids. Every emitted task uses
one static maximum-size tile. Ragged partitions clamp or overlap their final
tile, so repeated edge work is idempotent and is included in modeled traffic.
Candidate task counts are selected around the hardware's 48-vector-core wave
shape.

For each candidate, the planner builds an explicit `VectorSchedulePlan`. It
contains:

- row and column partitions and the exact work-unit count;
- full-region and streamed strip or chunk extents;
- phase operation lists and boundary-input first/last-use records;
- pipeline trip counts and depths;
- logical and DMA-padded reduction extents;
- full and emitted UB peaks plus modeled compute and transfer cycles.

The emitter validates this descriptor against the source graph before creating
IR. It does not rediscover tiling, lifetime, split, or phase choices. This
one-way plan-to-emission contract is important: a cost is meaningful only when
the emitted algorithm performs the work and owns the memory that the plan
priced.

At `INFO` log level, every successful rewrite prints one `AutoTile[name]` line
containing the selected schedule, grid, work units, tile/strip/chunk extents,
pipeline depths, UB peaks, phase traffic, modeled cycles, and whether the
reduction estimate is grounded or a fallback. The Ascend 910B coefficients are
ported without refitting from the silicon-grounded scheduler model; AutoTile
changes ownership of planning and emission, not those measurements.

## Schedule reports

Every successful `ir.compile()` of an AutoTile function also writes two small,
deterministic artifacts:

```text
<output_dir>/report/auto_tile/<function>.json
<output_dir>/report/auto_tile/<function>.txt
```

The JSON file is a versioned compiler-artifact schema. It records the selected
grid, balanced partitions, representative region, strip/chunk loops, serial
tails, phase operation order, boundary-input lifetimes, logical and physical
tile extents, dtype-specific element sizes, UB peaks, traffic, and modeled
cycles. It is intended for tooling and is not yet a stable public API.

The text file renders the same structured descriptor as tile-centric
pseudocode. It describes one representative SPMD work unit rather than drawing
all cores. For example, an online-softmax report separates the serial first
chunk, the two-stage statistics loop, a serial ragged tail when present, and the
two-stage apply/store loop. Ping/pong slots and persistent running statistics
are explicit. A line such as `lifetime ends: x(t0)` describes the logical
last-use point; it does not imply that the IR contains an explicit free
instruction.

Bare calls to `passes.auto_tile()` have no compilation artifact directory and
therefore keep the concise `INFO` log only. Use the schedule report together
with [IR Lowering Trace](../07-ir-lower-trace.md) to inspect the transformation
and [Memory Map](../07-memory-map.md) to inspect final UB addresses and physical
reuse.

## Schedule families

### Materialized

The complete per-core region fits in UB. Boundary tensors are sliced once per
phase, reused through their last topological use, and every returned value stays
live through its distinct `tensor.assemble` store.

### Pointwise stream

An oversized pointwise region is split along one axis. The emitted strip loop is
a two-stage `ForKind::Pipeline`, allowing load/store work for one strip to
overlap vector work for another after [`LowerPipelineLoops`](29-lower_pipeline_loops.md).
All returned values are loop-carried and stored; no intermediate GM tensor is
introduced. DMA-aligned physical tiles retain their exact logical valid shape
through every generated operation, so a ragged store cannot write padded rows
or columns.

### Folded and spanning reduction

The stats phase reduces fixed-size chunks and carries a thin accumulator.
A folded schedule runs the remaining thin operations once after the reduction.
A spanning schedule makes a second chunked pass over the wide input and applies
the reduced statistic. Full chunks may use a two-stage pipeline; initialization
and the ragged tail are serial.

### Online softmax

The canonical softmax schedule carries a running maximum and corrected running
sum across chunks, then makes one chunked output pass. It is numerically stable
without materializing exponentials in GM.

### Column reduction

Column reductions stream chunks of the reduced row axis while carrying one thin
column accumulator. A consumer over the original wide tensor uses a second
streaming apply phase. AutoTile does not emit a seed kernel or atomic partial
stores: if a column-reduction graph cannot be realized by one capacity-safe
kernel, the marked function is rejected.

## UB and transfer accounting

UB planning is dtype- and lifetime-aware. It accounts for boundary loads,
intermediates, all returned live-outs, DMA padding, the second bank of a
two-stage pipeline, padded row-reduction scratch tiles inserted by tensor-to-tile
lowering, high-precision `rsqrt` scratch, and thin accumulators. Metadata-only
`set_validshape` aliases do not allocate a second buffer. Column reductions
lower without a scratch tile and are priced that way.

Ordinary casts are conversions with distinct source and destination storage;
MemoryReuse must not turn `tile.cast` into an in-place operation. An explicitly
requested equal-byte `tile.reinterpret_view` remains the zero-copy opt-in and
lowers through the bitcast/view path instead of `TCVT`.

The cost model combines:

1. primitive cycle estimates and grounded Ascend 910B FP32/FP16 row/column
   reduction tables, interpolated at the emitted extent;
2. the exact primitive sequence generated for online-softmax initialization and
   updates;
3. summed logical GM-to-UB input and UB-to-GM output traffic for each phase;
4. `max(compute, transfer)` only for an emitted two-stage phase, otherwise the
   serial sum; and
5. task and wave fill terms.

Every capacity-safe reduction chunk is evaluated and the minimum modeled cost
wins; the largest fitting chunk is not assumed fastest. Both admitted compute
dtypes use grounded reduction tables. The implementation retains an explicit
conservative fallback for future backend expansion rather than presenting an
ungrounded estimate as measured data.

The model intentionally keeps the conservative summed directional GM term. It
does not assume independent MTE2/MTE3 overlap, fit a new bandwidth coefficient,
or infer an implicit pipeline that is absent from the IR.

## Outputs and calls

For an entry function, returned tensors become explicit `Out` parameters while
the original return tuple is retained for later normalization. For a marked
helper called elsewhere in the program, output storage is created inside the
helper so its call signature remains valid. Multiple live-outs always receive
distinct stores. Existing explicit `Out` parameters are reused positionally;
direct calls and `Submit` sites keep that declared signature.

Successful emission contains one `pl.spmd` scope and one non-split Vector
InCore body. Later hierarchy outlining therefore produces one AIV kernel for
the marked tensor DAG.

## Relationship to other tilers

This pass owns tensor-level Vector scheduling from GM through UB. It does not
replace [`AutoTileMatmulL0`](16-auto_tile_matmul_l0.md), which works later on a
single Cube matmul's L0 geometry. Cube, mixed-kernel, and graph-partitioning
support are intentionally outside this first AutoTile contract.
