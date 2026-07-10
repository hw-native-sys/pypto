# Unified `valid_shape` Semantics

`valid_shape` marks the valid sub-region of a shaped value. An access inside the
physical `shape_` but outside `valid_shape` reads the view's `pad` value
(`PadValue.null` / `zero` / `max` / `min`). Both view structs carry it —
`TensorView.valid_shape` and `TileView.valid_shape` — under one shared ruleset, so a
ragged-edge kernel can be written as

```text
slice / load  ->  compute  ->  assemble / store
```

without hand-threading a `min(TILE, N - i*TILE)` extent through every op.

See also [`02-types.md`](02-types.md) for `TensorType` / `TileType` themselves.

## Core rules

**D1 — tiles are physically 2-D.** A tile's physical shape is always `ConstInt`.
Dynamism rides entirely on `valid_shape`, which may be a `ConstInt`, a `Var`, or a
`Call` such as `pl.min(...)`. `FlattenTileNdTo2d` is the sole ND → 2-D lowering point,
so PTO codegen reading only `valid_shape[0]` / `valid_shape[1]` is correct by
construction.

**D2 — unset means fully valid.** An absent or empty `valid_shape` denotes the full
physical shape.

**Canonical encoding.** A `valid_shape` equal to the physical shape carries no
information and is collapsed at construction, so exactly one in-memory form exists per
semantic state. `TileType` resets the **whole** view (nothing else is meaningful on a
fully-valid tile); `TensorType` clears only the `valid_shape` **field**, because
`stride` / `layout` / `pad` remain independently meaningful.

```python
from pypto import DataType, ir

span = ir.Span.unknown()
ci = lambda v: ir.ConstInt(v, DataType.INT64, span)

# TILE: explicit valid_shape == shape collapses the whole view to the no-view form.
full = ir.TileType([ci(128), ci(128)], DataType.FP32, None,
                   ir.TileView(valid_shape=[ci(128), ci(128)]), ir.Mem.Vec)
assert full.tile_view is None
ir.assert_structural_equal(
    full, ir.TileType([ci(128), ci(128)], DataType.FP32, None, None, ir.Mem.Vec))

# TENSOR: only the valid_shape field is cleared; the stride/layout view survives.
# A PARTIAL valid_shape (e.g. [32, 64]) is kept verbatim on either view.
tv = ir.TensorView(stride=[ci(128), ci(1)], layout=ir.TensorLayout.ND,
                   valid_shape=[ci(128), ci(128)])
t = ir.TensorType([ci(128), ci(128)], DataType.FP32, tensor_view=tv)
assert len(t.tensor_view.valid_shape) == 0
```

**Standing invariant** (always-on `TypeCheck` verifier): `rank(valid) == rank(shape)`
and `0 <= valid[i] <= shape[i]` for every static dim (symbolic dims deferred). A
`pl.load(valid_shapes=[999, 999])` on a `[128, 128]` tile is rejected.

**`GetValidShape()` is the single source of truth**
([`include/pypto/ir/type_inference.h`](../../../../include/pypto/ir/type_inference.h)):
it returns the view's `valid_shape` when set, else the physical `shape_`, so "unset"
and "explicit-full" are indistinguishable to every consumer.

## Propagation and rejection

Op type inference must never corrupt a valid region. Two failure modes, both forbidden:

- **Widening** — marking padding as real data. Garbage is written into a tensor.
- **Narrowing** — silently dropping real data. Writes are truncated.

When a correct output region cannot be *proven*, the op **rejects** with `CHECK_SPAN`
rather than defaulting to the full shape or to arg0's region. The shared helpers live in
[`src/ir/op/type_inference.cpp`](../../../../src/ir/op/type_inference.cpp);
`ComputeAssembleUnionValidShape` is the standard of rigor.

| Op family | `valid_shape` rule | Rejects (`CHECK_SPAN`) when |
| --------- | ------------------ | --------------------------- |
| `load(valid_shapes=)` | Sets it; **intersects** the source tensor's region; inherits `pad` | a negative (non-origin-anchored) window offset |
| unary / cast / scalar-binary / move | Copy the input's region | — |
| elementwise binary / multi-operand | Per-dim **agreement**: equal extents on non-broadcast dims; `valid_shape` never broadcasts | a provable static extent disagreement |
| `part_add` / `part_mul` / `part_max` / `part_min` | Per-dim **union** (valid where *either* source is valid) | the union is not an origin-anchored rectangle |
| reduction | Drop the reduced axis, keep the rest | — |
| `matmul(A[M,K], B[K,N])` | `[valid(A)[M], valid(B)[N]]`; K must agree | a provable static K mismatch |
| `assemble(target, source, off)` | Per-dim bounding box of the target + written rectangles | the union is not a provable origin-anchored rectangle (gap / L-shape / unprovable) |
| `slice(clamp=True)` | Clip the window to the source region at `offset` (never widens) | a negative offset |
| `reshape` | Map the region through the flattened buffer | the input region is not a contiguous flat prefix |
| `transpose` / `extract` / `concat` | Permute / carve / stack the region | a `concat` non-final operand is partially valid (L-shape) |
| `sort` / `mrgsort` | Full-shape output | a partially-valid input (padding would enter the compare) |
| `store` | Writes exactly `valid(tile)` | — |
| tensor→tile `load`; tensor compute ops | Inherit the source region / carry the view | same as the tile peer |

### Why `assemble` rejects

`assemble` unions the target's valid rectangle `[0, valid(target))` with the written
rectangle `[offset, offset + valid(source))`. A `(valid_row, valid_col)` pair can only
describe **one origin-anchored rectangle**, so the union is representable only when it
*is* one — in which case it equals the per-dim bounding box:

```text
out_valid[i] = min( shape[i], max( valid(target)[i], offset[i] + valid(source)[i] ) )
```

A gap (`offset > valid(target)` on the growing dim) or an L-shape (one dim not fully
covered while another grows) would make the bounding box mark cells valid that *neither*
operand wrote. Those are rejected. The proof is **per-dim**: a symbolic passenger dim
does not stop a provable static L-shape from being rejected.

The proof also discharges a **symbolic contiguous append** — `offset` and
`valid(target)` being the *same expression* proves there is no gap, even when neither is
a `ConstInt`. That is the accumulator idiom:

```text
assemble(acc /*valid [v, 128]*/, src /*valid [32, 128]*/, offset=[v, 0])
    -> valid [min(128, max(v, v + 32)), 128]
```

Two *different* symbolic extents remain unprovable and are still rejected.

### The empty accumulator

Because unset means fully valid, a fresh `pl.create_tile([BM, BD], pl.FP32)` is fully
valid and the union can never narrow. Creating the accumulator empty is what makes the
rule usable:

```python
dst    = pl.create_tile([BM, BD], pl.FP32, valid_shape=[0, 0])
packed = pl.tile.assemble(dst, y, [0, 0])   # -> valid_shape == valid(y)
```

Successive appends grow the bounding box monotonically, provided each append stays
representable (contiguous growth along one dim, full coverage on the others).

## ND → 2-D lowering contract

For tile shape `[d0 ... d_{k-1}]` flattened to `rows = Π(d0 ... d_{k-2})`,
`cols = d_{k-1}`, the flat row index is

```text
flat_row = i0*(d1*...*d_{k-2}) + i1*(d2*...*d_{k-2}) + ... + i_{k-2}
```

The ND valid region is a **contiguous row prefix** — the only thing
`(valid_row, valid_col)` can express — iff there is a single *free* row dim: reading the
row dims most-significant first, every one before it is pinned (`valid[j] == 1`, so `i_j`
is forced to 0 and contributes no stride) and every one after it is fully valid.

```text
exists f in [0, k-2] such that
    valid[j] == 1           for all j <  f     // pinned: index forced to 0
    valid[j] == shape[j]    for all f < j <= k-2
```

Under this precondition the product fold `Π(valid[0..k-2])` is exactly correct. A partial
middle dim below a non-unit outer dim yields a *strided* region and is rejected.

Two worked examples. Shape `[16,4,8]` with valid `[1,2,8]`: `i0` is pinned, so
`flat_row = i1 ∈ {0,1}` — contiguous, and the fold gives `1*2 = 2`. ✓
Shape `[4,8,16]` with valid `[3,1,16]`: the free dim is `0` but `valid[1]=1 != 8`, so
`flat_row = i0*8 ∈ {0,8,16}` — strided, and the fold would wrongly give `3`. ✗

The pinning test is on `valid[j] == 1`, not `shape[j] == 1`: a `valid[j]` of 1 forces
`i_j = 0` regardless of the physical extent. An **empty** region (any dim provably `0`) is
trivially a prefix and folds to zero rows.

See [`../passes/13-flatten_tile_nd_to_2d.md`](../passes/13-flatten_tile_nd_to_2d.md).

## User-facing surface

| API | Purpose |
| --- | ------- |
| `pl.load(t, offs, shapes, valid_shapes=...)` | Attach a valid region to a loaded tile |
| `pl.slice(x, shape, offset, valid_shape=..., clamp=...)` | Slice; `clamp=True` derives the ragged-tail extent |
| `pl.create_tile(shape, dtype, valid_shape=...)` | Create a tile with an explicit (possibly empty) region |
| `pl.valid_dim(t, i)` | Compile-time query of the valid extent on axis `i` |
| `pl.fillpad(t, pad_value=...)` | Fill the invalid region with a pad value |
| `pl.store(t, offs, out, shapes=...)` | Partial write-back |

`pl.set_validshape` exists but is an internal, compiler-facing API (rank-2 only).
