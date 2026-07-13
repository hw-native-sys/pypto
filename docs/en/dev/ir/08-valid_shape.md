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

**D1 — backend tile buffers are physically 2-D.** Before `FlattenTileNdTo2D`, logical
`TileType` values may be N-D. That pass is the stage boundary: afterward every
`TileType` has rank at most 2 and `ConstInt` physical extents. PTO's `tile_buf` is
always exactly 2-D; a logical rank-1 `[N]` tile is normalized to physical `[1, N]` by
`ExtractTileTypeInfo`. Runtime dynamism rides on `valid_shape`, whose elements may be
a `ConstInt`, a `Var`, or a `Call` such as `pl.min(...)`. This rank/static-extent rule
is a post-flatten invariant, not a global constructor rule.

**D2 — unset means fully valid.** An absent or empty `valid_shape` denotes the full
physical shape.

**Canonical encoding.** A `valid_shape` structurally equal to the physical shape
carries no information, so its field is cleared at construction. The whole optional
view is reset only when every remaining field matches the implicit view. This preserves
independently meaningful `stride` / `layout` / `start_offset` / `pad` metadata on a
fully-valid `TileType`, and `stride` / `layout` / `pad` on a `TensorType`.

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

# A non-default tile view survives; only its redundant valid_shape is cleared.
strided = ir.TileType(
    [ci(128), ci(128)], DataType.FP32, None,
    ir.TileView(valid_shape=[ci(128), ci(128)], stride=[ci(128), ci(1)]),
    ir.Mem.Vec)
assert len(strided.tile_view.valid_shape) == 0
assert len(strided.tile_view.stride) == 2

# TENSOR: only the valid_shape field is cleared; the stride/layout view survives.
# A PARTIAL valid_shape (e.g. [32, 64]) is kept verbatim on either view.
tv = ir.TensorView(stride=[ci(128), ci(1)], layout=ir.TensorLayout.ND,
                   valid_shape=[ci(128), ci(128)])
t = ir.TensorType([ci(128), ci(128)], DataType.FP32, tensor_view=tv)
assert len(t.tensor_view.valid_shape) == 0
```

**Well-formedness invariant** (when the `TypeCheck` property verifier is enabled):
`rank(valid) == rank(shape)` and `0 <= valid[i] <= shape[i]`. The arithmetic analyzer
rejects every relation it can prove false; genuinely unknown symbolic bounds are
deferred. Automatic property verification can be disabled with
`VerificationLevel.None`, so this is not an always-on constructor guarantee. Operators
that accept or derive a valid region also call `ValidateValidShapeBounds` at type
inference boundaries and reject provable violations even when automatic verification is
off. For example, `pl.load(valid_shapes=[999, 999])` on a `[128, 128]` tile is rejected.

**`GetValidShape()` is the single source of truth**
([`include/pypto/ir/type_inference.h`](../../../../include/pypto/ir/type_inference.h)):
it returns the view's `valid_shape` when set, else the physical `shape_`, so "unset"
and "explicit-full" are indistinguishable to every consumer.

**Fresh compute is not a view.** A newly computed value preserves the semantic
effective `valid_shape`, but it must not accidentally inherit metadata belonging to the
source allocation. Fresh tensor results use default ND layout, empty strides, and null
padding. Fresh tile results retain only the block/scatter layout and fractal constraints
needed to produce the result; source `stride`, `start_offset`, `memref`, and `pad` do not
propagate. View-producing and in-place operations preserve or recompute alias/padding
metadata only when their own contract intentionally requires it.

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
| `load(valid_shapes=)` | Sets it; **intersects** the source tensor's region; inherits `pad` | window/source ranks differ; requested validity exceeds tile capacity; the requested transfer is out of the source allocation; or an offset is negative |
| unary / cast / scalar-binary / move | Preserve the input's effective region on a fresh result | — |
| elementwise binary / multi-operand | Per-dim **agreement**: non-broadcast contributors must be provably equal; `valid_shape` never broadcasts | equality is false **or unknown**; a broadcast singleton is not provably valid with extent 1 |
| `part_add` / `part_mul` / `part_max` / `part_min` | Per-dim **union** (valid where *either* source is valid) | the union is not an origin-anchored rectangle |
| reduction | Drop the reduced axis, keep the rest | — |
| 2-D `matmul(A[M,K], B[K,N])` | `[valid(A)[M], valid(B)[N]]`; PTO contracts over `valid(A)[K]` | `valid(A)[K] <= valid(B)[K]` is false **or unknown** |
| `tile.batch_matmul` | Broadcast the physical batch shape; propagate partial M/N and use the same directional K rule | any input batch axis is not provably fully valid, including unknown symbolic equality |
| `assemble(target, source, off)` | Per-dim bounding box of the target + source-valid write rectangles | an offset is negative; the tensor-valid transfer (or tile physical subview) cannot fit the target; or the union is not a provable origin-anchored rectangle (gap / L-shape / unprovable) |
| `slice(clamp=True)` | Clip the full-rank window to the source region at `offset` (never widens) | window/offset/source ranks differ, an offset is negative, or the effective valid transfer extends past the source allocation |
| `slice(..., drop_dims=...)` | Erase only static physical unit axes whose post-intersection valid extent is provably 1 | a dropped axis is empty or its unit validity is unproven |
| `reshape` | Map a contiguous flat prefix; inserting/removing provably-full physical unit axes preserves any rectangle exactly; an empty region stays empty | a genuine data-repartitioning reshape receives a non-prefix region, or a removed unit axis is not provably valid |
| `transpose` / `extract` / `concat` | Permute / carve / stack the region | a `concat` non-final operand is partially valid (L-shape) |
| indexed / gather / scatter / sort families | Fail closed: every data, index, workspace, accumulator, or destination operand consumed as a full contract must be provably fully valid | any required operand is partial or full-valid equality is unknown |
| AIV shard / gather boundary | The split axis must be provably fully valid; validity on non-split axes is preserved | partial or symbolically unknown split-axis validity |
| `store` | Transfer exactly `valid(tile)`, or the explicit original-rank partition injected by flattening | transfer rank differs from the destination; an offset is provably negative; or `offset + transfer > destination` is provable |
| tensor→tile `load`; tensor compute ops | Preserve the effective region across the boundary; fresh compute follows the metadata rule above | same proof obligations as the tile peer |

Control-flow joins are part of the contract too. Every `if` branch yield and its
declared return variable must agree on dtype, physical shape, effective `valid_shape`,
and padding policy (`PadValue.null` for an absent view); loop-carried values obey the
corresponding invariant. A symbolic validity equality that cannot be proven is rejected
rather than allowing the join annotation to widen either path.

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
row dims most-significant first, every one before it is either pinned by
`valid[j] == 1` or physically unit-sized (`shape[j] == 1`, so a symbolic 0/1
validity is only an empty/non-empty gate), and every one after it is fully valid.

```text
exists f in [0, k-2] such that
    valid[j] == 1 OR shape[j] == 1  for all j < f
    valid[j] == shape[j]            for all f < j <= k-2
```

Under this precondition the product fold `Π(valid[0..k-2])` is exactly correct. A partial
middle dim below a non-unit outer dim yields a *strided* region and is rejected.

Two worked examples. Shape `[16,4,8]` with valid `[1,2,8]`: `i0` is pinned, so
`flat_row = i1 ∈ {0,1}` — contiguous, and the fold gives `1*2 = 2`. ✓
Shape `[4,8,16]` with valid `[3,1,16]`: the free dim is `0` but `valid[1]=1 != 8`, so
`flat_row = i0*8 ∈ {0,8,16}` — strided, and the fold would wrongly give `3`. ✗

A `valid[j]` of 1 pins the index regardless of physical extent. A physical unit axis is
also safe when its validity is symbolic: the standing bounds invariant limits it to 0 or
1, so it merely gates the whole flattened prefix. An **empty** region (any dim provably
`0`) is trivially a prefix and folds to zero rows.

See [`../passes/13-flatten_tile_nd_to_2d.md`](../passes/13-flatten_tile_nd_to_2d.md).

## User-facing surface

| API | Purpose |
| --- | ------- |
| `pl.load(t, offs, shapes, valid_shapes=...)` | Attach a valid region to a loaded tile |
| `pl.slice(x, shape, offset, valid_shape=..., drop_dims=..., clamp=...)` | Full-rank slice; `drop_dims` explicitly erases valid unit axes and `clamp=True` derives the ragged-tail extent |
| `pl.create_tile(shape, dtype, valid_shape=...)` | Create a tile with an explicit (possibly empty) region |
| `pl.valid_dim(t, i)` | Compile-time query of the valid extent on axis `i` |
| `pl.fillpad(t, pad_value=...)` | Fill the invalid region with a pad value |
| `pl.store(t, offs, out, shapes=...)` | Partial write-back |

`pl.set_validshape` exists but is an internal, compiler-facing API (rank-2 only).
