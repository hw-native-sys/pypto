# FoldNoOpReshape Pass

Folds `tile.reshape` calls that change neither physical shape nor allocation
into plain Var-to-Var assignments, removing the trivial reshape Call from the
IR before PTO codegen.

## Overview

After `InitMemRef` and `MaterializeSemanticAliases` finalize allocation
identities, the LHS and RHS of a `tile.reshape` may already point at the same
`MemRef` root *and* carry identical
`TileBufSignature`s. In that case the reshape is a no-op at the PTO level —
the per-var alloc model has pre-declared LHS with the same shape, layout,
fractal, valid-shape and pad as RHS, and they share one allocation identity. There
is nothing for `pto.treshape` to do.

Historically PTO codegen detected this case at emission time and silently
dropped the `pto.treshape` line via a peephole. That hid an IR-to-IR
optimization inside the codegen layer; this pass moves the optimization to
where it belongs and rewrites:

```python
lhs: pl.Tile[..., MemRef(R)] = pl.tile.reshape(rhs, [...])  # rhs has same MemRef + sig
```

into:

```python
lhs: pl.Tile[..., MemRef(R)] = rhs
```

PTO codegen can then translate the `tile.reshape` op 1:1 in all surviving
cases, knowing the no-op cases were already removed upstream.

The pass also folds an identity reshape of a **buffer-less** tile — a
cross-core `tile.tpop_from_aic` / `tile.tpop_from_aiv` result, or a view chained
off one, which `InitMemRef` leaves MemRef-less. With no allocation to compare,
the proof is that the two tile types are structurally equal. This fold is
required for correctness, not just size: a surviving reshape lowers to a
`pto.treshape` view, which has no `valid_row` / `valid_col` operands, so a
symbolic `valid_shape` on it is never set at runtime and consumers such as
`pto.tstore` read an uninitialized extent. An identity reshape of such a tile
(for example a rank-raising `[16, 128] -> [16, 1, 128]`, which
`FlattenTileNdTo2D` collapses back to `[16, 128]`) is a no-op, so folding it
lets consumers read the popped tile directly, whose `pto.tpop_from_*` carries
the runtime extent as operands. A reshape that really changes the shape still
lowers to `pto.treshape`; this pass does not address a symbolic `valid_shape`
on one.

**Requirements**:

- `IRProperty::SplitIncoreOrch` — Orchestration is split out from InCore code
- `IRProperty::IncoreTileOps` — InCore functions use tile types
- `IRProperty::HasMemRefs` — `MemRef` slots populated by `InitMemRef`
- `IRProperty::TileOps2D` — tile ops are at most 2D
- `MaterializeSemanticAliases` must have finalized semantics-required sharing.
  `PYPTO` and `DSA_RP` run `AllocateMemoryAddr` before this pass; `PTOAS`
  deliberately skips address assignment, but the same-root identity is already
  sufficient because ptoas must place both values in one allocation.
- Only InCore-type functions (`InCore`, `AIC`, `AIV`) are scanned; Opaque
  and Orchestration functions are returned unchanged.

**When to use**: In the `Default` strategy, immediately after the selected
planner's memory stage (`AllocateMemoryAddr`
when PyPTO owns placement, otherwise the finalized semantic aliases) and before
`FuseCreateAssembleToSlice`.

## API

| C++ | Python | Level |
| --- | ------ | ----- |
| `pass::FoldNoOpReshape()` | `passes.fold_no_op_reshape()` | Function-level |

**Python usage**:

```python
from pypto.pypto_core import passes

fold_pass = passes.fold_no_op_reshape()
program_folded = fold_pass(program)
```

## Algorithm

For each InCore-type function (others returned unchanged) `FoldNoOpReshapeMutator`
walks the body. For every `AssignStmt` whose value is a `Call` to
`tile.reshape`, it first requires that **LHS and source are tiles**: both
`assign.var.type` and the source argument's type cast successfully to
`TileType`. It then takes one of two routes, depending on MemRefs:

**Both sides MemRef-backed** — three more conditions:

1. **Both MemRefs non-null**: `tile_type.memref_` is set on both, and neither
   is null.
2. **Same MemRef root**: `CompareBaseAddress(lhs_memref, rhs_memref)` is `kSame`.
3. **Identical signatures**: `TileBufSignature::FromTileType(lhs) == TileBufSignature::FromTileType(rhs)`.

**Neither side MemRef-backed** (a buffer-less tpop result or a view off one) —
one condition:

1. **Structurally equal types**: `structural_equal(lhs.type, rhs.type)`. This
   is stricter than the signature check on purpose: `TileBufSignature`
   records every symbolic valid dim only as "dynamic", so `[vr, 128]` and
   `[vc, 128]` would compare equal, while an alias makes the LHS denote the
   source's runtime extent.

When the conditions of either route hold, the
`AssignStmt(lhs, Call(tile.reshape, [src, shape]))` is replaced by
`AssignStmt(lhs, src)`. The Call is dropped entirely; LHS becomes a pure alias
of RHS at that statement, and downstream uses see exactly the same MemRef and
type they did before. A reshape with a MemRef on only one side is never folded.

The pass touches no other statement form and never modifies a reshape that
fails its route's conditions — those cases require real `pto.treshape`
emission.

| Source pattern | Action |
| -------------- | ------ |
| `lhs = tile.reshape(rhs, shape)` with same MemRef + same `TileBufSignature` | Rewrite to `lhs = rhs`; drop Call |
| `lhs = tile.reshape(rhs, shape)` with different MemRef root | Unchanged |
| `lhs = tile.reshape(rhs, shape)` with same MemRef but different `TileBufSignature` | Unchanged (real reshape) |
| `lhs = tile.reshape(rhs, shape)`, both MemRef-less, structurally equal types | Rewrite to `lhs = rhs`; drop Call |
| `lhs = tile.reshape(rhs, shape)`, both MemRef-less, types differ | Unchanged (real view) |
| `lhs = tile.reshape(rhs, shape)` with a MemRef on only one side | Unchanged |
| Any non-`tile.reshape` Call | Unchanged |
| Function is Opaque / Orchestration | Function returned unchanged |

## Example

### Trivial reshape after MemRef sharing

```python
# Before pass (TileBufSignature equal on both sides; same MemRef R after
# semantic alias materialization and any PyPTO-owned placement)
@pl.function(type=pl.FunctionType.InCore)
def kernel(x, out):
    a: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec, MemRef(R)] = pl.tile.load(x, ...)
    b: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec, MemRef(R)] = pl.tile.reshape(a, [64, 64])
    pl.tile.store(b, [0, 0], out)
```

```python
# After FoldNoOpReshape
@pl.function(type=pl.FunctionType.InCore)
def kernel(x, out):
    a: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec, MemRef(R)] = pl.tile.load(x, ...)
    b: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec, MemRef(R)] = a   # Var-to-Var
    pl.tile.store(b, [0, 0], out)
```

PTO codegen now never sees the reshape Call for this case. Downstream
passes such as `Simplify` may further inline the alias.

### Genuine reshape preserved

```python
# Different physical shape — must NOT be folded
a: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec, MemRef(R)] = pl.tile.load(x, ...)
b: pl.Tile[[4096, 1], pl.FP32, pl.Mem.Vec, MemRef(R)] = pl.tile.reshape(a, [4096, 1])
```

`TileBufSignature::FromTileType` produces different `rows`/`cols` for `a`
vs `b`, so `lhs_sig == rhs_sig` is false and the pass leaves the Call in
place. PTO codegen will emit a real `pto.treshape`.

### Identity reshape over a cross-core tile with a runtime valid extent

```python
# DSL (InCore): cv is a cube result moved to Vec, carrying valid rows `vr`
r = pl.reshape(cv, [16, 1, 128])
back = pl.reshape(r, [16, 128])
pl.store(back, [0, 0], out)
```

After `FlattenTileNdTo2D` and `ExpandMixedKernel`, both reshapes are
identities over the MemRef-less popped tile:

```python
# Before pass (AIV)
cv: pl.Tile[[16, 128], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[vr, 128])] = pl.tile.tpop_from_aic(split=0)
r: pl.Tile[[16, 128], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[vr, 128])] = pl.tile.reshape(cv, [16, 128])
back: pl.Tile[[16, 128], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[vr, 128])] = pl.tile.reshape(r, [16, 128])
```

```python
# After FoldNoOpReshape
r: pl.Tile[[16, 128], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[vr, 128])] = cv
back: pl.Tile[[16, 128], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[vr, 128])] = r
```

The store now reads the popped tile, whose `pto.tpop_from_aic(%vr, %c128)`
sets the runtime extent. Left unfolded, each reshape would become a
`pto.treshape` with `v_row=?, v_col=?` whose extent is never set, and the store
would write nothing.

## Verification

**Tests**: `tests/ut/ir/transforms/test_fold_no_op_reshape.py`

- `test_noop_reshape_is_folded` — a same-MemRef, same-signature reshape folds
- `test_genuine_reshape_kept` — physical-shape-changing reshapes survive
- `test_same_allocation_different_window_is_kept` — one allocation base at
  distinct offsets is not the same value
- `test_buffer_less_identity_reshape_is_folded` — an identity reshape of a
  MemRef-less tpop result with a symbolic valid shape folds
- `test_buffer_less_reshape_with_a_different_valid_extent_is_kept` — equal
  signatures but different symbolic valid extents do not fold
- `test_buffer_less_shape_change_is_kept` — a shape-changing view of a
  MemRef-less tile survives
- `test_pass_runs_without_error_on_simple_kernel` — smoke test on a
  no-reshape kernel returns unchanged

End to end, `tests/ut/codegen/test_pto_codegen_tpop_view.py::test_identity_reshape_over_dynamic_valid_tpop_reads_the_popped_tile`
checks under both memory planners that the example above emits no
`pto.treshape` and that `pto.tstore` reads the popped tile.

The codegen-side peephole that previously dropped no-op reshape emission
remains in place as defence-in-depth and can be removed in a follow-up
once this pass is observed to handle every case in the field.

## Pass Properties

| Property | Value |
| -------- | ----- |
| Required | `SplitIncoreOrch`, `IncoreTileOps`, `HasMemRefs`, `TileOps2D` |
| Produced | — |
| Invalidated | — |

The pass preserves every input property: it only rewrites the value of an
`AssignStmt` from a Call to a Var, both of the same `TileType`. SSA form,
type checks, MemRef bindings, and tile-op shape constraints are
unaffected.

## Scope

| Function type | Action |
| ------------- | ------ |
| InCore (InCore, AIC, AIV) | Scanned; eligible no-op reshapes folded |
| Orchestration | Returned unchanged |
| Opaque | Returned unchanged |

The pass is a no-op when no InCore-type function contains a foldable
`tile.reshape` AssignStmt.
