# FoldNoOpReshape Pass

Folds `tile.reshape` calls that change neither physical shape nor allocation
into plain Var-to-Var assignments, removing the trivial reshape Call from the
IR before PTO codegen.

## Overview

After `LegalizePTOBufferReuse` runs, the LHS and RHS of a `tile.reshape` may
already point at the same `MemRef` root *and* carry identical
`TileBufSignature`s. In that case the reshape is a no-op at the PTO level —
the per-var alloc model has pre-declared LHS with the same shape, layout,
fractal, valid-shape and pad as RHS, and they share a memory address. There
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

**Requirements**:

- `IRProperty::SplitIncoreOrch` — Orchestration is split out from InCore code
- `IRProperty::IncoreTileOps` — InCore functions use tile types
- `IRProperty::HasMemRefs` — `MemRef` slots populated by `InitMemRef`
- `IRProperty::TileOps2D` — tile ops are at most 2D
- The pass requires `LegalizePTOBufferReuse` to have run so that
  view-merging decisions are reflected on the canonical alloc — otherwise
  LHS and RHS may not yet share a `MemRef` even though they should.
- Only InCore-type functions (`InCore`, `AIC`, `AIV`) are scanned; Opaque
  and Orchestration functions are returned unchanged.

**When to use**: 29th pass in the `Default` strategy, immediately after
`AllocateMemoryAddr` (so `MemRef` merging is finalized) and before
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
`tile.reshape`, it checks four conditions:

1. **LHS and source are tiles**: both `assign.var.type` and the source
   argument's type cast successfully to `TileType`.
2. **Both are MemRef-backed**: `tile_type.memref_` is set on both, and
   neither is null.
3. **Same MemRef root**: `lhs_tile.memref->base.get() == rhs_tile.memref->base.get()`.
4. **Identical signatures**: `TileBufSignature::FromTileType(lhs) == TileBufSignature::FromTileType(rhs)`.

When all four hold, the `AssignStmt(lhs, Call(tile.reshape, [src, shape]))`
is replaced by `AssignStmt(lhs, src)`. The Call is dropped entirely; LHS
becomes a pure alias of RHS at that statement, and downstream uses see
exactly the same MemRef and type they did before.

The pass touches no other statement form and never modifies a reshape
whose LHS/RHS differ in any of those four ways — those cases require real
`pto.treshape` emission.

| Source pattern | Action |
| -------------- | ------ |
| `lhs = tile.reshape(rhs, shape)` with same MemRef + same `TileBufSignature` | Rewrite to `lhs = rhs`; drop Call |
| `lhs = tile.reshape(rhs, shape)` with different MemRef root | Unchanged |
| `lhs = tile.reshape(rhs, shape)` with same MemRef but different `TileBufSignature` | Unchanged (real reshape) |
| Any non-`tile.reshape` Call | Unchanged |
| Function is Opaque / Orchestration | Function returned unchanged |

## Example

### Trivial reshape after MemRef sharing

```python
# Before pass (TileBufSignature equal on both sides; same MemRef R after
# LegalizePTOBufferReuse)
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

## Verification

**Tests**: `tests/ut/ir/transforms/test_fold_no_op_reshape.py`

- `test_genuine_reshape_kept` — physical-shape-changing reshapes survive
- `test_pass_runs_without_error_on_simple_kernel` — smoke test on a
  no-reshape kernel returns unchanged

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
