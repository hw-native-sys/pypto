# CanonicalizeMatSlice Pass

Lowers Mat-resident `tile.slice` into the canonical `tile.extract` form, so all Mat→Left/Right movement is unified on `pto.textract`.

## Overview

A `tile.slice` whose result tile is `Mem.Mat` is a legal high-level "sub-window of a Mat tile" construct. [`FlattenTileNdTo2D`](15-flatten_tile_nd_to_2d.md) emits one per batch page when it unrolls a `tile.batch_matmul`: the page offset is `batch_index * page_rows`, and for a leading-dim-1 batch that offset is 0 and the window covers the whole tile — but it is still a `tile.slice`.

A standalone Mat-resident `tile.slice` has **no hardware lowering**: codegen would materialize it as a `loc=mat -> loc=mat` `pto.tmov`, and targets such as Ascend 910C have no L1→L1 DMA path. This pass eliminates every Mat-resident `tile.slice` by folding its offset into each consumer, then drops the now-dead slice.

**Pipeline position**: After [`AutoTileMatmulL0`](16-auto_tile_matmul_l0.md) (so the per-iter `tile.extract`s that read the batch-page slices already exist), before [`InferTileMemorySpace`](18-infer_tile_memory_space.md).

**Requirements**: `SSAForm`, `SplitIncoreOrch`, `IncoreTileOps`, `TileOps2D`, `NormalizedStmtStructure`.

**Produces**: same as required (property-preserving rewrite).

**Invalidates**: nothing.

**When to use**: Always, as part of the default tile-stage pipeline. The pass is a no-op when no Mat-resident `tile.slice` exists.

## API

| C++ | Python | Level |
| --- | ------ | ----- |
| `pass::CanonicalizeMatSlice()` | `passes.canonicalize_mat_slice()` | Function-level |

```python
from pypto.pypto_core import passes

program_canon = passes.canonicalize_mat_slice()(program)
```

## Algorithm

For each InCore-typed function, in three phases:

1. **Collect** — index every `AssignStmt` whose value is a Mat-resident `tile.slice(src, shape, offset)` (canonical 3-argument form). A slice whose `src` is itself a Mat slice is peeled, accumulating the offset, so each entry resolves to a non-slice base tile plus a total `(off_row, off_col)`. Slices carrying `valid_shape` / `drop_dims` (4–5 arguments) are not plain windows and are skipped.

2. **Rewrite consumers** — for each Mat slice:
   - **`tile.extract(slice, ir, ic, shape)`** → `tile.extract(base, ir + off_row, ic + off_col, shape)`. The extract reads the slice's source directly; the index add is constant-folded when both terms are `ConstInt`.
   - **`tile.matmul` / `tile.matmul_acc` / `tile.matmul_bias` operand** → the operand is replaced by a fresh `tile.extract(base, off_row, off_col, slice_shape, target_memory=Left|Right)` — `Left` for the lhs operand, `Right` for the rhs. (The `tile.matmul_acc` accumulator operand is `Acc`-resident and never a Mat slice.)

3. **Drop dead slices** — a `tile.slice` whose result no longer has any use is removed. A chained slice only becomes dead once the slice consuming it is dropped, so this iterates to a fixpoint (bounded by the Mat-slice count). A Mat slice still used at the end had a consumer this pass does not canonicalize; it is left intact — no regression versus the pre-pass IR.

The pass is a `FunctionPass`; functions are returned unchanged when no Mat-resident `tile.slice` is present.

## Examples

### Slice folded into `tile.extract`

The offset-0 full-shape slice [`FlattenTileNdTo2D`](15-flatten_tile_nd_to_2d.md) emits for a leading-dim-1 batch operand:

**Before**:

```python
lhs_slice: pl.Tile[[32, 512], pl.INT8, pl.Mem.Mat] = pl.tile.slice(x_mat, [32, 512], [0, 0])
a:         pl.Tile[[32, 256], pl.INT8, pl.Mem.Left] = pl.tile.extract(
    lhs_slice, 0, ko, shape=[32, 256], target_memory=pl.Mem.Left)
```

**After** (slice dropped; extract reads the loaded Mat tile directly):

```python
a: pl.Tile[[32, 256], pl.INT8, pl.Mem.Left] = pl.tile.extract(
    x_mat, 0, ko, shape=[32, 256], target_memory=pl.Mem.Left)
```

A non-zero page offset folds into the extract index — e.g. a slice at `[32, 0]` turns `extract(slice, 0, ko, ...)` into `extract(x_mat, 32, ko, ...)`.

### Slice folded into a `tile.matmul` operand

When `AutoTileMatmulL0` leaves a matmul untiled (already L0-sized), its Mat-slice operands are converted directly:

**Before**:

```python
lhs_slice: pl.Tile[[16, 256], pl.BF16, pl.Mem.Mat] = pl.tile.slice(lhs_mat, [16, 256], [0, 0])
rhs_slice: pl.Tile[[256, 64], pl.BF16, pl.Mem.Mat] = pl.tile.slice(rhs_mat, [256, 64], [0, 0])
c:         pl.Tile[[16, 64],  pl.FP32, pl.Mem.Acc] = pl.tile.matmul(lhs_slice, rhs_slice)
```

**After**:

```python
lhs_left:  pl.Tile[[16, 256], pl.BF16, pl.Mem.Left]  = pl.tile.extract(
    lhs_mat, 0, 0, shape=[16, 256], target_memory=pl.Mem.Left)
rhs_right: pl.Tile[[256, 64], pl.BF16, pl.Mem.Right] = pl.tile.extract(
    rhs_mat, 0, 0, shape=[256, 64], target_memory=pl.Mem.Right)
c:         pl.Tile[[16, 64],  pl.FP32, pl.Mem.Acc]   = pl.tile.matmul(lhs_left, rhs_right)
```

## Implementation

**Header**: `include/pypto/ir/transforms/passes.h`

**Properties**: `include/pypto/ir/transforms/pass_properties.h` (`kCanonicalizeMatSliceProperties`)

**Implementation**: `src/ir/transforms/canonicalize_mat_slice_pass.cpp`

**Python binding**: `python/bindings/modules/passes.cpp`

**Tests**: `tests/ut/ir/transforms/test_canonicalize_mat_slice.py`

## Pass Properties

| Property | Value |
| -------- | ----- |
| Required | SSAForm, SplitIncoreOrch, IncoreTileOps, TileOps2D, NormalizedStmtStructure |
| Produced | SSAForm, SplitIncoreOrch, IncoreTileOps, TileOps2D, NormalizedStmtStructure |
| Invalidated | — |

## Scope

| Op | Action |
| -- | ------ |
| Mat-resident `tile.slice` (3-arg) feeding `tile.extract` | Folded into the extract; slice dropped |
| Mat-resident `tile.slice` (3-arg) feeding a matmul-family operand | Replaced by `tile.extract(target_memory=Left\|Right)`; slice dropped |
| Chained Mat `tile.slice` (slice of a slice) | Peeled; offsets accumulated |
| Mat `tile.slice` with `valid_shape` / `drop_dims` | Skipped (not a plain window) |
| Vec/Left/Right/Acc-resident `tile.slice` | Untouched (only Mat slices are lowered) |
| Functions with no Mat `tile.slice` | Returned unchanged |

## See also

- [`FlattenTileNdTo2D`](15-flatten_tile_nd_to_2d.md) — upstream pass; emits the Mat-resident batch-page `tile.slice` this pass lowers
- [`AutoTileMatmulL0`](16-auto_tile_matmul_l0.md) — upstream pass; emits the `tile.extract`s that consume the batch-page slices
