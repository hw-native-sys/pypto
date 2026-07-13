# CanonicalizeTileSlice Pass

Lowers a `tile.slice` into the canonical `tile.extract` form so that movement is unified on `pto.textract` — both Mat-resident slices (folded into matmul / `tile.extract` consumers) and Vec slices whose lazy materialization would corrupt their source (materialized for the `tile.col_expand_*` family, issues #1640 and #2010).

## Overview

A `tile.slice` whose result tile is `Mem.Mat` is a legal high-level "sub-window of a Mat tile" construct. [`FlattenTileNdTo2D`](13-flatten_tile_nd_to_2d.md) emits one per batch page when it unrolls a `tile.batch_matmul`: the page offset is `batch_index * page_rows`, and for a leading-dim-1 batch that offset is 0 and the window covers the whole tile — but it is still a `tile.slice`.

PTO ISA supports `pto.subview` on Mat as a zero-copy alias (no data movement), so a standalone Mat slice is valid when its consumer accepts the subview SSA directly. However, consumers that trigger lazy materialization (via `MaterializeSubviewOperandIfNeeded`) would attempt a `loc=mat → loc=mat` `pto.textract` — an unsupported L1→L1 DMA path on targets such as Ascend 910C. This pass eliminates Mat-resident `tile.slice` nodes whose consumers it can canonicalize (extract/matmul) by folding the offset into each consumer for efficiency, then drops the now-dead slice. A Mat slice with a consumer that is not canonicalized (e.g. `tile.move`) is left intact — it lowers to a valid `pto.subview`.

The pass also canonicalizes a **Vec** `tile.slice` consumed by the `tile.col_expand_*` family (issues #1640, #2010). Those ops cannot read a `pto.subview` operand, so codegen lazily materializes the slice via `pto.textract` into the slice's own result buffer — and because `tile.slice` inherits its source's memory, that buffer sits **inside the still-live source**. The extract therefore runs in place over its own input, which is only safe when it is an **identity copy**. Two conditions must hold:

| Condition | Why it can fail |
| --------- | --------------- |
| The destination **address** is right | `AllocateMemoryAddr` folds a `ConstInt` offset into `base + off`, but a **dynamic** offset cannot be encoded as a `ConstInt` address and falls back to the bare source base — the extracted window lands on the source's row 0 (#1640). |
| The destination **layout** matches | The slice's buffer is dense (row pitch = slice cols) while the source window is strided (row pitch = source cols). These coincide only for a **contiguous** window: a single row, or one spanning every column. A column slice of a multi-row tile (`t[:, a:b]`) repacks strided → dense on top of its own source and destroys it — only row 0 survives, because its dense destination happens to equal its source address (#2010). |

When either condition fails, the operand is replaced by a fresh `tile.extract(..., target_memory=Vec)`, whose result gets its own non-inherited allocation. `tile.extract` is registered `not_inplace_safe()`, so [`MemoryReuse`](30-memory_reuse.md) cannot place that fresh buffer back onto the source either. A slice whose materialization *is* an identity copy is left untouched, so it keeps sharing the source buffer rather than paying for a duplicate allocation.

**Pipeline position**: After [`AutoTileMatmulL0`](14-auto_tile_matmul_l0.md) (so the per-iter `tile.extract`s that read the batch-page slices already exist), before [`InferTileMemorySpace`](16-infer_tile_memory_space.md).

**Requirements**: `SSAForm`, `SplitIncoreOrch`, `IncoreTileOps`, `TileOps2D`, `NormalizedStmtStructure`.

**Produces**: same as required (property-preserving rewrite).

**Invalidates**: nothing.

**When to use**: Always, as part of the default tile-stage pipeline. The pass is a no-op when no canonical `tile.slice` exists.

## API

| C++ | Python | Level |
| --- | ------ | ----- |
| `pass::CanonicalizeTileSlice()` | `passes.canonicalize_tile_slice()` | Function-level |

```python
from pypto.pypto_core import passes

program_canon = passes.canonicalize_tile_slice()(program)
```

## Algorithm

For each InCore-typed function, in three phases:

1. **Collect** — index every `AssignStmt` whose value is a Mat-resident `tile.slice(src, shape, offset)` (canonical 3-argument form). A slice whose `src` is itself a Mat slice is peeled, accumulating the offset, so each entry resolves to a non-slice base tile plus a total `(off_row, off_col)`. Slices carrying `valid_shape` / `drop_dims` (4–5 arguments) are not plain windows and are skipped.

2. **Rewrite consumers** — for each slice:
   - **`tile.extract(slice, ir, ic, shape)`** (Mat slices only) → `tile.extract(base, ir + off_row, ic + off_col, shape)`. The extract reads the slice's source directly; the index add is constant-folded when both terms are `ConstInt`.
   - **`tile.matmul` / `tile.matmul_acc` / `tile.matmul_bias` operand** (Mat slices only) → the operand is replaced by a fresh `tile.extract(base, off_row, off_col, slice_shape, target_memory=Left|Right)` — `Left` for the lhs operand, `Right` for the rhs. (The `tile.matmul_acc` accumulator operand is `Acc`-resident and never a Mat slice.)
   - **`tile.col_expand_*` operand** (Vec slices only) → when the lazy `pto.textract` would not be an identity copy — a dynamic offset, or a window that is not contiguous in the base tile (more than one row *and* narrower than the base) — the operand is replaced by a fresh `tile.extract(base, off_row, off_col, slice_shape, target_memory=Vec)`. Both operands are checked. Contiguous const-offset windows are left untouched.

3. **Drop dead slices** — a `tile.slice` whose result no longer has any use is removed. A chained slice only becomes dead once the slice consuming it is dropped, so this iterates to a fixpoint (bounded by the slice count). A slice still used at the end had a consumer this pass does not canonicalize; it is left intact — no regression versus the pre-pass IR.

The pass is a `FunctionPass`; functions are returned unchanged when no canonical `tile.slice` is present.

## Examples

### Slice folded into `tile.extract`

The offset-0 full-shape slice [`FlattenTileNdTo2D`](13-flatten_tile_nd_to_2d.md) emits for a leading-dim-1 batch operand:

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

### Vec slice materialized into a `tile.col_expand_mul` operand (#1640)

A dynamic-offset slice of a local tile feeding `col_expand_mul` (the same rewrite applies to `col_expand_add`):

**Before**:

```python
row:    pl.Tile[[1, 256], pl.FP32, pl.Mem.Vec] = pl.tile.slice(local, [1, 256], [row_off, 0])
scaled: pl.Tile[[1, 256], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(row, gamma_t)
```

**After** (slice dropped; the operand is materialized into a fresh, non-aliasing tile):

```python
row_ext: pl.Tile[[1, 256], pl.FP32, pl.Mem.Vec] = pl.tile.extract(
    local, row_off, 0, shape=[1, 256], target_memory=pl.Mem.Vec)
scaled:  pl.Tile[[1, 256], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(row_ext, gamma_t)
```

### Column slice of a multi-row Vec tile (#2010)

`t[:, 64:128]` on a `[16, 128]` tile is a *static*-offset slice, but its window is not contiguous — 16 rows, 64 of the source's 128 columns — so it is materialized too:

**Before**:

```python
hi:     pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.slice(t, [16, 64], [0, 64])
scaled: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(hi, gamma_t)
```

**After**:

```python
hi_ext: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.extract(
    t, 0, 64, shape=[16, 64], target_memory=pl.Mem.Vec)
scaled: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(hi_ext, gamma_t)
```

Without the rewrite, `hi` allocates a dense `[16, 64]` buffer at `t + 256 B` — inside `t` — and the lazy `pto.textract` repacks `t`'s strided columns into it, overwriting `t` as it reads it. Only row 0 comes back correct, which is why the same construct is harmless on a single-row tile.

## Implementation

**Header**: `include/pypto/ir/transforms/passes.h`

**Properties**: `include/pypto/ir/transforms/pass_properties.h` (`kCanonicalizeTileSliceProperties`)

**Implementation**: `src/ir/transforms/canonicalize_tile_slice_pass.cpp`

**Python binding**: `python/bindings/modules/passes.cpp`

**Tests**: `tests/ut/ir/transforms/test_canonicalize_tile_slice.py`

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
| Dynamic-offset Vec `tile.slice` (3-arg) feeding a `tile.col_expand_*` op | Replaced by `tile.extract(target_memory=Vec)`; slice dropped (#1640 — the address falls back to the bare source base) |
| Static-offset **non-contiguous** Vec `tile.slice` (multi-row *and* narrower than its base, e.g. `t[:, a:b]`) feeding a `tile.col_expand_*` op | Replaced by `tile.extract(target_memory=Vec)`; slice dropped (#2010 — the dense repack would run on top of its own live source) |
| Static-offset **contiguous** Vec `tile.slice` (single row, or full source width) feeding a `tile.col_expand_*` op | Untouched (the lazy textract is a safe identity copy; keeps sharing the source buffer) |
| Chained Mat `tile.slice` (slice of a slice) | Peeled; offsets accumulated |
| `tile.slice` with `valid_shape` / `drop_dims` | Skipped (not a plain window). If such a slice *also* fails either identity-copy condition above — a dynamic offset (e.g. a rank-reducing `t[i]`) or a non-contiguous window — while feeding a col-expand op, codegen rejects it with an `INTERNAL_CHECK` rather than emitting the source-corrupting materialization |
| Other Vec/Left/Right/Acc-resident `tile.slice` | Untouched (no matching consumer) |
| Functions with no canonical `tile.slice` | Returned unchanged |

## See also

- [`FlattenTileNdTo2D`](13-flatten_tile_nd_to_2d.md) — upstream pass; emits the Mat-resident batch-page `tile.slice` this pass lowers
- [`AutoTileMatmulL0`](14-auto_tile_matmul_l0.md) — upstream pass; emits the `tile.extract`s that consume the batch-page slices
