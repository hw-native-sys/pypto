# FlattenTileNdTo2D Pass

Flattens ND tile operations (3D+) to 2D in InCore functions by merging all dimensions except the last.

## Overview

PTO-ISA only accepts 2D tiles. After `ConvertTensorToTileOps`, tiles may have rank > 2 (matching tensor shapes). This pass flattens all >2D tile operations to 2D by merging higher axes into one dimension and keeping the last axis unchanged. For example, a tile `[2, 3, 4]` becomes `[6, 4]`.

For batched matrix multiplication, `ConvertTensorToTileOps` first preserves the
high-level intent as `tile.batch_matmul` (or `tile.batch_matmul_acc` when an
accumulator is involved). `FlattenTileNdTo2D` then becomes the canonical
legalization point that expands them into broadcast-aware per-batch
2D `tile.matmul` / `tile.matmul_acc` operations.

**Requirements**:

- Input IR must be in SSA form
- Input IR must have tile ops (run `ConvertTensorToTileOps` first)
- Every tile's **physical** shape must be static (`ConstInt`); a tile's `valid_shape` may be dynamic
  and is preserved through the flatten (see [Dynamic valid_shape](#dynamic-tile-dimensions-issue-1578))
- All tile reduce ops must reduce along the last axis
- All tile memory must be contiguous

**When to use**: Run after `ConvertTensorToTileOps` and before `ExpandMixedKernel` / `InitMemRef`.

## API

| C++ | Python | Level |
| --- | ------ | ----- |
| `pass::FlattenTileNdTo2D()` | `passes.flatten_tile_nd_to_2d()` | Function-level |

**Python usage**:

```python
from pypto.pypto_core import passes

flatten_pass = passes.flatten_tile_nd_to_2d()
program_2d = flatten_pass(program)
```

## Algorithm

For each InCore function (InCore, AIC, AIV):

1. **Validate preconditions**: Check static physical shapes, last-axis reduction, no `tile.read`/`tile.write`/`tile.slice` on >2D
2. **Transform statements**: Walk function body and convert >2D tile ops to 2D, preserving any dynamic `valid_shape` (see [Dynamic valid_shape](#dynamic-tile-dimensions-issue-1578))
3. **Verify postconditions**: The `TileOps2D` property verifier independently checks that the rewritten InCore IR contains only supported tile ranks and codegen-ready transpose forms

Per-statement handling:

| Tile op | Transformation |
| ------- | -------------- |
| `tile.load` (>2D) | Change result type to 2D directly (load produces a 2D tile from a rank>2 tensor window) |
| `tile.store` (rank>2 tensor) | Inject the original tensor-rank partition `shapes` as an extra 4th operand in the transformed IR so backend codegen can reconstruct the `partition_view`; the DSL source is unchanged. If the tile operand itself is still rank>2 (e.g. a user-written `tile.reshape` to 3D feeding `pl.assemble` into an N-D tensor view), insert a `tile.reshape` to flatten the tile operand to 2D first — the codegen requires a 2D tile while the original tile shape still flows through as the `shapes` partition operand |
| `tile.store` (2D tensor) | Pass through unchanged |
| `tile.create`/`tile.full` (>2D) | Rebuild with flattened 2D shape directly |
| `tile.sum`/`tile.max`/`tile.min` (>2D) | Remap axis to 1 (last axis of 2D) |
| `tile.transpose` | Sole owner of `pto.ttrans` scratch materialization. Arrives 3-arg (input, axis1, axis2). **2D**: create one scratch tile (shape = SOURCE page, in the input's memory space) and emit the codegen-ready 4-arg `tile.transpose(in, a1, a2, scratch)`. **>2D** (last-two-axes swap): unroll into per-batch 2D transposes, each a 4-arg form with scratch sliced from a flat `[batch*A, B]` pool, assembled into the merged 2D output. A batch-axis swap is a user error |
| `tile.batch_matmul` | Expand to per-batch 2D `tile.matmul`, honoring batch broadcast. A b_trans/a_trans operand arrives as a zero-copy `tile.transpose_view` over a natural load (no transpose-at-load, no copy); the tile-level op carries no transpose semantic. Each operand is handled identically (see operand handling below) |
| `tile.batch_matmul_acc` | Expand to per-batch 2D `tile.matmul_acc`, slicing the (already-flattened) accumulator per batch index. Memory-space decisions on the accumulator (Vec/Acc round-trips, retargetable producer promotion of an upstream `tile.create`, TileView refresh) are deferred to `InferTileMemorySpace` (pass 17) — flatten emits no inline `tile.move` |
| Other tile ops (>2D) | Substitute vars, re-create with 2D types |
| 1D/2D tile ops | Unchanged |

**Unified operand handling — whole-fit slice vs. per-batch load.** Every
batch_matmul operand (lhs or rhs, transposed or not, load- or move-sourced) is
treated identically. The routing is decided **per operand**: keep the whole tile
only when the operands' whole tiles fit Mat (L1) together (`BatchOperandsWholeFit`,
a capacity gate) **and** this operand's whole load collapses contiguously
(`WholeLoadContiguous`); otherwise re-emit it per batch.

- **whole (default):** the operand is brought whole into Mat once and
  per-batch **sliced** — a row slice for a plain (row-batched `[B*rows, cols]`)
  operand, a column slice for a `tile.transpose_view` (column-batched
  `[K, B*N]`) operand. A natural Mat load of a 3D `[B, N, K]` tensor keeps its ND
  source window here; the hardware ND2NZ "2-dim GlobalTensor" collapse to
  `[B*N, K]` is owned by the `tile.load` codegen — it fires when the load's result
  is an NZ Mat tile and emits the 2D `make_tensor_view` there — so this pass only
  flattens the load's **result tile** to 2D. A broadcast operand reuses its single
  page.
- **per batch** (the whole tile would overflow L1, **or** the whole load is
  non-contiguous): re-emit the operand from its underlying natural `tile.load`
  one batch at a time (a per-batch `[1, .., X, Y]` window → 2D `[X, Y]`, using the
  load's own window dims so a partial sub-tile re-emits correctly), with a
  per-batch `tile.transpose_view` when transposed. The dead whole load/view is
  then dropped.
  - *Non-contiguous* means a multi-batch load that also partially slices the
    matrix-row (middle) dim — e.g. `[2, K0<K, N]` from `[2, K, N]`. Flattened to
    `[2*K, N]` such a window has gaps between batches, so it cannot be one 2D
    ND2NZ load; per batch each page is `[1, K0, N]` (contiguous) and collapses
    cleanly. This routing keeps the codegen contiguity guard from ever firing on
    a batch_matmul operand.

**Dead-load elimination (per-batch only).** When an operand re-emits per-batch
loads (capacity !fit or non-contiguous), the original whole load/view becomes
dead and the pass drops it. The drop pre-scan applies the **same per-operand
routing** as `LowerBatchMatmul`, so a non-contiguous operand's chain is recognized
as per-batch here too. A chain is drop-eligible when **every** use is a
`tile.batch_matmul[_acc]` operand (the chain `tile.load → tile.transpose_view` is
walked back), and it is dropped only when **every** consuming matmul routes it
per-batch — a chain shared with any whole-kept matmul stays whole. Uses are
counted **recursively** (including nested `If`/`For`/`While`/`Scope` bodies) so a
load also consumed in a nested block is never dropped. The capacity gate is
backend-gated (no backend → reports fit), but the contiguity check is not, so the
non-contiguous routing fires in unit tests too.

> The per-batch V2C move case (a move-sourced operand that does not fit L1) is a
> deferred follow-up; such an operand currently stays on the whole-slice path,
> correct only while the moved tile fits the fixed cross-core ring.

## Example

**Before**:

```python
@pl.program
class Before:
    @pl.function(type=pl.FunctionType.InCore)
    def main_incore_0(self, x: pl.Tensor[[2, 3, 4], pl.FP32],
                      out_0: pl.Out[pl.Tensor[[2, 3, 4], pl.FP32]]) -> pl.Tensor[[2, 3, 4], pl.FP32]:
        x_tile: pl.Tile[[2, 3, 4], pl.FP32] = pl.load(x, [0, 0, 0], [2, 3, 4])
        y_tile: pl.Tile[[2, 3, 4], pl.FP32] = pl.tile.add(x_tile, x_tile)
        out_0 = pl.store(y_tile, [0, 0, 0], out_0)
        return out_0
```

**After**:

```python
@pl.program
class After:
    @pl.function(type=pl.FunctionType.InCore)
    def main_incore_0(self, x: pl.Tensor[[2, 3, 4], pl.FP32],
                      out_0: pl.Out[pl.Tensor[[2, 3, 4], pl.FP32]]) -> pl.Tensor[[2, 3, 4], pl.FP32]:
        x_tile: pl.Tile[[6, 4], pl.FP32] = pl.load(x, [0, 0, 0], [2, 3, 4])
        y_tile: pl.Tile[[6, 4], pl.FP32] = pl.tile.add(x_tile, x_tile)
        out_0 = pl.store(y_tile, [0, 0, 0], out_0)
        return out_0
```

The 3D tile `[2, 3, 4]` is flattened to `[6, 4]`. `tile.load` directly produces a 2D tile —
no `tile.reshape` is inserted. `tile.store` accepts the 2D tile and writes to the original rank>2 tensor. For
rank>2 tensors, the pass injects the original partition `shapes` as an extra 4th operand into the
transformed IR (e.g. `pl.store(y_tile, [0, 0, 0], out_0, (2, 3, 4))`); this operand is only
present in the transformed IR and is not part of the source DSL.

## Dynamic tile dimensions (issue #1578)

Hardware tiles map to fixed-size on-chip buffers, so every **physical** tile dimension must be a
compile-time constant; the runtime extent lives in `TileView.valid_shape`. To process a dynamic
dimension the user **writes the chunk loop themselves**: iterate the dynamic dim with `pl.range` in a
static `CHUNK` step, and load each chunk as a static physical `[1, CHUNK, 512]` tile whose
`valid_shapes` carries the runtime tail `min(CHUNK, s - c)`. The chunk size is the user's choice — it
strongly affects performance, so it is not auto-selected by the pass.

```python
# User-written: chunk the dynamic S dim, clamp the tail in valid_shapes.
for c, (o,) in pl.range(0, s_dim, CHUNK, init_values=(out,)):
    valid = pl.min(CHUNK, s_dim - c)
    t = pl.load(x, [b, c, 0], [1, CHUNK, 512], valid_shapes=[1, valid, 512])
    t = pl.cast(t, target_type=pl.FP32)
    o = pl.store(t, [b, c, 0], o)        # static physical [1, CHUNK, 512], dynamic valid
    pl.yield_(o)
```

Each per-chunk tile is physically `[1, CHUNK, 512]` (static) with a dynamic `valid_shape`
`[1, min(CHUNK, s - c), 512]`. **FlattenTileNdTo2D's only job here is to lower that >2D tile to
`[CHUNK, 512]` while preserving the dynamic `valid_shape`** — `ComputeMergedValidShape` merges the
leading dims of `valid_shape` the same way `ComputeMergedShape` merges the physical shape, but tolerates
dynamic entries, so the runtime tail survives the flatten instead of being reset to the full physical
shape. The loop itself is the user's; the pass does **not** synthesize it.

> The chunk must fit on-chip Vec (UB) memory (`CHUNK * <kept dims> * <live tile bytes> <= UB capacity`),
> otherwise `AllocateMemoryAddr` rejects the kernel with a "Vec buffer usage exceeds platform limit"
> error. Picking the chunk is the user's responsibility.

If a >2D tile reaches the pass with a **dynamic physical shape** (the user did not slice a static
chunk), it cannot be flattened and the pass raises an actionable error pointing to the two fixes:
chunk the dynamic dim with `pl.range`/`pl.parallel`, or reshape to 2D before the InCore (`pl.at`) scope.

## Implementation

**Header**: `include/pypto/ir/transforms/passes.h`

The implementation is split by responsibility:

| Phase | File | Responsibility |
| ----- | ---- | -------------- |
| Coordination | `src/ir/transforms/flatten_tile_nd_to_2d/pass.cpp` | Select InCore functions and sequence analysis before rewrite |
| Analysis | `src/ir/transforms/flatten_tile_nd_to_2d/analysis.cpp` | Read-only precondition validation |
| Rewrite | `src/ir/transforms/flatten_tile_nd_to_2d/rewrite.cpp` | ND-to-2D IR transformation |
| Verification | `src/ir/transforms/flatten_tile_nd_to_2d/verification.cpp` | Independent `TileOps2D` postcondition verification |

The phase entry points are private to the transform implementation; the public API remains `pass::FlattenTileNdTo2D()`.

**Python binding**: `python/bindings/modules/passes.cpp`

**Tests**: `tests/ut/ir/transforms/test_flatten_tile_nd_to_2d.py`, `tests/st/codegen/dsl/test_flatten_dynamic_tile_3d.py` (issue #1578 end-to-end)

## Pass Properties

| Property | Value |
| -------- | ----- |
| Required | SSAForm, IncoreTileOps |
| Produced | SSAForm, TileOps2D |
| Invalidated | — |

## Scope

| Tile rank | Action |
| --------- | ------ |
| 1D | Unchanged |
| 2D | Unchanged |
| 3D+ | Flattened to 2D |

Only InCore-type functions (InCore, AIC, AIV) are processed. Orchestration and Opaque functions are returned unchanged.
