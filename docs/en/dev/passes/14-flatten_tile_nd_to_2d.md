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
- All tile dimensions must be static (`ConstInt`)
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

1. **Validate preconditions**: Check static shapes, last-axis reduction, no `tile.read`/`tile.write`/`tile.slice` on >2D
2. **Transform statements**: Walk function body and convert >2D tile ops to 2D

Per-statement handling:

| Tile op | Transformation |
| ------- | -------------- |
| `tile.load` (>2D) | Change result type to 2D directly (load produces a 2D tile from a rank>2 tensor window) |
| `tile.store` (rank>2 tensor) | Inject the original tensor-rank partition `shapes` as an extra 4th operand in the transformed IR so backend codegen can reconstruct the `partition_view`; the DSL source is unchanged |
| `tile.store` (2D tensor) | Pass through unchanged |
| `tile.create`/`tile.full` (>2D) | Rebuild with flattened 2D shape directly |
| `tile.sum`/`tile.max`/`tile.min` (>2D) | Remap axis to 1 (last axis of 2D) |
| `tile.batch_matmul` | Expand to per-batch 2D `tile.matmul`, honoring batch broadcast and any operand-side transpose carried in the producer `tile.load(target_memory=Mat, transpose=True)` |
| `tile.batch_matmul_acc` | Expand to per-batch 2D `tile.matmul_acc`, slicing the (already-flattened) accumulator per batch index; an explicit `tile.move(target_memory=Acc)` is inserted when the accumulator is in another memory space |
| Other tile ops (>2D) | Substitute vars, re-create with 2D types |
| 1D/2D tile ops | Unchanged |

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

## Implementation

**Header**: `include/pypto/ir/transforms/passes.h`

**Implementation**: `src/ir/transforms/flatten_tile_nd_to_2d_pass.cpp`

**Python binding**: `python/bindings/modules/passes.cpp`

**Tests**: `tests/ut/ir/transforms/test_flatten_tile_nd_to_2d.py`

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
