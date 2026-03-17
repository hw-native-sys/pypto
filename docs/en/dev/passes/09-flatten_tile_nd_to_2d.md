# FlattenTileNdTo2D Pass

Flattens ND tile operations (3D+) to 2D in InCore functions by merging all dimensions except the last.

## Overview

PTO-ISA only accepts 2D tiles. After `ConvertTensorToTileOps`, tiles may be ND (matching tensor shapes). This pass flattens all >2D tile operations to 2D by merging higher axes into one dimension and keeping the last axis unchanged. For example, a tile `[2, 3, 4]` becomes `[6, 4]`.

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
| `tile.load` (>2D) | Keep load as-is, insert `tile.reshape` to 2D after |
| `tile.store` (tile covers full tensor or ND-tracked) | Insert `tile.reshape` back to ND before store |
| `tile.store` (2D slice of larger tensor) | Pass through unchanged |
| `tile.create`/`tile.full` (>2D) | Rebuild with flattened 2D shape directly |
| `tile.sum`/`tile.max`/`tile.min` (>2D) | Remap axis to 1 (last axis of 2D) |
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
        x_tile_nd: pl.Tile[[2, 3, 4], pl.FP32] = pl.load(x, [0, 0, 0], [2, 3, 4])
        x_tile: pl.Tile[[6, 4], pl.FP32] = pl.tile.reshape(x_tile_nd, [6, 4])
        y_tile: pl.Tile[[6, 4], pl.FP32] = pl.tile.add(x_tile, x_tile)
        y_tile_nd: pl.Tile[[2, 3, 4], pl.FP32] = pl.tile.reshape(y_tile, [2, 3, 4])
        out_0 = pl.store(y_tile_nd, [0, 0, 0], out_0)
        return out_0
```

The 3D tile `[2, 3, 4]` is flattened to `[6, 4]`. `tile.load` keeps its ND shape (interfacing with the tensor), and `tile.reshape` converts between ND and 2D. Before `tile.store`, the tile is reshaped back to ND.

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
