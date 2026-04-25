# OptimizeOrchTensors Pass

Optimizes tensor buffer usage across orchestration and InCore functions by eliminating redundant allocations and improving data flow.

## Overview

After `ConvertTensorToTileOps`, orchestration functions allocate output tensors (`tensor.create`) at every InCore call site, even inside loops where the same buffer could be reused. This pass applies three optimization patterns to reduce allocations and improve buffer layout information.

**Requirements**:

- Input IR must have InCore scopes outlined with tile ops (run `ConvertTensorToTileOps` first)

**When to use**: Run immediately after `ConvertTensorToTileOps` and before `FlattenTileNdTo2D`.

## API

| C++ | Python | Level |
| --- | ------ | ----- |
| `pass::OptimizeOrchTensors()` | `passes.optimize_orch_tensors()` | Program-level |

**Python usage**:

```python
from pypto.pypto_core import passes

opt_pass = passes.optimize_orch_tensors()
program_opt = opt_pass(program)
```

## Patterns

The pass applies four patterns in sequence. Each pattern sees the results of the previous one.

### Pattern 1: Iter-Arg Reuse (IterArgReuseOptimizer)

**Problem**: Inside a `for`/`while` loop, each iteration allocates a new output tensor via `tensor.create`, even though the InCore result feeds back as an iter-arg to the next iteration.

**Solution**: Merge the `Out` param into the corresponding `In` param (promoted to `InOut`), remove the `tensor.create`, and redirect `tile.store` to write into the reused buffer.

**Before**:

```python
for i in pl.range(N, init_values=[init_buf]):
    out: pl.Tensor = pl.tensor.create(shape, dtype=pl.FP32)  # redundant alloc
    result: pl.Tensor = self.incore_fn(iter_arg, out)          # In + Out params
    pl.yield_(result)
```

**After**:

```python
for i in pl.range(N, init_values=[init_buf]):
    result: pl.Tensor = self.incore_fn(iter_arg)  # InOut param (reuses iter-arg buffer)
    pl.yield_(result)
```

### Pattern 2: Assemble Parent Strides (AssembleParentStridesOptimizer)

**Problem**: When orchestration scatters InCore results into a larger tensor via `tensor.assemble`, the InCore function's `tile.store` doesn't know the parent tensor's strides, which can lead to suboptimal memory layout.

**Solution**: Analyze `tensor.assemble(parent, incore_result, offset)` patterns in orchestration. Attach the parent tensor's shape as `TensorView` strides on the InCore function's `Out` param type, so `tile.store` can use the correct memory layout.

### Pattern 4: Slice Input Strides (SliceInputStridesOptimizer)

**Problem**: When orchestration passes a sliced tensor (`tensor.slice`) as an `In` argument to an InCore function, the InCore function's parameter has contiguous strides (computed from its own shape), not the parent tensor's strides. This causes incorrect memory access when the slice is a non-contiguous view of the parent.

**Solution**: Analyze `tensor.slice(parent, size, offset)` patterns in orchestration. When a slice result is passed as an `In` argument to an InCore call, attach the parent tensor's shape-derived strides via `TensorView` on the InCore function's `In` param type, so `tile.load` uses the correct memory layout.

### Pattern 3: Assemble-Loop Rewrite (AssembleLoopRewriter)

**Problem**: An InCore function contains a `for` loop that accumulates results via `tile.assemble` into an iter-arg, then stores the final result. The `tile.assemble` creates intermediate tile copies each iteration.

**Solution**: Rewrite the loop body to use `tile.store` directly (writing into the `Out` param), initializing the iter-arg from the `Out` param instead of a `tile.create`.

## Example (Pattern 1)

**Before**:

```python
@pl.program
class Before:
    @pl.function(type=pl.FunctionType.InCore)
    def compute(self, x: pl.Tensor[[64], pl.FP32],
                out_0: pl.Out[pl.Tensor[[64], pl.FP32]]) -> pl.Tensor[[64], pl.FP32]:
        x_tile = pl.load(x, (0,), (64,))
        y_tile = pl.tile.add(x_tile, x_tile)
        ret = pl.store(y_tile, (0,), out_0)
        return ret

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, buf: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        for i in pl.range(10, init_values=[buf]):
            out_0 = pl.tensor.create((64,), dtype=pl.FP32)
            result = self.compute(iter_arg, out_0)
            pl.yield_(result)
        return loop_result
```

**After** (Pattern 1 merges Out into In as InOut):

```python
@pl.program
class After:
    @pl.function(type=pl.FunctionType.InCore)
    def compute(self, x: pl.InOut[pl.Tensor[[64], pl.FP32]]) -> pl.Tensor[[64], pl.FP32]:
        x_tile = pl.load(x, (0,), (64,))
        y_tile = pl.tile.add(x_tile, x_tile)
        ret = pl.store(y_tile, (0,), x)
        return ret

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, buf: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        for i in pl.range(10, init_values=[buf]):
            result = self.compute(iter_arg)
            pl.yield_(result)
        return loop_result
```

The `tensor.create` is eliminated; the iter-arg buffer is reused across iterations.

## Implementation

**Header**: `include/pypto/ir/transforms/passes.h`

**Implementation**: `src/ir/transforms/optimize_orch_tensors_pass.cpp`

**Python binding**: `python/bindings/modules/passes.cpp`

**Tests**: `tests/ut/ir/transforms/test_optimize_orch_tensors.py`

## Pass Properties

| Property | Value |
| -------- | ----- |
| Required | SplitIncoreOrch, IncoreTileOps |
| Produced | SplitIncoreOrch, IncoreTileOps |
| Invalidated | — |

## Key Components

| Component | Role |
| --------- | ---- |
| `IterArgReuseOptimizer` | Pattern 1 — merges Out params into In params for loop-carried buffers |
| `AssembleParentStridesOptimizer` | Pattern 2 — attaches parent strides via TensorView |
| `SliceInputStridesOptimizer` | Pattern 4 — attaches parent strides to In params via TensorView for slice patterns |
| `AssembleLoopRewriter` | Pattern 3 — rewrites tile.assemble loops to tile.store loops |
| `BuildOutParamReturnMappings` | Shared helper — maps Out params to return indices via tile.store |
| `ComputeRowMajorStrides` | Shared helper — computes row-major strides from a shape |

## Scope

| Function type | Action |
| ------------- | ------ |
| InCore | Params/body rewritten (Patterns 1, 3, 4) |
| Orchestration / Opaque | Call sites rewritten (Patterns 1, 2) |
| Group | Unchanged |
