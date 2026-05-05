# ConvertTensorToTileOps Pass

Converts tensor operations to tile operations in InCore functions and updates orchestration call sites.

## Overview

After `OutlineIncoreScopes` extracts InCore scopes into separate functions, those functions still operate on `TensorType` variables using `tensor.*` operations. This pass lowers them to `TileType` variables with `tile.*` operations that map directly to PTO-ISA instructions.

The pass also updates call sites in orchestration/opaque functions: for each new output parameter added to an InCore function, a `tensor.create` is inserted at the call site.

**Requirements**:

- Input IR must be in SSA form
- InCore scopes must be outlined (run `OutlineIncoreScopes` first)
- Statement structure must be normalized

**When to use**: Run after `OutlineClusterScopes` and before `OptimizeOrchTensors`.

## API

| C++ | Python | Level |
| --- | ------ | ----- |
| `pass::ConvertTensorToTileOps()` | `passes.convert_tensor_to_tile_ops()` | Program-level |

**Python usage**:

```python
from pypto.pypto_core import passes

convert_pass = passes.convert_tensor_to_tile_ops()
program_tiled = convert_pass(program)
```

## Algorithm

The pass operates in three program-level phases:

### Phase 1: Transform InCore Functions

For each `FunctionType::InCore` function:

1. **Pre-scan MatmulSlice patterns**: Collect `tensor.slice` results consumed by `tensor.matmul` / `tensor.matmul_acc`. These need `tile.load(Mat, transpose=...)` instead of the default `tile.load(Vec)`.

2. **Insert tile.load (entry loads)**: For each `TensorType` parameter directly consumed by a converted op, insert `tile.load(param, zeros, shape, shape, target_memory=Vec, transpose=False)` at function entry. Parameters only referenced by self-loading ops (`tensor.slice`, `tensor.matmul`, `tensor.read`, `tensor.write`, `tensor.assemble`) are skipped — they manage their own loads.

3. **Convert body via TensorToTileMutator**: Walk the function body and convert each `tensor.*` call to its `tile.*` equivalent using `OpConversionRegistry`. The mutator propagates type changes through control flow (IterArgs, ForStmt/WhileStmt return_vars, IfStmt return_vars).

4. **Insert tile.store (exit stores)**: For each return value converted from `TensorType` to `TileType`, add an `Out` parameter and insert `tile.store(tile, zeros, out_param)`. If the return value comes from a `tile.assemble` loop, the loop is rewritten to use `tile.store` directly (conversion-time assemble-loop rewrite; distinct from `OptimizeOrchTensors` Pattern 3 which handles cross-function optimization).

### Phase 2a: Propagate Added Outputs Through Spmd/Group Wrappers

`OutlineClusterScopes` produces Spmd/Group wrappers that are transparent 1:1
forwarders of their params to a single inner InCore call. When Phase 1 appends
`Out` params to that InCore callee, the wrapper must mirror the appended params
on its own signature and forward them through the inner call — otherwise
orchestration codegen's `BuildWrapperReorderedParams` invariant (every inner-call
`Var` arg resolves to a wrapper param) breaks.

For each `FunctionType::Spmd` / `FunctionType::Group` function:

1. `ForwardedCallFinder` locates the first call to a transformed InCore (one
   whose Phase 1 added at least one `Out` param).
2. If found, the wrapper signature is extended with matching `Out` params (same
   type as the InCore's appended params, reusing the `name_hint_`), and
   `WrapperForwardMutator` rewrites the inner call to append the new vars as
   forward args and adopt the callee's new return type. `tensor.create` is
   *not* synthesised in the wrapper — allocation remains the caller's
   responsibility.
3. If no forwarded transformed-InCore call is found, the wrapper is left
   unchanged.

### Phase 2b: Update Orchestration Call Sites

For each orchestration / opaque function that calls a transformed InCore
function or a wrapper that absorbed output params in Phase 2a:

1. Insert `tensor.create` for each added output parameter
2. Append created tensors as extra arguments to the call

InCore, Spmd, and Group functions are skipped from this phase — they were
already rewritten in Phase 1 / 2a.

## MatmulSlice Pattern

When `tensor.slice` feeds into `tensor.matmul` or `tensor.matmul_acc`, the slice must produce a Mat-space tile instead of a Vec-space tile. The pass pre-scans for this pattern and emits `tile.load(Mat, transpose=...)` with the transpose flag from the matmul kwargs (`a_trans` for LHS, `b_trans` for RHS).

## Example

**Before**:

```python
@pl.program
class Before:
    @pl.function(type=pl.FunctionType.InCore)
    def main_incore_0(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
        return y

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        y: pl.Tensor[[64], pl.FP32] = self.main_incore_0(x)
        return y
```

**After**:

```python
@pl.program
class After:
    @pl.function(type=pl.FunctionType.InCore)
    def main_incore_0(
        self, x: pl.Tensor[[64], pl.FP32],
        ret0_out: pl.Out[pl.Tensor[[64], pl.FP32]]
    ) -> pl.Tensor[[64], pl.FP32]:
        x_tile: pl.Tile[[64], pl.FP32] = pl.load(x, (0,), (64,))
        y_tile: pl.Tile[[64], pl.FP32] = pl.tile.add(x_tile, x_tile)
        ret0_store: pl.Tensor[[64], pl.FP32] = pl.store(y_tile, (0,), ret0_out)
        return ret0_store

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        ret0_out: pl.Tensor[[64], pl.FP32] = pl.tensor.create((64,), dtype=pl.FP32)
        y: pl.Tensor[[64], pl.FP32] = self.main_incore_0(x, ret0_out)
        return y
```

Key changes:

- `pl.add(x, x)` → `pl.tile.add(x_tile, x_tile)` (op conversion)
- `tile.load` inserted at entry, `tile.store` at exit
- `Out` parameter `ret0_out` added to InCore function
- `tensor.create` inserted at orchestration call site

## Implementation

**Header**: `include/pypto/ir/transforms/passes.h`

**Implementation**: `src/ir/transforms/convert_tensor_to_tile_ops_pass.cpp`

**Python binding**: `python/bindings/modules/passes.cpp`

**Tests**: `tests/ut/ir/transforms/test_convert_tensor_to_tile_ops.py`

## Pass Properties

| Property | Value |
| -------- | ----- |
| Required | SSAForm, SplitIncoreOrch, NormalizedStmtStructure |
| Produced | SSAForm, IncoreTileOps, NormalizedStmtStructure |
| Invalidated | — |

## Key Components

| Component | Role |
| --------- | ---- |
| `TensorArgsInConvertedOpsCollector` | IRVisitor — identifies tensor params needing entry loads |
| `MatmulSlicePatternCollector` | IRVisitor — finds slice→matmul patterns for Mat-space loads |
| `TypePropagatingMutator` | Base IRMutator — propagates type changes through control flow |
| `TensorToTileMutator` | IRMutator — converts tensor ops to tile ops via OpConversionRegistry |
| `ForwardedCallFinder` | IRVisitor — locates the wrapper's call into a transformed InCore (Phase 2a) |
| `WrapperForwardMutator` | IRMutator — appends new Out args to the wrapper's inner call (Phase 2a) |
| `CallSiteUpdateMutator` | IRMutator — inserts tensor.create at orchestration call sites (Phase 2b) |
| `IncoreTileOpsVerifier` | IRVisitor — verifies no TensorType ops remain in InCore functions |

## Scope

| Function type | Action |
| ------------- | ------ |
| InCore | Converted (tensor ops → tile ops); Phase 1 may append `Out` params |
| Spmd / Group (forwarding to a transformed InCore) | Signature mirrors the InCore's new `Out` params; inner call forwards them (Phase 2a) |
| Spmd / Group (no transformed-InCore forwarding) | Unchanged |
| Orchestration / Opaque | Call sites updated — `tensor.create` inserted for each new `Out` param (Phase 2b) |
