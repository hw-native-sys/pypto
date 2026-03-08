# ExpandMixedKernel Pass

Expands mixed InCore functions into separate AIC (Cube) + AIV (Vector) kernels wrapped in a Group function.

## Overview

After `OutlineIncoreScopes` and `ConvertTensorToTileOps`, InCore functions may contain both Cube ops (`tile.matmul`, `tile.gemv`, etc.) and Vector ops (`tile.load`, `tile.add`, `tile.store`, etc.). These are **mixed InCore functions**. Hardware requires Cube and Vector operations to run on separate core types, so this pass splits them into:

- **AIC function** (`FunctionType::AIC`) — contains only Cube + shared ops
- **AIV function** (`FunctionType::AIV`) — contains only Vector + shared ops
- **Group function** (`FunctionType::Group`) — calls AIC then AIV, replaces the original

Cross-core data dependencies are bridged with `tpush_to_aiv`/`tpop_from_aic` (Cube→Vector) and `tpush_to_aic`/`tpop_from_aiv` (Vector→Cube) ops.

**Requirements**:

- Input IR must have tile ops (run `ConvertTensorToTileOps` first)
- Input IR must have InCore scopes outlined (run `OutlineIncoreScopes` first)

**When to use**: Run after `ConvertTensorToTileOps` when InCore functions may contain both Cube and Vector tile operations.

## API

| C++ | Python | Level |
| --- | ------ | ----- |
| `pass::ExpandMixedKernel()` | `passes.expand_mixed_kernel()` | Program-level |

**Python usage**:

```python
from pypto.pypto_core import passes

expand_pass = passes.expand_mixed_kernel()
program_expanded = expand_pass(program)
```

## Algorithm

```text
for each InCore function F in program:
  1. Recursively analyze affinity of all statements (including inside loops/conditionals)
  2. If not mixed (no CUBE ops or no VECTOR ops): skip
  3. Find cross-affinity boundaries (variables flowing between CUBE and VECTOR stmts)
  4. Build AIC body: keep CUBE + SHARED stmts, prune VECTOR, recurse into MIXED loops
     - Insert tpop_from_aiv before CUBE stmts using VECTOR-defined vars
     - Insert tpush_to_aiv after CUBE stmts defining vars used by VECTOR
  5. Build AIV body: symmetric (keep VECTOR + SHARED, prune CUBE)
  6. Run dead code elimination on both bodies (recursive into loops)
  7. Create AIC function (no return), AIV function (original return), Group function (calls both)
  8. Replace original InCore function with Group + AIC + AIV
```

**Affinity classification**:

| Affinity | Ops |
| -------- | --- |
| CUBE | `tile.matmul`, `tile.matmul_acc`, `tile.matmul_bias`, `tile.gemv`, `tile.gemv_acc`, `tile.gemv_bias`, `tile.batch_matmul` |
| VECTOR | All other `tile.*` ops (`tile.load`, `tile.store`, `tile.add`, `tile.exp`, etc.) |
| SHARED | Non-tile ops, function calls, control flow, scalar ops |
| MIXED | Compound statements (ForStmt, IfStmt) containing both CUBE and VECTOR children |

**Nested structure handling**: ForStmt, IfStmt, and WhileStmt containing mixed ops are duplicated into both AIC and AIV bodies with recursively pruned contents.

## Example

**Before**:

```python
@pl.program
class Before:
    @pl.function(type=pl.FunctionType.InCore)
    def compute_incore_0(self, x: pl.Tensor[[16, 128], pl.BF16],
                         y: pl.Tensor[[128, 128], pl.BF16],
                         out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]]
                         ) -> pl.Tensor[[16, 128], pl.FP32]:
        x_tile: pl.Tile[[16, 128], pl.BF16] = pl.load(x, [0, 0], [16, 128])
        y_tile: pl.Tile[[128, 128], pl.BF16] = pl.load(y, [0, 0], [128, 128])
        z_tile: pl.Tile[[16, 128], pl.FP32] = pl.matmul(x_tile, y_tile)
        out_0 = pl.store(z_tile, [0, 0], out_0)
        return out_0

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, x, y) -> pl.Tensor[[16, 128], pl.FP32]:
        out_0 = pl.create_tensor([16, 128], dtype=pl.FP32)
        return self.compute_incore_0(x, y, out_0)
```

**After** (conceptual — actual IR includes type annotations on all variables):

```python
@pl.program
class After:
    @pl.function(type=pl.FunctionType.AIC)
    def compute_incore_0_aic(self, x, y, out_0):
        x_tile: pl.Tile[[16, 128], pl.BF16] = pl.system.tpop_from_aiv(aiv_idx=0)   # receive from AIV
        y_tile: pl.Tile[[128, 128], pl.BF16] = pl.system.tpop_from_aiv(aiv_idx=0)   # receive from AIV
        z_tile = pl.matmul(x_tile, y_tile)
        pl.system.tpush_to_aiv(z_tile, aiv_idx=0)     # send to AIV

    @pl.function(type=pl.FunctionType.AIV)
    def compute_incore_0_aiv(self, x, y, out_0) -> pl.Tensor[[16, 128], pl.FP32]:
        x_tile = pl.load(x, [0, 0], [16, 128])
        pl.system.tpush_to_aic(x_tile, aiv_idx=0)     # send to AIC
        y_tile = pl.load(y, [0, 0], [128, 128])
        pl.system.tpush_to_aic(y_tile, aiv_idx=0)     # send to AIC
        z_tile: pl.Tile[[16, 128], pl.FP32] = pl.system.tpop_from_aic(aiv_idx=0)   # receive from AIC
        out_0 = pl.store(z_tile, [0, 0], out_0)
        return out_0

    @pl.function(type=pl.FunctionType.Group)
    def compute_incore_0(self, x, y, out_0) -> pl.Tensor[[16, 128], pl.FP32]:
        self.compute_incore_0_aic(x, y, out_0)
        result = self.compute_incore_0_aiv(x, y, out_0)
        return result

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, x, y) -> pl.Tensor[[16, 128], pl.FP32]:
        out_0 = pl.create_tensor([16, 128], dtype=pl.FP32)
        return self.compute_incore_0(x, y, out_0)  # calls Group (same name)
```

## Implementation

**Header**: `include/pypto/ir/transforms/passes.h`

**Implementation**: `src/ir/transforms/expand_mixed_kernel_pass.cpp`

**Python binding**: `python/bindings/modules/passes.cpp`

**Tests**: `tests/ut/ir/transforms/test_expand_mixed_kernel.py`

## Pass Properties

| Property | Value |
| -------- | ----- |
| Required | IncoreTileOps, SplitIncoreOrch |
| Produced | MixedKernelExpanded |
| Invalidated | — |

## Property Verifier

`MixedKernelExpandedPropertyVerifier` checks that no remaining `FunctionType::InCore` function contains both Cube and Vector tile ops. AIC/AIV/Group functions are not checked (they are already split by definition).

## Design Decisions

| Decision | Rationale |
| -------- | --------- |
| Op-name classification (not memory-space) | Pass runs before `InitMemRef`, so memory spaces aren't assigned yet |
| Group keeps original function name | Orchestration call sites work unchanged — no call-site rewriting needed |
| Parameters copied to all three functions | Simplifies wiring; DCE removes unused params in downstream passes |
| Recursive compound-stmt handling | Correctly splits mixed ops inside `ForStmt`, `IfStmt`, `WhileStmt` |
