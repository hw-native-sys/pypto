# ResolveTransposeLayout Pass

Annotates InCore tensor parameters that source `tile.load(..., transpose=True)` with the `DN` (column-major) layout.

## Overview

When a `tile.load` is issued with `transpose=True`, PTO codegen needs the source tensor to be materialized in column-major (`DN`) layout — the transpose is realized by the layout choice rather than by reshaping data. This pass propagates that layout requirement back from the load site to the function parameter type, so that downstream passes and codegen can rely on the parameter's `TensorType` as the single source of truth for layout.

The pass annotates the parameter only — **shape is preserved**. `DN` is a layout/codegen hint; the logical tensor dimensions are not swapped. (This is the invariant enforced by the regression test for #606: a partial transpose load on `[128, 128]` must keep the parameter shape at `[128, 128]`, not the load-window shape.)

**Requirements**:

- Input IR must be in SSA form
- InCore functions must already be split out (`SplitIncoreOrch`)
- Tile ops must be present and 2D (`IncoreTileOps`, `TileOps2D`)
- Annotated tensor parameters must have rank ≥ 2

**When to use**: Run as the 15th pass in the `Default` strategy, after `InferTileMemorySpace` and before `ResolveBackendOpLayouts`. The 2D shape produced by `FlattenTileNdTo2D` is a precondition.

## API

| C++ | Python | Level |
| --- | ------ | ----- |
| `pass::ResolveTransposeLayout()` | `passes.resolve_transpose_layout()` | Program-level |

**Python usage**:

```python
from pypto.pypto_core import passes

resolve_pass = passes.resolve_transpose_layout()
program_dn = resolve_pass(program)
```

## Algorithm

For each function in the program:

1. **Skip non-InCore functions**: Orchestration and Opaque functions are returned unchanged. Only InCore-type functions (InCore, AIC, AIV) are processed.
2. **Scan body for transposed loads**: walk the function body and collect, for each `tile.load` call whose kwarg `transpose=True` and whose first argument is one of the function's parameters, the index of that parameter. Duplicates across multiple load sites are deduplicated.
3. **Rewrite parameters**: for each collected parameter:
   - **Skip if already DN**: if the parameter's `TensorType` already carries `TensorView{layout=DN}`, no rewrite is needed (idempotent).
   - **Require rank ≥ 2**: a 1D tensor cannot meaningfully be column-major; the pass aborts with a `CHECK` if it sees one.
   - Build a new `Var` with the same `name_hint`, span, and shape, but with a new `TensorType` whose `tensor_view_` is `TensorView({}, TensorLayout::DN)`.
4. **Substitute**: rewrite all uses of the old `Var` inside the function body via `Substitute`, then rebuild the function via `MutableCopy` with the new parameter list and body.

No Orchestration-side rewrite happens. Downstream passes and codegen consume the InCore signature as the layout source of truth.

| Behavior | Trigger |
| -------- | ------- |
| Annotate param with `DN` | InCore function param is the source of `tile.load(..., transpose=True)` |
| Skip param | Already `DN`, or no transposed load reaches it |
| Skip whole function | Function is Orchestration or Opaque |
| `CHECK` failure | Annotated param is not a `TensorType`, or rank < 2 |

## Example

**Before**:

```python
@pl.program
class Before:
    @pl.function(type=pl.FunctionType.InCore)
    def matmul_incore(
        self,
        a: pl.Tensor[[64, 128], pl.FP32],
        b: pl.Tensor[[32, 128], pl.FP32],
        c: pl.Out[pl.Tensor[[64, 32], pl.FP32]],
    ) -> pl.Tensor[[64, 32], pl.FP32]:
        tile_a = pl.load(a, [0, 0], [64, 128], target_memory=pl.MemorySpace.Mat)
        tile_b = pl.load(b, [0, 0], [32, 128], target_memory=pl.MemorySpace.Mat, transpose=True)
        tile_a_l0a = pl.move(tile_a, target_memory=pl.MemorySpace.Left)
        tile_b_l0b = pl.move(tile_b, target_memory=pl.MemorySpace.Right)
        tile_c = pl.matmul(tile_a_l0a, tile_b_l0b)
        c_store = pl.store(tile_c, [0, 0], c)
        return c_store

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self, a: pl.Tensor[[64, 128], pl.FP32], b: pl.Tensor[[32, 128], pl.FP32]
    ) -> pl.Tensor[[64, 32], pl.FP32]:
        c: pl.Tensor[[64, 32], pl.FP32] = pl.create_tensor([64, 32], dtype=pl.FP32)
        c_result = self.matmul_incore(a, b, c)
        return c_result
```

**After**:

```python
@pl.program
class After:
    @pl.function(type=pl.FunctionType.InCore)
    def matmul_incore(
        self,
        a: pl.Tensor[[64, 128], pl.FP32],
        b: pl.Tensor[[32, 128], pl.FP32, pl.DN],
        c: pl.Out[pl.Tensor[[64, 32], pl.FP32]],
    ) -> pl.Tensor[[64, 32], pl.FP32]:
        tile_a = pl.load(a, [0, 0], [64, 128], target_memory=pl.MemorySpace.Mat)
        tile_b = pl.load(b, [0, 0], [32, 128], target_memory=pl.MemorySpace.Mat, transpose=True)
        tile_a_l0a = pl.move(tile_a, target_memory=pl.MemorySpace.Left)
        tile_b_l0b = pl.move(tile_b, target_memory=pl.MemorySpace.Right)
        tile_c = pl.matmul(tile_a_l0a, tile_b_l0b)
        c_store = pl.store(tile_c, [0, 0], c)
        return c_store

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self, a: pl.Tensor[[64, 128], pl.FP32], b: pl.Tensor[[32, 128], pl.FP32]
    ) -> pl.Tensor[[64, 32], pl.FP32]:
        c: pl.Tensor[[64, 32], pl.FP32] = pl.create_tensor([64, 32], dtype=pl.FP32)
        c_result = self.matmul_incore(a, b, c)
        return c_result
```

`b` is the source of a `tile.load` with `transpose=True`, so the InCore parameter type gains the `pl.DN` layout annotation. The shape `[32, 128]` is unchanged. `a` is loaded without transpose, so it is left alone. The Orchestration `orchestrator` signature is **not** rewritten.

## Implementation

**Header**: `include/pypto/ir/transforms/passes.h`

**Implementation**: `src/ir/transforms/resolve_transpose_layout_pass.cpp`

**Python binding**: `python/bindings/modules/passes.cpp`

**Tests**: `tests/ut/ir/transforms/test_resolve_transpose_layout_pass.py`

## Pass Properties

| Property | Value |
| -------- | ----- |
| Required | SSAForm, IncoreTileOps, SplitIncoreOrch, TileOps2D |
| Produced | SSAForm, IncoreTileOps, SplitIncoreOrch, TileOps2D |
| Invalidated | — |

The pass preserves all input properties: it only rewrites tensor parameter type annotations, not statement structure or SSA form.

## Scope

| Function type | Action |
| ------------- | ------ |
| InCore (InCore, AIC, AIV) | Scanned and possibly rewritten |
| Orchestration | Unchanged |
| Opaque | Unchanged |

| Parameter state | Action |
| --------------- | ------ |
| Sourced by `tile.load(..., transpose=True)`, layout != DN, rank ≥ 2 | Rewritten to add `DN` |
| Sourced by `tile.load(..., transpose=True)`, layout already DN | Unchanged (idempotent) |
| Not sourced by any transposed load | Unchanged |
| Rank < 2 candidate | `CHECK` failure |

The pass is a no-op when no InCore function contains a `tile.load(..., transpose=True)` whose source is a parameter (verified by the `TestResolveTransposeLayoutNoOp` test class).

## Interaction with `tensor.transpose` at Orchestration

`tensor.transpose` at the Orchestration layer (issue #1209) deduces a result type that toggles `layout` between `ND` and `DN` for trailing-two-dim swaps (in `DeduceTensorTransposeType`). When the transposed tensor flows into an InCore parameter, that parameter inherits the `DN` tag and matches the same kernel-side codegen path this pass produces — so a downstream `tile.load(..., target_memory=Mat, transpose=True)` (the matmul B^T pattern) consumes the orchestration-side transpose for free, no explicit `pl.DN` annotation required.

**Vec target is a known limitation.** Cross-layout `TLOAD(VecTile, GlobalTensor)` is rejected by PTO ISA (only `ND→ND` / `DN→DN` / `NZ→NZ` are supported), and `tile.load` further restricts `transpose=True` to `target_memory=Mat`. Consequently, `pl.transpose` at orchestration followed by a Vec-target consumer (e.g. `pl.slice` inside a non-matmul `pl.at(level=CORE_GROUP)` block) can compile its IR correctly but will fail PTO ISA compilation on Ascend 910B. Workarounds: route the load through a Mat tile (matmul-style), perform an explicit incore `tile.transpose` after the load, or materialize a contiguous transposed copy at orchestration.
