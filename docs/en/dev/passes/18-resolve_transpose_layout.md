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

`tensor.transpose` at the Orchestration layer (issue #1209) records two complementary pieces of information on its result type (in `DeduceTensorTransposeType`):

1. **Layout tag.** For canonical trailing-two-dim swaps, `layout` toggles between `ND` and `DN`. PTOAS reads this tag to validate kernel boundaries.
2. **Explicit physical strides.** The runtime `Tensor::transpose` is a metadata-only swap of `shapes` / `offsets` — the underlying GM bytes stay in the source's row-major layout. So the post-transpose view's physical strides are the input's strides reordered at `(axis1, axis2)`. `DeduceTensorTransposeType` builds row-major strides over the input shape (`MakeIndexMul` collapses ConstInt chains, so static shapes get plain ConstInt strides; dynamic shapes get symbolic ones) and swaps them at the same axes as the shape. Non-trailing transposes are also supported via this path — they keep `layout = ND` and rely solely on the strides.

Codegen disambiguates the two callers of the DN tag by checking `tensor_view_->stride`:

| Source of DN | `stride` | `EmitMakeTensorViews` / partition emit |
| ------------ | -------- | -------------------------------------- |
| This pass (`tile.load(transpose=True)`) | empty | implicit "swap last two dims" — emit `[N, M]` shape with `[1, M]` strides over the IR shape `[M, N]` |
| `tensor.transpose` | non-empty | skip the implicit swap — emit IR shape directly with the recorded strides |

The codegen rule above lets *non-transposed* downstream consumers (e.g. `tile.load(..., transpose=False)`) read a DN-tagged `tensor.transpose` result via the explicit-stride path without re-applying the implicit shape swap.

**`tile.load(transpose=True)` directly on a `tensor.transpose` result is not yet supported.** That specific combination — an InCore param that arrives carrying both explicit physical strides AND `layout = DN` (the unique signature of a `tensor.transpose` result) — is rejected by this pass with a `CHECK` failure, because the two encodings would compose as a double transpose at codegen time and emit wrong addresses. Slice-derived inputs (explicit strides + `layout = ND`, attached by `OptimizeOrchTensors`) are unaffected and continue to flow through the standard "promote ND → DN, drop strides" path used by patterns like paged_attention's matmul B^T. Reconciling explicit physical strides with the pass's DN-only output convention requires a separate design — see follow-up tracked from issue #1209. Workaround for the rejected case: do the transpose at the tile level via `tile.load(transpose=True)` directly on the source tensor instead of via orchestration `tensor.transpose`.

**Vec target is supported on a5 only.** Cross-layout `TLOAD(VecTile_RowMajor, GlobalTensor<DN>)` is rejected by PTOAS on a2a3 (the static assertion is `TLOAD(VecTile, GlobalTensor) only support ND2ND/DN2DN/NZ2NZ`), and `tile.load` further restricts `transpose=True` to `target_memory=Mat`. a5 (and the `a5sim` simulator) lifts this restriction, so a `pl.transpose` followed by a Vec-target consumer (e.g. `pl.slice` inside a non-matmul `pl.at(level=CORE_GROUP)` block) compiles and runs correctly there. The regression test `tests/st/runtime/test_trans.py::test_transpose_slice_assemble[a5sim]` covers exactly this case. On a2a3, the same DSL pattern compiles to correct IR/.pto but fails at the kernel C++ stage; workarounds are unchanged from before — route the load through a Mat tile (matmul-style), perform an explicit `tile.transpose` after the load, or materialize a contiguous transposed copy at orchestration.
