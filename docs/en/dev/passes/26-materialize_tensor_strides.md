# MaterializeTensorStrides Pass

Fills every `view.has_value() && view.stride.empty()` slot on every `TensorType` reachable from the program with the packed canonical stride for the carried layout (per RFC #1300 §2.4). After this pass runs, the codegen-entry contract holds: every `TensorView` that exists has explicit stride matching its layout / shape, and the strict-mode `TensorViewCanonical` verifier accepts the IR.

> **Status**: this pass is registered (`passes.materialize_tensor_strides()`) and fully tested in isolation, but it is intentionally **not yet inserted into the default pipeline**. Materializing DN strides per the canonical (logical-shape) formula conflicts with the legacy "source shape + post-emit swap" path still living in PTO codegen (`get_shape_source_idx` / `dn_swap`). The pipeline insertion will land alongside the codegen cleanup in RFC #1300 P6/P7. The `pass_manager.py` insertion site carries a comment pointer for the next phase.

## Overview

PyPTO's IR allows two equivalent forms for `TensorType.tensor_view_`:

- **Implicit** — `view.has_value() && view.stride.empty()`: the layout tag is set (e.g. `DN`) but the per-dimension stride is left blank. Downstream consumers must treat empty stride as "use the packed canonical stride for this layout."
- **Explicit** — every dimension has its `ExprPtr` stride spelled out.

Codegen needs one machine-readable contract, so `MaterializeTensorStrides` walks the program and rewrites every implicit `TensorView` into its explicit packed canonical form using `BuildLogicalStridesFromLayout` from `tensor_view_semantics.h`. Bare `TensorType`s (`!view.has_value()`) are left untouched: the `TensorViewCanonical` verifier accepts them in both modes as implicitly ND-packed and the bare form is unambiguous.

**Requirements**:

- `SSAForm`, `SplitIncoreOrch`, `IncoreTileOps`, `TileOps2D`, `TileMemoryInferred`, `NormalizedStmtStructure`

**Produces**:

- `TensorViewCanonical` — `PassPipeline` auto-verifies after the pass using the registry's **strict-mode** verifier (empty stride on a present `TensorView` is rejected — that is the state this pass is responsible for eliminating)

**Position in the default pipeline** (active since RFC #1300 P6): between [`CanonicalizeIOOrder`](25-canonicalize_io_order.md) and [`InitMemRef`](27-init_memref.md). This is the codegen-prep boundary — every layout-mutating pass (`LowerTransposeLoadParamLayout`, `ResolveBackendOpLayouts`, `ExpandMixedKernel`, `SplitVectorKernel`) has finished, and `InitMemRef` is the first consumer that needs explicit stride.

## API

| C++ | Python | Level |
| --- | ------ | ----- |
| `pass::MaterializeTensorStrides()` | `passes.materialize_tensor_strides()` | Program-level |

```python
from pypto.pypto_core import passes

mat_pass = passes.materialize_tensor_strides()
program_canon = mat_pass(program)
```

## Algorithm

The pass uses an `IRMutator` with a Var-substitution cache, mirroring the pattern used by `InferTileMemorySpace`. It walks every `TypePtr` reachable from the program:

1. **For each function**, rebuild parameters / return types / body:
   - Walk parameter types; if a parameter's `TensorType` materializes to a different type, build a fresh `Var` with the same `name_hint` / span and register the substitution.
   - Walk return types similarly.
   - Walk the body via `IRMutator::VisitStmt`. Inside:
     - `VisitExpr_(VarPtr)`: if the Var's type changes after `MaterializeType`, build a fresh Var with the new type (consulting `var_cache_` so every reference to a rebuilt Var resolves to the same new Var).
     - `VisitExpr_(IterArgPtr)`: same as Var, plus the `init_value_` is recursed.
     - `VisitExpr_(CallPtr)`: rebuild via `OpRegistry` when registered, falling back to a direct `Call` constructor for `GlobalVar` calls / unregistered ops.
     - `VisitStmt_(AssignStmtPtr)`: rebuild RHS first; if the RHS Call's return type is more specific than the current LHS Var type, sync the Var.

2. **Type rewriting** — `MaterializeType(type)`:
   - `TensorType` with `view.has_value() && view.stride.empty() && layout != NZ`: rebuild with `BuildLogicalStridesFromLayout(shape, layout)` filled in. Other `TensorType` shapes pass through unchanged (identity preserved).
   - `TensorType` with `layout == NZ`: untouched (NZ on `TensorType` is invalid IR; the verifier flags it instead of `BuildLogicalStridesFromLayout` `CHECK`-failing).
   - `TupleType`: recurse into element types; rebuild only if any sub-type changed.
   - Anything else: pass through.

The pass is **idempotent**: re-running on already-materialized IR is a no-op, since every type comparison short-circuits on identity and `MutableCopy` is skipped when nothing changed.

| Behavior | Trigger |
| -------- | ------- |
| Fill stride with packed canonical | `view.has_value() && view.stride.empty()` and `layout in {ND, DN}` |
| Identity pass-through | `!view.has_value()` (bare tensor) |
| Identity pass-through | `view.has_value() && !view.stride.empty()` (already explicit) |
| Identity pass-through | `view.layout == NZ` on `TensorType` (verifier rejects this separately) |

## Example

**Before** — InCore param with empty-stride DN view (user-written `pl.Tensor[..., pl.DN]` without an explicit stride hint):

```python
@pl.function(type=pl.FunctionType.InCore)
def kernel(b: pl.Tensor[[2, 4, 8], pl.FP32, pl.TensorView(stride=[], layout=pl.TensorLayout.DN)],
           out: pl.Out[pl.Tensor[[2, 4, 8], pl.FP32]]) -> pl.Tensor[[2, 4, 8], pl.FP32]:
    ...
```

**After**:

```python
@pl.function(type=pl.FunctionType.InCore)
def kernel(b: pl.Tensor[[2, 4, 8], pl.FP32, pl.TensorView(stride=[32, 1, 4], layout=pl.TensorLayout.DN)],
           out: pl.Out[pl.Tensor[[2, 4, 8], pl.FP32]]) -> pl.Tensor[[2, 4, 8], pl.FP32]:
    ...
```

The DN packed canonical stride for shape `[2, 4, 8]` is computed as:

- `stride[1] = 1` (DN trailing-pair innermost)
- `stride[2] = shape[1] = 4`
- `stride[0] = shape[1] * shape[2] = 32`

For ND, the formula reduces to the standard row-major packed strides.

## Stride Formulas

See `BuildLogicalStridesFromLayout` in [`tensor_view_semantics.h`](../../../../include/pypto/ir/transforms/utils/tensor_view_semantics.h).

| Layout | Formula |
| ------ | ------- |
| `ND` | `stride[n-1] = 1; stride[k] = stride[k+1] * shape[k+1]` for `k = n-2 .. 0` |
| `DN` (`n ≥ 2`) | `stride[n-2] = 1`; `stride[n-1] = shape[n-2]`; `stride[n-3] = shape[n-2] * shape[n-1]`; `stride[k] = stride[k+1] * shape[k+1]` for `k = n-4 .. 0` |
| `NZ` | not representable as flat strides (fractal, tile-only) — `BuildLogicalStridesFromLayout` `CHECK`-fails |

`MakeIndexMul` folds `ConstInt * ConstInt` (with `__builtin_mul_overflow` guard so an overflow falls back to a symbolic `Mul` rather than silently wrapping) and the multiplicative identity, so symbolic dims are preserved as `Mul` expressions while static chains collapse to a single `ConstInt`.

## Verifier interaction

Because the pass declares `produced = {... ∪ TensorViewCanonical}`, `PassPipeline` automatically runs the registry's `TensorViewCanonical` verifier after the pass. The registry default is the **strict-mode** verifier (RFC #1300 §2.4 codegen-entry contract): it rejects `view.has_value() && stride.empty()` since this pass is responsible for materializing those slots. Bare `TensorType` (`!view.has_value()`) is still accepted — implicit ND-packed is canonical by construction. The same verifier is callable directly via `passes.verify_tensor_view_canonical(program, require_materialized=True)`; pass `require_materialized=False` for the weak mode used during the parse-time / early-pass window before materialization runs.

## Related

- [`CanonicalizeIOOrder`](25-canonicalize_io_order.md) — runs immediately before; produces the program state the materialization consumes
- [`InitMemRef`](27-init_memref.md) — first downstream consumer that depends on explicit stride
- [`LowerTransposeLoadParamLayout`](18-lower_transpose_load_param_layout.md) — primary source of empty-stride DN views in the default pipeline (active since RFC #1300 P6)
- [`tensor_view_semantics.h`](../../../../include/pypto/ir/transforms/utils/tensor_view_semantics.h) — the helpers (`BuildLogicalStridesFromLayout`, `CheckCanonicalView`, `CanonicalizeView`)
- RFC [#1300](https://github.com/hw-native-sys/pypto/issues/1300) — Self-consistent IR TensorType layout representation
