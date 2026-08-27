# MaterializeTensorStrides Pass

Fills every `view.has_value() && view.stride.empty()` slot on every `TensorType` / `DistributedTensorType` reachable from the program with the packed canonical stride for the carried layout (per RFC #1300 §2.4). After this pass runs, the codegen-entry contract holds: every `TensorView` that exists has explicit stride matching its layout / shape, and the strict-mode `TensorViewCanonical` verifier accepts the IR.

> **Status**: this pass is registered (`passes.materialize_tensor_strides()`), covered by unit tests, and wired into the default tile/PTO pipeline between `CanonicalizeIOOrder` and `InitMemRef` starting from RFC #1300 P6.

## Overview

PyPTO's IR allows two equivalent forms for `TensorType.tensor_view_`:

- **Implicit** — `view.has_value() && view.stride.empty()`: the layout tag is set (e.g. `DN`) but the per-dimension stride is left blank. Downstream consumers must treat empty stride as "use the packed canonical stride for this layout."
- **Explicit** — every dimension has its `ExprPtr` stride spelled out.

Codegen needs one machine-readable contract, so `MaterializeTensorStrides` walks the program and rewrites every implicit `TensorView` into its explicit packed canonical form using `BuildLogicalStridesFromLayout` from `tensor_view_semantics.h`. Bare `TensorType`s (`!view.has_value()`) are left untouched: the `TensorViewCanonical` verifier accepts them in both modes as implicitly ND-packed and the bare form is unambiguous. When the input type is a `DistributedTensorType`, the rebuilt type remains distributed and preserves its `memref`, non-stride `TensorView` metadata such as `pad`, and `window_buffer` back-reference.

**Requirements**:

- `SSAForm`, `SplitIncoreOrch`, `IncoreTileOps`, `TileOps2D`, `TileMemoryInferred`, `NormalizedStmtStructure`

**Produces**:

- `TensorViewCanonical` — `PassPipeline` auto-verifies after the pass using the registry's **strict-mode** verifier (empty stride on a present `TensorView` is rejected — that is the state this pass is responsible for eliminating)

**Position in the default pipeline** (active since RFC #1300 P6): between [`CanonicalizeIOOrder`](30-canonicalize_io_order.md) and [`InitMemRef`](32-init_memref.md). This is the codegen-prep boundary — every layout-mutating pass (`ResolveBackendOpLayouts`, `ExpandMixedKernel`, `SplitVectorKernel`) has finished, and `InitMemRef` is the first consumer that needs explicit stride.

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

2. **Type rewriting** — `MaterializeType(type, span)`:
   - `TensorType` / `DistributedTensorType` with `layout == NZ` whose shape is **not blocked**: **rejected** with an `INTERNAL_CHECK_SPAN`, whether or not the stride is explicit. NZ is legal on a tensor type, but only in the blocked rank-(r+2) form `[..., C/c0, R/16, 16, c0]` that [BlockNzTensorViews](14-block_nz_tensor_views.md) produces — that is the only shape for which the row-major stride built below actually describes the NZ byte order. Reaching here unblocked means pass 14 did not run or missed a slot, so this is a pass-ordering invariant, not a user error (the user-facing alignment diagnostics live in `BlockNzShape`). The `span` argument (the `Var` / `IterArg` / `Call` / `Submit` / param / function node carrying the type) locates the offending annotation in the message.
   - `TensorType` / `DistributedTensorType` with `view.has_value() && view.stride.empty()`: rebuild with `BuildLogicalStridesFromLayout(shape, layout)` filled in. The distributed wrapper and optional metadata (`memref`, `TensorView.pad`, `window_buffer`) are preserved. Other tensor shapes pass through unchanged (identity preserved).
   - `TupleType`: recurse into element types (same `span`); rebuild only if any sub-type changed.
   - Anything else: pass through.

The unblocked-NZ rejection lives in the pass rather than being delegated to the paired verifier. Delegating made the rejection conditional on verification being enabled: under `PYPTO_VERIFY_LEVEL=none` the pass returned the malformed slot untouched while still declaring `TensorViewCanonical` as produced, and the invalid layout only resurfaced downstream as an opaque backend layout mismatch.

The pass is **idempotent**: re-running on already-materialized IR is a no-op, since every type comparison short-circuits on identity and `MutableCopy` is skipped when nothing changed.

| Behavior | Trigger |
| -------- | ------- |
| Fill stride with packed canonical | `view.has_value() && view.stride.empty()` and `layout in {ND, DN}` |
| Identity pass-through | `!view.has_value()` (bare tensor) |
| Identity pass-through | `view.has_value() && !view.stride.empty()` and `layout in {ND, DN}` (already explicit) |
| Reject (`InternalError`) | `view.layout == NZ` with an unblocked shape, with or without explicit stride (BlockNzTensorViews should have blocked it) |

The rows are mutually exclusive: the blocked-NZ check runs first, so an
explicit-stride unblocked NZ view is rejected rather than passed through.

## Example

**Before** — InCore param with empty-stride DN view (`pl.TensorView(layout=DN)` written without an explicit stride hint):

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
| `NZ` | row-major over the *blocked* shape — identical to the ND rule. Row-major over `[..., C/c0, R/16, 16, c0]` reproduces pto-isa's `BaseShape2D<..., Layout::NZ>` exactly, so NZ needs no rule of its own. An unblocked NZ shape is rejected before this point. |

`MakeIndexMul` folds `ConstInt * ConstInt` (with `__builtin_mul_overflow` guard so an overflow falls back to a symbolic `Mul` rather than silently wrapping) and the multiplicative identity, so symbolic dims are preserved as `Mul` expressions while static chains collapse to a single `ConstInt`.

## Verifier interaction

Because the pass declares `produced = {... ∪ TensorViewCanonical}`, `PassPipeline` automatically runs the registry's `TensorViewCanonical` verifier after the pass. The registry default is the **strict-mode** verifier (RFC #1300 §2.4 codegen-entry contract): it rejects `view.has_value() && stride.empty()` since this pass is responsible for materializing those slots. Bare `TensorType` (`!view.has_value()`) is still accepted — implicit ND-packed is canonical by construction. The same verifier is callable directly via `passes.verify_tensor_view_canonical(program, require_materialized=True)`; pass `require_materialized=False` for the weak mode used during the parse-time / early-pass window before materialization runs.

The verifier is a paired check, not the sole enforcement point. Verification is configurable (`PYPTO_VERIFY_LEVEL` / `PassContext`) and may be off, so every invariant the pass claims to produce is established by the pass itself — the unblocked-NZ rejection above included. The verifier then re-checks it, catching regressions in this pass and in anything downstream that rewrites tensor types.

## Related

- [`CanonicalizeIOOrder`](30-canonicalize_io_order.md) — runs immediately before; produces the program state the materialization consumes
- [`InitMemRef`](32-init_memref.md) — first downstream consumer that depends on explicit stride
- [`tensor_view_semantics.h`](../../../../include/pypto/ir/transforms/utils/tensor_view_semantics.h) — the helpers (`BuildLogicalStridesFromLayout`, `CheckCanonicalView`, `CanonicalizeView`)
- RFC [#1300](https://github.com/hw-native-sys/pypto/issues/1300) — Self-consistent IR TensorType layout representation
