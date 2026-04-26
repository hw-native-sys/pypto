# ResolveBackendOpLayouts Pass

Repairs backend-required tile layouts for elementwise ops by reshaping `[N, 1]` col-major vector inputs into `[1, N]` row-major views and reshaping the result back. Runs in the tile-PTO stage between `ResolveTransposeLayout` and the trailing `NormalizeStmtStructure`.

## Overview

After `FlattenTileNdTo2D` and `ResolveTransposeLayout`, every tile op is in 2-D form with a known layout. Several PTO elementwise ops (registered in `src/backend/common/pto_ops_common.cpp`) require their tile operands and result to be `row_major`. A column vector of shape `[N, 1]` has the same memory layout as a row vector of shape `[1, N]`, so this pass repairs the violation locally:

1. For each `AssignStmt` / `EvalStmt` whose RHS is a `Call`, query `Backend::GetTileLayoutSpec(op_name)`.
2. Skip if no spec is registered, or if no `[N, 1]` col-major input violates a `row_major` requirement.
3. Otherwise insert `tile.reshape(arg, [1, N])` before the call, substitute the reshaped value into the call, and — for `AssignStmt` results, unless the result type is already a `[1, N]` row-major tile — append `tile.reshape(tmp, original_shape)` to restore the user-visible shape.

The pass is **backend-driven**: the set of constrained ops and their per-input requirements come from each op's `BackendOpRegistryEntry` (see `set_input_layout` / `set_output_layout` in `pto_ops_common.cpp`). The pass code itself stays backend-agnostic — adding a new constrained op only requires registering its layout spec, not editing this pass.

**Requirements**:

- Run after `FlattenTileNdTo2D` (assumes 2-D tile ops).
- Function must be `InCore` — Orchestration / Group functions are skipped.
- A backend must be configured via `BackendConfig::Set(...)`. Otherwise the pass is a no-op.

**When to use**: As part of the `Default` tile-PTO pipeline, after layout-altering passes (`FlattenTileNdTo2D`, `InferTileMemorySpace`, `ResolveTransposeLayout`) and before `NormalizeStmtStructure`. The pass manager already places it in the correct slot.

## API

| C++ | Python | Level |
| --- | ------ | ----- |
| `pass::ResolveBackendOpLayouts()` | `passes.resolve_backend_op_layouts()` | Function-level |

**Python usage**:

```python
from pypto.pypto_core import passes

repair = passes.resolve_backend_op_layouts()
program = repair(program)
```

## Algorithm

```text
For each function in the program:
  Skip if function is not InCore.
  Skip if no backend is configured.

  Walk the body with IRMutator. For each AssignStmt / EvalStmt whose
  RHS is a Call:
    spec = backend.GetTileLayoutSpec(call.op.name)
    if spec is None: continue
    if no input violates spec.input_layouts (only [N,1] col-major inputs
       targeting a row_major slot are repairable): continue

    For each input i targeting a row_major slot:
      Skip if the input is non-tile, or [1, N] row-major, or not [N, 1].
      reshape_var = fresh temp
        (AssignStmt: name derived from the result variable.
         EvalStmt:  name derived from the literal "layout_fix".
         Both forms add "row_major" + "arg<i>" qualifiers.)
      emit  reshape_var = tile.reshape(arg_i, [1, N])
      substitute reshape_var into the call

    repaired = OpRegistry.Create(call.op.name, new_args, call.kwargs)

    If statement is AssignStmt and result_type is not [1, N] row-major:
      tmp = fresh row-major temp ("row_major" qualifier on the result name)
      emit  tmp = repaired
      emit  result_var = tile.reshape(tmp, original_result_shape)
    Else:
      emit  result_var = repaired   (or EvalStmt with repaired)
```

Non-tile inputs (scalars, shapes) and inputs whose required layout is `nullopt` are left untouched. For tile inputs targeting a `row_major` slot, the per-input rewrite loop reshapes any `[N, 1]` operand (col-major *or* row-major) once the call has been classified as repairable; only `[1, N]` row-major and non-`[N, 1]` shapes are skipped. The call is classified as repairable (`IsRepairableCall`) only when at least one input violates `row_major` and every violating input is `[N, 1]` col-major — if any violating input falls outside that pattern, the statement is left alone.

## Example

(adapted from `tests/ut/ir/transforms/test_resolve_backend_op_layouts_pass.py::test_rewrites_column_vector_add_through_row_major_reshape`, with the Ascend910B backend active)

**Before**:

```python
@pl.program
class Before:
    @pl.function(type=pl.FunctionType.InCore)
    def repro(
        self,
        data: pl.Tensor[[16, 256], pl.FP32],
        out: pl.Out[pl.Tensor[[16, 1], pl.FP32]],
    ) -> pl.Tensor[[16, 1], pl.FP32]:
        acc_0: pl.Tile[[16, 1], pl.FP32, pl.MemorySpace.Vec] = pl.tile.create(
            [16, 1], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
        )
        acc_1: pl.Tile[[16, 1], pl.FP32, pl.MemorySpace.Vec] = pl.tile.muls(acc_0, 0.0)
        chunk: pl.Tile[[16, 256], pl.FP32, pl.MemorySpace.Vec] = pl.load(data, [0, 0], [16, 256])
        tmp: pl.Tile[[16, 256], pl.FP32, pl.MemorySpace.Vec] = pl.tile.create(
            [16, 256], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
        )
        partial: pl.Tile[[16, 1], pl.FP32, pl.MemorySpace.Vec] = pl.tile.row_sum(chunk, tmp)
        updated: pl.Tile[[16, 1], pl.FP32, pl.MemorySpace.Vec] = pl.tile.add(acc_1, partial)
        stored: pl.Tensor[[16, 1], pl.FP32] = pl.store(updated, [0, 0], out)
        return stored
```

**After**:

```python
@pl.program
class After:
    @pl.function(type=pl.FunctionType.InCore)
    def repro(
        self,
        data: pl.Tensor[[16, 256], pl.FP32],
        out: pl.Out[pl.Tensor[[16, 1], pl.FP32]],
    ) -> pl.Tensor[[16, 1], pl.FP32]:
        acc_0: pl.Tile[[16, 1], pl.FP32, pl.MemorySpace.Vec] = pl.tile.create(
            [16, 1], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
        )
        acc_0_rm: pl.Tile[[1, 16], pl.FP32, pl.MemorySpace.Vec] = pl.tile.reshape(acc_0, [1, 16])
        acc_1_rm: pl.Tile[[1, 16], pl.FP32, pl.MemorySpace.Vec] = pl.tile.muls(acc_0_rm, 0.0)
        acc_1: pl.Tile[[16, 1], pl.FP32, pl.MemorySpace.Vec] = pl.tile.reshape(acc_1_rm, [16, 1])
        chunk: pl.Tile[[16, 256], pl.FP32, pl.MemorySpace.Vec] = pl.load(data, [0, 0], [16, 256])
        tmp: pl.Tile[[16, 256], pl.FP32, pl.MemorySpace.Vec] = pl.tile.create(
            [16, 256], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
        )
        partial: pl.Tile[[16, 1], pl.FP32, pl.MemorySpace.Vec] = pl.tile.row_sum(chunk, tmp)
        acc_1_rm2: pl.Tile[[1, 16], pl.FP32, pl.MemorySpace.Vec] = pl.tile.reshape(acc_1, [1, 16])
        partial_rm: pl.Tile[[1, 16], pl.FP32, pl.MemorySpace.Vec] = pl.tile.reshape(partial, [1, 16])
        updated_rm: pl.Tile[[1, 16], pl.FP32, pl.MemorySpace.Vec] = pl.tile.add(acc_1_rm2, partial_rm)
        updated: pl.Tile[[16, 1], pl.FP32, pl.MemorySpace.Vec] = pl.tile.reshape(updated_rm, [16, 1])
        stored: pl.Tensor[[16, 1], pl.FP32] = pl.store(updated, [0, 0], out)
        return stored
```

`tile.muls`, `tile.add`, and similar elementwise PTO ops require `row_major` inputs and output. Each constrained call is wrapped: the `[16, 1]` operand is reshaped to `[1, 16]` immediately before the call, the call runs in row-major form, and the result is reshaped back to `[16, 1]` so downstream code (`tile.store`, return type) keeps the user-visible shape. `tile.row_sum` is unconstrained, so its inputs and output are left as-is.

## Implementation

| File | Role |
| ---- | ---- |
| `include/pypto/ir/transforms/passes.h` (`ResolveBackendOpLayouts`) | Public C++ factory |
| `src/ir/transforms/resolve_backend_op_layouts_pass.cpp` | Mutator and pass body |
| `include/pypto/ir/transforms/pass_properties.h` (`kResolveBackendOpLayoutsProperties`) | Pass properties |
| `python/bindings/modules/passes.cpp` (`resolve_backend_op_layouts`) | Python binding |
| `python/pypto/pypto_core/passes.pyi` (`resolve_backend_op_layouts`) | Type stub |
| `tests/ut/ir/transforms/test_resolve_backend_op_layouts_pass.py` | Unit tests (binary, unary, scalar-binary on `[N, 1]` vectors) |

Layout constraints are registered per op via `BackendOpRegistryEntry::set_input_layout` / `set_output_layout` in `src/backend/common/pto_ops_common.cpp` (e.g. row-major elementwise ops listed in `RequiresRowMajorLayout`, `tile.rsqrt`, `tile.cmps`, `tile.sort32`, `tile.mscatter`, ...).

Key helpers in the pass source:

- `IsRepairableCall` — true iff at least one tile input violates a `row_major` requirement and every violating input is a `[N, 1]` col-major tile.
- `BackendLayoutRepairMutator::VisitStmt_(const AssignStmtPtr&)` / `VisitStmt_(const EvalStmtPtr&)` — emit the pre-call reshape(s), rebuild the call, and (for `AssignStmt`) emit the post-call reshape when the result was a col-major column vector.
- `RewriteFunction` — bypasses non-`InCore` functions and the unconfigured-backend case before invoking the mutator.

## Pass Properties

| Property | Value |
| -------- | ----- |
| Required | SSAForm, IncoreTileOps, SplitIncoreOrch, TileOps2D |
| Produced | SSAForm, IncoreTileOps, SplitIncoreOrch, TileOps2D |
| Invalidated | NormalizedStmtStructure |

`NormalizedStmtStructure` is invalidated because each repair sequence wraps a previously single-statement op with extra `tile.reshape` assignments, breaking the canonical statement layout. The default pipeline re-runs `NormalizeStmtStructure` immediately after this pass to restore the invariant.

## Design Decisions

| Decision | Rationale |
| -------- | --------- |
| Drive layout requirements from `Backend::GetTileLayoutSpec` rather than hard-coded op lists in the pass | Each backend declares its own constraints next to its codegen registration. The pass stays backend-agnostic per `pass-context-config.md`; new constrained ops cost one `set_input_layout` call, not a pass edit. |
| Repair via two `tile.reshape` ops instead of rejecting the program | `[N, 1]` col-major and `[1, N]` row-major share the same flat memory; a local reshape preserves user-visible shapes while satisfying the backend ISA without forcing kernel rewrites. |
| Only repair `[N, 1]` col-major inputs (not arbitrary layout mismatches) | This is the only mismatch class observed for current PTO row-major elementwise ops. Broader cases would require a real layout-conversion pass and are out of scope; if one is found, `IsRepairableCall` returns false and the statement is left alone. |
| Bypass when no backend is configured | Many tests build IR without selecting a backend. A no-op fast path keeps those green and avoids spurious mutations. |
| Skip non-`InCore` functions | Layout constraints apply to per-core elementwise execution; Orchestration and Group functions only contain calls to lower-level kernels and have nothing to repair. |
