# ResolveBackendOpLayouts Pass

Repairs backend-required tile layouts for elementwise ops. `[N, 1]` col-major vectors are reshaped into `[1, N]` row-major views, while general non-row-major tiles are coerced through `tile.move(..., blayout=row_major)`. Runs in the tile-PTO stage between `LowerTransposeLoadParamLayout` and `ExpandMixedKernel`; the pass re-normalizes statement structure internally before returning, so `NormalizedStmtStructure` is preserved across the pass.

## Overview

After `FlattenTileNdTo2D` and `LowerTransposeLoadParamLayout`, every tile op is in 2-D form with a known layout. Several PTO elementwise ops (registered in `src/backend/common/pto_ops_common.cpp`) require their tile operands and result to be `row_major`. This pass repairs those local violations at the consumer:

1. For each `AssignStmt` / `EvalStmt` whose RHS is a `Call`, query `Backend::GetTileLayoutSpec(op_name)`.
2. Skip if no spec is registered, or if all constrained tile inputs and output already use `row_major`.
3. For `[N, 1]` col-major inputs, insert `tile.reshape(arg, [1, N])` before the call. This is a metadata-only view repair because `[N, 1]` col-major and `[1, N]` row-major have the same flat memory order.
4. For other non-row-major tile inputs, insert `tile.move(arg, target_memory=<same>, blayout=row_major, slayout=none_box)` before the call.
5. For `AssignStmt` results whose original result type is not row-major, assign the repaired call to a row-major temporary, then restore the original result layout with either `tile.reshape` for column vectors or `tile.move` for general matrix tiles.

The pass is **backend-driven**: the set of constrained ops and their per-input requirements come from each op's `BackendOpRegistryEntry` (see `set_input_layout` / `set_output_layout` in `pto_ops_common.cpp`). The pass code itself stays backend-agnostic — adding a new constrained op only requires registering its layout spec, not editing this pass.

**Requirements**:

- Run after `FlattenTileNdTo2D` (assumes 2-D tile ops).
- Function must be `InCore` — Orchestration / Group functions are skipped.
- A backend must be configured via `BackendConfig::Set(...)`. Otherwise the pass is a no-op.

**When to use**: As part of the `Default` tile-PTO pipeline, after layout-altering passes (`FlattenTileNdTo2D`, `InferTileMemorySpace`, `LowerTransposeLoadParamLayout`) and before `ExpandMixedKernel`. The pass manager already places it in the correct slot.

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
    if no constrained input/output needs row_major repair: continue

    For each input i targeting a row_major slot:
      Skip if the input is non-tile or already row-major.
      reshape_var = fresh temp
        (AssignStmt: name derived from the result variable.
         EvalStmt:  name derived from the literal "layout_fix".
         Both forms add "row_major" + "arg<i>" qualifiers.)
      If input is [N, 1] col-major:
        emit  reshape_var = tile.reshape(arg_i, [1, N])
      Else:
        emit  reshape_var = tile.move(arg_i, target_memory=<same>,
                                      blayout=row_major, slayout=none_box)
      substitute reshape_var into the call

    repaired = OpRegistry.Create(call.op.name, new_args, call.kwargs)

    If statement is AssignStmt and result_type is constrained but not row-major:
      tmp = fresh row-major temp ("row_major" qualifier on the result name)
      emit  tmp = repaired
      If original result is a column vector:
        emit  result_var = tile.reshape(tmp, original_result_shape)
      Else:
        emit  result_var = tile.move(tmp, target_memory=<same>,
                                     blayout=<original>, slayout=<original>)
    Else:
      emit  result_var = repaired   (or EvalStmt with repaired)
```

Non-tile inputs (scalars, shapes) and inputs whose required layout is `nullopt` are left untouched. The repair is local to the constrained op: downstream code still sees the original variable type and layout because `AssignStmt` results are restored after the row-major operation.

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
| `tests/ut/ir/transforms/test_resolve_backend_op_layouts_pass.py` | Unit tests (binary, unary, scalar-binary on `[N, 1]` vectors, plus matrix layout coercion through `tile.move`) |

Layout constraints are registered per op via `BackendOpRegistryEntry::set_input_layout` / `set_output_layout` in `src/backend/common/pto_ops_common.cpp` (e.g. row-major elementwise ops listed in `RequiresRowMajorLayout`, `tile.rsqrt`, `tile.cmps`, `tile.sort32`, `tile.mscatter`, ...).

Key helpers in the pass source:

- `NeedsInputRepair` / `NeedsOutputRepair` — detect constrained `row_major` slots whose current tile layout is not row-major.
- `CreateLayoutMoveCall` — emits the `tile.move` used for general matrix layout coercion and result restoration.
- `BackendLayoutRepairMutator::VisitStmt_(const AssignStmtPtr&)` / `VisitStmt_(const EvalStmtPtr&)` — emit the pre-call reshape/move repairs, rebuild the call, and (for `AssignStmt`) emit the post-call reshape/move restoration when needed.
- `RewriteFunction` — bypasses non-`InCore` functions and the unconfigured-backend case before invoking the mutator.

## Pass Properties

| Property | Value |
| -------- | ----- |
| Required | SSAForm, IncoreTileOps, SplitIncoreOrch, TileOps2D |
| Produced | SSAForm, IncoreTileOps, SplitIncoreOrch, TileOps2D, NormalizedStmtStructure |
| Invalidated | — |

Each repair sequence temporarily wraps a previously single-statement op with extra `tile.reshape` assignments, which would break the canonical statement layout. To keep `NormalizedStmtStructure` intact across the pass, `ResolveBackendOpLayouts` invokes `NormalizeStmtStructure` on its own output before returning, so the property is **produced** rather than invalidated.

## Design Decisions

| Decision | Rationale |
| -------- | --------- |
| Drive layout requirements from `Backend::GetTileLayoutSpec` rather than hard-coded op lists in the pass | Each backend declares its own constraints next to its codegen registration. The pass stays backend-agnostic per `pass-context-config.md`; new constrained ops cost one `set_input_layout` call, not a pass edit. |
| Prefer `tile.reshape` for `[N, 1]` vectors | `[N, 1]` col-major and `[1, N]` row-major share the same flat memory, so reshape is cheaper and preserves the existing vector repair behavior. |
| Use `tile.move` for general matrix layout coercion | Full tiles such as `[16, 256]` cannot be repaired by a shape-only vector reshape. A same-memory `tile.move` materializes a row-major view before row-major PTO ops such as `tile.exp`, then restores the original layout when needed. |
| Bypass when no backend is configured | Many tests build IR without selecting a backend. A no-op fast path keeps those green and avoids spurious mutations. |
| Skip non-`InCore` functions | Layout constraints apply to per-core elementwise execution; Orchestration and Group functions only contain calls to lower-level kernels and have nothing to repair. |
