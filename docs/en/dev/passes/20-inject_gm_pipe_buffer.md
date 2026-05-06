# InjectGMPipeBuffer Pass

Injects the `__gm_pipe_buffer` workspace parameter for cross-core pipes on backends that route slot data through GM (currently Ascend910B). Runs immediately after `ExpandMixedKernel`.

## Overview

On Ascend910B, cross-core `tpush`/`tpop` rides through a shared GM buffer instead of a direct inter-core fabric. After `ExpandMixedKernel` has split mixed InCore functions into AIC/AIV pairs and prepended `aic_initialize_pipe` / `aiv_initialize_pipe`, this pass:

1. Finds every function whose body issues `aic_initialize_pipe` or `aiv_initialize_pipe`.
2. Adds a fresh `__gm_pipe_buffer` Out-tensor parameter to each such function.
3. Propagates the parameter upward through the call graph: any caller of a function that took the new parameter also gets it added (so the workspace flows from Orchestration down to AIC/AIV).
4. Stops at Orchestration functions — they do **not** receive the parameter. Instead, the pass injects a per-call-site `tensor.create` so the workspace is materialized locally on the host side and passed in.

The pass is **backend-gated** on `BackendHandler::RequiresGMPipeBuffer()`. Backends without GM-routed pipes (e.g. Ascend950 with its direct cross-core fabric) see this pass as a no-op.

**Requirements**:

- Input IR must have AIC/AIV split with cross-core pipe setup already in place (run `ExpandMixedKernel` first).
- Backend must report `RequiresGMPipeBuffer() == true`. Otherwise the pass is a no-op.

**When to use**: Run after `ExpandMixedKernel` when targeting Ascend910B (or any backend that signals `RequiresGMPipeBuffer`). The default tile pipeline already places it in the correct slot.

> **Note**: This pass was extracted from `ExpandMixedKernel` to keep that pass focused on the AIC/AIV split logic and to scope the GM-workspace concern to a single, backend-gated transform.

## API

| C++ | Python | Level |
| --- | ------ | ----- |
| `pass::InjectGMPipeBuffer()` | `passes.inject_gm_pipe_buffer()` | Program-level |

**Python usage**:

```python
from pypto.pypto_core import passes

inject_pass = passes.inject_gm_pipe_buffer()
program = inject_pass(program)
```

## Algorithm

```text
Phase 1 — Discover seed functions:
  Walk every function. A function is a "seed" if its (recursively flattened)
  body contains an aic_initialize_pipe or aiv_initialize_pipe op.

Phase 2 — Append the parameter to each seed (and propagate upward):
  Use a worklist seeded with seed functions. For each function F popped:
    - Append __gm_pipe_buffer (Out-tensor) to F's parameter list.
    - For every caller C of F that also lives in the program:
        - If C is Orchestration: record C in orch_needs_tensor_create.
        - Otherwise: rewrite C's call site to forward C's own __gm_pipe_buffer
          argument, and enqueue C onto the worklist if it has not been
          rewritten yet.

Phase 3 — Inject tensor.create at Orchestration call sites:
  For each Orchestration function recorded in orch_needs_tensor_create,
  prepend a tensor.create that materializes the workspace and rewrite each
  affected call to pass it.
```

## Example

**Before** (after `ExpandMixedKernel`, Ascend910B):

```python
@pl.program
class Before:
    @pl.function(type=pl.FunctionType.AIC)
    def compute_aic(self, x, y, out_0):
        # ... aic_initialize_pipe(...) and Cube ops ...

    @pl.function(type=pl.FunctionType.AIV)
    def compute_aiv(self, x, y, out_0):
        # ... aiv_initialize_pipe(...) and Vector ops ...

    @pl.function(type=pl.FunctionType.Group)
    def compute(self, x, y, out_0):
        self.compute_aic(x, y, out_0)
        return self.compute_aiv(x, y, out_0)

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, x, y):
        out_0 = pl.create_tensor([16, 128], dtype=pl.FP32)
        return self.compute(x, y, out_0)
```

**After**:

```python
@pl.program
class After:
    @pl.function(type=pl.FunctionType.AIC)
    def compute_aic(self, x, y, out_0, __gm_pipe_buffer):
        # ... aic_initialize_pipe(...) referencing __gm_pipe_buffer ...

    @pl.function(type=pl.FunctionType.AIV)
    def compute_aiv(self, x, y, out_0, __gm_pipe_buffer):
        # ... aiv_initialize_pipe(...) referencing __gm_pipe_buffer ...

    @pl.function(type=pl.FunctionType.Group)
    def compute(self, x, y, out_0, __gm_pipe_buffer):
        self.compute_aic(x, y, out_0, __gm_pipe_buffer)
        return self.compute_aiv(x, y, out_0, __gm_pipe_buffer)

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, x, y):
        out_0 = pl.create_tensor([16, 128], dtype=pl.FP32)
        # Injected: workspace sized in FP32 elements (ceil(required_bytes / 4)).
        # FP32 is the contract today because tensor.create's element-count shape
        # paired with FP32 yields a backing of `4 * elements` bytes.
        __gm_pipe_buffer = pl.create_tensor([math.ceil(required_bytes / 4)], dtype=pl.FP32)
        return self.compute(x, y, out_0, __gm_pipe_buffer)
```

## Implementation

**Header**: `include/pypto/ir/transforms/passes.h`

```cpp
Pass InjectGMPipeBuffer();
```

**Implementation**: `src/ir/transforms/inject_gm_pipe_buffer_pass.cpp`

- `HasInitializePipeOps` — recursive scan for `aic_initialize_pipe` / `aiv_initialize_pipe` (uses `op_predicates::IsInitializePipe`)
- `AddGMSlotBufferParam` — append the Out-tensor parameter
- `RewriteCallsForGMBuffer` — rewrite a caller's call sites
- `CreateGMPipeBufferTensorCreate` — synthesize the Orchestration-side `tensor.create`
- `RewriteCallsWithPerCallGMBuffer` — drive the Orchestration-side rewrite, hoisting the `tensor.create` and forwarding the workspace per call site

**Python binding**: `python/bindings/modules/passes.cpp`

```cpp
passes.def("inject_gm_pipe_buffer", &pass::InjectGMPipeBuffer,
           "Inject __gm_pipe_buffer workspace parameter for GM-routed cross-core pipes");
```

**Tests**: covered transitively via `tests/ut/ir/transforms/test_expand_mixed_kernel_a2a3.py` (Ascend910B pipelines).

## Pass Properties

| Property | Value |
| -------- | ----- |
| Required | SSAForm, MixedKernelExpanded, NormalizedStmtStructure |
| Produced | SSAForm, MixedKernelExpanded, NormalizedStmtStructure |
| Invalidated | — |

The pass preserves all properties it requires (no-op on non-910B backends; same-shape rewrite on 910B).

## Design Decisions

| Decision | Rationale |
| -------- | --------- |
| Separate from `ExpandMixedKernel` | Keeps the kernel split focused on AIC/AIV body construction and confines the GM-workspace concern to a single, easily disabled pass |
| Backend-gated via `BackendHandler::RequiresGMPipeBuffer()` | Lets each backend opt in without scattering `if (backend == "910B")` checks in pass code (see `pass-context-config.md`) |
| Detect via `initialize_pipe` ops, not function-name patterns | Robust to renaming and to functions that legitimately do not need cross-core setup |
| Stop propagation at Orchestration | Orchestration is the host-side scheduler; it is the right layer to materialize the workspace and hand it to device-side callees |
| Materialize workspace via `tensor.create` at the call site | Keeps the Orchestration signature stable for the user and lets the codegen emit a regular host-side allocation |
