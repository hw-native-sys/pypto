# OptimizeOrchTensors Pass

Optimizes tensor buffer usage across orchestration and InCore functions by eliminating redundant allocations and improving data flow.

## Overview

After `ConvertTensorToTileOps`, orchestration functions allocate output tensors (`tensor.create`) at every InCore call site, even inside loops where the same buffer could be reused. This pass applies five optimization patterns to reduce allocations, improve buffer layout information, and make statically provable local tensor windows explicit at orchestration call sites.

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

The pass accepts a single `window_policy` parameter (default `"auto"`):

```python
passes.optimize_orch_tensors(window_policy="auto")
```

`window_policy` controls the window externalization strategy:

| Value | Output behavior | Input behavior | Notes |
| ----- | --------------- | -------------- | ----- |
| `"auto"` | ABI-safe single-piece windows | ABI-safe static windows | Default; proof-based and submit-signature preserving |
| `"all"` | All correctness-proven output windows | All correctness-proven input windows, including dynamic carrier readers | Aggressive mode; may expand submit signatures and carrier args |
| `"none"` | No windowing by default | No windowing by default | Per-kernel attrs can still opt in |

The global policy maps to per-kernel defaults:

| Global policy | Default `window_outputs` | Default `window_inputs` |
| ------------- | ------------------------ | ----------------------- |
| `"auto"` | `"auto"` | `"auto"` |
| `"all"` | `"all"` | `"all"` |
| `"none"` | `"off"` | `"off"` |

**Per-kernel overrides** via function attrs:

```python
@pl.function(
    type=pl.FunctionType.InCore,
    attrs={
        "window_policy":  "auto",      # optional per-kernel default: none | auto | all
        "window_outputs": "coalesce",  # off | auto | all | exact | coalesce
        "window_inputs":  "off",       # off | auto | all
    }
)
def my_kernel(...): ...
```

Per-kernel attrs override the global default. Missing attrs inherit the global
policy; explicit attrs win, including explicit opt-in when the global default is
`"none"`.

For `window_inputs`, `"auto"` covers ABI-safe static input windows. Dynamic
carrier readers that require extra carrier scalar/current args need effective
`window_inputs="all"`, either from global `"all"` or from an explicit per-kernel
attr. The pass does not create private dynamic reader scans without a proven
writer-reader carrier flow.

`window_outputs` and `window_inputs` are independent switches. Setting only
`window_outputs="off"` disables output rewriting for that kernel, but input-side
windowing can still clone the callee when `window_inputs` remains `"auto"`.
Set both attrs to `"off"` to disable all window externalization for one kernel:

```python
attrs={"window_outputs": "off", "window_inputs": "off"}
```

Use `window_outputs="coalesce"` when the caller only needs a coalesced output
bounding box. This chooses the writer output shape; it does not by itself opt a
downstream dynamic reader into extra carrier args:

```python
attrs={"window_outputs": "coalesce"}
```

Use `window_inputs="all"` on a compatible downstream reader when it should be
allowed to reuse a proven output-derived carrier:

```python
attrs={"window_inputs": "all"}
```

For example, in a Qwen prefill-style dynamic-indexed KV-cache writer/reader
chain, the expected boundary is:

- `"auto"`: keep local/simple windows only. Dynamic KV-cache writer and reader
  windows stay full-tensor, and no carrier/remat is introduced.
- `"all"`: rewrite every correctness-proven writer/reader chain the pass can
  prove, including output-derived carrier paths for dynamic readers such as
  qk/sv cache reads. Dynamic readers without a proven carrier still keep the full
  parent tensor (no private dynamic reader).
- `window_outputs="coalesce"`: opt one writer into multi-piece output bounding-box
  coalescing, but do not opt a downstream dynamic reader into extra carrier
  arguments.
- `window_outputs="coalesce"` plus reader `window_inputs="all"`: if the
  writer/reader flow is proven compatible, generate the output-derived
  carrier/remat path. This explicit input-side opt-in may add carrier
  scalar/current arguments and therefore may expand the runtime submit
  signature.

The carrier path depends on all of these conditions:

- the producer's effective `window_outputs` is not `"off"`
- the producer output rewrite actually survives as a `__windowed` call
- the reader's effective `window_inputs` is `"all"` when the carrier path needs
  extra runtime args
- the writer/reader pair satisfies the shared-root, loop-order, and min/max
  proof checks

If any condition fails, the dynamic reader keeps the full parent tensor and the
pass must not insert a runtime-current barrier just because the attr is present.

Output windows whose parent shape has dynamic dimensions are handled
conservatively. A static-offset/static-size partial window over a dynamic parent
stays full-tensor, because the same compiled graph may run with a smaller
dynamic extent. Dynamic-offset windows, such as KV-cache writes at a runtime
slot, can still be rewritten when the static proof and policy gate allow them.

For debugging, `PYPTO_WINDOW_EXTERNALIZE_INCLUDE` and
`PYPTO_WINDOW_EXTERNALIZE_EXCLUDE` filter candidates by callee or parameter
name. `PYPTO_WINDOW_EXTERNALIZE_LOG=1` prints `auto` accept/reject decisions.

## Patterns

The pass applies five patterns in sequence. Each pattern sees the results of the previous one.

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

### Pattern 3: Assemble-Loop Rewrite (AssembleLoopRewriter)

**Problem**: An InCore function contains a `for` loop that accumulates results via `tile.assemble` into an iter-arg, then stores the final result. The `tile.assemble` creates intermediate tile copies each iteration.

**Solution**: Rewrite the loop body to use `tile.store` directly (writing into the `Out` param), initializing the iter-arg from the `Out` param instead of a `tile.create`.

### Pattern 4: Slice Input Strides (SliceInputStridesOptimizer)

**Problem**: When orchestration passes a sliced tensor (`tensor.slice`) as an `In` argument to an InCore function, the InCore function's parameter has contiguous strides (computed from its own shape), not the parent tensor's strides. This causes incorrect memory access when the slice is a non-contiguous view of the parent.

**Solution**: Analyze `tensor.slice(parent, size, offset)` patterns in orchestration. When a slice result is passed as an `In` argument to an InCore call, attach the parent tensor's shape-derived strides via `TensorView` on the InCore function's `In` param type, so `tile.load` uses the correct memory layout.

### Pattern 5: Static Window Externalization (OutWindowExternalizer)

**Problem**: An outlined callee may write only a statically provable local window of a large `Out` tensor, or consume only a statically provable local window of a large `In` tensor, but the call site still passes the whole tensor. Downstream dependence analysis then sees whole-buffer accesses and may add unnecessary serialization.

**Solution**: Clone the callee to a `__windowed` variant with narrowed rewritten tensor parameter types and localized internal offsets. Rewrite the orchestration call site to explicit local slices. Output windows use `slice + __windowed call + assemble`:

```python
out_window = pl.tensor.slice(out, shape, offset)
out_window_next = self.kernel__windowed(..., out_window)
out = pl.tensor.assemble(out, out_window_next, offset)
```

Input windows use the same call-site-local slice materialization, without an assemble:

```python
in_window = pl.tensor.slice(inp, shape, offset)
result = self.consumer__windowed(in_window, ...)
```

When a materialized slice would otherwise use a loop-return alias as its parent,
the pass rewrites that parent to the loop's visible init tensor for both
`ForStmt` and `WhileStmt`. This keeps generated orchestration C++ from
referencing loop-return SSA names outside their scope. Loop-carried iter-args
inside the loop body are not folded this way.

This pass intentionally keeps window eligibility conservative. It does not special-case operator names such as `topk`; a tensor is windowed only when the callee body proves the access pattern below.

After static eligibility, the default `auto` policy applies one more conservative
cost gate. It rejects candidates that would increase the number of rewritten
tensor arguments and rejects multi-piece output rewrites by default. A
dynamic-indexed input is windowed only when a proven writer-reader carrier flow
exists and the reader's effective `window_inputs` policy allows any required
extra carrier args; otherwise the reader keeps the full parent tensor. This keeps the
default pipeline from trading local window precision for larger dispatch
signatures, extra call-site orchestration, or private dynamic-window scans whose
performance benefit is workload-dependent. Use `window_policy="all"` when the
intent is "cut every correctness-proven window even if submit signatures grow";
use reader `window_inputs="all"` for a local carrier-reader opt-in.

Supported rewrite shapes:

- `FinalStore`: the callee returns the result of a final `tile.store(...)` into one local window
- `AggregateWindowLoop`: the callee carries one or more `Out` tensors through a loop and writes a statically provable aggregate window, such as the outlined `kv_proj` group shape
- `PureInputWindowConsumer`: an `In` tensor parameter in a data-returning callee is used only through the same local input window
- `AggregateInputWindowLoop`: together with an `AggregateWindowLoop` output rewrite, an `In` tensor parameter is read only through loop-local `tile.load`/`tensor.slice` windows whose offsets expand across that same internal loop into one statically provable parent-shaped region, such as q/k inputs of qk norm
- `RuntimeCurrentAggregator`: when a surviving carrier-producing `__windowed` producer loop feeds a later repeated full-tensor `In` reader, the pass inserts a small runtime-visible current marker. Plain `coalesce` is only an output bounding-box policy. The match is attrs-driven and does not depend on model-specific callee or variable names.

Output-window eligibility:

- the write must be a statically provable local `tile.store` window or aggregate window loop
- window shape and offset must be statically known enough to materialize a `tensor.slice`
- offsets must be affine in the surrounding loop variables accepted by the pass
- multi-`Out` rewrites are all-or-nothing
- if multiple externalized `Out` params at the same callsite resolve to the same parent tensor, that callsite stays full-tensor; Pattern 5 does not chain multiple `tensor.assemble` updates into one parent state
- sequential-loop siblings are rewritten only when every rewritten `Out` can be proven disjoint across sibling iterations
- same-scope sibling writers to the same parent or aliased parent tensor may still be externalized when each individual writer satisfies the static output-window eligibility rules; however, if that parent also has a sibling full writer (`Out` or `InOut`) that cannot be externalized as an output window, other writers to the same parent stay full-tensor so the non-window writer does not hide partially initialized regions
- write/write and write/read ordering for the remaining windowed writers is delegated to runtime TensorMap overlap on the actually submitted window descriptors
- sibling-writer alias collection descends into nested `SeqStmts`, `ForStmt`, `WhileStmt`, and `IfStmt` bodies, so tensor aliases such as loop returns and tuple projections are resolved to the visible parent before call-site slicing
- later full-parent reads do not disable output windowing; correctness is delegated to runtime TensorMap overlap dependence once the call site exposes the actual window tensor

Input-window eligibility:

- the parameter must be an `In` tensor
- every reference to that parameter inside the callee must match the same local window
- supported references are `tile.load` and `tensor.slice`
- transpose loads are rejected
- the `tile.load` read shape must equal the candidate window shape
- all matched references must agree on window shape and offset
- if any reference is unsupported, the whole input parameter stays full-tensor
- pure input-window shape and callee-local offset expressions may reference only callee params; after call-site substitution, those params may carry outer loop-affine values, and the windowed callee reads relative to `[0, ...]`
- for `PureInputWindowConsumer`, if the matched window is full shape at zero offset, the pass skips it because slicing would not expose a narrower dependency
- for `PureInputWindowConsumer`, callees with no data return stay full-tensor because such consumers may be side-effect or fence tasks whose full input intentionally carries a wider dependency
- input-only `Submit` callsites stay full-tensor; inside `manual_scope`, a full input may intentionally carry a wider dependency even when the callee body reads a local window
- when a callee also has an eligible output-window rewrite, any already proven pure input windows are preserved and materialized at the same callsite
- for `AggregateInputWindowLoop`, all references must be inside one static `ForStmt`, at least one offset dimension must vary with that loop, and the aggregate window must equal the input parent shape; partial aggregate reads such as weight sub-windows remain full-tensor
- dynamic-indexed reader windows require a provable min/max scan, a writer whose output rewrite survives as a `__windowed` call, and a reader policy that allows any required carrier args. Eligible writer-reader chains with a carrier use the carrier/remat path; without that proven flow, the dynamic reader keeps full parent tensor (no private dynamic reader).

Non-goals and dependence model:

- the pass does not add explicit dependency edges
- the pass does not reintroduce a later full-parent-read guard
- the pass does not precompute global window descriptor arrays
- the pass does not split SPMD launches or externalize per-block SPMD windows
- unsupported consumers, including full-tensor readers, remain baseline/full-tensor inputs
- selected prefill full-tensor readers may still receive a runtime-current marker when the producer loop is a surviving carrier-producing fan-in pattern; this does not narrow the reader region, but gives runtime dependency tracking one current tensor to match instead of many producer windows
- `DeriveCallDirections` keeps its existing sound sequential `Out -> InOut` rule; Pattern 5 only exposes proven local windows before that pass runs

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
| `OutWindowExternalizer` | Pattern 5 — rewrites eligible local Out writes and eligible In-window consumers to explicit call-site slices |
| `BuildOutParamReturnMappings` | Shared helper — maps Out params to return indices via tile.store |
| `ComputeRowMajorStrides` | Shared helper — computes row-major strides from a shape |

## Scope

| Function type | Action |
| ------------- | ------ |
| InCore / outlined non-builtin callee | Params/body rewritten (Patterns 1, 3, 4, 5) |
| Orchestration / Opaque | Call sites rewritten (Patterns 1, 2, 5) |
| Group | Unchanged |
