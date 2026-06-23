# OptimizeOrchTensors Pass

Optimizes tensor buffer usage across orchestration and InCore functions by eliminating redundant allocations and improving data flow.

## Overview

After `ConvertTensorToTileOps`, orchestration functions allocate output tensors (`tensor.create`) at every InCore call site, even inside loops where the same buffer could be reused. This pass applies six optimization/lowering patterns to reduce allocations, improve buffer layout information, make statically provable local tensor windows explicit at orchestration call sites, and lower proven linked-flow runtime-current markers.

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

Most users should choose one preset with `window_option`:

```python
passes.optimize_orch_tensors(window_option="stable")
```

| `window_option` | Expanded config | Meaning |
| --------------- | --------------- | ------- |
| `"none"` | `window_policy="none", window_flow="local"` | Do not window by default. Explicit kernel attrs can still opt in. |
| `"stable"` | `window_policy="stable", window_flow="local"` | Default. Use stable exact windows without expanding runtime submit args. |
| `"exact"` | `window_policy="exact", window_flow="local"` | Allow exact windows that expand submit args. No boundingBox holes and no linked flow. |

`window_option="none"` is not a kill switch; it only changes the global
default. A kernel with explicit `window_outputs`, `window_inputs`, or
`window_flow` attrs may still opt in.

Advanced users can use the explicit two-axis API:

```python
passes.optimize_orch_tensors(window_policy="stable", window_flow="local")
```

Do not pass `window_option` together with `window_policy` or `window_flow`.
Use either the preset API or the explicit two-axis API.

`window_policy` controls the coverage shape and submit-argument budget for one
call site. `window_flow` controls whether a proven coverage relationship may be
linked across writer and reader call sites. Carrier/base/extent/current/remat
is not a user-facing option; it is one possible lowering of a proven linked
flow.

`boundingBox` and `linked` are intentionally not `window_option` presets. Use
explicit `window_policy="boundingBox"` and/or `window_flow="linked"` when the
larger coverage/flow semantics are desired.

| `window_policy` | Meaning |
| --------------- | ------- |
| `"none"` | Do not window by default. Explicit kernel attrs can still opt in. |
| `"stable"` | Default. Use only ABI-safe exact windows; do not expand submit tensor/scalar args. |
| `"exact"` | Use correctness-proven exact pieces; submit signatures may expand. No boundingBox holes. |
| `"boundingBox"` | Exact-first, but may use a continuous boundingBox when it reduces view/argument count or linked proof needs continuous coverage. Holes are explicitly allowed. |

| `window_flow` | Meaning |
| ------------- | ------- |
| `"local"` | Rewrite each call site independently. No cross-callsite current/carrier lowering. |
| `"linked"` | Allow proven writer-reader coverage to be linked when both endpoints opt in. Proof failure falls back; it is not an error. |

Useful global combinations:

| Global config | Submit args | Coverage | Holes | Linked flow | Carrier/base/extent/barrier |
| ------------- | ----------- | -------- | ----- | ----------- | --------------------------- |
| `none + local` | unchanged by default | no default windows | no | no | no |
| `none + linked` | unchanged by default | only explicit opt-in kernels | per opt-in | only opt-in writer/reader edges with both effective flows linked | only if the opt-in edge proves valid |
| `stable + local` | no expansion | ABI-safe exact | no | no | no |
| `stable + linked` | no expansion | ABI-safe exact | no | proof may run, but most dynamic readers fall back | no base/extent/barrier if new args would be needed |
| `exact + local` | may expand | exact pieces | no | no | no |
| `exact + linked` | may expand | exact pieces | no | single dense exact linked coverage only | no implicit boundingBox carrier |
| `boundingBox + local` | may expand | exact or boundingBox | yes | no | no |
| `boundingBox + linked` | may expand | exact or boundingBox | yes | proven writer-reader linked coverage | carrier/base/extent/remat and barriers may appear |

Linked flow has two independent lowering outcomes:

- **Dynamic range forwarding** may materialize carrier base/extent and remat a
  continuous writer window for a dynamic reader.
- **Aggregate current joining** may insert a runtime-current barrier between a
  loop-produced tensor and a later full-parent or inexact consumer.

An aggregate barrier does not require a dynamic carrier. Disabling one
writer/reader carrier path does not disable unrelated proven aggregate joins.
Neither lowering is permitted under `window_flow="local"`. Under
`stable + linked`, any lowering that would expand submit arguments falls back.

**Per-kernel overrides** via function attrs:

```python
@pl.function(
    type=pl.FunctionType.InCore,
    attrs={
        "window_outputs": "exact",       # off | stable | exact | boundingBox
        "window_inputs":  "off",         # off | stable | exact | boundingBox
        "window_flow":    "linked",      # local | linked
    }
)
def my_kernel(...): ...
```

Missing attrs inherit the global setting. Explicit side attrs win, including
opt-in when the global default is `"none"`. `window_outputs` and
`window_inputs` are independent coverage permissions. `InOut` dependency
coverage may be analyzed as read coverage plus write coverage, but runtime
submit still has one TensorView for that argument; if the read and write submit
views conflict, the pass uses a conservative view or falls back.

Set both side attrs to `"off"` to disable all local window externalization for
one kernel:

```python
attrs={"window_outputs": "off", "window_inputs": "off"}
```

`boundingBox` is permission, not a force mode. Exact coverage is preferred by
default. A boundingBox is selected only when exact pieces exceed the
submit/view expression budget, linked proof needs one continuous coverage, or a
mechanical local cost rule shows fewer runtime tensor views/params. If the
boundingBox equals the full parent or gives no expected benefit, the pass falls
back to exact or full parent without error.

```python
attrs={"window_outputs": "boundingBox"}
```

`exact` is also permission, not a force mode. Separated exact pieces may become
separate runtime tensor arguments. The pass rejects a rewrite that would exceed
the runtime limit of 32 tensor arguments or 16 scalar arguments, but a legal
multi-piece rewrite can still increase scheduling overhead substantially.
Inspect generated orchestration when opting a multi-piece kernel into `exact`.

`window_flow="linked"` is not a carrier switch. It only permits a writer-reader
edge to reuse coverage when both endpoints are effectively linked and the proof
succeeds. The lowering may use carrier/base/extent/remat today; future lowering
forms may differ.

```python
attrs={"window_inputs": "boundingBox", "window_flow": "linked"}
```

For example, in a Qwen prefill-style dynamic-indexed KV-cache writer/reader
chain, the expected boundary is:

- `stable + local`: keep ABI-safe local exact windows only. Dynamic KV-cache
  readers stay full-tensor.
- `exact + local`: allow exact pieces even when submit signatures grow, but do
  not create cross-callsite carrier lowering.
- `boundingBox + local`: allow local continuous coverage with holes, but still
  no carrier/base/extent.
- `boundingBox + linked`: allow qk/sv-style dynamic readers to use a proven
  output-derived linked flow. If proof fails, the reader stays full parent.

The linked dynamic-reader path depends on all of these conditions:

- the producer's effective `window_outputs` can provide the required exact or
  boundingBox coverage
- the reader's effective `window_inputs` permits the required coverage
- both endpoints have effective `window_flow="linked"`
- the writer/reader pair satisfies the shared-root, loop-order, and min/max
  proof checks

If any condition fails, the dynamic reader keeps the full parent tensor and the
pass must not insert a runtime-current barrier just because coverage attrs are
present.

Legacy configuration spellings are accepted only where an unambiguous mapping
exists: global `auto` maps to `stable`, and side `coalesce` maps to
`boundingBox`. New code should use the canonical names. Ambiguous settings are
rejected: `all`, `carrier`, `coalesce_carrier`, and the kernel-level
`window_policy` attr. Replace them with explicit `window_outputs`,
`window_inputs`, and `window_flow` settings.

Output windows whose parent shape has dynamic dimensions are handled
conservatively. A static-offset/static-size partial window over a dynamic parent
stays full-tensor, because the same compiled graph may run with a smaller
dynamic extent. Dynamic-offset windows, such as KV-cache writes at a runtime
slot, can still be rewritten when the static proof and policy gate allow them.

Diagnostic-only environment variables are available while debugging this pass:
`PYPTO_WINDOW_EXTERNALIZE_INCLUDE` and `PYPTO_WINDOW_EXTERNALIZE_EXCLUDE`
filter candidates by callee or parameter name, and
`PYPTO_WINDOW_EXTERNALIZE_LOG=1` prints stable/coverage accept/reject
decisions. These variables are not part of the public policy semantics.

## Patterns

The pass applies six patterns in sequence. Each pattern sees the results of the previous one.

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

After static eligibility, the default `stable + local` policy applies one more
conservative cost gate. It keeps only exact windows that do not increase
`add_inout`, `add_input`, `add_output`, or `add_scalar` submit budgets.
Multi-piece output rewrites, boundingBox coverage, dynamic carrier args, and
cross-callsite current/remat are outside the stable default. Use
`window_policy="exact"` when exact pieces may expand submit signatures. Use
`window_policy="boundingBox"` when continuous coverage with holes is acceptable.
Use `window_flow="linked"` only when proven writer-reader coverage may be
propagated across call sites.

Supported rewrite shapes:

- `FinalStore`: the callee returns the result of a final `tile.store(...)` into one local window
- `AggregateWindowLoop`: the callee carries one or more `Out` tensors through a loop and writes a statically provable aggregate window, such as the outlined `kv_proj` group shape
- `PureInputWindowConsumer`: an `In` tensor parameter in a data-returning callee is used only through the same local input window
- `AggregateInputWindowLoop`: together with an `AggregateWindowLoop` output rewrite, an `In` tensor parameter is read only through loop-local `tile.load`/`tensor.slice` windows whose offsets expand across that same internal loop into one statically provable parent-shaped region, such as q/k inputs of qk norm
- Linked-flow lowering: when both writer and reader endpoints opt into linked
  flow and proof succeeds, the pass may lower the relationship with
  carrier/base/extent/remat. Runtime-current markers are emitted only by that
  explicit linked-flow plan; they must not be inferred from callee names or from
  the presence of a `__windowed` clone.

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
- dynamic-indexed reader windows require a provable min/max scan, a writer whose
  coverage survives as a `__windowed` call or linked boundingBox variant, reader
  coverage that allows the required shape, and effective linked flow on both
  endpoints. Without that proven flow, the dynamic reader keeps the full parent
  tensor (no private dynamic reader).

Non-goals and dependence model:

- the pass does not add explicit dependency edges
- the pass does not reintroduce a later full-parent-read guard
- the pass does not precompute global window descriptor arrays
- the pass does not split SPMD launches or externalize per-block SPMD windows
- unsupported consumers, including full-tensor readers, remain baseline/full-tensor inputs
- runtime-current markers are emitted only by explicit linked-flow analysis; the
  follow-up lowering pass only materializes those markers into runtime barriers
  and strips the marker calls
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
| `RuntimeCurrentAggregator` | Pattern 6 — lowers explicit linked-flow runtime-current markers to barriers and strips marker calls |
| `BuildOutParamReturnMappings` | Shared helper — maps Out params to return indices via tile.store |
| `ComputeRowMajorStrides` | Shared helper — computes row-major strides from a shape |

## Scope

| Function type | Action |
| ------------- | ------ |
| InCore / outlined non-builtin callee | Params/body rewritten (Patterns 1, 3, 4, 5) |
| Orchestration / Opaque | Call sites rewritten (Patterns 1, 2, 5) |
| Group | Unchanged |
