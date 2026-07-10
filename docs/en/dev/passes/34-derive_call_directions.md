# DeriveCallDirections Pass

A pass that first materializes the effective `ParamDirection` of every Group/Spmd wrapper into its signature, then walks each `Function` body deriving a per-argument `ArgDirection` for every cross-function `Call` from callee `ParamDirection` and buffer lineage, writing the resolved vector to `Call.attrs["arg_directions"]`. It does **not** touch manual-scope dependency edges.

## Overview

PyPTO uses a **two-layer direction model** (introduced in commit `c53dac0d`):

- `ParamDirection` (`In` / `Out` / `InOut`) lives on the callee `Function` and describes the function-signature contract — *"I read/write this parameter."*
- `ArgDirection` (`Input` / `Output` / `InOut` / `OutputExisting` / `NoDep` / `Scalar`) lives on each `Call` site and describes the runtime task-submission semantics — *"this submission establishes these dependencies and uses this memory ownership model."*

The two layers must agree but are not identical: under `DeriveCallDirections`, a callee `Out` parameter may become either `OutputExisting` or `InOut` at the call site depending on whether other writers have already touched the same buffer. `ArgDirection::Output` is reserved for explicitly populated call sites where the runtime should allocate a fresh output buffer; this pass never infers it.

`DeriveCallDirections` is the pass that bridges the two layers. It walks every non-builtin `Call` in every `Function` body and writes the resolved per-argument vector to `Call.attrs["arg_directions"]` (the reserved key `kAttrArgDirections`, value type `std::vector<ArgDirection>`). Downstream consumers — orchestration codegen and the runtime task-submission layer — read `Call.attrs["arg_directions"]` instead of recomputing it from raw param directions.

**Submit is preserved, not lowered.** A task launch — `pl.submit(...)` inside `pl.manual_scope`, or a captured auto-scope dispatch (`with pl.at(...) as tid:` / `with pl.spmd(...) as tid:`) — is an `ir.Submit`, a sibling kind of `Call`. `DeriveCallDirections` derives `arg_directions` for the `Submit` (via a transient `SubmitToCallView` used only to inspect args) and re-attaches them to a **fresh `Submit`**, keeping the typed `deps_` field and the TASK_ID-augmented `Tuple[<outputs>..., Scalar[TASK_ID]]` return shape (pass-submit-awareness.md rule 3). Lowering `Submit → Call` here would produce a plain `Call` carrying a Tuple type its callee never declares — a malformed node that cannot survive print → reparse. Downstream consumers (orchestration codegen, `ExpandManualPhaseFence`, `CollectCommGroups`, the `CallDirectionsResolved` verifier) funnel `Submit` through `SubmitToCallView` where they need the Call-shaped view.

**Manual-scope deps are a separate layer.** A `Submit` carries its `deps=[...]` edges in the first-class `deps_` field. The attrs encoding — `Call.attrs["manual_dep_edges"]` (a `vector<VarPtr>` of `Scalar[TASK_ID]` / `Array[N, TASK_ID]` entries) — exists only inside the transient `SubmitToCallView`, which synthesises it from `deps_` for Call-shaped consumers; no plain cross-function `Call` in the IR carries it (the ManualDepsOnSubmitOnly structural property verifies this). `DeriveCallDirections` reads and writes only `arg_directions`; the later `ExpandManualPhaseFence` pass may rewrite a consumer's full TaskId-array dep to a dummy-barrier TaskId (preserving the consumer's kind: a `Submit` stays a `Submit`).

**When to use**: Run after the tile pipeline has stabilized (`SplitIncoreOrch` is required) and before any consumer that observes `Call.attrs["arg_directions"]`. In the `Default` strategy it sits between `FuseCreateAssembleToSlice` and the final `Simplify`.

## Properties

| Required | Produced | Invalidated |
| -------- | -------- | ----------- |
| `SplitIncoreOrch` | `CallDirectionsResolved` | — |

The `CallDirectionsResolved` property is verified by the registered `CallDirectionsResolved` property verifier (factory `CreateCallDirectionsResolvedPropertyVerifier()` in `src/ir/verifier/verify_call_directions.cpp`), so the pipeline auto-checks the integrity of the produced `arg_directions` after this pass runs — no separate verify pass exists. See [Verifier](99-verifier.md).

## API

| C++ | Python | Level |
| --- | ------ | ----- |
| `pass::DeriveCallDirections()` | `passes.derive_call_directions()` | Program-level |

**Factory function**:

```cpp
Pass DeriveCallDirections();
```

**Python usage**:

```python
from pypto.pypto_core import passes

derive_pass = passes.derive_call_directions()
program_with_dirs = derive_pass(program)
```

## Algorithm

The pass is a `ProgramPass`. It first rewrites Group/Spmd wrapper signatures program-wide (phase 0), then runs three sub-passes over each `Function` body.

### 0. Wrapper direction materialization

A Group/Spmd wrapper forwards its params 1:1 to an inner kernel call, but its own `param_directions_` can still read `In` for a param the inner kernel writes — the scope outliners infer directions from the body they extract, and later passes (`ExpandMixedKernel`, `SplitVectorKernel`) rebuild wrapper bodies around freshly split callees without revisiting the signature.

`MaterializeWrapperDirections` calls `ComputeWrapperEffectiveDirections` (`include/pypto/ir/transforms/utils/wrapper_call_utils.h`) once for the whole program, then writes each result back into `Function::param_directions_`. For each wrapper that helper walks the inner calls, matches each arg to a wrapper param by `Var` pointer identity, and merges the inner callee's direction with the lattice `InOut` > `Out` > `In`. Nested Group→Group chains resolve recursively, with a cycle guard that falls back to the declared directions. One shared memo across the whole program means each wrapper body is visited once, keeping the phase linear in total body size.

Every wrapper is resolved against the *pre-phase-0 snapshot* of the program, so the result does not depend on function map iteration order.

This establishes the invariant that **`callee->param_directions_` is the single source of truth for every callee kind**. The three sub-passes below, plus `AutoDeriveTaskDependencies`, the `CallDirectionsResolved` verifier, and orchestration codegen, all read that field directly instead of recomputing the effective view — which is what keeps codegen a strict 1-to-1 translation. The verifier re-checks the invariant on every program it inspects.

### 1. Buffer-root collection

`BufferRootCollector` (defined in `include/pypto/codegen/orchestration/orchestration_analysis.h`) walks the function body and maps every `Var*` to the `Var*` that owns its underlying buffer, propagating root identity through assignments, loops, and call outputs. The pass also builds a `param_vars` set from the function's formal parameters for fast *"rooted at a function param?"* lookups.

### 2. Prior-writer analysis

`PriorWriterCollector` decides, per `(Call, local-root)`, whether the call is the *first writer* of that root within its enclosing scope. It runs in two phases:

1. **Bottom-up cache** (`PrecomputeWrittenRoots`): for every subtree, cache the union of locally allocated roots written by any non-builtin `Call` inside it. The result becomes the *writer footprint* of that subtree when it appears as a sibling in an outer scope.
2. **Top-down scan** (`AnalyzeScope`): walk the IR maintaining a `seen_roots` set of roots already written by prior siblings. For each `Call`, every callee-`Out` argument whose root is *not* in `seen_roots` is recorded as a first writer. Every `ForStmt` (regardless of `ForKind`) / `WhileStmt` / `IfStmt` is entered with a *snapshot copy* of `seen_roots` (so writes inside the unit do not leak into sibling tracking) and treated as an opaque writer-unit; `ScopeStmt` and `SeqStmts` share the same `seen_roots`.

### 3. Direction rewrite

`CallDirectionMutator` walks every non-builtin `Call`, reading the callee's `param_directions_` — correct for every callee kind, since phase 0 already materialized the Group/Spmd wrappers. A `sequential_depth_` counter is incremented on non-`Parallel` `For` and on `While`, driving the *R-seq* promotion below.

For each positional argument the mutator picks a direction by this table. A callee `Out` is resolved by trying three promotion rules in order — R-seq → R-prior → R-enclosing; if none fires it stays `OutputExisting`:

| Callee `ParamDirection` | Argument | `sequential_depth > 0`? | Prior writer in scope? | Enclosing param `InOut`? | Result |
| ----------------------- | -------- | ----------------------- | ---------------------- | ------------------------ | ------ |
| any | non-tensor | — | — | — | `Scalar` |
| `In` | tensor | — | — | — | `Input` |
| `InOut` | tensor | — | — | — | `InOut` |
| `Out` | tensor | yes (R-seq) | — | — | `InOut` |
| `Out` | tensor | no | yes (R-prior) | — | `InOut` |
| `Out` | tensor | no | no | yes (R-enclosing) | `InOut` |
| `Out` | tensor | no | no | no | `OutputExisting` |

**R-seq** keeps cross-iteration write-after-write chains correct inside sequential loops: a callee `Out` under any sequential ancestor is promoted to `InOut` **unconditionally**. An earlier "disjoint variable-offset store" exception — which kept such a call as `OutputExisting` when the callee's `tile.store` offset depended on a parameter — was removed: soundly proving that cross-iteration writes are disjoint needs a real dependence analysis (affine offset extraction, stride-vs-tile-extent, offset injectivity, cross-procedural composition), and the cheap syntactic check it used could silently drop real WAW edges. **R-prior** preserves the cross-sibling WAW dependency when an earlier writer-unit in the same scope already touched the same root. **R-enclosing** honours an explicit `pl.InOut` declaration on the enclosing function parameter that the argument is rooted at.

A pre-populated `Call.attrs["arg_directions"]` is treated as authoritative and left untouched (some directions like `NoDep` are not derivable structurally). The `Call` / `Submit` constructor's `ValidateArgDirectionsAttr` only enforces arity when the vector is non-empty. The `CallDirectionsResolved` verifier requires a populated `arg_directions` vector for every call-like node **with arguments**; a 0-arg dispatch (e.g. a bare `pl.submit(self.kernel)` whose callee takes no positional tensor args) legitimately has an empty vector and is accepted.

**Idempotency**: the mutator skips any call that already has `attrs["arg_directions"]` (`HasArgDirections()`), so a second run leaves resolved calls untouched. Running the pass twice therefore produces structurally identical IR (regression-tested by `TestDeriveIdempotent::test_idempotent`).

## Example

Two consecutive calls writing the same locally allocated buffer at top level. The first is the only writer-unit so it stays `OutputExisting`; the second hits R-prior and is promoted to `InOut` so the runtime preserves the cross-call WAW dependency on `local`.

### Before

```python
@pl.program
class Prog:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        x: pl.Tensor[[64], pl.FP32],
        out: pl.Out[pl.Tensor[[64], pl.FP32]],
    ) -> pl.Tensor[[64], pl.FP32]:
        t: pl.Tile[[64], pl.FP32] = pl.load(x, [0], [64])
        ret: pl.Tensor[[64], pl.FP32] = pl.store(t, [0], out)
        return ret

    @pl.function
    def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        local: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
        local = self.kernel(x, local)   # arg_directions = []  (pre-pass)
        local = self.kernel(x, local)   # arg_directions = []  (pre-pass)
        return local
```

### After

```python
# Same IR shape; only Call.attrs["arg_directions"] changes:
local = self.kernel(x, local)   # arg_directions = [Input, OutputExisting]
local = self.kernel(x, local)   # arg_directions = [Input, InOut]
```

The `kernel` callee declares `Out` for parameter `out`. Because `local` is locally allocated (rooted at `pl.create_tensor`, not at a `main` parameter), the first call gets `OutputExisting` (no sequential ancestor, no prior writer-unit) while the second sees a prior writer in the same scope and is promoted to `InOut`.

## Implementation

**Header**: `include/pypto/ir/transforms/passes.h`

```cpp
Pass DeriveCallDirections();
```

**Properties**: `include/pypto/ir/transforms/pass_properties.h`

```cpp
inline const PassProperties kDeriveCallDirectionsProperties{
    .required = {IRProperty::SplitIncoreOrch},
    .produced = {IRProperty::CallDirectionsResolved}};
```

**Implementation**: `src/ir/transforms/derive_call_directions_pass.cpp`

- `MaterializeWrapperDirections` — phase 0: writes effective Group/Spmd directions into `Function::param_directions_`
- `PriorWriterCollector` — per-scope first-writer analysis (bottom-up cache + top-down scan)
- `CallDirectionMutator` — `IRMutator` that rewrites every non-builtin `Call` with the resolved `arg_directions` vector
- Reuses `ComputeWrapperEffectiveDirections` from `include/pypto/ir/transforms/utils/wrapper_call_utils.h` and `BufferRootCollector` from `include/pypto/codegen/orchestration/orchestration_analysis.h`

**Property verifier**: `src/ir/verifier/verify_call_directions.cpp` (factory in `include/pypto/ir/verifier/verifier.h`)

**Python binding**: `python/bindings/modules/passes.cpp`

```cpp
passes.def("derive_call_directions", &pass::DeriveCallDirections,
           "Derive Call attrs['arg_directions'] from callee param directions and buffer lineage.");
```

**Type stub**: `python/pypto/pypto_core/passes.pyi`

**Hand-written-IR helper**: `python/pypto/ir/directions.py` (`make_call`, lowercase aliases) — for tests and hand-built IR fragments that want to attach explicit directions before the pass runs.

**Tests**: `tests/ut/ir/transforms/test_derive_call_directions.py`

- `TestDeriveDirectionMatrix` — one test per cell of the (callee_dir, origin) → ArgDirection mapping table, including R-seq (`pl.range`, `while`) and R-prior (top-level + branch / parallel-after-top-level) edge cases
- `TestDeriveIdempotent` — running the pass twice yields structurally equal IR
- `TestDerivePreservesExplicit` — pre-populated `arg_directions` is not overwritten
- `TestVerifyPositive` / `TestVerifyNegative` — the `CallDirectionsResolved` property verifier accepts the pass output and rejects ill-formed `arg_directions` assignments
