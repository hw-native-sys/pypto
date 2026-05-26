# DeriveCallDirections Pass

Derives per-argument `ArgDirection` for every cross-function `Call` based on callee `ParamDirection` and buffer lineage.

## Overview

PyPTO uses a **two-layer direction model** (introduced in commit `c53dac0d`):

- `ParamDirection` (`In` / `Out` / `InOut`) lives on the callee `Function` and describes the function-signature contract — *"I read/write this parameter."*
- `ArgDirection` (`Input` / `Output` / `InOut` / `OutputExisting` / `NoDep` / `Scalar`) lives on each `Call` site and describes the runtime task-submission semantics — *"this submission establishes these dependencies and uses this memory ownership model."*

The two layers must agree but are not identical: under `DeriveCallDirections`, a callee `Out` parameter may become either `OutputExisting` or `InOut` at the call site depending on whether other writers have already touched the same buffer. `ArgDirection::Output` is reserved for explicitly populated call sites where the runtime should allocate a fresh output buffer; this pass never infers it.

`DeriveCallDirections` is the pass that bridges the two layers. It walks every non-builtin `Call` in every `Function` body and writes the resolved per-argument vector to `Call.attrs["arg_directions"]` (the reserved key `kAttrArgDirections`, value type `std::vector<ArgDirection>`). Downstream consumers — orchestration codegen and the runtime task-submission layer — read this vector instead of recomputing it from raw param directions.

**When to use**: Run after the tile pipeline has stabilized (`SplitIncoreOrch` is required) and before any consumer that observes `Call.attrs["arg_directions"]`. In the `Default` strategy it sits between `FuseCreateAssembleToSlice` and the final `Simplify` (doc slot 28 — the 28th pipeline pass excluding utility passes; see `.claude/rules/pass-doc-ordering.md` for the numbering scheme).

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

The pass is a `ProgramPass` that runs three phases per `Function` body.

### 1. Buffer-root collection

`BufferRootCollector` (defined in `include/pypto/codegen/orchestration/orchestration_analysis.h`) walks the function body and maps every `Var*` to the `Var*` that owns its underlying buffer, propagating root identity through assignments, loops, and call outputs. The pass also builds a `param_vars` set from the function's formal parameters for fast *"rooted at a function param?"* lookups.

### 2. Prior-writer analysis

`PriorWriterCollector` decides, per `(Call, local-root)`, whether the call is the *first writer* of that root within its enclosing scope. It runs in two phases:

1. **Bottom-up cache** (`PrecomputeWrittenRoots`): for every subtree, cache the union of locally allocated roots written by any non-builtin `Call` inside it. The result becomes the *writer footprint* of that subtree when it appears as a sibling in an outer scope.
2. **Top-down scan** (`AnalyzeScope`): walk the IR maintaining a `seen_roots` set of roots already written by prior siblings. For each `Call`, every callee-`Out` argument whose root is *not* in `seen_roots` is recorded as a first writer. Every `ForStmt` (regardless of `ForKind`) / `WhileStmt` / `IfStmt` is entered with a *snapshot copy* of `seen_roots` (so writes inside the unit do not leak into sibling tracking) and treated as an opaque writer-unit; `ScopeStmt` and `SeqStmts` share the same `seen_roots`.

### 3. Direction rewrite

`CallDirectionMutator` walks every non-builtin `Call`. For Group/Spmd callees the effective per-position directions are recovered via `ComputeGroupEffectiveDirections` (`orchestration_analysis.h`); other callees use their declared `param_directions_`. A `sequential_depth_` counter is incremented on non-`Parallel` `For` and on `While`, driving the *R-seq* promotion below.

For each positional argument the mutator picks a direction by this table:

| Callee `ParamDirection` | Argument origin | `sequential_depth > 0`? | Prior writer in scope? | Result |
| ----------------------- | --------------- | ----------------------- | ---------------------- | ------ |
| any | non-tensor | — | — | `Scalar` |
| `In` | tensor | — | — | `Input` |
| `InOut` | tensor | — | — | `InOut` |
| `Out` | rooted at param | — | — | `OutputExisting` |
| `Out` | local buffer | yes (R-seq) | — | `InOut` |
| `Out` | local buffer | no | yes (R-prior) | `InOut` |
| `Out` | local buffer | no | no | `OutputExisting` |

**R-seq** keeps cross-iteration write-after-write chains correct inside sequential loops: the same buffer slot is written once per iteration, so the runtime must serialize iterations on it. **R-prior** preserves the cross-sibling WAW dependency when an earlier writer-unit in the same scope already touched the same root.

A pre-populated `Call.attrs["arg_directions"]` is treated as authoritative and left untouched (some directions like `NoDep` are not derivable structurally). The `Call` constructor's `ValidateArgDirectionsAttr` only enforces arity when the vector is non-empty; an empty vector can still be attached and will later be rejected by the `CallDirectionsResolved` verifier.

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

- `PriorWriterCollector` — per-scope first-writer analysis (bottom-up cache + top-down scan)
- `CallDirectionMutator` — `IRMutator` that rewrites every non-builtin `Call` with the resolved `arg_directions` vector
- Reuses `BufferRootCollector` and `ComputeGroupEffectiveDirections` from `include/pypto/codegen/orchestration/orchestration_analysis.h`

**Property verifier**: `src/ir/verifier/verify_call_directions.cpp` (factory in `include/pypto/ir/verifier/verifier.h`)

**Python binding**: `python/bindings/modules/passes.cpp`

```cpp
passes.def("derive_call_directions", &pass::DeriveCallDirections,
           "Derive Call attrs['arg_directions'] from callee param directions and buffer lineage. ...");
```

**Type stub**: `python/pypto/pypto_core/passes.pyi`

**Hand-written-IR helper**: `python/pypto/ir/directions.py` (`make_call`, lowercase aliases) — for tests and hand-built IR fragments that want to attach explicit directions before the pass runs.

**Tests**: `tests/ut/ir/transforms/test_derive_call_directions.py`

- `TestDeriveDirectionMatrix` — one test per cell of the (callee_dir, origin) → ArgDirection mapping table, including R-seq (`pl.range`, `while`) and R-prior (top-level + branch / parallel-after-top-level) edge cases
- `TestDeriveIdempotent` — running the pass twice yields structurally equal IR
- `TestDerivePreservesExplicit` — pre-populated `arg_directions` is not overwritten
- `TestVerifyPositive` / `TestVerifyNegative` — the `CallDirectionsResolved` property verifier accepts the pass output and rejects ill-formed `arg_directions` assignments
