# DeriveCallDirections Pass

A two-phase pass over each `Function` body: **Phase 1** derives per-argument `ArgDirection` for every cross-function `Call` based on callee `ParamDirection` and buffer lineage, and **Phase 2** lowers user-declared `deps=[...]` edges inside `with pl.manual_scope():` regions into a TaskId-based dependency graph that the orchestration codegen can emit as `params.add_dep(<task_id>)` calls. Phase 2 was a separate pass called `DeriveManualScopeDeps` until it was folded into this pass as an implementation detail; the Python API (`passes.derive_call_directions()`) and the Default-strategy entry are unchanged.

## Overview

PyPTO uses a **two-layer direction model** (introduced in commit `c53dac0d`):

- `ParamDirection` (`In` / `Out` / `InOut`) lives on the callee `Function` and describes the function-signature contract — *"I read/write this parameter."*
- `ArgDirection` (`Input` / `Output` / `InOut` / `OutputExisting` / `NoDep` / `Scalar`) lives on each `Call` site and describes the runtime task-submission semantics — *"this submission establishes these dependencies and uses this memory ownership model."*

The two layers must agree but are not identical: under `DeriveCallDirections`, a callee `Out` parameter may become either `OutputExisting` or `InOut` at the call site depending on whether other writers have already touched the same buffer. `ArgDirection::Output` is reserved for explicitly populated call sites where the runtime should allocate a fresh output buffer; this pass never infers it.

`DeriveCallDirections` is the pass that bridges the two layers. Phase 1 walks every non-builtin `Call` in every `Function` body and writes the resolved per-argument vector to `Call.attrs["arg_directions"]` (the reserved key `kAttrArgDirections`, value type `std::vector<ArgDirection>`). Phase 2 then runs only on function bodies that contain a `RuntimeScopeStmt(manual=true)` and rewrites kernel calls inside the manual scope so they expose TaskId-typed dependency edges. Downstream consumers — orchestration codegen and the runtime task-submission layer — read both `Call.attrs["arg_directions"]` and `Call.attrs["manual_dep_edges"]` instead of recomputing them from raw param directions.

**When to use**: Run after the tile pipeline has stabilized (`SplitIncoreOrch` is required) and before any consumer that observes `Call.attrs["arg_directions"]` / `Call.attrs["manual_dep_edges"]`. In the `Default` strategy it sits between `FuseCreateAssembleToSlice` and the final `Simplify`.

## Properties

| Required | Produced | Invalidated |
| -------- | -------- | ----------- |
| `SplitIncoreOrch` | `CallDirectionsResolved` | — |

The `CallDirectionsResolved` property is verified by the registered `CallDirectionsResolved` property verifier (factory `CreateCallDirectionsResolvedPropertyVerifier()` in `src/ir/verifier/verify_call_directions.cpp`), so the pipeline auto-checks the integrity of the produced `arg_directions` after this pass runs — no separate verify pass exists. See [Verifier](99-verifier.md).

No new IRProperty is introduced for Phase 2: codegen reads `Call.attrs["manual_dep_edges"]` directly. Phase 2 is a no-op when no `RuntimeScopeStmt(manual=true)` exists in the program.

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

The pass is a `ProgramPass`. For each `Function` body, it runs **Phase 1 (arg directions)** unconditionally, then runs **Phase 2 (manual-scope lowering)** only if the body contains a `RuntimeScopeStmt(manual=true)`.

## Phase 1 — Arg directions

Phase 1 runs three sub-passes per `Function` body.

### 1.1 Buffer-root collection

`BufferRootCollector` (defined in `include/pypto/codegen/orchestration/orchestration_analysis.h`) walks the function body and maps every `Var*` to the `Var*` that owns its underlying buffer, propagating root identity through assignments, loops, and call outputs. The pass also builds a `param_vars` set from the function's formal parameters for fast *"rooted at a function param?"* lookups.

### 1.2 Prior-writer analysis

`PriorWriterCollector` decides, per `(Call, local-root)`, whether the call is the *first writer* of that root within its enclosing scope. It runs in two phases:

1. **Bottom-up cache** (`PrecomputeWrittenRoots`): for every subtree, cache the union of locally allocated roots written by any non-builtin `Call` inside it. The result becomes the *writer footprint* of that subtree when it appears as a sibling in an outer scope.
2. **Top-down scan** (`AnalyzeScope`): walk the IR maintaining a `seen_roots` set of roots already written by prior siblings. For each `Call`, every callee-`Out` argument whose root is *not* in `seen_roots` is recorded as a first writer. Every `ForStmt` (regardless of `ForKind`) / `WhileStmt` / `IfStmt` is entered with a *snapshot copy* of `seen_roots` (so writes inside the unit do not leak into sibling tracking) and treated as an opaque writer-unit; `ScopeStmt` and `SeqStmts` share the same `seen_roots`.

### 1.3 Direction rewrite

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

## Phase 2 — Manual-scope lowering

Phase 2 is implemented in the public utility `LowerManualDepsToTaskId(StmtPtr body) -> StmtPtr` (header `include/pypto/ir/transforms/utils/lower_manual_deps_to_task_id.h`, implementation `src/ir/transforms/utils/lower_manual_deps_to_task_id.cpp`), invoked from `DeriveCallDirections` once Phase 1 finishes. It was previously a separate pipeline pass called `DeriveManualScopeDeps`; the lowering logic was extracted to this utility so the pipeline can run Phase 1 and Phase 2 back-to-back on the same function body without a second program-wide traversal.

PyPTO has two scope flavours for orchestrator dependency tracking:

- **Auto scope** (default `PTO2_SCOPE()`): the runtime auto-tracks dependencies from buffer read/write overlap (OverlapMap).
- **Manual scope** (`with pl.manual_scope():` → `RuntimeScopeStmt(manual=true)`): the user takes full ownership of ordering. The runtime skips OverlapMap, and **every required edge must be declared by the user via `kernel(..., deps=[var, ...])`**. The pass does not derive any edge from data flow on its own — the previous auto-dataflow inference path was removed because in practice it produced false positives whenever a buffer was reused in-place across unrelated kernels.

`LowerManualDepsToTaskId` runs four ordered sub-passes per function body.

### 2.1 `ManualDepResolveMutator`

For every kernel `Call` inside a manual scope, copy `Call.attrs["user_manual_dep_edges"]` (the Tensor-Var edges written by the parser when the DSL passed `deps=[var, ...]`) into `Call.attrs["manual_dep_edges"]`. The copy deduplicates edges while preserving the original user order. Auto-derivation of edges from data-flow was intentionally removed (it over-serialised parallel kernels sharing an `Out` parameter); the user controls the dep set explicitly via `deps=[...]`. The per-submit cap of 16 explicit deps (`PTO2_MAX_EXPLICIT_DEPS`) is enforced later at orchestration codegen, which raises a `pypto::ValueError` when the resolved edge list exceeds the cap.

### 2.2 `TaskRelevantVarCollector` (closure analysis)

Starting from the Tensor Vars named in every `kAttrManualDepEdges` set, propagate the "needs-a-TaskId-companion" property through:

- **Var aliases** (`b = a` AssignStmts and `b = tuple[i]` TupleGetItem extracts).
- **`ForStmt.iter_args` ↔ `init_value`** (a TaskId carry must exist for every iter_arg whose init flows from a tagged Var, and vice versa).
- **`ForStmt.return_vars` ↔ `iter_args`** (the rv produced by a TaskId-carrying iter_arg is itself a TaskId carry).
- **`YieldStmt` source ↔ destination** (bidirectional — both directions are needed: `deps=[<iter_arg>]` flows dest→src, while `deps=[<kernel_lhs>]` flows src→dest to the carry destination).

The fixed-point closure builds three sets: `needs_tid_` (every Var needing a companion), `kernel_lhs_` (Vars that are LHS of a user kernel Call — they get the `system.task_id_of` synthesis path), and `import_vars_` (Vars in `needs_tid_` that have no AssignStmt def, typically function parameters used as iter_arg init values).

### 2.3 `PreallocateTaskIdVars`

Allocate one TaskId companion per Var in `needs_tid_`:

- Plain `Var` (non-IterArg, e.g. a kernel LHS or function param) → a fresh `Var` named `<name_hint>__tid` with type `ScalarType(DataType::TASK_ID)`.
- `IterArg` → a fresh `IterArg` with the same name suffix; its init value is wired to the outer Var's companion (looked up in the partial `tid_map_`). For nested loops this lookup needs the outer companion to exist first, so the IterArg allocation pass sweeps to fixed-point: iter_args whose init companion is not yet allocated are re-tried until the chain converges.

The map `tid_map_: const Var* → VarPtr` is the single source of identity for companions; every other stage looks up through it to avoid pointer-identity drift.

### 2.4 `TaskIdLoweringMutator` (IR mutation)

A single IRMutator pass rewrites the function body to install the TaskId infrastructure:

- After each kernel `Call` AssignStmt whose LHS is in `needs_tid_`, inject `<lhs>__tid = system.task_id_of(<lhs>)`.
- After each `tensor.create` AssignStmt whose LHS is in `needs_tid_` (a placeholder buffer with no prior task), inject `<lhs>__tid = system.task_invalid()`.
- After each plain Var-alias AssignStmt (`b = a`), inject `b__tid = a__tid`.
- After each TupleGetItem AssignStmt (`b = tuple[i]`), inject `b__tid = tuple_var__tid` (all unpacked elements share the tuple-producing call's task id).
- On every kernel `Call`, rewrite the Tensor Vars in `kAttrManualDepEdges` to their TaskId companions, and attach `kAttrTaskIdVar` pointing at the LHS's companion (so a later sibling can resolve `deps=[lhs]` through this attr without re-running the closure).
- On every `ForStmt` inside a manual scope, append a TaskId iter_arg and return-var companion for each existing iter_arg in `needs_tid_`. Yield-value lists are extended symmetrically.
- For `import_vars_` (function parameters used as TaskId iter_arg seeds), prepend `<param>__tid = system.task_invalid()` AssignStmts at function-body entry so the companion has an SSA definition the codegen can reference.

The kernel-Call rewrite places **the post-lowering form** in `kAttrManualDepEdges` (TaskId Vars). The codegen consumes this attr; the original Tensor-Var form in `kAttrUserManualDepEdges` is preserved for round-trip printing.

## Examples

### Phase 1 — Arg directions

Two consecutive calls writing the same locally allocated buffer at top level. The first is the only writer-unit so it stays `OutputExisting`; the second hits R-prior and is promoted to `InOut` so the runtime preserves the cross-call WAW dependency on `local`.

#### Before

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

#### After

```python
# Same IR shape; only Call.attrs["arg_directions"] changes:
local = self.kernel(x, local)   # arg_directions = [Input, OutputExisting]
local = self.kernel(x, local)   # arg_directions = [Input, InOut]
```

The `kernel` callee declares `Out` for parameter `out`. Because `local` is locally allocated (rooted at `pl.create_tensor`, not at a `main` parameter), the first call gets `OutputExisting` (no sequential ancestor, no prior writer-unit) while the second sees a prior writer in the same scope and is promoted to `InOut`.

### Phase 2 — Single manual dep edge

```python
@pl.function(type=pl.FunctionType.Orchestration)
def main(self, x: pl.Tensor[[64], pl.FP32],
         out: pl.Out[pl.Tensor[[64], pl.FP32]]) -> pl.Tensor[[64], pl.FP32]:
    with pl.manual_scope():
        scratch = self.stage1(x, scratch)
        out = self.stage2(scratch, out, deps=[scratch])
    return out
```

After the pass:

```python
scratch__ssa_v0__tid: pl.Scalar[pl.TASK_ID] = self.system.task_invalid()  # import var seed
with pl.manual_scope():
    scratch__ssa_v5 = self.stage1(x, scratch)
    scratch__ssa_v5__tid: pl.Scalar[pl.TASK_ID] = self.system.task_id_of(scratch__ssa_v5)
    out__ssa_v7 = self.stage2(scratch__ssa_v5, out, deps=[scratch__ssa_v5__tid])
```

Codegen emits `params_t1.add_dep(scratch__ssa_v5__tid);` from the rewritten dep edge.

### Phase 2 — Multiple deps + loop carry

```python
with pl.manual_scope():
    for phase in pl.range(N_PHASES):
        for branch in pl.parallel(N_BRANCHES):
            row = (phase * N_BRANCHES + branch) * TILE_M
            out = self.kernel_stripe(data, row, 1.0, out, deps=[out])
```

`out` is rebound on every iteration, so both ForStmts have a TaskId iter_arg added (carrying the previous-iteration task id). After the pass:

```python
for phase, (out__iter_v1, out__iter_v1__tid) in pl.range(4, init_values=(out, out__ssa_v0__tid)):
    for branch, (out__iter_v3, out__iter_v3__tid) in pl.parallel(4, init_values=(out__iter_v1, out__iter_v1__tid)):
        out__ssa_v5 = self.kernel_stripe(..., deps=[out__iter_v3__tid])
        out__ssa_v5__tid = self.system.task_id_of(out__ssa_v5)
        out__rv_v4, out__rv_v4__tid = pl.yield_(out__ssa_v5, out__ssa_v5__tid)
    out__rv_v2, out__rv_v2__tid = pl.yield_(out__rv_v4, out__rv_v4__tid)
```

The orchestration codegen treats the `pl.parallel` TaskId iter_arg as **array carry of size `N_BRANCHES`** when the trip count is statically known: it allocates `PTO2TaskId arr[N_BRANCHES]`, per-iteration yields write one slot, and downstream consumers get one `add_dep` per slot. This guarantees a phase-fence on **all** parallel iters, not just the last-dispatched one. The size cap is the same `PTO2_MAX_EXPLICIT_DEPS = 16`; a const trip count beyond that fails at codegen time with a clear error. A non-const trip count under `pl.parallel` carrying a manual dep is rejected at codegen with a "statically-known trip count" message.

### Phase 2 — Var aliases and tuple unpacking

```python
with pl.manual_scope():
    a = self.k1(x)
    c = a                          # plain Var alias
    p, q = self.kpair(x)           # tuple unpack
    d = self.k2(x, deps=[c, p])    # deps reference an alias and an unpacked element
```

The pass synthesises:

```python
a__tid    = self.system.task_id_of(a)
c__tid    = a__tid                  # alias forwards the producer's task id
kpair_tmp = self.kpair(x)           # tuple value
kpair_tmp__tid = self.system.task_id_of(kpair_tmp)
p__tid    = kpair_tmp__tid          # tuple extracts share the producer's task id
q__tid    = kpair_tmp__tid
d = self.k2(x, deps=[c__tid, p__tid])
```

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

**Phase 1 implementation**: `src/ir/transforms/derive_call_directions_pass.cpp`

- `PriorWriterCollector` — per-scope first-writer analysis (bottom-up cache + top-down scan)
- `CallDirectionMutator` — `IRMutator` that rewrites every non-builtin `Call` with the resolved `arg_directions` vector
- Reuses `BufferRootCollector` and `ComputeGroupEffectiveDirections` from `include/pypto/codegen/orchestration/orchestration_analysis.h`

**Phase 2 implementation**: `src/ir/transforms/utils/lower_manual_deps_to_task_id.cpp` (header `include/pypto/ir/transforms/utils/lower_manual_deps_to_task_id.h`)

- Public entry point `LowerManualDepsToTaskId(StmtPtr body) -> StmtPtr` — Phase 1 of `DeriveCallDirections` calls this on each function body once arg directions have been resolved. No-op when no `RuntimeScopeStmt(manual=true)` is reachable from the body.
- `ManualDepResolveMutator`, `TaskRelevantVarCollector`, `PreallocateTaskIdVars`, `TaskIdLoweringMutator` — the four ordered sub-passes described in Phase 2 above.

**Property verifier**: `src/ir/verifier/verify_call_directions.cpp` (factory in `include/pypto/ir/verifier/verifier.h`)

**Python binding**: `python/bindings/modules/passes.cpp`

```cpp
passes.def("derive_call_directions", &pass::DeriveCallDirections,
           "Derive Call attrs['arg_directions'] from callee param directions and buffer lineage, "
           "then lower user-declared deps=[...] edges inside manual scopes to TaskId companions.");
```

**Type stub**: `python/pypto/pypto_core/passes.pyi`

**Hand-written-IR helper**: `python/pypto/ir/directions.py` (`make_call`, lowercase aliases) — for tests and hand-built IR fragments that want to attach explicit directions before the pass runs.

**Tests**: `tests/ut/ir/transforms/test_derive_call_directions.py`

- `TestDeriveDirectionMatrix` — one test per cell of the (callee_dir, origin) → ArgDirection mapping table, including R-seq (`pl.range`, `while`) and R-prior (top-level + branch / parallel-after-top-level) edge cases
- `TestDeriveIdempotent` — running the pass twice yields structurally equal IR
- `TestDerivePreservesExplicit` — pre-populated `arg_directions` is not overwritten
- `TestVerifyPositive` / `TestVerifyNegative` — the `CallDirectionsResolved` property verifier accepts the pass output and rejects ill-formed `arg_directions` assignments

Phase 2 coverage lives alongside the original tests in `tests/ut/ir/transforms/` covering manual-scope lowering (single-dep, multi-dep, loop carry, alias, tuple unpacking).
