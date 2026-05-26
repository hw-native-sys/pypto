# DeriveManualScopeDeps Pass

Lowers user-declared `deps=[...]` edges in `with pl.manual_scope():` regions into a TaskId-based dependency graph that the orchestration codegen can emit as `params.add_dep(<task_id>)` calls.

## Overview

PyPTO has two scope flavours for orchestrator dependency tracking:

- **Auto scope** (default `PTO2_SCOPE()`): the runtime auto-tracks dependencies from buffer read/write overlap (OverlapMap).
- **Manual scope** (`with pl.manual_scope():` → `RuntimeScopeStmt(manual=true)`): the user takes full ownership of ordering. The runtime skips OverlapMap, and **every required edge must be declared by the user via `kernel(..., deps=[var, ...])`**. The pass does not derive any edge from data flow on its own — the previous auto-dataflow inference path was removed because in practice it produced false positives whenever a buffer was reused in-place across unrelated kernels.

The pass name kept by Python (`passes.derive_manual_scope_deps()`) and by the pipeline manager is `DeriveManualScopeDeps`; the C++ implementation is the `LowerManualDepsToTaskId` lowering. The two names refer to the same pass slot — the pass name is preserved for binding-API compatibility.

**When to use**: Run after `DeriveCallDirections` (which resolves per-arg `ArgDirection` and produces `CallDirectionsResolved`) and before the trailing `Simplify`. In the `Default` strategy it is the 33rd pipeline pass.

## Properties

| Required | Produced | Invalidated |
| -------- | -------- | ----------- |
| `CallDirectionsResolved` | — | — |

No verifiable post-condition is declared. The pass is structurally idempotent on already-lowered IR (the second run finds the same `kAttrUserManualDepEdges` entries, resolves the same closure, and re-allocates the same TaskId companions by pointer identity from the previously-built `tid_map_`).

## API

| C++ | Python | Level |
| --- | ------ | ----- |
| `pass::DeriveManualScopeDeps()` | `passes.derive_manual_scope_deps()` | Program-level |

```cpp
Pass DeriveManualScopeDeps();
```

```python
from pypto.pypto_core import passes
pass_obj = passes.derive_manual_scope_deps()
program_after = pass_obj(program)
```

## Algorithm

`LowerOneFunction` runs four ordered sub-passes per function body whose IR contains a manual scope.

### Stage 1 — `ManualDepResolveMutator`

For every kernel `Call` inside a manual scope, copy `Call.attrs["user_manual_dep_edges"]` (the Tensor-Var edges written by the parser when the DSL passed `deps=[var, ...]`) into `Call.attrs["manual_dep_edges"]`. No auto-derivation runs. No per-call cap is enforced — codegen sizes the emitted `ArgWithDeps<N>` to the exact dep count, and the runtime's `Arg::set_dependencies(ptr, count)` primitive has no upper bound.

### Stage 2 — `TaskRelevantVarCollector` (closure analysis)

Starting from the Tensor Vars named in every `kAttrManualDepEdges` set, propagate the "needs-a-TaskId-companion" property through:

- **Var aliases** (`b = a` AssignStmts and `b = tuple[i]` TupleGetItem extracts).
- **`ForStmt.iter_args` ↔ `init_value`** (a TaskId carry must exist for every iter_arg whose init flows from a tagged Var, and vice versa).
- **`ForStmt.return_vars` ↔ `iter_args`** (the rv produced by a TaskId-carrying iter_arg is itself a TaskId carry).
- **`YieldStmt` source ↔ destination** (bidirectional — both directions are needed: `deps=[<iter_arg>]` flows dest→src, while `deps=[<kernel_lhs>]` flows src→dest to the carry destination).

The fixed-point closure builds three sets: `needs_tid_` (every Var needing a companion), `kernel_lhs_` (Vars that are LHS of a user kernel Call — they get the `system.task_id_of` synthesis path), and `import_vars_` (Vars in `needs_tid_` that have no AssignStmt def, typically function parameters used as iter_arg init values).

### Stage 3 — `PreallocateTaskIdVars`

Allocate one TaskId companion per Var in `needs_tid_`:

- Plain `Var` (non-IterArg, e.g. a kernel LHS or function param) → a fresh `Var` named `<name_hint>__tid` with type `ScalarType(DataType::TASK_ID)`.
- `IterArg` → a fresh `IterArg` with the same name suffix; its init value is wired to the outer Var's companion (looked up in the partial `tid_map_`). For nested loops this lookup needs the outer companion to exist first, so the IterArg allocation pass sweeps to fixed-point: iter_args whose init companion is not yet allocated are re-tried until the chain converges.

The map `tid_map_: const Var* → VarPtr` is the single source of identity for companions; every other stage looks up through it to avoid pointer-identity drift.

### Stage 4 — `TaskIdLoweringMutator` (IR mutation)

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

### Single dep edge

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

### Multiple deps + loop carry (case1 shape)

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

The orchestration codegen treats the `pl.parallel` TaskId iter_arg as **array carry of size `N_BRANCHES`** when the trip count is statically known: it allocates `PTO2TaskId arr[N_BRANCHES]`, per-iteration yields write one slot, and downstream consumers get one `add_dep` per slot. This guarantees a phase-fence on **all** parallel iters, not just the last-dispatched one. The size cap is the emitted `ArgWithDeps<>` capacity (default 16); a const trip count beyond that fails at codegen time with a clear error. A non-const trip count under `pl.parallel` carrying a manual dep is rejected at codegen with a "statically-known trip count" message.

### Var aliases and tuple unpacking

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

## See also

- [DeriveCallDirections (slot 32)](32-derive_call_directions.md) — required predecessor; resolves per-arg `ArgDirection`.
- [System ops: `task_invalid` / `task_id_of`](../ir/05-operators.md#syncop-synchronization-operations) — the two builtins synthesised by this pass.
- [DataType: `TASK_ID`](../ir/02-types.md#scalartype) — opaque 64-bit handle used for companions.
- [Orchestration codegen: manual scope + array carry](../codegen/01-orchestration_codegen.md) — how the post-lowering IR is emitted.
- [Python syntax: manual scope and deps](../language/00-python_syntax.md#manual-dependency-primitives) — surface-form semantics.
