# DeriveManualScopeDeps Pass

Resolves task-dependency edges for kernel `Call`s inside `with pl.manual_scope():` regions and writes them to `Call.attrs["manual_dep_edges"]` for codegen.

## Overview

PyPTO has two scope flavours for orchestrator dependency tracking:

- **Auto scope** (the default `PTO2_SCOPE()` region): the runtime auto-derives task dependencies from buffer read/write overlap (OverlapMap).
- **Manual scope** (`with pl.manual_scope():` → `RuntimeScopeStmt(manual=true)`): the user takes ownership of ordering. The runtime skips OverlapMap inside this region, and codegen must emit explicit `params.add_dep(task_<m>);` calls instead.

`DeriveManualScopeDeps` is the pass that bridges DSL intent and runtime semantics. For every kernel `Call` inside a manual scope, it computes the union of:

1. **User-supplied edges** — any vars passed via the DSL `kernel(..., deps=[var, ...])` kwarg, surfaced by the parser as `Call.attrs["user_manual_dep_edges"]`.
2. **Data-flow edges** — every tensor argument whose `ArgDirection` is not `NoDep` and whose `Var` resolves to a producer (the LHS of a prior kernel `AssignStmt`) in the same manual scope.

The resolved set is written to `Call.attrs["manual_dep_edges"]` (`std::vector<VarPtr>`) and read by the orchestration codegen to emit `params_t<n>.add_dep(task_<m>);` per edge. The list is capped at 16 to mirror the runtime's `PTO2_MAX_EXPLICIT_DEPS`; exceeding the cap raises an internal error pinned to the offending call site.

**When to use**: Run after `DeriveCallDirections` (the data-flow analysis depends on resolved per-arg directions to honor `NoDep` slots) and before the final `Simplify`. In the `Default` strategy it's the 31st pipeline pass, between `DeriveCallDirections` (slot 30) and the trailing `Simplify` (utility slot 91).

## Properties

| Required | Produced | Invalidated |
| -------- | -------- | ----------- |
| `CallDirectionsResolved` | — | — |

The pass does not (yet) declare a verifiable post-condition; it is structurally idempotent — running it twice writes the same `manual_dep_edges` set.

## API

| C++ | Python | Level |
| --- | ------ | ----- |
| `pass::DeriveManualScopeDeps()` | `passes.derive_manual_scope_deps()` | Program-level |

**Factory function**:

```cpp
Pass DeriveManualScopeDeps();
```

**Python usage**:

```python
from pypto.pypto_core import passes

pass_obj = passes.derive_manual_scope_deps()
program_with_edges = pass_obj(program)
```

## Algorithm

The pass is a `ProgramPass` that runs a `ManualDepMutator` on each function body.

### 1. Manual-scope tracking

The mutator carries an integer `manual_depth_` counter incremented on entry to a `RuntimeScopeStmt(manual=true)` and decremented on exit. Outside a manual scope (`manual_depth_ == 0`) the mutator is a no-op for `AssignStmt` / `EvalStmt` visits.

When a manual scope is entered, the producer-Var map (see step 2) is **saved and cleared**, then restored on exit. This isolates each scope: a producer in scope A cannot be referenced as a manual dep edge from scope B.

### 2. Producer-Var map

Inside a manual scope, every `AssignStmt` whose RHS is a non-builtin kernel `Call` registers its LHS `Var*` in `producer_map_`. Builtin ops (`tensor.*`, `tile.*`, `system.*`) never become producers because they don't submit runtime tasks.

### 3. Edge resolution per call

For each kernel `Call` (whether RHS of `AssignStmt` or expression of `EvalStmt`) inside a manual scope, the helper `ResolveManualDepsForCall` collects edges in two phases, deduplicating by `Var*`:

1. **User-supplied** — scan `Call.attrs["user_manual_dep_edges"]` (parser-set when the DSL passes `deps=[var]`), preserving order.
2. **Data-flow** — for each positional arg, if its `ArgDirection` is not `NoDep` and `AsVarLike(arg)` resolves to a `Var*` in `producer_map_`, append that producer.

The result is written back as `Call.attrs["manual_dep_edges"]` (a fresh `std::vector<VarPtr>`). The cap check uses `INTERNAL_CHECK_SPAN(deps.size() <= 16, call->span_)` so the diagnostic carries the source location.

### 4. Var-typed attrs survive remap

Because the edge attrs hold `VarPtr` references that must stay in sync with their definition sites, the base `IRMutator::VisitExpr_(Call)` and `IRVisitor::VisitExpr_(Call)` were extended (in the same change) to walk Vars in `kAttrManualDepEdges` / `kAttrUserManualDepEdges` through the standard `var_remap_` / use-site visit paths. `ConvertToSSA` does the same with its own `cur_` map.

## Examples

### Auto-derived edges from data flow

```python
@pl.function(type=pl.FunctionType.Orchestration)
def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    with pl.manual_scope():
        a = self.k1(x)
        b = self.k2(a)        # k2 reads `a` -> auto edge: task_0
    return b
```

After the pass, the call to `k2` carries `attrs["manual_dep_edges"] = [a]`. Codegen emits:

```cpp
PTO2_SCOPE(PTO2ScopeMode::MANUAL) {
    Arg params_t0;
    params_t0.add_input(ext_x);
    TaskOutputTensors task_0_outs = rt_submit_aiv_task(0, params_t0);
    PTO2TaskId task_0 = task_0_outs.task_id();

    Arg params_t1;
    params_t1.add_input(/*alias of a*/);
    params_t1.add_dep(task_0);          // <-- from manual_dep_edges
    TaskOutputTensors task_1_outs = rt_submit_aiv_task(1, params_t1);
    PTO2TaskId task_1 = task_1_outs.task_id();
}
```

### Explicit user deps

`deps=[var, ...]` adds edges that aren't visible from data flow alone:

```python
with pl.manual_scope():
    a = self.k1(x)
    b = self.k2(x)
    c = self.k3(x, deps=[a, b])    # c reads only x, but ordering needs a,b
```

After the pass, `c`'s call carries `manual_dep_edges = [a, b]`; codegen emits both `params_t2.add_dep(task_0)` and `params_t2.add_dep(task_1)`.

### Suppressing an auto edge with `pl.no_dep`

A `pl.no_dep(arg)` marker (Phase 2) sets the per-arg direction to `NoDep` at parse time; this pass excludes such args from the data-flow scan, so the producer behind that arg does not become a manual dep edge.

## See also

- [DeriveCallDirections (slot 30)](30-derive_call_directions.md) — required predecessor; resolves `NoDep` from `pl.no_dep(...)` markers.
- [IR hierarchy: ScopeStmt](../ir/01-hierarchy.md#scopestmt-details) — `RuntimeScopeStmt` and the auto/manual flag.
- [Python syntax: scope context managers](../language/00-python_syntax.md#scope-context-managers) — `with pl.manual_scope():` surface form.
