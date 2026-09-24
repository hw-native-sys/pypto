# OutlineClusterScopes Pass

Outlines Cluster scopes into Group functions and standalone Spmd scopes into Spmd functions.

## Overview

This pass transforms `ClusterScopeStmt` nodes into separate `Function(Group)` definitions and replaces the scope with a Call to the outlined function. It also transforms standalone `SpmdScopeStmt` nodes, those not nested inside a Cluster, into `Function(Spmd)` definitions. Group functions represent co-scheduled AIC (Cube) + AIV (Vector) kernel groups that share the same physical cluster resources, while the standalone launch semantics (`core_num` / `sync_start`) move onto the synthesised dispatch — see [Where the launch spec lives](#where-the-launch-spec-lives).

**Requirements**:

- Input IR must be in SSA form (run ConvertToSSA first)
- Only processes Opaque and Orchestration functions

**When to use**: Run after `OutlineIncoreScopes` when the IR contains `with pl.cluster():` scopes or standalone `with pl.spmd(...):` / `for i in pl.spmd(...)` scopes that need to be extracted into wrapper functions. The loop form is a parser-level desugaring for `SpmdScopeStmt(body=InCoreScopeStmt(...))`; `OutlineIncoreScopes` outlines the InCore body first, leaving a single-call Spmd body for this pass to lift into a `Function(Spmd)`.

## API

| C++ | Python | Level |
| --- | ------ | ----- |
| `pass::OutlineClusterScopes()` | `passes.outline_cluster_scopes()` | Program-level |

**Python usage**:

```python
from pypto.pypto_core import passes

outline_pass = passes.outline_cluster_scopes()
program_outlined = outline_pass(program)
```

## Algorithm

1. **Scan for Cluster Scopes**: Find all `ClusterScopeStmt` nodes in Opaque/Orchestration functions
2. **Outline Cluster Scopes**: Extract each Cluster body into `Function(func_type=Group)`
3. **Scan for Standalone Spmd Scopes**: On the transformed body, find `SpmdScopeStmt` nodes that are not nested inside a Cluster
4. **Outline Standalone Spmd Scopes**: Extract each standalone Spmd body into `Function(func_type=Spmd)` and attach `core_num` / `sync_start` to the synthesised **dispatch** (Call attrs, or the `Submit` fields for an `as tid` scope) — never to the outlined function
5. **Unwrap Nested Spmd in Group**: For `pl.cluster(): with pl.spmd(...): ...`, keep a single Group function, move `core_num` / `sync_start` onto its **dispatch** (translated back through the dispatch's args, since the spec is lifted from inside the callee), and stamp the self-contained `spmd_unwrapped` marker on the Group
6. **Replace Scope**: Replace each outlined scope with a Call to the outlined function + output assignments
7. **Add to Program**: Prepend outlined functions to the program's function list

**Naming**: `{original_func}_cluster_{counter}` (e.g., `main_cluster_0`)

**Param-explicit returns**: like `OutlineIncoreScopes`, the
outlined Group/Spmd functions return their own parameters whenever a tensor
output writes through a parameter — store targets return the param directly,
other outputs are traced via the shared `return_lineage` utility; only
kernel-allocated outputs keep their SSA value. This keeps the
`ReturnParamsExplicit` invariant so orchestration codegen maps returns to
args by pointer identity.

## Example

**Before**:

```python
@pl.program
class Before:
    @pl.function
    def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        with pl.cluster():
            with pl.at(level=pl.Level.CORE_GROUP):
                y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
        return y
```

**After**:

```python
@pl.program
class After:
    @pl.function(type=pl.FunctionType.Group)
    def main_cluster_0(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP):
            y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
        return y

    @pl.function
    def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        y: pl.Tensor[[64], pl.FP32] = self.main_cluster_0(x)
        return y
```

Note: InCore scopes inside the Cluster are preserved in the outlined Group function. Run `OutlineIncoreScopes` first to outline InCore scopes before clustering, or after to outline them within Group functions.

## Standalone Spmd Example

**Before**:

```python
@pl.program
class Before:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, x: pl.Tensor[[64], pl.FP32],
               out: pl.Out[pl.Tensor[[64], pl.FP32]]) -> pl.Tensor[[64], pl.FP32]:
        tile = pl.load(x, [0], [64])
        out = pl.store(pl.add(tile, tile), [0], out)
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, x: pl.Tensor[[64], pl.FP32],
             out: pl.Out[pl.Tensor[[64], pl.FP32]]) -> pl.Tensor[[64], pl.FP32]:
        with pl.spmd(4, sync_start=True):
            out = self.kernel(x, out)
        return out
```

**After**:

```python
@pl.program
class After:
    # kernel definition unchanged — omitted for brevity
    @pl.function(type=pl.FunctionType.Spmd)
    def main_spmd_0(self, x: pl.Tensor[[64], pl.FP32],
                    out: pl.Out[pl.Tensor[[64], pl.FP32]]) -> pl.Tensor[[64], pl.FP32]:
        out = self.kernel(x, out)
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, x: pl.Tensor[[64], pl.FP32],
             out: pl.Out[pl.Tensor[[64], pl.FP32]]) -> pl.Tensor[[64], pl.FP32]:
        out = self.main_spmd_0(x, out, attrs={"core_num": 4, "sync_start": True})
        return out
```

### Where the launch spec lives

`core_num` / `sync_start` ride the **dispatch**, not the outlined function:

| Dispatch shape | Carrier |
| -------------- | ------- |
| plain `Call` — a standalone `with pl.spmd(...)` / `for i in pl.spmd(...)` carrying no Submit-only metadata | `attrs["core_num"]` (`ExprPtr`) and `attrs["sync_start"]` (`bool`, only when true) |
| `Submit` — a standalone `with pl.spmd(...)` / `for i in pl.spmd(...)` with `as tid`, or any of `deps=` / `allow_early_resolve=` / `predicate=` (for which the outliner synthesizes the TaskId Var) | the first-class `Submit::core_num_` / `sync_start_` fields |
| `Group` from `pl.cluster(): with pl.spmd(...)` | the dispatch, same as above; the Group keeps only the `spmd_unwrapped` marker |

`core_num` is an expression evaluated in the *dispatching* function's scope — it
may reference caller-local scalars (e.g. `pl.spmd(m // 16)` where
`m = pl.tensor.dim(a, 0)`). A `Function` is a closed scope: every Var it
references must resolve to one of its params or a body-local definition. Storing
the expression on the callee therefore produced a Function referencing names it
does not bind — it printed as a decorator evaluated before any body binds those
names (so the program could not be re-parsed) and was invisible to the visitor /
mutator, which walk one function at a time. Keeping the spec at the launch site
removes all three problems and needs no special handling: the existing Call-attr
printer/parser codec round-trips a general `ExprPtr`, and the Vars it references
are ordinary local uses for def-use and DCE.

The Group case needs one extra step: its spec is lifted from *inside* the callee,
where the count references a Group **param** (the cluster outliner captured the
caller's scalar as one). `LaunchSpecStamper` maps `params_[i] -> args_[i]` through
the dispatch so the attr names a Var that is actually live at the call site.

What stays on the Group is `spmd_unwrapped` (`bool`) — legitimately function-scoped:
it states a property of that function's own body and references nothing outside it.
It tells a launch-site consumer that dispatching this Group launches the kernels its
body calls, rather than the Group itself as a mixed kernel — the distinction the
occupancy verifier previously drew from the presence of a `core_num` attr.

Orchestration codegen reads the spec via `EffectiveLaunchSpec`, which prefers the
dispatch's own attrs; the launch-function fallback remains for hand-written or
deserialized IR that still spells a constant `core_num` on the function.

## Implementation

**Header**: `include/pypto/ir/transforms/passes.h`

**Implementation**: `src/ir/transforms/outline_cluster_scopes_pass.cpp`

**Python binding**: `python/bindings/modules/passes.cpp`

**Tests**: `tests/ut/ir/transforms/test_outline_cluster_scopes.py`

## Pass Properties

| Property | Value |
| -------- | ----- |
| Required | SSAForm |
| Produced | SSAForm, ClusterOutlined |
| Invalidated | — |

## Relationship to OutlineIncoreScopes

| Aspect | OutlineIncoreScopes | OutlineClusterScopes |
| ------ | ------------------- | -------------------- |
| Scope kind | `ScopeKind::InCore` | `ScopeKind::Cluster` / standalone `ScopeKind::Spmd` |
| Output function type | `FunctionType::InCore` | `FunctionType::Group` / `FunctionType::Spmd` |
| Naming pattern | `{func}_incore_{n}` | `{func}_cluster_{n}` / `{func}_spmd_{n}` |
| Promotes parent to | Orchestration | *(unchanged)* |
| Processes | Opaque functions only | Opaque + Orchestration |
