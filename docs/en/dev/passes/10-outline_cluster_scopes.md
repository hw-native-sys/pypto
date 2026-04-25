# OutlineClusterScopes Pass

Outlines Cluster scopes into Group functions and standalone Spmd scopes into Spmd functions.

## Overview

This pass transforms `ClusterScopeStmt` nodes into separate `Function(Group)` definitions and replaces the scope with a Call to the outlined function. It also transforms standalone `SpmdScopeStmt` nodes, those not nested inside a Cluster, into `Function(Spmd)` definitions. Group functions represent co-scheduled AIC (Cube) + AIV (Vector) kernel groups that share the same physical cluster resources, while Spmd functions preserve standalone launch semantics such as `core_num` and `sync_start`.

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
4. **Outline Standalone Spmd Scopes**: Extract each standalone Spmd body into `Function(func_type=Spmd)` and copy `core_num` / `sync_start` into function attrs
5. **Unwrap Nested Spmd in Group**: For `pl.cluster(): with pl.spmd(...): ...`, keep a single Group function and move `core_num` / `sync_start` onto the Group attrs
6. **Replace Scope**: Replace each outlined scope with a Call to the outlined function + output assignments
7. **Add to Program**: Prepend outlined functions to the program's function list

**Naming**: `{original_func}_cluster_{counter}` (e.g., `main_cluster_0`)

## Example

**Before**:

```python
@pl.program
class Before:
    @pl.function
    def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        with pl.cluster():
            with pl.incore():
                y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
        return y
```

**After**:

```python
@pl.program
class After:
    @pl.function(type=pl.FunctionType.Group)
    def main_cluster_0(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        with pl.incore():
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
    @pl.function(type=pl.FunctionType.Spmd, attrs={"core_num": 4, "sync_start": True})
    def main_spmd_0(self, x: pl.Tensor[[64], pl.FP32],
                    out: pl.Out[pl.Tensor[[64], pl.FP32]]) -> pl.Tensor[[64], pl.FP32]:
        out = self.kernel(x, out)
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, x: pl.Tensor[[64], pl.FP32],
             out: pl.Out[pl.Tensor[[64], pl.FP32]]) -> pl.Tensor[[64], pl.FP32]:
        out = self.main_spmd_0(x, out)
        return out
```

## Implementation

**Header**: `include/pypto/ir/transforms/passes.h`

**Implementation**: `src/ir/transforms/outline_cluster_scopes_pass.cpp`

**Python binding**: `python/bindings/modules/passes.cpp`

**Tests**: `tests/ut/ir/transforms/test_outline_cluster_scopes.py`

## Pass Properties

| Property | Value |
| -------- | ----- |
| Required | TypeChecked, SSAForm |
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
