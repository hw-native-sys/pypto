# OutlineClusterScopes Pass

Outlines Cluster scopes into separate Group functions.

## Overview

This pass transforms `ScopeStmt(Cluster)` nodes into separate `Function(Group)` definitions and replaces the scope with a Call to the outlined function. Group functions represent co-scheduled AIC (Cube) + AIV (Vector) kernel groups that share the same physical cluster resources.

**Requirements**:

- Input IR must be in SSA form (run ConvertToSSA first)
- Only processes Opaque and Orchestration functions

**When to use**: Run after `OutlineIncoreScopes` when the IR contains `with pl.cluster():` scopes that need to be extracted into Group-typed functions.

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

1. **Scan for Cluster Scopes**: Find all `ScopeStmt(scope_kind=Cluster)` in Opaque/Orchestration functions
2. **Analyze Inputs**: Variables referenced inside scope but defined outside
3. **Analyze Outputs**: Variables defined inside scope and used after it
4. **Create Function**: Extract scope body into `Function(func_type=Group)` with input params and output returns
5. **Replace Scope**: Replace `ScopeStmt` with Call to outlined function + output assignments
6. **Add to Program**: Prepend outlined functions to the program's function list

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

## Implementation

**Header**: `include/pypto/ir/transforms/passes.h`

**Implementation**: `src/ir/transforms/outline_cluster_scopes_pass.cpp`

**Python binding**: `python/bindings/modules/passes.cpp`

**Tests**: `tests/ut/ir/transforms/test_outline_cluster_scopes.py`

## Pass Properties

| Property | Value |
| -------- | ----- |
| Required | TypeChecked, SSAForm |
| Produced | ClusterOutlined |
| Invalidated | — |

## Relationship to OutlineIncoreScopes

| Aspect | OutlineIncoreScopes | OutlineClusterScopes |
| ------ | ------------------- | -------------------- |
| Scope kind | `ScopeKind::InCore` | `ScopeKind::Cluster` |
| Output function type | `FunctionType::InCore` | `FunctionType::Group` |
| Naming pattern | `{func}_incore_{n}` | `{func}_cluster_{n}` |
| Promotes parent to | Orchestration | *(unchanged)* |
| Processes | Opaque functions only | Opaque + Orchestration |
