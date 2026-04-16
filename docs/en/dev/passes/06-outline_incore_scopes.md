# OutlineIncoreScopes Pass

Outlines `HierarchyScopeStmt` regions with `level_ == CORE_GROUP` into
dedicated `Function(InCore)` definitions and promotes the enclosing parent
function from `Opaque` to `Orchestration`.

## Overview

This pass specifically targets the `CORE_GROUP` form of
`HierarchyScopeStmt` — the per-core-group kernel region introduced by
`with pl.at(level=pl.Level.CORE_GROUP):`. Each such scope is extracted
into a new `Function` whose `func_type_` is `FunctionType::InCore`, and the
original scope is replaced with a `Call` to that outlined function. Whenever
any `CORE_GROUP` scope is outlined out of a given parent function, that
parent's `func_type_` is promoted from `Opaque` to `Orchestration`.

This pass is the CORE_GROUP counterpart of
[`OutlineHierarchyScopes`](05-outline_hierarchy_scopes.md), which handles
the remaining (non-CORE_GROUP) hierarchy levels by emitting
`Function(Opaque)` and leaving the parent type alone.

| Scope `level_` | Outlined function type | Parent function type after pass |
| -------------- | ---------------------- | ------------------------------- |
| `Level.CORE_GROUP` | `FunctionType::InCore` | promoted `Opaque` → `Orchestration` |
| any other level | *(not handled — already outlined by `OutlineHierarchyScopes`)* | — |

When a `CORE_GROUP` scope carries a `split_` optimization hint, the hint is
attached to the outlined `InCore` function as a `split` attribute so that
downstream passes — notably
[`ExpandMixedKernel`](11-expand_mixed_kernel.md) — can honour it when
deciding how to split the kernel into AIC / AIV halves.

**Requirements**:

- Input IR must be in SSA form (run `ConvertToSSA` first). SSA form is
  preserved (produced) by this pass.
- Expects `OutlineHierarchyScopes` to have already run, so only
  `CORE_GROUP` `HierarchyScopeStmt` nodes remain to be outlined.
- Only processes `Opaque` functions (which may contain residual
  `CORE_GROUP` scopes). Functions already typed as `Orchestration`,
  `InCore`, `AIC`, `AIV`, or `Group` are left untouched.

**When to use**: Run immediately after
[`OutlineHierarchyScopes`](05-outline_hierarchy_scopes.md) and before
[`OutlineClusterScopes`](07-outline_cluster_scopes.md). By the time this
pass finishes, the `HierarchyOutlined` property holds: no
`HierarchyScopeStmt` nodes remain in `Opaque` or `Orchestration` functions.

## API

| C++ | Python | Level |
| --- | ------ | ----- |
| `pass::OutlineIncoreScopes()` | `passes.outline_incore_scopes()` | Program-level |

**Factory function**:

```cpp
Pass OutlineIncoreScopes();
```

**Python usage**:

```python
from pypto.pypto_core import passes

outline_pass = passes.outline_incore_scopes()
program_outlined = outline_pass(program)
```

## Algorithm

1. **Scan for CORE_GROUP Scopes**: Find every `HierarchyScopeStmt` in each
   `Opaque` function body whose `level_ == CORE_GROUP`.
2. **Analyze Inputs/Outputs**: Use the shared scope-outline helpers to
   compute the set of variables defined outside but used inside (inputs)
   and defined inside but used outside (outputs).
3. **Create Outlined InCore Function**: Extract the scope body into a new
   `Function`:
   - Parameters = input variables
   - Returns = output variables
   - Body = the scope body
   - `func_type_` = `InCore`
   - Copy `role_` into function attrs.
   - If the scope carries a `split_` optimization hint, copy it into the
     function's `split` attr (consumed by `ExpandMixedKernel`).
4. **Replace the Scope**: Substitute the original `HierarchyScopeStmt`
   with a `Call` to the outlined InCore function followed by `AssignStmt`s
   that bind its return values.
5. **Promote Parent**: If any `CORE_GROUP` scope was outlined from the
   parent function, re-type that parent from `Opaque` to `Orchestration`.
6. **Add to Program**: Prepend the outlined InCore functions to the
   program's function list.

**Naming**: `{original_func}_core_group_{counter}` (e.g.
`main_core_group_0`). Outlined InCore functions use a `_incore_`-style
name suffix in their attrs and are easily identifiable in printed IR. When
`HierarchyScopeStmt.name_hint` is non-empty the hint is used directly.

## Example

### CORE_GROUP → InCore + Orchestration

**Before** (after `OutlineHierarchyScopes`, non-CORE_GROUP scopes are
already outlined; the CORE_GROUP scope still sits inline in `main`):

```python
@pl.program
class Before:
    @pl.function  # Opaque
    def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        y = x + 1

        with pl.at(level=pl.Level.CORE_GROUP):
            tile = pl.load(y, [0], [64])
            tile_sq = pl.mul(tile, tile)
            result_tile = tile_sq + 1
            result = pl.store(result_tile, [0], x)

        z = result + 2
        return z
```

**After**:

```python
@pl.program
class After:
    @pl.function(type=pl.FunctionType.Orchestration)  # promoted
    def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        y = x + 1
        result = self.main_core_group_0(y, x)  # Call to outlined InCore fn
        z = result + 2
        return z

    @pl.function(type=pl.FunctionType.InCore)  # outlined
    def main_core_group_0(self, y: pl.Tensor[[64], pl.FP32],
                          x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        tile = pl.load(y, [0], [64])
        tile_sq = pl.mul(tile, tile)
        result_tile = tile_sq + 1
        result = pl.store(result_tile, [0], x)
        return result
```

### CORE_GROUP with split hint

```python
with pl.at(level=pl.Level.CORE_GROUP,
           optimizations=[pl.split(pl.SplitMode.UP_DOWN)]):
    ...
```

The outlined `InCore` function receives the `split` hint in its attrs,
which `ExpandMixedKernel` later reads to split the kernel into AIC + AIV
halves.

### Multiple outputs

```python
with pl.at(level=pl.Level.CORE_GROUP):
    a_tile = pl.load(a, [0], [64])
    b_tile = pl.load(b, [0], [64])
    c_tile = pl.add(a_tile, b_tile)
    out_a = pl.store(c_tile, [0], out)
    out_b = pl.mul(c_tile, 2.0)
# both out_a and out_b used after the scope
x = out_a + out_b
```

After outlining, the parent body becomes:

```python
out_a, out_b = self.main_core_group_0(a, b, out)  # multiple return values
x = out_a + out_b
```

## Implementation

**Header**: `include/pypto/ir/transforms/passes.h`

```cpp
Pass OutlineIncoreScopes();
```

**Implementation**: `src/ir/transforms/outline_incore_scopes.cpp`

- Uses the shared `scope_outline_utils` to compute inputs/outputs
- Builds a new `Function(InCore)` per `CORE_GROUP` scope
- Copies `role_` / `split_` metadata onto the outlined function's attrs
- Re-types the parent function from `Opaque` to `Orchestration` when at
  least one `CORE_GROUP` scope was outlined out of it

**Python binding**: `python/bindings/modules/passes.cpp`

```cpp
passes.def("outline_incore_scopes", &pass::OutlineIncoreScopes,
           "Outline CORE_GROUP HierarchyScopeStmt regions into Function(InCore) "
           "and promote the parent function to Orchestration");
```

**Tests**: `tests/ut/ir/transforms/test_outline_incore_scopes.py`

- Tests `CORE_GROUP` scope → `InCore` function + parent `Orchestration`
- Tests `split_` propagation onto the outlined InCore function
- Tests input/output analysis
- Tests multiple `CORE_GROUP` scopes in the same parent function
- Tests SSA preservation

## Pass Properties

| Property | Value |
| -------- | ----- |
| Required | `SSAForm` |
| Produced | `SSAForm`, `HierarchyOutlined` |
| Invalidated | — |

`HierarchyOutlined` is produced here (not by
[`OutlineHierarchyScopes`](05-outline_hierarchy_scopes.md)): after both
outline passes have run, no `HierarchyScopeStmt` nodes remain in
`Opaque`/`Orchestration` functions.

## Pipeline Position

```text
... → ConvertToSSA → NormalizeStmtStructure → FlattenCallExpr →
OutlineHierarchyScopes → OutlineIncoreScopes → OutlineClusterScopes →
ConvertTensorToTileOps → ...
```
