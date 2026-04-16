# OutlineHierarchyScopes Pass

Outlines non-`CORE_GROUP` `HierarchyScopeStmt` regions into separate
`Opaque` functions, carrying the scope's level/role metadata onto the
outlined function.

## Overview

This pass transforms each `HierarchyScopeStmt` whose `level_` is not
`Level.CORE_GROUP` into a dedicated `Function` definition and replaces the
scope with a `Call` to that function. The outlined function is always typed
`FunctionType::Opaque`; the parent function's type is preserved.

| Scope `level_` | Handled by this pass | Outlined function type | Parent function type after pass |
| -------------- | -------------------- | ---------------------- | ------------------------------- |
| `Level.HOST`, `Level.CLUSTER`, `Level.GLOBAL`, ... | Yes | `FunctionType::Opaque` | unchanged (preserved) |
| `Level.CORE_GROUP` | **No — intentionally left alone** | *(handled by [`OutlineIncoreScopes`](06-outline_incore_scopes.md))* | *(promoted to `Orchestration` by the next pass)* |

`CORE_GROUP` scopes are intentionally left untouched here; the immediately
following pass, [`OutlineIncoreScopes`](06-outline_incore_scopes.md),
outlines them into `Function(InCore)` and promotes the parent function from
`Opaque` to `Orchestration`.

**Requirements**:

- Input IR must be in SSA form (run `ConvertToSSA` first). SSA form is
  preserved (produced) by this pass.
- Processes `Opaque` functions. Functions already typed as
  `Orchestration`, `InCore`, `AIC`, `AIV`, or `Group` are left untouched.

**When to use**: Run after `ConvertToSSA`/`FlattenCallExpr` when the IR
contains `with pl.at(level=...):` scopes for non-`CORE_GROUP` levels that
need to be extracted into callable helper functions.

## API

| C++ | Python | Level |
| --- | ------ | ----- |
| `pass::OutlineHierarchyScopes()` | `passes.outline_hierarchy_scopes()` | Program-level |

**Factory function**:

```cpp
Pass OutlineHierarchyScopes();
```

**Python usage**:

```python
from pypto.pypto_core import passes

outline_pass = passes.outline_hierarchy_scopes()
program_outlined = outline_pass(program)
```

## Algorithm

1. **Scan for Hierarchy Scopes**: Find every `HierarchyScopeStmt` inside each
   `Opaque` function body whose `level_` is **not** `CORE_GROUP`.
2. **Analyze Inputs/Outputs**: Use the shared scope-outline helpers to compute
   the set of variables defined outside but used inside (inputs) and defined
   inside but used outside (outputs).
3. **Create Outlined Function**: Extract the scope body into a new `Function`:
   - Parameters = input variables
   - Returns = output variables
   - Body = the scope body
   - `func_type_` = `Opaque`
   - Copy `role_` metadata into function attrs.
4. **Replace the Scope**: Substitute the original `HierarchyScopeStmt` with
   a `Call` to the outlined function followed by `AssignStmt`s that bind its
   return values.
5. **Preserve Parent Type**: The parent function's `func_type_` is not
   changed by this pass. Parent type promotion for `CORE_GROUP` scopes is
   the responsibility of [`OutlineIncoreScopes`](06-outline_incore_scopes.md).
6. **Add to Program**: Prepend the outlined functions to the program's
   function list.

**Naming**: `{original_func}_{level}_{counter}` (e.g. `main_host_0`,
`main_global_0`). When `HierarchyScopeStmt.name_hint` is non-empty the hint
is used directly.

## Example

### Non-CORE_GROUP level (HOST)

**Before**:

```python
@pl.program
class Before:
    @pl.function  # Opaque
    def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        with pl.at(level=pl.Level.HOST):
            y = helper(x)
        return y
```

**After** (parent stays `Opaque`, outlined function is `Opaque`):

```python
@pl.program
class After:
    @pl.function  # unchanged
    def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        y = self.main_host_0(x)
        return y

    @pl.function  # Opaque
    def main_host_0(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        y = helper(x)
        return y
```

### Multiple outputs

```python
with pl.at(level=pl.Level.HOST):
    a_tile = pl.load(a, [0], [64])
    b_tile = pl.load(b, [0], [64])
    c_tile = pl.add(a_tile, b_tile)
    out_a = pl.store(c_tile, [0], out)
    out_b = pl.mul(c_tile, 2.0)
# both out_a and out_b used after the scope
x = out_a + out_b
```

After outlining, the body becomes:

```python
out_a, out_b = self.main_host_0(a, b, out)  # multiple return values
x = out_a + out_b
```

### CORE_GROUP scopes are skipped

```python
@pl.function  # Opaque
def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    with pl.at(level=pl.Level.CORE_GROUP):   # <-- NOT outlined here
        tile = pl.load(x, [0], [64])
        result = pl.store(tile, [0], x)
    return result
```

This pass leaves the `CORE_GROUP` scope in place. The next pipeline pass,
[`OutlineIncoreScopes`](06-outline_incore_scopes.md), outlines it into
`Function(InCore)` and promotes the parent to `Orchestration`.

## Implementation

**Header**: `include/pypto/ir/transforms/passes.h`

```cpp
Pass OutlineHierarchyScopes();
```

**Implementation**: `src/ir/transforms/outline_hierarchy_scopes.cpp`

- Uses the shared `scope_outline_utils` to compute inputs/outputs
- Builds a new `Function(Opaque)` per non-`CORE_GROUP` scope
- Copies `role_` metadata onto the outlined function's attrs
- Never modifies the parent function's `func_type_`

**Python binding**: `python/bindings/modules/passes.cpp`

```cpp
passes.def("outline_hierarchy_scopes", &pass::OutlineHierarchyScopes,
           "Outline non-CORE_GROUP HierarchyScopeStmt regions into Opaque functions");
```

**Tests**: `tests/ut/ir/transforms/test_outline_hierarchy_scopes.py`

- Tests non-`CORE_GROUP` scope → `Opaque` function + parent unchanged
- Tests that `CORE_GROUP` scopes are left in place
- Tests input/output analysis
- Tests multiple non-`CORE_GROUP` scopes in the same parent function
- Tests SSA preservation

## Pass Properties

| Property | Value |
| -------- | ----- |
| Required | `SSAForm` |
| Produced | `SSAForm` |
| Invalidated | — |

`HierarchyOutlined` is produced by
[`OutlineIncoreScopes`](06-outline_incore_scopes.md), which runs next and
handles the remaining `CORE_GROUP` scopes.

## Pipeline Position

```text
... → ConvertToSSA → NormalizeStmtStructure → FlattenCallExpr →
OutlineHierarchyScopes → OutlineIncoreScopes → OutlineClusterScopes →
ConvertTensorToTileOps → ...
```
