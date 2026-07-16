# OutlineIncoreScopes Pass

Outlines InCore scopes into separate functions.

## Overview

This pass transforms `InCoreScopeStmt` nodes into separate `Function(InCore)` definitions and replaces the scope with a Call to the outlined function.

**Requirements**:

- Input IR must be in SSA form (run ConvertToSSA first); SSAForm is preserved (produced) by this pass
- Only processes Opaque functions (InCore functions are left unchanged)

**When to use**: Run after ConvertToSSA when you need to extract InCore computation regions into separate callable functions.

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

1. **Scan for InCore Scopes**: Find all `InCoreScopeStmt` nodes in Opaque functions
2. **Analyze Inputs**: Determine external variable references (variables defined outside scope, used inside)
3. **Analyze Outputs**: Determine internal definitions used after scope (variables defined inside, used outside)
4. **Create Function**: Extract scope body into new `Function(scope_type=InCore)` with:
   - Parameters = input variables
   - Returns = output variables
   - Body = scope body
5. **Replace Scope**: Replace `InCoreScopeStmt` with:
   - Call to outlined function with input arguments
   - AssignStmt for each output variable
6. **Add to Program**: Add outlined function to program's function list

**Param-explicit returns**: the outlined function returns its
own parameters, not SSA result vars, whenever a tensor output writes through
a parameter — store-target outputs return the param directly, other outputs
are traced via the shared `return_lineage` utility. Kernel-allocated outputs
keep their SSA value. This makes the return→param mapping a pointer-identity
lookup for orchestration codegen (`ReturnParamsExplicit` invariant).

**Naming**:

- Default: `{original_func}_incore_{counter}` (e.g., `main_incore_0`, `main_incore_1`)
- User-provided: when `InCoreScopeStmt.name_hint` is non-empty, that name is used directly
  - `with pl.at(level=pl.Level.CORE_GROUP, name_hint="fused_add"):` → function named `fused_add`

**Name collisions** (`name_hint` is a *hint*, not a unique identifier — outlined
functions share one program-wide namespace, so collisions are resolved
automatically):

- **In-function** — two scopes in the same function sharing a `name_hint` get a
  numeric suffix: `my_kernel`, `my_kernel_0`.
- **Cross-function** — two *different* functions outlining scopes with the same
  `name_hint` (typically a reused `@pl.jit.inline` helper composed into a host
  program) are disambiguated by namespacing the collision under the originating
  function. The first function keeps the bare hint (stable, matching its
  standalone compilation); the later one is prefixed:
  - `single_a` → `dup_scope`, `single_b` → `single_b_dup_scope`

  This lets independently-runnable child kernels be composed into one
  `@pl.jit.host` program without manually renaming shared helper internals. The
  same rule applies to the sibling `OutlineHierarchyScopes` and
  `OutlineClusterScopes` passes (which share the outlining utility).

## Example

### Basic Outlining

**Before**:

```python
@pl.program
class Before:
    @pl.function  # Opaque function
    def main(self, x: Tensor[[64], FP32]) -> Tensor[[64], FP32]:
        y = x + 1

        with pl.at(level=pl.Level.CORE_GROUP):  # InCore scope
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
    @pl.function  # Opaque function
    def main(self, x: Tensor[[64], FP32]) -> Tensor[[64], FP32]:
        y = x + 1

        # Scope replaced with call + assignments
        result = self.main_incore_0(y, x)  # Call outlined function

        z = result + 2
        return z

    @pl.function(scope_type=InCore)  # Outlined InCore function
    def main_incore_0(self, y: Tensor[[64], FP32], x: Tensor[[64], FP32]) -> Tensor[[64], FP32]:
        # Scope body moved here
        tile = pl.load(y, [0], [64])
        tile_sq = pl.mul(tile, tile)
        result_tile = tile_sq + 1
        result = pl.store(result_tile, [0], x)
        return x  # store target: returns the param, not `result`
```

### Multiple Outputs

**Before**:

```python
with pl.at(level=pl.Level.CORE_GROUP):
    a_tile = pl.load(a, [0], [64])
    b_tile = pl.load(b, [0], [64])
    c_tile = pl.add(a_tile, b_tile)
    out_a = pl.store(c_tile, [0], out)
    out_b = pl.mul(c_tile, 2.0)
# Both out_a and out_b used after scope
x = out_a + out_b
```

**After**:

```python
out_a, out_b = self.main_incore_0(a, b, out)  # Multiple outputs
x = out_a + out_b

# Outlined function:
def main_incore_0(self, a, b, out):
    a_tile = pl.load(a, [0], [64])
    b_tile = pl.load(b, [0], [64])
    c_tile = pl.add(a_tile, b_tile)
    out_a = pl.store(c_tile, [0], out)
    out_b = pl.mul(c_tile, 2.0)
    return (out, out_b)  # out_a → param `out`; out_b is kernel-local, kept as-is
```

## Implementation

**Header**: `include/pypto/ir/transforms/passes.h`

```cpp
Pass OutlineIncoreScopes();
```

**Implementation**: `src/ir/transforms/outline_incore_scopes.cpp`

- Uses SSA analysis to determine inputs/outputs
- Creates new Function nodes with InCore scope type
- Replaces InCoreScopeStmt with Call + AssignStmt
- Manages function naming and counters

**Python binding**: `python/bindings/modules/passes.cpp`

```cpp
passes.def("outline_incore_scopes", &pass::OutlineIncoreScopes, "Outline InCore scopes");
```

**Tests**: `tests/ut/ir/transforms/test_outline_incore_scopes.py`

- Tests basic scope outlining
- Tests input/output analysis
- Tests multiple scopes in same function
- Tests nested scopes
- Tests SSA preservation

## Requirements

**SSA form required**: The pass relies on SSA properties:

- Single assignment ensures clear input/output analysis
- No variable shadowing simplifies scope analysis
- YieldStmt in control flow handled correctly

**Run ConvertToSSA first** if IR is not in SSA form.

**Mutually exclusive AIV-split mechanisms**: a function-level AUTO split
(`optimizations=[pl.split(mode)]`, carried as the scope's own `split_`) and
explicit `pl.split_aiv` regions (`SplitAivScopeStmt`) cannot coexist on one
scope. This pass rejects the combination (it bridges a single region's mode into
a function-level representative `split`, which would silently collide with the
user's `pl.split`). See
[`LowerAutoVectorSplit`](18-lower_auto_vector_split.md) for how the surviving
mechanism is lowered.
