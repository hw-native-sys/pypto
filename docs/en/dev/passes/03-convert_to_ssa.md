# ConvertToSSA Pass

Converts non-SSA IR to Static Single Assignment (SSA) form with variable renaming, phi nodes, and iter_args.

## Overview

This pass transforms IR with multiple assignments to the same variable into SSA form where each variable is assigned exactly once. It handles:

- **Straight-line code**: Multiple assignments to the same variable
- **If statements**: Variables modified in one or both branches
- **For loops**: Variables modified inside the loop body
- **Mixed SSA/non-SSA**: Preserves existing SSA structure while converting non-SSA parts

**Requires**: `TypeChecked` property. `TypeChecked` is verified automatically at BASIC level once produced; use a `VerificationInstrument` via `PassContext` to validate required properties before this pass runs.

**When to use**: Run this pass before any optimization or analysis that requires SSA form (e.g., OutlineIncoreScopes, memory optimization passes).

## API

| C++ | Python | Level |
| --- | ------ | ----- |
| `pass::ConvertToSSA()` | `passes.convert_to_ssa()` | Function-level |

**Factory function**:

```cpp
Pass ConvertToSSA();
```

**Python usage**:

```python
from pypto.pypto_core import passes

ssa_pass = passes.convert_to_ssa()
program_ssa = ssa_pass(program)
```

## Algorithm

1. **Variable Renaming**: Rename variables with SSA suffixes (x → x__ssa_v0, x__ssa_v1, x__ssa_v2) for each assignment
2. **Phi Nodes for If**: Add phi nodes (return_vars + YieldStmt) for variables modified in if branches, including variables defined independently in both branches
3. **Iter_args for Loops**: Convert loop-modified variables to iter_args + return_vars pattern with YieldStmt
4. **Escaping Variable Promotion**: Variables first defined inside a loop body but used after the loop are promoted to iter_args + return_vars via forward-use analysis
5. **Scope Tracking**: Track variable definitions across nested scopes
6. **Preservation**: Keep existing SSA constructs unchanged

**Key transformations**:

- `x = 1; x = 2` → `x__ssa_v0 = 1; x__ssa_v1 = 2`
- If with divergent assignments → add return_vars and YieldStmt in both branches
- For loops with loop-carried dependencies → add iter_args/return_vars/YieldStmt
- Loop-escaping variables → promoted to iter_args with matching Out parameter as initial value

## Example

### Straight-line Code

**Before**:

```python
x = 1
y = x + 2
x = 3  # Multiple assignment
z = x + 4
```

**After**:

```python
x__ssa_v0 = 1
y = x__ssa_v0 + 2
x__ssa_v1 = 3
z = x__ssa_v1 + 4
```

### If Statement

**Before**:

```python
x = 1
if condition:
    x = 2  # Modified in then branch
z = x + 3  # Uses x after if
```

**After**:

```python
x__ssa_v0 = 1
if condition:
    x__ssa_v1 = 2
    yield (x__ssa_v1,)  # Yield modified variable
else:
    yield (x__ssa_v0,)  # Yield original variable
return_vars = (x__phi_v2,)  # Phi node
z = x__phi_v2 + 3
```

### For Loop

**Before**:

```python
sum = 0
for i in range(10):
    sum = sum + i  # Loop-carried dependency
```

**After**:

```python
sum__ssa_v0 = 0
for i in range(10):
    iter_args = (sum__iter_v1,)
    init_values = (sum__ssa_v0,)
    # Loop body
    sum__ssa_v2 = sum__iter_v1 + i
    yield (sum__ssa_v2,)
return_vars = (sum__rv_v3,)
```

### Loop-Escaping Variable

**Before** (variable `out` first defined inside loop, used after):

```python
for i in range(4):
    tile_c = tile.add(tile_a, tile_b)
    out = tile.store(tile_c, [offset, 0], c)  # first assignment inside loop
return out  # used after loop
```

**After** (promoted to iter_arg + return_var):

```python
for i in range(4):
    iter_args = (out__iter_v0,)
    init_values = (c__ssa_v0,)  # Out parameter as initial value
    tile_c__ssa_v0 = tile.add(tile_a__ssa_v0, tile_b__ssa_v0)
    out__ssa_v1 = tile.store(tile_c__ssa_v0, [offset__ssa_v0, 0], out__iter_v0)
    yield (out__ssa_v1,)
return_vars = (out__rv_v2,)
return out__rv_v2
```

## Implementation

**Header**: `include/pypto/ir/transforms/passes.h`

```cpp
Pass ConvertToSSA();
```

**Implementation**: `src/ir/transforms/convert_to_ssa_pass.cpp`

- Uses manual statement dispatch with explicit scope management
- Maintains version maps for variable renaming
- Inserts YieldStmt and manages return_vars/iter_args

**Python binding**: `python/bindings/modules/passes.cpp`

```cpp
passes.def("convert_to_ssa", &pass::ConvertToSSA, "Convert to SSA form");
```

**Tests**: `tests/ut/ir/transforms/test_convert_to_ssa_pass.py`

- Tests straight-line renaming
- Tests if statement phi nodes
- Tests for loop iter_args
- Tests nested scopes
- Tests mixed SSA/non-SSA
