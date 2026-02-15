# ConvertToSSA Pass

Converts non-SSA IR to Static Single Assignment (SSA) form with variable renaming, phi nodes, and iter_args.

## Overview

This pass transforms IR with multiple assignments to the same variable into SSA form where each variable is assigned exactly once. It handles:

- **Straight-line code**: Multiple assignments to the same variable
- **If statements**: Variables modified in one or both branches
- **For loops**: Variables modified inside the loop body
- **Mixed SSA/non-SSA**: Preserves existing SSA structure while converting non-SSA parts

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

1. **Variable Renaming**: Rename variables with version suffixes (x → x_0, x_1, x_2) for each assignment
2. **Phi Nodes for If**: Add phi nodes (return_vars + YieldStmt) for variables modified in if branches
3. **Iter_args for Loops**: Convert loop-modified variables to iter_args + return_vars pattern with YieldStmt
4. **Scope Tracking**: Track variable definitions across nested scopes
5. **Preservation**: Keep existing SSA constructs unchanged

**Key transformations**:

- `x = 1; x = 2` → `x_0 = 1; x_1 = 2`
- If with divergent assignments → add return_vars and YieldStmt in both branches
- For loops with loop-carried dependencies → add iter_args/return_vars/YieldStmt

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
x_0 = 1
y = x_0 + 2
x_1 = 3
z = x_1 + 4
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
x_0 = 1
if condition:
    x_1 = 2
    yield (x_1,)  # Yield modified variable
else:
    yield (x_0,)  # Yield original variable
return_vars = (x_2,)  # Phi node
z = x_2 + 3
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
sum_0 = 0
for i in range(10):
    iter_args = (sum_1,)
    init_values = (sum_0,)
    # Loop body
    sum_2 = sum_1 + i
    yield (sum_2,)
return_vars = (sum_3,)
```

## Implementation

**Header**: `include/pypto/ir/transforms/passes.h`

```cpp
Pass ConvertToSSA();
```

**Implementation**: `src/ir/transforms/convert_to_ssa.cpp`

- Uses IRMutator pattern to traverse and transform IR
- Maintains version maps for variable renaming
- Inserts YieldStmt and manages return_vars/iter_args

**Python binding**: `python/bindings/modules/passes.cpp`

```cpp
passes.def("convert_to_ssa", &pass::ConvertToSSA, "Convert to SSA form");
```

**Tests**: `tests/ut/ir/transforms/test_convert_to_ssa.py`

- Tests straight-line renaming
- Tests if statement phi nodes
- Tests for loop iter_args
- Tests nested scopes
- Tests mixed SSA/non-SSA
