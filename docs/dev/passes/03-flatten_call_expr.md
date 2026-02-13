# FlattenCallExpr Pass

Flattens nested call expressions into three-address code form.

## Overview

This pass ensures that call expressions do not appear in nested contexts by extracting them into temporary variables. It enforces a three-address code constraint where:

1. Call arguments cannot be calls
2. If conditions cannot be calls
3. For loop ranges (start/stop/step) cannot be calls
4. Binary/unary expression operands cannot be calls

**When to use**: Run this pass after SSA conversion and before code generation to simplify downstream analysis and code generation.

## API

| C++ | Python | Level |
|-----|--------|-------|
| `pass::FlattenCallExpr()` | `passes.flatten_call_expr()` | Function-level |

**Factory function**:
```cpp
Pass FlattenCallExpr();
```

**Python usage**:
```python
from pypto.pypto_core import passes

flatten_pass = passes.flatten_call_expr()
program_flat = flatten_pass(program)
```

## Algorithm

1. **Detect Nested Calls**: Identify call expressions in nested contexts
2. **Extract to Temps**: Create temporary variables (named _t0, _t1, etc.)
3. **Insert AssignStmt**: Add assignment statements before the original statement
4. **Replace with Var**: Replace nested call with temporary variable reference
5. **Handle Control Flow**: For if/for statements, insert into last OpStmts or create new one

**Extraction locations**:
- Before AssignStmt/EvalStmt: Insert directly before
- Before IfStmt/ForStmt: Insert into last OpStmts in preceding SeqStmts, or create new OpStmts

## Example

### Nested Call Arguments

**Before**:
```python
c = foo(bar(a))  # bar(a) is nested in foo's arguments
```

**After**:
```python
_t0 = bar(a)
c = foo(_t0)
```

### Nested Call in If Condition

**Before**:
```python
if is_valid(compute(x)):
    y = 1
```

**After**:
```python
_t0 = compute(x)
_t1 = is_valid(_t0)
if _t1:
    y = 1
```

### Multiple Nested Calls

**Before**:
```python
result = add(mul(a, b), div(c, d))
```

**After**:
```python
_t0 = mul(a, b)
_t1 = div(c, d)
result = add(_t0, _t1)
```

### Nested in Binary Expression

**Before**:
```python
x = compute(a) + compute(b)
```

**After**:
```python
_t0 = compute(a)
_t1 = compute(b)
x = _t0 + _t1
```

## Implementation

**Header**: `include/pypto/ir/transforms/passes.h`
```cpp
Pass FlattenCallExpr();
```

**Implementation**: `src/ir/transforms/flatten_call_expr.cpp`
- Uses IRMutator to traverse expressions
- Maintains temporary variable counter
- Collects extracted assignments
- Rebuilds statements with flattened expressions

**Python binding**: `python/bindings/modules/passes.cpp`
```cpp
passes.def("flatten_call_expr", &pass::FlattenCallExpr, "Flatten nested calls");
```

**Tests**: `tests/ut/ir/transforms/test_flatten_call_expr.py`
- Tests call arguments extraction
- Tests if condition extraction
- Tests for range extraction
- Tests binary/unary expression extraction
- Tests multiple nested calls

## Error Types

The pass can detect and report nested call violations via `NestedCallErrorType`:

- `CALL_IN_CALL_ARGS`: Call in call arguments
- `CALL_IN_IF_CONDITION`: Call in if condition
- `CALL_IN_FOR_RANGE`: Call in for range
- `CALL_IN_BINARY_EXPR`: Call in binary expression
- `CALL_IN_UNARY_EXPR`: Call in unary expression
