# Utility Passes

Normalization and cleanup passes for IR structure.

## Overview

These utility passes handle IR normalization and cleanup tasks:

1. **NormalizeStmtStructure**: Ensures consistent statement structure
2. **FlattenSingleStmt**: Removes unnecessary nesting
3. **VerifyNoNestedCall**: Verification pass for three-address code form

These are typically used internally by other passes or for specific normalization needs.

## NormalizeStmtStructure

Ensures IR is in a normalized form with consistent structure.

**Requires**: TypeChecked property.

### Purpose

Normalizes statement structure by:

1. Wrapping function/if/for bodies in SeqStmts
2. Wrapping consecutive AssignStmt/EvalStmt in OpStmts within SeqStmts

### API

| C++ | Python |
| --- | ------ |
| `pass::NormalizeStmtStructure()` | `passes.normalize_stmt_structure()` |

### Algorithm

1. **Ensure SeqStmts**: Wrap non-SeqStmts bodies in SeqStmts
2. **Group Operations**: Wrap consecutive AssignStmt/EvalStmt in OpStmts
3. **Preserve Control Flow**: Keep IfStmt/ForStmt/WhileStmt unwrapped

### Example

**Before**:

```python
def func(...):
    x = 1  # Direct AssignStmt (not in SeqStmts)
```

**After**:

```python
def func(...):
    SeqStmts([OpStmts([AssignStmt(x, 1)])])
```

**Before**:

```python
SeqStmts([
    AssignStmt(a, 1),  # Consecutive operations
    AssignStmt(b, 2),
    IfStmt(...)
])
```

**After**:

```python
SeqStmts([
    OpStmts([AssignStmt(a, 1), AssignStmt(b, 2)]),  # Wrapped in OpStmts
    IfStmt(...)
])
```

### Implementation

**Factory**: `pass::NormalizeStmtStructure()`
**File**: `src/ir/transforms/normalize_stmt_structure.cpp`
**Tests**: `tests/ut/ir/transforms/test_normalize_stmt_structure_pass.py`

---

## FlattenSingleStmt

Recursively flattens single-statement blocks to simplify IR.

**Requires**: TypeChecked property.

### Purpose

Removes unnecessary nesting:

- SeqStmts with only one statement → that statement
- OpStmts with only one statement → that statement
- Applied recursively

**Note**: This pass does NOT enforce that Function/IfStmt/ForStmt body must be SeqStmts. It will flatten them if they contain only a single statement.

### API

| C++ | Python |
| --- | ------ |
| `pass::FlattenSingleStmt()` | `passes.flatten_single_stmt()` |

### Algorithm

1. **Traverse IR**: Visit all SeqStmts and OpStmts nodes
2. **Check Count**: If node contains exactly one statement
3. **Replace**: Replace node with that single statement
4. **Recurse**: Continue until no more single-statement blocks

### Example

**Before**:

```python
SeqStmts([OpStmts([AssignStmt(x, 1)])])
```

**After**:

```python
AssignStmt(x, 1)
```

**Before**:

```python
SeqStmts([OpStmts([AssignStmt(x, 1), AssignStmt(y, 2)])])
```

**After**:

```python
OpStmts([AssignStmt(x, 1), AssignStmt(y, 2)])
# Only outer SeqStmts flattened, OpStmts preserved (has 2 statements)
```

### Implementation

**Factory**: `pass::FlattenSingleStmt()`
**File**: `src/ir/transforms/flatten_single_stmt.cpp`
**Tests**: `tests/ut/ir/transforms/test_flatten_single_stmt_pass.py`

---

## Verify NoNestedCall (Part of RunVerifier)

Verifies that IR is in three-address code form (no nested calls).

### Purpose

This verification rule (part of IRVerifier) checks that FlattenCallExpr pass has been run successfully. It detects:

- `CALL_IN_CALL_ARGS`: Call in call arguments
- `CALL_IN_IF_CONDITION`: Call in if condition
- `CALL_IN_FOR_RANGE`: Call in for range
- `CALL_IN_BINARY_EXPR`: Call in binary expression
- `CALL_IN_UNARY_EXPR`: Call in unary expression

### API

Part of `RunVerifier` pass (not standalone):

```python
# Enable/disable via RunVerifier
verifier_pass = passes.run_verifier(disabled_rules=["NoNestedCall"])
```

### Implementation

**File**: `src/ir/verifier/verify_no_nested_call_pass.cpp`
**Rule name**: `"NoNestedCall"`
**Tests**: `tests/ut/ir/transforms/test_verifier.py`

---

## Usage Patterns

### Normalization Pipeline

```python
# Typical normalization sequence
program = passes.normalize_stmt_structure()(program)
program = passes.flatten_single_stmt()(program)
```

### Cleanup After Transformation

```python
# After a pass that might create single-statement blocks
program = some_transformation_pass()(program)
program = passes.flatten_single_stmt()(program)  # Clean up
```

### Verification

```python
# Verify three-address code form
verifier = passes.run_verifier()  # Includes NoNestedCall by default
verified_program = verifier(program)  # Throws if nested calls found
```

---

## When to Use

| Pass | When to Use |
| ---- | ----------- |
| **NormalizeStmtStructure** | Before passes that expect consistent SeqStmts/OpStmts structure |
| **FlattenSingleStmt** | After transformations to clean up unnecessary nesting |
| **VerifyNoNestedCall** | After FlattenCallExpr to ensure correctness |

## Implementation Files

| Pass | Header | Implementation | Tests |
| ---- | ------ | -------------- | ----- |
| NormalizeStmtStructure | `passes.h` | `normalize_stmt_structure.cpp` | `test_normalize_stmt_structure_pass.py` |
| FlattenSingleStmt | `passes.h` | `flatten_single_stmt.cpp` | `test_flatten_single_stmt_pass.py` |
| VerifyNoNestedCall | `passes.h` | `ir_verifier.cpp` | `test_verifier.py` |
