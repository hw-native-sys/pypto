# Utility Passes

Normalization and cleanup passes for IR structure.

## Overview

These utility passes handle IR normalization and cleanup tasks:

1. **NormalizeStmtStructure**: Ensures consistent statement structure
2. **VerifyNoNestedCall**: Verification pass for three-address code form

These are typically used internally by other passes or for specific normalization needs.

## SeqStmts::Flatten

Static helper method for creating well-formed `SeqStmts` nodes. Code that constructs `SeqStmts` should prefer `Flatten()` to satisfy the `NoRedundantBlocks` structural property.

### SeqStmts::Flatten

```cpp
// Signature (include/pypto/ir/stmt.h)
static StmtPtr SeqStmts::Flatten(std::vector<StmtPtr> stmts, Span span);
```

| Input | Output |
| ----- | ------ |
| `Flatten({a, SeqStmts({b, c}), d}, span)` | `SeqStmts({a, b, c, d})` |
| `Flatten({a}, span)` | `a` (unwrapped) |
| `Flatten({}, span)` | `SeqStmts({})` |

Nested `SeqStmts` children are absorbed (flattened). Single-child results are unwrapped.

### Usage in IRMutator

The base `IRMutator::VisitStmt_(SeqStmtsPtr)` calls `SeqStmts::Flatten()` automatically. All passes inheriting from `IRMutator` produce well-formed IR without extra effort.

### Usage in Passes

Passes that construct `SeqStmts` directly (not via the mutator) should use `Flatten()`:

```cpp
// ✅ Good — always well-formed
return SeqStmts::Flatten(std::move(new_stmts), op->span_);

// ❌ Bad — may produce single-child or nested SeqStmts
return std::make_shared<SeqStmts>(new_stmts, op->span_);
```

### NoRedundantBlocks (Structural Property)

`NoRedundantBlocks` is a **structural property** — verified at pipeline start and expected to hold at all times. It checks:

| Check | SeqStmts |
| ----- | -------- |
| Single-child (should unwrap) | yes |
| Nested (should flatten) | yes |

---

## NormalizeStmtStructure

Ensures IR is in a normalized form with consistent structure.

**Requires**: TypeChecked property.

### Purpose

Normalizes statement structure by:

1. Flattening nested `SeqStmts`
2. Unwrapping single-child `SeqStmts`
3. Leaving `AssignStmt`/`EvalStmt` as direct children of `SeqStmts`

### API

| C++ | Python |
| --- | ------ |
| `pass::NormalizeStmtStructure()` | `passes.normalize_stmt_structure()` |

### Algorithm

1. **Flatten Nesting**: Absorb nested `SeqStmts` into the parent
2. **Unwrap Single-child**: Return single child directly (no redundant `SeqStmts` wrapper)
3. **Preserve Control Flow**: Keep `IfStmt`/`ForStmt`/`WhileStmt` bodies otherwise unchanged

### Example

**Before**:

```python
def func(...):
    x = 1  # Direct AssignStmt
```

**After**:

```python
def func(...):
    AssignStmt(x, 1)  # Single-child SeqStmts is unwrapped
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
    AssignStmt(a, 1),
    AssignStmt(b, 2),
    IfStmt(...)
])
```

### Implementation

**Factory**: `pass::NormalizeStmtStructure()`
**File**: `src/ir/transforms/utils/normalize_stmt_structure.cpp`
**Tests**: `tests/ut/ir/transforms/test_normalize_stmt_structure_pass.py`

---

## Verify NoNestedCall (Part of IRVerifier)

Verifies that IR is in three-address code form (no nested calls).

### Purpose

This verification rule (part of IRVerifier) checks that FlattenCallExpr pass has been run successfully. It detects:

- `CALL_IN_CALL_ARGS`: Call in call arguments
- `CALL_IN_IF_CONDITION`: Call in if condition
- `CALL_IN_FOR_RANGE`: Call in for range
- `CALL_IN_BINARY_EXPR`: Call in binary expression
- `CALL_IN_UNARY_EXPR`: Call in unary expression

### API

Verified via PropertyVerifierRegistry (not a standalone pass):

```python
# Verify with default properties (includes NoNestedCalls)
verify_pass = passes.run_verifier()

# Or exclude NoNestedCalls from verification
props = passes.get_default_verify_properties()
props.remove(passes.IRProperty.NoNestedCalls)
verify_pass = passes.run_verifier(properties=props)
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
| **NormalizeStmtStructure** | Before passes that expect flat `SeqStmts` structure |
| **VerifyNoNestedCall** | After FlattenCallExpr to ensure correctness |

## Implementation Files

| Pass | Header | Implementation | Tests |
| ---- | ------ | -------------- | ----- |
| NormalizeStmtStructure | `passes.h` | `normalize_stmt_structure.cpp` | `test_normalize_stmt_structure_pass.py` |
| VerifyNoNestedCall | `passes.h` | `verify_no_nested_call_pass.cpp` | `test_verifier.py` |
