# Plans and Proposals

## Core Principle

**When presenting a plan, proposal, or implementation strategy, always include detailed, comprehensive examples.**

Abstract descriptions are insufficient. Every proposed change must be grounded in concrete code snippets, file paths, and before/after comparisons so the user can evaluate the plan accurately.

## Requirements

### 1. Show Concrete Code, Not Abstract Descriptions

````text
# ❌ Vague
"We should add a validation method to the TensorExpr class."

# ✅ Detailed
"Add a `validate_shape()` method to `TensorExpr` in `include/pypto/ir/expr.h`:

```cpp
// include/pypto/ir/expr.h
class TensorExpr : public Expr {
 public:
  // ... existing methods ...

  /// Validate that the tensor shape is well-formed.
  /// Checks: all dimensions positive, rank within limits.
  void ValidateShape() const;
};
```

Implementation in `src/ir/expr.cpp`:

```cpp
void TensorExpr::ValidateShape() const {
  CHECK(GetRank() > 0) << "Tensor must have at least 1 dimension";
  for (int i = 0; i < GetRank(); ++i) {
    CHECK(GetShape(i) > 0)
        << "Dimension " << i << " must be positive, got " << GetShape(i);
  }
}
```

Python binding in `python/bindings/ir_binding.cpp`:

```cpp
.def("validate_shape", &TensorExpr::ValidateShape,
     "Validate tensor shape is well-formed")
```

Type stub in `python/pypto/pypto_core/__init__.pyi`:

```python
def validate_shape(self) -> None:
    """Validate tensor shape is well-formed."""
    ...
```
"
````

### 2. Include Before/After for Modifications

When proposing changes to existing code, show the current state and the proposed state:

````text
# ❌ Vague
"Refactor the print method to handle the new node type."

# ✅ Detailed
"Modify `PythonPrinter::VisitStmt` in `src/ir/printing/python_printer.cpp`:

Before:
```cpp
void PythonPrinter::VisitStmt(const ForStmt& stmt) {
  PrintIndent();
  os_ << "for " << stmt.GetVar() << " in range(...):" << std::endl;
  PrintBody(stmt.GetBody());
}
```

After:
```cpp
void PythonPrinter::VisitStmt(const ForStmt& stmt) {
  PrintIndent();
  os_ << "for " << stmt.GetVar() << " in ";
  PrintRange(stmt.GetRange());
  os_ << ":" << std::endl;
  PrintBody(stmt.GetBody());
}
```
"
````

### 3. List All Affected Files With Specific Changes

````text
# ❌ Vague
"This change touches the IR layer, bindings, and tests."

# ✅ Detailed
"Files to modify:
1. `include/pypto/ir/stmt.h` — Add `GetRange()` method to `ForStmt` (line ~142)
2. `src/ir/stmt.cpp` — Implement `GetRange()` returning the loop range expression
3. `python/bindings/ir_binding.cpp` — Expose `get_range()` on `ForStmt` (line ~305)
4. `python/pypto/pypto_core/__init__.pyi` — Add `get_range() -> RangeExpr` stub
5. `tests/ut/ir/statements/test_for_stmt.py` — Add test for `get_range()` accessor

New files:
- None
"
````

### 4. Specify Step Order and Dependencies

````text
# ❌ Vague
"We need to update C++, bindings, and stubs."

# ✅ Detailed
"Implementation order:
1. C++ header (`include/pypto/ir/stmt.h`): Declare `GetRange()` — must come first
2. C++ impl (`src/ir/stmt.cpp`): Implement `GetRange()` — depends on step 1
3. Build C++: `cmake --build build` — verify compilation before binding work
4. Python binding (`python/bindings/ir_binding.cpp`): Add `.def("get_range", ...)`
5. Type stub (`python/pypto/pypto_core/__init__.pyi`): Add `get_range()` signature
6. Test (`tests/ut/ir/statements/test_for_stmt.py`): Add `test_get_range()`
7. Build and test: `cmake --build build && cd build && ctest` — full verification
"
````

### 5. Address Edge Cases and Alternatives

When the plan involves design decisions, explain the trade-offs:

````text
# ❌ Vague
"We could use either approach."

# ✅ Detailed
"Two approaches for range validation:

Option A — Validate in constructor:
```cpp
ForStmt::ForStmt(Var var, RangeExpr range, Body body) {
  CHECK(range.GetStep() != 0) << "Loop step cannot be zero";
  // ...
}
```
Pro: Invalid ForStmt can never exist
Con: Makes deserialization harder (must validate during parsing)

Option B — Validate in pass:
```cpp
void ValidatePass::VisitStmt(const ForStmt& stmt) {
  CHECK(stmt.GetRange().GetStep() != 0)
      << "Loop step cannot be zero at " << stmt.GetLocation();
}
```
Pro: Flexible — allows constructing incomplete IR during transformations
Con: Invalid IR can exist temporarily

Recommendation: Option A — aligns with existing pattern in IfStmt and WhileStmt
constructors (see include/pypto/ir/stmt.h:89,112).
"
````

## Summary

| Element | Required in Plan |
| ------- | ---------------- |
| Code snippets | Yes — show actual proposed code |
| File paths with line numbers | Yes — pinpoint every change |
| Before/after comparisons | Yes — for modifications |
| Step ordering | Yes — with dependencies noted |
| Edge cases and alternatives | Yes — with trade-off analysis |
| Test strategy | Yes — describe what tests cover |
