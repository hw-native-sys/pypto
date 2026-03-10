# Plans and Proposals

## Core Principle

**When presenting a plan, proposal, or implementation strategy, always include detailed, comprehensive examples.**

Abstract descriptions are insufficient. Every proposed change must be grounded in concrete code snippets, file paths, and before/after comparisons so the user can evaluate the plan accurately.

## Requirements

### 1. Show Concrete Code, Not Abstract Descriptions

````text
# ❌ Vague
"We should add a validation method to the Call class."

# ✅ Detailed
"Add a `ValidateArgs()` method to `Call` in `include/pypto/ir/expr.h`:

```cpp
// include/pypto/ir/expr.h
class Call : public Expr {
 public:
  // ... existing methods ...

  /// Validate that call arguments match the callee signature.
  void ValidateArgs() const;
};
```

Implementation in `src/ir/expr.cpp`:

```cpp
void Call::ValidateArgs() const {
  CHECK(op_.defined()) << "Call must have a valid callee";
  for (size_t i = 0; i < args_.size(); ++i) {
    CHECK(args_[i].defined())
        << "Call argument " << i << " must not be null";
  }
}
```

Python binding in `python/bindings/modules/ir.cpp`:

```cpp
.def("validate_args", &Call::ValidateArgs,
     "Validate call arguments match callee signature")
```

Type stub in `python/pypto/pypto_core/ir.pyi`:

```python
def validate_args(self) -> None:
    """Validate call arguments match callee signature."""
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
"Modify `IRPythonPrinter::VisitStmt_` in `src/ir/transforms/python_printer.cpp`:

Before:
```cpp
void IRPythonPrinter::VisitStmt_(const ForStmtPtr& op) {
  PrintIndent();
  os_ << "for " << op->loop_var_->name_ << " in ";
  os_ << prefix_ << ".range(" << Print(op->start_) << ", "
      << Print(op->stop_) << "):" << std::endl;
  PrintBody(op->body_);
}
```

After:
```cpp
void IRPythonPrinter::VisitStmt_(const ForStmtPtr& op) {
  PrintIndent();
  os_ << "for " << op->loop_var_->name_ << " in ";
  os_ << prefix_ << ".range(" << Print(op->start_) << ", "
      << Print(op->stop_) << ", " << Print(op->step_) << "):"
      << std::endl;
  PrintBody(op->body_);
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
1. `include/pypto/ir/stmt.h` — Add `IsChunked()` method to `ForStmt` (line ~483)
2. `src/ir/stmt.cpp` — Implement `IsChunked()` checking `chunk_size_.has_value()`
3. `python/bindings/modules/ir.cpp` — Expose `is_chunked()` on `ForStmt`
4. `python/pypto/pypto_core/ir.pyi` — Add `is_chunked() -> bool` stub
5. `tests/ut/ir/statements/test_for_stmt.py` — Add test for `is_chunked()` accessor

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
1. C++ header (`include/pypto/ir/stmt.h`): Declare `IsChunked()` — must come first
2. C++ impl (`src/ir/stmt.cpp`): Implement `IsChunked()` — depends on step 1
3. Build C++: `cmake --build build` — verify compilation before binding work
4. Python binding (`python/bindings/modules/ir.cpp`): Add `.def("is_chunked", ...)`
5. Type stub (`python/pypto/pypto_core/ir.pyi`): Add `is_chunked()` signature
6. Test (`tests/ut/ir/statements/test_for_stmt.py`): Add `test_is_chunked()`
7. Build and test: `cmake --build build && cd build && ctest` — full verification
"
````

### 5. Address Edge Cases and Alternatives

When the plan involves design decisions, explain the trade-offs:

````text
# ❌ Vague
"We could use either approach."

# ✅ Detailed
"Two approaches for step validation in ForStmt:

Option A — Validate in constructor (include/pypto/ir/stmt.h:443):
```cpp
ForStmt::ForStmt(VarPtr loop_var, ExprPtr start, ExprPtr stop,
                 ExprPtr step, ...) : Stmt(std::move(span)), ... {
  CHECK(step_.defined()) << "ForStmt step expression must not be null";
}
```
Pro: Invalid ForStmt can never exist
Con: Makes deserialization harder (must validate during parsing)

Option B — Validate in a verification pass:
```cpp
void VerifyPass::VisitStmt_(const ForStmtPtr& op) {
  CHECK(op->step_.defined())
      << "ForStmt step must not be null";
}
```
Pro: Flexible — allows constructing incomplete IR during transformations
Con: Invalid IR can exist temporarily

Recommendation: Option B — aligns with existing pattern where OpStmts
validates via INTERNAL_CHECK (see src/ir/stmt.cpp:25) rather than CHECK,
and IfStmt/WhileStmt constructors (stmt.h:288,544) do no validation.
"
````

### 6. Describe the Test Strategy

````text
# ❌ Vague
"I will add tests."

# ✅ Detailed
"Test strategy:
1. Unit test (`tests/ut/ir/statements/test_for_stmt.py`):
   - Add `test_is_chunked_true` to verify `is_chunked()` returns True
     when `chunk_size` is provided.
   - Add `test_is_chunked_false` to verify it returns False when
     `chunk_size` is None.
2. Printer test (`tests/ut/ir/printing/`):
   - Update ForStmt printing test to verify the new step expression
     appears in printed output.
3. Round-trip test (`tests/ut/ir/parser/`):
   - Add a ForStmt with explicit step to ensure it survives
     parse → print → reparse correctly.
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
