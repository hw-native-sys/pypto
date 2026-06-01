# Error Handling

PyPTO's error handling framework provides structured exceptions with C++ stack traces, assertion macros with IR source location tracking, and a diagnostic system for verification errors.

## Overview

| Component | Header | Purpose |
| --------- | ------ | ------- |
| **Exception hierarchy** | `include/pypto/core/error.h` | Typed exceptions (`ValueError`, `InternalError`, …) with automatic stack trace capture |
| **Assertion macros** | `include/pypto/core/logging.h` | `CHECK` / `CHECK_SPAN`, `INTERNAL_CHECK_SPAN`, `UNREACHABLE` / `UNREACHABLE_SPAN`, etc. |
| **Diagnostic system** | `include/pypto/core/error.h` | `Diagnostic` / `VerificationError` for verification passes |
| **Span** | `include/pypto/ir/span.h` | IR source location attached to diagnostics and internal checks |

## Exception Hierarchy

All exceptions inherit from `Error`, which captures the C++ stack trace at construction time via `libbacktrace`.

```text
std::runtime_error
  └── Error                  (base: auto stack trace capture)
        ├── ValueError       (→ Python ValueError)
        ├── TypeError        (→ Python TypeError)
        ├── RuntimeError     (→ Python RuntimeError)
        ├── NotImplementedError
        ├── IndexError
        ├── AssertionError
        ├── InternalError    (→ Python RuntimeError — internal bugs)
        └── VerificationError (carries vector<Diagnostic>)
```

`Error::GetFullMessage()` returns the error message plus a formatted C++ stack trace.

## Assertion Macros

### User-facing checks — `CHECK` / `CHECK_SPAN`

Throw `ValueError` when a user-visible contract is violated. `CHECK_SPAN` attaches the IR source location, just like `INTERNAL_CHECK_SPAN` — prefer it whenever a `Span` is reachable so the user can see which DSL line tripped the check:

```cpp
CHECK(args.size() == 2) << "op requires exactly 2 arguments, got " << args.size();
CHECK_SPAN(shape.size() == 2, span) << "tensor.matmul: only 2D inputs are supported";
```

The `span` argument follows the same safety rule as `INTERNAL_CHECK_SPAN`: it is evaluated only on failure, but unconditionally evaluated there. The span source must therefore be safe to dereference at the failure point (typically a local `Span` variable or a sibling IR node known non-null).

### Internal invariant checks — `INTERNAL_CHECK_SPAN`

Throw `InternalError` when an internal invariant is violated. Always attach the IR node's `Span` so the error message includes the user's source location:

```cpp
INTERNAL_CHECK_SPAN(op->var_, op->span_) << "AssignStmt has null var";
INTERNAL_CHECK_SPAN(new_value, op->span_) << "AssignStmt value mutated to null";
```

When the check fails, the error message includes both the IR source location and the C++ location:

```text
AssignStmt has null var [user_model.py:42:1]
Check failed: op->var_ at src/ir/transforms/mutator.cpp:301
```

There is also `INTERNAL_UNREACHABLE_SPAN` for code paths that should never be reached:

```cpp
INTERNAL_UNREACHABLE_SPAN(span) << "Unknown binary expression kind";
```

### Variants without span

`INTERNAL_CHECK` and `INTERNAL_UNREACHABLE` do not carry IR source location. They are appropriate when no `Span` is available (e.g., in non-IR contexts like arithmetic utilities or registry lookups). When an IR node is being processed and `op->span_` is accessible, prefer the `_SPAN` variants.

### Unreachable code paths — `UNREACHABLE` / `UNREACHABLE_SPAN`

Throw `ValueError` for code paths that should be unreachable from a user perspective. Prefer `UNREACHABLE_SPAN` when an IR span is available:

```cpp
UNREACHABLE << "Unsupported data type: " << dtype;
UNREACHABLE_SPAN(node->span_) << "Unsupported data type: " << dtype;
```

### Macro Reference

| Macro | Exception Type | Span | Status |
| ----- | -------------- | ---- | ------ |
| `CHECK(expr)` | `ValueError` | No | Active |
| `CHECK_SPAN(expr, span)` | `ValueError` | Yes | **Preferred** when span available |
| `UNREACHABLE` | `ValueError` | No | Active |
| `UNREACHABLE_SPAN(span)` | `ValueError` | Yes | **Preferred** when span available |
| `INTERNAL_CHECK_SPAN(expr, span)` | `InternalError` | Yes | **Preferred** |
| `INTERNAL_UNREACHABLE_SPAN(span)` | `InternalError` | Yes | **Preferred** |
| `INTERNAL_CHECK(expr)` | `InternalError` | No | Active (use `_SPAN` when span available) |
| `INTERNAL_UNREACHABLE` | `InternalError` | No | Active (use `_SPAN` when span available) |

## Diagnostic System

The diagnostic system is used by [IR verification passes](passes/99-verifier.md) to collect multiple issues before reporting.

Each `Diagnostic` carries:

| Field | Type | Purpose |
| ----- | ---- | ------- |
| `severity` | `DiagnosticSeverity` | Error or Warning |
| `rule_name` | `string` | Which verification rule detected the issue |
| `error_code` | `int` | Numeric error identifier |
| `message` | `string` | Human-readable description |
| `span` | `Span` | IR source location |

`VerificationError` is thrown when verification fails, carrying all collected diagnostics.

## Span and Source Location

Every IR node inherits a `span_` field from `IRNode` (see [IR Overview](ir/00-overview.md)). This field tracks the user's source location (filename, line, column) and is used in two error paths:

1. **Verification diagnostics** — verifier passes record `op->span_` into `Diagnostic` objects
2. **Assertion checks** — `CHECK_SPAN` / `UNREACHABLE_SPAN` / `INTERNAL_CHECK_SPAN` / `INTERNAL_UNREACHABLE_SPAN` embed `span.to_string()` into the failure message

When a `Span` is valid, the error output appends `[file:line:col]` to the message. When `Span::unknown()` is used, no source location is shown.

## Python API

```python
import pypto

# User-facing check (raises ValueError)
pypto.check(condition, "error message")

# Internal invariant check with span (raises RuntimeError)
pypto.internal_check_span(condition, "error message", span)

# Raise InternalError with span (for testing or unconditional error paths)
pypto.raise_internal_error_with_span("error message", span)

# Internal invariant check without span
pypto.internal_check(condition, "error message")
```

## Migration Guide

When writing or touching code in IR transforms, passes, or codegen:

1. Identify the current IR node being processed (`op`, `stmt`, `expr`, etc.)
2. Replace `INTERNAL_CHECK(expr)` with `INTERNAL_CHECK_SPAN(expr, op->span_)` (and `INTERNAL_UNREACHABLE` with `INTERNAL_UNREACHABLE_SPAN`)
3. Likewise replace user-facing `CHECK(expr)` with `CHECK_SPAN(expr, op->span_)` (and `UNREACHABLE` with `UNREACHABLE_SPAN`) when a span is reachable
4. If a `Span` is available as a function parameter (e.g., in `Reconstruct*` helpers or op-conversion lambdas), use that directly

```cpp
// Before:
INTERNAL_CHECK(op->body_) << "ForStmt has null body";
CHECK(args.size() == 2) << "tensor.matmul requires 2 args";

// After (preferred when span is available):
INTERNAL_CHECK_SPAN(op->body_, op->span_) << "ForStmt has null body";
CHECK_SPAN(args.size() == 2, span) << "tensor.matmul requires 2 args";
```

### Inside passes: CHECK vs INTERNAL_CHECK

Passes operate on IR that has already been verified by earlier passes. A failed invariant inside a pass therefore almost always indicates a **compiler bug**, not a user error — use `INTERNAL_CHECK_SPAN` / `INTERNAL_UNREACHABLE_SPAN`. Reserve `CHECK_SPAN` for genuine user-facing limitations that the user can work around (e.g. "4D scatter_update is not yet lowered — use 2D"). If you're unsure, ask: *would the message read "this is a PyPTO bug, please report" or "please change your code"?*

## Related

- [IR Overview — Source Location Tracking](ir/00-overview.md)
- [IR Verifier — Diagnostic System](passes/99-verifier.md)
- `include/pypto/core/error.h` — Exception classes and `Diagnostic`
- `include/pypto/core/logging.h` — Assertion macros and `FatalLogger`
- `include/pypto/ir/span.h` — `Span` class
