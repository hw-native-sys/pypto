# Error Handling

PyPTO's error handling framework provides structured exceptions with C++ stack traces, assertion macros with IR source location tracking, and a diagnostic system for verification errors.

## Overview

| Component | Header | Purpose |
| --------- | ------ | ------- |
| **Exception hierarchy** | `include/pypto/core/error.h` | Typed exceptions (`ValueError`, `InternalError`, …) with automatic stack trace capture |
| **Assertion macros** | `include/pypto/core/logging.h` | `CHECK`, `INTERNAL_CHECK_SPAN`, `UNREACHABLE`, etc. |
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

### User-facing checks — `CHECK`

Throw `ValueError` when a user-visible contract is violated:

```cpp
CHECK(args.size() == 2) << "op requires exactly 2 arguments, got " << args.size();
```

### Internal invariant checks — `INTERNAL_CHECK_SPAN`

Throw `InternalError` when an internal invariant is violated. Always attach the IR node's `Span` so the error message includes the user's source location:

```cpp
INTERNAL_CHECK_SPAN(op->var_, op->span_) << "AssignStmt has null var";
INTERNAL_CHECK_SPAN(new_value, op->span_) << "AssignStmt value mutated to null";
```

When the check fails, the error message includes both the IR source location and the C++ location:

```text
AssignStmt has null var
  Source location: user_model.py:42:1
Check failed: op->var_ at src/ir/transforms/mutator.cpp:301
```

There is also `INTERNAL_UNREACHABLE_SPAN` for code paths that should never be reached:

```cpp
INTERNAL_UNREACHABLE_SPAN(span) << "Unknown binary expression kind";
```

### Variants without span

`INTERNAL_CHECK` and `INTERNAL_UNREACHABLE` do not carry IR source location. They are appropriate when no `Span` is available (e.g., in non-IR contexts like arithmetic utilities or registry lookups). When an IR node is being processed and `op->span_` is accessible, prefer the `_SPAN` variants.

### Unreachable code paths — `UNREACHABLE`

Throw `ValueError` for code paths that should be unreachable from a user perspective:

```cpp
UNREACHABLE << "Unsupported data type: " << dtype;
```

### Macro Reference

| Macro | Exception Type | Span | Status |
| ----- | -------------- | ---- | ------ |
| `CHECK(expr)` | `ValueError` | No | Active |
| `UNREACHABLE` | `ValueError` | No | Active |
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
2. **Internal assertion checks** — `INTERNAL_CHECK_SPAN` / `INTERNAL_UNREACHABLE_SPAN` embed `span.to_string()` into the `InternalError` message

When a `Span` is valid, the error output includes a `Source location:` line pointing to the user's code. When `Span::unknown()` is used, no source location line appears.

## Python API

```python
import pypto

# User-facing check (raises ValueError)
pypto.check(condition, "error message")

# Internal invariant check with span (raises RuntimeError)
pypto.internal_check_span(condition, "error message", span)

# Internal invariant check without span (deprecated)
pypto.internal_check(condition, "error message")
```

## Migration Guide

When writing new code in IR transforms, passes, or codegen that uses `INTERNAL_CHECK`:

1. Identify the current IR node being processed (`op`, `stmt`, `expr`, etc.)
2. Replace `INTERNAL_CHECK(expr)` with `INTERNAL_CHECK_SPAN(expr, op->span_)`
3. Replace `INTERNAL_UNREACHABLE` with `INTERNAL_UNREACHABLE_SPAN(op->span_)`
4. If a `Span` is available as a function parameter (e.g., in `Reconstruct*` helpers), use that directly

```cpp
// Before (deprecated — triggers compiler warning):
INTERNAL_CHECK(op->body_) << "ForStmt has null body";

// After:
INTERNAL_CHECK_SPAN(op->body_, op->span_) << "ForStmt has null body";
```

## Related

- [IR Overview — Source Location Tracking](ir/00-overview.md)
- [IR Verifier — Diagnostic System](passes/99-verifier.md)
- `include/pypto/core/error.h` — Exception classes and `Diagnostic`
- `include/pypto/core/logging.h` — Assertion macros and `FatalLogger`
- `include/pypto/ir/span.h` — `Span` class
