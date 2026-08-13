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
  └── Error                  (base: auto stack trace capture, → Python pypto.Error)
        ├── ValueError       (→ Python ValueError)
        ├── TypeError        (→ Python TypeError)
        ├── RuntimeError     (→ Python RuntimeError)
        ├── NotImplementedError
        ├── IndexError
        ├── AssertionError
        ├── InternalError    (→ Python RuntimeError — internal bugs)
        └── VerificationError (carries vector<Diagnostic>, → Python pypto.Error)
```

Subclasses without a dedicated translation fall back to `pypto.Error` — a real Python type, not a
bare `Exception`, so `VerificationError` stays catchable by type. `pypto.Error` derives from
`Exception`, so `except Exception` keeps working. Tests must assert the concrete type rather than
`Exception`; `tests/lint/check_no_broad_raises.py` enforces this.

**The Python mirror is flat, not nested.** Each row above maps to an *independent* Python type, so
the C++ inheritance shown in the tree does not carry over: `pypto.InternalError` derives from
Python's `RuntimeError`, not from `pypto.Error`. `except pypto.Error` therefore catches
`VerificationError` but **not** `InternalError`. Catch `Exception` if you need both.

`Error::GetFullMessage()` returns the error message plus a formatted C++ stack trace.

### When the stack trace is shown

Every `Error` captures a trace at construction, but the exception translator
(`python/bindings/modules/error.cpp`) only attaches it to the Python message when it helps:

| Exception | Raised by | Trace in the Python message |
| --------- | --------- | --------------------------- |
| `InternalError`, `AssertionError` | `INTERNAL_CHECK` family | Always — a failed invariant is a PyPTO bug, and the frames are the primary artefact |
| `ValueError`, `TypeError`, `RuntimeError`, `IndexError`, `NotImplementedError`, `VerificationError` | `CHECK` family, user input | Only under `PTO_BACKTRACE=1` |

A user error already carries its own DSL source snippet; the C++ frames name PyPTO internals the
caller cannot act on, and printing them in between pushes the snippet further down. That includes
`NotImplementedError`: an unlowered feature is a documented limitation surfaced to the user, the
same category `CHECK` covers, not a failed invariant. `PTO_BACKTRACE=1` is the same switch the DSL
diagnostics advertise, so one variable turns on both backtraces.

```bash
PTO_BACKTRACE=1 python my_kernel.py   # C++ frames on every error, not just internal ones
```

Only the exact value `1` enables C++ traces; any other value leaves them off. The DSL parser is
stricter — it rejects anything other than `0` or `1` — so stick to those two values.

`Backtrace::FormatStackTrace` drops frames belonging to infrastructure rather than to the call
path — `libbacktrace`, `nanobind`, libc, the C++ standard library, and the `error.h` / `logging.h`
throw sites (`kFileNameFilter` in `src/core/backtrace.cpp`).

### Platform support for stack traces

`3rdparty/libbacktrace` tracks upstream [ianlancetaylor/libbacktrace](https://github.com/ianlancetaylor/libbacktrace).
Upstream's Mach-O reader accepts only `MH_EXECUTE`, `MH_DYLIB` and `MH_DSYM`, but a CPython
extension module is an `MH_BUNDLE` — so on **macOS** symbolization fails and `GetFullMessage()`
falls back to `No stack trace available`. No build mode changes this; it is a filetype gate, not
missing debug info. Linux (ELF) is unaffected, but still needs debug info: `Debug` or
`RelWithDebInfo` (the default) pass `-g` and produce full traces, while a plain `Release` build is
`-O2 -DNDEBUG` with no `-g`, so frames carry no source location and the same fallback appears.

`Backtrace::ErrorCallback` reports each distinct `(message, errno)` pair only once. Without that,
macOS would emit one `no debug info in Mach-O executable` line per stack frame of every captured
trace, since the dyld init path succeeds overall and installs `macho_nodebug` as the fileline
handler.

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

### Span attribution in passes

A pass that synthesizes or re-creates an IR node must give it the span of the
**node it stands for**, not the enclosing function's span. Reaching for
`func->span_` is convenient — it is in scope for the whole transform — but it
reports the `def` line for every node the pass touches, which silently degrades
each consumer that reads a span off a `Call`: `CHECK_SPAN` / `INTERNAL_CHECK_SPAN`
diagnostics raised by later passes, IR-trace reports, and span-keyed verifier
checks (the PH001 perf hint deduplicates by source site, so coarsened spans merge
unrelated transfers into one bogus "N occurrences" hint).

```cpp
// ❌ every synthesized op reports the `def` line
const auto& span = func->span_;
for (const auto& stmt : body) { /* ... */ op_registry.Create(name, args, span); }

// ✅ attribute each node to the statement being rewritten
for (const auto& stmt : body) {
  const Span& span = stmt->span_;
  /* ... */ op_registry.Create(name, args, span);
}
```

Pick the nearest node that motivated the new one: the statement being rewritten,
the `Call` being converted (`call->span_`), the parameter a prologue load reads
(`var->span_`), or the `ReturnStmt` an epilogue store serves. `func->span_` is
correct only for nodes that genuinely belong to the whole function — the rebuilt
`Function` itself and its body `SeqStmts`.

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
