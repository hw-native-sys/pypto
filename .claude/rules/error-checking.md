# Error Checking Conventions

## Overview

PyPTO uses two distinct macros: `CHECK` for user errors, `INTERNAL_CHECK` for internal bugs. When operating on IR and a `Span` is in scope, prefer the `_SPAN` variants (`CHECK_SPAN`, `INTERNAL_CHECK_SPAN`, `UNREACHABLE_SPAN`, `INTERNAL_UNREACHABLE_SPAN`) — they auto-emit IR source location on failure and dramatically improve debuggability for both users and developers.

**Inside passes**: passes operate on IR that earlier passes have already verified. A failed invariant inside a pass therefore almost always indicates a **compiler bug** — use `INTERNAL_CHECK_SPAN`, not `CHECK`. Reserve `CHECK` / `CHECK_SPAN` for documented user-facing limitations (e.g. "feature X not yet supported — use Y").

## CHECK / CHECK_SPAN - User Input Validation

**Use for validating user-provided input or external conditions.**

**When to use:**

- Function arguments passed by users
- User-provided data (tensor dimensions, indices)
- External configuration or file input (`.pto` deserialization)
- API contract enforcement
- Documented user-facing limitations inside passes (e.g. "feature X not yet lowered — use Y")

**Behavior:** Raises `pypto::ValueError` with helpful error message. `CHECK_SPAN` additionally embeds the IR source location.

**Example:**

```cpp
void SetTensorShape(const std::vector<int>& shape) {
  CHECK(!shape.empty()) << "Tensor shape cannot be empty";
  for (size_t i = 0; i < shape.size(); ++i) {
    CHECK(shape[i] > 0) << "Tensor dimension " << i
                        << " must be positive, got " << shape[i];
  }
}

// Inside an op-conversion lambda where `span` is a parameter:
CHECK_SPAN(input_tile->shape_.size() == 2, span)
    << "scatter_update: only 2D input is currently supported, got rank "
    << input_tile->shape_.size();
```

`CHECK_SPAN` follows the same span-safety rule as `INTERNAL_CHECK_SPAN` (see below): the span expression must be safe to evaluate when the check fails.

**Error messages should:**

- Be clear and actionable
- Include context (show invalid value and expected value)
- Help users fix their code

## INTERNAL_CHECK - Internal Invariant Verification

**Use for internal consistency checks and invariants.**

**When to use:**

- Verifying internal invariants and postconditions
- Double-checking algorithm correctness
- Validating internal state consistency
- Preconditions guaranteed by earlier checks
- Conditions that can only fail due to implementation bugs

**Example:**

```cpp
void InternalTransform(IRNode* node) {
  INTERNAL_CHECK(node != nullptr) << "Internal error: node should not be null";
  INTERNAL_CHECK_SPAN(node->GetRefCount() > 0, node->span_)
    << "Internal error: invalid reference count";
  // Transform logic...
}
```

**Error messages should:**

- Be technical (for developers debugging PyPTO)
- Include context for debugging
- Mark as "Internal error" to indicate bug

### Prefer `INTERNAL_CHECK_SPAN` when an IR `Span` is in scope

`INTERNAL_CHECK_SPAN(expr, span)` and `INTERNAL_UNREACHABLE_SPAN(span)` attach the IR source location to the failure message. Use them whenever a `Span` is reachable — typically `op->span_`, `node->span_`, `expr->span_`, or `stmt->span_`. See `docs/en/dev/02-error-handling.md` for the full reference.

**Safety rule:** the `span` argument is evaluated only on failure, but it is evaluated unconditionally there. So **the span expression must be safe to evaluate exactly when the check fails.**

```cpp
// ❌ UNSAFE — if `func` is null, `RequiresX(func)` short-circuits to false,
// then `func->span_` dereferences null in the failure path.
INTERNAL_CHECK_SPAN(RequiresX(func), func->span_) << "...";

// ❌ UNSAFE — guarding the very pointer whose span we read.
INTERNAL_CHECK_SPAN(node, node->span_) << "...";

// ✅ Plain INTERNAL_CHECK when the predicate is the null guard.
INTERNAL_CHECK(node) << "Internal error: node must not be null";

// ✅ SPAN form when the span source is a *different* IR node guaranteed
// non-null at the failure point (often a sibling parameter or a parent).
INTERNAL_CHECK_SPAN(child_value, parent->span_) << "...";
```

**Quick decision:**

| Situation | Use |
| --------- | --- |
| Check guards the same pointer whose `->span_` you'd read | `INTERNAL_CHECK` |
| Predicate may short-circuit before dereferencing the span source | `INTERNAL_CHECK` |
| Span source is a separate IR node known non-null at the failure point | `INTERNAL_CHECK_SPAN` |
| No IR node in scope (utility / arithmetic helper / non-IR code) | `INTERNAL_CHECK` |

## PyPTO Exception Types

**IMPORTANT: Always use PyPTO exceptions, not native C++ exceptions.**

```cpp
// ✅ Use PyPTO exceptions
pypto::ValueError      // Invalid values (used by CHECK)
pypto::TypeError       // Type mismatches
pypto::RuntimeError    // Runtime issues
pypto::IndexError      // Index out of bounds

// ❌ Don't use native C++ exceptions
std::runtime_error     // DON'T USE
std::invalid_argument  // DON'T USE
std::logic_error       // DON'T USE
```

**Why?** PyPTO exceptions work properly across C++/Python boundary with better error messages.

**Manual exception throwing:**

```cpp
// ✅ Good
if (complex_validation_fails) {
  throw pypto::ValueError("Detailed explanation");
}

// ❌ Bad
if (complex_validation_fails) {
  throw std::runtime_error("Error");  // DON'T
}
```

## Decision Guide

```text
Could this fail due to user error?
├─ YES → Use CHECK / CHECK_SPAN (raises pypto::ValueError)
│        ↳ Inside a pass? Almost always NO — earlier passes verified the IR.
│          Only YES when surfacing a documented limitation as a user error.
│
└─ NO → Could this fail due to PyPTO bug?
    └─ YES → Use INTERNAL_CHECK / INTERNAL_CHECK_SPAN
```

## Common Patterns

**Public API:**

```cpp
ObjectRef CreateObject(const std::string& name, int value) {
  CHECK(!name.empty()) << "Object name cannot be empty";
  CHECK(value >= 0) << "Value must be non-negative, got " << value;
  auto obj = InternalCreate(name, value);
  INTERNAL_CHECK(obj.defined()) << "Internal error: failed to create";
  return obj;
}
```

**Internal functions:**

```cpp
void InternalHelper(IRNode* node) {
  INTERNAL_CHECK(node != nullptr);
  INTERNAL_CHECK(node->IsValid());
}
```

## Error Message Guidelines

**CHECK (user-facing):** Clear, actionable, include context

```cpp
// ✅ Good
CHECK(dim > 0) << "Tensor dimension must be positive, got " << dim;
```

**INTERNAL_CHECK (developer-facing):** Technical, mark as "Internal error"

```cpp
// ✅ Good
INTERNAL_CHECK(ref_count_ > 0) << "Internal error: ref count is " << ref_count_;
```

## Best Practices

1. Validate early with `CHECK` at API boundaries
2. Use `INTERNAL_CHECK` for guaranteed preconditions
3. Always use PyPTO exceptions, not C++ exceptions
4. Provide helpful error messages with context

## Summary

| - | CHECK / CHECK_SPAN | INTERNAL_CHECK / INTERNAL_CHECK_SPAN |
| - | ------------------ | ------------------------------------ |
| **For** | User errors / documented user-facing limitations | Internal bugs (pass invariants, postconditions) |
| **Raises** | `pypto::ValueError` | `pypto::InternalError` |
| **Message** | User-friendly | Technical, mark as "Internal error" |
| **Inside passes** | Rare — only for surfaced limitations | Default |

**Remember:** `CHECK` = user error, `INTERNAL_CHECK` = PyPTO bug. Prefer the `_SPAN` variants whenever an IR `Span` is reachable. Always use PyPTO exceptions!
