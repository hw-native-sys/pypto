# Error Checking Conventions

## Overview

PyPTO uses two distinct macros: `CHECK` for user errors, `INTERNAL_CHECK` for internal bugs.

## CHECK - User Input Validation

**Use for validating user-provided input or external conditions.**

**When to use:**

- Function arguments passed by users
- User-provided data (tensor dimensions, indices)
- External configuration or file input
- API contract enforcement
- Conditions that may fail due to user mistakes

**Behavior:** Raises `pypto::ValueError` with helpful error message

**Example:**

```cpp
void SetTensorShape(const std::vector<int>& shape) {
  CHECK(!shape.empty()) << "Tensor shape cannot be empty";
  for (size_t i = 0; i < shape.size(); ++i) {
    CHECK(shape[i] > 0) << "Tensor dimension " << i
                        << " must be positive, got " << shape[i];
  }
}
```

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
  INTERNAL_CHECK(node->GetRefCount() > 0)
    << "Internal error: invalid reference count";
  // Transform logic...
}
```

**Error messages should:**

- Be technical (for developers debugging PyPTO)
- Include context for debugging
- Mark as "Internal error" to indicate bug

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
├─ YES → Use CHECK (raises pypto::ValueError)
│
└─ NO → Could this fail due to PyPTO bug?
    └─ YES → Use INTERNAL_CHECK
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

| - | CHECK | INTERNAL_CHECK |
| - | ----- | -------------- |
| **For** | User errors | Internal bugs |
| **Raises** | `pypto::ValueError` | Internal error |
| **Message** | User-friendly | Technical |

**Remember:** `CHECK` = user error, `INTERNAL_CHECK` = PyPTO bug. Always use PyPTO exceptions!
