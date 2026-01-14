# Error Checking Conventions

## Overview

PyPTO uses two distinct macros for error checking: `CHECK` and `INTERNAL_CHECK`. Understanding when to use each is crucial for proper error handling and code maintainability.

## Core Principles

### `CHECK` - User Input Validation

**Use `CHECK` for validating user-provided input or external conditions.**

`CHECK` is designed to catch errors that may occur due to incorrect user input, invalid API usage, or external conditions that the user can control or fix.

**Behavior:**
- When the condition fails, `CHECK` raises a `pypto::ValueError`
- This allows proper exception handling in both C++ and Python layers
- The error message is propagated to the user with full context

**When to use `CHECK`:**
- Validating function arguments passed by users
- Checking user-provided data (e.g., tensor dimensions, indices)
- Validating external configuration or file input
- Enforcing API contracts that users must follow
- Conditions that may fail due to user mistakes

**Example:**

```cpp
void SetTensorShape(const std::vector<int>& shape) {
  // User might provide empty or negative dimensions
  CHECK(!shape.empty()) << "Tensor shape cannot be empty";
  for (size_t i = 0; i < shape.size(); ++i) {
    CHECK(shape[i] > 0) << "Tensor dimension " << i
                        << " must be positive, got " << shape[i];
  }
}
```

**Characteristics:**
- Raises `pypto::ValueError` when condition fails
- Provides user-friendly error messages
- Indicates user error, not a bug in the library
- Should be triggered by incorrect usage, not internal bugs
- Error messages should help users fix their code

### `INTERNAL_CHECK` - Internal Invariant Verification

**Use `INTERNAL_CHECK` for internal consistency checks and invariants.**

`INTERNAL_CHECK` is similar to `assert` - it verifies conditions that should always be true if the library implementation is correct. An `INTERNAL_CHECK` failure indicates a bug in PyPTO itself, not user error.

**When to use `INTERNAL_CHECK`:**
- Verifying internal invariants and postconditions
- Double-checking algorithm correctness
- Validating internal state consistency
- Preconditions that should be guaranteed by earlier checks
- Conditions that can only fail due to implementation bugs

**Example:**

```cpp
void InternalTransform(IRNode* node) {
  // This should have been validated by caller
  INTERNAL_CHECK(node != nullptr) << "Internal error: node should not be null";

  // This is an internal invariant - if it fails, there's a bug
  INTERNAL_CHECK(node->GetRefCount() > 0)
    << "Internal error: node has invalid reference count";

  // Transform logic...
}
```

**Characteristics:**
- Indicates internal bugs in PyPTO implementation
- Used for debugging and development
- May be optimized out in release builds (like `assert`)
- Error messages can be technical/developer-focused

## Exception Handling Policy

**IMPORTANT: Always prefer PyPTO error types over native C++ exceptions.**

- **DO**: Throw `pypto::ValueError`, `pypto::TypeError`, `pypto::RuntimeError`, etc.
- **DON'T**: Throw `std::runtime_error`, `std::invalid_argument`, `std::logic_error`, etc.
- **WHY**: PyPTO errors are properly handled across the C++/Python boundary with better error messages and stack traces

**Exception hierarchy:**
```cpp
// Prefer these PyPTO exceptions
pypto::ValueError      // For invalid values (used by CHECK)
pypto::TypeError       // For type mismatches
pypto::RuntimeError    // For runtime issues
pypto::IndexError      // For index out of bounds

// Avoid these C++ exceptions
std::runtime_error     // DON'T USE
std::invalid_argument  // DON'T USE
std::logic_error       // DON'T USE
```

**When to throw manually:**
If you need to throw an exception outside of `CHECK`/`INTERNAL_CHECK`, always use PyPTO exceptions:

```cpp
// Good - PyPTO exception
if (some_complex_validation()) {
  throw pypto::ValueError("Detailed explanation of what went wrong");
}

// Bad - Native C++ exception
if (some_complex_validation()) {
  throw std::runtime_error("Error");  // DON'T DO THIS
}
```

## Decision Flowchart

When adding a check, ask yourself:

```
Could this condition fail due to user error?
├─ YES → Use CHECK (raises pypto::ValueError)
│   └─ Example: Invalid user input, wrong API usage
│
└─ NO → Could this fail due to a bug in PyPTO?
    ├─ YES → Use INTERNAL_CHECK
    │   └─ Example: Internal invariant violation
    │
    └─ Should never fail → Use INTERNAL_CHECK for safety
        └─ Example: Postcondition that should be guaranteed
```

## Common Patterns

### Pattern 1: Public API Functions

```cpp
// Public API - use CHECK for user input
ObjectRef CreateObject(const std::string& name, int value) {
  CHECK(!name.empty()) << "Object name cannot be empty";
  CHECK(value >= 0) << "Value must be non-negative, got " << value;

  auto obj = InternalCreate(name, value);

  // Use INTERNAL_CHECK for internal invariants
  INTERNAL_CHECK(obj.defined()) << "Internal error: failed to create object";
  return obj;
}
```

### Pattern 2: Internal Helper Functions

```cpp
// Internal function - can use INTERNAL_CHECK if preconditions are guaranteed
void InternalHelper(IRNode* node) {
  // Caller should have validated this
  INTERNAL_CHECK(node != nullptr);
  INTERNAL_CHECK(node->IsValid());

  // Process node...
}
```

### Pattern 3: Mixed Validation

```cpp
void ProcessTensor(const Tensor& tensor, int index) {
  // CHECK for user-provided parameters (raises pypto::ValueError)
  CHECK(index >= 0) << "Index must be non-negative, got " << index;
  CHECK(index < tensor.size()) << "Index " << index
                                << " out of bounds for tensor of size "
                                << tensor.size();

  auto element = tensor[index];

  // INTERNAL_CHECK for internal invariants
  INTERNAL_CHECK(element.IsValid())
    << "Internal error: tensor contains invalid element at index " << index;
}
```

### Pattern 4: Manual Exception Throwing

```cpp
// When CHECK is not sufficient, manually throw PyPTO exceptions
void ComplexValidation(const Config& config) {
  if (config.mode == "advanced") {
    // Complex validation that needs custom logic
    if (!IsValidAdvancedConfig(config)) {
      // Good - use PyPTO exception
      throw pypto::ValueError(
        "Advanced mode requires specific configuration: " +
        config.GetErrorDetails());
    }
  }

  // DON'T do this:
  // throw std::runtime_error("Invalid config");  // BAD!
}
```

## Error Messages

### For `CHECK` (User Errors)

- **Be clear and actionable**: Tell users what went wrong and how to fix it
- **Include context**: Show the invalid value and what was expected
- **Be polite**: Assume the user made an honest mistake

```cpp
// Good
CHECK(dim > 0) << "Tensor dimension must be positive, got " << dim
               << ". Please provide a positive dimension value.";

// Bad
CHECK(dim > 0) << "Invalid dimension";
```

### For `INTERNAL_CHECK` (Internal Errors)

- **Be technical**: These are for developers debugging PyPTO
- **Include context**: Help developers understand what went wrong
- **Mark as internal**: Make it clear this is a bug in PyPTO

```cpp
// Good
INTERNAL_CHECK(ref_count_ > 0)
  << "Internal error: reference count is " << ref_count_
  << ", should be positive. This indicates a bug in reference counting logic.";

// Bad
INTERNAL_CHECK(ref_count_ > 0) << "Bad ref count";
```

## Best Practices

1. **Validate early**: Use `CHECK` at API boundaries to catch user errors early
2. **Trust internal calls**: Use `INTERNAL_CHECK` for conditions that should be guaranteed by design
3. **Don't over-check**: Avoid redundant checks that hurt performance
4. **Provide context**: Always include helpful error messages
5. **Be consistent**: Follow these conventions throughout the codebase
6. **Use PyPTO exceptions**: Always prefer `pypto::ValueError` and other PyPTO exceptions over native C++ exceptions
7. **Prefer macros**: Use `CHECK` and `INTERNAL_CHECK` macros when possible; manually throw PyPTO exceptions only for complex validation logic

## Summary

| Aspect | `CHECK` | `INTERNAL_CHECK` |
|--------|---------|----------|
| **Purpose** | User input validation | Internal invariant verification |
| **Audience** | Library users | Library developers |
| **Indicates** | User error | Implementation bug |
| **When fails** | Invalid usage | Internal bug |
| **Exception** | Raises `pypto::ValueError` | Internal error (may abort) |
| **Error message** | User-friendly, actionable | Technical, debug-focused |
| **Like** | Throwing exceptions | `assert()` macro |

## Remember

- **`CHECK`** = "The user did something wrong" → Raises `pypto::ValueError`
- **`INTERNAL_CHECK`** = "We (developers) did something wrong" → Internal bug
- **Always use PyPTO exceptions**, never native C++ exceptions (`std::runtime_error`, etc.)

When in doubt, prefer `CHECK` for public APIs and `INTERNAL_CHECK` for internal logic.
