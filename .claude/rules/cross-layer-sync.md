# Cross-Layer Synchronization

## Overview

PyPTO has three layers that must stay synchronized:
1. **C++ Implementation** - `include/pypto/` and `src/`
2. **Python Bindings** - `python/bindings/` (nanobind)
3. **Type Stubs** - `python/pypto/pypto_core/__init__.pyi`

**All three must be updated together when changing public APIs.**

## Example: Adding a Method

**1. C++ Header** (`include/pypto/ir/expr.h`):
```cpp
class TensorExpr : public Expr {
 public:
  bool IsScalar() const;  // New method
};
```

**2. Python Binding** (`python/bindings/ir_binding.cpp`):
```cpp
nb::class_<TensorExpr>(m, "TensorExpr")
    .def("is_scalar", &TensorExpr::IsScalar, "Check if tensor is scalar");
```

**3. Type Stub** (`python/pypto/pypto_core/__init__.pyi`):
```python
class TensorExpr(Expr):
    def is_scalar(self) -> bool:
        """Check if tensor is scalar."""
        ...
```

## Workflow

1. Modify C++ header (`include/pypto/`)
2. Implement in C++ source (`src/`)
3. Update Python binding (`python/bindings/`)
4. Update type stub (`python/pypto/pypto_core/__init__.pyi`)
5. Build and test all layers work together

## Naming Conventions

| C++ | Python Binding | Example |
|-----|----------------|---------|
| `GetValue()` | `get_value()` | Use snake_case |
| `SetRank()` | `set_rank()` | Convert methods |
| `TensorExpr` | `TensorExpr` | Keep class names |

**Always use Python snake_case for method names in bindings!**

## Common Pitfalls

**❌ Forgetting type stub** → No IDE autocomplete
```python
# Always update all three layers together
```

**❌ Inconsistent signatures** → Type checker fails
```python
# Double-check return types match across layers
def is_valid(self) -> bool: ...  # Must match C++ bool
```

**❌ Wrong naming** → Inconsistent API
```cpp
.def("GetValue", &Class::GetValue)  // Wrong! Use snake_case
.def("get_value", &Class::GetValue) // Correct!
```

**❌ Missing docstrings** → Poor developer experience
```python
def method(self, arg: Type) -> Result:
    """Always add helpful docstrings."""
    ...
```

## Verification Checklist

After cross-layer changes:
- [ ] C++ header and implementation complete
- [ ] Python binding exposes API with snake_case names
- [ ] Type stub matches binding signature
- [ ] Type hints are accurate
- [ ] Docstrings present in binding and stub
- [ ] Project builds (`cmake --build build`)
- [ ] Python can import and use the API
- [ ] Type checker passes (mypy/pyright)

## Quick Test

```python
from pypto import ir

# Create and use
expr = ir.TensorExpr()
rank = expr.get_rank()

# Verify types
assert isinstance(rank, int)
```

## Remember

**When you touch one layer, update all three.**

This ensures IDE autocomplete, type checking, and a consistent developer experience.
