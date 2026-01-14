# Operator Implementation Organization

This document describes the organization of operator implementations in the PyPTO codebase.

## Overview

Operator implementations are organized into separate source files under `src/ir/op/`, categorized by operator type and functionality.

## File Structure

```
src/ir/op/
├── README.md                    # Documentation
├── type_inference.cpp           # Type inference utilities implementation
├── tensor_ops/                  # Tensor operator implementations
│   └── elementwise.cpp          # Element-wise operations
└── tile_ops/                    # Tile operator implementations
    └── elementwise.cpp          # Element-wise operations
```

## Why This Organization?

### Previous Structure (✗)
```
include/pypto/ir/op_traits.h     # All operator traits in one header
src/ir/op_registry.cpp           # Registry + all operator implementations
```

**Problems:**
- All operator implementations in one or two large files
- Changes to any operator triggered recompilation of many files
- Difficult to navigate as the number of operators grows
- No clear separation by operator category

### New Structure (✓)
```
src/ir/op/
├── type_inference.cpp           # Shared type inference utilities
├── tensor_ops/elementwise.cpp   # Tensor elementwise ops
└── tile_ops/elementwise.cpp     # Tile elementwise ops
```

**Benefits:**
1. **Modularity**: Each operator category is self-contained
2. **Build Performance**: Changes to one category don't rebuild others
3. **Maintainability**: Easy to find and modify specific operators
4. **Scalability**: Adding new operators is straightforward
5. **Automatic Registration**: Uses static initialization via `REGISTER_OP` macro

## Design Patterns

### 1. Category-Based Organization

Operators are grouped into category files based on their functionality:

```cpp
// src/ir/op/tensor_ops/elementwise.cpp

// Helper function for common type deduction logic
TypePtr DeduceTensorOpElementwiseBinaryType(const std::vector<ExprPtr>& args, const std::string& op_name) {
  // Validate arguments, promote types, broadcast shapes
  // ...
}

// Register operators using fluent API
REGISTER_OP("tensor.add")
    .set_op_category("TensorOp")
    .set_description("Element-wise addition of two tensors with broadcasting")
    .add_argument("lhs", "Left-hand side tensor (TensorType)")
    .add_argument("rhs", "Right-hand side tensor (TensorType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args) {
      return DeduceTensorOpElementwiseBinaryType(args, "tensor.add");
    });

REGISTER_OP("tensor.sub")
    .set_op_category("TensorOp")
    .set_description("Element-wise subtraction of two tensors with broadcasting")
    .add_argument("lhs", "Left-hand side tensor (TensorType)")
    .add_argument("rhs", "Right-hand side tensor (TensorType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args) {
      return DeduceTensorOpElementwiseBinaryType(args, "tensor.sub");
    });
// ...
```

### 2. Static Initialization Pattern

The `REGISTER_OP` macro uses static initialization to automatically register operators when the shared library is loaded:

```cpp
// In op_registry.h
#define REGISTER_OP(OpName)                                                                           \
  static PYPTO_STR_CONCAT(PYPTO_UNUSED ::pypto::ir::OpRegistryEntry& OpRegistryEntry_, __COUNTER__) = \
      ::pypto::ir::OpRegistry::GetInstance().Register(OpName)
```

This eliminates the need for manual registration function calls - operators are registered automatically before `main()` runs.

## Comparison with Other Projects

### TVM/Relax
```
src/relax/op/
├── tensor/
│   ├── binary.cc
│   ├── create.cc
│   └── manipulate.cc
└── nn/
    ├── convolution.cc
    └── pooling.cc
```

### PyTorch
```
aten/src/ATen/native/
├── BinaryOps.cpp
├── ReduceOps.cpp
└── TensorShape.cpp
```

Our structure follows similar principles with clearer categorization for our specific needs (tensor vs tile operations).

## Future Extensions

As more operators are added, new category files can be created:

```
src/ir/op/
├── tensor_ops/
│   ├── elementwise.cpp         # ✓ Exists
│   ├── reduction.cpp           # TODO: Sum, Max, Min, etc.
│   ├── matmul.cpp              # TODO: Matrix multiplication
│   └── transform.cpp           # TODO: Reshape, Transpose, etc.
└── tile_ops/
    ├── elementwise.cpp         # ✓ Exists
    ├── matmul.cpp              # TODO: Tile matmul for hardware
    └── load_store.cpp          # TODO: Tile load/store operations
```

## Adding New Operators

See the [README in src/ir/op/](../../src/ir/op/README.md) for step-by-step instructions on adding new operators.

## Build System Integration

The CMakeLists.txt includes all operator implementation files:

```cmake
set(PYPTO_SOURCES
    # ... other sources ...
    src/ir/op_registry.cpp
    src/ir/op/type_inference.cpp
    src/ir/op/tensor_ops/elementwise.cpp
    src/ir/op/tile_ops/elementwise.cpp
    # Add new category files here
)
```

## Testing

All operators continue to be tested through the same test suite in `tests/ut/ir/test_op_registry.py`. The organization change is transparent to tests.

## Summary

This reorganization:
- ✅ Improves code organization and maintainability
- ✅ Reduces compilation dependencies
- ✅ Makes it easier to add new operators
- ✅ Maintains all existing functionality
- ✅ Passes all 307 IR tests
