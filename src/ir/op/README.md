# Operator Implementations

This directory contains the implementations for all PyPTO IR operators, organized by category.

## Directory Structure

```
src/ir/op/
├── README.md                    # This file
├── type_inference.cpp           # Type inference utilities implementation
├── tensor_ops/                  # Tensor operator implementations
│   └── elementwise.cpp          # Element-wise ops (Add, Sub, Mul, Div)
└── tile_ops/                    # Tile operator implementations
    └── elementwise.cpp          # Element-wise ops (Add, Sub, Mul, Div)
```

## Organization Principles

### By Operation Type
- `tensor_ops/` - Operations on N-dimensional tensors
- `tile_ops/` - Operations on 2D tiles (at most 2 dimensions)

### By Operation Category (within each type)
- `elementwise.cpp` - Element-wise binary operations (Add, Sub, Mul, Div)
- `reduction.cpp` - Reduction operations (Sum, Max, Min, etc.) - *to be added*
- `matmul.cpp` - Matrix multiplication operations - *to be added*
- `transform.cpp` - Shape transformation operations (Reshape, Transpose, etc.) - *to be added*

## Adding a New Operator

### 1. Choose or create a category file
Select the appropriate category file under `tensor_ops/` or `tile_ops/`, or create a new one:
- Element-wise ops: `elementwise.cpp`
- Matrix ops: `matmul.cpp` (create if needed)
- Reduction ops: `reduction.cpp` (create if needed)

### 2. Register the operator using the fluent API
Add the operator registration to the category file:

```cpp
// Example: in tensor_ops/matmul.cpp (new file)
#include "pypto/core/logging.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/type.h"
#include "pypto/ir/type_inference.h"

namespace pypto {
namespace ir {

// Helper function for type deduction (optional, for code reuse)
TypePtr DeduceTensorMatMulType(const std::vector<ExprPtr>& args, const std::string& op_name) {
  CHECK(args.size() == 2) << op_name << " requires exactly 2 arguments";

  auto tensor1 = std::dynamic_pointer_cast<const TensorType>(args[0]->GetType());
  auto tensor2 = std::dynamic_pointer_cast<const TensorType>(args[1]->GetType());
  CHECK(tensor1 && tensor2) << op_name << " requires TensorType arguments";

  // Matrix multiplication type inference logic
  // ...

  return result_type;
}

// Register the operator
REGISTER_OP("tensor.matmul")
    .set_op_category("TensorOp")
    .set_description("Matrix multiplication of two tensors")
    .add_argument("lhs", "Left-hand side tensor")
    .add_argument("rhs", "Right-hand side tensor")
    .f_deduce_type([](const std::vector<ExprPtr>& args) {
      return DeduceTensorMatMulType(args, "tensor.matmul");
    });

}  // namespace ir
}  // namespace pypto
```

The `REGISTER_OP` macro uses static initialization, so the operator is automatically registered when the library loads. No manual registration function calls are needed.

### 3. Update CMakeLists.txt
Add the new source file if you created one:

```cmake
set(PYPTO_SOURCES
    # ... existing files ...
    src/ir/op/tensor_ops/matmul.cpp  # Add this
    # ... rest of files ...
)
```

### 4. Write tests
Add tests in `tests/ut/ir/test_op_registry.py` to verify the operator works correctly.

## Benefits of This Structure

1. **Modularity**: Each operator category is in its own file
2. **Maintainability**: Easy to find and modify specific operator implementations
3. **Scalability**: Adding new operators doesn't bloat existing files
4. **Build Performance**: Changes to one category don't trigger recompilation of others
5. **Clear Organization**: Operators grouped by type (tensor/tile) and category (elementwise/reduction/etc.)

## Current Operators

### Tensor Operations
- **Element-wise** (`tensor_ops/elementwise.cpp`):
  - `tensor.add` - Element-wise addition with broadcasting
  - `tensor.sub` - Element-wise subtraction with broadcasting
  - `tensor.mul` - Element-wise multiplication with broadcasting
  - `tensor.div` - Element-wise division with broadcasting

### Tile Operations
- **Element-wise** (`tile_ops/elementwise.cpp`):
  - `tile.add` - Element-wise addition with 2D broadcasting
  - `tile.sub` - Element-wise subtraction with 2D broadcasting
  - `tile.mul` - Element-wise multiplication with 2D broadcasting
  - `tile.div` - Element-wise division with 2D broadcasting

## See Also

- [Operator Registration Documentation](../../../docs/dev/02-operator_registration.md)
- [Operator Organization Documentation](../../../docs/dev/03-operator_organization.md)
- [Type Inference Header](../../../include/pypto/ir/type_inference.h)
- [Type Inference Implementation](type_inference.cpp)
- [Operator Registry Header](../../../include/pypto/ir/op_registry.h)
- [Operator Registry Implementation](../op_registry.cpp)
