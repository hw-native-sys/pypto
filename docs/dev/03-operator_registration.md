# Operator Registration System

This document describes the operator registration system for PyPTO IR, which provides type-safe operator definitions with automatic type deduction.

## Overview

The operator registration system supports four kinds of operations:
- **ScalarOp**: Operations on scalar values (existing system, unchanged)
- **TensorOp**: Operations on N-dimensional tensors with broadcasting
- **TileOp**: Operations on 2D tiles (at most 2 dimensions) for hardware optimization
- **BlockOp**: Block-level operations for hardware-optimized programming with tiles and scalar broadcasting

## Key Features

1. **Fluent API registration**: Expressive operator registration with method chaining
2. **Automatic type deduction**: Result types are automatically deduced from input types
3. **Broadcasting support**: NumPy-style broadcasting for tensor/tile operations
4. **Type promotion**: Automatic data type promotion (e.g., INT32 + FP32 → FP32)
5. **Dynamic dimensions**: Support for dynamic dimensions using `kDynamicDim`

## Architecture

```
OpRegistry (Singleton)
    ├── TensorOp
    │   ├── TensorAdd
    │   ├── TensorSub
    │   ├── TensorMul
    │   └── TensorDiv
    ├── TileOp
    │   ├── TileAdd
    │   ├── TileSub
    │   ├── TileMul
    │   └── TileDiv
    └── BlockOp
        ├── BlockGetBlockIdx
        ├── BlockUbCopyIn
        ├── BlockUbCopyOut
        ├── BlockAdd
        ├── BlockMul
        ├── BlockDiv
        ├── BlockSum
        └── BlockSqrt
```

## Type System

### TensorType
N-dimensional tensor with arbitrary dimensions:
```cpp
TensorType(DataType::FP32, {dim1, dim2, dim3, ...})
```

### TileType
2D tensor with at most 2 dimensions (validated at construction):
```cpp
TileType(DataType::FP16, {dim1, dim2})  // OK
TileType(DataType::FP16, {dim1})        // OK (1D)
TileType(DataType::FP16, {d1, d2, d3})  // Error: too many dimensions
```

### Dynamic Dimensions
Use the `kDynamicDim` constant for dynamic dimensions:
```cpp
// Dynamic dimension constant (defined in pypto/core/common.h)
constexpr int64_t kDynamicDim = -1;

// Use in dimension expressions
auto dynamic_dim = make_int(kDynamicDim);
```

## C++ Usage

### Defining a New Operator

The operator registration uses a fluent API pattern where you register the operator and configure its behavior in a single chain of method calls.

**In the appropriate category file** (e.g., `src/ir/op/tensor_ops/elementwise.cpp`):
```cpp
REGISTER_OP("tensor.add")
    .set_op_category("TensorOp")
    .set_description("Element-wise addition of two tensors with broadcasting")
    .add_argument("lhs", "Left-hand side tensor (TensorType)")
    .add_argument("rhs", "Right-hand side tensor (TensorType)")
    .set_attr<std::string>("device", "cpu")
    .set_attr<int>("priority", 10)
    .set_attr<bool>("commutative", true)
    .f_deduce_type([](const std::vector<ExprPtr>& args) {
      // Validate we have exactly 2 arguments
      CHECK(args.size() == 2) << "tensor.add requires exactly 2 arguments";

      // Validate argument types
      auto tensor1 = std::dynamic_pointer_cast<const TensorType>(args[0]->GetType());
      auto tensor2 = std::dynamic_pointer_cast<const TensorType>(args[1]->GetType());
      CHECK(tensor1) << "First argument must be a TensorType";
      CHECK(tensor2) << "Second argument must be a TensorType";

      // Promote data types
      auto result_dtype = PromoteDataTypes(tensor1->dtype_, tensor2->dtype_);
      CHECK(result_dtype) << "Incompatible data types";

      // Broadcast shapes
      auto broadcast_result = BroadcastShapes(tensor1->shape_, tensor2->shape_);
      CHECK(broadcast_result.success) << "Incompatible shapes for broadcasting";

      // Return result type
      return std::make_shared<TensorType>(*result_dtype, broadcast_result.shape);
    });
```

The `REGISTER_OP` macro uses static initialization, so operators are automatically registered when the shared library is loaded. No manual registration function calls are needed.

### Using Operators

```cpp
// Create tensor variables
auto tensor_a = std::make_shared<Var>("a",
    std::make_shared<TensorType>(DataType::FP32, {make_int(4), make_int(8)}),
    span);
auto tensor_b = std::make_shared<Var>("b",
    std::make_shared<TensorType>(DataType::FP32, {make_int(8)}),
    span);

// Create operation via registry (with automatic type deduction)
auto result = OpRegistry::Get().Create("tensor.add", {tensor_a, tensor_b}, span);
// result->GetType() is TensorType(FP32, [4, 8]) due to broadcasting
```

## Python Usage

### Creating Tensors and Tiles

```python
from pypto.pypto_core import DataType, ir

span = ir.Span.unknown()

# Create a 2D tensor [4, 8]
dim4 = ir.ConstInt(4, DataType.INT32, span)
dim8 = ir.ConstInt(8, DataType.INT32, span)
tensor_type = ir.TensorType(DataType.FP32, [dim4, dim8])
tensor_var = ir.Var("t", tensor_type, span)

# Create a 2D tile [16, 16]
dim16 = ir.ConstInt(16, DataType.INT32, span)
tile_type = ir.TileType(DataType.FP16, [dim16, dim16])
tile_var = ir.Var("tile", tile_type, span)
```

### Using Operators

```python
# Create tensor variables
tensor_a = ir.Var("a", ir.TensorType(DataType.FP32, [dim4, dim8]), span)
tensor_b = ir.Var("b", ir.TensorType(DataType.FP32, [dim8]), span)

# Create tensor add operation (with automatic type deduction)
result = ir.create_op_call("tensor.add", [tensor_a, tensor_b], span)

# result.type is TensorType(FP32, [4, 8]) due to broadcasting
print(result.type.dtype)  # FP32
print(len(result.type.shape))  # 2
```

### Query Operator Registry

```python
# Check if operator is registered
assert ir.is_op_registered("tensor.add")
assert ir.is_op_registered("tile.mul")

# Get operator instance
op = ir.get_op("tensor.add")
print(op.name)  # "tensor.add"
```

### Operator Attributes

Operators can store arbitrary typed attributes for metadata like device affinity, optimization hints, operator properties, etc.

#### C++ - Setting Attributes

```cpp
REGISTER_OP("tensor.matmul")
    .set_op_category("TensorOp")
    .set_description("Matrix multiplication")
    .add_argument("lhs", "Left matrix")
    .add_argument("rhs", "Right matrix")
    .set_attr<std::string>("device", "gpu")
    .set_attr<int>("priority", 5)
    .set_attr<bool>("commutative", false)
    .f_deduce_type([](const std::vector<ExprPtr>& args) {
      // type deduction logic
    });
```

#### Python - Accessing Attributes

```python
# Get operator instance
op = ir.get_op("test.op")

# Check if attribute exists
if op.has_attr("string_attr"):
    string_attr = op.get_attr("string_attr")
    print(f"String attr: {string_attr}")  # "String attr: string_attr_value"

# Get different attribute types (automatically determined)
int_attr = op.get_attr("int_attr")  # 42
bool_attr = op.get_attr("bool_attr")  # True

# Get all attribute keys
keys = op.get_attr_keys()
print(keys)  # ['int_attr', 'string_attr', 'bool_attr']
```

#### Common Attribute Use Cases

- **Device affinity**: `set_attr<std::string>("device", "gpu")` - preferred execution device
- **Priority**: `set_attr<int>("priority", 10)` - scheduling priority for execution
- **Operator properties**: `set_attr<bool>("commutative", true)` - mathematical properties
- **Cost models**: `set_attr<int>("compute_cost", 100)` - estimated computation cost
- **Memory hints**: `set_attr<bool>("inplace", true)` - whether operation can be in-place
- **Optimization flags**: `set_attr<bool>("fused", true)` - whether operator is fused version

### Dynamic Dimensions

```python
# Use dynamic dimension constant
assert ir.DYNAMIC_DIM == -1

# Create tile with dynamic dimension
span = ir.Span.unknown()
dynamic_dim = ir.ConstInt(ir.DYNAMIC_DIM, DataType.INT32, span)
tile_type = ir.TileType(DataType.FP32, [dynamic_dim, ir.ConstInt(16, DataType.INT32, span)])
```

## Broadcasting Rules

### NumPy-style Broadcasting

Dimensions are aligned from right to left:
```
[4, 8] + [4, 8] → [4, 8]  # Exact match
[4, 8] + [8]    → [4, 8]  # Missing left dimension = 1
[4, 1] + [8]    → [4, 8]  # Size 1 broadcasts
[1, 8] + [4, 8] → [4, 8]  # Size 1 broadcasts
```

### Error Cases
```
[4, 8] + [5]     → Error: 8 ≠ 5
[4, 8] + [3, 5]  → Error: incompatible dimensions
```

## Type Promotion

Type promotion follows standard numeric rules:
- Float types take precedence over integer types
- Larger types take precedence over smaller types
- Signed types take precedence over unsigned types of the same size

Examples:
```
INT32 + INT32 → INT32
INT32 + FP32  → FP32  (float takes precedence)
INT32 + INT64 → INT64 (larger size)
UINT32 + INT32 → INT32 (signed takes precedence)
```

## Modern C++ Features

The implementation demonstrates several modern C++ (C++17) features:

1. **Fluent API**: Method chaining for expressive operator registration
2. **std::optional**: Fallible type operations
3. **std::function**: Type-erased callable for type deduction functions
4. **Lambda Expressions**: Clean inline type deduction logic
5. **Smart Pointers**: `std::shared_ptr` for memory management
6. **Static Initialization**: Automatic operator registration on library load
7. **Type Traits**: `std::dynamic_pointer_cast` for type checking
8. **constexpr**: Compile-time constants like `kDynamicDim`

## Error Handling

The system provides clear error messages:

```python
# Wrong argument count
try:
    ir.create_op_call("tensor.add", [tensor_a], span)
except Exception as e:
    print(e)  # "Operator 'tensor.add' expects 2 arguments, got 1"

# Type mismatch
try:
    ir.create_op_call("tensor.add", [scalar, tensor], span)
except Exception as e:
    print(e)  # "TensorAdd: first argument must be a TensorType, got ScalarType"

# Tile dimension constraint violation
try:
    ir.TileType(DataType.FP32, [dim1, dim2, dim3])
except Exception as e:
    print(e)  # "TileType can have at most 2 dimensions, got 3"
```

## Adding New Operations

To add a new operator (e.g., `TensorMatMul`):

1. Choose or create appropriate category file under `src/ir/op/tensor_ops/`, `src/ir/op/tile_ops/`, or `src/ir/op/block_ops/`
   - Element-wise ops: `elementwise.cpp`
   - Matrix ops: `matmul.cpp` (create if needed)
   - Reduction ops: `reduction.cpp` (create if needed)
   - Memory ops: `memory.cpp` (block_ops only)
   - Unary ops: `unary.cpp` (block_ops only)
2. Add `REGISTER_OP()` call with complete configuration:
   ```cpp
   REGISTER_OP("tensor.matmul")
       .set_op_category("TensorOp")
       .set_description("Matrix multiplication of two tensors")
       .add_argument("lhs", "Left-hand side tensor")
       .add_argument("rhs", "Right-hand side tensor")
       .f_deduce_type([](const std::vector<ExprPtr>& args) {
         // Type deduction logic here
       });
   ```
3. Add tests in `tests/ut/ir/test_op_registry.py`
4. Update `CMakeLists.txt` if adding a new operator category file

## References

- Common constants: `include/pypto/core/common.h`
- Type definitions: `include/pypto/ir/type.h`
- Operator registry: `include/pypto/ir/op_registry.h`
- Type inference utilities: `include/pypto/ir/type_inference.h`
- Type inference implementation: `src/ir/op/type_inference.cpp`
- Operator registry implementation: `src/ir/op_registry.cpp`
- Tensor operator implementations: `src/ir/op/tensor_ops/`
- Tile operator implementations: `src/ir/op/tile_ops/`
- Block operator implementations: `src/ir/op/block_ops/`
- [Block Operations Documentation](06-block_operations.md) - Detailed guide for block operations
