# Operator Registration System

This document describes the operator registration system for PyPTO IR, which provides type-safe operator definitions with automatic type deduction.

## Overview

The operator registration system supports three kinds of operations:
- **ScalarOp**: Operations on scalar values (existing system, unchanged)
- **TensorOp**: Operations on N-dimensional tensors with broadcasting
- **BlockOp**: Block-level operations for hardware-optimized programming with tiles and scalar broadcasting

## Key Features

1. **Fluent API registration**: Expressive operator registration with method chaining
2. **Automatic type deduction**: Result types are automatically deduced from input types
3. **Kwargs support**: Separate Expr arguments from metadata parameters (e.g., out_dtype, axis, mode)
4. **Broadcasting support**: NumPy-style broadcasting for tensor/tile operations
5. **Type promotion**: Automatic data type promotion (e.g., INT32 + FP32 → FP32)
6. **Dynamic dimensions**: Support for dynamic dimensions using `kDynamicDim`

## Architecture

```
OpRegistry (Singleton)
    ├── TensorOp
    │   ├── TensorAdd
    │   ├── TensorSub
    │   ├── TensorMul
    │   └── TensorDiv
    └── BlockOp
        ├── BlockGetBlockIdx
        ├── BlockLoad
        ├── BlockStore
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

**Type Deduction Function Signature:**
```cpp
std::function<TypePtr(const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs)>
```

The type deduction function receives:
- `args`: Positional Expr arguments (tensors, scalars, etc.)
- `kwargs`: Keyword arguments for metadata (out_dtype, axis, mode, etc.)

**Example 1: Simple elementwise operator (no kwargs needed):**

**In `src/ir/op/tensor_ops/elementwise.cpp`:**
```cpp
REGISTER_OP("tensor.add")
    .set_op_category("TensorOp")
    .set_description("Element-wise addition of two tensors with broadcasting")
    .add_argument("lhs", "Left-hand side tensor (TensorType)")
    .add_argument("rhs", "Right-hand side tensor (TensorType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs)) {
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
      return std::make_shared<TensorType>(broadcast_result.shape, *result_dtype);
    });
```

**Example 2: Operator with kwargs (matmul with transpose flags):**

**In `src/ir/op/tensor_ops/matmul.cpp`:**
```cpp
// Helper to get kwargs value with default
template <typename T>
T GetKwarg(const std::vector<std::pair<std::string, std::any>>& kwargs),
           const std::string& key, const T& default_value = T{}) {
  auto it = kwargs.find(key);
  if (it == kwargs.end()) {
    return default_value;
  }
  return std::any_cast<T>(it->second);
}

TypePtr DeduceTensorMatMulType(const std::vector<ExprPtr>& args,
                                const std::vector<std::pair<std::string, std::any>>& kwargs)) {
  CHECK(args.size() == 2) << "tensor.matmul requires exactly 2 Expr arguments";

  auto lhs_type = std::dynamic_pointer_cast<const TensorType>(args[0]->GetType());
  auto rhs_type = std::dynamic_pointer_cast<const TensorType>(args[1]->GetType());

  // Read kwargs with defaults
  DataType out_dtype;
  auto it = kwargs.find("out_dtype");
  if (it != kwargs.end()) {
    out_dtype = static_cast<DataType>(std::any_cast<int>(it->second));
  } else {
    // Infer from inputs
    auto promoted = PromoteDataTypes(lhs_type->dtype_, rhs_type->dtype_);
    out_dtype = *promoted;
  }

  bool a_trans = GetKwarg<bool>(kwargs, "a_trans", false);
  bool b_trans = GetKwarg<bool>(kwargs, "b_trans", false);

  // Compute output shape based on transpose flags...
  std::vector<ExprPtr> output_shape;
  if (lhs_type->shape_.size() == 2 && rhs_type->shape_.size() == 2) {
    ExprPtr m_dim = a_trans ? lhs_type->shape_[1] : lhs_type->shape_[0];
    ExprPtr n_dim = b_trans ? rhs_type->shape_[0] : rhs_type->shape_[1];
    output_shape = {m_dim, n_dim};
  }
  // ... handle other cases ...

  return std::make_shared<TensorType>(output_shape, out_dtype);
}

REGISTER_OP("tensor.matmul")
    .set_op_category("TensorOp")
    .set_description("Matrix multiplication with optional transpose")
    .add_argument("lhs", "Left-hand side tensor (TensorType)")
    .add_argument("rhs", "Right-hand side tensor (TensorType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs)) {
      return DeduceTensorMatMulType(args, kwargs);
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
auto result = OpRegistry::GetInstance().Create("tensor.add", {tensor_a, tensor_b}, span);
// result->GetType() is TensorType(FP32, [4, 8]) due to broadcasting

// Create operation with kwargs
std::unordered_map<std::string, std::any> kwargs;
kwargs["out_dtype"] = static_cast<int>(DataType::FP32);
kwargs["a_trans"] = true;
auto matmul_result = OpRegistry::GetInstance().Create("tensor.matmul",
                                                       {tensor_a, tensor_b},
                                                       kwargs,
                                                       span);
```

## Python Usage

### Creating Tensors and Tiles

```python
from pypto.pypto_core import DataType, ir

span = ir.Span.unknown()

# Create a 2D tensor [4, 8]
dim4 = ir.ConstInt(4, DataType.INT32, span)
dim8 = ir.ConstInt(8, DataType.INT32, span)
tensor_type = ir.TensorType([dim4, dim8], DataType.FP32)
tensor_var = ir.Var("t", tensor_type, span)

# Create a 2D tile [16, 16]
dim16 = ir.ConstInt(16, DataType.INT32, span)
tile_type = ir.TileType([dim16, dim16], DataType.FP16)
tile_var = ir.Var("tile", tile_type, span)
```

### Using Operators

#### Simple Operators (Expr arguments only)

```python
# Create tensor variables
tensor_a = ir.Var("a", ir.TensorType([dim4, dim8], DataType.FP32), span)
tensor_b = ir.Var("b", ir.TensorType([dim8], DataType.FP32), span)

# Create tensor add operation (with automatic type deduction)
result = ir.create_op_call("tensor.add", [tensor_a, tensor_b], {}, span)

# result.type is TensorType(FP32, [4, 8]) due to broadcasting
print(result.type.dtype)  # FP32
print(len(result.type.shape))  # 2
```

#### Operators with Kwargs (Metadata parameters)

For operators with metadata parameters (like transpose flags, axis, mode), use kwargs:

```python
# Matrix multiplication with kwargs
a = ir.Var("a", ir.TensorType([64, 128], DataType.FP16), span)
b = ir.Var("b", ir.TensorType([128, 64], DataType.FP16), span)

# Kwargs separate metadata from Expr arguments
kwargs = {
    "out_dtype": DataType.FP32,  # Output data type (can pass DataType directly)
    "a_trans": True,              # Transpose first matrix
    "b_trans": False              # Don't transpose second matrix
}
matmul_call = ir.create_op_call("tensor.matmul", [a, b], kwargs, span)

# Using the high-level API (recommended)
from pypto.ir import op
matmul_call = op.tensor.matmul(a, b, out_dtype=DataType.FP32, a_trans=True)
```

#### High-Level Python API

The `pypto.ir.op` module provides convenient functions that handle kwargs:

```python
from pypto import DataType, ir

span = ir.Span.unknown()
a = ir.Var("a", ir.TensorType([64, 128], DataType.FP16), span)
b = ir.Var("b", ir.TensorType([128, 64], DataType.FP16), span)
c = ir.Var("c", ir.TensorType([64, 64], DataType.FP16), span)

# Matrix multiplication with kwargs
result = ir.op.tensor.matmul(a, b, out_dtype=DataType.FP32, a_trans=True)
# Python print: tensor.matmul(a, b, a_trans=True, out_dtype=51)

# Type casting with kwargs
casted = ir.op.tensor.cast(c, target_type=DataType.FP32, mode="floor")
# Python print: tensor.cast(c, mode=1, target_type=51)

# Reduction with kwargs
reduced = ir.op.tensor.row_max(c, axis=-1, keep_dim=True)
# Python print: tensor.row_max(c, keep_dim=True, axis=-1)

# Simple operators (no kwargs)
summed = ir.op.tensor.add(a, a)
# Python print: tensor.add(a, a)
```

### Query Operator Registry

```python
# Check if operator is registered
assert ir.is_op_registered("tensor.add")
assert ir.is_op_registered("block.mul")

# Get operator instance
op = ir.get_op("tensor.add")
print(op.name)  # "tensor.add"
```

### Kwargs (Keyword Arguments)

Call expressions support kwargs to separate Expr arguments from metadata parameters. Kwargs are stored per-Call instance and are used for operator configuration.

#### Call Structure

```cpp
class Call : public Expr {
public:
  OpPtr op_;                                           // Shared operator definition
  std::vector<ExprPtr> args_;                          // Positional Expr arguments
  std::unordered_map<std::string, std::any> kwargs_;   // Instance-specific kwargs
};
```

#### When to Use Kwargs

Use kwargs for:
- **Data type parameters**: `out_dtype`, `target_type`
- **Boolean flags**: `a_trans`, `b_trans`, `keep_dim`
- **Integer parameters**: `axis`, `mode`
- **Configuration**: Any metadata that's not an Expr

Use args for:
- **Tensors/tiles**: Input data
- **Shape dimensions**: Expr that represent dimensions
- **Offsets**: Expr that represent positions

#### C++ - Reading Kwargs in Type Deduction

```cpp
TypePtr DeduceTensorCastType(const std::vector<ExprPtr>& args,
                             const std::vector<std::pair<std::string, std::any>>& kwargs)) {
  CHECK(args.size() == 1) << "tensor.cast requires 1 argument";

  auto input_type = std::dynamic_pointer_cast<const TensorType>(args[0]->GetType());

  // Read required kwarg
  auto it = kwargs.find("target_type");
  CHECK(it != kwargs.end()) << "tensor.cast requires 'target_type' kwarg";
  DataType target_dtype = static_cast<DataType>(std::any_cast<int>(it->second));

  // Read optional kwarg with default
  int mode = 0;  // default: round
  auto mode_it = kwargs.find("mode");
  if (mode_it != kwargs.end()) {
    mode = std::any_cast<int>(mode_it->second);
  }

  return std::make_shared<TensorType>(input_type->shape_, target_dtype);
}
```

#### Python - Using Kwargs

```python
from pypto import DataType, ir

# Operators with kwargs
result = ir.op.tensor.matmul(a, b,
                             out_dtype=DataType.FP32,  # kwarg
                             a_trans=True,              # kwarg
                             b_trans=False)             # kwarg

# Accessing kwargs from Call object
print(result.kwargs)  # {'out_dtype': 51, 'a_trans': True, 'b_trans': False}

# Kwargs appear in Python printing
print(ir.python_print(result))
# Output: tensor.matmul(a, b, a_trans=True, b_trans=False, out_dtype=51)
```

#### Kwargs vs Op Attributes

**Op Attributes** (set during registration):
- Global metadata for all instances of an operator
- Examples: device preference, operator category, priority
- Accessed via `op.get_attr(key)`

**Call Kwargs** (set per Call instance):
- Instance-specific parameters
- Examples: transpose flags, axis, mode, out_dtype
- Accessed via `call.kwargs[key]`

```cpp
// Op attribute (shared across all calls)
REGISTER_OP("tensor.matmul")
    .set_attr<std::string>("device", "gpu")  // All matmul calls prefer GPU

// Call kwargs (per-instance)
auto call1 = Create("tensor.matmul", args, {{"a_trans", true}}, span);   // This call transposes A
auto call2 = Create("tensor.matmul", args, {{"b_trans", true}}, span);   // This call transposes B
```

### Operator Attributes (Kwarg Schema)

Operators use `set_attr` to define the schema of allowed kwargs. This specifies what kwargs an operator accepts and their expected types. Actual kwarg values are provided per-Call instance.

#### C++ - Defining Kwarg Schema

```cpp
REGISTER_OP("tensor.matmul")
    .set_op_category("TensorOp")
    .set_description("Matrix multiplication")
    .add_argument("lhs", "Left matrix")
    .add_argument("rhs", "Right matrix")
    .set_attr<DataType>("out_dtype")    // Accepts DataType kwarg
    .set_attr<bool>("a_trans")          // Accepts bool kwarg
    .set_attr<bool>("b_trans")          // Accepts bool kwarg
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs)) {
      // type deduction logic can read from kwargs
    });
```

**Note:** `set_attr` only defines the kwarg schema (allowed keys and types), not static metadata values. Only specific types are allowed: `bool`, `int`, `std::string`, `double`, `DataType`. This is enforced at compile-time.

#### Python - Checking Kwarg Schema

```python
# Get operator instance
op = ir.get_op("tensor.matmul")

# Check if a kwarg is registered in the schema
if op.has_attr("out_dtype"):
    print("out_dtype kwarg is supported")

# Get all registered kwarg keys
keys = op.get_attr_keys()
print(keys)  # ['out_dtype', 'a_trans', 'b_trans']
```

**Note:** `op.has_attr()` and `op.get_attr_keys()` now return information about the kwarg schema, not stored values. Actual kwarg values are stored in `Call.kwargs`.

#### Common Kwarg Types

- **Data types**: `set_attr<DataType>("out_dtype")` - output data type specification
- **Boolean flags**: `set_attr<bool>("transpose")` - operation mode toggles
- **Integer parameters**: `set_attr<int>("axis")` - dimensional parameters
- **String modes**: `set_attr<std::string>("mode")` - operation mode selection
- **Floating point**: `set_attr<double>("threshold")` - numerical thresholds

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

1. Choose or create appropriate category file under `src/ir/op/tensor_ops/` or `src/ir/op/block_ops/`
   - Element-wise ops: `elementwise.cpp`
   - Matrix ops: `matmul.cpp` (create if needed)
   - Reduction ops: `reduction.cpp` (create if needed)
   - Memory ops: `memory.cpp` (block_ops only)
   - Unary ops: `unary.cpp` (block_ops only)

2. Implement type deduction function:
   ```cpp
   TypePtr DeduceTensorMatMulType(const std::vector<ExprPtr>& args,
                                  const std::vector<std::pair<std::string, std::any>>& kwargs)) {
     // Validate args (only Expr arguments)
     CHECK(args.size() == 2) << "tensor.matmul requires 2 arguments";

     auto lhs_type = std::dynamic_pointer_cast<const TensorType>(args[0]->GetType());
     auto rhs_type = std::dynamic_pointer_cast<const TensorType>(args[1]->GetType());

     // Read kwargs with defaults
     bool a_trans = false;
     auto it = kwargs.find("a_trans");
     if (it != kwargs.end()) {
       a_trans = std::any_cast<bool>(it->second);
     }

     // Compute output type...
     return result_type;
   }
   ```

3. Register operator with `REGISTER_OP()`:
   ```cpp
   REGISTER_OP("tensor.matmul")
       .set_op_category("TensorOp")
       .set_description("Matrix multiplication of two tensors")
       .add_argument("lhs", "Left-hand side tensor")
       .add_argument("rhs", "Right-hand side tensor")
       .f_deduce_type([](const std::vector<ExprPtr>& args,
                         const std::vector<std::pair<std::string, std::any>>& kwargs)) {
         return DeduceTensorMatMulType(args, kwargs);
       });
   ```

4. Add Python wrapper function in `python/pypto/ir/op/tensor_ops.py`:
   ```python
   def matmul(lhs: Expr, rhs: Expr,
              out_dtype: Optional[Union[int, DataType]] = None,
              a_trans: bool = False,
              b_trans: bool = False) -> Call:
       """Matrix multiplication with optional transpose."""
       span = Span.unknown()

       args = [lhs, rhs]  # Only Expr arguments
       kwargs: Dict[str, Any] = {}

       if out_dtype is not None:
           kwargs["out_dtype"] = out_dtype.code() if isinstance(out_dtype, DataType) else out_dtype
       if a_trans:
           kwargs["a_trans"] = a_trans
       if b_trans:
           kwargs["b_trans"] = b_trans

       return _ir_core.create_op_call("tensor.matmul", args, kwargs, span)
   ```

5. Add tests in `tests/ut/ir/test_tensor_ops.py` or similar
6. Update `CMakeLists.txt` if adding a new operator category file

## References

- Common constants: `include/pypto/core/common.h`
- Type definitions: `include/pypto/ir/type.h`
- Operator registry: `include/pypto/ir/op_registry.h`
- Type inference utilities: `include/pypto/ir/type_inference.h`
- Type inference implementation: `src/ir/op/type_inference.cpp`
- Operator registry implementation: `src/ir/op_registry.cpp`
- Tensor operator implementations: `src/ir/op/tensor_ops/`
- Block operator implementations: `src/ir/op/block_ops/`
- [Block Operations Documentation](06-block_operations.md) - Detailed guide for block operations
