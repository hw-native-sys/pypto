# Operator Implementation Organization

This document describes the organization and implementation of operators in the PyPTO codebase, covering all operator categories: TensorOp, TileOp, and BlockOp.

## Table of Contents

- [Overview](#overview)
- [File Structure](#file-structure)
- [Operator Categories](#operator-categories)
  - [TensorOp](#tensorop-n-dimensional-tensor-operations)
  - [TileOp](#tileop-2d-tile-operations)
  - [BlockOp](#blockop-hardware-optimized-block-operations)
- [Type System](#type-system)
- [Organization Benefits](#organization-benefits)
- [Design Patterns](#design-patterns)
- [Implementation Guide](#implementation-guide)
- [Testing](#testing)

## Overview

PyPTO organizes operator implementations into separate source files under `src/ir/op/`, categorized by operator type and functionality. This modular approach supports three main operator categories, each designed for specific use cases and hardware optimization strategies.

## File Structure

```
src/ir/op/
├── README.md                    # Documentation
├── type_inference.cpp           # Type inference utilities implementation
├── tensor_ops/                  # Tensor operator implementations
│   └── elementwise.cpp          # Element-wise operations (add, sub, mul, div)
├── tile_ops/                    # Tile operator implementations
│   └── elementwise.cpp          # Element-wise operations (add, sub, mul, div)
└── block_ops/                   # Block operator implementations
    ├── memory.cpp               # Memory operations (ub_copy_in, ub_copy_out)
    ├── elementwise.cpp          # Element-wise operations (add, mul, div)
    ├── reduction.cpp            # Reduction operations (sum)
    └── unary.cpp                # Unary operations (sqrt)
```

## Operator Categories

### TensorOp: N-Dimensional Tensor Operations

**Purpose**: General N-dimensional tensor operations with full broadcasting support.

**Type**: Works on `TensorType` (arbitrary dimensions)

**Location**: `src/ir/op/tensor_ops/`

**Python API**: `from pypto.ir.op import tensor`

#### Operations

| Operation | Description | Broadcasting |
|-----------|-------------|--------------|
| `tensor.add` | Element-wise addition | ✅ Full |
| `tensor.sub` | Element-wise subtraction | ✅ Full |
| `tensor.mul` | Element-wise multiplication | ✅ Full |
| `tensor.div` | Element-wise division | ✅ Full |

#### Example Usage

```python
from pypto.ir.op import tensor
from pypto.ir.builder import IRBuilder
from pypto.pypto_core import DataType, ir

ib = IRBuilder()

with ib.function("tensor_example") as f:
    # N-dimensional tensors
    input_a = f.param("input_a", ir.TensorType([128, 64, 32], DataType.FP32))
    input_b = f.param("input_b", ir.TensorType([128, 64, 32], DataType.FP32))
    f.return_type(ir.TensorType([128, 64, 32], DataType.FP32))

    # Tensor operations with broadcasting
    result = ib.let("result", tensor.add(input_a, input_b))
    ib.return_stmt(result)

func = f.get_result()
```

#### C++ Implementation

```cpp
// src/ir/op/tensor_ops/elementwise.cpp

TypePtr DeduceTensorOpElementwiseBinaryType(
    const std::vector<ExprPtr>& args,
    const std::vector<std::pair<std::string, std::any>>& kwargs,
    const std::string& op_name) {

  auto tensor_type1 = std::dynamic_pointer_cast<const TensorType>(args[0]->GetType());
  auto tensor_type2 = std::dynamic_pointer_cast<const TensorType>(args[1]->GetType());

  // Promote data types
  auto result_dtype = PromoteDataTypes(tensor_type1->dtype_, tensor_type2->dtype_);

  // Broadcast shapes (supports N dimensions)
  auto broadcast_result = BroadcastShapes(tensor_type1->shape_, tensor_type2->shape_);

  return std::make_shared<TensorType>(broadcast_result.shape, *result_dtype);
}

REGISTER_OP("tensor.add")
    .set_op_category("TensorOp")
    .set_description("Element-wise addition of two tensors with broadcasting")
    .add_argument("lhs", "Left-hand side tensor (TensorType)")
    .add_argument("rhs", "Right-hand side tensor (TensorType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorOpElementwiseBinaryType(args, kwargs, "tensor.add");
    });
```

### TileOp: 2D Tile Operations

**Purpose**: 2D tile operations with shape constraints for hardware optimization.

**Type**: Works on `TileType` (at most 2 dimensions, with optional TileView)

**Location**: `src/ir/op/tile_ops/`

**Python API**: `from pypto.ir.op import tile`

#### Operations

| Operation | Description | Broadcasting |
|-----------|-------------|--------------|
| `tile.add` | Element-wise addition | ✅ 2D |
| `tile.sub` | Element-wise subtraction | ✅ 2D |
| `tile.mul` | Element-wise multiplication | ✅ 2D |
| `tile.div` | Element-wise division | ✅ 2D |

#### Key Features

- **2D Constraint**: Shape validation ensures at most 2 dimensions
- **TileView Support**: Optional view information for advanced tiling patterns
- **Memory Reference**: Can track memory location (DDR, UB, L0A, L0B, L0C)

#### Example Usage

```python
from pypto.ir.op import tile
from pypto.ir.builder import IRBuilder
from pypto.pypto_core import DataType, ir

ib = IRBuilder()

with ib.function("tile_example") as f:
    # 2D tiles only
    tile_a = f.param("tile_a", ir.TileType([32, 32], DataType.FP32))
    tile_b = f.param("tile_b", ir.TileType([32, 32], DataType.FP32))
    f.return_type(ir.TileType([32, 32], DataType.FP32))

    # Tile operations
    result = ib.let("result", tile.mul(tile_a, tile_b))
    ib.return_stmt(result)

func = f.get_result()
```

#### C++ Implementation

```cpp
// src/ir/op/tile_ops/elementwise.cpp

TypePtr DeduceTileOpElementwiseBinaryType(
    const std::vector<ExprPtr>& args,
    const std::vector<std::pair<std::string, std::any>>& kwargs,
    const std::string& op_name) {

  auto tile_type1 = std::dynamic_pointer_cast<const TileType>(args[0]->GetType());
  auto tile_type2 = std::dynamic_pointer_cast<const TileType>(args[1]->GetType());

  // Validate 2D shapes
  CHECK(tile_type1->shape_.size() <= 2) << "TileType must have at most 2 dimensions";
  CHECK(tile_type2->shape_.size() <= 2) << "TileType must have at most 2 dimensions";

  // Promote data types
  auto result_dtype = PromoteDataTypes(tile_type1->dtype_, tile_type2->dtype_);

  // Broadcast shapes (2D only)
  auto broadcast_result = BroadcastShapes(tile_type1->shape_, tile_type2->shape_);

  return std::make_shared<TileType>(broadcast_result.shape, *result_dtype);
}

REGISTER_OP("tile.mul")
    .set_op_category("TileOp")
    .set_description("Element-wise multiplication of two tiles with broadcasting")
    .add_argument("lhs", "Left-hand side tile (TileType)")
    .add_argument("rhs", "Right-hand side tile (TileType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileOpElementwiseBinaryType(args, kwargs, "tile.mul");
    });
```

### BlockOp: Hardware-Optimized Block Operations

**Purpose**: Hardware-optimized block-level programming with explicit memory management between tensors and unified buffers.

**Type**: Works on `TileType` (2D tiles/blocks in unified buffers)

**Location**: `src/ir/op/block_ops/`

**Python API**: `from pypto.ir.op import block`

#### Design Decision

Block operations use `TileType` instead of a separate `BlockType` because:

1. **Redundancy**: TileType provides all necessary features (2D constraints, MemRef, TileView)
2. **Consistency**: Leverages existing infrastructure (printer, hash, equality, serialization)
3. **Simplicity**: No duplication of type handling across transform modules
4. **Semantic Clarity**: "Tiles" and "blocks" are synonymous in hardware programming

The operation namespace `block.*` combined with `TileType` clearly communicates the intent: hardware-optimized block operations working on tiles in unified buffers.

#### Operations

##### Memory Operations (`memory.cpp`)

| Operation | Description | Input → Output |
|-----------|-------------|----------------|
| `block.get_block_idx` | Get current block index | None → ScalarType(INT32) |
| `block.ub_copy_in` | Copy from tensor to unified buffer | TensorType → TileType |
| `block.ub_copy_out` | Copy from unified buffer to tensor | TileType → TensorType |

##### Element-wise Operations (`elementwise.cpp`)

| Operation | Description | Support |
|-----------|-------------|---------|
| `block.add` | Element-wise addition | Tile-Tile, Tile-Scalar |
| `block.mul` | Element-wise multiplication | Tile-Tile, Tile-Scalar |
| `block.div` | Element-wise division | Tile-Tile, Tile-Scalar |

##### Unary Operations (`unary.cpp`)

| Operation | Description |
|-----------|-------------|
| `block.sqrt` | Element-wise square root |

##### Reduction Operations (`reduction.cpp`)

| Operation | Description | Parameters |
|-----------|-------------|------------|
| `block.sum` | Sum reduction along axis | axis, keepdim |

#### Data Flow Pattern

```
TensorType (DDR Memory)
    ↓ block.ub_copy_in
TileType (Unified Buffer)
    ↓ block.{add,mul,div,sqrt,sum}
TileType (Unified Buffer)
    ↓ block.ub_copy_out
TensorType (DDR Memory)
```

#### Example Usage

##### Basic Block Operations

```python
from pypto.ir.op import block
from pypto.ir.builder import IRBuilder
from pypto.pypto_core import DataType, ir

ib = IRBuilder()

with ib.function("block_multiply") as f:
    # Parameters: TensorType (full tensors in memory)
    input_a = f.param("input_a", ir.TensorType([128, 128], DataType.FP32))
    input_b = f.param("input_b", ir.TensorType([128, 128], DataType.FP32))
    output = f.param("output", ir.TensorType([128, 128], DataType.FP32))
    f.return_type(ir.TensorType([128, 128], DataType.FP32))

    # Load tiles from tensors into unified buffer
    tile_a = ib.let("tile_a", block.ub_copy_in(input_a, 0, 0, 32, 32))
    tile_b = ib.let("tile_b", block.ub_copy_in(input_b, 0, 0, 32, 32))

    # Perform computation on tiles
    tile_c = ib.let("tile_c", block.mul(tile_a, tile_b))

    # Store result back to tensor
    result = ib.let("result", block.ub_copy_out(tile_c, 0, 0, 32, 32, output))
    ib.return_stmt(result)

func = f.get_result()
```

##### Tile-Scalar Operations

```python
with ib.function("block_scale") as f:
    input_a = f.param("input_a", ir.TensorType([128, 128], DataType.FP32))
    output = f.param("output", ir.TensorType([128, 128], DataType.FP32))
    f.return_type(ir.TensorType([128, 128], DataType.FP32))

    tile_a = ib.let("tile_a", block.ub_copy_in(input_a, 0, 0, 32, 32))
    tile_scaled = ib.let("tile_scaled", block.mul(tile_a, 2.0))  # Tile-scalar
    result = ib.let("result", block.ub_copy_out(tile_scaled, 0, 0, 32, 32, output))
    ib.return_stmt(result)

func = f.get_result()
```

##### Complex Computation with Reduction

```python
with ib.function("complex_block_computation") as f:
    input_a = f.param("input_a", ir.TensorType([128, 128], DataType.FP32))
    input_b = f.param("input_b", ir.TensorType([128, 128], DataType.FP32))
    input_c = f.param("input_c", ir.TensorType([128, 128], DataType.FP32))
    output = f.param("output", ir.TensorType([128, 1], DataType.FP32))
    f.return_type(ir.TensorType([128, 1], DataType.FP32))

    # Load tiles
    tile_a = ib.let("tile_a", block.ub_copy_in(input_a, 0, 0, 32, 128))
    tile_b = ib.let("tile_b", block.ub_copy_in(input_b, 0, 0, 32, 128))
    tile_c = ib.let("tile_c", block.ub_copy_in(input_c, 0, 0, 32, 128))

    # Compute: sqrt(a * b + c)
    tile_mul = ib.let("tile_mul", block.mul(tile_a, tile_b))
    tile_add = ib.let("tile_add", block.add(tile_mul, tile_c))
    tile_sqrt = ib.let("tile_sqrt", block.sqrt(tile_add))

    # Reduce along axis 1 (column reduction)
    tile_sum = ib.let("tile_sum", block.sum(tile_sqrt, axis=1, keepdim=True))

    # Store result
    result = ib.let("result", block.ub_copy_out(tile_sum, 0, 0, 32, 1, output))
    ib.return_stmt(result)

func = f.get_result()
```

#### C++ Implementation

##### Memory Operations

```cpp
// src/ir/op/block_ops/memory.cpp

TypePtr DeduceBlockUbCopyInType(const std::vector<ExprPtr>& args,
                                const std::vector<std::pair<std::string, std::any>>& kwargs,
                                const std::string& op_name) {
  auto tensor_type = std::dynamic_pointer_cast<const TensorType>(args[0]->GetType());

  std::vector<ExprPtr> tile_shape;
  if (args.size() >= 5) {
    // Use provided height and width
    tile_shape.push_back(args[3]);  // height
    tile_shape.push_back(args[4]);  // width
  } else {
    // Use dynamic dimensions
    auto dynamic_dim_height = std::make_shared<ConstInt>(
        static_cast<int>(kDynamicDim), DataType::INT32, Span::unknown());
    auto dynamic_dim_width = std::make_shared<ConstInt>(
        static_cast<int>(kDynamicDim), DataType::INT32, Span::unknown());
    tile_shape.push_back(dynamic_dim_height);
    tile_shape.push_back(dynamic_dim_width);
  }

  // Return TileType with same dtype as tensor
  return std::make_shared<TileType>(tile_shape, tensor_type->dtype_);
}

REGISTER_OP("block.ub_copy_in")
    .set_op_category("BlockOp")
    .set_description("Copy data from tensor to unified buffer (tile)")
    .add_argument("tensor", "Source tensor (TensorType)")
    .add_argument("row_offset", "Row offset (scalar)")
    .add_argument("col_offset", "Column offset (scalar)")
    .add_argument("height", "Tile height (scalar)")
    .add_argument("width", "Tile width (scalar)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceBlockUbCopyInType(args, kwargs, "block.ub_copy_in");
    });
```

##### Element-wise Operations with Scalar Support

```cpp
// src/ir/op/block_ops/elementwise.cpp

TypePtr DeduceBlockOpElementwiseBinaryType(
    const std::vector<ExprPtr>& args,
    const std::vector<std::pair<std::string, std::any>>& kwargs,
    const std::string& op_name) {

  auto tile_type1 = std::dynamic_pointer_cast<const TileType>(args[0]->GetType());
  auto tile_type2 = std::dynamic_pointer_cast<const TileType>(args[1]->GetType());
  auto scalar_type2 = std::dynamic_pointer_cast<const ScalarType>(args[1]->GetType());

  if (tile_type2) {
    // Tile-Tile: use broadcasting
    auto result_dtype = PromoteDataTypes(tile_type1->dtype_, tile_type2->dtype_);
    auto broadcast_result = BroadcastShapes(tile_type1->shape_, tile_type2->shape_);
    return std::make_shared<TileType>(broadcast_result.shape, *result_dtype);
  } else if (scalar_type2) {
    // Tile-Scalar: result has same shape as tile
    auto result_dtype = PromoteDataTypes(tile_type1->dtype_, scalar_type2->dtype_);
    return std::make_shared<TileType>(tile_type1->shape_, *result_dtype);
  }

  CHECK(false) << "Invalid operand types for " << op_name;
  return nullptr;
}

REGISTER_OP("block.mul")
    .set_op_category("BlockOp")
    .set_description("Element-wise multiplication of two tiles or tile and scalar")
    .add_argument("lhs", "Left-hand side tile (TileType)")
    .add_argument("rhs", "Right-hand side tile (TileType) or scalar (ScalarType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceBlockOpElementwiseBinaryType(args, kwargs, "block.mul");
    });
```

##### Reduction Operations

```cpp
// src/ir/op/block_ops/reduction.cpp

TypePtr DeduceBlockSumType(const std::vector<ExprPtr>& args,
                           const std::vector<std::pair<std::string, std::any>>& kwargs,
                           const std::string& op_name) {
  auto tile_type = std::dynamic_pointer_cast<const TileType>(args[0]->GetType());

  // Extract axis and keepdim parameters
  auto const_axis = std::dynamic_pointer_cast<const ConstInt>(args[1]);
  int axis_value = const_axis->value_;

  bool keepdim = false;
  if (args.size() == 3) {
    auto const_keepdim = std::dynamic_pointer_cast<const ConstInt>(args[2]);
    keepdim = (const_keepdim->value_ == 1);
  }

  // Compute output shape based on reduction
  std::vector<ExprPtr> output_shape;
  const auto& input_shape = tile_type->shape_;

  if (keepdim) {
    // Keep dimension as 1
    for (size_t i = 0; i < input_shape.size(); ++i) {
      if (static_cast<int>(i) == axis_value) {
        output_shape.push_back(std::make_shared<ConstInt>(1, DataType::INT32, Span::unknown()));
      } else {
        output_shape.push_back(input_shape[i]);
      }
    }
  } else {
    // Remove dimension
    for (size_t i = 0; i < input_shape.size(); ++i) {
      if (static_cast<int>(i) != axis_value) {
        output_shape.push_back(input_shape[i]);
      }
    }
  }

  // Return TileType with reduced shape or ScalarType if fully reduced
  if (output_shape.empty()) {
    return std::make_shared<ScalarType>(tile_type->dtype_);
  }
  return std::make_shared<TileType>(output_shape, tile_type->dtype_);
}

REGISTER_OP("block.sum")
    .set_op_category("BlockOp")
    .set_description("Sum reduction of a tile along specified axes")
    .add_argument("tile", "Input tile (TileType)")
    .add_argument("axes", "Reduction axes (required)")
    .add_argument("keepdim", "Keep reduced dimensions as 1 (optional, default false)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceBlockSumType(args, kwargs, "block.sum");
    });
```

#### Best Practices

1. **Variable Naming**: Use `tile_xxx` for TileType variables to distinguish from TensorType
2. **Type Safety**: Ensure tiles have correct dimensions (at most 2D)
3. **Memory Management**: Explicit ub_copy_in/ub_copy_out for data transfer
4. **Shape Compatibility**: Verify shapes are compatible for element-wise operations
5. **Reduction Axes**: Validate axes are within valid range (0 to ndim-1)

## Type System

### Type Hierarchy

```
Type (abstract)
├── UnknownType
├── ScalarType(dtype)
├── ShapedType(dtype, shape, memref?)
│   ├── TensorType(shape, dtype, memref?)
│   └── TileType(shape, dtype, memref?, tile_view?)
└── TupleType(types[])
```

### Type Comparison

| Feature | TensorType | TileType |
|---------|-----------|----------|
| Dimensions | N-dimensional | At most 2D |
| Use Case | General tensor ops | Hardware-optimized tile/block ops |
| Memory Ref | Optional | Optional |
| Special Fields | None | Optional TileView |
| Broadcasting | Full N-D | 2D only |
| Validation | None | Shape constraint check |

### When to Use Which Type?

- **TensorType**:
  - General purpose N-dimensional tensors
  - Main memory (DDR) storage
  - Function parameters and return types
  - Flexible shape operations

- **TileType**:
  - 2D tiles/blocks in unified buffers
  - Hardware-optimized computations
  - Explicit memory management
  - Block-level programming

## Organization Benefits

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
├── tile_ops/elementwise.cpp     # Tile elementwise ops
└── block_ops/                   # Block operations
    ├── memory.cpp
    ├── elementwise.cpp
    ├── reduction.cpp
    └── unary.cpp
```

**Benefits:**
1. **Modularity**: Each operator category is self-contained
2. **Build Performance**: Changes to one category don't rebuild others
3. **Maintainability**: Easy to find and modify specific operators
4. **Scalability**: Adding new operators is straightforward
5. **Automatic Registration**: Uses static initialization via `REGISTER_OP` macro

## Design Patterns

### 1. Category-Based Organization

Operators are grouped into category files based on their functionality and type:

```cpp
// src/ir/op/block_ops/elementwise.cpp

// Helper function for common type deduction logic
TypePtr DeduceBlockOpElementwiseBinaryType(...) {
  // Validate arguments, promote types, broadcast shapes
  // ...
}

// Register multiple related operators
REGISTER_OP("block.add").f_deduce_type([](auto& args, auto& kwargs) {
  return DeduceBlockOpElementwiseBinaryType(args, kwargs, "block.add");
});

REGISTER_OP("block.mul").f_deduce_type([](auto& args, auto& kwargs) {
  return DeduceBlockOpElementwiseBinaryType(args, kwargs, "block.mul");
});
```

### 2. Static Initialization Pattern

The `REGISTER_OP` macro uses static initialization to automatically register operators:

```cpp
// In op_registry.h
#define REGISTER_OP(OpName)                                                                           \
  static PYPTO_STR_CONCAT(PYPTO_UNUSED ::pypto::ir::OpRegistryEntry& OpRegistryEntry_, __COUNTER__) = \
      ::pypto::ir::OpRegistry::GetInstance().Register(OpName)
```

This eliminates manual registration - operators are registered automatically before `main()` runs.

### 3. Type Deduction Functions

Each operation category has helper functions for type deduction:

```cpp
// Tensor ops: Full N-D broadcasting
TypePtr DeduceTensorOpElementwiseBinaryType(...)

// Tile ops: 2D broadcasting with validation
TypePtr DeduceTileOpElementwiseBinaryType(...)

// Block ops: 2D broadcasting + scalar support
TypePtr DeduceBlockOpElementwiseBinaryType(...)
```

## Implementation Guide

### Adding New Operators

See the [README in src/ir/op/](../../src/ir/op/README.md) for step-by-step instructions.

### Future Extensions

As more operators are added, new category files can be created:

```
src/ir/op/
├── tensor_ops/
│   ├── elementwise.cpp         # ✓ Exists
│   ├── reduction.cpp           # TODO: Sum, Max, Min, etc.
│   ├── matmul.cpp              # TODO: Matrix multiplication
│   └── transform.cpp           # TODO: Reshape, Transpose, etc.
├── tile_ops/
│   ├── elementwise.cpp         # ✓ Exists
│   ├── matmul.cpp              # TODO: Tile matmul for hardware
│   └── load_store.cpp          # TODO: Tile load/store operations
└── block_ops/
    ├── memory.cpp              # ✓ Exists
    ├── elementwise.cpp         # ✓ Exists
    ├── reduction.cpp           # ✓ Exists
    └── unary.cpp               # ✓ Exists
```

## Testing

### Test Organization

All operators are tested through:
- `tests/ut/ir/test_op_registry.py` - Operator registration and type deduction
- `tests/ut/ir/test_tensor_ops.py` - Tensor operations
- `tests/ut/ir/test_tile_ops.py` - Tile operations
- `tests/ut/ir/test_block_ops.py` - Block operations with integration tests

### Example Test

```python
def test_ub_copy_in(self):
    """Test block.ub_copy_in operation."""
    ib = IRBuilder()

    with ib.function("test_ub_copy_in") as f:
        input_tensor = f.param("input", ir.TensorType([128, 128], DataType.FP32))
        f.return_type(ir.TileType([
            ir.ConstInt(32, DataType.INT32, ir.Span.unknown()),
            ir.ConstInt(32, DataType.INT32, ir.Span.unknown()),
        ], DataType.FP32))

        tile = ib.let("tile", block.ub_copy_in(input_tensor, 0, 0, 32, 32))
        ib.return_stmt(tile)

    func = f.get_result()
    assert func is not None
    assert "block.ub_copy_in" in str(func)
```

## Build System Integration

The CMakeLists.txt includes all operator implementation files:

```cmake
set(PYPTO_SOURCES
    # ... other sources ...
    src/ir/op_registry.cpp
    src/ir/op/type_inference.cpp
    src/ir/op/tensor_ops/elementwise.cpp
    src/ir/op/tile_ops/elementwise.cpp
    src/ir/op/block_ops/memory.cpp
    src/ir/op/block_ops/elementwise.cpp
    src/ir/op/block_ops/reduction.cpp
    src/ir/op/block_ops/unary.cpp
    # Add new category files here
)
```

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

Our structure follows similar principles with clearer categorization for hardware-specific needs (tensor vs tile vs block operations).

## Summary

This reorganization:
- ✅ Improves code organization and maintainability
- ✅ Reduces compilation dependencies
- ✅ Makes it easier to add new operators
- ✅ Provides clear separation between operator categories
- ✅ Supports hardware-optimized programming patterns
- ✅ Maintains all existing functionality
- ✅ Passes all IR tests

## Related Documentation

- [Operator Registration (03-operator_registration.md)](03-operator_registration.md) - Details on the operator registration system
- [IR Builder (06-ir_builder.md)](06-ir_builder.md) - Using IRBuilder for IR construction
- [Python Syntax (05-python_syntax.md)](05-python_syntax.md) - Python IR syntax specification
