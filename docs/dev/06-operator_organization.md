# Operator Implementation Organization

PyPTO organizes operators into three categories (TensorOp, BlockOp, SyncOp) with modular source files under `src/ir/op/`. See [05-operator_registration.md](05-operator_registration.md) for registration details.

## File Structure

| Directory/File | Contents |
|----------------|----------|
| `src/ir/op/type_inference.cpp` | Shared type inference utilities |
| `tensor_ops/elementwise.cpp` | TensorOp: add, sub, mul, div |
| `block_ops/memory.cpp` | BlockOp: load, store, get_block_idx |
| `block_ops/elementwise.cpp` | BlockOp: add, mul, div, adds, muls, etc. |
| `block_ops/reduction.cpp` | BlockOp: sum (with axis, keepdim) |
| `block_ops/unary.cpp` | BlockOp: sqrt |
| `sync_ops/sync.cpp` | SyncOp: sync_src, sync_dst, barriers |

## Operator Categories

### TensorOp: N-Dimensional Tensor Operations

**Purpose**: General N-dimensional tensors with full broadcasting
**Type**: `TensorType` (arbitrary dimensions) | **Location**: `src/ir/op/tensor_ops/` | **Python API**: `from pypto.ir.op import tensor`

**Operations:** `tensor.add/sub/mul/div` (element-wise with full N-D broadcasting)

**Example:**
```python
from pypto.ir.op import tensor

ib = IRBuilder()
with ib.function("tensor_example") as f:
    input_a = f.param("input_a", ir.TensorType([128, 64, 32], DataType.FP32))
    input_b = f.param("input_b", ir.TensorType([128, 64, 32], DataType.FP32))
    f.return_type(ir.TensorType([128, 64, 32], DataType.FP32))
    result = ib.let("result", tensor.add(input_a, input_b))
    ib.return_stmt(result)
```

**C++ Implementation:**
```cpp
// src/ir/op/tensor_ops/elementwise.cpp
TypePtr DeduceTensorOpElementwiseBinaryType(args, kwargs, op_name) {
  auto tensor_type1 = cast<TensorType>(args[0]->GetType());
  auto tensor_type2 = cast<TensorType>(args[1]->GetType());
  auto result_dtype = PromoteDataTypes(tensor_type1->dtype_, tensor_type2->dtype_);
  auto broadcast_result = BroadcastShapes(tensor_type1->shape_, tensor_type2->shape_);
  return make_shared<TensorType>(broadcast_result.shape, *result_dtype);
}

REGISTER_OP("tensor.add")
    .set_op_category("TensorOp")
    .set_description("Element-wise addition with broadcasting")
    .add_argument("lhs", "Left tensor").add_argument("rhs", "Right tensor")
    .f_deduce_type(DeduceTensorOpElementwiseBinaryType);
```

### BlockOp: Hardware-Optimized Block Operations

**Purpose**: Hardware-optimized block operations with explicit memory management
**Type**: `TileType` (2D tiles in unified buffers)
**Location**: `src/ir/op/block_ops/`
**Python API**: `from pypto.ir.op import block`

**Design**: Uses `TileType` (not separate `BlockType`) for consistency with existing infrastructure. Namespace `block.*` + `TileType` clearly indicates hardware-optimized tile operations.

#### Operations

| Category | Operations | Description |
|----------|-----------|-------------|
| **Memory** | `block.get_block_idx` | Get block index (→ ScalarType) |
| | `block.load` | TensorType → TileType (DDR to unified buffer) |
| | `block.store` | TileType → TensorType (unified buffer to DDR) |
| **Element-wise** | `block.add/sub/mul/div` | Tile-Tile operations |
| | `block.adds/subs/muls/divs` | Tile-Scalar operations |
| **Unary** | `block.sqrt` | Element-wise square root |
| **Reduction** | `block.sum` | Reduction along axis (axis, keepdim) |

**Data Flow:** `TensorType (DDR) → block.load → TileType (Unified Buffer) → block.{ops} → TileType → block.store → TensorType (DDR)`

#### Example Usage

**Low-level API (IRBuilder):**
```python
from pypto.ir.op import block

ib = IRBuilder()
with ib.function("block_computation") as f:
    input_a = f.param("input_a", ir.TensorType([128, 128], DataType.FP32))
    input_b = f.param("input_b", ir.TensorType([128, 128], DataType.FP32))
    output = f.param("output", ir.TensorType([128, 1], DataType.FP32))
    f.return_type(ir.TensorType([128, 1], DataType.FP32))

    # Load, compute, reduce, store
    tile_a = ib.let("tile_a", block.load(input_a, 0, 0, 32, 128))
    tile_b = ib.let("tile_b", block.load(input_b, 0, 0, 32, 128))
    tile_mul = ib.let("tile_mul", block.mul(tile_a, tile_b))
    tile_sqrt = ib.let("tile_sqrt", block.sqrt(tile_mul))
    tile_sum = ib.let("tile_sum", block.sum(tile_sqrt, axis=1, keepdim=True))
    result = ib.let("result", block.store(tile_sum, 0, 0, 32, 1, output))
    ib.return_stmt(result)
```

**High-level API (Language DSL):**
```python
import pypto.language as pl

@pl.program
class MyProgram:
    @pl.function
    def block_computation(
        self,
        input_a: pl.Tensor[[128, 128], pl.FP32],
        input_b: pl.Tensor[[128, 128], pl.FP32],
        output: pl.Tensor[[128, 1], pl.FP32],
    ) -> pl.Tensor[[128, 1], pl.FP32]:
        tile_a: pl.Tile[[32, 128], pl.FP32] = pl.op.block.load(input_a, 0, 0, 32, 128)
        tile_b: pl.Tile[[32, 128], pl.FP32] = pl.op.block.load(input_b, 0, 0, 32, 128)
        tile_mul: pl.Tile[[32, 128], pl.FP32] = pl.op.block.mul(tile_a, tile_b)
        tile_sqrt: pl.Tile[[32, 128], pl.FP32] = pl.op.block.sqrt(tile_mul)
        tile_sum: pl.Tile[[32, 1], pl.FP32] = pl.op.block.row_sum(tile_sqrt)
        result: pl.Tensor[[128, 1], pl.FP32] = pl.op.block.store(tile_sum, 0, 0, 32, 1, output)
        return result
```

#### C++ Implementation Patterns

**Memory:**
```cpp
// src/ir/op/block_ops/memory.cpp
TypePtr DeduceBlockLoadType(args, kwargs, op_name) {
  auto tensor_type = cast<TensorType>(args[0]->GetType());
  std::vector<ExprPtr> tile_shape = args.size() >= 5
      ? std::vector{args[3], args[4]} : std::vector{dynamic_dim_expr, dynamic_dim_expr};
  return make_shared<TileType>(tile_shape, tensor_type->dtype_);
}
REGISTER_OP("block.load").set_op_category("BlockOp")
    .add_argument("tensor", "Source").add_argument("row_offset", "Row")
    .add_argument("col_offset", "Col").add_argument("height", "H").add_argument("width", "W")
    .f_deduce_type(DeduceBlockLoadType);
```

**Element-wise (Tile-Tile and Tile-Scalar):**
```cpp
// src/ir/op/block_ops/elementwise.cpp
TypePtr DeduceBlockOpElementwiseBinaryType(args, kwargs, op_name) {
  auto tile_type1 = cast<TileType>(args[0]->GetType());
  if (auto tile_type2 = cast<TileType>(args[1]->GetType())) {
    auto result_dtype = PromoteDataTypes(tile_type1->dtype_, tile_type2->dtype_);
    auto broadcast_result = BroadcastShapes(tile_type1->shape_, tile_type2->shape_);
    return make_shared<TileType>(broadcast_result.shape, *result_dtype);
  } else if (auto scalar_type2 = cast<ScalarType>(args[1]->GetType())) {
    auto result_dtype = PromoteDataTypes(tile_type1->dtype_, scalar_type2->dtype_);
    return make_shared<TileType>(tile_type1->shape_, *result_dtype);
  }
}
```

**Reduction:**
```cpp
// src/ir/op/block_ops/reduction.cpp
TypePtr DeduceBlockSumType(args, kwargs, op_name) {
  auto tile_type = cast<TileType>(args[0]->GetType());
  int axis = GetKwarg<int>(kwargs, "axis");
  bool keepdim = GetKwarg<bool>(kwargs, "keepdim", false);
  if (axis < 0) axis += tile_type->shape_.size();

  std::vector<ExprPtr> output_shape;
  for (size_t i = 0; i < tile_type->shape_.size(); ++i) {
    if (i == axis && keepdim) output_shape.push_back(const_int_expr(1));
    else if (i != axis) output_shape.push_back(tile_type->shape_[i]);
  }
  return output_shape.empty()
      ? make_shared<ScalarType>(tile_type->dtype_)
      : make_shared<TileType>(output_shape, tile_type->dtype_);
}
```

### SyncOp: Synchronization Operations

**Purpose**: Hardware synchronization and barriers | **Type**: `UnknownType` (no return), use in `EvalStmt`
**Location**: `src/ir/op/sync_ops/` | **Python API**: `from pypto.ir.op import system`

**Operations:** `system.sync_src/sync_dst` (set/wait flags), `system.bar_v/bar_m/bar_all` (barriers)

**Example:**
```python
from pypto.ir.op import system

with ib.function("sync_example") as f:
    ib.emit(system.bar_all())  # Global barrier
    ib.emit(system.sync_src(set_pipe=2, wait_pipe=4, event_id=0))
    ib.emit(system.sync_dst(set_pipe=2, wait_pipe=4, event_id=0))
```

**C++ Implementation:**
```cpp
// src/ir/op/sync_ops/sync.cpp
REGISTER_OP("system.bar_all")
    .set_op_category("SyncOp").set_pipe(PipeType::S)
    .no_argument().f_deduce_type(DeduceUnknownType);
```

**Note:** Use `ib.emit()` for ops returning no value. Associated with PipeType::S.

## Type System

| Type | Dimensions | Use Case | Memory | Special Fields |
|------|-----------|----------|--------|----------------|
| **TensorType** | N-D | General tensors, function params/returns | DDR (optional MemRef) | None |
| **TileType** | ≤2D | Hardware-optimized tiles in unified buffers | Unified buffer (optional MemRef) | Optional TileView |
| **ScalarType** | 0D | Scalar values | Register | dtype only |
| **UnknownType** | N/A | No return value (sync ops) | N/A | None |

**Type Hierarchy:**
```
Type (abstract)
├── UnknownType
├── ScalarType(dtype)
├── ShapedType(dtype, shape, memref?)
│   ├── TensorType(shape, dtype, memref?)
│   └── TileType(shape, dtype, memref?, tile_view?)
└── TupleType(types[])
```

**When to use:**
- **TensorType**: N-D tensors, DDR storage, function boundaries, flexible shapes
- **TileType**: 2D tiles, unified buffers, hardware-optimized computations, explicit memory management

## Organization Benefits

**Previous structure (✗):** All operators in 1-2 large files → recompilation overhead, difficult navigation

**New structure (✓):** Modular files by category

| Benefit | Description |
|---------|-------------|
| **Modularity** | Self-contained operator categories |
| **Build Performance** | Changes to one category don't rebuild others |
| **Maintainability** | Easy to locate and modify operators |
| **Scalability** | Straightforward to add new operators |
| **Registration** | Automatic via `REGISTER_OP` static initialization |

## Design Patterns

**1. Category-Based Organization:** Group related operators, share type deduction helpers.

```cpp
// src/ir/op/block_ops/elementwise.cpp
TypePtr DeduceBlockOpElementwiseBinaryType(...) { /* Shared logic */ }
REGISTER_OP("block.add").f_deduce_type(DeduceBlockOpElementwiseBinaryType);
REGISTER_OP("block.mul").f_deduce_type(DeduceBlockOpElementwiseBinaryType);
```

**2. Static Initialization:** `REGISTER_OP` macro auto-registers operators before `main()`.

**3. Type Deduction Helpers:** Per-category functions handle type inference.

```cpp
DeduceTensorOpElementwiseBinaryType(...)  // Tensor: Full N-D broadcasting
DeduceBlockOpElementwiseBinaryType(...)   // Block: 2D + scalar support
DeduceBlockSumType(...)                   // Block: Reduction with axis/keepdim
```

## Implementation Guide

**Adding operators:** See [src/ir/op/README.md](../../src/ir/op/README.md)

**Future extensions:**
- `tensor_ops/reduction.cpp` (sum, max, min), `matmul.cpp`, `transform.cpp` (reshape, transpose)
- `block_ops/` (complete), `sync_ops/` (complete)

## Testing & Build

**Tests:** `tests/ut/ir/test_op_registry.py`, `test_tensor_ops.py`, `test_block_ops.py`

**CMakeLists.txt:**
```cmake
set(PYPTO_SOURCES
    src/ir/op_registry.cpp src/ir/op/type_inference.cpp
    src/ir/op/tensor_ops/elementwise.cpp
    src/ir/op/block_ops/memory.cpp src/ir/op/block_ops/elementwise.cpp
    src/ir/op/block_ops/reduction.cpp src/ir/op/block_ops/unary.cpp
    src/ir/op/sync_ops/sync.cpp  # Add new files here
)
```

## Related Documentation

- [05-operator_registration.md](05-operator_registration.md) - Operator registration system details
- [08-ir_builder.md](08-ir_builder.md) - IR construction with IRBuilder
- [07-python_syntax.md](07-python_syntax.md) - Python IR syntax specification
