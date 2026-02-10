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
**Type**: `TileType` (tiles in unified buffers)
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
    tile_a = ib.let("tile_a", block.load(input_a, [0, 0], [32, 128]))
    tile_b = ib.let("tile_b", block.load(input_b, [0, 0], [32, 128]))
    tile_mul = ib.let("tile_mul", block.mul(tile_a, tile_b))
    tile_sqrt = ib.let("tile_sqrt", block.sqrt(tile_mul))
    tile_sum = ib.let("tile_sum", block.sum(tile_sqrt, axis=1, keepdim=True))
    result = ib.let("result", block.store(tile_sum, [0, 0], [32, 1], output))
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
        tile_a: pl.Tile[[32, 128], pl.FP32] = pl.load(input_a, [0, 0], [32, 128])
        tile_b: pl.Tile[[32, 128], pl.FP32] = pl.load(input_b, [0, 0], [32, 128])
        tile_mul: pl.Tile[[32, 128], pl.FP32] = pl.mul(tile_a, tile_b)
        tile_sqrt: pl.Tile[[32, 128], pl.FP32] = pl.sqrt(tile_mul)
        tile_sum: pl.Tile[[32, 1], pl.FP32] = pl.row_sum(tile_sqrt)
        result: pl.Tensor[[128, 1], pl.FP32] = pl.store(tile_sum, [0, 0], [32, 1], output)
        return result
```

#### C++ Implementation Patterns

**Memory** (`src/ir/op/block_ops/memory.cpp`): `DeduceBlockLoadType` extracts tile shape from load args, returns `TileType`.

**Element-wise** (`src/ir/op/block_ops/elementwise.cpp`): `DeduceBlockOpElementwiseBinaryType` handles both Tile-Tile (broadcast) and Tile-Scalar (preserve tile shape) cases.

**Reduction** (`src/ir/op/block_ops/reduction.cpp`): `DeduceBlockSumType` computes output shape based on axis/keepdim, returns `TileType` or `ScalarType`.

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
| **TileType** | N-D | Hardware-optimized tiles in unified buffers | Unified buffer (optional MemRef) | Optional TileView |
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
- **TileType**: Tiles in unified buffers, hardware-optimized computations, explicit memory management

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
DeduceBlockOpElementwiseBinaryType(...)   // Block: Tile + scalar support
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

## Unified Language API (`pl.*`)

At the language level, a **unified namespace** auto-dispatches between tensor and block operations based on the first argument's type (`Tensor` vs `Tile`). The explicit `pl.tensor.*` and `pl.block.*` namespaces remain available.

### Dispatch Rules

| First arg type | `pl.add(a, b)` dispatches to | Scalar rhs handling |
|----------------|----------------------------------|---------------------|
| `Tensor` | `tensor.add` | Handled internally by `tensor.add` |
| `Tile` + Tile rhs | `block.add` | N/A |
| `Tile` + scalar rhs | `block.adds` | Auto-selects scalar variant |

### Unified Ops

| Category | Operations |
|----------|-----------|
| **Binary arithmetic** | `add`, `sub`, `mul`, `div` (scalar auto-dispatch for Tile) |
| **Element-wise** | `maximum`, `exp` |
| **Shape** | `reshape`, `transpose`, `view` |
| **Matrix** | `matmul` (Tensor path accepts extra kwargs) |
| **Reduction** | `row_max`, `row_sum` |
| **Tensor-only** | `cast`, `create`, `assemble` |

### Promoted Ops (single-module only)

Block-only ops like `load`, `store`, `neg`, `sqrt`, etc. are promoted to `pl.*` for convenience. Scalar-specific ops (`adds`, `subs`, `muls`, `divs`) are **not** promoted — use `pl.add(tile, scalar)` instead.

### Example

```python
import pypto.language as pl

@pl.program
class Example:
    @pl.function
    def compute(
        self,
        a: pl.Tensor[[64, 64], pl.FP32],
        b: pl.Tensor[[64, 64], pl.FP32],
        out: pl.Tensor[[64, 64], pl.FP32],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        # Unified API — dispatches to tensor.add
        c: pl.Tensor[[64, 64], pl.FP32] = pl.add(a, b)

        # Block path — unified API dispatches to block.add
        tile_a: pl.Tile[[64, 64], pl.FP32] = pl.load(a, [0, 0], [64, 64])
        tile_b: pl.Tile[[64, 64], pl.FP32] = pl.load(b, [0, 0], [64, 64])
        tile_c: pl.Tile[[64, 64], pl.FP32] = pl.add(tile_a, tile_b)

        # Scalar auto-dispatch — dispatches to block.muls
        tile_d: pl.Tile[[64, 64], pl.FP32] = pl.mul(tile_c, 2.0)

        result: pl.Tensor[[64, 64], pl.FP32] = pl.store(tile_d, [0, 0], [64, 64], out)
        return result
```
