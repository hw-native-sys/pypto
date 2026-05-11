# Operator System

Type-safe operator definitions with automatic type deduction, organized into modular categories (TensorOp, TileOp, SyncOp, CrossCoreOp).

## Operator Categories

| Category | Types | Use Case | File Location |
| -------- | ----- | -------- | ------------- |
| **TensorOp** | TensorType | N-D tensor operations with broadcasting | `src/ir/op/tensor_ops/` |
| **TileOp** | TileType | Hardware-optimized tile operations | `src/ir/op/tile_ops/` |
| **SyncOp** | UnknownType | Pipeline barriers and synchronization | `src/ir/op/sync_ops/sync.cpp` |
| **CrossCoreOp** | UnknownType/TileType | AIC↔AIV cross-core communication | `src/ir/op/sync_ops/cross_core.cpp` |

**Key Features**: Fluent API, automatic type deduction, kwargs for metadata, NumPy-style broadcasting, type promotion, dynamic dimensions (`kDynamicDim`)

## Type System

```cpp
// TensorType: N-dimensional tensors
TensorType(DataType::FP32, {dim1, dim2, dim3, ...})

// TileType: Hardware-optimized tiles
TileType(DataType::FP16, {dim1, dim2})

// Dynamic dimensions (pypto/core/common.h)
constexpr int64_t kDynamicDim = -1;
auto dynamic_dim = make_int(kDynamicDim);
```

| Type | Dimensions | Use Case | Memory |
| ---- | ---------- | -------- | ------ |
| **TensorType** | N-D | General tensors, function params/returns | DDR (optional MemRef) |
| **TileType** | N-D | Hardware-optimized tiles in unified buffers | Unified buffer (optional MemRef) |
| **ScalarType** | 0D | Scalar values | Register |
| **UnknownType** | N/A | No return value (sync ops) | N/A |

## REGISTER_OP Fluent API

| Method | Purpose | Example |
| ------ | ------- | ------- |
| `set_op_category(str)` | Operator category | `.set_op_category("TensorOp")` |
| `set_description(str)` | Human-readable description | `.set_description("Element-wise add")` |
| `add_argument(name, desc)` | Positional Expr argument | `.add_argument("lhs", "Left tensor")` |
| `no_argument()` | No arguments (sync ops) | `.no_argument()` |
| `set_attr<T>(name)` | Kwarg schema (T: bool, int, DataType, etc.) | `.set_attr<bool>("a_trans")` |
| `f_deduce_type(fn)` | Type deduction function | `.f_deduce_type(DeduceAddType)` |

**Type Deduction Signature:**

```cpp
std::function<TypePtr(const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs)>
```

## C++ Registration Examples

### Simple Elementwise Operator

```cpp
// src/ir/op/tensor_ops/elementwise.cpp
REGISTER_OP("tensor.add")
    .set_op_category("TensorOp")
    .add_argument("lhs", "Left tensor")
    .add_argument("rhs", "Right tensor")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      CHECK(args.size() == 2);
      auto t1 = std::dynamic_pointer_cast<const TensorType>(args[0]->GetType());
      auto t2 = std::dynamic_pointer_cast<const TensorType>(args[1]->GetType());
      auto dtype = PromoteDataTypes(t1->dtype_, t2->dtype_);
      auto shape = BroadcastShapes(t1->shape_, t2->shape_);
      return std::make_shared<TensorType>(shape.shape, *dtype);
    });
```

### Operator with Kwargs

```cpp
// src/ir/op/tensor_ops/matmul.cpp
TypePtr DeduceMatMul(const std::vector<ExprPtr>& args,
                     const std::vector<std::pair<std::string, std::any>>& kwargs) {
  auto lhs = std::dynamic_pointer_cast<const TensorType>(args[0]->GetType());
  auto rhs = std::dynamic_pointer_cast<const TensorType>(args[1]->GetType());

  auto get = [&](const std::string& k, bool d) {
    for (const auto& [name, val] : kwargs)
      if (name == k) return std::any_cast<bool>(val);
    return d;
  };

  DataType dtype = [&]() {
    for (const auto& [k, v] : kwargs)
      if (k == "out_dtype") return static_cast<DataType>(std::any_cast<int>(v));
    return *PromoteDataTypes(lhs->dtype_, rhs->dtype_);
  }();

  bool a_t = get("a_trans", false), b_t = get("b_trans", false);
  ExprPtr m = a_t ? lhs->shape_[1] : lhs->shape_[0];
  ExprPtr n = b_t ? rhs->shape_[0] : rhs->shape_[1];
  return std::make_shared<TensorType>(std::vector<ExprPtr>{m, n}, dtype);
}

REGISTER_OP("tensor.matmul")
    .set_op_category("TensorOp")
    .add_argument("lhs", "Left matrix")
    .add_argument("rhs", "Right matrix")
    .set_attr<DataType>("out_dtype")
    .set_attr<bool>("a_trans")
    .set_attr<bool>("b_trans")
    .f_deduce_type(DeduceMatMul);
```

At the tile layer, `tile.batch_matmul` provides batched semantics for
`TileType` operands. It accepts rank >= 2 tiles, broadcasts the leading batch
dimensions, and keeps the same operand-only interface style as `tile.matmul`.
If batch operands need transpose semantics, that can be expressed either with
an explicit `tile.transpose(...)` on the inputs or by feeding a tile produced
by `tile.load(..., transpose=True)`. During later lowering to 2D
`tile.matmul`, both forms are normalized to the same operand-transpose
semantics.

`tile.batch_matmul_acc(acc, lhs, rhs)` is the accumulating counterpart for
the batched path: `acc = acc + lhs @ rhs` with the same rank>=2 + batch
broadcasting rules as `tile.batch_matmul`. The acc batch shape must match the
broadcast batch shape of `lhs`/`rhs` exactly; the matmul (M, N) dims must
match the trailing dims of acc; and the K dimension must match between lhs
and rhs. The inner accumulator type defaults to FP32 for floating inputs and
INT32 for integer inputs (mirroring `tile.matmul_acc`). At conversion time
`ConvertTensorToTileOps` dispatches `tensor.matmul` / `tensor.matmul_acc` to
this batched path whenever any operand has rank > 2; `FlattenTileNdTo2D`
later unrolls the batched form into per-batch 2D ops.

## Python Usage

```python
from pypto.pypto_core import DataType, ir
from pypto.ir import op

span = ir.Span.unknown()
dim4, dim8 = ir.ConstInt(4, DataType.INT32, span), ir.ConstInt(8, DataType.INT32, span)

# Create tensors
tensor_a = ir.Var("a", ir.TensorType([dim4, dim8], DataType.FP32), span)
tensor_b = ir.Var("b", ir.TensorType([dim8], DataType.FP32), span)

# Simple operators
result = op.tensor.add(tensor_a, tensor_b)  # Broadcasting: [4,8] + [8] → [4,8]

# Operators with kwargs
dim64, dim128 = ir.ConstInt(64, DataType.INT32, span), ir.ConstInt(128, DataType.INT32, span)
a = ir.Var("a", ir.TensorType([dim64, dim128], DataType.FP16), span)
b = ir.Var("b", ir.TensorType([dim128, dim64], DataType.FP16), span)
matmul = op.tensor.matmul(a, b, out_dtype=DataType.FP32, a_trans=True)

# Query registry
assert ir.is_op_registered("tensor.add")
op_instance = ir.get_op("tensor.add")
```

## Kwargs (Keyword Arguments)

Call expressions separate Expr arguments from metadata parameters using kwargs.

### Kwargs vs Args vs Attributes

| - | **Args** | **Kwargs** | **Op Attributes** |
| - | -------- | ---------- | ----------------- |
| **Type** | `ExprPtr` | `std::any` | Type-erased |
| **Scope** | Per-Call | Per-Call | Global |
| **Use** | Tensors, dims, offsets | `out_dtype`, flags, modes | Device, category |
| **Access** | `call.args_` | `call.kwargs_` | `op.get_attr()` |

### C++ - Reading Kwargs

```cpp
TypePtr DeduceCastType(const std::vector<ExprPtr>& args,
                       const std::vector<std::pair<std::string, std::any>>& kwargs) {
  auto input = std::dynamic_pointer_cast<const TensorType>(args[0]->GetType());

  // Required kwarg
  auto it = kwargs.find("target_type");
  CHECK(it != kwargs.end()) << "tensor.cast requires 'target_type'";
  DataType target = static_cast<DataType>(std::any_cast<int>(it->second));

  // Optional with default
  int mode = 0;
  auto mode_it = kwargs.find("mode");
  if (mode_it != kwargs.end()) mode = std::any_cast<int>(mode_it->second);

  return std::make_shared<TensorType>(input->shape_, target);
}
```

### Python - Using Kwargs

```python
result = op.tensor.matmul(a, b, out_dtype=DataType.FP32, a_trans=True)
print(result.kwargs)  # {'out_dtype': 51, 'a_trans': True}
```

## Broadcasting and Type Promotion

### NumPy-style Broadcasting

Dimensions aligned right to left:

```text
[4, 8] + [4, 8] → [4, 8]  # Exact match
[4, 8] + [8]    → [4, 8]  # Missing left dimension = 1
[4, 1] + [8]    → [4, 8]  # Size 1 broadcasts
[1, 8] + [4, 8] → [4, 8]  # Size 1 broadcasts
[4, 8] + [5]    → Error   # 8 ≠ 5
```

### Type Promotion

Standard numeric rules: float > int, larger > smaller, signed > unsigned (same size).

```text
INT32 + INT32 → INT32
INT32 + FP32  → FP32   (float precedence)
INT32 + INT64 → INT64  (larger size)
UINT32 + INT32 → INT32 (signed precedence)
```

## TensorOp: N-Dimensional Tensor Operations

**Purpose**: General N-dimensional tensors with full broadcasting
**Type**: `TensorType` (arbitrary dimensions)
**Location**: `src/ir/op/tensor_ops/`
**Python API**: `from pypto.ir.op import tensor`

**Operations:** `tensor.add/sub/mul/div` (element-wise with full N-D broadcasting), `tensor.maximum/minimum` (element-wise max/min; rhs may be tensor or scalar — `ConvertTensorToTileOps` dispatches to `tile.maximum/minimum` or `tile.maximums/minimums` based on the rhs operand type), `tensor.set_validshape` (internal, update valid-shape metadata without data movement — compiler-generated only), `tensor.sort32` / `tensor.mrgsort_format1` / `tensor.mrgsort_format2` (sorting; tensor-level counterparts of `tile.sort32` / `tile.mrgsort` — converted to tile ops by `ConvertTensorToTileOps`), `tensor.gather` (per-dim indexing; MVP supports rank-2 inputs with `dim=-1` and lowers to a per-row `tile.gather` loop via `ConvertTensorToTileOps`), `tensor.gather_mask` (mask-pattern gather; tensor-level counterpart of `tile.gather_mask`, with optional same-bit-width `output_dtype`), `tensor.ci` / `tensor.arange` (contiguous integer sequence generation; lowers to `tile.ci`; also exposed at top level as `pl.arange`)

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

## TileOp: Hardware-Optimized Tile Operations

**Purpose**: Hardware-optimized tile operations with explicit memory management
**Type**: `TileType` (tiles in unified buffers)
**Location**: `src/ir/op/tile_ops/`
**Python API**: `from pypto.ir.op import tile`

**Design**: Uses `TileType` (not separate `BlockType`) for consistency. Namespace `tile.*` + `TileType` clearly indicates hardware-optimized tile operations.

### Operations

| Category | Operations | Description |
| -------- | ---------- | ----------- |
| **Memory** | `tile.get_block_idx` | Get hardware block index (→ ScalarType(DataType::UINT64)) |
| - | `tile.load` | TensorType → TileType (DDR to unified buffer) |
| - | `tile.store` | TileType → TensorType (unified buffer to DDR) |
| **Element-wise** | `tile.add/sub/mul/div` | Tile-Tile operations |
| - | `tile.adds/subs/muls/divs` | Tile-Scalar operations |
| **Unary** | `tile.sqrt` | Element-wise square root |
| **Transform** | `tile.slice` | Extract a sub-tile with static shape and optional dynamic valid_shape |
| - | `tile.extract` | Extract a sub-tile from `src` at `(index_row, index_col)` — ISA TEXTRACT Variant 1 (Mat→Left/Right, Acc→Mat) |
| - | `tile.reshape` | Reshape tile to new dimensions (element count must match) |
| - | `tile.transpose` | Swap two axes of a tile |
| - | `tile.set_validshape` | Update valid-shape metadata without data movement |
| - | `tile.ci` | Generate contiguous integer sequence (start + k / start - k); dtype ∈ {INT16, INT32}; innermost dim != 1 |
| **Reduction** | `tile.sum` | Reduction along axis (axis, keepdim) |

**Data Flow:** `TensorType (DDR) → tile.load → TileType (Unified Buffer) → tile.{ops} → TileType → tile.store → TensorType (DDR)`

### Example Usage

```python
from pypto.ir.op import tile

ib = IRBuilder()
with ib.function("tile_computation") as f:
    input_a = f.param("input_a", ir.TensorType([128, 128], DataType.FP32))
    input_b = f.param("input_b", ir.TensorType([128, 128], DataType.FP32))
    output = f.param("output", ir.TensorType([128, 1], DataType.FP32))
    f.return_type(ir.TensorType([128, 1], DataType.FP32))

    # Load, compute, reduce, store
    tile_a = ib.let("tile_a", tile.load(input_a, [0, 0], [32, 128]))
    tile_b = ib.let("tile_b", tile.load(input_b, [0, 0], [32, 128]))
    tile_mul = ib.let("tile_mul", tile.mul(tile_a, tile_b))
    tile_sqrt = ib.let("tile_sqrt", tile.sqrt(tile_mul))
    tile_sum = ib.let("tile_sum", tile.sum(tile_sqrt, axis=1, keepdim=True))
    result = ib.let("result", tile.store(tile_sum, [0, 0], output))
    ib.return_stmt(result)
```

## SyncOp: Synchronization Operations

**Purpose**: Hardware synchronization and barriers
**Type**: `UnknownType` (no return), use in `EvalStmt`
**Location**: `src/ir/op/sync_ops/sync.cpp`
**Python API**: `from pypto.ir.op import system`

| Operation | Description | Kwargs |
| --------- | ----------- | ------ |
| `system.bar_all` | Global barrier | None |
| `system.bar_v` | Vector barrier | None |
| `system.bar_m` | Matrix barrier | None |
| `system.sync_src` | Set sync flag | `set_pipe`, `wait_pipe`, `event_id` |
| `system.sync_dst` | Wait sync flag | `set_pipe`, `wait_pipe`, `event_id` |
| `system.task_invalid` | Sentinel `PTO2TaskId::invalid()` — "no producer" seed for a TaskId carry (synthesised by `DeriveManualScopeDeps` only) | None |
| `system.task_id_of` | Extract the `PTO2TaskId` of a kernel-Call producer's LHS (synthesised by `DeriveManualScopeDeps` only) | None; sole positional arg is the producer Var |

`system.task_invalid` and `system.task_id_of` both return [`ScalarType(DataType::TASK_ID)`](02-types.md#scalartype). They are not user-callable from the DSL — they appear only after the [DeriveManualScopeDeps pass](../passes/33-derive_manual_scope_deps.md) inserts them while lowering `with pl.manual_scope()` regions. Source: `src/ir/op/sync_ops/task.cpp`.

**Python Example:**

```python
from pypto.ir.op import system
ib.emit(system.bar_all())
ib.emit(system.sync_src(set_pipe=2, wait_pipe=4, event_id=0))
```

**C++ Registration (`src/ir/op/sync_ops/sync.cpp`):**

```cpp
REGISTER_OP("system.bar_all")
    .set_op_category("SyncOp")
    .no_argument()
    .f_deduce_type(DeduceUnknownType);

REGISTER_OP("system.sync_src")
    .set_op_category("SyncOp")
    .no_argument()
    .set_attr<int>("set_pipe")
    .set_attr<int>("wait_pipe")
    .set_attr<int>("event_id")
    .no_argument()
    .f_deduce_type(DeduceUnknownType);
```

## CrossCoreOp: AIC↔AIV Communication

**Purpose**: Cross-core data transfer and pipe management between AIC (Cube) and AIV (Vector) kernels
**Type**: `UnknownType` (push/init/buffer/free ops) or `TileType` passthrough (pop ops)
**Location**: `src/ir/op/tile_ops/cross_core.cpp` (tpush/tpop) and `src/ir/op/sync_ops/cross_core.cpp` (tfree/pipe init/buffers)
**Python API**: `import pypto.language as pl` (promoted ops) or `from pypto.ir.op import tile, system`

### Data Transfer Operations

| Operation | Args | Description | Kwargs |
| --------- | ---- | ----------- | ------ |
| `tile.tpush_to_aiv` | 1 (tile) | Push tile from Cube to Vector | `split`, optional `id` |
| `tile.tpush_to_aic` | 1 (tile) | Push tile from Vector to Cube | `split`, optional `id` |
| `tile.tpop_from_aic` | 0 | Pop tile from Cube pipe (→ TileType) | `split`, optional `id` |
| `tile.tpop_from_aiv` | 0 | Pop tile from Vector pipe (→ TileType) | `split`, optional `id` |
| `system.tfree_to_aic` | 1 (tile) | Release slot back to Cube producer | optional `id` |
| `system.tfree_to_aiv` | 1 (tile) | Release slot back to Vector producer | optional `id` |

### Pipe Initialization Operations

| Operation | Args | Description | Kwargs |
| --------- | ---- | ----------- | ------ |
| `system.aic_initialize_pipe` | 2 | Init cross-core pipe on Cube side (positional: `c2v_consumer_buf`, `v2c_consumer_buf`, i32 SSA) | `dir_mask`, `slot_size`, optional `id` |
| `system.aiv_initialize_pipe` | 2 | Init cross-core pipe on Vector side (positional: `c2v_consumer_buf`, `v2c_consumer_buf`, i32 SSA) | `dir_mask`, `slot_size`, optional `id` |

### Buffer Management Operations

| Operation | Args | Description | Kwargs |
| --------- | ---- | ----------- | ------ |
| `system.reserve_buffer` | 0 | Reserve named cross-core buffer (consumer side) | `name`, `size`, `base`* |
| `system.import_peer_buffer` | 0 | Import buffer from peer function (producer side) | `name`, `peer_func` |

\* `base` defaults to `AUTO (-1)` for compiler-assigned address.

### DSL Example (cross-core V2C unidirectional)

`dir_mask=2` enables V2C only, so the C2V buffer operand must be an inactive-direction placeholder (`0`, or `pl.const(0, pl.INT32)`); the active side passes the reserved/imported buffer handle as the first positional operand.

```python
import pypto.language as pl

@pl.program
class CrossCoreExample:
    @pl.function(type=pl.FunctionType.InCore)
    def vector_producer(self, a: pl.Tensor[[16, 16], pl.FP16]):
        peer = pl.import_peer_buffer(name="v2c_buf", peer_func="cube_consumer")
        pl.aiv_initialize_pipe(pl.const(0, pl.INT32), peer, dir_mask=2, slot_size=512)

        tile_a: pl.Tile[[16, 16], pl.FP16] = pl.load(a, [0, 0], [16, 16])
        pl.tpush_to_aic(tile_a, split=0)

    @pl.function(type=pl.FunctionType.InCore)
    def cube_consumer(self, out: pl.Tensor[[16, 16], pl.FP32]) -> pl.Tensor[[16, 16], pl.FP32]:
        buf = pl.reserve_buffer(name="v2c_buf", size=4096, base=0x1000)
        pl.aic_initialize_pipe(pl.const(0, pl.INT32), buf, dir_mask=2, slot_size=512)

        received: pl.Tile[[16, 16], pl.FP16] = pl.tpop_from_aiv(split=0)
        pl.tfree_to_aiv(received)
        result: pl.Tensor[[16, 16], pl.FP32] = pl.store(received, [0, 0], out)
        return result
```

See [TPUSH/TPOP ISA Reference](../../reference/pto-isa/01-tpush_tpop.md) and [Buffer Management](../../reference/pto-isa/02-buffer_management.md) for hardware details.

## File Organization

| Directory/File | Contents |
| -------------- | -------- |
| `src/ir/op/type_inference.cpp` | Shared type inference utilities |
| `tensor_ops/elementwise.cpp` | TensorOp: add, sub, mul, div |
| `tile_ops/memory.cpp` | TileOp: load, store, read, get_block_idx |
| `tile_ops/elementwise.cpp` | TileOp: add, mul, div, adds, muls, etc. |
| `tile_ops/reduction.cpp` | TileOp: sum (with axis, keepdim) |
| `tile_ops/unary.cpp` | TileOp: sqrt |
| `sync_ops/sync.cpp` | SyncOp: sync_src, sync_dst, barriers |
| `sync_ops/cross_core.cpp` | CrossCoreOp: tpush, tpop, pipe init, buffers |

**Benefits**:

- **Modularity**: Self-contained operator categories
- **Build Performance**: Changes to one category don't rebuild others
- **Maintainability**: Easy to locate and modify operators
- **Scalability**: Straightforward to add new operators

## Adding New Operations

1. **Choose category file**: `src/ir/op/tensor_ops/elementwise.cpp`, `matmul.cpp`, `reduction.cpp`, or `src/ir/op/tile_ops/memory.cpp`, `unary.cpp`

2. **Implement type deduction**:

   ```cpp
   TypePtr DeduceType(const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
     CHECK(args.size() == 2) << "op requires 2 arguments";
     // Validate types, read kwargs, compute output type
     return result_type;
   }
   ```

3. **Register**:

   ```cpp
   REGISTER_OP("tensor.matmul")
       .set_op_category("TensorOp")
       .add_argument("lhs", "Left tensor")
       .add_argument("rhs", "Right tensor")
       .set_attr<DataType>("out_dtype")
       .f_deduce_type(DeduceType);
   ```

4. **Python wrapper** (`python/pypto/ir/op/tensor_ops.py`):

   ```python
   def matmul(lhs: Expr, rhs: Expr, out_dtype=None, a_trans=False) -> Call:
       kwargs = {}
       if out_dtype: kwargs["out_dtype"] = out_dtype.code() if isinstance(out_dtype, DataType) else out_dtype
       if a_trans: kwargs["a_trans"] = a_trans
       return _ir_core.create_op_call("tensor.matmul", [lhs, rhs], kwargs, Span.unknown())
   ```

5. **Add tests** in `tests/ut/ir/` and update `CMakeLists.txt` if needed

## References

- Common constants: `include/pypto/core/common.h`
- Type definitions: `include/pypto/ir/type.h`
- Operator registry: `include/pypto/ir/op_registry.h`
- Type inference utilities: `include/pypto/ir/type_inference.h`
- Type inference implementation: `src/ir/op/type_inference.cpp`
- Operator registry implementation: `src/ir/op_registry.cpp`
- Tensor operator implementations: `src/ir/op/tensor_ops/`
- Tile operator implementations: `src/ir/op/tile_ops/`
- Sync operator implementations: `src/ir/op/sync_ops/`
