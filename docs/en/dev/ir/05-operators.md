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
an explicit `tile.transpose(...)` on the inputs or by a zero-copy
`tile.transpose_view(...)` over a natural `tile.load`. During later lowering to
2D `tile.matmul`, both forms are normalized to the same operand-transpose
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

**Operations:** `tensor.add/sub/mul/div` (element-wise with full N-D broadcasting), `tensor.maximum/minimum` (element-wise max/min; rhs may be tensor or scalar — `ConvertTensorToTileOps` dispatches to `tile.maximum/minimum` or `tile.maximums/minimums` based on the rhs operand type), `tensor.set_validshape` (internal, update valid-shape metadata without data movement — compiler-generated only), `tensor.sort32` / `tensor.mrgsort_format1` / `tensor.mrgsort_format2` (sorting; tensor-level counterparts of `tile.sort32` / `tile.mrgsort` — converted to tile ops by `ConvertTensorToTileOps`), `tensor.gather` (per-dim indexing; MVP supports rank-2 inputs with `dim=-1` and lowers to a per-row `tile.gather` loop via `ConvertTensorToTileOps`), `tensor.gather_mask` (mask-pattern gather; tensor-level counterpart of `tile.gather_mask`, with optional same-bit-width `output_dtype` — see [Mask patterns](#mask-patterns)), `tensor.scatter` (column scatter; the column-wise inverse of `tensor.gather`, MVP supports rank-2 inputs with `dim=-1` — `out[b, index[b, k]] = src[b, k]`, `index` same shape as `src` — and lowers to `tile.scatter` via `ConvertTensorToTileOps`), `tensor.scatter_mask` (mask-pattern row-scatter; tensor-level counterpart of `tile.scatter_mask`, expands a compact `input` tensor into the mask-marked columns of `dst` — see [Mask patterns](#mask-patterns)), `tensor.ci` / `tensor.arange` (contiguous integer sequence generation; lowers to `tile.ci`; also exposed at top level as `pl.arange`)

`tensor.view` is a metadata-only zero-copy shape/layout reinterpret. It is
registered as a `TensorOp` passthrough in `ConvertTensorToTileOps`; PTO in-core
codegen lowers it to `pto.make_tensor_view` over the original base pointer.
Targets require rank at least 1 (DN requires rank at least 2); orchestration
shape reinterpret is ND-only and cannot also change layout.
Shape reinterpretation of a partially valid source is limited to a packed ND
leading-dimension collapse to 2D and requires an explicit target `valid_shape`;
this form preserves the source tensor kind and backing metadata.

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
| **Transform** | `tile.slice` | Extract a sub-tile with static shape, optional dynamic valid_shape, and optional `drop_dims` (numpy-style rank reduction over static unit axes; result clamped to a 2D minimum) |
| - | `tile.extract` | Extract a sub-tile from `src` at `(index_row, index_col)` — ISA TEXTRACT Variant 1 (Mat→Left/Right, Acc→Mat) |
| - | `tile.reshape` | Reshape tile to new dimensions (element count must match) |
| - | `tile.transpose` | Swap two axes of a tile |
| - | `tile.set_validshape` | Update valid-shape metadata without data movement |
| - | `tile.ci` | Generate contiguous integer sequence (start + k / start - k); dtype ∈ {INT16, INT32}; innermost dim != 1 |
| **Reduction** | `tile.row_*` / `tile.col_*` | Direction-specific reduction (`row_sum`/`row_max`/`row_min`/`row_prod` collapse the last axis; `col_*` collapse axis 0). There is no axis-parameterized reduction — the ISA has only direction-specific intrinsics (`pto.trowsum`, `pto.tcolsum`, …) |
| **Scatter** | `tile.scatter` | Row-scatter `src` into `dst` at per-row indices (`pto.tscatter` index form; DPS — `dst` is in/out, the result aliases `dst`). `src`/`dst` dtype ∈ {I8, I16, I32, FP16, FP32, BF16}; `indexes` dtype ∈ {I16, I32}; element-size matching rule: 4-byte dst ↔ INT32, 2-byte dst ↔ INT16, 1-byte dst ↔ INT16. |
| - | `tile.scatter_mask` | Mask-pattern row-scatter: write each `src` row into the mask-marked columns of `dst` (DPS — `dst` is in/out). A PyPTO codegen form lowered to a `pto.tscatter` mask emission — **not** a distinct pto-isa instruction (unlike `tile.gather_mask`). See [Mask patterns](#mask-patterns). |

**Data Flow:** `TensorType (DDR) → tile.load → TileType (Unified Buffer) → tile.{ops} → TileType → tile.store → TensorType (DDR)`

### Mask patterns

`*.gather_mask` / `*.scatter_mask` use a compile-time `MaskPattern` (`pl.tile.MaskPattern`, integer values 1–7, matching the hardware `VREDUCEv2` pattern modes) to mark a per-row subset of columns (names read **right-to-left**, rightmost bit = column 0). The same mark set drives the two ops in opposite directions. **`gather_mask`** *selects & compacts*: it reads the marked columns of a wide input into the leading columns of a narrower output (`out_cols = cols / stride`); this is a real pto-isa instruction (`pto.tgather` mask form), supported on A2/A3 **and A5**. **`scatter_mask`** *places & expands*: it writes a compact input into the marked columns of a wider `dst` (`dst_cols = cols * stride`), leaving unmarked columns at their prior `dst` value (DPS); this is a **PyPTO codegen-level form, not a distinct pto-isa instruction** — there is no `pto.tscatter` mask instruction (unlike gather) — and PyPTO emits it for A2/A3 / CPU-sim style lowering paths. E.g. for `[a0 a1 a2 a3 a4 a5 a6 a7]`: gather `P0101 → [a0 a2 a4 a6]`; scatter of `[s0 s1 s2 s3]` `P0101 → [s0 · s1 · s2 · s3 ·]` (`·` = preserved `dst`).

| Pattern | int | Marks column `c` when | Marked columns | Stride |
| ------- | --- | --------------------- | -------------- | ------ |
| `P0101` | 1 | `c % 2 == 0` | 0, 2, 4, … | 2 |
| `P1010` | 2 | `c % 2 == 1` | 1, 3, 5, … | 2 |
| `P0001` | 3 | `c % 4 == 0` | 0, 4, 8, … | 4 |
| `P0010` | 4 | `c % 4 == 1` | 1, 5, 9, … | 4 |
| `P0100` | 5 | `c % 4 == 2` | 2, 6, 10, … | 4 |
| `P1000` | 6 | `c % 4 == 3` | 3, 7, 11, … | 4 |
| `P1111` | 7 | always | all | 1 |

The last dim must be divisible by the stride. `gather_mask` also accepts an optional same-bit-width `output_dtype` (bit-reinterpret, not a value cast). Reference: gather selection is `MaskSelect` in `pto-isa` `include/pto/cpu/TGather.hpp`; pypto type deduction in `src/ir/op/tile_ops/gather.cpp` (gather) / `src/ir/op/tile_ops/scatter.cpp` (scatter).

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
    # row_sum collapses the last axis -> [32, 1]. Its scratch tile must have
    # the same dtype and rank and be at least as large as the input in every dimension.
    tmp_tile = ib.let("tmp_tile", tile.create([32, 128], DataType.FP32))
    tile_sum = ib.let("tile_sum", tile.row_sum(tile_sqrt, tmp_tile))
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
| `system.fence` | Memory barrier over global memory (lowers to `pto.fence.barrier_all #pto.fence_scope<gm>`) | None |
| `system.cacheinvalid` | Invalidate the cache lines backing a tensor sub-region. Args: `tensor`, `shapes` (N-D), `offsets` (N-D). All-1 `shapes` (scalar write) lowers to `pto.addptr` + `pto.cmo.cacheinvalid %write_ptr single_cache_line`; a larger region (tile store) lowers to `pto.partition_view` + `pto.cmo.cacheinvalid %payload_view single_cache_line : !pto.partition_tensor_view<...>` | None |
| `system.syncall` | Cross-core all-participant barrier (`pto::SYNCALL`). `mode="hard"` (FFTS, no operands) or `mode="soft"` (GM-polling, operands) | `core_type` (`"aiv_only"` \| `"aic_only"` \| `"mix"`), `mode` (`"hard"` \| `"soft"`) |
| `system.sync_src` | Set sync flag | `set_pipe`, `wait_pipe`, `event_id` |
| `system.sync_dst` | Wait sync flag | `set_pipe`, `wait_pipe`, `event_id` |
| `system.task_invalid` | Sentinel `PTO2TaskId::invalid()` — "no producer" seed for a TaskId carry | None |
| `system.task_is_valid` | Test whether a `TASK_ID` value is a valid (non-sentinel) handle | None; sole positional arg is the TaskId Var |

`system.syncall` has two modes. The **hard** form (`mode="hard"`, default) emits an FFTS barrier that waits for **all** physical cores of the selected `core_type`; the kernel must be launched at full occupancy (one block per physical core) **and with `sync_start=True`** (so all blocks are co-resident — a non-sync_start launch may dispatch blocks in waves and deadlock the barrier), or it deadlocks (AICore error 507018). The **soft** form (`mode="soft"`) polls a shared GM workspace and so works at **partial** occupancy. `gm_workspace` is a shared, zero-initialized GM `INT32` tensor with `used_cores * 8` slots (pass it as a kernel parameter so all blocks share one buffer); the scratch tile(s) are compiler-synthesized local staging buffers; `used_cores` is the participant count. Soft mode is supported for every `core_type`, with operands that vary by participant set:

- `aiv_only`: `[gm_workspace, ub_scratch, used_cores]` — one UB (Vec) staging tile.
- `aic_only`: `[gm_workspace, l1_scratch, used_cores]` — one flat L1 (Mat, `slayout=none_box`) staging tile.
- `mix`: `[gm_workspace, ub_scratch, l1_scratch, used_cores]` — both a UB and a flat L1 tile. The barrier rendezvouses AIC + AIV cores, so `used_cores` is the *total* participant count (AIC blocks + AIV subblocks). The op is duplicated onto both the cube and vector lanes; each lane uses its own tile (the other is dead), matching pto-isa's soft-mix lowering.

The flat L1 staging tile is created via `pl.tile.create(..., target_memory=pl.Mem.Mat, flat_layout=True)`, which keeps the contiguous `slayout=none_box` layout (a normal boxed NZ Mat tile would mis-place the 8-int32 counter slots).

The unified `mode=` keyword API (`mode="hard"` / `mode="soft"`) is the **DSL** surface (`pl.system.syncall`). The Python IR helpers under `pypto.ir.op.system` are split instead: `syncall(core_type=...)` builds the hard form and `syncall_soft(core_type, args)` builds the soft form.

`system.task_invalid` returns [`ScalarType(DataType::TASK_ID)`](02-types.md#scalartype). It is the lowering target of the Python literal `None` when `None` appears in a TaskId position (a `deps=[None]` entry or a TaskId loop iter_arg seed) inside `with pl.manual_scope():` regions. There is no `system.task_id_of` op — producer task ids are obtained from the second tuple element returned by the `pl.submit(...)` parser construct, not from a builtin. Source: `src/ir/op/sync_ops/task.cpp`.

## CrossCoreOp: AIC↔AIV Communication

**Purpose**: Cross-core synchronization, data transfer, and pipe management between AIC (Cube) and AIV (Vector) kernels
**Type**: `UnknownType` (sync/push/init/buffer/free ops) or `TileType` passthrough (pop ops)
**Location**: `src/ir/op/tile_ops/cross_core.cpp` (tpush/tpop) and `src/ir/op/sync_ops/cross_core.cpp` (sync/tfree/pipe init/buffers)
**Python API**: `import pypto.language as pl` (promoted ops) or `from pypto.ir.op import tile, system`

### Explicit Event Synchronization

| Operation | Args | Description | Kwargs |
| --------- | ---- | ----------- | ------ |
| `system.sync_set` | 0 or 1 (`event_id_dyn`) | Emit `pto.sync.set` from one core type | `pipe`, static `event_id`, optional `ffts_mode`, optional `core_type` |
| `system.sync_wait` | 0 or 1 (`event_id_dyn`) | Emit `pto.sync.wait` on the peer core type | `pipe`, static `event_id`, optional `core_type` |
| `system.set_ffts` | 1 (`workspace`) | Declare the A3 FFTS setup required by explicit cross-core events | — |

Use `pl.system.sync_set(event_id, pipe=..., ffts_mode=...)` and `pl.system.sync_wait(event_id, pipe=...)` in explicitly typed AIC/AIV kernels. In a mixed InCore kernel, pass `core_type="aiv"` or `core_type="aic"` to retain each event operation on the intended lane when the kernel is expanded. On A3, call `pl.system.set_ffts(workspace)` in every participating AIC/AIV function before its first explicit event operation; `workspace` must be a one-dimensional `INT64` tensor with at least 256 elements and acts as the PTOAS setup operand. PyPTO's persistent runtime keeps the hardware FFTS control address installed, so generated runtime wrappers do not replace it with this operand. A5 does not require this setup. `event_id` may be an integer in the user-available range 0–13 or a dynamic `pl.Scalar[pl.INDEX]`; IDs 14 and 15 are reserved. `ffts_mode`, when supplied to `sync_set`, must be 0, 1, or 2. The author of a manual cross-core protocol is responsible for pairing event IDs and pipes. PyPTO's normal automatic intra-core dependency insertion remains enabled and uses the separate `set_flag`/`wait_flag` mechanism, so it does not allocate from these explicit cross-core event IDs.

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
| `system.aic_initialize_pipe` | 2 | Init cross-core pipe on Cube side (positional: `c2v_consumer_buf`, `v2c_consumer_buf`, i32 SSA) | `dir_mask`, `slot_size`, optional `slot_num`, optional `local_slot_num`, optional `id` |
| `system.aiv_initialize_pipe` | 2 | Init cross-core pipe on Vector side (positional: `c2v_consumer_buf`, `v2c_consumer_buf`, i32 SSA) | `dir_mask`, `slot_size`, optional `slot_num`, optional `local_slot_num`, optional `id` |

- `slot_num` (when set, must be > 0) pins the GM ring-buffer slot count; omit it to let PTOAS pick its default (8 unidirectional, 4 per direction bidirectional).
- `local_slot_num` (a2/a3 only, must be > 0 and `<= slot_num`) pins the local slot count.
- **Sizing the reserved/imported buffer is your responsibility and is architecture-dependent:** on **a3** use `slot_size * local_slot_num`; on **a5** use `slot_size * slot_num`.

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

Core definitions live in `include/pypto/core/common.h` and `include/pypto/ir/`; registry and type-inference implementations are in `src/ir/`, with operator implementations grouped under `src/ir/op/{tensor_ops,tile_ops,sync_ops}/`.
