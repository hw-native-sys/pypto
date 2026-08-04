# Operator System

Type-safe operator definitions with automatic type deduction, organized into modular categories (TensorOp, TileOp, SyncOp, CrossCoreOp).

## Operator Categories

| Category | Types | Use Case | File Location |
| -------- | ----- | -------- | ------------- |
| **TensorOp** | TensorType | N-D tensor operations with broadcasting | `src/ir/op/tensor_ops/` |
| **TileOp** | TileType | Hardware-optimized tile operations | `src/ir/op/tile_ops/` |
| **SyncOp** | UnknownType (barriers); ScalarType (task / launch queries) | Pipeline barriers, synchronization, TaskId and SPMD launch-shape queries | `src/ir/op/sync_ops/` |
| **CrossCoreOp** | UnknownType/TileType | AIC↔AIV cross-core communication | `src/ir/op/sync_ops/cross_core.cpp` |
| **PrefetchOp** | Opaque handles | Asynchronous GM→L2 cache prefetch | `src/ir/op/prefetch/prefetch_async.cpp` |

**Key Features**: Fluent API, automatic type deduction, kwargs for metadata, NumPy-style broadcasting, type promotion, dynamic dimensions (`kDynamicDim`)

## Type System

```cpp
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

For 2D `tile.matmul`, the physical boxed K dimensions must match. PTO derives
the contraction extent from the lhs valid K, so that extent may be smaller than
the rhs valid K but must be contained by it. `tile.matmul_acc` likewise keeps
exact physical M/N/K box compatibility while allowing the accumulator's valid
M/N rectangle and the rhs valid K extent to contain the smaller rectangle PTO
computes from lhs M/K and rhs N.

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

### MX block-scale matmul (Ascend950)

MX uses dedicated `LeftScale` / `RightScale` memory spaces and the `FP8E8M0`
scale dtype. PyPTO supports the host-prequant MXFP8 path on Ascend950 through
the `matmul_mx` family. `InsertMxScaleAddr` (after `InferTileMemorySpace`)
inserts the internal `tile.tget_scale_addr` bindings once operand memory
spaces are resolved.

| IR / DSL | Notes |
| -------- | ----- |
| `tile.load` of `pl.Tensor[..., pl.MX_A_ZZ \| pl.MX_B_NN]` | The source TensorLayout carries the MX scale GM layout. Dtype is FP8E8M0 or UINT8, `target_memory=Mat` is required, and strided sources are rejected. |
| `tile.move(..., target_memory=LeftScale/RightScale)` | Mat-to-Scale move with hardware-fixed row/row/32 (left) or col/col/32 (right) layout; the source Mat tile and layout overrides must match exactly. |
| `tile.create(..., target_memory=LeftScale/RightScale)` | Not supported; load MX scale data into Mat and then move it into scale memory. |
| `tile.matmul_mx` / `pl.matmul_mx` | `Left, LeftScale, Right, RightScale → Acc`; data is `FP8E4M3FN` only; scale is `FP8E8M0`; physical `M % 16 == 0`, `K % 64 == 0`, `N % 32 == 0`; valid K must satisfy `ceil(validK/32) == ceil(physicalK/32)`. Alignment / scale-group checks run only for constant extents; symbolic dims skip the numeric checks and fall back to the declared scale tile geometry (later PTOAS still verifies). |
| `tile.matmul_mx_acc` / `pl.matmul_mx_acc` | `Acc, Left, LeftScale, Right, RightScale → Acc`; in-place through `set_output_reuses_input(0)`; accumulator physical and valid M/N must match the matmul output. |
| `tile.matmul_mx_bias` / `pl.matmul_mx_bias` | `Left, LeftScale, Right, RightScale, Bias → Acc`; bias is `[1, N]` FP32. |
| `tile.tget_scale_addr` | Compiler-generated A5 binding from `LeftScale↔Left` or `RightScale↔Right`; DPS in-place on `dst_scale`. Users write only the `matmul_mx` family. |

The canonical shape is `M=128, K=64, N=64`, with FP8E4M3FN data,
FP8E8M0 scales shaped `[128,2]` and `[2,64]`, and `mx_a_zz` / `mx_b_nn`
host layouts. Align M↑16, K↑64, N↑32 (fp8).

MX tensor subviews are a legacy limitation. `tensor.slice`, `tensor.reshape`,
`tensor.transpose`, `tensor.reinterpret_view`, and `tensor.view` reject
MX-layout sources because the hardware path cannot represent a subview base
offset. `pld.tile.remote_load` also rejects MX layouts until its complete scale
layout contract is implemented.

#### MX / Ascend950: pto-isa constraints

| Constraint | Detail |
| ---------- | ------ |
| Distinct scale buffers | Cube does not fold scales into Left/Right data. `TileType::ScaleLeft` / `ScaleRight` sidecars map to PyPTO `LeftScale` / `RightScale`. |
| Payload | Scale is `float8_e8m0_t` / `FP8E8M0`; MX data is **`FP8E4M3FN` only** (**rejects `FP8E5M2`**). Physical `K%64==0`, with `ceil(K/32)` scale groups and fractal 32. |
| Layouts | `mx_a_zz` is row-major ZZ; `mx_b_nn` is col-major NN; loads use `TLoadMxCube*` (AZZ2ZZ). |
| `TMov` `CommonCheckMX` | Allows UINT8 Mat → FP8E8M0 ScaleLeft/Right; canonical path: ui8 Mat reshape then ui8→f8 Scale. |
| Bind then fill | Fill **after** `GetScaleAddr(Left/Right)`; writing the provisional alloc address is orphaned once rebound. |
| Alignment | Physical `M%16==0`, `K%64==0`, and `N%32==0` (fp8); `DeduceTileMatMulMxType` enforces these for **constant** extents only. Symbolic dims skip numeric checks. |

#### MX / Ascend950: PTOAS constraints

| Constraint | Detail |
| ---------- | ------ |
| Single `loc=scaling` | PTOAS has no distinct left/right scale locations; EmitC recovers ScaleLeft/Right. |
| FP8E8M0 scaling dtype | UINT8 with `loc=scaling` is treated as Fixpipe scaling, so promote before entering LeftScale/RightScale. |
| No Mat↔Scaling `treshape` | Different locs; reshape stays in Mat (ui8), then `tmov` into scaling. |
| Shape-matched Mat→Scale `tmov` | Flat `[1,G]` must `treshape` to `[M,K/32]` (or B-side shape) first. |
| Order | PyPTO emits Mat→scaling `tmov` in source order; PTOAS `PTOA5NormalizeTMovPass` reorders `tget_scale_addr` before it (ISA bind-then-fill). |
| `#pto.layout` / mx load | `mx_a_zz` / `mx_b_nn` / …; this stage uses **host ZZ/NN** (AZZ2ZZ). |
| Coverage | `pto.tmatmul.mx` / `.acc` / `.bias` + `pto.tget_scale_addr`. |

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

  // `kwargs` is a vector of pairs, not a map — scan it to look a key up.
  auto find_kwarg = [&kwargs](const std::string& key) {
    return std::find_if(kwargs.begin(), kwargs.end(),
                        [&key](const auto& kv) { return kv.first == key; });
  };

  // Required kwargs — `cast` declares both `target_type` and `mode`, and codegen
  // reads `mode` unconditionally, so a missing one must fail here rather than
  // silently default to round_mode NONE.
  auto it = find_kwarg("target_type");
  CHECK(it != kwargs.end()) << "tensor.cast requires 'target_type'";
  DataType target = static_cast<DataType>(std::any_cast<int>(it->second));

  CHECK(find_kwarg("mode") != kwargs.end()) << "tensor.cast requires 'mode'";

  return std::make_shared<TensorType>(input->shape_, target);
}
```

A genuinely optional kwarg (one codegen reads with a fallback, such as `tile.log`'s
`high_precision`) is read via `Call::GetKwarg<T>(key, default_value)` instead of a
`CHECK` — see `include/pypto/ir/expr.h`.

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

**Operations:** `tensor.add/sub/mul/div` (element-wise with full N-D broadcasting), `tensor.maximum/minimum` (element-wise max/min; rhs may be tensor or scalar — `ConvertTensorToTileOps` dispatches to `tile.maximum/minimum` or `tile.maximums/minimums` based on the rhs operand type), `tensor.set_validshape` (internal, update valid-shape metadata without data movement — compiler-generated only), `tensor.sort32` / `tensor.mrgsort_format1` / `tensor.mrgsort_format2` (sorting; tensor-level counterparts of `tile.sort32` / `tile.mrgsort` — converted to tile ops by `ConvertTensorToTileOps`), `tensor.gather` (per-dim indexing; MVP supports rank-2 inputs with `dim=-1`, lowered by `ConvertTensorToTileOps` with a backend-specific strategy — on A5 (Ascend950) a last-dim gather becomes a single full-tile `tile.gather` over flat element offsets `flat[i, j] = i * src_cols + index[i, j]`, first materializing a strided tile source (e.g. a `tile.slice` view) into a contiguous tile so the flat index addresses it correctly; on A2A3 (Ascend910B) it keeps the legacy per-row `tile.gather` loop where the column index equals the flat index within each 1-row slice), `tensor.gather_mask` (mask-pattern gather; tensor-level counterpart of `tile.gather_mask`, with optional same-bit-width `output_dtype` — see [Mask patterns](#mask-patterns)), `tensor.scatter` (column scatter; the column-wise inverse of `tensor.gather`, MVP supports rank-2 inputs with `dim=-1` — `out[b, index[b, k]] = src[b, k]`, `index` same shape as `src` — and lowers to `tile.scatter` via `ConvertTensorToTileOps`), `tensor.scatter_mask` (mask-pattern row-scatter; tensor-level counterpart of `tile.scatter_mask`, expands a compact `input` tensor into the mask-marked columns of `dst` — see [Mask patterns](#mask-patterns)), `tensor.ci` / `tensor.arange` (contiguous integer sequence generation; lowers to `tile.ci`; also exposed at top level as `pl.arange`), `tensor.and/ands/or/ors/xor/xors/not/shl/shls/shr/shrs` (integer-only bitwise and shift ops. These are the registered *IR* names; the Python spellings for the three whose leaf is a Python keyword carry a trailing underscore -- `tensor.and_`, `tensor.or_`, `tensor.not_` -- and the printer emits that form so IR round-trips as valid Python; tensor-level counterparts of the matching `tile.*` ops. Both operands of a tensor-tensor form must have the same shape — there is no `tile.row_expand_and`, so broadcasting is rejected at type deduction rather than failing later in the pass. `tensor.not` is int16/uint16 only, matching `tile.not`/TNOT. Shifts keep the lhs element type; `and`/`or`/`xor` promote across integer widths, as their tile counterparts do. `ConvertTensorToTileOps` lowers nine of them 1:1, and synthesizes the `pto.txor` scratch operand for `tensor.xor`/`tensor.xors` so tensor-level callers never supply a `tmp`)

`tensor.view` is a metadata-only zero-copy shape/layout reinterpret. It is registered as a `TensorOp` passthrough in `ConvertTensorToTileOps`; PTO in-core codegen lowers it to `pto.make_tensor_view` over the original base pointer. Targets require rank at least 1 (DN requires rank at least 2); orchestration shape reinterpret is ND-only and cannot also change layout. Shape reinterpretation of a partially valid source is limited to either a packed ND leading-dimension collapse to 2D or a contiguous-prefix linear collapse to `[1, product(shape)]`; both require an explicit target `valid_shape`. These forms preserve the source tensor kind and backing metadata.

`pl.reinterpret_view(data, dtype, *, shape=None)` dispatches to the equivalent `pl.tensor` or `pl.tile` operator and returns the same kind. It is a zero-copy view over exactly the same bytes, so `dtype` must differ and be one of signed/unsigned 8/16/32/64-bit integers, FP16, BF16, or FP32. With no `shape`, ND/row-major scales the last axis and DN/col-major scales the penultimate axis by the source/target byte-width ratio. An explicit shape must be byte-equivalent and fully static unless it is provably identical to the auto-inferred shape; a partial `valid_shape` only permits that auto-equivalent shape. Zero/null padding metadata is preserved, while dtype-dependent max/min padding is cleared. The initial executable path supports packed ND in-core tensors and packed flat (`none_box`) row/col-major tiles; DN tensor inference is available but Tensor-to-Tile lowering rejects it, and orchestration tensors are unsupported.

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
| - | `tile.move` | Move a tile between memory spaces (`target_memory`) — see [Result view of tile.move](#result-view-of-tilemove) |
| **Element-wise** | `tile.add/sub/mul/div` | Tile-Tile operations |
| - | `tile.adds/subs/muls/divs` | Tile-Scalar operations. A **constant** scalar operand adopts the tile's element dtype (a bare int literal is otherwise parsed as `index`, which no `pto.t*s` op accepts) — except a float literal on an integer tile, which keeps FP32 so promotion is preserved. An explicit `pl.const(v, dtype)` is a deliberate annotation and is left as-is, as is any non-constant expression; a non-constant `index` scalar (loop var, `pl.dim`) is rejected — convert it with `pl.cast`. Same rule for `tensor.*s`. |
| **Unary** | `tile.sqrt` | Element-wise square root |
| **Transform** | `tile.slice` | Extract a sub-tile with static shape, optional dynamic valid_shape, and optional `drop_dims` (numpy-style rank reduction over static unit axes; result clamped to a 2D minimum) |
| - | `tile.extract` | Extract a sub-tile from `src` at `(index_row, index_col)` — ISA TEXTRACT Variant 1 (Mat→Left/Right, Acc→Mat). The result's layout comes from `target_memory`'s implicit view, except `Left`/`Right`, which take the TEXTRACT-side L0 formats (these differ from `tile.move`'s TMOV-side ones) |
| - | `tile.reshape` | Reshape tile to new dimensions (element count must match). Carries the source's `valid_shape` through without widening it — see [Reshape and the valid region](#reshape-and-the-valid-region) |
| - | `tile.reinterpret_view` | Zero-copy view with a different dtype and the same exact bytes; optional shape uses layout-aware inference (packed flat tiles only) |
| - | `tile.transpose` | Swap two axes of a tile |
| - | `tile.set_validshape` | Update valid-shape metadata without data movement |
| - | `tile.ci` | Generate contiguous integer sequence (start + k / start - k); dtype ∈ {INT16, INT32}; innermost dim != 1 |
| - | `tile.tri` | Generate a lower/upper triangular 0/1 mask with an INT32 diagonal offset; supports an optional partial `valid_shape`; maps to `pto.ttri`. |
| **Reduction** | `tile.row_*` / `tile.col_*` | Direction-specific reduction (`row_sum`/`row_max`/`row_min`/`row_prod` collapse the last axis; `col_*` collapse axis 0). There is no axis-parameterized reduction — the ISA has only direction-specific intrinsics (`pto.trowsum`, `pto.tcolsum`, …) |
| **Gather** | `tile.gatherb` | Gather 32-byte source blocks. Each UINT32 offset selects one block; each offset column expands to `32 / sizeof(output_dtype)` output elements, and valid shape expands identically. `output_dtype` defaults to the source dtype and may select another supported byte interpretation. Offset rows contain a positive multiple of eight entries. A sliced source must have a byte address provably aligned to 32 bytes; dynamic column offsets are rejected, while dynamic row offsets remain valid when the physical row stride preserves alignment. Maps to `pto.tgatherb`. |
| - | `tile.mgather` | Gather from a GM tensor into a fresh Vec or Mat tile. Vec output uses an INT32 index tile (`[1,R]`, or A5 `[R,1]`); Mat output uses ND-layout GM source and INT32 index tensors plus canonical NZ layout, with physical rows aligned to 16 and columns aligned to `C0 = 32 / sizeof(dtype)`. Mat output accepts a smaller 2D `valid_shape` for padded tails. `coalesce="row"` gathers complete rows, while `"elem"` flat-indexes elements and requires a same-dtype, contiguous-ND GM `scratch` tensor with at least as many elements as the physical output. `gather_oob` selects `undefined`, `clamp`, `wrap`, or `zero`. Payload dtypes are I8/U8/I16/U16/I32/U32/FP16/BF16/FP32, plus the A5-only FP8E4M3FN/FP8E5M2/HF8 forms. |
| **Scatter** | `tile.scatter` | Row-scatter `src` into `dst` at per-row indices (`pto.tscatter` index form; DPS — `dst` is in/out, the result aliases `dst`). `src`/`dst` dtype ∈ {I8, I16, I32, FP16, FP32, BF16}; `indexes` dtype ∈ {I16, I32}; element-size matching rule: 4-byte dst ↔ INT32, 2-byte dst ↔ INT16, 1-byte dst ↔ INT16. |
| - | `tile.scatter_mask` | Mask-pattern row-scatter: write each `src` row into the mask-marked columns of `dst` (DPS — `dst` is in/out). A PyPTO codegen form lowered to a `pto.tscatter` mask emission — **not** a distinct pto-isa instruction (unlike `tile.gather_mask`). See [Mask patterns](#mask-patterns). |

`tile.reshape` preserves dtype, element count, and the source's valid region (see below); `tile.reinterpret_view(data, dtype, *, shape=None)` changes dtype while preserving exact byte size. Without `shape`, it scales the physically contiguous axis using the source/target dtype byte widths and tile layout. Under PTOAS memory planning, it lowers to the aliasing PTO `treshape` primitive for both same-shape and width-changing views.

### Result view of `tile.move`

The deduced result `TileView` splits by field:

| Field | Source of the result value |
| ----- | -------------------------- |
| `blayout` / `slayout` | The **destination** space's implicit layout wherever it has one of its own (`Mat`, `Acc`, `Left`, `Right`, `LeftScale`, `RightScale`); for the flat spaces (`Vec`, `Bias`, …) the source tile's effective layout carries over. A `blayout` / `slayout` kwarg overrides either |
| `fractal` | The **destination** space's boxing granularity, never the source's: `Acc` (L0C, NZ-boxed) is 1024, MX scale tiles are 32, everything else 512 |
| `valid_shape` / `pad` | Carried over from the source |
| `stride` / `start_offset` | Dropped — the destination is a dense buffer |

The layout comes from the destination because it describes how that buffer is
boxed; `tile_view_semantics::GetImplicitTileLayout` supplies it. `Right` needs a
local override — L0B requires `blayout=row_major` even for an `[N, 1]` shape,
whose implicit `blayout` is `col_major`.

`tile.move` stamps the destination `memory_space` itself (see the `TileType`
contract in [Types](02-types.md#tiletype)), so a result view matching the
destination's implicit view collapses to `nullopt` — the same per-space view
[`InferTileMemorySpace`](../passes/18-infer_tile_memory_space.md) refreshes a
retyped tile to.

### Reshape and the valid region

A reshape is a zero-copy view, so it cannot invent data: `tensor.reshape` and
`tile.reshape` share one rule that carries the source's `valid_shape` into the
target shape and never widens it. A valid region is an origin-anchored box, so
not every source region survives a repartition — the rule maps what it can:

| Source region | Result |
| ------------- | ------ |
| Fully valid | `new_shape` — canonicalized away, so no view survives and no existing program changes |
| Provably empty | An all-zero box |
| Only full unit axes added / removed | Surviving axes map 1:1; an arbitrary rectangle is preserved exactly |
| A contiguous flat prefix | The rectangle of `new_shape` spanning those same cells, if one exists |
| Anything else | **Rejected** — `valid_shape` cannot describe the reshaped region |

So `[8, 16]` valid `[5, 16]` (a flat prefix of 80 cells) maps to `[16, 8]` valid
`[10, 8]` and to `[128]` valid `[80]`, while `[4, 32]` is rejected — 80 cells is
not a whole number of 32-wide rows. `[1, 8, 16]` valid `[1, 8, 5]` is not a flat
prefix at all, yet `[8, 16]` valid `[8, 5]` is exact, because dropping a full
unit axis keeps rows as rows. `tensor.reshape`'s optional third `valid_shape`
operand may only *narrow* the derived region, never claim data outside it.

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

**Purpose**: Hardware synchronization and barriers, plus the TaskId and SPMD launch-shape queries sharing the `system.` namespace
**Type**: `UnknownType` (no return, used in `EvalStmt`) for barriers; `ScalarType` for the value-binding query ops (`task_invalid`, `task_is_valid`, `available_cluster_count`, `available_aiv_count`)
**Location**: `src/ir/op/sync_ops/` — `sync.cpp` (barriers), `task.cpp` (TaskId), `launch.cpp` (launch-shape queries)
**Python API**: `from pypto.ir.op import system`

| Operation | Description | Kwargs |
| --------- | ----------- | ------ |
| `system.bar_all` | Global barrier (lowers to `pto.barrier <PIPE_ALL>`) | None |
| `system.bar_v` | Vector barrier (lowers to `pto.barrier <PIPE_V>`) | None |
| `system.bar_m` | Matrix barrier (lowers to `pto.barrier <PIPE_M>`) | None |
| `system.fence` | Memory barrier over global memory (lowers to `pto.fence.barrier_all #pto.fence_scope<gm>`) | None |
| `system.cacheinvalid` | Invalidate the cache lines backing a tensor sub-region. Args: `tensor`, `shapes` (N-D), `offsets` (N-D). Every region size — a single element included — lowers to `pto.partition_view` + `pto.cmo.cacheinvalid %payload_view single_cache_line : !pto.partition_tensor_view<...>` | None |
| `system.syncall` | Cross-core all-participant barrier (`pto::SYNCALL`). `mode="hard"` (FFTS, no operands) or `mode="soft"` (GM-polling, operands) | `core_type` (`"aiv_only"` \| `"aic_only"` \| `"mix"`), `mode` (`"hard"` \| `"soft"`) |
| `system.sync_src` | Set sync flag | `set_pipe`, `wait_pipe`, `event_id` |
| `system.sync_dst` | Wait sync flag | `set_pipe`, `wait_pipe`, `event_id` |
| `system.task_invalid` | Sentinel `PTO2TaskId::invalid()` — "no producer" seed for a TaskId carry | None |
| `system.task_is_valid` | Test whether a `TASK_ID` value is a valid (non-sentinel) handle | None; sole positional arg is the TaskId Var |
| `system.available_cluster_count` | This run's MIX cluster (= AIC) count, read from the device. Result `Scalar[INT32]` | None |
| `system.available_aiv_count` | This run's standalone AIV core count, read from the device. Result `Scalar[INT32]` | None |

`system.syncall` has two modes. The **hard** form (`mode="hard"`, default) emits an FFTS barrier that waits for **all** physical cores of the selected `core_type`; the kernel must be launched at full occupancy (one block per physical core) **and with `sync_start=True`** (so all blocks are co-resident — a non-sync_start launch may dispatch blocks in waves and deadlock the barrier), or it deadlocks (AICore error 507018). The **soft** form (`mode="soft"`) polls a shared GM workspace and so works at **partial** occupancy. `gm_workspace` is a shared, zero-initialized GM `INT32` tensor with `used_cores * 8` slots (pass it as a kernel parameter so all blocks share one buffer); the scratch tile(s) are compiler-synthesized local staging buffers; `used_cores` is the participant count. Soft mode is supported for every `core_type`, with operands that vary by participant set:

- `aiv_only`: `[gm_workspace, ub_scratch, used_cores]` — one UB (Vec) staging tile.
- `aic_only`: `[gm_workspace, l1_scratch, used_cores]` — one flat L1 (Mat, `slayout=none_box`) staging tile.
- `mix`: `[gm_workspace, ub_scratch, l1_scratch, used_cores]` — both a UB and a flat L1 tile. The barrier rendezvouses AIC + AIV cores, so `used_cores` is the *total* participant count (AIC blocks + AIV subblocks). The op is duplicated onto both the cube and vector lanes; each lane uses its own tile (the other is dead), matching pto-isa's soft-mix lowering.

The flat L1 staging tile is created via `pl.tile.create(..., target_memory=pl.Mem.Mat, flat_layout=True)`, which keeps the contiguous `slayout=none_box` layout (a normal boxed NZ Mat tile would mis-place the 8-int32 counter slots).

The unified `mode=` keyword API (`mode="hard"` / `mode="soft"`) is the **DSL** surface (`pl.system.syncall`). The Python IR helpers under `pypto.ir.op.system` are split instead: `syncall(core_type=...)` builds the hard form and `syncall_soft(core_type, args)` builds the soft form.

`system.available_cluster_count` / `system.available_aiv_count` are the SPMD **launch-shape queries**: pass one as `pl.spmd(...)`'s `core_num` so the launch sizes itself on the device the run lands on. Orchestration codegen lowers them to `rt_available_cluster_count()` / `rt_available_aiv_count()`. Use the cluster count for a mixed (AIC+AIV) or cube-only kernel — one block per core-group — and the AIV count for a vector-only kernel. This is the only launch width that stays at full occupancy across devices, which the hard `system.syncall` requires; the `HardSyncallOccupancy` verifier accepts these widths without a count comparison and rejects the query for the *other* core type. Pass the call inline (`pl.spmd(pl.system.available_cluster_count())`) rather than binding it to a name first — a name reaches the outlined `Spmd` wrapper as a variable defined in the caller, which the IR printer cannot re-parse. Source: `src/ir/op/sync_ops/launch.cpp`.

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

## PrefetchOp: Asynchronous GM→L2 Prefetch

A latency-hiding cache hint. `async_prefetch` starts an SDMA-backed pull of a
global-memory region into L2 while unrelated compute proceeds; `wait` blocks
until it lands. The prefetch changes no tensor values — a kernel is numerically
identical with or without it, so only performance differs.

Unlike most PTO intrinsics, `TPREFETCH_ASYNC` carries no implicit wait-event
synchronization, so completion is explicit via an event/session pair.

### Operations

| DSL | Operands | Result | PTOAS op |
| --- | -------- | ------ | -------- |
| `pl.prefetch.make_context()` | None | `PrefetchAsyncContextType` | `pto.make_prefetch_async_context` |
| `pl.prefetch.async_prefetch(src, ctx)` | GM Tensor, context | `AsyncEventType` | `pto.tprefetch_async` |
| `pl.prefetch.session(ctx)` | context | `AsyncSessionType` | `pto.get_prefetch_async_session` |
| `pl.prefetch.wait(evt, session)` | event, session | `BOOL` scalar | `pto.comm.wait_async_event` |

The three result types are opaque singleton markers (no shape, no buffer), in
the same family as `CommCtxType`. The SDMA workspace is not a program operand:
the runtime owns it, and codegen injects a hidden pointer into prefetch kernels.

### Constraints

- `src` must be a **flat contiguous logical-1D GM** region: a fully static shape
  whose dimensions are all `1` except the last (`[N]`, `[1, N]`, `[1, 1, N]`).
  This mirrors the PTOAS `TPrefetchAsyncOp::verify()` check, so a shape mistake
  fails at PyPTO IR construction rather than at PTOAS verification.

### Example Usage

```python
@pl.program
class PrefetchExample:
    @pl.function(type=pl.FunctionType.InCore)
    def main(
        self, x: pl.Tensor[[1, 4096], pl.FP32],
        out: pl.Tensor[[1, 128], pl.FP32],
    ) -> pl.Tensor[[1, 128], pl.FP32]:
        ctx = pl.prefetch.make_context()
        evt = pl.prefetch.async_prefetch(x, ctx)     # warms L2, does not block
        session = pl.prefetch.session(ctx)
        # ... unrelated compute overlaps the prefetch ...
        pl.prefetch.wait(evt, session)               # x is now resident in L2
        tile = pl.load(x, [0, 0], [1, 128])
        return pl.store(tile, [0, 0], out)
```

**Core placement**: this family is **AIV-only**. `TPREFETCH_ASYNC` drives its
SDMA `tmpBuf` from a Vec(UB) scratch tile held inside `PrefetchAsyncContext`
(pto-isa static_asserts `ScratchTile::Loc == TileType::Vec`), and UB lives on
the vector core. The ops declare `CoreAffinity::VECTOR`, so in a mixed kernel
`ExpandMixedKernel` keeps them on the vector lane — they are neither placed on
nor duplicated onto the cube lane.

**Runtime ownership and support**: normal one-shot execution reads the generated
artifact's SDMA requirement and automatically constructs an enabled worker. No
workspace appears in the user, orchestration, or runtime tensor signature. For
an explicitly reused L2 worker, opt in when constructing it:

```python
with ChipWorker(
    config=RunConfig(platform="a2a3", device_id=0), enable_sdma=True
):
    compiled(a, out, config=cfg)
```

The current runtime-provisioned execution path is covered only on onboard a2a3.
An enabled worker on simulator, a5, or another runtime without an SDMA provider
fails during runtime initialization. PyPTO does not allocate a fallback
workspace or silently turn a requested prefetch into a no-op. See
`tests/st/runtime/ops/test_prefetch_async.py` for the a2a3 system test.

## File Organization

| Directory/File | Contents |
| -------------- | -------- |
| `src/ir/op/type_inference.cpp` | Shared type inference utilities |
| `tensor_ops/elementwise.cpp` | TensorOp: add, sub, mul, div |
| `tile_ops/matmul.cpp` | TileOp: matmul, gemv |
| `tile_ops/matmul_mx.cpp` | TileOp: matmul_mx, matmul_mx_acc, matmul_mx_bias, internal tget_scale_addr binding |
| `tile_ops/memory.cpp` | TileOp: load, store, read, get_block_idx |
| `tile_ops/elementwise.cpp` | TileOp: add, mul, div, adds, muls, etc. |
| `tile_ops/reduction.cpp` | TileOp: sum (with axis, keepdim) |
| `tile_ops/unary.cpp` | TileOp: sqrt |
| `sync_ops/sync.cpp` | SyncOp: sync_src, sync_dst, barriers |
| `sync_ops/task.cpp` | SyncOp: TaskId sentinel and predicate |
| `sync_ops/launch.cpp` | SyncOp: SPMD launch-shape queries |
| `sync_ops/cross_core.cpp` | CrossCoreOp: tpush, tpop, pipe init, buffers |
| `prefetch/prefetch_async.cpp` | PrefetchOp: make_context, async_prefetch, session, wait |

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
