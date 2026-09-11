# Tensor and Tile Operators

Part of the [Operator System](05-operators.md) reference.

## TensorOp: N-Dimensional Tensor Operations

**Purpose**: General N-dimensional tensors with full broadcasting
**Type**: `TensorType` (arbitrary dimensions)
**Location**: `src/ir/op/tensor_ops/`
**Python API**: `from pypto.ir.op import tensor`

**Operations:** `tensor.add/sub/mul/div` (element-wise with full N-D broadcasting), `tensor.maximum/minimum` (element-wise max/min; rhs may be tensor or scalar — `ConvertTensorToTileOps` dispatches to `tile.maximum/minimum` or `tile.maximums/minimums` based on the rhs operand type), `tensor.set_validshape` (update valid-shape metadata without data movement; also reachable as `pl.set_validshape`), `tensor.sort32` / `tensor.mrgsort_format1` / `tensor.mrgsort_format2` (sorting; tensor-level counterparts of `tile.sort32` / `tile.mrgsort` — converted to tile ops by `ConvertTensorToTileOps`), `tensor.gather` (axis indexing with `dim`, or flat element indexing without `dim`; see below), `tensor.gather_mask` (mask-pattern gather; tensor-level counterpart of `tile.gather_mask`, with optional same-bit-width `output_dtype` — see [Mask patterns](#mask-patterns)), `tensor.scatter` (column scatter; the column-wise inverse of `tensor.gather`, MVP supports rank-2 inputs with `dim=-1` — `out[b, index[b, k]] = src[b, k]`, `index` same shape as `src` — and lowers to `tile.scatter` via `ConvertTensorToTileOps`), `tensor.scatter_mask` (mask-pattern row-scatter; tensor-level counterpart of `tile.scatter_mask`, expands a compact `input` tensor into the mask-marked columns of `dst` — see [Mask patterns](#mask-patterns)), `tensor.ci` / `tensor.arange` (contiguous integer sequence generation; lowers to `tile.ci`; also exposed at top level as `pl.arange`), `tensor.and/ands/or/ors/xor/xors/not/shl/shls/shr/shrs` (integer-only bitwise and shift ops. These are the registered *IR* names; the Python spellings for the three whose leaf is a Python keyword carry a trailing underscore -- `tensor.and_`, `tensor.or_`, `tensor.not_` -- and the printer emits that form so IR round-trips as valid Python; tensor-level counterparts of the matching `tile.*` ops. Both operands of a tensor-tensor form must have the same shape — there is no `tile.row_expand_and`, so broadcasting is rejected at type deduction rather than failing later in the pass. `tensor.not` is int16/uint16 only, matching `tile.not`/TNOT. Shifts keep the lhs element type; `and`/`or`/`xor` require matching 8/16/32-bit operand dtypes, and scalar forms use the same-width signless `iN` encoding required by their tile lowering. `ConvertTensorToTileOps` lowers nine of them 1:1, and synthesizes the `pto.txor` scratch operand for `tensor.xor`/`tensor.xors` so tensor-level callers never supply a `tmp`)

`tensor.view` is a metadata-only zero-copy shape/layout reinterpret. It is registered as a `TensorOp` passthrough in `ConvertTensorToTileOps`; PTO in-core codegen lowers it to `pto.make_tensor_view` over the original base pointer. Targets require rank at least 1 (DN requires rank at least 2). Orchestration shape reinterpret is normally ND-only and cannot also change layout. FP8E8M0 dynamic scale storage additionally permits an equal-element-count shaped alias between packed ND and `MX_A_ZZ` or `MX_B_NN`; orchestration preserves the same runtime tensor without calling `reshape`. Shape reinterpretation of a partially valid source is limited to either a packed ND leading-dimension collapse to 2D or a contiguous-prefix linear collapse to `[1, product(shape)]`; both require an explicit target `valid_shape`. These forms preserve the source tensor kind and backing metadata.

Flat gather reuses `pl.gather(src, index=idx)` (equivalently `pl.gather(src, idx)`):
`out = src.reshape(-1)[idx]`. The index is a 2D INT32 tensor or tile and can be
computed inside the kernel. The output is a Tensor with the index's shape and
valid shape, and the source's dtype (FP16/FP32/INT16/INT32). Indices must refer to
valid source elements; negative indexing and bounds checking are not provided.
A contiguous ND source still in GM lowers directly to `tile.mgather`, without
loading the entire source into UB. Both GM sources and indices may be local
`DistributedTensor` windows. Tile indices must be in Vec with an unboxed row-major
layout; non-Vec indices must be moved first, and transposed/boxed index layouts
are rejected before codegen. Physical index columns
must be a positive static multiple of 16 for FP16/INT16 sources, or 8 for
FP32/INT32, so both index and output rows are 32-byte aligned. This restriction
applies even to a single row and is checked at the flat-gather boundary, not
deferred to codegen. Pad the physical index tensor and use `pl.set_validshape`
for a narrower valid region; its valid row/column counts need not be aligned.
For example, FP16 with eight valid columns uses physical indices `[1, 16]`:

```python
indices = pl.set_validshape(padded_indices, 1, 8)
values = pl.gather(src, index=indices)  # Physical [1, 16], valid [1, 8].
```

An on-chip source lowers to `tile.gather`
with compiler-managed scratch; it must be static 2D row-major Vec with 32-byte-aligned
rows (or one row). Strided on-chip windows are packed first: `tile.extract` for
floating point, exact integer `tile.adds(..., 0)` for INT16/INT32. Proven packed
computed sources are reused without a copy; unknown storage still gets packed.
Specifying `dim` retains axis indexing (rank 2/3, any axis), and mask/compare
forms are unchanged. See [gather lowering](../passes/11-convert_tensor_to_tile_ops.md#flat-gather-lowering).

For plain `TensorType` operands, the supported Tensor-scalar arithmetic
operators (`adds`, `subs`, `muls`, `divs`, `fmods`, and scalar `maximum` or
`minimum`) and bitwise/shift operators (`ands`, `ors`, `shls`, and `shrs`)
create fresh storage but cannot create valid data in padding. Their results
therefore preserve the tensor operand's effective `valid_shape` while dropping
source alias, layout, stride, and padding metadata. This matches the existing
Tile-scalar rule and keeps a ragged tail narrow through Tensor-to-Tile lowering.
Scalar comparison and XOR (`cmp` and `xors`) remain excluded.

The ordinary arithmetic Tensor-tensor operators (`add`, `sub`, `mul`, `div`,
`fmod`, `maximum`, and `minimum`) also preserve the effective `valid_shape`
when both operands have identical physical shapes and their effective valid
regions are provably equal. The same exact-region rule applies to `and`, `or`,
`shl`, and `shr`. It needs no broadcast-axis mapping and agrees with the
corresponding Tile result contract; the result remains fresh storage and
therefore inherits no alias, layout, stride, or padding metadata. Comparison,
XOR, `part_*`, broadcasting, different valid regions, and direct distributed
window operands are not covered by this rule because their current lowering or
combination contracts require separate handling.

`pl.reinterpret_view(data, dtype, *, shape=None)` dispatches to the equivalent `pl.tensor` or `pl.tile` operator and returns the same kind. It is a zero-copy view over exactly the same bytes. General reinterpretation supports signed/unsigned 8/16/32/64-bit integers, FP16, BF16, and FP32; MX lowering additionally permits only the byte-identical INT8↔FP8E4M3FN and UINT8↔FP8E8M0 pairs. With no `shape`, ND/row-major scales the last axis and DN/col-major scales the penultimate axis by the source/target byte-width ratio. An explicit shape must be byte-equivalent and fully static unless it is provably identical to the auto-inferred shape; a partial `valid_shape` only permits that auto-equivalent shape. Zero/null padding metadata is preserved, while dtype-dependent max/min padding is cleared. The initial executable path supports packed ND in-core tensors and packed flat (`none_box`) row/col-major tiles; DN tensor inference is available but Tensor-to-Tile lowering rejects it, and orchestration tensors are unsupported.

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
| **Quantization** | `tile.tquant_mx` / `pl.quant_mx` | Ascend950-only **MXFP8** block-32 dynamic quantization returning `{FP8E4M3FN quant, FP8E8M0 scale}`. `dtype` must be `FP8E4M3FN`. `group_axis` is PTOAS `grpAxis` (`1` = A-side `[M,K]`, `0` = B-side `[N,K]` with transpose). Public scale shapes are `[M,K/32]` / `[K/32,N]`; requires a full valid region and `K % 64 == 0` (plus axis1 `M % 16 == 0`, axis0 `N % 32 == 0`). [Pass 13](../passes/13-lower_composite_ops.md) emits grouped TQUANT plus X-to-ZZ TMOV. In a mixed task, results may feed `matmul_mx` directly through V2C. MXFP4 quant is deferred. |
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

On Ascend950, `quant_mx` and `matmul_mx` may share the same InCore mixed task.
The compiler transports both quantized data and the FP8E8M0 scale directly over
V2C while retaining the scale's logical fractal-32 layout.

`tile.reshape` preserves dtype, element count, and the source's valid region (see below); `tile.reinterpret_view(data, dtype, *, shape=None)` changes dtype while preserving exact byte size. Without `shape`, it scales the physically contiguous axis using the source/target dtype byte widths and tile layout. Under PTOAS memory planning, it lowers to the aliasing PTO `treshape` primitive for both same-shape and width-changing views.

### Result view of `tile.move`

The deduced result `TileView` splits by field:

| Field | Source of the result value |
| ----- | -------------------------- |
| `blayout` / `slayout` | The **destination** space's implicit layout wherever it has one of its own (`Mat`, `Acc`, `Left`, `Right`, `LeftScale`, `RightScale`); for the flat spaces (`Vec`, `Bias`, …) the source tile's effective layout carries over. A `blayout` / `slayout` kwarg overrides either |
| `fractal` | The **destination** space's boxing granularity: `Acc` (L0C, NZ-boxed) is 1024, MX scale tiles are 32, everything else 512. A byte-valued MX-scale Vec-to-Vec reorder or Vec-to-Mat cross-core staging move is the narrow exception: it preserves the source's 32-byte scale boxes |
| `valid_shape` / `pad` | Carried over from the source |
| `stride` / `start_offset` | Dropped — the destination is a dense buffer |

The layout comes from the destination because it describes how that buffer is
boxed; `tile_view_semantics::GetImplicitTileLayout` supplies it. `Right` needs a
local override — L0B requires `blayout=row_major` even for an `[N, 1]` shape,
whose implicit `blayout` is `col_major`.

`tile.move` stamps the destination `memory_space` itself (see the `TileType`
contract in [Types](02-types.md#tiletype)), so a result view matching the
destination's implicit view collapses to `nullopt` — the same per-space view
[`InferTileMemorySpace`](../passes/20-infer_tile_memory_space.md) refreshes a
retyped tile to.

`tile.move` is not in-place safe: within one memory space, its source and result
must resolve to distinct addresses. The PyPTO and DSA-RP planners enforce this
constraint, and baked-address PTO codegen reports an error if an explicit
MemRef binding or hand-built IR still presents a same-address move.

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
| One the target shape cuts the same way | The box of `new_shape` spanning those same cells, if one exists |
| Anything else | **Rejected** — `valid_shape` cannot describe the reshaped region |

The last rule reads the region as the **runs** of elements it fills.
Neighbouring source axes stay in one run while the lower one is fully valid or
the upper one is pinned to a single coordinate; anywhere else the upper axis's
stride survives into the region and cuts it. Each run then holds a flat prefix
of its own volume, and the region maps exactly when `new_shape` groups its own
dimensions into the same runs and each prefix falls on a dimension boundary
there. For a static region the rule is **exact**: it accepts if and only if some
box of `new_shape` denotes the very same cells.

So `[8, 16]` valid `[5, 16]` is the one-run case (a flat prefix of 80 cells): it
maps to `[16, 8]` valid `[10, 8]` and to `[128]` valid `[80]`, while `[4, 32]` is
rejected — 80 cells is not a whole number of 32-wide rows. `[2, 2, 2]` valid
`[2, 1, 2]` is the two-run case `2 | 4` — flat cells `{0, 1, 4, 5}`, no prefix at
all — which `[2, 4]` spells as valid `[2, 2]` and `[8]` cannot spell, having no
dimension boundary every 4 elements. `[8, 16]` valid `[8, 5]` cuts into `8 | 16`
and `[16, 8]` cannot regroup that way, so it is rejected. `tensor.reshape`'s
optional third `valid_shape` operand may only *narrow* the derived region, never
claim data outside it.

A symbolic extent narrows what the rule can prove, but does not by itself
reject. **Any** run may carry a symbolic *valid* extent through unchanged, onto a
dimension of its own run whose step is exactly that run's trailing volume — so
`[4, 2, 8]` valid `[v, 1, 8]` maps to `[4, 16]` valid `[v, 8]` even though it
cuts into the two runs `4 | 16`. What has to be static is the *physical* geometry
the region is measured against: the target extents, the extents below each run's
free axis, the free axis itself on the symbolic path (its dimension must be
provably wide enough), and — once the region cuts into more than one run — each
run's volume. Anything less is rejected rather than guessed.

An **identity** `tile.reshape` — one whose target shape equals the source's —
additionally keeps the source's layout triple (`blayout` / `slayout` / `fractal`) and its
resolved memory space, instead of re-deriving the layout from the shape. Re-deriving
yields the space-agnostic flat layout, which `NormalizeImplicitTileView` rescues only for
a view that collapses; an Acc box that is narrowed, padded, or declared `compact` never
collapses, so the flat layout would stick and its reader would walk L0C as a plain
row-major buffer (issue #2470).

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
