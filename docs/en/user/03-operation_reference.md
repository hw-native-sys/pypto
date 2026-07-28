# Operation Reference

All operations are accessed via `import pypto.language as pl`.

**Notation:** `T` = `Tensor` or `Tile` (unified dispatch). `IntLike` = `int | Scalar | Expr`. `Mem` = `MemorySpace` (short alias; both `pl.Mem` and `pl.MemorySpace` work).

## Unified Dispatch (`pl.*`)

Auto-selects between tensor and tile implementation based on input type.

| Name | Signature | Description |
| ---- | --------- | ----------- |
| `add` | `(lhs: T, rhs: T \| int \| float \| Scalar) -> T` | Element-wise addition |
| `sub` | `(lhs: T, rhs: T \| int \| float \| Scalar) -> T` | Element-wise subtraction |
| `mul` | `(lhs: T, rhs: T \| int \| float \| Scalar) -> T` | Element-wise multiplication |
| `div` | `(lhs: T, rhs: T \| int \| float \| Scalar) -> T` | Element-wise division |
| `part_add` | `(lhs: T, rhs: T) -> T` | Partial add (copies the only valid input) |
| `part_mul` | `(lhs: T, rhs: T) -> T` | Partial multiply (copies the only valid input) |
| `part_max` | `(lhs: T, rhs: T) -> T` | Partial max (copies the only valid input) |
| `part_min` | `(lhs: T, rhs: T) -> T` | Partial min (copies the only valid input) |
| `fmod` | `(lhs: T, rhs: T \| int \| float \| Scalar) -> T` | Floating-point remainder (`torch.fmod`) |
| `fmods` | `(lhs: T, rhs: int \| float \| Scalar) -> T` | Floating-point remainder with scalar |
| `maximum` | `(lhs: T, rhs: T) -> T` | Element-wise maximum |
| `exp` | `(input: T) -> T` | Element-wise exponential |
| `cast` | `(input: T, target_type: int \| DataType, mode="round") -> T` | Type cast (`mode`: none, rint, round, floor, ceil, trunc, odd) |
| `reshape` | `(input: T, shape: Sequence[IntLike]) -> T` | Reshape to new dimensions |
| `transpose` | `(input: T, axis1: int, axis2: int) -> T` | Swap two axes |
| `slice` | `(input: T, shape: Sequence[IntLike], offset: Sequence[IntLike]) -> T` | Slice with offset |
| `matmul` | `(lhs: T, rhs: T, out_dtype=None, a_trans=False, b_trans=False, c_matrix_nz=False) -> T` | Matrix multiplication |
| `matmul_acc` | `(acc: T, lhs: T, rhs: T, a_trans=False, b_trans=False) -> T` | Matrix multiply with accumulation: `acc += lhs @ rhs` |
| `row_max` | `(input: T, tmp_tile: Tile \| None = None) -> T` | Row-wise max (tile path requires `tmp_tile`) |
| `row_sum` | `(input: T, tmp_tile: Tile \| None = None) -> T` | Row-wise sum (tile path requires `tmp_tile`) |
| `row_prod` | `(input: T, tmp_tile: Tile \| None = None) -> T` | Row-wise product (tile path requires `tmp_tile`) |
| `col_sum` | `(input: T, tmp_tile: Tile \| None = None) -> T` | Column-wise sum. On Tile, passing `tmp_tile` activates binary-tree reduction; omitting it uses sequential reduction. Tensor input lowers to the sequential path. |
| `col_max` | `(input: T) -> T` | Column-wise max |
| `col_min` | `(input: T) -> T` | Column-wise min |
| `col_prod` | `(input: T) -> T` | Column-wise product |
| `row_argmax` | `(input: T, tmp_tile: Tile \| None = None) -> T` | Row-wise argmax (column index of per-row max, int32 output; tile path requires `tmp_tile`) |
| `row_argmin` | `(input: T, tmp_tile: Tile \| None = None) -> T` | Row-wise argmin (column index of per-row min, int32 output; tile path requires `tmp_tile`) |
| `col_argmax` | `(input: T, tmp_tile: Tile \| None = None) -> T` | Column-wise argmax (row index of per-column max, int32 output; tile path requires `tmp_tile`) |
| `col_argmin` | `(input: T, tmp_tile: Tile \| None = None) -> T` | Column-wise argmin (row index of per-column min, int32 output; tile path requires `tmp_tile`) |
| `rsqrt` | `(input: T, high_precision: bool = False) -> T` | Reciprocal square root; `high_precision=True` selects the high-precision path (tensor input only — tile callers must use `pl.tile.rsqrt(src, tmp=...)`) |
| `create` / `create_tile` | `(shape: Sequence[IntLike], dtype: DataType, target_memory: Mem) -> Tile` | Tile-only (promoted from `pl.tile.create`): create tile at specific memory space |
| `read` | `(src: T, offset: IntLike \| Sequence[IntLike]) -> Scalar` | Read scalar at indices (dispatched by source type). Sugar: `A[i, j]` |
| `write` | `(dst: T, offset: IntLike \| Sequence[IntLike], value: Scalar) -> Expr` | Write scalar at indices (dispatched by destination type). Sugar: `A[i, j] = v` |

## Tensor-Only (`pl.tensor.*`)

Operate on `Tensor` objects (DDR memory).

| Name | Signature | Description |
| ---- | --------- | ----------- |
| `create` / `create_tensor` | `(shape: Sequence[IntLike], dtype: DataType, layout: TensorLayout = None, init_value: int \| float \| None = None) -> Tensor` | Create a new tensor (optional `layout`, e.g. `pl.DN`, `pl.NZ`; `init_value` AICPU-pre-fills the buffer — `0` zeroes any dtype, non-zero needs an int / 32-bit-or-wider float dtype) |
| `read` | `(tensor: Tensor, indices: IntLike \| Sequence[IntLike]) -> Scalar` | Read scalar at indices. Sugar: `A[i, j]` |
| `write` | `(tensor: Tensor, indices: IntLike \| Sequence[IntLike], value: Scalar \| Expr) -> Expr` | Write scalar at indices. Sugar: `A[i, j] = v` |
| `dim` | `(tensor: Tensor, axis: int) -> Scalar` | Get dimension size (supports negative indexing) |
| `slice` | `(tensor: Tensor, shape: Sequence[IntLike], offset: Sequence[IntLike]) -> Tensor` | Slice. Sugar: `A[0:16, :]` |
| `reshape` | `(tensor: Tensor, shape: Sequence[IntLike]) -> Tensor` | Reshape |
| `view` | `(tensor: Tensor, shape: Sequence[IntLike] \| None = None, *, layout: TensorLayout \| None = None) -> Tensor` | Zero-copy reinterpret over the same storage; target rank must be at least 1 and DN requires rank at least 2. Orchestration shape reinterpret is ND-only and cannot also change layout |
| `transpose` | `(tensor: Tensor, axis1: int, axis2: int) -> Tensor` | Swap two axes |
| `assemble` | `(target: Tensor, source: Tensor, offset: Sequence[IntLike], *, atomic: AtomicType = AtomicType.None_) -> Tensor` | Write source into target at offset. Sugar (pre-SSA only): `target[i:i+H, j:j+W] = source`. `atomic=AtomicType.Add` accumulates instead of overwriting (split-K) — only valid when the target is a function output (global memory); non-deterministic FP, target must be pre-zeroed, dtypes fp32/bf16/fp16/int32/int16/int8 (bf16 requires the Ascend910B/A2/A3 profile) |
| `scatter_update` | `(input: Tensor, dim: int, index: Tensor, src: Tensor) -> Tensor` | Update rows of `input` at sparse positions given by `index` with values from `src`. `input`/`src`: 2D `[rows, d]` or 4D `[B, S, 1, d]`; `index`: 2D `[b, s]` integer. Only `dim=-2` is supported |
| `random` | `(key0, key1, counter0, counter1, counter2, counter3: int \| Scalar, shape: Sequence[IntLike], dtype: DataType = UINT32, rounds: int = 10) -> Tensor` | Counter-based (Philox/ChaCha-style) RNG; lowers to `tile.random`. Deterministic from the key + counter seeds. `dtype` ∈ {INT32, UINT32}; `rounds` ∈ {7, 10}. Top-level alias `pl.random`. **A5 only** |
| `add` | `(lhs: Tensor, rhs: Tensor \| int \| float \| Scalar) -> Tensor` | Element-wise add |
| `sub` | `(lhs: Tensor, rhs: Tensor \| int \| float \| Scalar) -> Tensor` | Element-wise subtract |
| `mul` | `(lhs: Tensor, rhs: Tensor \| int \| float \| Scalar) -> Tensor` | Element-wise multiply |
| `div` | `(lhs: Tensor, rhs: Tensor \| int \| float \| Scalar) -> Tensor` | Element-wise divide |
| `adds` | `(lhs: Tensor, rhs: int \| float \| Scalar) -> Tensor` | Add scalar |
| `subs` | `(lhs: Tensor, rhs: int \| float \| Scalar) -> Tensor` | Subtract scalar |
| `muls` | `(lhs: Tensor, rhs: int \| float \| Scalar) -> Tensor` | Multiply by scalar |
| `divs` | `(lhs: Tensor, rhs: int \| float \| Scalar) -> Tensor` | Divide by scalar |
| `part_add` | `(lhs: Tensor, rhs: Tensor) -> Tensor` | Partial add (copies the only valid input) |
| `part_mul` | `(lhs: Tensor, rhs: Tensor) -> Tensor` | Partial multiply (copies the only valid input) |
| `part_max` | `(lhs: Tensor, rhs: Tensor) -> Tensor` | Partial max (copies the only valid input) |
| `part_min` | `(lhs: Tensor, rhs: Tensor) -> Tensor` | Partial min (copies the only valid input) |
| `fmod` | `(lhs: Tensor, rhs: Tensor \| int \| float \| Scalar) -> Tensor` | Floating-point remainder (`torch.fmod`) |
| `fmods` | `(lhs: Tensor, rhs: int \| float \| Scalar) -> Tensor` | Floating-point remainder with scalar |
| `maximum` | `(lhs: Tensor, rhs: Tensor) -> Tensor` | Element-wise maximum |
| `row_max` | `(input: Tensor) -> Tensor` | Row-wise max reduction |
| `row_sum` | `(input: Tensor) -> Tensor` | Row-wise sum reduction |
| `row_prod` | `(input: Tensor) -> Tensor` | Row-wise product reduction |
| `col_sum` | `(input: Tensor) -> Tensor` | Column-wise sum reduction (reduces along axis=-2) |
| `col_max` | `(input: Tensor) -> Tensor` | Column-wise max reduction (reduces along axis=-2) |
| `col_min` | `(input: Tensor) -> Tensor` | Column-wise min reduction (reduces along axis=-2) |
| `col_prod` | `(input: Tensor) -> Tensor` | Column-wise product reduction (reduces along axis=-2) |
| `row_argmax` | `(input: Tensor) -> Tensor` | Row-wise argmax reduction (int32 index output) |
| `row_argmin` | `(input: Tensor) -> Tensor` | Row-wise argmin reduction (int32 index output) |
| `col_argmax` | `(input: Tensor) -> Tensor` | Column-wise argmax reduction (reduces along axis=-2, int32 index output) |
| `col_argmin` | `(input: Tensor) -> Tensor` | Column-wise argmin reduction (reduces along axis=-2, int32 index output) |
| `rsqrt` | `(input: Tensor, high_precision: bool = False) -> Tensor` | Element-wise reciprocal square root; `high_precision=True` allocates a scratch tile during lowering for the higher-precision PTO path (requires static tile shape, same constraint as `row_max`/`row_sum`) |
| `exp` | `(input: Tensor) -> Tensor` | Element-wise exponential |
| `cast` | `(input: Tensor, target_type: DataType, mode="round") -> Tensor` | Type cast |
| `matmul` | `(lhs: Tensor, rhs: Tensor, out_dtype=None, a_trans=False, b_trans=False, c_matrix_nz=False) -> Tensor` | Matrix multiplication |
| `matmul_acc` | `(acc: Tensor, lhs: Tensor, rhs: Tensor, a_trans=False, b_trans=False) -> Tensor` | Matrix multiply with accumulation: `acc += lhs @ rhs` |

## Data Movement (`pl.tile.*`)

Transfer data between memory hierarchy levels.

| Name | Signature | Description |
| ---- | --------- | ----------- |
| `load` | `(tensor: Tensor, offsets: Sequence[IntLike], shapes: Sequence[IntLike], target_memory: Mem = Mem.Vec) -> Tile` | DDR → on-chip tile. Both `offsets` and `shapes` use the source tensor's coordinate system. For a transposed matmul operand, apply `transpose_view` to the loaded tile. |
| `store` | `(tile: Tile, offsets: Sequence[IntLike], output_tensor: Tensor, *, atomic: AtomicType = AtomicType.None_) -> Tensor` | Tile → DDR (pipe inferred from source memory). `atomic=AtomicType.Add` accumulates the tile into existing DDR contents (split-K); non-deterministic FP, destination must be pre-zeroed, dtypes fp32/bf16/fp16/int32/int16/int8 (bf16 requires the Ascend910B/A2/A3 profile) |
| `assemble` | `(target: Tile, source: Tile, offset: Sequence[IntLike]) -> Tile` | Write source tile into target at offset. Sugar (pre-SSA only): `target[i:i+H, j:j+W] = source` |
| `scatter_update` | `(input: Tile, dim: int, index: Tile, src: Tile) -> Tile` | Update rows of `input` tile at sparse positions given by `index` tile with values from `src` tile. `input`/`src`: 2D `[rows, d]` or 4D `[B, S, 1, d]`; `index`: 2D `[b, s]` integer. Lowered to `tile.scatter` (pto.tscatter, whole-row flat indices). Only `dim=-2` is supported |
| `read` | `(tile: Tile, indices: IntLike \| Sequence[IntLike]) -> Scalar` | Read scalar at indices. Sugar: `A[i, j]` |
| `write` | `(tile: Tile, indices: IntLike \| Sequence[IntLike], value: Scalar \| Expr) -> Expr` | Write scalar at indices. Sugar: `A[i, j] = v` |
| `move` | `(tile: Tile, target_memory: Mem) -> Tile` | Move tile between memory levels (including Vec→Vec) |
| `create` | `(shape: Sequence[IntLike], dtype: DataType, target_memory: Mem = Mem.Vec) -> Tile` | Create tile at memory space |
| `full` | `(shape: list[int], dtype: DataType, value: int \| float) -> Tile` | Create tile filled with constant |
| `random` | `(key0, key1, counter0, counter1, counter2, counter3: int \| Scalar, shape: Sequence[int], valid_shape: Sequence[int] \| None = None, dtype: DataType = UINT32, rounds: int = 10) -> Tile` | Fill a tile with counter-based (Philox/ChaCha-style) pseudo-random values from a 64-bit key + 128-bit counter. Deterministic: same seeds → same tile. Optional `valid_shape` (each dim `<= shape`) writes only the valid rows/cols, leaving the rest untouched. `dtype` ∈ {INT32, UINT32}; `rounds` ∈ {7, 10}. 2D shape only. **A5 only** (`pto.trandom`) |
| `fillpad` | `(input: Tensor \| Tile, pad_value: PadValue \| int \| float = PadValue.zero) -> Tensor \| Tile` | Fill invalid view elements using the requested pad value; accepts the `PadValue.zero/max/min` enum or the literal sugars `0`, `0.0`, `math.inf`, `-math.inf` (other values raise). Tensor inputs lower to tile fillpad in InCore code |
| `fillpad_expand` | `(input: Tensor \| Tile, shape: Sequence[IntLike], pad_value: PadValue \| int \| float = PadValue.zero) -> Tensor \| Tile` | Like `fillpad` but the destination `shape` may be **larger** than the source in either dimension: the source's valid region is copied to the top-left and every other element is filled with `pad_value`. Each destination dimension must be `>=` the source. Tensor inputs lower to tile fillpad_expand in InCore code |
| `get_block_idx` | `() -> Scalar` | Get current hardware block index (UINT64) |

## Tile Arithmetic (`pl.tile.*`)

### Binary (Tile × Tile)

| Name | Signature | Description |
| ---- | --------- | ----------- |
| `add` | `(lhs: Tile, rhs: Tile) -> Tile` | Element-wise add |
| `sub` | `(lhs: Tile, rhs: Tile) -> Tile` | Element-wise subtract |
| `mul` | `(lhs: Tile, rhs: Tile) -> Tile` | Element-wise multiply |
| `div` | `(lhs: Tile, rhs: Tile) -> Tile` | Element-wise divide |
| `maximum` | `(lhs: Tile, rhs: Tile) -> Tile` | Element-wise maximum |
| `minimum` | `(lhs: Tile, rhs: Tile) -> Tile` | Element-wise minimum |

### Binary (Tile × Scalar)

| Name | Signature | Description |
| ---- | --------- | ----------- |
| `adds` | `(lhs: Tile, rhs: int \| float \| Scalar) -> Tile` | Add scalar |
| `subs` | `(lhs: Tile, rhs: int \| float \| Scalar) -> Tile` | Subtract scalar |
| `muls` | `(lhs: Tile, rhs: int \| float \| Scalar) -> Tile` | Multiply by scalar |
| `divs` | `(lhs: Tile, rhs: int \| float \| Scalar) -> Tile` | Divide by scalar |
| `maximums` | `(lhs: Tile, rhs: int \| float \| Scalar) -> Tile` | Max with scalar |
| `minimums` | `(lhs: Tile, rhs: int \| float \| Scalar) -> Tile` | Min with scalar |

### Three-Input Arithmetic

| Name | Signature | Description |
| ---- | --------- | ----------- |
| `addc` | `(lhs: Tile, rhs: Tile, rhs2: Tile) -> Tile` | `lhs + rhs + rhs2` |
| `subc` | `(lhs: Tile, rhs: Tile, rhs2: Tile) -> Tile` | `lhs - rhs - rhs2` |
| `addsc` | `(lhs: Tile, rhs: int \| float \| Scalar, rhs2: Tile) -> Tile` | `lhs + scalar + rhs2` |
| `subsc` | `(lhs: Tile, rhs: int \| float \| Scalar, rhs2: Tile) -> Tile` | `lhs - scalar - rhs2` |

## Tile Math (`pl.tile.*`)

| Name | Signature | Description |
| ---- | --------- | ----------- |
| `neg` | `(tile: Tile) -> Tile` | Negate |
| `exp` | `(tile: Tile) -> Tile` | Exponential |
| `sqrt` | `(tile: Tile) -> Tile` | Square root |
| `rsqrt` | `(tile: Tile, tmp: Tile \| None = None) -> Tile` | Reciprocal square root; passing `tmp` (same shape/dtype as `tile`) selects the high-precision PTO lowering |
| `recip` | `(tile: Tile) -> Tile` | Reciprocal (1/x) |
| `log` | `(tile: Tile) -> Tile` | Natural logarithm |
| `abs` | `(tile: Tile) -> Tile` | Absolute value |

## Tile Reductions (`pl.tile.*`)

| Name | Signature | Description |
| ---- | --------- | ----------- |
| `row_max` | `(tile: Tile, tmp_tile: Tile) -> Tile` | Row-wise max (requires tmp buffer) |
| `row_sum` | `(tile: Tile, tmp_tile: Tile) -> Tile` | Row-wise sum (requires tmp buffer) |
| `row_min` | `(tile: Tile, tmp_tile: Tile) -> Tile` | Row-wise min (requires tmp buffer) |
| `row_prod` | `(tile: Tile, tmp_tile: Tile) -> Tile` | Row-wise product (requires tmp buffer) |
| `col_sum` | `(tile: Tile, tmp_tile: Tile \| None = None) -> Tile` | Column-wise sum. Passing `tmp_tile` activates binary-tree reduction; omitting it uses sequential reduction. |
| `col_max` | `(tile: Tile) -> Tile` | Column-wise max |
| `col_min` | `(tile: Tile) -> Tile` | Column-wise min |
| `col_prod` | `(tile: Tile) -> Tile` | Column-wise product |
| `row_argmax` | `(tile: Tile, tmp_tile: Tile) -> Tile` | Row-wise argmax, column index of per-row max (requires tmp buffer, int32 output) |
| `row_argmin` | `(tile: Tile, tmp_tile: Tile) -> Tile` | Row-wise argmin, column index of per-row min (requires tmp buffer, int32 output) |
| `col_argmax` | `(tile: Tile, tmp_tile: Tile) -> Tile` | Column-wise argmax, row index of per-column max (requires tmp buffer, int32 output) |
| `col_argmin` | `(tile: Tile, tmp_tile: Tile) -> Tile` | Column-wise argmin, row index of per-column min (requires tmp buffer, int32 output) |

## Linear Algebra (`pl.tile.*`)

| Name | Signature | Description |
| ---- | --------- | ----------- |
| `matmul` | `(lhs: Tile, rhs: Tile) -> Tile` | Matrix multiply: `C = A @ B` |
| `matmul_acc` | `(acc: Tile, lhs: Tile, rhs: Tile) -> Tile` | `acc += A @ B` |
| `matmul_bias` | `(lhs: Tile, rhs: Tile, bias: Tile) -> Tile` | `C = A @ B + bias` |
| `gemv` | `(lhs: Tile, rhs: Tile) -> Tile` | GEMV: `C[1,N] = A[1,K] @ B[K,N]` |
| `gemv_acc` | `(acc: Tile, lhs: Tile, rhs: Tile) -> Tile` | GEMV with accumulation |
| `gemv_bias` | `(lhs: Tile, rhs: Tile, bias: Tile) -> Tile` | GEMV with bias |

## Broadcast / Expand (`pl.tile.*`)

| Name | Signature | Description |
| ---- | --------- | ----------- |
| `row_expand` | `(target: Tile, row_vec: Tile) -> Tile` | Expand `row_vec[M,1]` to `target[M,N]` |
| `row_expand_add` | `(tile: Tile, row_vec: Tile) -> Tile` | `tile + row_vec[M,1]` broadcast |
| `row_expand_sub` | `(tile: Tile, row_vec: Tile) -> Tile` | `tile - row_vec` broadcast |
| `row_expand_mul` | `(tile: Tile, row_vec: Tile) -> Tile` | `tile * row_vec` broadcast |
| `row_expand_div` | `(tile: Tile, row_vec: Tile) -> Tile` | `tile / row_vec` broadcast |
| `row_expand_max` | `(tile: Tile, row_vec: Tile) -> Tile` | `max(tile, row_vec)` broadcast |
| `row_expand_min` | `(tile: Tile, row_vec: Tile) -> Tile` | `min(tile, row_vec)` broadcast |
| `row_expand_expdif` | `(tile: Tile, row_vec: Tile) -> Tile` | `exp(tile - row_vec[M,1])` broadcast |
| `col_expand` | `(target: Tile, col_vec: Tile) -> Tile` | Expand `col_vec[1,N]` to `target[M,N]` |
| `col_expand_mul` | `(tile: Tile, col_vec: Tile) -> Tile` | `tile * col_vec` broadcast |
| `col_expand_div` | `(tile: Tile, col_vec: Tile) -> Tile` | `tile / col_vec` broadcast |
| `col_expand_sub` | `(tile: Tile, col_vec: Tile) -> Tile` | `tile - col_vec` broadcast |
| `col_expand_add` | `(tile: Tile, col_vec: Tile) -> Tile` | `tile + col_vec[1,N]` broadcast |
| `col_expand_max` | `(tile: Tile, col_vec: Tile) -> Tile` | `max(tile, col_vec)` broadcast |
| `col_expand_min` | `(tile: Tile, col_vec: Tile) -> Tile` | `min(tile, col_vec)` broadcast |
| `col_expand_expdif` | `(tile: Tile, col_vec: Tile) -> Tile` | `exp(tile - col_vec[1,N])` broadcast |
| `expands` | `(target: Tile, scalar: int \| float \| Scalar) -> Tile` | Expand scalar to tile shape |

## Comparison / Selection (`pl.tile.*`)

Compare types: `EQ=0, NE=1, LT=2, LE=3, GT=4, GE=5`. `cmp` and `cmps` return
a target-packed predicate mask; use `sel` with an explicit `UINT8 [1, 32]`
scratch tile to materialize numeric results on A2/A3.

| Name | Signature | Description |
| ---- | --------- | ----------- |
| `cmp` | `(lhs: Tile, rhs: Tile, cmp_type: int = 0) -> Tile` | Compare two tiles |
| `cmps` | `(lhs: Tile, rhs: int \| float \| Scalar, cmp_type: int = 0) -> Tile` | Compare tile with scalar |
| `sel` | `(mask: Tile, lhs: Tile, rhs: Tile, tmp: Tile) -> Tile` | Select: `lhs if mask else rhs`; `tmp` is TSEL scratch |
| `sels` | `(lhs: Tile, rhs: Tile, select_mode: int \| float \| Scalar) -> Tile` | Select by scalar mode |

## Bitwise (`pl.tile.*`)

| Name | Signature | Description |
| ---- | --------- | ----------- |
| `and_` | `(lhs: Tile, rhs: Tile) -> Tile` | Bitwise AND |
| `ands` | `(lhs: Tile, rhs: int \| Scalar) -> Tile` | Bitwise AND with scalar |
| `or_` | `(lhs: Tile, rhs: Tile) -> Tile` | Bitwise OR |
| `ors` | `(lhs: Tile, rhs: int \| Scalar) -> Tile` | Bitwise OR with scalar |
| `xor` | `(lhs: Tile, rhs: Tile, tmp: Tile) -> Tile` | Bitwise XOR (requires tmp) |
| `xors` | `(lhs: Tile, rhs: int \| Scalar, tmp: Tile) -> Tile` | XOR with scalar (requires tmp) |
| `not_` | `(tile: Tile) -> Tile` | Bitwise NOT |
| `shl` | `(lhs: Tile, rhs: Tile) -> Tile` | Left shift |
| `shls` | `(lhs: Tile, rhs: int \| Scalar) -> Tile` | Left shift by scalar |
| `shr` | `(lhs: Tile, rhs: Tile) -> Tile` | Right shift |
| `shrs` | `(lhs: Tile, rhs: int \| Scalar) -> Tile` | Right shift by scalar |
| `rem` | `(lhs: Tile, rhs: Tile) -> Tile` | Remainder / modulo |
| `rems` | `(lhs: Tile, rhs: int \| float \| Scalar) -> Tile` | Remainder with scalar |
| `part_add` | `(lhs: Tile, rhs: Tile) -> Tile` | Partial add (copies the only valid input) |
| `part_mul` | `(lhs: Tile, rhs: Tile) -> Tile` | Partial multiply (copies the only valid input) |
| `part_max` | `(lhs: Tile, rhs: Tile) -> Tile` | Partial max (copies the only valid input) |
| `part_min` | `(lhs: Tile, rhs: Tile) -> Tile` | Partial min (copies the only valid input) |
| `fmod` | `(lhs: Tile, rhs: Tile) -> Tile` | Floating-point remainder (`torch.fmod`) |
| `fmods` | `(lhs: Tile, rhs: int \| float \| Scalar) -> Tile` | Floating-point remainder with scalar |

## Activations (`pl.tile.*`)

| Name | Signature | Description |
| ---- | --------- | ----------- |
| `relu` | `(tile: Tile) -> Tile` | ReLU: `max(0, x)` |
| `lrelu` | `(tile: Tile, slope: int \| float \| Scalar) -> Tile` | Leaky ReLU with scalar slope |
| `prelu` | `(tile: Tile, slope: Tile, tmp: Tile) -> Tile` | Parametric ReLU (requires tmp) |

## Shape Operations (`pl.tile.*`)

| Name | Signature | Description |
| ---- | --------- | ----------- |
| `slice` | `(tile: Tile, shape: Sequence[IntLike], offset: Sequence[IntLike]) -> Tile` | Slice (at most 2D). Sugar: `A[0:16, :]` |
| `reshape` | `(tile: Tile, shape: Sequence[IntLike]) -> Tile` | Reshape (at most 2D) |
| `transpose` | `(tile: Tile, axis1: int, axis2: int) -> Tile` | Swap two axes |
| `cast` | `(tile: Tile, target_type: DataType, mode="round") -> Tile` | Type cast |

## DSL Helpers (`pl.*`)

| Name | Signature | Description |
| ---- | --------- | ----------- |
| `range` | `(*args: int \| Scalar, init_values: tuple \| None = None) -> RangeIterator` | Sequential for-loop. Args: `(stop)`, `(start, stop)`, or `(start, stop, step)` |
| `parallel` | `(*args: int \| Scalar, init_values: tuple \| None = None) -> RangeIterator` | Parallel for-loop (same as range but parallel) |
| `while_` | `(*, init_values: tuple) -> WhileIterator` | While-loop (always requires init_values) |
| `yield_` | `(*values: Any) -> Any \| tuple[Any, ...]` | Yield values from for/if scope |
| `cond` | `(condition: bool \| Scalar) -> None` | Set while-loop condition (must be first statement) |
| `const` | `(value: int \| float, dtype: DataType) -> int \| float` | Typed constant |
| `incore` | `() -> IncoreContext` | Context manager for InCore scope |
| `dynamic` | `(name: str) -> DynVar` | Create dynamic dimension variable |
| `create_tensor` | `(shape: Sequence[IntLike], dtype: DataType, layout: TensorLayout = None, init_value: int \| float \| None = None) -> Tensor` | Create tensor (promoted from `pl.tensor`; `init_value` AICPU-pre-fills the buffer — `0` zeroes any dtype, non-zero needs an int / 32-bit-or-wider float dtype) |
| `max` | `(lhs: Scalar \| int \| Expr, rhs: Scalar \| int \| Expr) -> Scalar` | Max of two **scalars** (not a tile reduction — use `pl.tile.row_max` / `pl.tile.col_max`) |
| `min` | `(lhs: Scalar \| int \| Expr, rhs: Scalar \| int \| Expr) -> Scalar` | Min of two **scalars** (not a tile reduction — use `pl.tile.row_min` / `pl.tile.col_min`) |
