# ConvertTensorToTileOps Pass

Converts tensor operations to tile operations in InCore functions and updates orchestration call sites.

## Overview

After `OutlineIncoreScopes` extracts InCore scopes into separate functions, those functions still operate on `TensorType` variables using `tensor.*` operations. This pass lowers them to `TileType` variables with `tile.*` operations that map directly to PTO-ISA instructions.

The pass also updates call sites in orchestration/opaque functions: for each new output parameter added to an InCore function, a `tensor.create` is inserted at the call site.

**Requirements**:

- Input IR must be in SSA form
- InCore scopes must be outlined (run `OutlineIncoreScopes` first)
- Statement structure must be normalized

**When to use**: Run after `OutlineClusterScopes` and before `OptimizeOrchTensors`.

## API

| C++ | Python | Level |
| --- | ------ | ----- |
| `pass::ConvertTensorToTileOps()` | `passes.convert_tensor_to_tile_ops()` | Program-level |

**Python usage**:

```python
from pypto.pypto_core import passes

convert_pass = passes.convert_tensor_to_tile_ops()
program_tiled = convert_pass(program)
```

## Algorithm

The pass operates in three program-level phases:

### Phase 1: Transform InCore Functions

For each `FunctionType::InCore` function:

1. **Pre-scan MatmulSlice patterns**: Collect `tensor.slice` results consumed by `tensor.matmul` / `tensor.matmul_acc`. These need `tile.load(Mat, transpose=...)` instead of the default `tile.load(Vec)`.

2. **Insert tile.load (entry loads)**: For each `TensorType` parameter directly consumed by a converted op, insert `tile.load(param, zeros, shape, shape, target_memory=Vec, transpose=False)` at function entry. Parameters only referenced by self-loading ops (`tensor.slice`, `tensor.matmul`, `tensor.read`, `tensor.write`, `tensor.assemble`) are skipped — they manage their own loads.

3. **Convert body via TensorToTileMutator**: Walk the function body and convert each `tensor.*` call to its `tile.*` equivalent using `OpConversionRegistry`. The mutator propagates type changes through control flow (IterArgs, ForStmt/WhileStmt return_vars, IfStmt return_vars).

4. **Insert tile.store (exit stores)**: For each return value converted from `TensorType` to `TileType`, add an `Out` parameter and insert `tile.store(tile, zeros, out_param)`. If the return value comes from a `tile.assemble` loop, the loop is rewritten to use `tile.store` directly (conversion-time assemble-loop rewrite; distinct from `OptimizeOrchTensors` Pattern 3 which handles cross-function optimization).

### Phase 2a: Propagate Added Outputs Through Spmd/Group Wrappers

`OutlineClusterScopes` produces Spmd/Group wrappers that are transparent 1:1
forwarders of their params to a single inner InCore call. When Phase 1 appends
`Out` params to that InCore callee, the wrapper must mirror the appended params
on its own signature and forward them through the inner call — otherwise
orchestration codegen's `BuildWrapperReorderedParams` invariant (every inner-call
`Var` arg resolves to a wrapper param) breaks.

For each `FunctionType::Spmd` / `FunctionType::Group` function:

1. `ForwardedCallFinder` locates the first call to a transformed InCore (one
   whose Phase 1 added at least one `Out` param).
2. If found, the wrapper signature is extended with matching `Out` params (same
   type as the InCore's appended params, reusing the `name_hint_`), and
   `WrapperForwardMutator` rewrites the inner call to append the new vars as
   forward args and adopt the callee's new return type. `tensor.create` is
   *not* synthesised in the wrapper — allocation remains the caller's
   responsibility.
3. If no forwarded transformed-InCore call is found, the wrapper is left
   unchanged.

### Phase 2b: Update Orchestration Call Sites

For each orchestration / opaque function that calls a transformed InCore
function or a wrapper that absorbed output params in Phase 2a:

1. Insert `tensor.create` for each added output parameter
2. Append created tensors as extra arguments to the call

InCore, Spmd, and Group functions are skipped from this phase — they were
already rewritten in Phase 1 / 2a.

## MatmulSlice Pattern

When `tensor.slice` feeds into `tensor.matmul` or `tensor.matmul_acc`, the slice must produce a Mat-space tile instead of a Vec-space tile. The pass pre-scans for this pattern and emits `tile.load(Mat, transpose=...)` with the transpose flag from the matmul kwargs (`a_trans` for LHS, `b_trans` for RHS).

## Transpose Lowering

`tensor.transpose` lowers to a plain 3-arg **`tile.transpose(input, axis1, axis2)`**. The PTO `pto.ttrans` instruction needs a scratch workspace tile (same shape/dtype as the source), but that scratch is a pure codegen detail — not a semantic operand. [`FlattenTileNdTo2D`](15-flatten_tile_nd_to_2d.md) is the **sole owner** of scratch materialization: it emits the codegen-ready 4-arg form (`tile.create` + `tile.transpose(..., tmp)`) for both 2D and per-page >2D transposes, still before the memory allocator runs (so the scratch gets a real UB address). Keeping scratch out of the high-level op means `tensor.transpose` and the DSL `pl.tile.transpose(tile, axis1, axis2)` stay 1:1 with the semantic operation.

```python
# Before
y = tensor.transpose(x, 0, 1)

# After (this pass)
y_tile = pl.tile.transpose(x_tile, 0, 1)   # 3-arg, no scratch

# After FlattenTileNdTo2D (scratch materialized there)
transpose_tmp = pl.tile.create(x.shape, x.dtype, target_memory=x.memory_space)
y_tile = pl.tile.transpose(x_tile, 0, 1, transpose_tmp)
```

## Scatter Update Lowering

`tensor.scatter_update` / `tile.scatter_update` (whole-row scatter, `dim=-2` only) lower to a per-element `tile.scatter` (`pto.tscatter`) plus a `tile.sel` preserve-blend. The hardware `pto.tscatter` writes per element using a flattened destination index (`dst.flat[idx[k, c]] = src[k, c]`) and treats its `dst` operand as **write-only** (unwritten slots are not preserved), so the pass reconstructs the "keep `input` on unwritten rows" semantics itself.

The whole-row update `input[index.flat[k], :] = src[k, :]` is expressed as a flat index:

```text
flat_idx[k, c] = index.flat[k] * d + c          # d = feature width (= src cols)
```

The flat-index arithmetic is built **entirely in i32**, and only the finished row-major `[n, d]` index is narrowed to the `pto.tscatter`-required width (i16 for 2-byte data, i32 for 4-byte) via a single trailing `tile.cast`. Computing in i32 keeps every intermediate tile in a canonical, 32-byte-aligned, row-major layout — narrowing earlier would either cast a `col_major [n, 1]` view (which `tile.cast` mis-orders) or produce an unaligned 2-byte `[b, s]` tile (`cols * 2` bytes is not 32-byte aligned).

Generated PTO op sequence (FP32 `[32, 32]` input, `[2, 8]` index, `[16, 32]` src):

| # | PTO op | Produces |
| - | ------ | -------- |
| 1–3 | `pto.tload` ×3 | `input_tile`, `index_tile`, `src_tile` |
| 4 | `pto.tci` | column arange `[1, d]` = `0..d-1` |
| 5 | `pto.texpands` | zero template `[n, d]` |
| 6 | `pto.tcolexpand` | `col_nd[k, c] = c` |
| 7 | `pto.tmuls` | `row_base[k] = index.flat[k] * d` (index reshaped to `[n, 1]`) |
| 8 | `pto.trowexpandadd` | `flat_idx = col_nd + row_base` → `[n, d]` |
| 8a | `pto.tcvt` | narrow `flat_idx` i32→i16 (**2-byte dtypes only**) |
| 9 | `pto.texpands` | zeroed scatter base `[m, d]` |
| 10 | `pto.tscatter` | `scattered` = src into zeroed base (written = src, unwritten = 0) |
| 11–12 | `pto.texpands` ×2 | mask zero base `[m, d]`, ones src `[n, d]` |
| 13 | `pto.tscatter` | `mask` = ones into zeroed base (written = 1, unwritten = 0) |
| 14 | `pto.tcmps` | `pred = (mask != 0)` |
| 15 | `pto.tsel` | `out = sel(pred, scattered, input_tile)` |
| 16 | `pto.tstore` | write `out` to the output tensor |

`tile.sel` (not `input * mask`) reconstructs the preserve blend so the lowering emits no `pto.tmul`, which A2/A3 reject for bf16/i8. The index `reshape [b, s] → [n, 1]` is a buffer-view realias, not a separate PTO op.

## Paged Gather Lowering

`tensor.paged_gather(src, indices, block_table, ...)` gathers scattered rows of a paged KV pool directly into an on-chip buffer (L1 / `Mem.Mat` by default, or UB / `Mem.Vec`). The hardware `pto.tgather` instruction can only write UB, so paged-gather-to-L1 is **not** an indexed gather instruction — it is a fully-scalar per-row `GM → on-chip` DMA loop on the **Cube core (AIC)**. `src`, `indices`, and `block_table` are kept as GM tensors (the op is registered self-loading, so the framework does not preload them into Vec tiles).

The pass materializes the loop directly:

```text
rows = tensor.dim(indices, last_axis)                  # runtime gathered-row count
acc  = tile.create([max_indices, size], target_memory=space)   # static on-chip buffer
for i in [0, rows):                                    # ForStmt, iter_arg = acc
    idx   = tensor.read(indices, [i])                  # scalar GM read (pto.load_scalar)
    phys  = block_table[idx // block_size] * block_size + idx % block_size   # scalar
    acc   = tile.gather_row(acc, src, [i, 0], [phys, col_off], [1, size])    # GM->on-chip
    yield acc
```

`tile.gather_row` is a DPS op that writes one physical GM row straight into a
sub-region of the accumulator: `pto.subview` of `acc` + `pto.partition_view` of
`src` + `pto.tload` (`GM → on-chip`) — **no `pto.tmov`**. An L1→L1 `tmov` is
unsupported on a2a3 (L1 can only be filled via `GM → L1` `tload`), so the row is
loaded straight into the accumulator's sub-region rather than assembled.

Only the small index / page-table metadata is scalar-read from GM; the bulk KV data goes straight `GM → L1` via `pto.tload` and never touches UB — eliminating the GM round-trip that a `gather_kv → qk_pv` pipeline pays today. `is_trans=True` (Mat only) loads each row transposed into column offset `[0, i]`, giving the matmul B-operand layout. `max_indices` sizes the L1 buffer statically; the runtime `rows` count drives the loop bound, so dynamic gather counts are supported.

**Boxed (NZ) sub-region alignment.** An L1 (`Mem.Mat`) accumulator carries the matmul-operand NZ fractal layout, where `pto.subview` sizes must be whole multiples of the inner box (`M0 = 16` rows; `C0 = fractal_bytes / dtype_bytes / 16` cols). A per-row gather writes a single row, so `tile.gather_row` codegen emits a **box-aligned physical size** (`phys_rows = round_up(1, 16)`, `phys_cols = round_up(size, C0)`) while marking only the real extent valid (`valid = [1, size]`); the `tload` then fills just that row. UB (`Mem.Vec`, `slayout = none_box`) tiles have no inner box and use the exact `[1, size]` size. The gathered L1 tile is consumed by `tensor.matmul` directly (its natural use as a matmul operand).

### Kernel-Driven Gather (`tensor.create_l1` + `tensor.gather_row`)

`tensor.paged_gather` hardcodes its per-row source address (`block_table[idx // bs] * bs + idx % bs`). When the kernel needs arbitrary gather logic — multi-source selection, invalid-row clamping, overlay pools — it builds the same L1 accumulator itself from two tensor-level primitives, the flexible counterpart of `paged_gather`:

| Op | Lowers to | Role |
| -- | --------- | ---- |
| `tensor.create_l1(shape, dtype, transpose=...)` | `tile.create(target_memory=Mat, transpose=...)` | seed the loop-carried L1 accumulator |
| `tensor.gather_row(acc, src, dst_off, src_off, shapes, transpose=...)` | `tile.gather_row` (DPS) | DMA one **caller-addressed** GM row into `acc` |

Both deduce a `TensorType`, so the gathered result composes with tensor-level `tensor.matmul` / softmax; both are registered self-loading (`src` stays GM). The caller computes `src_off` and the `dst_off` slot, then fills the accumulator row by row in its own loop.

**Transpose (ZN) for a `b_trans` matmul operand.** `transpose=True` makes the gathered tile a transposed matmul B-operand without a GM round-trip:

- `tensor.create_l1(..., transpose=True)` allocates the **transposed Mat (ZN) fractal** (`blayout = row_major`, `slayout = col_major`) — the layout a `b_trans` operand carries.
- `tensor.gather_row(..., transpose=True)` places the GM row `[r, c]` as the L1 column `[c, r]`. `pto.tload` does not transpose, so codegen presents `src` as a **DN-strided view** (`pto.make_tensor_view ... {layout = #pto.layout<dn>}`, shape/strides swapped, same base ptr) and partitions the row as a column — the `tload` then runs `DN → NZ`, which *is* the transpose. (`paged_gather`'s `is_trans=True` shares this `tile.gather_row` path.) A straight `ND → NZ` `tload` would scramble the fractal layout.

## Example

**Before**:

```python
@pl.program
class Before:
    @pl.function(type=pl.FunctionType.InCore)
    def main_incore_0(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
        return y

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        y: pl.Tensor[[64], pl.FP32] = self.main_incore_0(x)
        return y
```

**After**:

```python
@pl.program
class After:
    @pl.function(type=pl.FunctionType.InCore)
    def main_incore_0(
        self, x: pl.Tensor[[64], pl.FP32],
        ret0_out: pl.Out[pl.Tensor[[64], pl.FP32]]
    ) -> pl.Tensor[[64], pl.FP32]:
        x_tile: pl.Tile[[64], pl.FP32] = pl.load(x, (0,), (64,))
        y_tile: pl.Tile[[64], pl.FP32] = pl.tile.add(x_tile, x_tile)
        ret0_store: pl.Tensor[[64], pl.FP32] = pl.store(y_tile, (0,), ret0_out)
        return ret0_store

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        ret0_out: pl.Tensor[[64], pl.FP32] = pl.tensor.create((64,), dtype=pl.FP32)
        y: pl.Tensor[[64], pl.FP32] = self.main_incore_0(x, ret0_out)
        return y
```

Key changes:

- `pl.add(x, x)` → `pl.tile.add(x_tile, x_tile)` (op conversion)
- `tile.load` inserted at entry, `tile.store` at exit
- `Out` parameter `ret0_out` added to InCore function
- `tensor.create` inserted at orchestration call site

## Implementation

**Header**: `include/pypto/ir/transforms/passes.h`

**Implementation**: `src/ir/transforms/convert_tensor_to_tile_ops_pass.cpp`

**Python binding**: `python/bindings/modules/passes.cpp`

**Tests**: `tests/ut/ir/transforms/test_convert_tensor_to_tile_ops.py`

## Pass Properties

| Property | Value |
| -------- | ----- |
| Required | SSAForm, SplitIncoreOrch, NormalizedStmtStructure |
| Produced | SSAForm, IncoreTileOps, NormalizedStmtStructure |
| Invalidated | — |

## Key Components

| Component | Role |
| --------- | ---- |
| `TensorArgsInConvertedOpsCollector` | IRVisitor — identifies tensor params needing entry loads |
| `MatmulSlicePatternCollector` | IRVisitor — finds slice→matmul patterns for Mat-space loads |
| `TypePropagatingMutator` | Base IRMutator — propagates type changes through control flow |
| `TensorToTileMutator` | IRMutator — converts tensor ops to tile ops via OpConversionRegistry |
| `ForwardedCallFinder` | IRVisitor — locates the wrapper's call into a transformed InCore (Phase 2a) |
| `WrapperForwardMutator` | IRMutator — appends new Out args to the wrapper's inner call (Phase 2a) |
| `CallSiteUpdateMutator` | IRMutator — inserts tensor.create at orchestration call sites (Phase 2b) |
| `IncoreTileOpsVerifier` | IRVisitor — verifies no TensorType ops remain in InCore functions |

## Scope

| Function type | Action |
| ------------- | ------ |
| InCore | Converted (tensor ops → tile ops); Phase 1 may append `Out` params |
| Spmd / Group (forwarding to a transformed InCore) | Signature mirrors the InCore's new `Out` params; inner call forwards them (Phase 2a) |
| Spmd / Group (no transformed-InCore forwarding) | Unchanged |
| Orchestration / Opaque | Call sites updated — `tensor.create` inserted for each new `Out` param (Phase 2b) |
