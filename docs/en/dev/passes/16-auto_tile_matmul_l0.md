# AutoTileMatmulL0 Pass

L0 tiling for `tile.matmul` / `tile.matmul_acc` ops with a Mat right operand (and a Mat- or Vec-resident left operand): pick an L0 tile shape `(m, n, k)` from the active backend's L0 capacities and rewrite the call into a 2-stage pipelined K-loop with per-iter Mat→Left/Right `tile.extract`s. When the `[M, N]` output itself exceeds L0c, the output is additionally tiled (M/N tiling) into a grid of `[m, n]` sub-tiles, each stored straight to the output tensor.

## Overview

Mat-resident matmuls produced upstream by `ConvertTensorToTileOps` + [`FlattenTileNdTo2D`](15-flatten_tile_nd_to_2d.md) carry full `(M, N, K)` operand shapes — almost always larger than the cube unit's L0a/L0b/L0c capacity. This pass picks an L0-fitting `(m, n, k)` and rewrites the matmul into a K-loop whose body extracts `[m, k]` and `[k, n]` slabs into `Left` / `Right` and accumulates into an `Acc`-resident iter-arg. The loop is marked `ForKind::Pipeline` with `pipeline_stages=2` so the downstream [`LowerPipelineLoops`](27-lower_pipeline_loops.md) pass produces a 2-deep ping-pong on the per-iter operand extracts.

**K-tiling vs M/N-tiling.** When the chooser returns `m == M` and `n == N` the output already fits L0c, so only the K dimension is tiled (one K-loop). When it returns `m < M` or `n < N` the `[M, N]` output Acc would overflow L0c. The operands are already Mat-resident, so *only* the output overflows: the pass tiles the **output** into a `ceil(M/m) × ceil(N/n)` grid of `[m, n]` sub-tiles (partial on the boundary — `m`/`n` need not divide `M`/`N`), computes each with the same pipelined K-loop, and stores each `[m, n]` Acc sub-tile straight to `out[mi:, ni:]` (the direct-store / DDR-output path). Every Acc tile is then ≤ L0c, so the matmul lowers through `AllocateMemoryAddr` with no overflow. The output tensor is chained through the per-sub-tile stores in SSA form (`out → out_t0 → out_t1 → …`).

**Pipeline position**: After [`FlattenTileNdTo2D`](15-flatten_tile_nd_to_2d.md), before [`InferTileMemorySpace`](18-infer_tile_memory_space.md). All tile ops are already 2D and memory spaces have not yet been inferred.

**Requirements**: `SSAForm`, `SplitIncoreOrch`, `IncoreTileOps`, `TileOps2D`, `NormalizedStmtStructure`.

**Produces**: same as required (property-preserving rewrite).

**Invalidates**: nothing.

**When to use**: Always, as part of the default tile-stage pipeline. The pass is a no-op when no Mat-resident matmul exceeds the backend's L0 capacity.

## API

| C++ | Python | Level |
| --- | ------ | ----- |
| `pass::AutoTileMatmulL0()` | `passes.auto_tile_matmul_l0()` | Program-level |

```python
from pypto.pypto_core import passes

l0_tile_pass = passes.auto_tile_matmul_l0()
program_tiled = l0_tile_pass(program)
```

## Algorithm

For each `tile.matmul` or `tile.matmul_acc` in an InCore-typed function:

1. **Filter** — operand layout: `(lhs, rhs)` for `tile.matmul`, `(acc, lhs, rhs)` for `tile.matmul_acc`. Both `lhs` and `rhs` must be `Var`/`IterArg` (via `AsVarLike`) of `TileType` with static 2D shape. The right (B) operand must be `memory_space == Mat` (loaded from DDR into L1, then fed to L0B). The left (A) operand may be `Mat` (the QK pattern) **or** `Vec` — the fused-attention `score·V` (PV) pattern, where the softmax/`exp` output reaches the matmul resident in `Vec` at the cube↔vector boundary. Other cases (Acc operands, a Vec right operand, dynamic shapes) are skipped silently. `tile.matmul_bias` is **not** rewritten — bias-add only after the final iteration needs extra rewriting that is not yet implemented.
2. **Pick L0 tile shape** — call `utils::ChooseL0Tile(cfg)` with the active `BackendHandler`'s `GetL0{a,b,c}CapacityBytes()` and `GetL0FractalAlignment()`, plus per-operand element width (`bytes_a/b/c`) read from the call's result type so the chooser sees the actual accumulator footprint. `c_read = is_matmul_acc` because `tile.matmul_acc` threads the caller's accumulator through the K-loop's iter-arg (γ_C = 2 in the chooser's traffic model). The chooser returns `(m, n, k)` — closed-form O(1) following the L0 tiling design note (continuous optimum + aligned candidates around it, scored by `(traffic, padded_compute, k_blocks, area, k)`).
3. **Skip if already L0-sized** — `(m, n, k) == (M, N, K)`.
4. **Skip with `PerfHint` for unsupported regimes**:
   - Sub-byte dtypes (cube path doesn't support them) — `PH-AT-003`.
   - `ChooseL0Tile` rejects the configuration — `PH-AT-005`.
   - `K % k != 0` — `PH-AT-007`. K-boundary handling (slice `valid_shape` on the last K iteration) is not yet implemented; applies to both K-only and M/N tiling.
5. **Build the K-loop** (per output sub-tile — the whole output when K-only, or each `[m, n]` sub-tile when M/N tiling):
   - `tile.matmul` — iter-arg init is an Acc-resident `tile.create([m, n], dtype, target_memory=Acc)` placeholder; the loop body branches on `ko == 0` between `tile.matmul` (fresh Acc) and `tile.matmul_acc` (accumulating into the iter-arg). The `IfStmt` materializes a phi return_var that the outer yield carries back to the iter-arg.
   - `tile.matmul_acc` — iter-arg init is the caller's accumulator directly (its type already matches the per-iter `tile.matmul_acc` output); every iteration is uniform `tile.matmul_acc`, so no if-else.
   - Per-iter operand extracts use `tile.extract(src, idx_row, idx_col, [shape], target_memory=Left|Right)` — the SSA-form fusion of the older `tile.slice` (Mat-resident result) + `tile.mov` (Mat→Left/Right) pair. This eliminates the intermediate Mat-resident slice tile and lowers to `pto.textract` rather than `pto.subview`, sidestepping the latter's `valid_row` codegen mismatch. For an output sub-tile at origin `(mi, ni)` the extracts slice `lhs[mi:mi+m, ko:ko+k]` and `rhs[ko:ko+k, ni:ni+n]`; the K-only case is `mi == ni == 0`, `m == M`, `n == N`.
   - **Vec left operand staging** — when the left (A) operand is `Vec`-resident (PV / `score·V`), a single `tile.move(lhs, target_memory=Mat)` is emitted **before** the K-loop and the per-iter Left extract slices from that staged Mat tile (so the extract source is Mat exactly like the QK path). Keeping the Vec→Mat crossing a `tile.move` lets [`ExpandMixedKernel`](21-expand_mixed_kernel.md) recognise it (`CollectCVBoundaryMoves` only matches `tile.move`) and lower it to the cross-core `tpop_from_aiv` handshake (which lands the data in Mat). Extracting straight from the Vec tile would instead leave the operand a dangling cross-boundary free variable on the cube side.
   - The K-loop is `ForKind::Pipeline` with `pipeline_stages=2`.
6. **M/N tiling (when `m < M` or `n < N`)** — the `[M, N]` output Acc overflows L0c. For a **plain `tile.matmul` whose result is consumed by exactly one 2D `tile.store(c, base, out)`**, the pass unrolls the output into a `ceil(M/m) × ceil(N/n)` grid: for each sub-tile origin `(mi, ni)` it builds the K-loop above for the `[m, n]` (partial on the boundary, `min(m, M-mi) × min(n, N-ni)`) sub-tile and emits `tile.store(c_sub, [base_r + mi, base_c + ni], out_prev)`. The stores chain the output tensor in SSA form; the final store's result replaces the original store downstream. Boundary sub-tiles use static partial extents (no `valid_shape` masking needed) — `m`/`n` need not divide `M`/`N`. The following M/N regimes are **deferred** and emit `PH-AT-006` (the matmul is left untouched): `tile.matmul_acc` (needs per-sub-tile slicing of the caller's `[M, N]` accumulator), a `Vec` left operand (PV path), `K / k < 2` (single K block — needs a straight-line per-sub-tile matmul), and a result not consumed by a single 2D store (needs the Mat-scratch / `tile.assemble` path for on-chip consumers).
7. **Rewrite the enclosing `SeqStmts`** — substitute uses of the original matmul's `Var` (K-only) or the consumer store's result (M/N) with the new `return_var`. Substitution is scoped to the `SeqStmts` that contains the rewrite, so it does not leak into sibling regions.

The pass is a `ProgramPass` and walks each function with an `IRMutator`; functions are returned unchanged when no rewrite fires (no `MutableCopy` cost for matmul-free programs).

## Examples

### Plain `tile.matmul`

**Before** (Mat-resident `tile.matmul` with `M = N = 128`, `K = 256`):

```python
@pl.program
class Before:
    @pl.function(type=pl.FunctionType.InCore)
    def main(self, ...):
        ...
        c: pl.Tile[[128, 128], pl.FP32] = pl.tile.matmul(a_mat, b_mat)
        ...
```

**After** (chooser picks `m = 128, n = 128, k = 64`):

```python
@pl.program
class After:
    @pl.function(type=pl.FunctionType.InCore)
    def main(self, ...):
        ...
        c_l0_init = pl.tile.create([128, 128], pl.FP32, target_memory=Acc)
        for ko, (c_iter,) in pl.pipeline(0, 256, 64, init_values=(c_l0_init,), stage=2):
            sa = pl.tile.extract(a_mat, 0, ko, [128, 64], target_memory=Left)
            sb = pl.tile.extract(b_mat, ko, 0, [64, 128], target_memory=Right)
            if ko == 0:
                c_first = pl.tile.matmul(sa, sb)
                c_phi = pl.yield_(c_first)
            else:
                c_acc = pl.tile.matmul_acc(c_iter, sa, sb)
                c_phi = pl.yield_(c_acc)
            c = pl.yield_(c_phi)
        # c (the yield-LHS) holds the accumulated Acc-typed result.
        ...
```

### `tile.matmul_acc`

The caller's accumulator threads through the iter-arg directly; no if-else is needed:

```python
for ko, (c_iter,) in pl.pipeline(0, K, k, init_values=(acc_init,), stage=2):
    sa = pl.tile.extract(a_mat, 0, ko, [m, k], target_memory=Left)
    sb = pl.tile.extract(b_mat, ko, 0, [k, n], target_memory=Right)
    c_new = pl.tile.matmul_acc(c_iter, sa, sb)
    c = pl.yield_(c_new)
# c (the yield-LHS) holds the accumulated Acc-typed result.
```

### M/N tiling (output exceeds L0c)

**Before** (`M = N = 512`, `K = 512`, FP32; the `[512, 512]` FP32 output is 1 MB > L0c, so the chooser picks `m = n = 256, k = 32`):

```python
c: pl.Tile[[512, 512], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(lhs_mat, rhs_mat)
out = pl.store(c, [0, 0], out)
```

**After** (2×2 grid of `[256, 256]` Acc sub-tiles, each a pipelined K-loop, each stored straight to the output — one sub-tile shown; the store chains `out → out_t0 → out_t1 → out_t2 → out_t3`):

```python
# Sub-tile (mi=256, ni=0): rows [256:512], cols [0:256].
c_t1_init = pl.tile.create([256, 256], dtype=pl.FP32, target_memory=Acc)
for ko, (c_iter,) in pl.pipeline(0, 512, 32, init_values=(c_t1_init,), stage=2):
    sa = pl.tile.extract(lhs_mat, 256, ko, [256, 32], target_memory=Left)
    sb = pl.tile.extract(rhs_mat, ko, 0, [32, 256], target_memory=Right)
    if ko == 0:
        c_first = pl.tile.matmul(sa, sb)
        c_phi = pl.yield_(c_first)
    else:
        c_acc = pl.tile.matmul_acc(c_iter, sa, sb)
        c_phi = pl.yield_(c_acc)
    c_t1 = pl.yield_(c_phi)
out_t1 = pl.store(c_t1, [256, 0], out_t0)  # store sub-tile to out[256:512, 0:256]
```

Boundary sub-tiles (when `m`/`n` do not divide `M`/`N`) use static partial extents `[min(m, M-mi), min(n, N-ni)]` — e.g. a 256×256 FP32 matmul on Ascend910B (chooser picks `m = 192, n = 160`) tiles into sub-tiles of `192×160`, `192×96`, `64×160`, `64×96`.

## Backend constraints

L0 capacities and fractal alignment come from the active `BackendHandler`. The pass reads from `PassContext::Current()->GetBackendHandler()` when a context is active, and falls back to `pypto::backend::GetBackend()->GetHandler()` for direct callers (e.g. tests that don't wrap in a `PassContext`).

| Handler call | Used as |
| ------------ | ------- |
| `GetL0aCapacityBytes()` | L0a (Left) capacity for chooser |
| `GetL0bCapacityBytes()` | L0b (Right) capacity for chooser |
| `GetL0cCapacityBytes()` | L0c (Acc) capacity for chooser |
| `GetL0FractalAlignment()` | M/N/K alignment grid for the chooser |
| `GetMinL0TileDim()` | Minimum per-axis tile dim |

Adding a new backend therefore only needs to provide these handler hooks — the pass itself is backend-neutral.

## Implementation

**Header**: `include/pypto/ir/transforms/passes.h`

**Properties**: `include/pypto/ir/transforms/pass_properties.h` (`kAutoTileMatmulL0Properties`)

**Implementation**: `src/ir/transforms/auto_tile_matmul_l0_pass.cpp`

**Chooser utility**: `src/ir/transforms/utils/l0_tile_chooser.cpp` — closed-form L0 shape picker, shared with future tilers.

**Python binding**: `python/bindings/modules/passes.cpp`

**Tests**: `tests/ut/ir/transforms/test_auto_tile_matmul_l0.py`, `tests/ut/ir/transforms/test_l0_tile_chooser.py`

## Pass Properties

| Property | Value |
| -------- | ----- |
| Required | SSAForm, SplitIncoreOrch, IncoreTileOps, TileOps2D, NormalizedStmtStructure |
| Produced | SSAForm, SplitIncoreOrch, IncoreTileOps, TileOps2D, NormalizedStmtStructure |
| Invalidated | — |

## Scope

| Op | Action |
| -- | ------ |
| `tile.matmul` over static-2D operands (Mat left, or Vec left for PV) + Mat right, output fits L0c | Rewritten to 2-stage pipelined K-loop; a Vec left operand is staged to Mat first |
| `tile.matmul` (plain, Mat left, Mat right) whose output exceeds L0c, consumed by one 2D `tile.store` | M/N-tiled: `ceil(M/m) × ceil(N/n)` grid of sub-tile K-loops, each stored straight to the output |
| `tile.matmul_acc` over static-2D operands (Mat left, or Vec left for PV) + Mat right, output fits L0c | Rewritten to 2-stage pipelined K-loop (uniform `matmul_acc` body) |
| `tile.matmul[_acc]` with a Vec **right** operand | Skipped (the B operand must feed L0B from L1) |
| `tile.matmul_bias` | Skipped (deferred — bias-add-only-after-final-iter rewrite not yet implemented) |
| Already L0-sized matmul (`(m, n, k) == (M, N, K)`) | Untouched |
| Output exceeds L0c but M/N fold not applicable (`matmul_acc`, Vec left, `K/k < 2`, or non-store consumer) | Skipped with `PerfHint` (`PH-AT-006`) |
| Sub-byte dtypes / `K % k != 0` | Skipped with `PerfHint` |
| Non-InCore functions (Orchestration, Opaque) | Untouched |

## Diagnostics

The pass emits `PerfHint` diagnostics rather than failing when it declines to rewrite — the original matmul is left intact and runs through the rest of the pipeline unchanged. PerfHint codes:

| Code | Meaning |
| ---- | ------- |
| `PH-AT-003` | Sub-byte dtype on operand or accumulator |
| `PH-AT-005` | `ChooseL0Tile` rejected the configuration |
| `PH-AT-006` | Output exceeds L0c but the M/N direct-store fold is not applicable — `tile.matmul_acc`, a Vec left operand, `K / k < 2`, or a result not consumed by a single 2D `tile.store` (the Mat-scratch / `tile.assemble` path for on-chip consumers is deferred) |
| `PH-AT-007` | `K % k != 0` (K-boundary handling not yet supported) |
| `PH-AT-008` | `ChooseL0Tile` returned a fallback configuration with a perf hint message |

## See also

- [`FlattenTileNdTo2D`](15-flatten_tile_nd_to_2d.md) — upstream pass; produces the static-2D Mat-resident tile shapes this pass consumes
- [`InferTileMemorySpace`](18-infer_tile_memory_space.md) — downstream pass; bridges Vec/Acc accumulators that this pass deliberately leaves alone
- [`LowerPipelineLoops`](27-lower_pipeline_loops.md) — consumes the `ForKind::Pipeline` + `pipeline_stages=2` produced here
