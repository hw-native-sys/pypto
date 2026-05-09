# AutoTileMatmulL0 Pass

L0 tiling for Mat-resident `tile.matmul` / `tile.matmul_acc` ops: pick an L0 tile shape `(m, n, k)` from the active backend's L0 capacities and rewrite the call into a 2-stage pipelined K-loop with per-iter Mat→Left/Right `tile.extract`s.

## Overview

Mat-resident matmuls produced upstream by `ConvertTensorToTileOps` + [`FlattenTileNdTo2D`](15-flatten_tile_nd_to_2d.md) carry full `(M, N, K)` operand shapes — almost always larger than the cube unit's L0a/L0b/L0c capacity. This pass picks an L0-fitting `(m, n, k)` and rewrites the matmul into a K-loop whose body extracts `[m, k]` and `[k, n]` slabs into `Left` / `Right` and accumulates into an `Acc`-resident iter-arg. The loop is marked `ForKind::Pipeline` with `pipeline_stages=2` so the downstream [`LowerPipelineLoops`](24-lower_pipeline_loops.md) pass produces a 2-deep ping-pong on the per-iter operand extracts.

**Pipeline position**: After [`FlattenTileNdTo2D`](15-flatten_tile_nd_to_2d.md), before [`InferTileMemorySpace`](17-infer_tile_memory_space.md). All tile ops are already 2D and memory spaces have not yet been inferred.

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

1. **Filter** — operand layout: `(lhs, rhs)` for `tile.matmul`, `(acc, lhs, rhs)` for `tile.matmul_acc`. Both `lhs` and `rhs` must be `Var`/`IterArg` (via `AsVarLike`) of `TileType` with static 2D shape and `memory_space == Mat`. Other cases (Vec/Acc operands, dynamic shapes) are skipped silently. `tile.matmul_bias` is **not** rewritten — bias-add only after the final iteration needs extra rewriting that is not yet implemented.
2. **Pick L0 tile shape** — call `utils::ChooseL0Tile(cfg)` with the active `BackendHandler`'s `GetL0{a,b,c}CapacityBytes()` and `GetL0FractalAlignment()`, plus per-operand element width (`bytes_a/b/c`) read from the call's result type so the chooser sees the actual accumulator footprint. `c_read = is_matmul_acc` because `tile.matmul_acc` threads the caller's accumulator through the K-loop's iter-arg (γ_C = 2 in the chooser's traffic model). The chooser returns `(m, n, k)` — closed-form O(1) following the L0 tiling design note (continuous optimum + aligned candidates around it, scored by `(traffic, padded_compute, k_blocks, area, k)`).
3. **Skip if already L0-sized** — `(m, n, k) == (M, N, K)`.
4. **Skip with `PerfHint` for unsupported regimes**:
   - Sub-byte dtypes (cube path doesn't support them) — `PH-AT-003`.
   - `ChooseL0Tile` rejects the configuration — `PH-AT-005`.
   - M/N tiling (`m != M || n != N`) — `PH-AT-006`. M/N tiling needs a Mat-resident output scratch + per-iter assemble that is not yet implemented.
   - `K % k != 0` — `PH-AT-007`. K-boundary handling (slice `valid_shape` on the last iteration) is not yet implemented.
5. **Build the K-loop**:
   - `tile.matmul` — iter-arg init is an Acc-resident `tile.create([m, n], dtype, target_memory=Acc)` placeholder; the loop body branches on `ko == 0` between `tile.matmul` (fresh Acc) and `tile.matmul_acc` (accumulating into the iter-arg). The `IfStmt` materializes a phi return_var that the outer yield carries back to the iter-arg.
   - `tile.matmul_acc` — iter-arg init is the caller's accumulator directly (its type already matches the per-iter `tile.matmul_acc` output); every iteration is uniform `tile.matmul_acc`, so no if-else.
   - Per-iter operand extracts use `tile.extract(src, idx_row, idx_col, [shape], target_memory=Left|Right)` — the SSA-form fusion of the older `tile.slice` (Mat-resident result) + `tile.mov` (Mat→Left/Right) pair. This eliminates the intermediate Mat-resident slice tile and lowers to `pto.textract` rather than `pto.subview`, sidestepping the latter's `valid_row` codegen mismatch.
   - The K-loop is `ForKind::Pipeline` with `pipeline_stages=2`.
6. **Rewrite the enclosing `SeqStmts`** — substitute uses of the original matmul's `Var` with the new `ForStmt`'s `return_var`. Substitution is scoped to the `SeqStmts` that contains the rewrite, so it does not leak into sibling regions.

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
| `tile.matmul` over Mat-resident static-2D operands | Rewritten to 2-stage pipelined K-loop |
| `tile.matmul_acc` over Mat-resident static-2D operands | Rewritten to 2-stage pipelined K-loop (uniform `matmul_acc` body) |
| `tile.matmul_bias` | Skipped (deferred — bias-add-only-after-final-iter rewrite not yet implemented) |
| Already L0-sized matmul (`(m, n, k) == (M, N, K)`) | Untouched |
| Sub-byte dtypes / `m != M` / `n != N` / `K % k != 0` | Skipped with `PerfHint` |
| Non-InCore functions (Orchestration, Opaque) | Untouched |

## Diagnostics

The pass emits `PerfHint` diagnostics rather than failing when it declines to rewrite — the original matmul is left intact and runs through the rest of the pipeline unchanged. PerfHint codes:

| Code | Meaning |
| ---- | ------- |
| `PH-AT-003` | Sub-byte dtype on operand or accumulator |
| `PH-AT-005` | `ChooseL0Tile` rejected the configuration |
| `PH-AT-006` | Chooser picked a shape that would need M/N tiling (not yet supported) |
| `PH-AT-007` | `K % k != 0` (K-boundary handling not yet supported) |
| `PH-AT-008` | `ChooseL0Tile` returned a fallback configuration with a perf hint message |

## See also

- [`FlattenTileNdTo2D`](15-flatten_tile_nd_to_2d.md) — upstream pass; produces the static-2D Mat-resident tile shapes this pass consumes
- [`InferTileMemorySpace`](17-infer_tile_memory_space.md) — downstream pass; bridges Vec/Acc accumulators that this pass deliberately leaves alone
- [`LowerPipelineLoops`](24-lower_pipeline_loops.md) — consumes the `ForKind::Pipeline` + `pipeline_stages=2` produced here
