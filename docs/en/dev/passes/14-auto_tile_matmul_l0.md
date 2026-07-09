# AutoTileMatmulL0 Pass

L0 tiling for `tile.matmul` / `tile.matmul_acc` ops with a Mat right operand (and a Mat- or Vec-resident left operand): pick an L0 tile shape `(m, n, k)` from the active backend's L0 capacities and rewrite the call into a 2-stage pipelined K-loop with per-iter Mat→Left/Right `tile.extract`s. When the `[M, N]` output itself exceeds L0c, the output is additionally tiled (M/N tiling) into a grid of `[m, n]` sub-tiles, each stored straight to the output tensor.

## Overview

Mat-resident matmuls produced upstream by `ConvertTensorToTileOps` + [`FlattenTileNdTo2D`](13-flatten_tile_nd_to_2d.md) carry full `(M, N, K)` operand shapes — almost always larger than the cube unit's L0a/L0b/L0c capacity. This pass picks an L0-fitting `(m, n, k)` and rewrites the matmul into a K-loop whose body extracts `[m, k]` and `[k, n]` slabs into `Left` / `Right` and accumulates into an `Acc`-resident iter-arg. The loop is marked `ForKind::Pipeline` with `pipeline_stages=2` so the downstream [`LowerPipelineLoops`](25-lower_pipeline_loops.md) pass produces a 2-deep ping-pong on the per-iter operand extracts.

**K-tiling vs M/N-tiling.** When the chooser returns `m == M` and `n == N` the output already fits L0c, so only the K dimension is tiled (one K-loop). When it returns `m < M` or `n < N` the `[M, N]` output Acc would overflow L0c. The operands are already Mat-resident, so *only* the output overflows: the pass tiles the **output** into a `ceil(M/m) × ceil(N/n)` grid of `[m, n]` sub-tiles (partial on the boundary — `m`/`n` need not divide `M`/`N`), computes each with the same pipelined K-loop, and stores each `[m, n]` Acc sub-tile straight to `out[mi:, ni:]` (the direct-store / DDR-output path). Every Acc tile is then ≤ L0c, so the matmul lowers through `AllocateMemoryAddr` with no overflow. The output tensor is chained through the per-sub-tile stores in SSA form (`out → out_t0 → out_t1 → …`).

**Fits-L0c chained cast-fold.** A chained matmul whose `[M, N]` result *fits* L0c (no M/N tiling) but feeds a second matmul through a downcast — `c = matmul(a, b); cb = cast(c, bf16); d = matmul(cb, e)` — needs the bf16 intermediate in **Mat** (L1) for the consumer. Left alone, `tile.cast` lowers to a **Vector** `pto.tcvt` (a cube→vector→cube round-trip that overflows the Vec buffer at `[128, 128]`). Instead the pass folds the cast into a **single full-window** Acc→Mat `tile.assemble` — the same `MatScratchPlacer` as the oversized Mat-scratch path, but one `PlaceAt` at offset `(0, 0)` rather than a grid — so the downcast stays on the cube as a FIXPIPE `pto.tinsert`. This is a cast-peephole independent of K tiling: it fires whether the producer was left whole (`k == K`) or K-looped (`k < K`), and only when every use of the cast result is a matmul operand (a non-matmul consumer keeps the Vector cast). The fold also mirrors exactly what FIXPIPE can reproduce — an **`f32 → bf16/f16`** downcast whose round mode is **`rint`** (round-half-to-*even*), FIXPIPE's fixed tie rule — the same on A2/A3 and A5 (the pto-isa CPU reference narrows via `std::bfloat16_t` with no arch branch, and `pto.tinsert` carries no `rmode`; the backends differ only in the scratch dtype, not the rounding). A non-`f32` accumulator (e.g. an `int32` matmul result, which would need a scaled *dequant*), the cast's default **`round`** mode (round-half-*away*), or a directional/truncating mode (`none`/`floor`/`ceil`/`trunc`/`odd`) all keep the Vector `pto.tcvt` — the only path that honors the requested `rmode` — and the pass emits a `PH-AT-010` hint pointing at `mode="rint"`. The same guard (`CastFoldableToFixpipeMat`) gates the oversized Mat-scratch fold below. Oversized results never reach this peephole — their cast is folded per sub-tile by the M/N path above.

**Pipeline position**: After [`FlattenTileNdTo2D`](13-flatten_tile_nd_to_2d.md), before [`InferTileMemorySpace`](16-infer_tile_memory_space.md). All tile ops are already 2D and memory spaces have not yet been inferred.

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
2. **Pick L0 tile shape** — call `utils::ChooseL0Tile(cfg)` with the active `BackendHandler`'s `GetL0{a,b,c}CapacityBytes()`, `GetL0FractalAlignment()` / `GetMinL0TileDim()`, and `GetL0CostModel()` (L1↔L0 bandwidths + MAD issue overhead), plus per-operand element width (`bytes_a/b/c`) read from the call's result type so the chooser sees the actual accumulator footprint. `c_read = is_matmul_acc` because `tile.matmul_acc` threads the caller's accumulator through the K-loop's iter-arg (γ_C = 2, doubling the C traffic the model charges). The chooser returns `(m, n, k)` plus the chosen design point — an **exhaustive roofline-`wall` minimum**, not a closed form; see [Cost model & design space](#cost-model--design-space-choosel0tile) below.
3. **Skip if already L0-sized** — `(m, n, k) == (M, N, K)`.
4. **Skip with `PerfHint` for unsupported regimes**:
   - Sub-byte dtypes (cube path doesn't support them) — `PH-AT-003`.
   - `ChooseL0Tile` rejects the configuration — `PH-AT-005`.
5. **Build the K-loop** (per output sub-tile — the whole output when K-only, or each `[m, n]` sub-tile when M/N tiling):
   - `tile.matmul` — iter-arg init is an Acc-resident `tile.create([m, n], dtype, target_memory=Acc)` placeholder; the loop body branches on `ko == 0` between `tile.matmul` (fresh Acc) and `tile.matmul_acc` (accumulating into the iter-arg). The `IfStmt` materializes a phi return_var that the outer yield carries back to the iter-arg.
   - `tile.matmul_acc` — iter-arg init is the caller's accumulator directly (its type already matches the per-iter `tile.matmul_acc` output); every iteration is uniform `tile.matmul_acc`, so no if-else.
   - Per-iter operand extracts use `tile.extract(src, idx_row, idx_col, [shape], target_memory=Left|Right)` — the SSA-form fusion of the older `tile.slice` (Mat-resident result) + `tile.mov` (Mat→Left/Right) pair. This eliminates the intermediate Mat-resident slice tile and lowers to `pto.textract` rather than `pto.subview`, sidestepping the latter's `valid_row` codegen mismatch. For an output sub-tile at origin `(mi, ni)` the extracts slice `lhs[mi:mi+m, ko:ko+k]` and `rhs[ko:ko+k, ni:ni+n]`; the K-only case is `mi == ni == 0`, `m == M`, `n == N`.
   - **Vec left operand staging** — when the left (A) operand is `Vec`-resident (PV / `score·V`), a single `tile.move(lhs, target_memory=Mat)` is emitted **before** the K-loop and the per-iter Left extract slices from that staged Mat tile (so the extract source is Mat exactly like the QK path). Keeping the Vec→Mat crossing a `tile.move` lets [`ExpandMixedKernel`](19-expand_mixed_kernel.md) recognise it (`CollectCVBoundaryMoves` only matches `tile.move`) and lower it to the cross-core `tpop_from_aiv` handshake (which lands the data in Mat). Extracting straight from the Vec tile would instead leave the operand a dangling cross-boundary free variable on the cube side.
   - The K-loop is `ForKind::Pipeline` with `pipeline_stages=2`.
   - **Non-divisor K (K-boundary peel)** — when the chosen `k` does not divide `K`, the pipelined loop covers only the `⌊K/k⌋` full blocks (bound `⌊K/k⌋·k`) and a straight-line `tile.matmul_acc` peels the partial last block of width `K − ⌊K/k⌋·k`; when only one full block fits (`⌊K/k⌋ == 1`), a single straight-line full block + tail replaces the loop. With `K` and `k` both 16-aligned (the cube fractal), the peeled tail width `K − ⌊K/k⌋·k` is itself 16-aligned — an ordinary `matmul_acc` block, no masking. (ptoas requires 16-aligned tile cols, so the operand dimensions must be 16-aligned; non-16-aligned `K` is **not** supported.) The chooser only returns a non-divisor `k` under `ChooseL0Tile`'s `allow_k_boundary`, which this pass sets; when the full (16-aligned) K fits one L0 block the chooser returns `k == K` (no loop) instead. A **non-16-aligned `K` is rejected outright** — there is no valid K-tiling (any peeled tail or whole-K block would have non-fractal cols), so the chooser returns no candidate and the pass skips the matmul with a `PH-AT-007` hint rather than emit invalid extracts.
6. **M/N tiling (when `m < M` or `n < N`)** — the `[M, N]` output Acc overflows L0c. For a **plain `tile.matmul` whose result is consumed by exactly one 2D `tile.store(c, base, out)`**, the pass tiles the output into a `ceil(M/m) × ceil(N/n)` grid: for each sub-tile origin `(mi, ni)` it computes the `[m, n]` (partial on the boundary, `min(m, M-mi) × min(n, N-ni)`) sub-tile and emits `tile.store(c_sub, [base_r + mi, base_c + ni], out_prev)`. When **K spans ≥ 2 L0 blocks**, each sub-tile is an independent **pipelined K-loop** (the `[m, K]`/`[K, n]` panel does not fit L0, so it is re-extracted per sub-tile). When **`k == K`** (the full K fits L0a/L0b at once), the grid is emitted as **nested `ForKind::Pipeline` loops** over the divisible `[0, full_m) × [0, full_n)` interior (`full_m = ⌊M/m⌋·m`, `full_n = ⌊N/n⌋·n`), so [`LowerPipelineLoops`](25-lower_pipeline_loops.md) double-buffers the moving-operand `tile.extract` (hidden behind the cube). The outer loop owns the **stationary** panel (re-extracted once per outer step); the inner loop is `pipeline_stages=2` so [`LowerPipelineLoops`](25-lower_pipeline_loops.md) double-buffers the moving panel and [`CanonicalizeIOOrder`](26-canonicalize_io_order.md) keeps each store adjacent to its matmul (`pipeline_overlap_stores=false` → one L0C accumulator, not two co-live). **Which operand is stationary, and its buffering, follow the chooser's design point.** For **output-stationary** the outer panel is the one the chooser scored as the cheaper hoist under the **bandwidth-weighted** interior load — rows-outer `held_A = P·A/BW_A + P·Q·B/BW_B` vs cols-outer `held_B = P·Q·A/BW_A + Q·B/BW_B` (`P`/`Q` = interior row/col block counts, `A = m·K·bytes_a` / `B = K·n·bytes_b`) — the *same* `min`-hoist the `wall` was scored under, recorded in the chooser's `os_holds_a` so the emit obeys the scored hoist instead of re-deriving it from raw bytes (weighting by the L0A/L0B bandwidths matters because the ~1.5:1 asymmetry makes fewest-bytes and fewest-cycles disagree — a square tile ties on bytes but not on cycles). **Both** operands double-buffer (outer + inner are `ForKind::Pipeline`). For **A/B-stationary** the *held* operand is the outer panel and the outer loop is `ForKind::Sequential`, so that operand is **single-buffered** in the full L0 buffer the chooser budgeted un-halved (only the moving inner panel double-buffers). The L-shaped **partial boundary** (`[full_m, M) × [0, N)` plus `[0, full_m) × [full_n, N)`) is peeled into straight-line partial tiles, so `m`/`n` need not divide `M`/`N` — no exact-divisor constraint that would collapse e.g. `M = 272 = 16·17` to a 16×16 tile. The stores chain the output tensor in SSA form; the final store's result replaces the original store downstream. The following M/N regimes are **deferred** and emit `PH-AT-006` (the matmul is left untouched): `tile.matmul_acc` (needs per-sub-tile slicing of the caller's `[M, N]` accumulator), a `Vec` left operand (PV path), and a result consumed on-chip that is **not** consumed *entirely* as a matmul operand (a mixed store-plus-on-chip or an elementwise use). A result consumed **entirely as a matmul operand** (a chained matmul) is *not* deferred — it takes the **Mat-scratch** placement below.

   **Placement (direct-store vs Mat-scratch).** Both grids hand each `[m, n]` Acc sub-tile to a `SubtilePlacer`. The **`DirectGmPlacer`** stores it to the DDR output (`tile.store`, above). The **`MatScratchPlacer`** instead keeps the whole `[M, N]` result on-chip in an L1/**Mat** scratch — created once with `tile.create(target_memory=Mat)` (whose implicit NZ TileView `col_major/row_major` is the matmul-operand layout), then each sub-tile is assembled in place via `tile.assemble(scratch, sub, [mi, ni])` (Acc→Mat, lowering to `pto.subview` + `pto.tmov`). The pass selects Mat-scratch when the matmul result's uses are **all** matmul-operand reads *and* the `[M, N]` scratch fits the backend handler's Mat capacity (`GetMatCapacityBytes()`) — a conservative necessary-condition gate that keeps oversized chained matmuls on the deferred `PH-AT-006` path instead of emitting an impossible on-chip allocation (a full packed-peak check that also accounts for coexisting Mat tensors is a follow-up). On selection it remaps the result `Var` to the scratch so the consumer reads it on-chip. `tile.assemble`'s `set_output_memory_inherit_input()` makes the chain share one Mat base, so the assemble is in place (no unsupported Mat→Mat preservation copy). Both the split-K (unrolled, constant offsets) and full-K (pipelined, loop-variable offsets) grids drive either placer.

   > **Follow-up — operand-stationary chained producers + L0 packing.** A chained-matmul (Mat-scratch) producer shares L0 with its consumer (sequential; the intermediate stays in L1, never DDR — the `L0C→L1→L0A` trip). For their L0 operand buffers to reuse the same space they currently need the **same buffer shape**: an A/B-stationary producer pins one monolithic full-L0 operand buffer that a double-buffered consumer's two half-size buffers cannot pack against, because `AllocateMemoryAddr` bump-stacks reuse classes and never subdivides a freed region (a 64 KB producer buffer reused for one 32 KB consumer half wastes 32 KB, and the other half spills → L0 overflow). So today the chooser's natural **output-stationary** choices (matching buffer shapes) are what coexist. Liveness-aware **offset-packing** in the allocator — place each buffer at the lowest offset free for its lifetime — would let either operand-stationary order pack; tracked as a separate follow-up.
7. **Rewrite the enclosing `SeqStmts`** — substitute uses of the original matmul's `Var` (K-only) or the consumer store's result (M/N) with the new `return_var`. Substitution is scoped to the `SeqStmts` that contains the rewrite, so it does not leak into sibling regions.

The pass is a `ProgramPass` and walks each function with an `IRMutator`; functions are returned unchanged when no rewrite fires (no `MutableCopy` cost for matmul-free programs).

## Cost model & design space (`ChooseL0Tile`)

`ChooseL0Tile` picks the L0 GEMM tile by an **exhaustive roofline search**, not a closed form. For every legal aligned `(m, n, k)` — each a multiple of `GetL0FractalAlignment()`, fitting the L0a/L0b/L0c budgets — it estimates wall-clock in core cycles and returns the minimum:

- `wall ≈ max(C_load, C_mad) + C_drain` when the FIXPIPE L0C→L1 drain is exposed (single L0C), or
- `wall ≈ max(C_load, C_mad, C_drain) + min(compute, C_drain) / T` when the drain is hidden behind compute (double-buffered L0C, `T` output tiles). The `+ min(…)/T` term is the pipeline **fill/drain bubble** — the first tile's compute (or the last tile's drain) has no partner to overlap, so the ideal all-hidden `T·max` roofline undercounts by one tile's non-dominant pipe (≈25% of the smaller pipe at a 2×2 grid). This keeps dbC=2 from being over-picked on small grids.

`C_load` is the L1→L0A/L0B operand traffic under the chosen loop order, scaled by the per-buffer bandwidths from `GetL0CostModel()` (on-device MTE1 sweep: `bw_l0a≈130`, `bw_l0b≈85` B/cyc, ~1.52:1); `C_mad` is the cube MAD cost (per-`TMATMUL` issue overhead × K-fractal count). `C_drain` is the FIXPIPE L0C writeback, charged **per output tile** as a **per-M-row** cost: `⌈M/m⌉·⌈N/n⌉ · (drain_fixed + m·(max(drain_row, bytes_c·n/bw_drain) + drain_penalty·(odd(⌈n/N0⌉)−1)))`. A direct fit of an on-device FIXPIPE sweep: FIXPIPE addresses one M-row of the `N1 M1 M0 N0` FRACTAL_NZ accumulator at a time (so cost ∝ `m`), each row a grouped `nburst`/`loop` over the `N1 = ⌈n/N0⌉` N-fractals (`N0 = 32/bytes_c = 8` for the fp32 L0C). The per-row cost is `max(floor, throughput)` — a fixed burst-issue **floor** `drain_row` (row addressing/setup, N-independent) that dominates narrow N, or the fractal **throughput** `bytes_c·n/bw_drain` that dominates wide N (crossover ~n=131) — plus the **misalignment** residual: a non-power-of-two fractal count serializes the odd part `odd(N1)−1` into extra passes at `drain_penalty` per M-row (the predicate is a **non-power-of-two `N1`**, not literally `N%32`: `n=80 → odd(10)=5` is penalized, and so is `n=96 → odd(12)=3` even though `96%32=0`; aligned power-of-two `N1` such as `n=128 → 16` pays nothing). Because the drain count is `⌈M/m⌉·⌈N/n⌉`, **splitting the output (M/N) adds drains but splitting K does not** (partial sums accumulate in one L0C, drained once per `(m,n)` block). The per-M-row form makes the chooser prefer **wide-N / small-M** tiles (fewer FIXPIPE rows per drain) and correctly prices a misaligned-N tile so it is not over-selected — e.g. `320×320` lands an aligned `(160,128,64)` instead of the drain-bound `160×80`. Device-validated (drain 0.93–1.09×, loads R²=0.993). The search is exhaustive over **all** legal `k` per `(m, n)` (not the largest legal k — `⌈K/k⌉·⌈k/kt⌉` is non-monotone in `k` when `kt ≠ align_k`). Wall ties break lexicographically on `(padded_compute, ⌈K/k⌉, C_load, …)`. The search is exhaustive over **all** legal `k` per `(m, n)` (not the largest legal k — `⌈K/k⌉·⌈k/kt⌉` is non-monotone in `k` when `kt ≠ align_k`). Wall ties break lexicographically on `(padded_compute, ⌈K/k⌉, C_load, …)`; the `C_load` key picks the lower-hidden-load aspect among MAD-bound `(m,n)`↔`(n,m)` ties (L0B's slower bandwidth favours fewer m-blocks).

The search ranges over the **design space** `P = (m, n, k, stationarity, dbC)`:

- **stationarity** `{output, A, B}` — which operand is pinned across the L0 grid. This *derives* the per-operand double-buffer depths (`dbA`/`dbB`): the moving operand(s) double-buffer (depth 2), the stationary one single-buffers (depth 1). They are not searched independently.
- **dbC** `{1, 2}` — whether the L0C accumulator is double-buffered to overlap the FIXPIPE drain with the next tile's compute.

A **realizable mask** (the `allow_a_stationary` / `allow_b_stationary` / `allow_double_buffer_c` config gates) restricts which design points are *enumerated and emitted* to those whose lowering exists — a gated-off axis is **not** explored (not scored); opening a gate adds those points to the search. The pass opens the **A/B-stationary** gates: the held operand is pinned **single-buffered** in the full L0 buffer across the moving grid (`k == K`), realized by a `ForKind::Sequential` outer loop in `BuildFullKPipelined` (a `Pipeline` outer would double-buffer the held operand → 2× its full-L0 budget → overflow). So the pass emits **output-stationary or operand-stationary**. **dbC=2** (the two-accumulator L0C ping-pong: tile *i*'s FIXPIPE drain overlaps tile *i+1*'s MAD) is opened unconditionally under `memory_planner=PTOAS`, and under the PyPTO planner as an **experimental opt-in** (`PassContext(enable_pypto_l0c_double_buffer=True)`, default off pending device validation of the numerics + drain-hidden win): `cfg.allow_double_buffer_c = ptoas_planner || (pypto_planner && flag)`. In both planners `BuildFullKPipelined` tags the moving loop with `kPipelineDoubleBufferCAttr` and `CanonicalizeIOOrder` floats **both** stores below **both** matmuls (`matmul, matmul, store, store` — co-live lifetimes rather than the default `matmul, store, …` disjoint ones). The two co-live accumulators then survive allocation differently per planner: under **PTOAS** because it skips `MemoryReuse` (`InitMemRef` gives the two stages distinct L0C bases and ptoas places them at distinct offsets); under **PyPTO** because [`LowerPipelineLoops`](25-lower_pipeline_loops.md) gives the dbC accumulator a **flat depth-2** `pipeline_membership` — only the moving (dbC) loop tags it; enclosing loops skip it since the cube serializes MADs — so `MemoryReuse`'s capacity gate (#1475) allocates exactly the two co-live L0C buffers instead of coalescing them (its former behaviour there, which shrank the tile to L0C/2 with no second buffer). dbC=2 requires full-K and a ≥2×2 grid; the Mat-scratch (`Acc→Mat`, `tile.assemble`) drain is floated the same way. A `PassManager` built under one planner and run under another **fails loud** (the pass list's `MemoryReuse`-skip and the chooser's dbC gate must agree). The cost-model formulas themselves are gate-independent. See [`26-canonicalize_io_order.md`](26-canonicalize_io_order.md) for the co-live float and the runtime device validation for numerics + the distinct `{0, L0C/2}` offsets.

> **This is a model-driven tile change, not a behavior-neutral refactor.** The roofline objective replaced an earlier traffic-minimizing closed-form chooser, so the selected `(m, n, k)` differs from before for MAD-bound shapes. The pre/post tiles for representative shapes are pinned in `test_l0_tile_chooser.py::TestL0TilingRooflineMigration`.

The full rationale (the perf-sim derivation of the bandwidth / MAD numbers, the stationarity and double-buffer findings) lives in the chooser header `l0_tile_chooser.h` and the perf-sim study `DESIGN_SPACE.md`. `ChooseL0Tile`'s optimum is verified against a brute-force re-enumeration of the same cost model in `tests/ut/ir/transforms/test_l0_tile_chooser.py` — an independent check of the *solver* (that it finds the model's global minimum), not of the model against hardware.

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

### Fits-L0c chained matmul (cast-fold)

**Before** (`[128, 128]` intermediate fits L0c; `K = 64` fits L0, so the producer is a single matmul):

```python
c  = pl.tile.matmul(a_mat, b_mat)          # [128, 128] Acc f32 — fits L0c
cb = pl.tile.cast(c, pl.BF16)              # would lower to a Vector pto.tcvt
d  = pl.tile.matmul(cb, e_mat)             # consumes the bf16 intermediate on-chip
out = pl.tile.store(d, [0, 0], out)
```

**After** (the cast is folded into one full-window Acc→Mat assemble; `cb`'s consumer reads the Mat scratch):

```python
c       = pl.tile.matmul(a_mat, b_mat)                       # unchanged (fits L0c)
c_mat   = pl.tile.create([128, 128], dtype=pl.BF16, target_memory=Mat)  # the L1/Mat scratch
c_mat_t0 = pl.tile.assemble(c_mat, c, [0, 0])                # Acc f32 → Mat bf16 (cube pto.tinsert)
d       = pl.tile.matmul(c_mat_t0, e_mat)                    # reads the scratch on-chip
out     = pl.tile.store(d, [0, 0], out)
```

The `tile.cast` is dropped. When the producer needs a K-loop (`k < K`), the K-loop is emitted as usual and its Acc result feeds the *same* single `tile.assemble` — the fold is independent of K tiling.

## Backend constraints

L0/Mat capacities and fractal alignment come from the active `BackendHandler`. The pass reads from `PassContext::Current()->GetBackendHandler()` when a context is active, and falls back to `pypto::backend::GetBackend()->GetHandler()` for direct callers (e.g. tests that don't wrap in a `PassContext`).

| Handler call | Used as |
| ------------ | ------- |
| `GetL0aCapacityBytes()` | L0a (Left) capacity for chooser |
| `GetL0bCapacityBytes()` | L0b (Right) capacity for chooser |
| `GetL0cCapacityBytes()` | L0c (Acc) capacity for chooser |
| `GetMatCapacityBytes()` | Mat (L1) capacity for Mat-scratch gate |
| `GetL0FractalAlignment()` | M/N/K alignment grid for the chooser |
| `GetMinL0TileDim()` | Minimum per-axis tile dim |

Adding a new backend therefore only needs to provide these handler hooks — the pass itself is backend-neutral.

## Implementation

**Header**: `include/pypto/ir/transforms/passes.h`

**Properties**: `include/pypto/ir/transforms/pass_properties.h` (`kAutoTileMatmulL0Properties`)

**Implementation**: `src/ir/transforms/auto_tile_matmul_l0_pass.cpp`

**Chooser utility**: `src/ir/transforms/utils/l0_tile_chooser.cpp` — roofline cost-model L0 tile picker (exhaustive over the legal aligned grid; see [Cost model & design space](#cost-model--design-space-choosel0tile)), shared with future tilers.

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
| `tile.matmul` (plain, Mat left, Mat right) whose output exceeds L0c, consumed by one 2D `tile.store` | M/N-tiled: `ceil(M/m) × ceil(N/n)` grid of sub-tile K-loops, each stored straight to the output (direct-store) |
| `tile.matmul` (plain) whose output exceeds L0c, consumed *entirely* as a matmul operand (chained matmul), and whose `[M, N]` scratch fits Mat/L1 | M/N-tiled into an L1/**Mat** scratch (per-sub-tile Acc→Mat `tile.assemble`), kept on-chip for the consumer (Mat-scratch) |
| `tile.matmul` whose output *fits* L0c, downcast via `tile.cast(c, bf16/f16)` whose result is consumed *entirely* as a matmul operand (chained) | Cast-fold: one full-window Acc→Mat `tile.assemble` (cube `pto.tinsert`); the cast is dropped — no Vector `pto.tcvt` round-trip |
| `tile.matmul_acc` over static-2D operands (Mat left, or Vec left for PV) + Mat right, output fits L0c | Rewritten to 2-stage pipelined K-loop (uniform `matmul_acc` body) |
| `tile.matmul[_acc]` with a Vec **right** operand | Skipped (the B operand must feed L0B from L1) |
| `tile.matmul_bias` | Skipped (deferred — bias-add-only-after-final-iter rewrite not yet implemented) |
| Already L0-sized matmul (`(m, n, k) == (M, N, K)`) | Untouched |
| Output exceeds L0c but no M/N placement applies — `matmul_acc`, Vec left, a non-matmul-operand consumer, or a chained-matmul scratch whose `[M, N]` exceeds Mat/L1 | Skipped with `PerfHint` (`PH-AT-006`) |
| `K` not a multiple of the cube fractal (16) | Skipped with `PerfHint` (`PH-AT-007`) — no fractal-aligned K-tiling |
| Sub-byte dtypes | Skipped with `PerfHint` |
| Non-InCore functions (Orchestration, Opaque) | Untouched |

## Diagnostics

The pass emits `PerfHint` diagnostics rather than failing when it declines to rewrite — the original matmul is left intact and runs through the rest of the pipeline unchanged. PerfHint codes:

| Code | Meaning |
| ---- | ------- |
| `PH-AT-003` | Sub-byte dtype on operand or accumulator |
| `PH-AT-005` | `ChooseL0Tile` rejected the configuration |
| `PH-AT-006` | Output exceeds L0c but neither M/N placement applies — `tile.matmul_acc`, a Vec left operand, or a result consumed on-chip that is **not** *entirely* a matmul operand (mixed store-plus-on-chip, or elementwise). A result consumed entirely as a matmul operand takes the **Mat-scratch** path (no hint) — *unless* its `[M, N]` scratch exceeds the backend's Mat/L1 capacity, in which case it is deferred here too (a conservative necessary-condition gate; a full packed-peak check is a follow-up). |
| `PH-AT-007` | Non-16-aligned `K` — no fractal-aligned K-tiling exists (any peeled tail or whole-K block would have non-fractal cols), so the matmul is left untouched |
| `PH-AT-008` | `ChooseL0Tile` returned a fallback configuration with a perf hint message |
| `PH-AT-009` | Backend needs a bf16/f16 on-chip Mat scratch (e.g. Ascend910B) but the oversized chained-matmul intermediate is f32 — cast the matmul result to bf16/f16 before the consumer matmul; left on the deferred path |
| `PH-AT-010` | A fits-L0c chained-matmul cast cannot fold onto the cube FIXPIPE (which narrows `f32 → bf16/f16` with round-half-to-even only): the source is non-f32, or the round mode is not `rint` (e.g. the default `round`, or `floor`/`ceil`/`trunc`/`odd`/`none`). Kept on the Vector `pto.tcvt` path — a cube→vector→cube round-trip that may overflow the Vec buffer at large `[M, N]`. Cast an f32 result with `mode="rint"` to keep it on the cube. |

## See also

- [`FlattenTileNdTo2D`](13-flatten_tile_nd_to_2d.md) — upstream pass; produces the static-2D Mat-resident tile shapes this pass consumes
- [`InferTileMemorySpace`](16-infer_tile_memory_space.md) — downstream pass; bridges Vec/Acc accumulators that this pass deliberately leaves alone
- [`LowerPipelineLoops`](25-lower_pipeline_loops.md) — consumes the `ForKind::Pipeline` + `pipeline_stages=2` produced here
