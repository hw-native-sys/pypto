# AutoTileMatmulL0 Pass

L0 tiling for static 2D `tile.matmul`, `tile.matmul_acc`, and `tile.matmul_bias`: pick an L0 tile shape `(m, n, k)` from the active backend's L0 capacities and rewrite the call into a 2-stage pipelined K-loop with per-iter Mat→Left/Right `tile.extract`s. Capacity checks use the accumulator's physical L0C footprint, not just its logical shape. When that footprint exceeds L0c, the pass M/N-tiles a fresh plain or biased matmul, or the canonical frontend split-K create/pipeline/store chain, into `[m, n]` output sub-tiles.

## Overview

Mat-resident matmuls produced upstream by `ConvertTensorToTileOps` + [`FlattenTileNdTo2D`](13-flatten_tile_nd_to_2d.md) carry full `(M, N, K)` operand shapes — almost always larger than the cube unit's L0a/L0b/L0c capacity. This pass picks an L0-fitting `(m, n, k)` and rewrites the matmul into a K-loop whose body extracts `[m, k]` and `[k, n]` slabs into `Left` / `Right` and accumulates into an `Acc`-resident iter-arg. The loop is marked `ForKind::Pipeline` with `pipeline_stages=2` so the downstream [`LowerPipelineLoops`](29-lower_pipeline_loops.md) pass produces a 2-deep ping-pong on the per-iter operand extracts.

An operand already placed in `Left` or `Right` is a manual L0 scheduling decision, so AutoTile does not silently replace or subdivide it. Legal manual tiles remain untouched. If one static operand by itself exceeds its backend space—for example an FP16 `Right[256, 256]` requiring 131072 bytes in a 65536-byte L0B—the pass instead fails at that operand with its variable name, physical shape, dtype, required/available bytes, and two fixes: keep the operand in `Mat` and pass it directly to the matmul so AutoTile can choose legal L0 tiles, or manually extract a smaller L0 tile. This early operation-specific error prevents the later allocator failure from being preceded by an irrelevant MemoryReuse fallback warning.

The pass also handles an already-tiled form under the PyPTO memory planner: a user-authored static `pl.pipeline(stage=F)` (`F ≥ 2`, trip count divisible by `F`) with one `tile.matmul(Left, Right)` per iteration and a canonical loop-carried drain. The selected moving operand must be produced by a direct per-iteration Mat→L0 transfer, while the other matmul operand is defined outside the loop. AutoTile enables a two-slot L0C ping-pong only when a path-specific profitability gate passes and the conservative whole-function Acc footprint—including the physical stage multiplicity that pipeline lowering can request for other Acc producers—plus one extra slot for every eligible loop fits in L0C. A direct-to-GM `tile.store` requires at least four iterations. The cheaper Acc-to-Mat `tile.assemble` path requires at least eight iterations and an aligned Acc tile occupying at least one quarter of L0C; this admits the independently measured 32/40 KiB Mat-scratch wins while excluding the measured 8 KiB regression and 16 KiB tie on a 128 KiB L0C. Pipeline lowering then schedules depth-two `matmul, matmul, drain, drain` chunks, overlapping tile *i*'s FIXPIPE drain with tile *i+1*'s MAD. A deeper operand pipeline keeps its original stage membership (subject to the allocator's ordinary capacity gate), while Acc membership rotates over two slots. Multiple Acc values, additional stores, nested control flow, indirect uses, loop-carried operands, separately lowered tail groups, and non-canonical drain/yield chains remain unchanged. PTOAS is left unchanged because its planner already separates the reproduced loop's physical Acc placements and device timing showed no benefit from the source marker.

**K-tiling vs M/N-tiling.** When the chooser returns `m == M` and `n == N` the output's **physical** allocation fits L0c, so only the K dimension is tiled (one K-loop). Ordinary capacity is `AlignUp(M, GetL0cMAlignment(dtype)) × N × bytes_c`; on Ascend910B, for example, a logical INT32 `M = 16` accumulator occupies 32 physical rows. The canonical split-K path also passes its operand layouts' Mat-box granularity into the chooser: it charges `AlignUp(AlignUp(m, box_m), l0c_align_m) × AlignUp(n, box_n) × bytes_c`, plus the correspondingly boxed L0A/L0B panels, before selecting a tile. When the chooser returns `m < M` or `n < N`, the output would overflow L0c. The pass tiles the **output** into a `ceil(M/m) × ceil(N/n)` grid. For a fresh plain or biased matmul it computes and places every sub-tile directly; a biased matmul replaces a single-use full bias load, with only sibling loads between it and the matmul, by `[1, n]` window loads from the same effect-free observation region and applies each window once on the first K block of its output tile. Biased matmul currently requires M/N/K to satisfy the operands' boxed layout alignment; plain matmul retains its padded boundary support. For the canonical frontend split-K form, the pass clones the complete source K reduction for every output sub-tile. An oversized full `[M, N]` Acc is never materialized. The output tensor is chained through the per-sub-tile stores in SSA form (`out → out_t0 → out_t1 → …`).

**One physical-size contract.** `ChooseL0Tile`, the existing-pipeline dbC capacity plan, and `InitMemRef` all use the same backend-aware L0C row-footprint helper. Canonical split-K additionally supplies its shared layout-aware Mat box alignment to `ChooseL0Tile`, so selection applies the same M/N padding that its load rebuilding will materialize before the common L0C row padding. A tile admitted by either planner is therefore allocated with the padded shape used by the capacity decision; two live padded accumulators cannot be assigned adjacent logical ranges that overlap physically.

**Fits-L0c chained cast-fold.** A chained matmul whose `[M, N]` result *fits* L0c (no M/N tiling) but feeds a second matmul through a downcast — `c = matmul(a, b); cb = cast(c, bf16); d = matmul(cb, e)` — needs the bf16 intermediate in **Mat** (L1) for the consumer. Left alone, `tile.cast` lowers to a **Vector** `pto.tcvt` (a cube→vector→cube round-trip that overflows the Vec buffer at `[128, 128]`). Instead the pass folds the cast into a **single full-window** Acc→Mat `tile.assemble` — the same `MatScratchPlacer` as the oversized Mat-scratch path, but one `PlaceAt` at offset `(0, 0)` rather than a grid — so the downcast stays on the cube as a FIXPIPE `pto.tinsert`. This is a cast-peephole independent of K tiling: it fires whether the producer was left whole (`k == K`) or K-looped (`k < K`), and only when every use of the cast result is a matmul operand (a non-matmul consumer keeps the Vector cast). The fold also mirrors exactly what FIXPIPE can reproduce — an **`f32 → bf16/f16`** downcast whose round mode is **`rint`** (round-half-to-*even*), FIXPIPE's fixed tie rule — the same on A2/A3 and A5 (the pto-isa CPU reference narrows via `std::bfloat16_t` with no arch branch, and `pto.tinsert` carries no `rmode`; the backends differ only in the scratch dtype, not the rounding). A non-`f32` accumulator (e.g. an `int32` matmul result, which would need a scaled *dequant*), the cast's default **`round`** mode (round-half-*away*), or a directional/truncating mode (`none`/`floor`/`ceil`/`trunc`/`odd`) all keep the Vector `pto.tcvt` — the only path that honors the requested `rmode` — and the pass emits a `PH-AT-010` hint pointing at `mode="rint"`. The same guard (`CastFoldableToFixpipeMat`) gates the oversized Mat-scratch fold below. Oversized results never reach this peephole — their cast is folded per sub-tile by the M/N path above.

**Pipeline position**: After [`LegalizeTileCast`](15-legalize_tile_cast.md), before [`CanonicalizeTileSlice`](17-canonicalize_tile_slice.md) and [`InferTileMemorySpace`](18-infer_tile_memory_space.md). All tile ops are already 2D and memory spaces have not yet been inferred.

**Requirements**: `SSAForm`, `SplitIncoreOrch`, `IncoreTileOps`, `TileOps2D`, `NormalizedStmtStructure`.

**Produces**: same as required (property-preserving rewrite).

**Invalidates**: nothing.

**When to use**: Always, as part of the default tile-stage pipeline. The pass is a no-op when no Mat-resident matmul needs tiling or cast folding and no existing PyPTO-planned L0 pipeline qualifies for automatic accumulator double buffering.

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

For each `tile.matmul`, `tile.matmul_acc`, or `tile.matmul_bias` in an InCore-typed function:

1. **Filter** — operand layout: `(lhs, rhs)` for `tile.matmul`, `(acc, lhs, rhs)` or `(acc, lhs, rhs, init_cond)` for `tile.matmul_acc`, and `(lhs, rhs, bias)` for `tile.matmul_bias`. Arity 4 is accepted **only** for the accumulate kind — a fresh matmul has no accumulator to predicate and `tile.matmul_bias` carries no `init_cond` operand. All matrix operands must be `Var`/`IterArg` (via `AsVarLike`) of `TileType` with static 2D shape. Before the residency filter, a static matrix operand already in its final `Left`/`Right` space is rejected when its physical footprint alone exceeds L0A/L0B; this impossible manual allocation cannot be repaired by reuse or planner choice. Individually legal manual L0 operands remain untouched. For automatic tiling, the right (B) operand must be `memory_space == Mat`. Plain/accumulating matmul retains the historical `Mat`-or-`Vec` left operand support; biased matmul deliberately requires both matrix operands in `Mat`. Its bias must be `[1, N]`, resident in `Mat` or `Bias`, and use the accumulator dtype (FP32 for floating-point matrix operands, INT32 for integer matrix operands). A Mat-resident bias additionally requires a backend-supported Mat→Bias dtype pair. Biased M/N/K must satisfy the matrix operands' layout-derived boxed alignment. Other cases are skipped or diagnosed conservatively.
2. **Pick L0 tile shape** — call `utils::ChooseL0Tile(cfg)` with the active `BackendHandler`'s `GetL0{a,b,c}CapacityBytes()`, `GetL0FractalAlignment()` / `GetMinL0TileDim()`, `GetL0cMAlignment(accumulator_dtype)`, and `GetL0CostModel()` (L1↔L0 bandwidths + MAD issue overhead), plus per-operand element width (`bytes_a/b/c`) read from the call's result type. Biased matmul additionally caps candidate N with `GetBiasCapacityBytes() / bytes_c`; this is a hard auxiliary-SRAM limit, not a roofline coefficient. For a Mat-backed bias reconstructed inside a full-K output grid, the same exhaustive search applies schedule-aware caps: B-stationary keeps one Bias slot, A-stationary and output-stationary with N outer use two, and output-stationary with N in the inner pipeline composes both pipeline memberships and uses four. An already-Bias-resident source is defined outside the grid and always consumes one slot. Candidate L0C legality is `AlignUp(AlignUp(m, box_align_m), l0c_align_m) × AlignUp(n, box_align_n) × bytes_c × dbC <= L0C`; the box alignments default to 1 and are set from the canonical split-K operands' effective Mat layouts when that path will physically pad a window. The boxed M/N extents also gate L0A/L0B. The logical work shape and roofline accounting remain `[m, n]`. `c_read = is_matmul_acc` because `tile.matmul_acc` threads the caller's accumulator through the K-loop's iter-arg (γ_C = 2, doubling the C traffic the model charges). The chooser returns `(m, n, k)` plus the chosen design point — an **exhaustive roofline-`wall` minimum**, not a closed form; see [Cost model & design space](#cost-model--design-space-choosel0tile) below.
3. **Skip if already L0-sized** — `(m, n, k) == (M, N, K)`.
4. **Skip with `PerfHint` for unsupported regimes**:
   - Sub-byte dtypes (cube path doesn't support them) — `PH-AT-003`.
   - `ChooseL0Tile` rejects the configuration — `PH-AT-005`.
5. **Build the K-loop** (per output sub-tile — the whole output when K-only, or each `[m, n]` sub-tile when M/N tiling):
   - `tile.matmul` — iter-arg init is the Acc-resident `tile.create([m, n], dtype, target_memory=Acc)` seed; the loop body is one predicated `tile.matmul_acc(c_iter, sa, sb, ko == 0)` whose predicate overwrites the accumulator on the first L0 block and accumulates into it afterwards. Because `tile.matmul_acc` declares `set_output_reuses_input(0)`, the create / call / yield / return_var chain is one L0C buffer by construction. The peeled `if ko == 0` shape this replaces put one logical value on **two** Acc buffers and merged them at a phi — a state no supported target can realize, since nothing reads L0C but the FIXPIPE drain and there is therefore no Acc→Acc copy to reconcile the arms.
   - `tile.matmul_acc` — iter-arg init is the caller's accumulator directly (its type already matches the per-iter `tile.matmul_acc` output); every iteration is uniform `tile.matmul_acc`. The 3-operand spelling emits no predicate at all, because the caller's accumulator is already live on the first iteration and must never be overwritten. The 4-operand spelling carries the caller's `init_cond`, which means "this is the first K step of the *user's* reduction" — the emitted call ANDs it with the loop's own `ko == 0` ("the first L0 block of that step"), so the accumulator is overwritten only where both hold. A lone straight-line full block (`⌊K/k⌋ == 1`) forwards `init_cond` verbatim: that block *is* the first K block, so composing a statically-true `ko == 0` would only add a foldable node. The peeled partial tail drops the predicate entirely and emits the 3-operand call — it runs at a K offset past every full block, so it is never the first block.
   - `tile.matmul_bias` — the first K block is **head-peeled** out of the loop: a straight-line `tile.matmul_bias` applies the bias exactly once *and* mints the accumulator, and the loop then runs over the remaining full blocks accumulating into it with plain `tile.matmul_acc`. `tile.matmul_bias` carries no `init_cond` operand, so it cannot use the predicated body a plain `tile.matmul` gets; peeling reaches the same one-buffer chain without a predicate. It deliberately does **not** peel into an `IfStmt`: a phi between the fresh `tile.matmul_bias` and the in-place `tile.matmul_acc` would give one logical value two producers on two L0C buffers, which no target can realize (there is no `Acc`→`Acc` copy) and which `MemoryReuse` can only repair after the fact by rewriting buffer identity. With exactly two full blocks the second is straight-line too, since a 1-trip pipeline loop would be degenerate. The pass therefore generates no accumulator phi at all. For N tiling, the matching window is reconstructed as an independent tensor→Mat `tile.load`, then transferred with `tile.move` to Bias (`pto.tload` + `pto.tmov`). A zero-copy one-row Mat `tile.slice` is not used because its boxed `pto.subview` is not PTOAS-legal.
   - Per-iter operand extracts use `tile.extract(src, idx_row, idx_col, [shape], target_memory=Left|Right)` — the SSA-form fusion of the older `tile.slice` (Mat-resident result) + `tile.mov` (Mat→Left/Right) pair. This eliminates the intermediate Mat-resident slice tile and lowers to `pto.textract` rather than `pto.subview`, sidestepping the latter's `valid_row` codegen mismatch. For an output sub-tile at origin `(mi, ni)` the extracts slice `lhs[mi:mi+m, ko:ko+k]` and `rhs[ko:ko+k, ni:ni+n]`; the K-only case is `mi == ni == 0`, `m == M`, `n == N`.
   - **Vec left operand staging** — when the left (A) operand is `Vec`-resident (PV / `score·V`), a single `tile.move(lhs, target_memory=Mat)` is emitted **before** the K-loop and the per-iter Left extract slices from that staged Mat tile (so the extract source is Mat exactly like the QK path). Keeping the Vec→Mat crossing a `tile.move` lets [`ExpandMixedKernel`](22-expand_mixed_kernel.md) recognise it (`CollectCVBoundaryMoves` only matches `tile.move`) and lower it to the cross-core `tpop_from_aiv` handshake (which lands the data in Mat). Extracting straight from the Vec tile would instead leave the operand a dangling cross-boundary free variable on the cube side.
   - The K-loop is `ForKind::Pipeline` with `pipeline_stages=2`.
   - **Non-divisor K (K-boundary peel)** — when the chosen `k` does not divide `K`, the pipelined loop covers only the `⌊K/k⌋` full blocks (bound `⌊K/k⌋·k`) and a straight-line `tile.matmul_acc` peels the partial last block of width `K − ⌊K/k⌋·k`; when only one full block fits (`⌊K/k⌋ == 1`), a single straight-line full block + tail replaces the loop. With `K` and `k` both 16-aligned (the cube fractal), the peeled tail width `K − ⌊K/k⌋·k` is itself 16-aligned — an ordinary `matmul_acc` block, no masking. (ptoas requires 16-aligned tile cols, so the operand dimensions must be 16-aligned; non-16-aligned `K` is **not** supported.) The chooser only returns a non-divisor `k` under `ChooseL0Tile`'s `allow_k_boundary`, which this pass sets; when the full (16-aligned) K fits one L0 block the chooser returns `k == K` (no loop) instead. A **non-16-aligned `K` is rejected outright** — there is no valid K-tiling (any peeled tail or whole-K block would have non-fractal cols), so the chooser returns no candidate and the pass skips the matmul with a `PH-AT-007` hint rather than emit invalid extracts.
6. **M/N tiling (when `m < M` or `n < N`)** — the physical `[M, N]` output Acc footprint overflows L0c.

   For a **fresh `tile.matmul` or `tile.matmul_bias` whose result is consumed by exactly one 2D `tile.store(c, base, out)`**, with that consumer store as the first non-load statement after the matmul, the pass tiles the output into a `ceil(M/m) × ceil(N/n)` grid: for each sub-tile origin `(mi, ni)` it computes the `[m, n]` (partial on the boundary, `min(m, M-mi) × min(n, N-ni)`) sub-tile and emits `tile.store(c_sub, [base_r + mi, base_c + ni], out_prev)`. Biased matmul also reloads the corresponding `[1, n]` bias window from the defining tensor into Mat and moves it to Bias, where the cube broadcasts it over that sub-tile's M rows. Requiring the store to be the first non-load prevents deferred grid emission from crossing an effect. When **K spans ≥ 2 L0 blocks**, each sub-tile is an independent **pipelined K-loop**. When **`k == K`**, the grid is emitted as nested loops over the divisible interior so [`LowerPipelineLoops`](29-lower_pipeline_loops.md) double-buffers the moving operand. The outer loop owns the stationary panel; output-stationary versus A/B-stationary loop order follows the chooser's design point. The L-shaped partial boundary is peeled into straight-line partial tiles, so `m`/`n` need not divide `M`/`N`. The stores chain the output tensor in SSA form; the final store's result replaces the original store downstream.

   A **canonical frontend split-K reduction** is matched as an adjacent `tile.create([M, N])` full-output accumulator placeholder, one pipeline carrying that value through the K reduction, and one 2D output store. The reduction itself is accepted in **either source spelling**:

   - **peeled** — an `if` with `tile.matmul` for the first K block and `tile.matmul_acc` for later blocks, merged at the branch's phi;
   - **predicated** — a single 4-operand `tile.matmul_acc(acc, lhs, rhs, init_cond)` accumulating into the loop's only iter-arg, with no branch and no phi.

   In the predicated spelling `init_cond` must be a split-K seed test: `<x> == 0` where `x` is this loop's own induction variable, or a body-local scalar defined as `<induction variable> * <nonzero constant>` (the `k0 = kb * K_TILE` spelling every peeled kernel already uses for its `if`). Matching the induction variable by identity is what rejects a caller-supplied flag or another loop's variable — neither is evidence that the accumulator is overwritten on the first K block. Such a triplet is left untouched and reported with a `PH-AT-006` hint rather than silently losing M/N tiling.

   The two distinct operands must come from direct per-iteration GM→Mat loads whose static shape and valid shape cover the full rectangular panels; the loop may contain only scalar address calculations besides those loads. The pass moves the output grid outside the source K loop: for each `(mi, ni)`, it creates a legal `[m, n]` Acc, clones the complete K loop, narrows both loads to that output window, completes the full K reduction, then stores the tile. The retiled loop keeps whichever spelling the source used, so both spellings produce the same output grid, the same narrowed loads and the same store chain, differing only in that reduction statement. The ordinary call-level AutoTile rewrite subsequently applies any needed inner-K tiling to the narrowed calls — where it does, the peeled spelling yields two inner K-loops behind the branch while the predicated one yields a single loop whose predicate is `init_cond and ko == 0`. This ordering is essential: slicing the existing full Acc would still require the impossible `[M, N]` L0C allocation.

   The following M/N regimes remain **deferred**: an arbitrary standalone `tile.matmul_acc` with a caller-owned accumulator (it does not match the canonical chain above), a `Vec` left operand (PV path), a biased matmul whose already-`Bias`-resident source would need an unsupported Bias-to-Bias N sub-window, a Mat bias that is not a single-use 2D load with only sibling loads before the matmul, and a result consumed on-chip that is **not** consumed entirely as a matmul operand. Treating every non-load statement as a barrier preserves the original bias snapshot across intervening stores or other effects; removing the replaced full load avoids redundant traffic. A result consumed entirely as a matmul operand takes the Mat-scratch placement below.

   **Placement (direct-store vs Mat-scratch).** Both grids hand each `[m, n]` Acc sub-tile to a `SubtilePlacer`. The **`DirectGmPlacer`** stores it to the DDR output (`tile.store`, above). The **`MatScratchPlacer`** instead keeps the whole `[M, N]` result on-chip in an L1/**Mat** scratch — created once with `tile.create(target_memory=Mat)` (whose implicit NZ TileView `col_major/row_major` is the matmul-operand layout), then each sub-tile is assembled in place via `tile.assemble(scratch, sub, [mi, ni])`. For the low-precision bf16/f16 scratch used by the chained path this Acc→Mat writeback lowers to FIXPIPE `pto.tinsert`; a supported same-dtype full-window assemble uses `pto.subview` + `pto.tmov`. The pass selects Mat-scratch when the matmul result's uses are **all** matmul-operand reads *and* the `[M, N]` scratch fits the backend handler's Mat capacity (`GetMatCapacityBytes()`) — a conservative necessary-condition gate that keeps oversized chained matmuls on the deferred `PH-AT-006` path instead of emitting an impossible on-chip allocation (a full packed-peak check that also accounts for coexisting Mat tensors is a follow-up). On selection it remaps the result `Var` to the scratch so the consumer reads it on-chip. `tile.assemble`'s `set_output_memory_inherit_input()` makes the chain share one Mat base, so the assemble is in place (no unsupported Mat→Mat preservation copy). Both the split-K (unrolled, constant offsets) and full-K (pipelined, loop-variable offsets) grids drive either placer.

   > **Legacy-PyPTO limitation — operand-stationary chained producers + L0 packing.** A chained-matmul (Mat-scratch) producer shares L0 with its consumer (sequential; the intermediate stays in L1, never DDR — the `L0C→L1→L0A` trip). An A/B-stationary producer pins one monolithic full-L0 operand buffer while a pipelined consumer needs multiple smaller buffers. The legacy PYPTO planner's `AllocateMemoryAddr` bump-stacks reuse classes and never subdivides a freed region (for example, reusing a 64 KB producer buffer for one 32 KB consumer slot wastes 32 KB, and the other slot spills → L0 overflow), so the pass forces Mat-scratch producers to **output-stationary only under that planner**. Enabling the experimental dbC opt-in can still expose this limitation on some output-stationary chains; one current reproducer requests 96 KB from a 64 KB L0B. `DSA_RP` and `PTOAS` already place buffers from their actual lifetimes, retain the chooser's operand-stationary schedule, and pack the consumer's smaller buffers into the released full-size range. Adding equivalent lifetime-aware subdivision to the legacy allocator is tracked by [issue #1908](https://github.com/hw-native-sys/pypto/issues/1908).
7. **Rewrite the enclosing `SeqStmts`** — substitute uses of the original matmul's `Var` (K-only) or the consumer store's result (M/N) with the new `return_var`. Substitution is scoped to the `SeqStmts` that contains the rewrite, so it does not leak into sibling regions.

8. **Recognize existing L0 pipelines** — independently of chooser-driven rewrites, inspect each static PyPTO-planned `ForKind::Pipeline` with `pipeline_stages=F ≥ 2` and a trip count divisible by `F`. Requiring complete groups avoids a separately lowered tail whose allocator placement can need another Acc slot. The loop's flat body must contain exactly one plain `tile.matmul` with static `Left`/`Right` operands; its selected moving operand must have a recognized direct per-iteration Mat→L0 producer, while the stationary operand is defined outside the loop. The body must also contain one canonical drain chain whose result is yielded back through its matching iter-arg: direct-to-GM `tile.store(acc, ..., iter_arg_i)` or Acc-to-Mat `tile.assemble(iter_arg_i, acc, ...)`. The direct path requires at least four iterations. The Mat-scratch path requires at least eight iterations and an aligned Acc footprint of at least `ceil(L0C/4)`, a separate conservative gate because its cheaper drain is not represented by the shared direct-GM roofline. Any other Acc definition/read or store-like operation defers the loop. Before attaching `pipeline_double_buffer_c=true` and `pipeline_overlap_stores=false`, the pass conservatively sums every static Acc value in the function. Ordinary cube accumulators count once because lowering serializes them; every other Acc producer is multiplied by the product of its enclosing source-pipeline stage depths, matching the physical memberships lowering may request. One additional aligned slot is then added for every profitable loop. All loops remain unchanged unless this post-lowering upper bound fits L0C, so enabling dbC cannot force another pipeline to shed buffering depth merely because its replicated Acc footprint was omitted. Explicitly attributed loops are left unchanged. For `F > 2`, lowering emits repeated depth-two `MMSS` chunks and rotates Acc memberships modulo two, while operand memberships retain depth `F`.

The pass is a `ProgramPass` and walks each function with an `IRMutator`; functions are returned unchanged when no rewrite fires (no `MutableCopy` cost for matmul-free programs).

## Cost model & design space (`ChooseL0Tile`)

`ChooseL0Tile` picks the L0 GEMM tile by an **exhaustive roofline search**, not a closed form. For every legal aligned `(m, n, k)` — each a multiple of `GetL0FractalAlignment()`, with `AlignUp(m, l0c_align_m) × n` used for the L0C budget — it estimates wall-clock in core cycles and returns the minimum:

- `wall ≈ max(C_load, C_mad) + C_drain` when the FIXPIPE L0C→L1 drain is exposed (single L0C), or
- `wall ≈ max(C_load, C_mad, C_drain) + min(compute, C_drain) / T` when the drain is hidden behind compute (double-buffered L0C, `T` output tiles). The `+ min(…)/T` term is the pipeline **fill/drain bubble** — the first tile's compute (or the last tile's drain) has no partner to overlap, so the ideal all-hidden `T·max` roofline undercounts by one tile's non-dominant pipe (50% of the smaller pipe for two output tiles and ≈25% at a 2×2 grid). This keeps dbC=2 from being over-picked on small grids.

`C_load` is the L1→L0A/L0B operand traffic under the chosen loop order, scaled by the per-buffer bandwidths from `GetL0CostModel()` (on-device MTE1 sweep: `bw_l0a≈130`, `bw_l0b≈85` B/cyc, ~1.52:1); `C_mad` is the cube MAD cost (per-`TMATMUL` issue overhead × K-fractal count). `C_drain` is the FIXPIPE L0C writeback, charged **per output tile** as a **per-M-row** cost: `⌈M/m⌉·⌈N/n⌉ · (drain_fixed + m·(max(drain_row, bytes_c·n/bw_drain) + drain_penalty·(odd(⌈n/N0⌉)−1)))`. A direct fit of an on-device FIXPIPE sweep: FIXPIPE addresses one M-row of the `N1 M1 M0 N0` FRACTAL_NZ accumulator at a time (so cost ∝ `m`), each row a grouped `nburst`/`loop` over the `N1 = ⌈n/N0⌉` N-fractals (`N0 = 32/bytes_c = 8` for the fp32 L0C). The per-row cost is `max(floor, throughput)` — a fixed burst-issue **floor** `drain_row` (row addressing/setup, N-independent) that dominates narrow N, or the fractal **throughput** `bytes_c·n/bw_drain` that dominates wide N (crossover ~n=131) — plus the **misalignment** residual: a non-power-of-two fractal count serializes the odd part `odd(N1)−1` into extra passes at `drain_penalty` per M-row (the predicate is a **non-power-of-two `N1`**, not literally `N%32`: `n=80 → odd(10)=5` is penalized, and so is `n=96 → odd(12)=3` even though `96%32=0`; aligned power-of-two `N1` such as `n=128 → 16` pays nothing). Because the drain count is `⌈M/m⌉·⌈N/n⌉`, **splitting the output (M/N) adds drains but splitting K does not** (partial sums accumulate in one L0C, drained once per `(m,n)` block). The per-M-row form makes the chooser prefer **wide-N / small-M** tiles (fewer FIXPIPE rows per drain) and correctly prices a misaligned-N tile so it is not over-selected — e.g. `320×320` lands an aligned `(160,128,64)` instead of the drain-bound `160×80`. Device-validated (drain 0.93–1.09×, loads R²=0.993). The search is exhaustive over **all** legal `k` per `(m, n)` (not the largest legal k — `⌈K/k⌉·⌈k/kt⌉` is non-monotone in `k` when `kt ≠ align_k`). Wall ties break lexicographically on `(padded_compute, ⌈K/k⌉, C_load, …)`; the `C_load` key picks the lower-hidden-load aspect among MAD-bound `(m,n)`↔`(n,m)` ties (L0B's slower bandwidth favours fewer m-blocks).

The search ranges over the **design space** `P = (m, n, k, stationarity, dbC)`:

- **stationarity** `{output, A, B}` — which operand is pinned across the L0 grid. This *derives* the per-operand double-buffer depths (`dbA`/`dbB`): the moving operand(s) double-buffer (depth 2), the stationary one single-buffers (depth 1). They are not searched independently.
- **dbC** `{1, 2}` — whether the L0C accumulator is double-buffered to overlap the FIXPIPE drain with the next tile's compute.

A **realizable mask** (the `allow_a_stationary` /
`allow_b_stationary` / `allow_double_buffer_c` config gates) restricts which
design points are *enumerated and emitted* to those whose lowering exists. A
gated-off axis is not scored. The pass opens the **A/B-stationary** gates: the
held operand is pinned **single-buffered** across the moving grid (`k == K`) by
a `ForKind::Sequential` outer loop in `BuildFullKPipelined`; a pipelined outer
loop would require twice its full-L0 budget.

**dbC=2** is the two-accumulator L0C ping-pong in which tile *i*'s FIXPIPE
drain overlaps tile *i+1*'s MAD. It is enabled unconditionally for
`memory_planner=DSA_RP` and `memory_planner=PTOAS`. The legacy `PYPTO`
planner retains an experimental opt-in
(`PassContext(enable_pypto_l0c_double_buffer=True)`, default off) because
issue #1908 can still overflow operand buffers in chained Mat-scratch layouts.
`BuildFullKPipelined` tags the moving loop with
`kPipelineDoubleBufferCAttr`, and `CanonicalizeIOOrder` floats both stores
below both matmuls (`matmul, matmul, store, store`) to make the two accumulator
lifetimes co-live.

The planners preserve that intent differently. For eligible PTOAS pipelines,
[`LowerPipelineToSlots`](28-lower_pipeline_to_slots.md) expresses the stages as
slots in one allocation; declined loops flow to
[`LowerPipelineLoops`](29-lower_pipeline_loops.md), and PTOAS assigns the
resulting stage buffers distinct offsets itself. `PYPTO` uses the flat depth-2
`pipeline_membership` emitted by `LowerPipelineLoops`, and `MemoryReuse`'s
capacity gate (#1475) keeps the buffers distinct to the affordable depth.
`DSA_RP` also skips `MemoryReuse`; it represents pipeline-stage separations as
hard constraints, runs its bounded strict search first, and relaxes only
pipeline-intent separations to soft penalties if that search finds no
capacity-fitting placement. dbC=2 requires full-K and at least two **full**
tiles on the moving inner axis; the stationary outer axis may have one tile.
The emitted loop orientation follows the chooser's stationarity/hoist decision,
and a peeled partial boundary does not count as a ping-pong stage. Thus rows
outer admits a 1×2 grid, while columns outer admits a 2×1 grid. The Mat-scratch
(`Acc→Mat`, `tile.assemble`) drain is floated the same way. A
`PassManager` built under one planner and run under another fails loudly
because its pass list and chooser gates must agree. The cost-model formulas
are gate-independent. See
[`30-canonicalize_io_order.md`](30-canonicalize_io_order.md) for the co-live
float and runtime validation of the distinct `{0, L0C/2}` offsets.

The full-K and moving-inner restrictions above apply only to chooser-emitted M/N
tiling. The separate existing-pipeline recognizer does not alter the chooser's
design space: for the legacy PyPTO planner, it applies the same two-Acc mechanism
to the canonical stationary-panel pattern after the conservative
function-level Acc fit described above.

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
            c_acc = pl.tile.matmul_acc(c_iter, sa, sb, ko == 0)
            c = pl.yield_(c_acc)
        # c (the yield-LHS) holds the accumulated Acc-typed result.
        ...
```

### Row-narrowed left operand: the seed declares `compact`

When the left operand's valid rows are not provably its physical rows, the
accumulator seed is emitted as a *narrowed, compact* placeholder:

```python
c_l0_init_storage = pl.tile.create([64, 128], pl.INT32, target_memory=Acc, compact=True)
c_l0_init = pl.tile.set_validshape(c_l0_init_storage, 16, 128)
```

`mad` takes M from the L0A operand's **valid** rows and lays the product out in
L0C with an N-fractal stride of `ceil(M/16)*16` (pto-isa `TMatmul.hpp`), and
only a compact tile makes a reader recompute that pitch instead of using the
physical row count. `tile.matmul` gets the mode from
`StampCompactForNarrowedAccRows`, but `tile.matmul_acc` **inherits** its
accumulator operand's mode — so a non-compact seed drags every accumulate step,
and the `tile.store` / `tile.tpush_to_aiv` after the loop, back to the physical
pitch and scrambles every N-fractal above the first (issues #2470, #2510).

The mode is *declared* on the `tile.create` rather than stamped onto the type
afterwards because a declaration is what survives: `InferTileMemorySpace`
re-deduces every call whose arguments changed, discarding pass-applied type
refinements, while a kwarg is re-read by the deducer each time.
`AccCompactValid` verifies the resulting contract.

### `tile.matmul_acc`

The caller's accumulator threads through the iter-arg directly; no `init_cond` predicate is emitted, because that accumulator is already live on the first iteration and must never be overwritten:

```python
for ko, (c_iter,) in pl.pipeline(0, K, k, init_values=(acc_init,), stage=2):
    sa = pl.tile.extract(a_mat, 0, ko, [m, k], target_memory=Left)
    sb = pl.tile.extract(b_mat, ko, 0, [k, n], target_memory=Right)
    c_new = pl.tile.matmul_acc(c_iter, sa, sb)
    c = pl.yield_(c_new)
# c (the yield-LHS) holds the accumulated Acc-typed result.
```

### `tile.matmul_acc` with a caller-supplied `init_cond`

`pl.tile.matmul_acc(acc_init, a_mat, b_mat, init_cond=user_cond)` is the split-K idiom: `user_cond` marks the first K step of the user's own reduction. The pass composes it with the `ko == 0` its K-loop introduces, so the accumulator is overwritten only on the first L0 block of that step:

```python
for ko, (c_iter,) in pl.pipeline(0, k_full, k, init_values=(acc_init,), stage=2):
    sa = pl.tile.extract(a_mat, 0, ko, [m, k], target_memory=Left)
    sb = pl.tile.extract(b_mat, ko, 0, [k, n], target_memory=Right)
    c_new = pl.tile.matmul_acc(c_iter, sa, sb, user_cond and ko == 0)
    c_kmain = pl.yield_(c_new)
# Peeled partial tail (k does not divide K): unpredicated 3-operand form —
# it runs at K offset k_full > 0, so it is never the first block.
sat = pl.tile.extract(a_mat, 0, k_full, [m, k_eff], target_memory=Left)
sbt = pl.tile.extract(b_mat, k_full, 0, [k_eff, n], target_memory=Right)
c = pl.tile.matmul_acc(c_kmain, sat, sbt)
```

A literal `True` predicate is composed the same way rather than folded back to `tile.matmul`: folding it would mint a second L0C buffer, and the backend emitter already selects the right instruction for a compile-time predicate — see [`init_cond` in the operator reference](../ir/05-operators.md), which also covers what the emitter does with the `ko == 0` this pass generates once `LowerPipelineLoops` folds it per replica.

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
    c_t1_acc = pl.tile.matmul_acc(c_iter, sa, sb, ko == 0)
    c_t1 = pl.yield_(c_t1_acc)
out_t1 = pl.store(c_t1, [256, 0], out_t0)  # store sub-tile to out[256:512, 0:256]
```

Boundary sub-tiles (when `m`/`n` do not divide `M`/`N`) have logical extents `[min(m, M-mi), min(n, N-ni)]` — e.g. a 256×256 FP32 matmul on Ascend910B (chooser picks `m = 192, n = 160`) tiles into logical sub-tiles of `192×160`, `192×96`, `64×160`, `64×96`. For the canonical split-K rewrite, each operand's physical Mat shape is rounded up to the effective boxed-layout granularity while `valid_shape` retains the logical extent. This granularity is part of chooser capacity legality, including when a full logical tile such as INT8 N=80 boxes to physical N=96. `tile.matmul` / `tile.matmul_acc` propagate the same physical/valid distinction to the loop-carried Acc, and `tile.store` transfers only the valid rectangle at the original logical offset. For example, an INT8 Right tile with a 16-column N tail is represented physically as `[K, 32]` with `valid_shape=[K, 16]`, producing an Acc with physical N=32 and valid N=16.

For a canonical split-K chain, the same grid encloses the **source** reduction rather than slicing its final Acc. In issue #2232, the logical INT32 `[16, 1152]` result occupies `32 × 1152 × 4 = 144 KiB` physically on Ascend910B, so it is split along N. Each generated N tile runs all eight source K blocks and stores its result before the next N tile starts.

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
| `GetBiasCapacityBytes()` | Bias-table capacity; caps `tile.matmul_bias` candidate N |
| `SupportsMatToBiasMove(src, dst)` | PTO-ISA dtype legality for materialising a Mat-resident bias |
| `GetMatCapacityBytes()` | Mat (L1) capacity for Mat-scratch gate |
| `GetL0FractalAlignment()` | M/N/K alignment grid for the chooser |
| `GetL0cMAlignment(dtype)` | Physical M-row alignment for L0C capacity; 32 for INT32 on Ascend910B |
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
| `tile.matmul` over static-2D operands (Mat left, or Vec left for PV) + Mat right, output fits L0c | Rewritten to 2-stage pipelined K-loop (predicated `tile.matmul_acc` body — one Acc buffer, no phi); a Vec left operand is staged to Mat first |
| `tile.matmul` (plain, Mat left, Mat right) whose output exceeds L0c, consumed by one 2D `tile.store` | M/N-tiled: `ceil(M/m) × ceil(N/n)` grid of sub-tile K-loops, each stored straight to the output (direct-store) |
| `tile.matmul` (plain) whose output exceeds L0c, consumed *entirely* as a matmul operand (chained matmul), and whose `[M, N]` scratch fits Mat/L1 | M/N-tiled into an L1/**Mat** scratch (per-sub-tile Acc→Mat `tile.assemble`), kept on-chip for the consumer (Mat-scratch) |
| `tile.matmul` whose output *fits* L0c, downcast via `tile.cast(c, bf16/f16)` whose result is consumed *entirely* as a matmul operand (chained) | Cast-fold: one full-window Acc→Mat `tile.assemble` (cube `pto.tinsert`); the cast is dropped — no Vector `pto.tcvt` round-trip |
| `tile.matmul_acc` over static-2D operands (Mat left, or Vec left for PV) + Mat right, output fits L0c | Rewritten to 2-stage pipelined K-loop (uniform `matmul_acc` body) |
| The same call written with a caller-supplied `init_cond` (4 operands) | Also K-tiled: the loop body carries `init_cond and ko == 0`, a lone straight-line full block carries `init_cond` verbatim, and the peeled tail stays 3-operand |
| Canonical split-K `create([M,N])` → pipeline (either `matmul` first + loop-carried `matmul_acc` later, or one predicated `matmul_acc(acc, lhs, rhs, <loop var> == 0)`) → one 2D store, physical output exceeds L0c | M/N-tiled outside the K loop; each `[m,n]` tile completes the full K reduction before it is stored |
| `tile.matmul[_acc]` with a Vec **right** operand | Skipped (the B operand must feed L0B from L1) |
| `tile.matmul_bias` with static Mat matrix operands and a `[1,N]` Mat/Bias source, output fits L0c but K does not | K-tiled; the first block is a head-peeled straight-line `matmul_bias` that initializes each output tile once, and the loop over the remaining blocks uses `matmul_acc` (no `IfStmt`, one L0C buffer) |
| `tile.matmul_bias` with static Mat matrix operands and a single-use, full rectangular `[1,N]` Mat bias load separated from the call only by sibling loads, output exceeds L0c, with one direct store or only later matmul-operand uses | M/N-tiled after replacing the full load with per-N tensor→Mat window loads; placed through the same direct-GM or Mat-scratch strategies as fresh `tile.matmul` |
| `tile.matmul_bias` with a Vec left operand, or an already-Bias-resident source requiring N tiling | Skipped; the new biased path requires native Mat operands and cannot emit Bias-to-Bias sub-window extracts |
| Already L0-sized matmul (`(m, n, k) == (M, N, K)`) | Untouched |
| Output exceeds L0c but no M/N placement applies — non-canonical standalone `matmul_acc`, Vec left, a non-matmul-operand consumer, or a chained-matmul scratch whose `[M, N]` exceeds Mat/L1 | Skipped with `PerfHint` (`PH-AT-006`) |
| `K` not a multiple of the cube fractal (16) | Skipped with `PerfHint` (`PH-AT-007`) — no fractal-aligned K-tiling |
| Sub-byte dtypes | Skipped with `PerfHint` |
| Non-InCore functions (Orchestration, Opaque) | Untouched |

## Diagnostics

The pass emits `PerfHint` diagnostics rather than failing when it declines to rewrite — the original matmul is left intact and runs through the rest of the pipeline unchanged. PerfHint codes:

| Code | Meaning |
| ---- | ------- |
| `PH-AT-003` | Sub-byte dtype on operand or accumulator |
| `PH-AT-005` | `ChooseL0Tile` rejected the configuration |
| `PH-AT-006` | Output exceeds L0c but no supported M/N placement applies — for `tile.matmul_acc`, this specifically means a caller-owned accumulator outside the canonical create/split-K-pipeline/store chain, or a 4-operand call inside an otherwise canonical chain whose `init_cond` is not a split-K seed test on the loop's induction variable. It also covers a Vec left operand, an already-Bias-resident `tile.matmul_bias` source that would need N sub-windowing, a result consumed on-chip that is not entirely a matmul operand, or a chained-matmul scratch that exceeds Mat/L1 capacity. The canonical split-K case from issue #2232 does not emit this hint. |
| `PH-AT-007` | Non-16-aligned `K` — no fractal-aligned K-tiling exists (any peeled tail or whole-K block would have non-fractal cols), so the matmul is left untouched |
| `PH-AT-008` | `ChooseL0Tile` returned a fallback configuration with a perf hint message |
| `PH-AT-009` | Backend needs a bf16/f16 on-chip Mat scratch (e.g. Ascend910B) but the oversized chained-matmul intermediate is f32 — cast the matmul result to bf16/f16 before the consumer matmul; left on the deferred path |
| `PH-AT-010` | A fits-L0c chained-matmul cast cannot fold onto the cube FIXPIPE (which narrows `f32 → bf16/f16` with round-half-to-even only): the source is non-f32, or the round mode is not `rint` (e.g. the default `round`, or `floor`/`ceil`/`trunc`/`odd`/`none`). Kept on the Vector `pto.tcvt` path — a cube→vector→cube round-trip that may overflow the Vec buffer at large `[M, N]`. Cast an f32 result with `mode="rint"` to keep it on the cube. |
| `PH-AT-011` | A biased matmul cannot form a legal Bias window: unsupported Mat→Bias dtype pair, non-Mat matrix operand, partial/dynamic/non-load-backed N window, insufficient Bias capacity, or layout-misaligned M/N/K. The call is unchanged. |

## See also

- [`FlattenTileNdTo2D`](13-flatten_tile_nd_to_2d.md) — upstream pass; produces the static-2D Mat-resident tile shapes this pass consumes
- [`InferTileMemorySpace`](18-infer_tile_memory_space.md) — downstream pass; bridges Vec/Acc accumulators that this pass deliberately leaves alone
- [`LowerPipelineLoops`](29-lower_pipeline_loops.md) — consumes the `ForKind::Pipeline` + `pipeline_stages=2` produced here
