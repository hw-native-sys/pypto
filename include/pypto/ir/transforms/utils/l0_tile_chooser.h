/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

#ifndef PYPTO_IR_TRANSFORMS_UTILS_L0_TILE_CHOOSER_H_
#define PYPTO_IR_TRANSFORMS_UTILS_L0_TILE_CHOOSER_H_

#include <cstdint>
#include <string>

namespace pypto {
namespace ir {
namespace utils {

/**
 * @brief Which GEMM operand is pinned (held resident) across the L0 tiling loops.
 *
 * Axis 2 of the design space (loop permutation -> stationarity; see
 * DESIGN_SPACE.md). The choice fixes the per-operand double-buffer depth: the
 * stationary operand is single-buffered (depth 1, full L0 buffer); the moving
 * operand(s) are double-buffered (depth 2).
 *
 *   - kOutputStationary: pin the L0C accumulator (the base when k < K); both
 *     operands stream (dbA = dbB = 2).
 *   - kAStationary: pin the left operand A (k == K); A loaded once per row
 *     (dbA = 1), B streams (dbB = 2).
 *   - kBStationary: pin the right operand B (k == K); B loaded once per column
 *     (dbB = 1), A streams (dbA = 2).
 */
enum class Stationarity { kOutputStationary, kAStationary, kBStationary };

/**
 * @brief Inputs to ChooseL0Tile.
 *
 * Captures the problem dimensions and the backend's hardware constraints in a
 * single struct so callers (the AutoTileMatmulL0 pass and tests) can build it
 * once and pass it around.
 */
struct L0TileConfig {
  // Problem dimensions (must be > 0). The L1 / Mat tile shape is static —
  // dynamic shapes are resolved earlier in the pipeline.
  int M = 0;
  int N = 0;
  int K = 0;

  // L0 capacities in bytes (typically read from BackendHandler::GetL0?CapacityBytes).
  uint32_t l0a_bytes = 0;
  uint32_t l0b_bytes = 0;
  uint32_t l0c_bytes = 0;

  // Element sizes in bytes for the three operand tiles.
  // Defaults match BF16 x BF16 -> FP32 GEMM.
  uint32_t bytes_a = 2;
  uint32_t bytes_b = 2;
  uint32_t bytes_c = 4;

  // Lower bounds and alignment for the L0 tile shape (m, n, k).
  // Defaults reflect the cube fractal across Ascend AI Core generations.
  int min_m = 16;
  int min_n = 16;
  int min_k = 16;
  int align_m = 16;
  int align_n = 16;
  int align_k = 16;

  // --- Realizable mask -----------------------------------------------------
  // The chooser scores the design space (stationarity x dbC), but only EMITS
  // design points whose pass lowering exists. These gates select the realizable
  // subset; they all default to today's mask -- output-stationary, single L0C --
  // so a caller that sets nothing gets the algorithm the pass realizes now. Each
  // new lowering flips one gate; the cost model is untouched. Operand
  // double-buffer depths (dbA, dbB) are NOT gates: they are derived from the
  // chosen stationarity (moving operand -> 2, stationary -> 1).

  // Allow A-stationary / B-stationary loop orders (pin an operand, k == K). When
  // false only output-stationary is considered.
  bool allow_a_stationary = false;
  bool allow_b_stationary = false;

  // Allow the L0C double-buffer (dbC = 2): two accumulators ping-ponged so the
  // FIXPIPE drain overlaps the next tile's compute (half the L0C budget, drain
  // hidden). When false only single-L0C (dbC = 1) is considered.
  bool allow_double_buffer_c = false;

  // Whether the matmul reads its accumulator (C = beta * C + A @ B). When
  // true, C traffic doubles in the cost estimate.
  bool c_read = false;

  // Closed-form cost-model parameters (filled from
  // BackendHandler::GetL0CostModel). Bandwidths are in BYTES PER CYCLE so the
  // chooser scores wall-clock directly: wall ~= max(C_load, C_mad) + C_drain.
  // Defaults are Ascend a2a3 (910B) for the common BF16 x BF16 -> FP32 GEMM, so
  // standalone callers (tests) get a sane model without wiring a backend.
  double bw_a = 200.0;                // L1->L0A bytes/cycle (op-sim work-fit; datasheet 238).
  double bw_b = 132.0;                // L1->L0B bytes/cycle (op-sim work-fit; ~1.5:1 vs L0A, not 2:1).
  double bw_drain = 118.0;            // FIXPIPE L0C drain bytes/cycle (op-sim work-fit; per-drain slope).
  double drain_fixed_cycles = 245.0;  // Per-FIXPIPE-drain fixed overhead (penalizes M/N-split, not K-split).
  int mad_head = 6;                   // Fixed per-TMATMUL issue overhead.
  int mad_k_fractal_bytes = 32;       // Cube K-fractal width (kt = this / bytes_a).

  // Whether the chooser may pick a tile dimension larger than the problem
  // dimension (i.e. pad M / N / K up to reach `min_m` / `min_n` / `min_k`).
  //
  // Default false: at L0 we do not pad up the problem dimensions. The cube
  // minimum (16) must already be satisfied by the input shape; callers that
  // see smaller shapes should skip the matmul with a perf hint rather than
  // ask the chooser to fabricate padding. Note this flag does NOT control
  // boundary-tile handling — when `M % m != 0` the outer loop's last
  // iteration is naturally partial; that is the pass's responsibility, not
  // the chooser's.
  bool allow_padding = false;

  // Allow a non-divisor final K block (K-boundary peel) -- valid ONLY for a
  // 16-aligned K. The chosen k need not divide K, but K must be a multiple of
  // align_k so the peeled tail K - floor(K/k)*k is itself 16-aligned (ptoas requires
  // 16-aligned tile cols). When the full (16-aligned) K fits one L0a/L0b block the
  // chooser returns k == K (single block, no loop); otherwise it tiles K and the pass
  // peels the partial last block. A NON-16-aligned K admits no legal tile at all (any
  // tail or whole-K block would have non-fractal cols): the chooser returns none and
  // the pass skips the matmul with a PerfHint (PH-AT-007). When false the chooser
  // walks k down to an aligned divisor of K (legacy — no K-boundary handling).
  bool allow_k_boundary = false;
};

/**
 * @brief Output of ChooseL0Tile.
 *
 * On success, `(m, n, k)` is the chosen L0 tile shape and `perf_hint` is
 * empty. On a fallback the chooser still returns a legal `(m, n, k)` and
 * `perf_hint` contains a diagnostic string the caller may forward via
 * EmitDiagnostics with severity PerfHint.
 */
struct L0TileResult {
  int m = 0;
  int n = 0;
  int k = 0;

  // Estimated L1 <-> L0 traffic in bytes for the chosen tile (lower is
  // better). Retained for tests / inspection; the chooser now ranks by
  // estimated_cost_cycles, not this value.
  int64_t estimated_traffic_bytes = 0;

  // Estimated wall-clock for the chosen tile in core cycles (the roofline
  // objective the chooser ranks by; lower is better):
  //   double_buffer_c == false : max(C_load, C_mad) + C_drain  (drain exposed)
  //   double_buffer_c == true  : max(C_load, C_mad, C_drain)   (drain hidden)
  int64_t estimated_cost_cycles = 0;

  // Padded compute volume = ceil(M/m)*m * ceil(N/n)*n * ceil(K/k)*k.
  // Used as a tie-breaker.
  int64_t padded_compute_volume = 0;

  // --- Chosen design point (beyond the (m, n, k) tile) ---------------------
  // The pinned operand / loop order. Output-stationary unless allow_a_stationary
  // / allow_b_stationary opened the operand-stationary routes (k == K). The
  // per-operand double-buffer depths (dbA, dbB) are NOT reported separately:
  // they are fully determined by this field (moving operand -> depth 2,
  // stationary -> depth 1 / full buffer); see the Stationarity enum doc.
  Stationarity stationarity = Stationarity::kOutputStationary;

  // For the full-K output-stationary emit (BuildFullKPipelined at k == K), which
  // operand is hoisted to the outer loop (loaded once per outer step, reused
  // across the inner sweep): true = hold A (rows outer), false = hold B (cols
  // outer). This is the SAME bandwidth-weighted decision the wall cost is scored
  // under (LoadCycles' min-hoist), recorded here so the emit obeys the scored
  // hoist by construction rather than re-deriving it from raw byte traffic (the
  // ~200:132 L0A:L0B ratio makes "fewest bytes" and "fewest cycles" disagree).
  // Only meaningful for output-stationary k == K; A/B-stationary force the loop
  // order from `stationarity`, and split-K (k < K) uses a different emitter.
  bool os_holds_a = true;

  // Whether the chooser chose to double-buffer L0C (two accumulators ping-ponged
  // to overlap the FIXPIPE drain). True only when allow_double_buffer_c was set,
  // the single-L0C optimum already tiles the output (a full [M, N, K] tile that
  // fits one L0C is left untiled instead), the tile forms a >= 2x2 output grid,
  // reduces K in one pass (k == K), and the double-buffered wall was strictly
  // lower.
  //
  // NOTE: the caller must REALIZE this with a genuine two-accumulator schedule
  // (two co-live L0C buffers). A full-K emitter that threads the output as a
  // single iter-arg chain yields ONE L0C buffer regardless of
  // pipeline_overlap_stores, so it must not set allow_double_buffer_c until that
  // lowering exists — otherwise the chooser only shrinks the tile (budgeting
  // L0C/2) without hiding the drain, a regression.
  bool double_buffer_c = false;

  // Empty on success. Non-empty when the chooser couldn't pick an "ideal"
  // tile but landed on a legal fallback (e.g., M < min_m so we padded up).
  std::string perf_hint;
};

/**
 * @brief Pick the minimum-wall L0 GEMM design point under the roofline model.
 *
 * Scores the roofline cost model (see DESIGN_SPACE.md in the pto-isa cost-model
 * study) over the design space and returns the minimum-wall design point
 *   P = (m, n, k, stationarity, dbA, dbB, dbC),
 * where dbA / dbB are derived from `stationarity` (moving operand -> depth 2,
 * stationary -> depth 1) and prefetch depth is fixed at <= 2 (no multistage).
 * A "realizable mask" (the allow_* gates) restricts which points may be emitted;
 * it defaults to today's realizable subset {output-stationary, dbC = 1}, so a
 * caller that opens no gate gets exactly the algorithm the pass realizes now.
 *
 *   1. Enumerate the allowed (stationarity, dbC) combinations. For each, derive
 *      the operand depths and L0 budgets:
 *        A0 = L0A/(bytes_a*dbA), B0 = L0B/(bytes_b*dbB), C0 = L0C/(bytes_c*dbC).
 *   2. Enumerate every legal aligned (m, n) with m*n <= C0, and for each (m, n)
 *      every legal aligned k (m*k <= A0, n*k <= B0, k >= min_k; k | K when neither
 *      padding nor k_boundary; plus k == K when the full K fits one block). ALL k
 *      are scored -- ceil(K/k)*ceil(k/kt) is non-monotone in k when kt != align_k,
 *      so the largest legal k is not always wall-optimal.
 *   3. Score each tile by wall (cycles):
 *        C_mad  = ceil(M/m)*ceil(N/n) *
 *                 (ceil(K/k)*mad_head + cpr*ceil(m/16)*Kfrac*ceil(n/16))
 *          Kfrac = floor(K/k)*ceil(k/kt) + ceil((K - floor(K/k)*k)/kt)
 *                  (the K-peel tail is scored at its own width, not rounded to k)
 *        C_drain = ceil(M/m)*ceil(N/n) * (drain_fixed + gamma_c*bytes_c*m*n/BW_drain)
 *                  (per output tile: fixed issue overhead + its bytes at BW_drain;
 *                   M/N-split adds drains, K-split does not)
 *        C_load (BW-weighted, by stationarity):
 *          OS full-K : min(held-A, held-B) route (hoist the cheaper operand; the
 *                      recorded os_holds_a drives the emit) — held-A / held-B as below
 *          OS split-K: ba*M*K*ceil(N/n)/BW_A + bb*K*N*ceil(M/m)/BW_B  (both re-streamed)
 *          A-stat    : ba*M*K/BW_A           + bb*K*N*ceil(M/m)/BW_B   (k==K, held-A)
 *          B-stat    : ba*M*K*ceil(N/n)/BW_A + bb*K*N/BW_B             (k==K, held-B)
 *        wall   = max(C_load, C_mad) + C_drain        (dbC == 1, drain exposed)
 *               = max(C_load, C_mad, C_drain)         (dbC == 2, drain hidden)
 *      with kt = mad_k_fractal_bytes/bytes_a, cpr = bytes_a/2. Ties break by lex
 *      (padded_compute, ceil(K/k), C_load, -m*n, -k) -- the C_load key picks the
 *      lower-hidden-load aspect among MAD-bound (m,n)<->(n,m) ties.
 *   4. Pick the global minimum-wall point across all enumerated combinations.
 *      A-stationary / B-stationary require k == K (operand residency). dbC = 2
 *      additionally requires a >= 2x2 grid, k == K, and that the single-L0C
 *      optimum already tiles. The
 *      (m, n) grid is bounded by m*n <= C0 and the operand caps, k by K/align_k:
 *      O((C0/align^2) * (K/align_k)) per matmul -- a hardware constant, independent
 *      of IR size; the chooser runs once per matmul op, so the pass stays linear
 *      in the IR.
 *
 * @param cfg All inputs (problem dims + hardware + realizable-mask gates).
 * @return Chosen design point and metadata. Throws ValueError if the inputs
 *   are invalid (e.g., non-positive dims, capacities too small to fit any
 *   legal tile).
 */
L0TileResult ChooseL0Tile(const L0TileConfig& cfg);

}  // namespace utils
}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORMS_UTILS_L0_TILE_CHOOSER_H_
