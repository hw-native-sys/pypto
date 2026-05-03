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
 * @brief Inputs to ChooseL0Tile.
 *
 * Captures the problem dimensions and the backend's hardware constraints in a
 * single struct so callers (the AutoTileMatmulL0 pass and tests) can build it
 * once and pass it around.
 */
struct L0TileConfig {
  // Problem dimensions (must be > 0). The L1 / Mat tile shape is static per
  // the L0_TILING.md design — dynamic shapes are resolved earlier in the
  // pipeline.
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

  // Double-buffering schedule knobs. Halve the effective capacity for each
  // operand whose buffer is double-allocated, so candidate (m, n, k) accounts
  // for the ping-pong before evaluating capacity constraints.
  bool double_buffer_a = true;
  bool double_buffer_b = true;
  bool double_buffer_c = false;

  // Whether the matmul reads its accumulator (C = beta * C + A @ B). When
  // true, C traffic doubles in the cost estimate.
  bool c_read = false;

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
  // better). Used by tests; the pass does not consume this value directly.
  int64_t estimated_traffic_bytes = 0;

  // Padded compute volume = ceil(M/m)*m * ceil(N/n)*n * ceil(K/k)*k.
  // Used as a tie-breaker.
  int64_t padded_compute_volume = 0;

  // Empty on success. Non-empty when the chooser couldn't pick an "ideal"
  // tile but landed on a legal fallback (e.g., M < min_m so we padded up).
  std::string perf_hint;
};

/**
 * @brief Pick an approximately-optimal L0 tile shape (m, n, k).
 *
 * Closed-form O(1) algorithm following the L0 tiling design note:
 *   1. Compute effective capacities A0, B0, C0 (account for double-buffering).
 *   2. Compute the continuous optimum m_star = sqrt(bytes_b * N * C0 /
 *      (bytes_a * M)) and n_star = C0 / m_star.
 *   3. Generate a constant number of aligned candidates around (m_star, n_star)
 *      plus a "full C tile" candidate if M*N <= C0.
 *   4. For each (m, n), pick the largest aligned k satisfying
 *      m*k <= A0, n*k <= B0, k >= min_k.
 *   5. Score each candidate by lex (traffic, padded_compute, ceil(K/k),
 *      -m*n, -k) and return the best.
 *
 * @param cfg All inputs (problem dims + hardware + schedule knobs).
 * @return Chosen tile shape and metadata. Throws ValueError if the inputs
 *   are invalid (e.g., non-positive dims, capacities too small to fit any
 *   legal tile).
 */
L0TileResult ChooseL0Tile(const L0TileConfig& cfg);

}  // namespace utils
}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORMS_UTILS_L0_TILE_CHOOSER_H_
