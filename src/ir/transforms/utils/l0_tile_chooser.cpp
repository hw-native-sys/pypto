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

#include "pypto/ir/transforms/utils/l0_tile_chooser.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <sstream>
#include <vector>

#include "pypto/core/logging.h"

namespace pypto {
namespace ir {
namespace utils {

namespace {

// ===========================================================================
// Small numerical helpers
// ===========================================================================

constexpr int64_t AlignDown(int64_t x, int64_t a) { return (x / a) * a; }

constexpr int64_t AlignUp(int64_t x, int64_t a) { return ((x + a - 1) / a) * a; }

constexpr int64_t CeilDiv(int64_t a, int64_t b) { return (a + b - 1) / b; }

// ===========================================================================
// Candidate scoring
// ===========================================================================

struct Candidate {
  int m = 0;
  int n = 0;
  int k = 0;
  int64_t traffic = 0;
  int64_t padded_compute = 0;
};

// Largest legal aligned k for a given (m, n). Returns 0 if no legal k exists.
// Callers must pass m, n >= min_m, min_n (positive).
//
// When `allow_padding` is false, the returned k must additionally divide K:
// the downstream `AutoTileMatmulL0` pass has no K-boundary handling yet (see
// `auto_tile_matmul_l0_pass.cpp` PH-AT-007) and will skip the matmul if
// `K % k != 0`, leaving the caller-supplied Mat tile to overflow L0.  Walking
// k_max down to the largest aligned divisor of K avoids that silent skip and
// keeps the chooser-pass contract intact.
int LargestLegalK(int m, int n, const L0TileConfig& cfg, int64_t A0, int64_t B0) {
  const int64_t k_from_a = A0 / m;
  const int64_t k_from_b = B0 / n;
  // Padded callers want k bounded by the *aligned-up* problem size — without
  // AlignUp, K=17 with align_k=16 would cap at max(17, 16)=17 then align-down
  // to 16, so the natural padded k=32 would never be considered.
  const int64_t k_from_problem =
      cfg.allow_padding ? std::max<int64_t>(AlignUp(static_cast<int64_t>(cfg.K), cfg.align_k), cfg.min_k)
                        : cfg.K;
  int64_t k_max = std::min({k_from_a, k_from_b, static_cast<int64_t>(k_from_problem)});
  k_max = AlignDown(k_max, cfg.align_k);
  if (k_max < cfg.min_k) return 0;
  if (!cfg.allow_padding) {
    // O(K / align_k) per call; constant in IR size.
    while (k_max >= cfg.min_k && cfg.K % k_max != 0) {
      k_max -= cfg.align_k;
    }
    if (k_max < cfg.min_k) return 0;
  }
  return static_cast<int>(k_max);
}

// Largest legal aligned n for a given m. n is bounded by C0 / m (since m*n <=
// C0) and by allowing at least min_k to fit in B0 (so n <= B0 / min_k).
int LargestLegalN(int m, const L0TileConfig& cfg, int64_t B0, int64_t C0) {
  const int64_t n_from_c = C0 / m;
  const int64_t n_from_b = B0 / cfg.min_k;
  const int64_t n_from_problem = cfg.allow_padding ? std::max(cfg.N, cfg.min_n) : cfg.N;
  int64_t n_max = std::min({n_from_c, n_from_b, static_cast<int64_t>(n_from_problem)});
  n_max = AlignDown(n_max, cfg.align_n);
  if (n_max < cfg.min_n) return 0;
  return static_cast<int>(n_max);
}

// Largest legal aligned m for a given n (symmetric to LargestLegalN).
int LargestLegalM(int n, const L0TileConfig& cfg, int64_t A0, int64_t C0) {
  const int64_t m_from_c = C0 / n;
  const int64_t m_from_a = A0 / cfg.min_k;
  const int64_t m_from_problem = cfg.allow_padding ? std::max(cfg.M, cfg.min_m) : cfg.M;
  int64_t m_max = std::min({m_from_c, m_from_a, static_cast<int64_t>(m_from_problem)});
  m_max = AlignDown(m_max, cfg.align_m);
  if (m_max < cfg.min_m) return 0;
  return static_cast<int>(m_max);
}

// Cost model from L0_TILING.md §5.
//
//   A traffic ≈ bytes_a * M * K * ceil(N / n)
//   B traffic ≈ bytes_b * K * N * ceil(M / m)
//   C traffic ≈ gamma_c * bytes_c * M * N
//
// gamma_c is 2 when the matmul reads its accumulator (C = beta*C + A@B),
// else 1.
int64_t EstimateTraffic(int m, int n, int k, const L0TileConfig& cfg) {
  (void)k;  // k does not appear in the traffic model under C-stationary scheduling.
  const int64_t M = cfg.M;
  const int64_t N = cfg.N;
  const int64_t K = cfg.K;
  const int64_t a_traffic = static_cast<int64_t>(cfg.bytes_a) * M * K * CeilDiv(N, n);
  const int64_t b_traffic = static_cast<int64_t>(cfg.bytes_b) * K * N * CeilDiv(M, m);
  const int64_t gamma_c = cfg.c_read ? 2 : 1;
  const int64_t c_traffic = gamma_c * static_cast<int64_t>(cfg.bytes_c) * M * N;
  return a_traffic + b_traffic + c_traffic;
}

int64_t PaddedComputeVolume(int m, int n, int k, const L0TileConfig& cfg) {
  return CeilDiv(cfg.M, m) * m * CeilDiv(cfg.N, n) * n * CeilDiv(cfg.K, k) * k;
}

// Lexicographic ordering: lower is better.
//   (traffic, padded_compute, ceil(K/k), -m*n, -k)
bool Better(const Candidate& a, const Candidate& b, const L0TileConfig& cfg) {
  if (a.traffic != b.traffic) return a.traffic < b.traffic;
  if (a.padded_compute != b.padded_compute) return a.padded_compute < b.padded_compute;
  const int64_t a_kblocks = CeilDiv(cfg.K, a.k);
  const int64_t b_kblocks = CeilDiv(cfg.K, b.k);
  if (a_kblocks != b_kblocks) return a_kblocks < b_kblocks;
  const int64_t a_area = static_cast<int64_t>(a.m) * a.n;
  const int64_t b_area = static_cast<int64_t>(b.m) * b.n;
  if (a_area != b_area) return a_area > b_area;  // larger area preferred
  return a.k > b.k;                              // larger k preferred
}

// Try a candidate (m, n) by picking the largest legal k and recording the
// scored result. No-op if the candidate is illegal.
void TryCandidate(int m, int n, const L0TileConfig& cfg, int64_t A0, int64_t B0, int64_t C0,
                  std::vector<Candidate>& out) {
  if (m < cfg.min_m || n < cfg.min_n) return;
  // Without padding, the chosen tile must not exceed the problem dimensions.
  // Aligned-down boundary tiles (m <= M but M % m != 0) are still permitted —
  // those are handled by the outer loop, not by chooser-introduced padding.
  if (!cfg.allow_padding) {
    if (m > cfg.M || n > cfg.N) return;
  }
  m = static_cast<int>(AlignDown(m, cfg.align_m));
  n = static_cast<int>(AlignDown(n, cfg.align_n));
  if (m < cfg.min_m || n < cfg.min_n) return;
  if (static_cast<int64_t>(m) * n > C0) return;
  const int k = LargestLegalK(m, n, cfg, A0, B0);
  if (k == 0) return;
  Candidate c;
  c.m = m;
  c.n = n;
  c.k = k;
  c.traffic = EstimateTraffic(m, n, k, cfg);
  c.padded_compute = PaddedComputeVolume(m, n, k, cfg);
  out.push_back(c);
}

}  // namespace

L0TileResult ChooseL0Tile(const L0TileConfig& cfg) {
  // 1. Validate inputs.
  CHECK(cfg.M > 0 && cfg.N > 0 && cfg.K > 0)
      << "ChooseL0Tile: M, N, K must all be positive (got " << cfg.M << ", " << cfg.N << ", " << cfg.K << ")";
  CHECK(cfg.l0a_bytes > 0 && cfg.l0b_bytes > 0 && cfg.l0c_bytes > 0)
      << "ChooseL0Tile: L0 capacities must be positive";
  CHECK(cfg.bytes_a > 0 && cfg.bytes_b > 0 && cfg.bytes_c > 0)
      << "ChooseL0Tile: element byte sizes must be positive";
  CHECK(cfg.min_m > 0 && cfg.min_n > 0 && cfg.min_k > 0)
      << "ChooseL0Tile: minimum tile dimensions must be positive";
  CHECK(cfg.align_m > 0 && cfg.align_n > 0 && cfg.align_k > 0)
      << "ChooseL0Tile: tile alignments must be positive";

  // Without padding, the problem dimensions themselves must already meet the
  // cube minimum. Callers (the pass) should pre-screen and skip with a
  // perf-hint rather than rely on the chooser to fabricate padding.
  if (!cfg.allow_padding) {
    CHECK(cfg.M >= cfg.min_m)
        << "ChooseL0Tile: allow_padding=false but M=" << cfg.M << " is below the cube minimum tile dimension "
        << cfg.min_m << ". The pass should skip this matmul with a perf hint instead of calling the chooser.";
    CHECK(cfg.N >= cfg.min_n) << "ChooseL0Tile: allow_padding=false but N=" << cfg.N
                              << " is below the cube minimum tile dimension " << cfg.min_n;
    CHECK(cfg.K >= cfg.min_k) << "ChooseL0Tile: allow_padding=false but K=" << cfg.K
                              << " is below the cube minimum tile dimension " << cfg.min_k;
  }

  // 2. Effective capacity (in elements). Halve for double-buffered operands.
  const int64_t A0 = static_cast<int64_t>(cfg.l0a_bytes) /
                     (static_cast<int64_t>(cfg.bytes_a) * (cfg.double_buffer_a ? 2 : 1));
  const int64_t B0 = static_cast<int64_t>(cfg.l0b_bytes) /
                     (static_cast<int64_t>(cfg.bytes_b) * (cfg.double_buffer_b ? 2 : 1));
  const int64_t C0 = static_cast<int64_t>(cfg.l0c_bytes) /
                     (static_cast<int64_t>(cfg.bytes_c) * (cfg.double_buffer_c ? 2 : 1));

  CHECK(A0 >= static_cast<int64_t>(cfg.min_m) * cfg.min_k)
      << "ChooseL0Tile: L0a capacity " << A0 << " elements is too small to fit the minimum tile ("
      << cfg.min_m << " x " << cfg.min_k << ")";
  CHECK(B0 >= static_cast<int64_t>(cfg.min_n) * cfg.min_k)
      << "ChooseL0Tile: L0b capacity " << B0 << " elements is too small to fit the minimum tile ("
      << cfg.min_k << " x " << cfg.min_n << ")";
  CHECK(C0 >= static_cast<int64_t>(cfg.min_m) * cfg.min_n)
      << "ChooseL0Tile: L0c capacity " << C0 << " elements is too small to fit the minimum tile ("
      << cfg.min_m << " x " << cfg.min_n << ")";

  // 3. Continuous optimum (L0_TILING.md §6).
  //    Minimises bytes_a * M / n + bytes_b * N / m subject to m*n <= C0.
  const double m_star_raw =
      std::sqrt(static_cast<double>(cfg.bytes_b) * static_cast<double>(cfg.N) * static_cast<double>(C0) /
                (static_cast<double>(cfg.bytes_a) * static_cast<double>(cfg.M)));

  // 4. Generate candidates. We accumulate then deduplicate before scoring.
  std::vector<Candidate> candidates;

  // Continuous-optimum candidates around m_star: floor and ceil aligned.
  const int64_t m_floor =
      std::max<int64_t>(cfg.min_m, AlignDown(static_cast<int64_t>(m_star_raw), cfg.align_m));
  const int64_t m_ceil = std::max<int64_t>(cfg.min_m, AlignUp(static_cast<int64_t>(m_star_raw), cfg.align_m));
  for (int64_t m_cand : {m_floor, m_ceil}) {
    const int n_legal = LargestLegalN(static_cast<int>(m_cand), cfg, B0, C0);
    TryCandidate(static_cast<int>(m_cand), n_legal, cfg, A0, B0, C0, candidates);
  }

  // Symmetric: candidates around n_star = C0 / m_star.
  const double n_star_raw = (m_star_raw > 0.0) ? static_cast<double>(C0) / m_star_raw : 0.0;
  const int64_t n_floor =
      std::max<int64_t>(cfg.min_n, AlignDown(static_cast<int64_t>(n_star_raw), cfg.align_n));
  const int64_t n_ceil = std::max<int64_t>(cfg.min_n, AlignUp(static_cast<int64_t>(n_star_raw), cfg.align_n));
  for (int64_t n_cand : {n_floor, n_ceil}) {
    const int m_legal = LargestLegalM(static_cast<int>(n_cand), cfg, A0, C0);
    TryCandidate(m_legal, static_cast<int>(n_cand), cfg, A0, B0, C0, candidates);
  }

  // Full-C candidate: pad/align M and N, take if it fits.
  {
    const int m_full =
        static_cast<int>(std::max<int64_t>(cfg.min_m, AlignUp(static_cast<int64_t>(cfg.M), cfg.align_m)));
    const int n_full =
        static_cast<int>(std::max<int64_t>(cfg.min_n, AlignUp(static_cast<int64_t>(cfg.N), cfg.align_n)));
    if (cfg.allow_padding ||
        (m_full <= AlignUp(static_cast<int64_t>(cfg.M), cfg.align_m) && m_full <= cfg.M && n_full <= cfg.N)) {
      TryCandidate(m_full, n_full, cfg, A0, B0, C0, candidates);
    }
  }

  // Defensive bottom-floor candidate at the minimum tile shape, in case the
  // continuous optimum and full-C all fall outside the legal region.
  TryCandidate(cfg.min_m, cfg.min_n, cfg, A0, B0, C0, candidates);

  CHECK(!candidates.empty()) << "ChooseL0Tile: no legal (m, n, k) tile found for M=" << cfg.M
                             << ", N=" << cfg.N << ", K=" << cfg.K
                             << ". This indicates the hardware capacity is below the configured "
                             << "minimum tile shape; check L0a/L0b/L0c bytes and min_m/min_n/min_k.";

  // 5. Pick best by lex (traffic, padded_compute, k_blocks, -area, -k).
  const Candidate* best = &candidates.front();
  for (const auto& c : candidates) {
    if (Better(c, *best, cfg)) best = &c;
  }

  L0TileResult result;
  result.m = best->m;
  result.n = best->n;
  result.k = best->k;
  result.estimated_traffic_bytes = best->traffic;
  result.padded_compute_volume = best->padded_compute;

  // 6. Perf-hint diagnostics for borderline cases (callers may forward via
  //    EmitDiagnostics with severity PerfHint).
  if (cfg.M < cfg.min_m || cfg.N < cfg.min_n || cfg.K < cfg.min_k) {
    std::stringstream ss;
    ss << "Matmul shape (M=" << cfg.M << ", N=" << cfg.N << ", K=" << cfg.K
       << ") is below the cube minimum tile dimension " << cfg.min_m << "; tile shape padded up to ("
       << result.m << ", " << result.n << ", " << result.k
       << "). Consider reshaping or fusing with adjacent ops to amortise the padding.";
    result.perf_hint = ss.str();
  }

  return result;
}

}  // namespace utils
}  // namespace ir
}  // namespace pypto
