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
#include <optional>
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

// One enumerated design-space combination (the axes beyond the tile shape).
// dbA / dbB are derived from `stat`; dbc is the L0C double-buffer choice.
struct Regime {
  Stationarity stat = Stationarity::kOutputStationary;
  bool dbc = false;  // true => dbC = 2 (drain hidden)
};

// Per-operand double-buffer depth derived from stationarity: the stationary
// operand is single-buffered (false), the moving operand(s) double-buffered.
struct OperandDB {
  bool a = true;
  bool b = true;
};
OperandDB DeriveOperandDB(Stationarity stat) {
  switch (stat) {
    case Stationarity::kAStationary:
      return {/*a=*/false, /*b=*/true};  // A pinned
    case Stationarity::kBStationary:
      return {/*a=*/true, /*b=*/false};  // B pinned
    case Stationarity::kOutputStationary:
      break;
  }
  return {/*a=*/true, /*b=*/true};  // both stream
}

struct Candidate {
  int m = 0;
  int n = 0;
  int k = 0;
  int64_t traffic = 0;
  int64_t cost_cycles = 0;
  int64_t padded_compute = 0;
  double load_cycles = 0;  // C_load — a wall-tie-break (hidden under max() when MAD-bound)
  Regime regime;           // the (stationarity, dbC) this tile was scored under
};

// All legal k values for a fixed (m, n), ascending. k is a REAL search axis:
// the MAD term ceil(K/k)*ceil(k/kt) is NOT monotonic in k when kt != align_k
// (e.g. bytes_a=1 -> kt=32 > align_k=16), so the largest legal k is not always
// wall-optimal. The caller must score every returned k, not just the largest.
//
// Legality mirrors the three regimes:
//   * legacy (no relaxation): aligned k that DIVIDE K -- the pass has no
//     K-boundary handling, so a non-divisor k would be skipped (PH-AT-007).
//   * allow_k_boundary: any aligned k <= capacity, PLUS k == K when the full K
//     reduction fits one L0 block (a single block; K is 16-aligned, so k == K is
//     too -- ptoas requires 16-aligned tile cols).
//   * allow_padding: aligned k bounded by the aligned-up problem size.
std::vector<int> EnumerateLegalKs(int m, int n, const L0TileConfig& cfg, int64_t A0, int64_t B0) {
  std::vector<int> ks;
  const int64_t k_from_a = A0 / m;
  const int64_t k_from_b = B0 / n;
  const int64_t cap = std::min(k_from_a, k_from_b);  // max k that fits L0a and L0b
  const int64_t k_problem =
      cfg.allow_padding ? std::max<int64_t>(AlignUp(static_cast<int64_t>(cfg.K), cfg.align_k), cfg.min_k)
                        : static_cast<int64_t>(cfg.K);
  const int64_t k_hi = AlignDown(std::min(cap, k_problem), cfg.align_k);
  // allow_k_boundary admits a NON-DIVISOR k (the K-peel) ONLY when K is itself
  // align_k-aligned: then every full block AND the peeled tail (K - floor(K/k)*k)
  // are 16-aligned, which ptoas requires for tile cols. A non-16-aligned K has no
  // valid k-tiling (a non-fractal tail or whole-K block), so it yields no candidate
  // here and the pass skips the matmul (PH-AT-007) rather than emit invalid extracts.
  const bool peel_ok = cfg.allow_k_boundary && (cfg.K % cfg.align_k == 0);
  for (int64_t k = cfg.min_k; k <= k_hi; k += cfg.align_k) {
    if (!cfg.allow_padding && !peel_ok && cfg.K % k != 0) continue;  // divisors only
    ks.push_back(static_cast<int>(k));
  }
  return ks;
}

// L1<->L0 operand + drain traffic in BYTES for the chosen tile under its regime.
// Inspection-only (the chooser ranks by the roofline wall, not this value); the
// reload counts follow the regime's stationarity, mirroring LoadCycles so the
// reported traffic is honest for A/B-stationary too (not just output-stationary):
//   A traffic ≈ bytes_a * M * K * (ceil(N/n), or 1 when A is held / A-stationary)
//   B traffic ≈ bytes_b * K * N * (ceil(M/m), or 1 when B is held / B-stationary)
//   C traffic ≈ gamma_c * bytes_c * M * N   (gamma_c = 2 when the accumulator is read)
int64_t EstimateTraffic(int m, int n, int k, const L0TileConfig& cfg, const Regime& r) {
  (void)k;  // k does not change the C-stationary reload counts.
  const int64_t M = cfg.M;
  const int64_t N = cfg.N;
  const int64_t K = cfg.K;
  const int64_t ceil_n = CeilDiv(N, n);
  const int64_t ceil_m = CeilDiv(M, m);
  int64_t a_traffic = 0;
  int64_t b_traffic = 0;
  switch (r.stat) {
    case Stationarity::kAStationary:
      a_traffic = static_cast<int64_t>(cfg.bytes_a) * M * K;  // A loaded once (k == K)
      b_traffic = static_cast<int64_t>(cfg.bytes_b) * K * N * ceil_m;
      break;
    case Stationarity::kBStationary:
      a_traffic = static_cast<int64_t>(cfg.bytes_a) * M * K * ceil_n;
      b_traffic = static_cast<int64_t>(cfg.bytes_b) * K * N;  // B loaded once (k == K)
      break;
    case Stationarity::kOutputStationary:
      a_traffic = static_cast<int64_t>(cfg.bytes_a) * M * K * ceil_n;
      b_traffic = static_cast<int64_t>(cfg.bytes_b) * K * N * ceil_m;
      break;
  }
  const int64_t gamma_c = cfg.c_read ? 2 : 1;
  const int64_t c_traffic = gamma_c * static_cast<int64_t>(cfg.bytes_c) * M * N;
  return a_traffic + b_traffic + c_traffic;
}

int64_t PaddedComputeVolume(int m, int n, int k, const L0TileConfig& cfg) {
  return CeilDiv(cfg.M, m) * m * CeilDiv(cfg.N, n) * n * CeilDiv(cfg.K, k) * k;
}

// Roofline cost model (output-stationary, single L0C -- the algorithm
// the pass realizes today). See l0_tile_chooser.h and the pto-isa cost-model
// study (DESIGN_SPACE.md). All costs are in core cycles.

// Cube MAD cost over the full padded grid. Per-TMATMUL cost is
//   mad_head + cpr * ceil(m/16) * ceil(k/kt) * ceil(n/16),
// with kt = mad_k_fractal_bytes/bytes_a and cpr = bytes_a/2 (1 BF16 / 2 FP32).
int64_t MadCycles(int m, int n, int k, const L0TileConfig& cfg) {
  const int64_t kt = std::max<int64_t>(1, cfg.mad_k_fractal_bytes / static_cast<int64_t>(cfg.bytes_a));
  const int64_t cpr = std::max<int64_t>(1, static_cast<int64_t>(cfg.bytes_a) / 2);
  const int64_t per_tile =
      cfg.mad_head + cpr * CeilDiv(m, cfg.align_m) * CeilDiv(k, kt) * CeilDiv(n, cfg.align_n);
  return CeilDiv(cfg.M, m) * CeilDiv(cfg.N, n) * CeilDiv(cfg.K, k) * per_tile;
}

// L1->L0 load cost (cycles). The MTE1 pipe is shared, so A and B loads serialize;
// each is weighted by its port bandwidth (the 2:1 L0A/L0B asymmetry). The reload
// counts depend on the stationarity / reuse route (DESIGN_SPACE.md):
//   OS     : A re-streamed once per n-block (ceil(N/n)); B once per m-block
//            (ceil(M/m)).
//   A-stat : A held -> loaded once (k == K); B once per m-block.
//   B-stat : B held -> loaded once (k == K); A once per n-block.
double LoadCycles(int m, int n, const L0TileConfig& cfg, const Regime& r) {
  const double M = cfg.M, N = cfg.N, K = cfg.K;
  const double ceil_n = static_cast<double>(CeilDiv(cfg.N, n));
  const double ceil_m = static_cast<double>(CeilDiv(cfg.M, m));
  double a_bytes = 0.0;
  double b_bytes = 0.0;
  switch (r.stat) {
    case Stationarity::kAStationary:
      a_bytes = static_cast<double>(cfg.bytes_a) * M * K;  // A loaded once (k == K)
      b_bytes = static_cast<double>(cfg.bytes_b) * K * N * ceil_m;
      break;
    case Stationarity::kBStationary:
      a_bytes = static_cast<double>(cfg.bytes_a) * M * K * ceil_n;
      b_bytes = static_cast<double>(cfg.bytes_b) * K * N;  // B loaded once (k == K)
      break;
    case Stationarity::kOutputStationary:
      a_bytes = static_cast<double>(cfg.bytes_a) * M * K * ceil_n;
      b_bytes = static_cast<double>(cfg.bytes_b) * K * N * ceil_m;
      break;
  }
  return a_bytes / cfg.bw_a + b_bytes / cfg.bw_b;
}

// L0C drain cost. Shape-invariant (every output element drains once; gamma_c=2
// when the accumulator is read), so under dbC=1 it shifts wall uniformly across
// candidates and does not change the pick -- included so the value is honest.
double DrainCycles(const L0TileConfig& cfg) {
  const double gamma_c = cfg.c_read ? 2.0 : 1.0;
  return gamma_c * static_cast<double>(cfg.bytes_c) * cfg.M * cfg.N / cfg.bw_drain;
}

// Roofline wall in cycles. With a single L0C (drain_hidden=false) the FIXPIPE
// drain is exposed -- the cube stalls on each tile's store -- so it ADDS to the
// pipe maximum. With L0C double-buffered (drain_hidden=true) the drain overlaps
// the next tile's compute, so it JOINS the maximum instead of adding.
int64_t WallCycles(int m, int n, int k, const L0TileConfig& cfg, const Regime& r) {
  const double compute = std::max(LoadCycles(m, n, cfg, r), static_cast<double>(MadCycles(m, n, k, cfg)));
  const double drain = DrainCycles(cfg);
  const double wall = r.dbc ? std::max(compute, drain) : compute + drain;
  // Guard the float->int cast: a non-finite or out-of-exact-range wall would be UB.
  // Given the validated positive bandwidths and aligned-bounded dims this never fires.
  INTERNAL_CHECK(std::isfinite(wall) && wall <= 9007199254740992.0)  // 2^53
      << "Internal error: ChooseL0Tile wall cycles " << wall << " is non-finite or out of range";
  return static_cast<int64_t>(std::llround(wall));
}

// Ordering: lower is better. Primary key is the roofline wall (cycles); ties
// (equal cycles) break by lex (padded_compute, ceil(K/k), C_load, -m*n, -k).
bool Better(const Candidate& a, const Candidate& b, const L0TileConfig& cfg) {
  if (a.cost_cycles != b.cost_cycles) return a.cost_cycles < b.cost_cycles;
  if (a.padded_compute != b.padded_compute) return a.padded_compute < b.padded_compute;
  const int64_t a_kblocks = CeilDiv(cfg.K, a.k);
  const int64_t b_kblocks = CeilDiv(cfg.K, b.k);
  if (a_kblocks != b_kblocks) return a_kblocks < b_kblocks;
  // Among wall-ties (MAD-bound: the load is hidden under max(C_load, C_mad)),
  // prefer the lower HIDDEN load. The L0A/L0B bandwidth asymmetry (2:1) favours
  // one aspect, and that aspect wins the moment the real shape leaves the perfect
  // MAD-bound knee. This disambiguates aspect-swapped (m,n)<->(n,m) tiles that the
  // symmetric area/k keys below cannot -- otherwise the ascending-m scan would
  // pick the load-suboptimal small-m aspect.
  if (a.load_cycles != b.load_cycles) return a.load_cycles < b.load_cycles;
  const int64_t a_area = static_cast<int64_t>(a.m) * a.n;
  const int64_t b_area = static_cast<int64_t>(b.m) * b.n;
  if (a_area != b_area) return a_area > b_area;  // larger area preferred
  return a.k > b.k;                              // larger k preferred
}

// Build the scored candidate for an explicit (m, n, k). nullopt if illegal
// (below min, or the output overflows L0C, or m/n exceed the problem dims
// without padding). The operand-capacity legality of k is the caller's job
// (EnumerateLegalKs only returns k that fit L0a/L0b).
std::optional<Candidate> MakeCandidate(int m, int n, int k, const L0TileConfig& cfg, int64_t C0,
                                       const Regime& regime) {
  if (m < cfg.min_m || n < cfg.min_n || k < cfg.min_k) return std::nullopt;
  // Without padding, the chosen tile must not exceed the problem dimensions.
  // Aligned-down boundary tiles (m <= M but M % m != 0) are still permitted —
  // the full-K emitter peels the partial boundary into a straight-line tail.
  if (!cfg.allow_padding && (m > cfg.M || n > cfg.N)) return std::nullopt;
  if (static_cast<int64_t>(m) * n > C0) return std::nullopt;
  Candidate c;
  c.m = m;
  c.n = n;
  c.k = k;
  c.traffic = EstimateTraffic(m, n, k, cfg, regime);
  c.cost_cycles = WallCycles(m, n, k, cfg, regime);
  c.padded_compute = PaddedComputeVolume(m, n, k, cfg);
  c.load_cycles = LoadCycles(m, n, cfg, regime);
  c.regime = regime;
  return c;
}

// Enumerate the legal aligned (m, n, k) grid under capacity C0 for one
// design-space regime and return its minimum-wall tile. EVERY legal k per
// (m, n) is scored (k is not monotone in wall -- see EnumerateLegalKs), so this
// is a true exhaustive search over the regime's tile shapes, not (m, n) with a
// largest-k shortcut.
//
// require_2d: only tiles forming a >= 2x2 output grid are considered -- L0C
//   double-buffering overlaps drains in the inner pipelined loop, which needs
//   >= 2 tiles on each axis.
// require_full_k: only tiles that reduce K in a single pass (k == K) are
//   considered -- needed for the operand-stationary routes (A/B held across K)
//   and for the dbC=2 ping-pong (realized only by the full-K pipelined emitter).
//
// Complexity: O((C0 / align^2) * (K / align_k)) per matmul -- the (m, n) grid is
// bounded by m*n <= C0 and the L0A/L0B capacities, k by K/align_k. A hardware
// constant per op, independent of IR size. The chooser runs once per matmul op
// (matmul ops are O(N)), so the pass stays linear in the IR.
std::optional<Candidate> EnumerateBest(const L0TileConfig& cfg, const Regime& regime, int64_t A0, int64_t B0,
                                       int64_t C0, bool require_2d, bool require_full_k) {
  const int64_t m_hi = cfg.allow_padding ? AlignUp(static_cast<int64_t>(cfg.M), cfg.align_m) : cfg.M;
  const int64_t n_hi = cfg.allow_padding ? AlignUp(static_cast<int64_t>(cfg.N), cfg.align_n) : cfg.N;
  std::optional<Candidate> best;
  for (int64_t m = cfg.min_m; m <= m_hi; m += cfg.align_m) {
    // n >= min_n must fit m*n <= C0; once it cannot, no larger m can either.
    if (m * static_cast<int64_t>(cfg.min_n) > C0) break;
    if (require_2d && CeilDiv(static_cast<int64_t>(cfg.M), m) < 2) continue;
    const int64_t n_max = std::min<int64_t>(n_hi, C0 / m);
    for (int64_t n = cfg.min_n; n <= n_max; n += cfg.align_n) {
      if (require_2d && CeilDiv(static_cast<int64_t>(cfg.N), n) < 2) continue;
      for (const int k : EnumerateLegalKs(static_cast<int>(m), static_cast<int>(n), cfg, A0, B0)) {
        if (require_full_k && k != cfg.K) continue;
        auto c = MakeCandidate(static_cast<int>(m), static_cast<int>(n), k, cfg, C0, regime);
        if (c && (!best || Better(*c, *best, cfg))) best = c;
      }
    }
  }
  return best;
}

// Operand (L0A/L0B) element budgets for a regime: stationary operand uses the
// full buffer (depth 1), the moving operand is halved (depth 2).
int64_t L0aBudget(const L0TileConfig& cfg, const OperandDB& db) {
  return static_cast<int64_t>(cfg.l0a_bytes) / (static_cast<int64_t>(cfg.bytes_a) * (db.a ? 2 : 1));
}
int64_t L0bBudget(const L0TileConfig& cfg, const OperandDB& db) {
  return static_cast<int64_t>(cfg.l0b_bytes) / (static_cast<int64_t>(cfg.bytes_b) * (db.b ? 2 : 1));
}
// L0C element budget per accumulator: halved for dbC=2 (m * n * bytes_c <= L0C / dbC).
int64_t L0cBudget(const L0TileConfig& cfg, const Regime& r) {
  return static_cast<int64_t>(cfg.l0c_bytes) / (static_cast<int64_t>(cfg.bytes_c) * (r.dbc ? 2 : 1));
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
  CHECK(cfg.bw_a > 0.0 && cfg.bw_b > 0.0 && cfg.bw_drain > 0.0)
      << "ChooseL0Tile: roofline bandwidths must be strictly positive (got bw_a=" << cfg.bw_a
      << ", bw_b=" << cfg.bw_b << ", bw_drain=" << cfg.bw_drain << ") -- they divide the load/drain cost.";

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

  // 2. Baseline budgets for the always-present output-stationary / dbC=1 regime
  //    (both operands double-buffered, one full L0C accumulator). The capacity
  //    sanity checks use this most-constrained operand budget.
  const OperandDB os_db = DeriveOperandDB(Stationarity::kOutputStationary);
  const int64_t A0 = L0aBudget(cfg, os_db);
  const int64_t B0 = L0bBudget(cfg, os_db);
  const Regime base_regime;  // OS, dbC=1
  const int64_t C0_base = L0cBudget(cfg, base_regime);

  CHECK(A0 >= static_cast<int64_t>(cfg.min_m) * cfg.min_k)
      << "ChooseL0Tile: L0a capacity " << A0 << " elements is too small to fit the minimum tile ("
      << cfg.min_m << " x " << cfg.min_k << ")";
  CHECK(B0 >= static_cast<int64_t>(cfg.min_n) * cfg.min_k)
      << "ChooseL0Tile: L0b capacity " << B0 << " elements is too small to fit the minimum tile ("
      << cfg.min_k << " x " << cfg.min_n << ")";
  CHECK(C0_base >= static_cast<int64_t>(cfg.min_m) * cfg.min_n)
      << "ChooseL0Tile: L0c capacity " << C0_base << " elements is too small to fit the minimum tile ("
      << cfg.min_m << " x " << cfg.min_n << ")";

  // 3. Score the design space. The baseline (output-stationary, dbC=1) is today's
  //    realizable algorithm and is always scored. Within a regime the wall
  //    objective couples m, n, k non-separably (the BW-weighted load-optimal
  //    aspect m:n = bytes_b*BW_A : bytes_a*BW_B = 2:1 for BF16 trades against the
  //    per-tile MAD head and ceil waste), so we score every legal tile.
  std::optional<Candidate> best =
      EnumerateBest(cfg, base_regime, A0, B0, C0_base, /*require_2d=*/false, /*require_full_k=*/false);
  CHECK(best) << "ChooseL0Tile: no legal (m, n, k) tile found for M=" << cfg.M << ", N=" << cfg.N
              << ", K=" << cfg.K << ". This indicates the hardware capacity is below the configured "
              << "minimum tile shape; check L0a/L0b/L0c bytes and min_m/min_n/min_k.";

  // Explore the rest of the design space only when the baseline actually tiles
  // the output. A full [M, N, K] tile that fits one L0C is "already L0-sized":
  // the caller skips tiling, so no richer algorithm (operand-stationary, dbC=2)
  // should turn it into a multi-tile grid for a marginal modelled gain the
  // lowering's loop/extract overhead would erase. Each enumerated regime is gated
  // by the realizable mask and adopted only on a STRICTLY lower wall (ties keep
  // the simpler, earlier regime -- the baseline first).
  const bool is_tiled = !(best->m == cfg.M && best->n == cfg.N && best->k == cfg.K);
  if (is_tiled) {
    std::vector<Stationarity> stats = {Stationarity::kOutputStationary};
    if (cfg.allow_a_stationary) stats.push_back(Stationarity::kAStationary);
    if (cfg.allow_b_stationary) stats.push_back(Stationarity::kBStationary);
    for (const Stationarity stat : stats) {
      const OperandDB db = DeriveOperandDB(stat);
      const int64_t a0 = L0aBudget(cfg, db);
      const int64_t b0 = L0bBudget(cfg, db);
      const bool is_os = stat == Stationarity::kOutputStationary;
      for (int dbc = 0; dbc <= (cfg.allow_double_buffer_c ? 1 : 0); ++dbc) {
        const Regime r{stat, /*dbc=*/dbc == 1};
        if (is_os && !r.dbc) continue;  // baseline, already scored
        const int64_t c0 = L0cBudget(cfg, r);
        if (c0 < static_cast<int64_t>(cfg.min_m) * cfg.min_n) continue;  // can't fit min tile
        // Operand-stationary pins an operand across K (k == K); dbC=2 needs the
        // full-K emitter (k == K) and a >= 2x2 grid for the ping-pong.
        const bool require_full_k = !is_os || r.dbc;
        const bool require_2d = r.dbc;
        auto cand = EnumerateBest(cfg, r, a0, b0, c0, require_2d, require_full_k);
        // Cross-regime tie policy: a non-baseline regime is adopted only on a
        // STRICTLY lower wall, so an equal-wall A/B-stationary or dbC=2 candidate
        // never displaces the already-scored output-stationary baseline. This is
        // a deterministic "prefer the simpler lowering" rule, not enumeration
        // order. (Within a regime, Better() applies the full lexicographic key.)
        if (cand && cand->cost_cycles < best->cost_cycles) best = cand;
      }
    }
  }

  L0TileResult result;
  result.m = best->m;
  result.n = best->n;
  result.k = best->k;
  result.estimated_traffic_bytes = best->traffic;
  result.estimated_cost_cycles = best->cost_cycles;
  result.padded_compute_volume = best->padded_compute;
  result.stationarity = best->regime.stat;
  result.double_buffer_c = best->regime.dbc;

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
