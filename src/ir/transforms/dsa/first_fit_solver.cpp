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

#include <algorithm>
#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pypto/ir/transforms/dsa/dsa_solver.h"

namespace pypto {
namespace ir {
namespace dsa {

namespace {

/// Round `x` up to a multiple of `align` (align 0/1 == no rounding).
uint64_t AlignUp(uint64_t x, uint64_t align) {
  if (align <= 1) return x;
  return ((x + align - 1) / align) * align;
}

/// Disjoint-set over BufferId array indices, for collapsing colocation classes.
class UnionFind {
 public:
  explicit UnionFind(size_t n) : parent_(n) {
    for (size_t i = 0; i < n; ++i) parent_[i] = i;
  }
  size_t Find(size_t x) {
    while (parent_[x] != x) {
      parent_[x] = parent_[parent_[x]];  // path halving
      x = parent_[x];
    }
    return x;
  }
  void Union(size_t a, size_t b) { parent_[Find(a)] = Find(b); }

 private:
  std::vector<size_t> parent_;
};

/// A colocation class: all members share one offset within one pool.
struct SuperBuffer {
  std::vector<size_t> members;  ///< indices into DsaProblem::buffers
  uint64_t size = 0;
  uint64_t align = 1;
  PoolId pool = 0;
  Interval interval;  ///< union hull of members
  uint64_t offset = 0;
};

/// Lowest aligned offset >= floor at which [off, off+size) avoids every occupied
/// range.  `occupied` need not be sorted or disjoint; we sort a copy.
uint64_t FirstFit(uint64_t size, uint64_t align, uint64_t floor,
                  std::vector<std::pair<uint64_t, uint64_t>> occupied) {
  std::sort(occupied.begin(), occupied.end());
  uint64_t candidate = AlignUp(floor, align);
  for (const auto& [lo, hi] : occupied) {
    if (lo >= candidate + size) break;  // the gap before this range fits
    if (hi > candidate) candidate = AlignUp(hi, align);
  }
  return candidate;
}

}  // namespace

SolverCapabilities FirstFitByLifetimeSolver::Capabilities() const {
  SolverCapabilities caps;
  caps.multi_interval = false;  // single hull per buffer
  caps.cost_model = false;      // core only — no sync/bank overlay
  caps.colocations = true;
  caps.separations = true;
  caps.pinned = false;
  caps.multi_pool = true;
  return caps;
}

DsaResult FirstFitByLifetimeSolver::Solve(const DsaProblem& problem) {
  DsaResult result;
  const size_t n = problem.buffers.size();

  // Map BufferId -> array index (ids need not be contiguous).
  std::unordered_map<BufferId, size_t> id_to_idx;
  id_to_idx.reserve(n);
  for (size_t i = 0; i < n; ++i) id_to_idx[problem.buffers[i].id] = i;

  // 1. Collapse colocation classes into super-buffers.
  UnionFind uf(n);
  auto idx_of = [&](BufferId id) -> int64_t {
    auto it = id_to_idx.find(id);
    return it == id_to_idx.end() ? -1 : static_cast<int64_t>(it->second);
  };
  for (const auto& [a, b] : problem.colocations) {
    int64_t ia = idx_of(a);
    int64_t ib = idx_of(b);
    if (ia < 0 || ib < 0) {
      result.diagnostics.emplace_back("colocation references unknown buffer id");
      continue;
    }
    uf.Union(static_cast<size_t>(ia), static_cast<size_t>(ib));
  }

  std::unordered_map<size_t, SuperBuffer> supers;  // root idx -> super
  for (size_t i = 0; i < n; ++i) {
    const Buffer& buf = problem.buffers[i];
    SuperBuffer& s = supers[uf.Find(i)];
    if (s.members.empty()) {
      s.pool = buf.pool;
      s.interval = buf.interval;
    } else {
      if (s.pool != buf.pool) {
        result.diagnostics.emplace_back(
            "colocated buffers span different pools (must-alias must be intra-pool)");
      }
      s.interval.start = std::min(s.interval.start, buf.interval.start);
      s.interval.end = std::max(s.interval.end, buf.interval.end);
    }
    s.members.push_back(i);
    s.size = std::max(s.size, buf.size);
    s.align = std::max(s.align, buf.align == 0 ? uint64_t{1} : buf.align);
  }

  // Separation adjacency at super-buffer (root) granularity.
  std::unordered_map<size_t, std::set<size_t>> sep_adj;
  for (const auto& [a, b] : problem.separations) {
    int64_t ia = idx_of(a);
    int64_t ib = idx_of(b);
    if (ia < 0 || ib < 0) {
      result.diagnostics.emplace_back("separation references unknown buffer id");
      continue;
    }
    size_t ra = uf.Find(static_cast<size_t>(ia));
    size_t rb = uf.Find(static_cast<size_t>(ib));
    if (ra == rb) {
      result.diagnostics.emplace_back("separation between colocated buffers is contradictory");
      continue;
    }
    sep_adj[ra].insert(rb);
    sep_adj[rb].insert(ra);
  }

  // 2. Group super-buffers by pool.
  std::map<PoolId, std::vector<size_t>> pool_to_roots;  // pool -> super roots
  for (const auto& [root, s] : supers) pool_to_roots[s.pool].push_back(root);

  DsaSolution solution;
  ObjectiveValue objective;
  bool over_cap = false;

  // 3. Independent per-pool first-fit-by-lifetime (decreasing size).
  for (auto& [pool, roots] : pool_to_roots) {
    uint64_t floor = 0;
    auto rb = problem.reserved_base.find(pool);
    if (rb != problem.reserved_base.end()) floor = rb->second;

    std::sort(roots.begin(), roots.end(), [&](size_t a, size_t b) {
      const SuperBuffer& sa = supers[a];
      const SuperBuffer& sb = supers[b];
      if (sa.size != sb.size) return sa.size > sb.size;                  // decreasing size
      if (sa.interval.start != sb.interval.start) return sa.interval.start < sb.interval.start;
      return problem.buffers[sa.members.front()].id < problem.buffers[sb.members.front()].id;
    });

    std::vector<size_t> placed;  // roots already placed in this pool
    uint64_t peak = floor;
    for (size_t root : roots) {
      SuperBuffer& s = supers[root];
      // Blocking ranges: placed supers whose lifetime overlaps OR that are separation partners.
      std::vector<std::pair<uint64_t, uint64_t>> occupied;
      const auto sep_it = sep_adj.find(root);
      for (size_t pr : placed) {
        const SuperBuffer& other = supers[pr];
        const bool conflict = s.interval.Overlaps(other.interval) ||
                              (sep_it != sep_adj.end() && sep_it->second.count(pr) > 0);
        if (conflict) occupied.emplace_back(other.offset, other.offset + other.size);
      }
      s.offset = FirstFit(s.size, s.align, floor, std::move(occupied));
      placed.push_back(root);
      peak = std::max(peak, s.offset + s.size);
      for (size_t m : s.members) solution.offsets[problem.buffers[m].id] = s.offset;
    }

    objective.peak_by_pool[pool] = peak;
    objective.peak = std::max(objective.peak, peak);

    auto cap_it = problem.pool_caps.find(pool);
    if (cap_it != problem.pool_caps.end() && cap_it->second > 0 && peak > cap_it->second) {
      over_cap = true;
      result.diagnostics.emplace_back("pool " + std::to_string(pool) + " peak " + std::to_string(peak) +
                                   " exceeds cap " + std::to_string(cap_it->second));
    }
  }

  result.solution = std::move(solution);
  result.objective = std::move(objective);
  result.status = over_cap ? SolveStatus::kBestEffortNoFit : SolveStatus::kFeasible;
  return result;
}

std::vector<std::string> Validate(const DsaProblem& problem, const DsaSolution& solution) {
  std::vector<std::string> errors;

  std::unordered_map<BufferId, const Buffer*> by_id;
  for (const auto& b : problem.buffers) by_id[b.id] = &b;

  auto offset_of = [&](BufferId id, uint64_t* out) -> bool {
    auto it = solution.offsets.find(id);
    if (it == solution.offsets.end()) return false;
    *out = it->second;
    return true;
  };

  // Every buffer placed, aligned, above its pool floor, within cap.
  for (const auto& b : problem.buffers) {
    uint64_t off = 0;
    if (!offset_of(b.id, &off)) {
      errors.emplace_back("buffer " + std::to_string(b.id) + " has no offset");
      continue;
    }
    const uint64_t align = b.align == 0 ? 1 : b.align;
    if (align > 1 && off % align != 0) {
      errors.emplace_back("buffer " + std::to_string(b.id) + " offset " + std::to_string(off) +
                       " violates alignment " + std::to_string(align));
    }
    auto rb = problem.reserved_base.find(b.pool);
    if (rb != problem.reserved_base.end() && off < rb->second) {
      errors.emplace_back("buffer " + std::to_string(b.id) + " offset " + std::to_string(off) +
                       " below reserved base " + std::to_string(rb->second));
    }
    auto cap = problem.pool_caps.find(b.pool);
    if (cap != problem.pool_caps.end() && cap->second > 0 && off + b.size > cap->second) {
      errors.emplace_back("buffer " + std::to_string(b.id) + " end " + std::to_string(off + b.size) +
                       " exceeds pool cap " + std::to_string(cap->second));
    }
  }

  // Colocated pairs share an offset.
  for (const auto& [a, b] : problem.colocations) {
    uint64_t oa = 0;
    uint64_t ob = 0;
    if (offset_of(a, &oa) && offset_of(b, &ob) && oa != ob) {
      errors.emplace_back("colocated buffers " + std::to_string(a) + "," + std::to_string(b) +
                       " have different offsets");
    }
  }

  // No two lifetime-overlapping same-pool buffers occupy overlapping address
  // ranges (colocated pairs are exempt — they alias deliberately).
  std::set<std::pair<BufferId, BufferId>> colocated;
  for (const auto& [a, b] : problem.colocations) colocated.insert({std::min(a, b), std::max(a, b)});
  const size_t n = problem.buffers.size();
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = i + 1; j < n; ++j) {
      const Buffer& x = problem.buffers[i];
      const Buffer& y = problem.buffers[j];
      if (x.pool != y.pool) continue;
      if (colocated.count({std::min(x.id, y.id), std::max(x.id, y.id)}) > 0) continue;
      if (!x.interval.Overlaps(y.interval)) continue;
      uint64_t ox = 0;
      uint64_t oy = 0;
      if (!offset_of(x.id, &ox) || !offset_of(y.id, &oy)) continue;
      if (ox < oy + y.size && oy < ox + x.size) {
        errors.emplace_back("lifetime-overlapping buffers " + std::to_string(x.id) + "," +
                         std::to_string(y.id) + " overlap in address");
      }
    }
  }

  // Separation pairs keep apart in address, regardless of lifetime.
  for (const auto& [a, b] : problem.separations) {
    const Buffer* ba = by_id.count(a) ? by_id[a] : nullptr;
    const Buffer* bb = by_id.count(b) ? by_id[b] : nullptr;
    uint64_t oa = 0;
    uint64_t ob = 0;
    if (ba && bb && offset_of(a, &oa) && offset_of(b, &ob)) {
      if (oa < ob + bb->size && ob < oa + ba->size) {
        errors.emplace_back("separated buffers " + std::to_string(a) + "," + std::to_string(b) +
                         " overlap in address");
      }
    }
  }

  return errors;
}

}  // namespace dsa
}  // namespace ir
}  // namespace pypto
