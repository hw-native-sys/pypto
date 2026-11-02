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
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/backend/common/backend.h"
#include "pypto/backend/common/backend_config.h"
#include "pypto/backend/common/backend_handler.h"
#include "pypto/core/dtype.h"
#include "pypto/core/error.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/memref.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_context.h"
#include "pypto/ir/transforms/structural_comparison.h"
#include "pypto/ir/transforms/utils/attrs.h"
#include "pypto/ir/transforms/utils/buffer_valid_analysis.h"
#include "pypto/ir/transforms/utils/lifetime_analysis.h"
#include "pypto/ir/transforms/utils/memory_footprint.h"
#include "pypto/ir/transforms/utils/memref_collectors.h"
#include "pypto/ir/transforms/utils/memref_utils.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/transforms/utils/op_predicates.h"
#include "pypto/ir/transforms/utils/tile_buf_signature.h"
#include "pypto/ir/type.h"
#include "src/ir/transforms/memory_reuse/internal.h"
#include "src/ir/transforms/memory_reuse/storage.h"

namespace pypto {
namespace ir {
namespace memory_reuse {

namespace {

// NOTE: The former tile-type reuse-compatibility gate (AreTileTypesCompatible)
// has been removed. PTO codegen binds a per-var alloc_tile to each tile, so two
// tiles that share a physical MemRef can legally carry different shapes, dtypes,
// or TileView attributes. In-place read/write hazards are handled precisely by
// not_inplace_safe() and per-operand forbid_output_alias() markers (see
// ForbidAliasCollector in allocation_constraints.cpp).

/**
 * @brief Check if two lifetimes overlap.
 *
 * Uses <= to allow "touching" lifetimes (last_use == def_point) to share
 * buffers: within a single statement, inputs are consumed before outputs
 * are produced.
 */
static bool LifetimesOverlap(const LifetimeInterval& a, const LifetimeInterval& b) {
  return !(a.last_use_point <= b.def_point || b.last_use_point <= a.def_point);
}

// Cross-group shed objective (see docs/en/dev/passes/37-memory_reuse.md, pipeline-stage guard): when a
// space overflows at every group's max-affordable depth,
// the packer lowers one pipeline group's double-buffering depth by a residue. The MaxRelief heuristic
// selects **which** group loses a level — lower score sheds first; ties always break by lowest group id
// (deterministic). MaxRelief is used unconditionally: the pluggable objective (and its ArrivalOrder
// alternative) was retired because it never diverged from MaxRelief on any real kernel — the multi-group
// forced-shed it needs to differ does not occur in practice.
struct ShedCandidate {
  int32_t group;         ///< pipeline group id (the tie-break key)
  uint64_t bytes_freed;  ///< align(slot_g): bytes reclaimed by dropping one residue buffer of this group
};

/// MaxRelief: shed the group that frees the most bytes per level ⇒ fewest levels lost to fit the space.
inline double ScoreMaxRelief(const ShedCandidate& c) { return -static_cast<double>(c.bytes_freed); }

struct PackedMember {
  size_t interval_index = 0;
  uint64_t byte_offset = 0;
};

struct PackedBuffer {
  std::vector<PackedMember> members;
  uint64_t capacity = 0;
  bool pinned = false;
  BufferRootCompatibility::Root buffer_root;
};

}  // namespace

/**
 * @brief Identify memory reuse opportunities from lifetime intervals.
 *
 * Global first-fit-decreasing packing: within each memory space, intervals are
 * placed **largest-first**, and every later interval joins the first existing
 * buffer all of whose members it can share with (see `can_share` below).  A
 * buffer's allocated size is fixed by its first (largest) member, so admitting a
 * smaller member afterwards is free — which makes first-fit cost-optimal for
 * this objective and, crucially, lets a *later, larger* interval host an
 * *earlier, smaller* one.  The former definition-order greedy had a
 * one-directional size gate (`source.size >= target.size`) so it could only let
 * a smaller interval reuse an earlier, larger buffer; two lifetime-disjoint L0
 * cube-input tiles whose small one was defined first were never coalesced.
 *
 * The representative (each buffer's first/largest member) is the MemRef every
 * other member is rebased onto downstream; its alloc dominates the whole
 * function (InitMemRef hoists every tile.alloc to the function-body head), so a
 * representative defined *after* some of its members is safe.  Because the
 * packer no longer processes intervals in program order, every pairwise gate
 * below (load+tpop hazard, no-alias) is checked in **both directions**.
 *
 * @param hazard  Ascend910 load + tpop_from_aic guard inputs.  Empty when the
 *                guard is inactive (non-910B / non-split-AIV), in which case the
 *                hazard check is a no-op and reuse behaviour is unchanged.
 * @param forbid_alias  Per-output forbidden-input map (ForbidAliasCollector):
 *                an op may forbid its output from sharing a buffer with specific
 *                input operands it reads while writing the output in place.
 *
 * Complexity: O(M log M) sort + O(M^2) worst-case pairwise checks over the M
 * tile intervals (M a subset of the N IR nodes). The capacity-gated shed loop
 * re-runs the O(M^2) FFD pack once per shed step, so the worst case is O(S·M^2)
 * where S = Σ(F_g − 1) is the total double-buffering depth shed across a space's
 * pipeline groups. S is bounded by (#pipeline groups × depth) — a small constant
 * on real kernels (groups few, depth 2–4) — so this stays the same O(M^2) class as
 * the prior greedy in practice; the FFD base was already O(M^2) pre-#1475.
 */
ReuseMap IdentifyReuseOpportunities(
    const std::vector<LifetimeInterval>& lifetimes, const HazardInputs& hazard,
    const ForbidAliasMap& forbid_alias, const std::map<const Var*, std::set<int>>& phi_family_ids,
    const std::map<VarPtr, std::vector<VarPtr>>& sharing_groups,
    const std::map<const Var*, std::pair<int, int>>& var_liveness,
    const std::map<const Var*, std::vector<std::pair<int32_t, int32_t>>>& pipeline_membership,
    const std::set<const Var*>& pipeline_load_tiles, const std::set<const Var*>& subrange_unsafe_groups,
    const std::map<MemorySpace, uint64_t>& reserved_end_by_space, const std::set<const Var*>& pinned_bases,
    const FunctionPtr& func, std::vector<Diagnostic>* out_hints) {
  ReuseMap reuse_map;
  std::vector<BufferRootCompatibility::Root> buffer_roots(lifetimes.size());
  const auto* buffer_context = PassContext::Current();
  if (buffer_context && buffer_context->GetEnableBufferIR()) {
    BufferRootCompatibility compatibility(func->body_);
    for (size_t i = 0; i < lifetimes.size(); ++i) buffer_roots[i] = compatibility.Get(lifetimes[i].variable);
  }

  // A declared allocation (`pl.Tile[..., pl.MemRef("name"), ...]`) has exactly the membership
  // the kernel author wrote. The packer neither adds tiles to it nor moves its
  // tiles into someone else's buffer — that is the whole point of naming it:
  // coalescing tiles whose lifetimes merely happen not to overlap introduces a
  // false dependency that serializes them.
  //
  // Resolved once per interval rather than inside `can_share`: that predicate is
  // the innermost step of the O(M^2) pack, which itself re-runs per shed step, so
  // a type cast + set lookup there would be paid M^2 times — and by every kernel,
  // including the vast majority that declare no allocation at all. Same reason
  // `pipeline_membership` is pre-parsed (see LifetimeAnalysisResult).
  std::vector<bool> is_pinned(lifetimes.size(), false);
  if (!pinned_bases.empty()) {
    for (size_t i = 0; i < lifetimes.size(); ++i) {
      if (!lifetimes[i].variable) continue;
      auto memref = GetTypeMemRef(lifetimes[i].variable->GetType());
      is_pinned[i] = memref.has_value() && (*memref)->base_ && pinned_bases.count((*memref)->base_.get()) > 0;
    }
  }

  // Members of a sharing group (the vars that already physically share one base).
  // Returns a pointer (or nullptr) to avoid copying the vector in the O(M^2) loop.
  auto group_members = [&sharing_groups](const VarPtr& rep) -> const std::vector<VarPtr>* {
    auto it = sharing_groups.find(rep);
    return it != sharing_groups.end() ? &it->second : nullptr;
  };
  auto ids_intersect = [](const std::set<int>& x, const std::set<int>& y) {
    auto ix = x.begin();
    auto iy = y.begin();
    while (ix != x.end() && iy != y.end()) {
      if (*ix < *iy) {
        ++ix;
      } else if (*iy < *ix) {
        ++iy;
      } else {
        return true;
      }
    }
    return false;
  };
  // Two vars are in the same phi family (the same logical value across
  // mutually-exclusive scf.if paths for one return slot) -> they never conflict.
  auto same_phi_family_var = [&](const Var* x, const Var* y) {
    auto ix = phi_family_ids.find(x);
    auto iy = phi_family_ids.find(y);
    return ix != phi_family_ids.end() && iy != phi_family_ids.end() && ids_intersect(ix->second, iy->second);
  };
  // Precise per-var overlap (touching allowed: last_use == def is fine).
  auto var_overlap = [&](const Var* x, const Var* y) {
    auto ix = var_liveness.find(x);
    auto iy = var_liveness.find(y);
    if (ix == var_liveness.end() || iy == var_liveness.end()) return true;  // unknown -> conservative
    return !(ix->second.second <= iy->second.first || iy->second.second <= ix->second.first);
  };
  // Decide whether a group-interval overlap between two intervals actually
  // blocks sharing.  This is invoked ONLY when their merged intervals overlap.
  //
  // To keep behaviour identical to the prior coarse gate for everything that is
  // not phi-related, we block unless a phi family makes the overlap spurious:
  // some cross-group var pair must be phi-family-related (the same value across
  // mutually-exclusive scf.if paths), AND no *non*-family cross-group pair may
  // truly overlap.  The latter still catches a real conflict hidden behind a
  // branch-local alias of an outside-live buffer (e.g. `result = seed` aliasing
  // a `seed` that overlaps the sibling branch).
  auto overlap_blocks_sharing = [&](const LifetimeInterval& a, const LifetimeInterval& b) {
    const auto* ga = group_members(a.variable);
    const auto* gb = group_members(b.variable);
    const std::vector<VarPtr> sa = ga ? std::vector<VarPtr>{} : std::vector<VarPtr>{a.variable};
    const std::vector<VarPtr> sb = gb ? std::vector<VarPtr>{} : std::vector<VarPtr>{b.variable};
    bool family_pair = false;
    for (const auto& x : (ga ? *ga : sa)) {
      for (const auto& y : (gb ? *gb : sb)) {
        if (same_phi_family_var(x.get(), y.get())) {
          family_pair = true;
        } else if (var_overlap(x.get(), y.get())) {
          return true;  // genuine non-family conflict
        }
      }
    }
    return !family_pair;  // no phi relation -> block exactly as the coarse gate did
  };

  // Maps a tile's *original* MemRef base to the base its buffer is coalesced
  // onto by a committed reuse decision.  A reuse retargets the reused var (and
  // every VIEW that inherited its base) from its own base onto the
  // representative's base; resolve_base() chases that chain so the no-alias
  // guard sees a forbidden operand's *physical* buffer even when the operand is
  // a view whose own base is now stale.  Populated as decisions are committed.
  std::map<const Var*, const Var*> base_remap;
  std::function<const Var*(const Var*)> resolve_base = [&](const Var* b) -> const Var* {
    while (b != nullptr) {
      auto it = base_remap.find(b);
      if (it == base_remap.end() || it->second == b) break;
      b = it->second;
    }
    return b;
  };

  // The physical buffer base a tile var currently occupies: follow any committed
  // reuse onto its representative, then chase base coalescing / VIEW inheritance.
  auto physical_base = [&](const VarPtr& v) -> const Var* {
    VarPtr root = v;
    while (reuse_map.count(root)) root = reuse_map.at(root).representative;
    return resolve_base(TileMemRefBase(root));
  };

  // Ascend910 split-AIV hazard (directional): `writer` consumes a
  // tile.tpop_from_aic value and is defined exactly where the load-derived
  // `input` is last read, so placing writer's output on input's buffer forms the
  // in-place load+tpop touch.  `hazard` is empty off-910B, so this is a no-op.
  auto hazard_blocks = [&hazard](const LifetimeInterval& writer, const LifetimeInterval& input) {
    return writer.def_point == input.last_use_point && hazard.reads_tpop.count(writer.variable.get()) != 0 &&
           hazard.load_derived.count(input.variable.get()) != 0;
  };

  // Written-workspace hazard (directional): the intrinsic writes this operand
  // itself, on a pipe PyPTO cannot see, and nothing orders that write against a
  // read of the same buffer still in flight.  Block only the direction that
  // lets the workspace INHERIT storage from a value that dies before it is
  // defined -- a later value joining the workspace's buffer is safe (it has
  // long retired), which keeps the legal `dst` reuse that
  // `test_sels_output_may_reuse_dead_tmp` pins.
  auto in_workspace_set = [](const LifetimeInterval& interval, const std::unordered_set<const Var*>& vars,
                             const std::unordered_set<const Var*>& bases) {
    if (vars.count(interval.variable.get()) != 0) return true;
    const Var* base = TileMemRefBase(interval.variable);
    return base != nullptr && bases.count(base) != 0;
  };
  auto is_written_workspace = [&hazard, &in_workspace_set](const LifetimeInterval& interval) {
    return in_workspace_set(interval, hazard.written_workspaces, hazard.written_workspace_bases);
  };
  auto is_looping_written_workspace = [&hazard, &in_workspace_set](const LifetimeInterval& interval) {
    return in_workspace_set(interval, hazard.looping_written_workspaces,
                            hazard.looping_written_workspace_bases);
  };
  auto workspace_blocks = [&is_written_workspace, &is_looping_written_workspace](
                              const LifetimeInterval& workspace, const LifetimeInterval& other) {
    // Inside a loop the directional argument collapses: the back edge re-writes
    // the workspace from the scalar pipe on iteration i+1 while iteration i's
    // vector reads of anything sharing that buffer -- the TSELS `dst`, say, and
    // whatever consumes it -- can still be in flight. Lifetime order within one
    // iteration proves nothing about the next, so a looping workspace takes an
    // exclusive buffer.
    if (is_looping_written_workspace(workspace)) return true;
    return is_written_workspace(workspace) && other.last_use_point <= workspace.def_point;
  };

  // No-alias guard (directional): `writer`'s defining op forbids its output from
  // sharing a buffer with one or more input operands — either because the op is
  // not_inplace_safe (forbids ALL inputs, e.g. tile.recip) or because it marks a
  // specific operand via forbid_output_alias (e.g. tile.sel's mask/tmp, read by
  // the TSEL intrinsic while writing dst).  Both feed `forbid_alias`.  Block if
  // any forbidden operand resolves to the physical buffer `occupant` sits on.
  // Each operand's buffer is found by following its reuse chain to its owner and
  // taking the MemRef base — catching both an operand reassigned by an earlier
  // reuse decision AND one occupying the buffer via VIEW inheritance (a reshape
  // / slice / extract shares its source's base with no reuse-map entry).
  auto forbid_blocks = [&](const LifetimeInterval& writer, const LifetimeInterval& occupant) {
    auto fa_it = forbid_alias.find(writer.variable.get());
    if (fa_it == forbid_alias.end()) return false;
    const Var* occ_base = physical_base(occupant.variable);
    if (occ_base == nullptr) return false;
    for (const VarPtr& operand : fa_it->second) {
      if (physical_base(operand) == occ_base) return true;
    }
    return false;
  };

  // Pipeline ping-pong guard (symmetric): two tiles that share a pipeline group
  // with a *different* stage index belong to replicated clones the scheduler
  // overlaps, so collapsing them onto one buffer injects a false write-after-read
  // that serializes the stages — even though their program-order lifetimes are
  // disjoint (that disjointness is exactly what pipelining hides). Non-pipelined
  // tiles carry no membership and never trigger this. See ``kPipelineMembershipAttr``.
  //
  // Default (capacity-gated, #1475): protect concurrent clones in EVERY space —
  // including the L0 matmul spaces (Left/Right/Acc/Bias) and regardless of whether
  // a tile is a load or a `tile.move` result — up to the max-affordable
  // double-buffering depth `F_g` (see below). This is what fixes the L0b operand
  // collapse: the binding matmul operands are L0 `tile.move` results, so the legacy
  // predicate below skipped them and merged 8 → 1.
  //
  // Legacy predicate (the `!gated` fallback, used only when a space's capacity is
  // unknown or when `force_legacy` re-packs after an unfittable shed): the L0
  // spaces are exempt entirely and only *load* buffers get per-stage privacy.
  // Forbidding *all* cross-stage reuse without a capacity check is infeasible — F
  // full copies of every intermediate overflow the budget (e.g. stage=4 RMSNorm
  // needs 4x67KB > 188KB UB), and 2x64KB Right > 64KB L0b — so the ungated path
  // blocks iff the two tiles are same-group / different-stage AND at least one is a
  // load. The capacity gate replaces this coarse heuristic with an exact per-space
  // fit; the predicate is retained verbatim only as the never-worse-than-legacy floor.
  auto is_l0_space = [](MemorySpace s) {
    return s == MemorySpace::Left || s == MemorySpace::Right || s == MemorySpace::Acc ||
           s == MemorySpace::Bias || s == MemorySpace::LeftScale || s == MemorySpace::RightScale;
  };
  // Capacity-gated (#1475): keep software-pipelined operands in separate buffers so the pipeline
  // stages double-buffer instead of serializing on a shared buffer. #1900's `pipeline_membership` tags
  // give each operand its (group, stage); `stage` is the clone index 0..F-1 of a `pl.pipeline(stage=F)`
  // group whose clones run concurrently and must occupy distinct buffers. Each group `g` gets an
  // affordable residue count `F_g = min(D_g, ⌊C_s / slot_g⌋)`; clone stage k lands in residue
  // `ordinal(k) mod F_g`, so clones < F_g apart never share (exact double-buffering when F_g == D_g,
  // maximal spread when capacity forces F_g < D_g). This adjacency guarantee is structural and
  // model-free — it is `mod k`, kept. `F_g` is *mutable*: the FFD driver's shed loop lowers one group's
  // depth (re-packing, exact SpaceFootprint fit) when a space overflows, and `F_g == 1` collapses that
  // group to the legacy merge. Unknown capacity (`cap == 0`) ⇒ `F_g = 1` (conservative merge — never a
  // separation we cannot verify fits). The whole-space fit is now the allocator's realized footprint
  // (Primitive A, in the driver below), not a `Σ` estimate — so co-resident tiles and alignment are exact.
  std::map<std::pair<MemorySpace, int32_t>, uint64_t> group_slot;  // (space, group) -> max tile
  std::map<std::pair<MemorySpace, int32_t>, int32_t> group_depth;  // (space, group) -> F_g (mutable)
  std::map<std::pair<MemorySpace, int32_t>, std::map<int32_t, int32_t>>
      group_stage_ordinal;                                                   // stage->ordinal
  std::map<std::pair<MemorySpace, int32_t>, int32_t> group_requested_depth;  // (space, group) -> D_g (stages)
  std::set<MemorySpace> force_legacy_spaces;  // spaces whose gated packing overflowed → legacy fallback fired
  // Spaces with a *known* (non-zero) capacity — only these are capacity-gated. A space whose bound is
  // unknown (`cap == 0`, incl. no backend configured) falls through to the legacy predicate: gating it to
  // F_g == 1 would merge *everything*, dropping even the legacy non-L0 load-only separation (#1900) and
  // thus separating strictly LESS than legacy. Legacy-for-unknown makes "never worse than legacy" hold
  // for separation too, not just for overflow.
  std::set<MemorySpace> capacity_known_spaces;
  {
    const backend::Backend* be = backend::BackendConfig::IsConfigured() ? backend::GetBackend() : nullptr;
    std::map<std::pair<MemorySpace, int32_t>, std::set<int32_t>> group_stages;
    for (const auto& iv : lifetimes) {
      auto it = pipeline_membership.find(iv.variable.get());
      if (it == pipeline_membership.end()) continue;
      for (const auto& [g, st] : it->second) {
        auto key = std::make_pair(iv.memory_space, g);
        group_slot[key] = std::max(group_slot[key], iv.size);
        group_stages[key].insert(st);
      }
    }
    for (const auto& [key, slot] : group_slot) {
      // Optimistic per-slot depth using the *raw* space size (not cap − reserved_start): this is only the
      // initial F_g. The shed loop below re-checks with the exact reserved-aware SpaceFootprint and lowers
      // F_g further if the reserved region or co-resident tiles don't actually leave room.
      const uint64_t cap = be ? be->GetMemSize(key.first) : 0;
      const int32_t depth = static_cast<int32_t>(group_stages[key].size());
      int32_t f = 1;  // cap == 0 (unknown capacity) ⇒ this space is NOT gated (legacy predicate below)
      if (cap != 0 && slot != 0) {
        f = static_cast<int32_t>(
            std::max<uint64_t>(1, std::min<uint64_t>(static_cast<uint64_t>(depth), cap / slot)));
        capacity_known_spaces.insert(key.first);
      }
      group_depth[key] = f;
      group_requested_depth[key] = depth;
      int32_t ordinal = 0;  // stages may be sparse; compare dense ordinals mod F, not raw stage mod F
      for (const int32_t st : group_stages[key]) group_stage_ordinal[key][st] = ordinal++;
    }
  }
  // `force_legacy` (set by the driver's fallback below) makes the guard behave exactly like the
  // legacy predicate for a space whose gated packing overflowed — so the fallback is legacy by construction.
  bool force_legacy = false;
  auto pipeline_blocks = [&pipeline_membership, &pipeline_load_tiles, &is_l0_space, &group_depth,
                          &group_stage_ordinal, &force_legacy,
                          &capacity_known_spaces](const LifetimeInterval& a, const LifetimeInterval& b) {
    // The binding matmul operands live in L0 and are `tile.move` results, so the legacy guard's
    // `is_l0_space` exemption AND its load-only restriction both skip them — the per-stage operands
    // merge (8 → 1) and the cube matmuls serialize. The capacity-gated path (always on) protects
    // pipeline operands in every space and regardless of load/move, up to the max-affordable
    // double-buffering depth. A space whose capacity is unknown (`cap == 0`) is NOT gated — it uses the
    // legacy predicate, so gated packing is never worse than legacy there. (a and b share a memory
    // space — reuse only happens within a space.)
    const bool gated = !force_legacy && capacity_known_spaces.count(a.memory_space) != 0;
    if (!gated && is_l0_space(a.memory_space)) return false;  // legacy: L0 exempt
    auto ia = pipeline_membership.find(a.variable.get());
    if (ia == pipeline_membership.end()) return false;
    auto ib = pipeline_membership.find(b.variable.get());
    if (ib == pipeline_membership.end()) return false;
    if (gated) {
      // Keep separate iff a shared group's dense stage ordinals fall in different residues mod F_g (the
      // current, possibly-shed, affordable residue count). F_g == 1 ⇒ same residue ⇒ mergeable (legacy).
      // The whole-space fit is decided by the driver's exact SpaceFootprint, not here.
      for (const auto& [ga, sa] : ia->second) {
        for (const auto& [gb, sb] : ib->second) {
          if (ga != gb) continue;
          const auto key = std::make_pair(a.memory_space, ga);
          auto dit = group_depth.find(key);
          const int32_t k = (dit != group_depth.end() && dit->second > 0) ? dit->second : 1;
          int32_t ra = sa;
          int32_t rb = sb;
          auto omit = group_stage_ordinal.find(key);
          if (omit != group_stage_ordinal.end()) {
            auto oa = omit->second.find(sa);
            if (oa != omit->second.end()) ra = oa->second;
            auto ob = omit->second.find(sb);
            if (ob != omit->second.end()) rb = ob->second;
          }
          if (((ra % k) + k) % k != ((rb % k) + k) % k) return true;  // different buffer → separate
        }
      }
      return false;
    }
    if (!PipelineMembershipsConflict(ia->second, ib->second)) return false;  // not cross-stage
    // Legacy: block only if at least one side is a load buffer.
    return pipeline_load_tiles.count(a.variable.get()) != 0 ||
           pipeline_load_tiles.count(b.variable.get()) != 0;
  };

  // Can `cand` join a single physical buffer that already holds `member`?
  // Lifetimes must not overlap (touching is allowed: a buffer's reader is
  // consumed before the writer at the same statement produces its output), and
  // neither directional gate may block. No tile-type / size check is needed:
  // PTO binds a per-var alloc_tile so differing shapes/dtypes legally alias one
  // base, and largest-first ordering guarantees the buffer is sized to its
  // representative (no member is ever larger than the buffer it joins).
  auto can_share = [&](const LifetimeInterval& cand, const LifetimeInterval& member) {
    // Group-interval overlap is a fast reject; when it fires, fall back to the
    // precise per-var check so mutually-exclusive / same-value phi-family tiles
    // may still share while a genuine conflict (incl. one hidden behind a
    // branch-local alias of an outside-live buffer) is still caught.
    if (LifetimesOverlap(cand, member) && overlap_blocks_sharing(cand, member)) return false;
    if (hazard_blocks(cand, member) || hazard_blocks(member, cand)) return false;
    if (workspace_blocks(cand, member) || workspace_blocks(member, cand)) return false;
    if (forbid_blocks(cand, member) || forbid_blocks(member, cand)) return false;
    if (pipeline_blocks(cand, member)) return false;  // symmetric — one call suffices
    return true;
  };

  // Group interval indices by memory space — reuse only happens within a space.
  std::map<MemorySpace, std::vector<size_t>> by_space;
  for (size_t i = 0; i < lifetimes.size(); ++i) {
    by_space[lifetimes[i].memory_space].push_back(i);
  }

  // Allocator policy for the exact per-space footprint (Primitive A) — only when a backend is configured.
  const backend::Backend* pack_be = backend::BackendConfig::IsConfigured() ? backend::GetBackend() : nullptr;
  auto alloc_policy = pack_be ? pack_be->CreateMemoryAllocatorPolicy() : nullptr;

  for (auto& [space_binding, indices_binding] : by_space) {
    // Bind the structured bindings to regular local references: the lambdas below capture by `[&]`,
    // and capturing a structured binding name directly is only a C++20 extension (project is C++17).
    auto& space = space_binding;
    auto& indices = indices_binding;
    // Largest-first; ties broken by definition order for determinism and so that
    // equal-size workloads reproduce the prior definition-order grouping.
    std::sort(indices.begin(), indices.end(), [&lifetimes](size_t a, size_t b) {
      if (lifetimes[a].size != lifetimes[b].size) return lifetimes[a].size > lifetimes[b].size;
      if (lifetimes[a].def_point != lifetimes[b].def_point) {
        return lifetimes[a].def_point < lifetimes[b].def_point;
      }
      return a < b;
    });

    // Dynamic non-slice views lose their symbolic address in
    // AllocateMemoryAddr and cannot reconstruct it in PTO codegen. Disable
    // subdivision for the complete space: compacting an unrelated group can be
    // enough to turn a legacy overflow into silently wrong generated code.
    const bool subrange_space_safe = std::none_of(indices.begin(), indices.end(), [&](size_t idx) {
      return subrange_unsafe_groups.count(lifetimes[idx].variable.get()) != 0;
    });

    // Subrange placement moves the complete root allocation, not the live
    // windows of its alias/view members.  The root must therefore have a static
    // allocation offset, while a symbolic member offset is safe: rebasing keeps
    // that expression relative to the root and the packer continues to reserve
    // the root's complete [placement, placement + size) range.  Constant member
    // offsets are still checked here so malformed static ranges fail closed.
    std::vector<bool> whole_root_subrange_eligible(lifetimes.size(), false);
    for (size_t idx : indices) {
      if (subrange_unsafe_groups.count(lifetimes[idx].variable.get()) != 0) continue;
      const auto root_memref = GetTypeMemRef(lifetimes[idx].variable->GetType());
      if (!root_memref.has_value() || !*root_memref) continue;
      const auto root_offset = As<ConstInt>((*root_memref)->byte_offset_);
      if (!root_offset || root_offset->value_ < 0) continue;
      bool eligible = true;
      const auto* members = group_members(lifetimes[idx].variable);
      const std::vector<VarPtr> singleton =
          members ? std::vector<VarPtr>{} : std::vector<VarPtr>{lifetimes[idx].variable};
      for (const VarPtr& member : (members ? *members : singleton)) {
        const auto member_memref = GetTypeMemRef(member->GetType());
        if (!member_memref.has_value() || !*member_memref ||
            (*member_memref)->base_.get() != (*root_memref)->base_.get()) {
          eligible = false;
          break;
        }
        const auto member_offset = As<ConstInt>((*member_memref)->byte_offset_);
        // A remaining dynamic member is a direct tile.slice: it does not expose
        // holes to the packer, and PTO rebuilds its address from the translated
        // source root plus the slice operands.
        if (!member_offset) continue;
        const __int128 relative = static_cast<__int128>(member_offset->value_) - root_offset->value_;
        const __int128 end = relative + (*member_memref)->size_;
        if (relative < 0 || end > lifetimes[idx].size) {
          eligible = false;
          break;
        }
      }
      whole_root_subrange_eligible[idx] = eligible;
    }

    // First-fit-decreasing pack with the current per-group residue counts
    // (pipeline_blocks reads the mutable F_g). The ordinary mode reproduces the
    // legacy whole-buffer FFD exactly. Subrange mode instead treats members that
    // cannot share bytes with the candidate as address blockers and picks the
    // lowest aligned gap. Members stay sorted by offset, so one candidate scans
    // each arena once and the pack remains O(M^2).
    auto pack = [&](bool allow_subranges) {
      std::vector<PackedBuffer> buffers;
      for (size_t idx : indices) {
        if (is_pinned[idx]) {
          buffers.push_back(
              PackedBuffer{{PackedMember{idx, 0}}, lifetimes[idx].size, true, buffer_roots[idx]});
          continue;
        }
        const auto& cand = lifetimes[idx];
        bool placed = false;
        for (PackedBuffer& buffer : buffers) {
          if (buffer.pinned || cand.size > buffer.capacity) continue;
          // A disjoint subrange is still the same symbolic byte root. Dynamic
          // metadata requires a compatible full-capacity typed allocation even
          // when the original lifetimes do not overlap.
          if (!BufferRootCompatibility::Compatible(buffer_roots[idx], buffer.buffer_root)) continue;

          std::optional<uint64_t> chosen_offset;
          if (!allow_subranges || !whole_root_subrange_eligible[idx] ||
              !whole_root_subrange_eligible[buffer.members.front().interval_index]) {
            bool fits_at_zero = true;
            for (const PackedMember& member : buffer.members) {
              if (!can_share(cand, lifetimes[member.interval_index])) {
                fits_at_zero = false;
                break;
              }
            }
            if (fits_at_zero) chosen_offset = 0;
          } else {
            uint64_t cursor = 0;
            for (const PackedMember& member : buffer.members) {
              if (can_share(cand, lifetimes[member.interval_index])) continue;
              const uint64_t start = member.byte_offset;
              INTERNAL_CHECK_SPAN(
                  lifetimes[member.interval_index].size <= std::numeric_limits<uint64_t>::max() - start,
                  lifetimes[member.interval_index].variable->span_)
                  << "Internal error: reuse placement range overflows uint64";
              const uint64_t end = start + lifetimes[member.interval_index].size;
              const uint64_t aligned = alloc_policy->AlignAddress(cursor, space);
              if (aligned <= start && cand.size <= start - aligned) {
                chosen_offset = aligned;
                break;
              }
              cursor = std::max(cursor, end);
            }
            if (!chosen_offset.has_value()) {
              const uint64_t aligned = alloc_policy->AlignAddress(cursor, space);
              if (aligned <= buffer.capacity && cand.size <= buffer.capacity - aligned) {
                chosen_offset = aligned;
              }
            }
          }

          if (!chosen_offset.has_value()) continue;
          const PackedMember placement{idx, *chosen_offset};
          auto pos = std::upper_bound(buffer.members.begin(), buffer.members.end(), placement,
                                      [](const PackedMember& lhs, const PackedMember& rhs) {
                                        return lhs.byte_offset < rhs.byte_offset;
                                      });
          buffer.members.insert(pos, placement);
          buffer.buffer_root = BufferRootCompatibility::Merge(buffer.buffer_root, buffer_roots[idx]);
          placed = true;
          break;
        }
        if (!placed) {
          buffers.push_back(PackedBuffer{{PackedMember{idx, 0}}, cand.size, false, buffer_roots[idx]});
        }
      }
      return buffers;
    };

    std::vector<PackedBuffer> buffers = pack(/*allow_subranges=*/false);

    // Capacity relief uses the exact allocator footprint (SpaceFootprint, Primitive A). First retry with
    // subranges; while that still overflows, lower the largest-slot pipeline group's depth by one residue
    // and re-pack. Re-running the FFD (not merging in place) avoids in-place non-monotonicity, but the FFD
    // is still not monotone
    // under the *relaxed* (F_g==1) predicate, so exhausting the shed does NOT guarantee a fit: the loop
    // ends in a from-scratch legacy re-pack + diagnostic (see the `!shed_group` branch below), which
    // guarantees no fit regression vs legacy. The shed objective is MaxRelief (largest slot first, tie
    // by lowest group id), applied unconditionally.
    if (alloc_policy) {
      const uint64_t cap = pack_be->GetMemSize(space);
      auto rit = reserved_end_by_space.find(space);
      const uint64_t reserved_start = rit != reserved_end_by_space.end() ? rit->second : 0;
      auto footprint = [&](const std::vector<PackedBuffer>& bufs) {
        SpaceFootprint fp(space, *alloc_policy, reserved_start);
        for (const auto& buf : bufs) {
          (void)fp.OpenBuffer(buf.capacity);
        }
        return fp.HighWater();
      };
      auto retry_with_subranges = [&]() {
        if (!subrange_space_safe || cap == 0 || footprint(buffers) <= cap) return;
        auto subdivided = pack(/*allow_subranges=*/true);
        if (footprint(subdivided) < footprint(buffers)) buffers = std::move(subdivided);
      };
      // Preserve the legacy layout whenever it fits. Only an actual overflow
      // enables subdivision, before sacrificing software-pipeline depth.
      retry_with_subranges();
      // Bound the shed re-packs so the loop stays within the pass-complexity budget
      // (.claude/rules/pass-complexity.md). A legitimate shed converges in Σ_g(F_g−1) steps over the few
      // co-live groups, but the pipeline-group count can grow with generated IR, so cap the full re-packs
      // at a constant. The cap only trips in a pathological many-group overflow, where the legacy fallback
      // is the correct safe outcome anyway — keeping the shed at O(kMaxShedRepacks · M²), the same O(M²)
      // class as the base FFD packer, instead of O(groups · M²).
      constexpr int kMaxShedRepacks = 256;
      int shed_repacks = 0;
      while (cap != 0 && footprint(buffers) > cap) {
        // Pick the group to shed by MaxRelief; ties break by lowest group id (deterministic). Once the
        // re-pack budget is spent, stop picking (shed_group stays empty) and take the legacy fallback.
        const bool within_budget = (++shed_repacks <= kMaxShedRepacks);
        std::optional<int32_t> shed_group;
        double best_score = 0.0;
        if (within_budget) {
          for (const auto& [key, f] : group_depth) {
            if (key.first != space || f <= 1) continue;
            const uint64_t slot = group_slot.count(key) ? group_slot.at(key) : 0;
            const double s =
                ScoreMaxRelief(ShedCandidate{key.second, alloc_policy->AlignAddress(slot, space)});
            if (!shed_group || s < best_score || (s == best_score && key.second < *shed_group)) {
              best_score = s;
              shed_group = key.second;
            }
          }
        }
        if (!shed_group) {
          // No group left to shed — either every group is at depth 1 (shed exhausted) or the re-pack
          // budget was spent (pathological many-group overflow). FFD is NOT monotone under the relaxed
          // (F_g==1) predicate, so this packing may be worse than legacy; re-pack in legacy mode
          // (`force_legacy`, byte-identical to the legacy predicate) — no fit regression vs legacy, and if
          // legacy also overflows it is a genuine overflow legacy would hit too (AllocateMemoryAddr surfaces
          // it).
          force_legacy = true;
          buffers = pack(/*allow_subranges=*/false);
          retry_with_subranges();
          force_legacy = false;
          force_legacy_spaces.insert(space);
          // Fold into the diagnostic channel (Warning) so all capacity-degradation signals go through one
          // place (perf hint for a partial reduction below; this Warning for a space that fits at no depth).
          if (out_hints != nullptr) {
            const bool still_overflows = footprint(buffers) > cap;
            // A single physical buffer larger than the entire space is not a
            // reuse or pipeline-depth failure.  Legacy packing cannot repair
            // it, and the final allocator (or an earlier operation-specific
            // check such as AutoTileMatmulL0's explicit-L0 operand diagnostic)
            // reports the impossible allocation.  Do not obscure that error
            // with a fallback warning that suggests reuse or stage count is
            // responsible.  Keep warning for aggregate pressure: every buffer
            // fits individually, but their required co-residency does not.
            const bool has_intrinsically_oversized_buffer = std::any_of(
                buffers.begin(), buffers.end(), [&](const PackedBuffer& buf) { return buf.capacity > cap; });
            const std::string why =
                within_budget ? "at any double-buffering depth" : "within the shed-repack budget";
            if (!has_intrinsically_oversized_buffer) {
              out_hints->emplace_back(
                  DiagnosticSeverity::Warning, "MemoryReuse", 0,
                  "capacity-gated reuse could not fit memory space " + MemorySpaceToString(space) + " " +
                      why + "; fell back to the legacy packing" +
                      (still_overflows ? " (which also overflows — reduce tile size or stage count)" : "") +
                      ".",
                  func ? func->span_ : Span::unknown());
            }
          }
          break;
        }
        group_depth[std::make_pair(space, *shed_group)] -= 1;
        buffers = pack(/*allow_subranges=*/false);
        retry_with_subranges();
      }
    }

    // Commit the final packing. Every member maps directly to the arena owner;
    // byte_offset is zero for legacy whole-buffer reuse and may be non-zero only
    // for the overflow-triggered subrange fallback.
    for (const auto& buf : buffers) {
      const size_t representative_index = buf.members.front().interval_index;
      const VarPtr& representative = lifetimes[representative_index].variable;
      for (const PackedMember& member : buf.members) {
        if (member.interval_index == representative_index) continue;
        const auto& cand = lifetimes[member.interval_index];
        reuse_map[cand.variable] = ReusePlacement{representative, member.byte_offset};
        if (const Var* cb = TileMemRefBase(cand.variable)) {
          if (const Var* rb = physical_base(representative)) {
            if (cb != rb) base_remap[cb] = rb;
          }
        }
      }
    }
  }

  // Loud diagnostic (perf hint): a pipeline group whose achieved depth F_g fell below the requested D_g
  // means the capacity gate could not honor the programmer's `pl.pipeline(stage=D)` — stages k and k+F_g
  // share a buffer and re-serialize (the false WAR the pipeline meant to avoid). Surface it (correct, just
  // slower) with the concrete fix, rather than silently degrading. Spaces that hit the legacy fallback
  // already emitted a Warning above, so skip them here to avoid a double signal.
  if (out_hints != nullptr) {
    for (const auto& [key, achieved] : group_depth) {
      if (capacity_known_spaces.count(key.first) == 0) continue;  // unknown capacity ⇒ not gated
      if (force_legacy_spaces.count(key.first) != 0) continue;    // already warned at the space level
      auto rit = group_requested_depth.find(key);
      const int32_t requested = rit != group_requested_depth.end() ? rit->second : achieved;
      if (achieved >= requested) continue;  // fully double-buffered as requested — nothing to report
      const uint64_t slot = group_slot.count(key) != 0 ? group_slot.at(key) : 0;
      const uint64_t cap = pack_be != nullptr ? pack_be->GetMemSize(key.first) : 0;
      // Effective (free) capacity is the total minus the reserved region — the same reserved_start the exact
      // SpaceFootprint fit begins at. Co-resident non-pipeline tiles and other pipeline groups also consume
      // the space, but they are not a single subtractable constant, so the byte threshold is only exact when
      // this operand's own footprint is the binding constraint. Distinguish the two shed causes so the fix
      // stays honest: (a) slot-bound — `slot*requested` overflows even an otherwise-empty free region, so
      // shrink to `free_cap/requested`; (b) space-pressure — it would fit alone, so the fix is to relieve the
      // co-residents, not shrink this tile (which already satisfies the per-slot bound).
      const uint64_t reserved =
          reserved_end_by_space.count(key.first) != 0 ? reserved_end_by_space.at(key.first) : 0;
      const uint64_t free_cap = cap > reserved ? cap - reserved : 0;
      // Use the *aligned* slot — the same per-buffer increment SpaceFootprint bumps by — so the slot-bound
      // vs space-pressure classification and the byte threshold match the real fit (raw `slot` under-counts
      // when the tile isn't an alignment multiple).
      const uint64_t aligned_slot =
          alloc_policy != nullptr ? alloc_policy->AlignAddress(slot, key.first) : slot;
      const uint64_t need = aligned_slot * static_cast<uint64_t>(requested);
      const bool slot_bound = free_cap == 0 || need > free_cap;
      // Source-agnostic wording: `pipeline_membership` is stamped both by an explicit `pl.pipeline(stage=)`
      // and by compiler-synthesized pipelines (e.g. #1900's cross-core skew clones), so blame "software
      // pipelining", not the user's `stage=`, and offer `pl.pipeline(stage=)` only as an example.
      std::ostringstream msg;
      msg << "software pipelining requested depth " << requested << " for pipeline group " << key.second
          << " in " << MemorySpaceToString(key.first) << ", but only " << achieved << " of " << requested
          << " buffers fit (" << aligned_slot << " B per stage, " << free_cap << " B free";
      if (reserved != 0) msg << " after " << reserved << " B reserved";
      msg << ") — stages " << achieved << " apart share storage and serialize. ";
      if (slot_bound) {
        msg << "This operand alone needs " << need << " B for depth " << requested
            << "; shrink the per-stage tile to <= "
            << (requested > 0 ? free_cap / static_cast<uint64_t>(requested) : free_cap)
            << " B, or reduce the pipeline depth (e.g. `pl.pipeline(stage=)`) to " << achieved << ".";
      } else {
        msg << "The operand would fit depth " << requested
            << " on its own, but co-resident buffers / other pipeline groups over-subscribe the space; "
            << "relieve the co-residents (smaller or fewer co-live tiles) or reduce the pipeline depth to "
            << achieved << ".";
      }
      out_hints->emplace_back(DiagnosticSeverity::PerfHint, "MemoryReuse", 0, "PH-MR-001", msg.str(),
                              func ? func->span_ : Span::unknown());
    }
  }

  return reuse_map;
}

/**
 * @brief Apply MemRef sharing to the statement tree
 */
StmtPtr ApplyMemRefSharing(const StmtPtr& stmt, const ReuseMap& reuse_map,
                           const std::map<VarPtr, std::vector<VarPtr>>& var_sharing_groups) {
  const auto allocation_sizes = memref_collectors::CollectAllocSizes(stmt);
  // Custom IRMutator for MemRef sharing
  class MemRefSharingMutator : public IRMutator {
   public:
    explicit MemRefSharingMutator(const ReuseMap& reuse_map,
                                  const std::map<VarPtr, std::vector<VarPtr>>& sharing_groups,
                                  const std::map<const Var*, uint64_t>& allocation_sizes)
        : reuse_map_(reuse_map), sharing_groups_(sharing_groups), allocation_sizes_(allocation_sizes) {}

    StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
      // Check if this variable should reuse another's MemRef
      if (reuse_map_.count(op->var_)) {
        const ReusePlacement& placement = reuse_map_.at(op->var_);
        INTERNAL_CHECK_SPAN(
            placement.byte_offset <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()), op->span_)
            << "Internal error: reuse placement offset does not fit an INDEX ConstInt";
        VarPtr source_var = placement.representative;

        // Get source's TileType and MemRef
        auto source_tile_type = As<TileType>(source_var->GetType());

        if (!source_tile_type || !source_tile_type->memref_.has_value()) {
          LOG_ERROR << "Source variable " << source_var->name_hint_ << " does not have MemRef";
          return IRMutator::VisitStmt_(op);
        }

        const auto source_memref = GetDefinedMemRef(source_tile_type);
        const auto allocation = allocation_sizes_.find(source_memref->base_.get());
        const uint64_t source_capacity =
            source_memref->slot_count_ <= 1 || allocation == allocation_sizes_.end()
                ? source_memref->size_
                : std::max(source_memref->size_, allocation->second);

        // Get current variable's TileType
        auto curr_tile_type = As<TileType>(op->var_->GetType());

        if (!curr_tile_type) {
          LOG_ERROR << "Current variable " << op->var_->name_hint_ << " is not TileType";
          return IRMutator::VisitStmt_(op);
        }

        // Rebase a sharing-group member onto the reuse target, keeping its
        // offset relative to the representative and its own size — substituting
        // the target MemRef wholesale would collapse every member onto the
        // target's base offset (issue #1723: all per-head row slices read row 0).
        // Members coinciding with the representative reuse the target MemRef
        // object so plain reuse keeps MemRef identity.
        //
        // Symbolic offsets retain the exact relative-address expression:
        // target_offset + (member_offset - representative_offset). Collapsing a
        // dynamic member onto the target MemRef would silently select the
        // representative's window.
        const MemRefPtr curr_memref = curr_tile_type->memref_.value_or(nullptr);
        auto rebase_memref = [&](const TileTypePtr& tile) -> std::optional<MemRefPtr> {
          if (!tile->memref_.has_value() || !curr_memref) return source_memref;
          const auto& old = tile->memref_.value();
          const auto& slot = placement.forward_slot ? source_memref : old;
          auto old_off = As<ConstInt>(old->byte_offset_);
          auto curr_off = As<ConstInt>(curr_memref->byte_offset_);
          if (!old_off || !curr_off) {
            ExprPtr relative;
            if (structural_equal(old->byte_offset_, curr_memref->byte_offset_)) {
              relative = MakeZeroByteOffset();
            } else if (curr_off && curr_off->value_ == 0) {
              relative = old->byte_offset_;
            } else {
              relative = MakeSub(old->byte_offset_, curr_memref->byte_offset_, old->span_);
            }
            auto placement_expr = std::make_shared<ConstInt>(static_cast<int64_t>(placement.byte_offset),
                                                             DataType::INDEX, Span::unknown());
            auto target_offset = AddByteOffsets(source_memref->byte_offset_, placement_expr);
            auto rebased_offset = AddByteOffsets(target_offset, relative);
            return std::make_shared<MemRef>(source_memref->base_, rebased_offset, old->size_, old->span_,
                                            slot->is_pinned_, slot->slot_count_, slot->slot_index_);
          }
          const __int128 wide_rel = static_cast<__int128>(old_off->value_) - curr_off->value_;
          const __int128 wide_placement = placement.byte_offset;
          INTERNAL_CHECK_SPAN(wide_rel >= 0 && wide_rel <= std::numeric_limits<int64_t>::max() &&
                                  wide_placement + wide_rel + old->size_ <= source_capacity,
                              old->span_)
              << "Internal error: sharing-group member offset " << old_off->value_
              << " cannot be represented inside reuse target (capacity " << source_capacity << ")";
          const int64_t placed_rel = static_cast<int64_t>(wide_placement + wide_rel);
          // Reusing bytes must not replace a pool member's own slot identity,
          // or turn an ordinary tile into a slot of its former allocation.
          if (placed_rel == 0 && old->size_ == source_memref->size_ && old->slot_count_ == 1 &&
              source_memref->slot_count_ == 1 && !old->slot_index_ && !source_memref->slot_index_) {
            return source_memref;
          }
          auto rel_expr = std::make_shared<ConstInt>(placed_rel, DataType::INDEX, Span::unknown());
          return std::make_shared<MemRef>(source_memref->base_,
                                          AddByteOffsets(source_memref->byte_offset_, rel_expr), old->size_,
                                          old->span_, slot->is_pinned_, slot->slot_count_, slot->slot_index_);
        };

        // Rebase the representative through the same path as every member of
        // its semantic sharing group.  A smaller tile may reuse a larger
        // allocation; substituting source_memref wholesale here would give the
        // representative the larger window while its aliases retain their
        // original window, splitting one loop-carried accumulator family.
        auto new_tile_type = std::dynamic_pointer_cast<const TileType>(
            CloneTypeWithMemRefAndRemapExprs(curr_tile_type, rebase_memref(curr_tile_type),
                                             [this](const ExprPtr& expr) { return VisitExpr(expr); }));

        // Create new Var
        auto new_var = std::make_shared<const Var>(op->var_->name_hint_, new_tile_type, op->var_->span_);

        // Record the variable substitution mapping (old -> new)
        // This ensures that all subsequent references to the old variable will be replaced with the new one
        var_substitution_map_[op->var_] = new_var;

        // CRITICAL: If this variable shares MemRef with others (view operations),
        // we need to update ALL of them to use the new MemRef
        if (sharing_groups_.count(op->var_)) {
          const auto& sharing_group = sharing_groups_.at(op->var_);
          for (const auto& shared_var : sharing_group) {
            if (shared_var != op->var_) {
              // Create new Var for shared variable with same reused MemRef
              auto shared_tile_type = As<TileType>(shared_var->GetType());
              if (shared_tile_type) {
                auto new_shared_tile_type =
                    std::dynamic_pointer_cast<const TileType>(CloneTypeWithMemRefAndRemapExprs(
                        shared_tile_type, rebase_memref(shared_tile_type),
                        [this](const ExprPtr& expr) { return VisitExpr(expr); }));
                auto new_shared_var = std::make_shared<const Var>(shared_var->name_hint_,
                                                                  new_shared_tile_type, shared_var->span_);
                var_substitution_map_[shared_var] = new_shared_var;

                LOG_DEBUG << "Propagating reuse to sharing group member: " << shared_var->name_hint_;
              }
            }
          }
        }

        // Visit value expression (this will recursively apply substitutions)
        ExprPtr new_value = VisitExpr(op->value_);

        return std::make_shared<const AssignStmt>(new_var, new_value, op->span_);
      }

      return IRMutator::VisitStmt_(op);
    }

    // Override VisitExpr_ to replace variable references with their new versions
    ExprPtr VisitExpr_(const VarPtr& op) override {
      // Check if this variable has been replaced (i.e., it's the old version of a reused variable)
      if (var_substitution_map_.count(op)) {
        // Return the new version of the variable
        return var_substitution_map_.at(op);
      }
      // Otherwise, keep the variable as-is
      return op;
    }

   private:
    const ReuseMap& reuse_map_;
    const std::map<VarPtr, std::vector<VarPtr>>& sharing_groups_;  // var -> sharing group
    const std::map<const Var*, uint64_t>& allocation_sizes_;
    // Maps old variable objects to new variable objects (with reused MemRef)
    // This is needed because IR nodes are immutable, so we create new Var objects
    // and need to replace all references to the old ones
    std::map<VarPtr, VarPtr> var_substitution_map_;
  };

  MemRefSharingMutator mutator(reuse_map, var_sharing_groups, allocation_sizes);
  return mutator.VisitStmt(stmt);
}

/// Forward a last-used logical input version through an in-place-safe op.
/// Scheduling only declares input versions. This placement decision runs after
/// InitMemRef and uses the shared lifetime facts and operator alias contracts.
/// A fixed number of indexed walks keeps the additional work O(N log N).
StmtPtr ReuseSoftwarePipelineInputs(const StmtPtr& body, const LifetimeAnalysisResult& lifetimes,
                                    const HazardInputs& hazard, const std::set<const Var*>& declared_bases) {
  class Facts : public IRVisitor {
   public:
    std::vector<AssignStmtPtr> assigns;
    std::map<const Var*, const Var*> aliases;
    std::set<const Var*> version_bases;
    std::set<const Var*> partial_view_bases;
    const Var* Root(const Var* var) const {
      auto it = aliases.find(var);
      return it == aliases.end() ? var : it->second;
    }
    void VisitStmt_(const AssignStmtPtr& stmt) override {
      if (auto tile = As<TileType>(stmt->var_->GetType()); tile && tile->memref_) {
        assigns.push_back(stmt);
        auto call = As<Call>(stmt->value_);
        auto source = AsVarLike(stmt->value_);
        if (call && op_predicates::IsBufferAliasingViewOp(call->op_->name_) && !call->args_.empty()) {
          source = AsVarLike(call->args_[0]);
        }
        if (source) {
          const auto* root = Root(source.get());
          aliases.emplace(stmt->var_.get(), root);
          const auto source_type = As<TileType>(root->GetType());
          const auto& ref = *tile->memref_;
          // Moving an ordinary result into a uniform slot must preserve every
          // view in its allocation family, not just the defining result type.
          if (ref->slot_count_ == 1 && source_type && source_type->memref_ &&
              (!TileBufSignature::FromTileType(*tile).IsFullSlotAlias(
                   TileBufSignature::FromTileType(*source_type)) ||
               !structural_equal(ref->byte_offset_, (*source_type->memref_)->byte_offset_))) {
            partial_view_bases.insert(ref->base_.get());
          }
        }
        if (call && call->GetAttr<bool>(kSoftwarePipelineSlotsAttr, false)) {
          version_bases.insert((*tile->memref_)->base_.get());
        }
      }
      IRVisitor::VisitStmt_(stmt);
    }
  } facts;
  facts.VisitStmt(body);
  if (facts.version_bases.empty()) return body;
  std::map<const Var*, int> last_use;
  for (const auto& [var, interval] : lifetimes.var_liveness) {
    last_use[facts.Root(var)] = std::max(last_use[facts.Root(var)], interval.second);
  }
  ReuseMap reuse;
  std::map<const Var*, VarPtr> storage;
  for (const auto& stmt : facts.assigns) {
    auto call = As<Call>(stmt->value_);
    if (!call || !OpRegistry::GetInstance().IsRegistered(call->op_->name_) ||
        facts.aliases.count(stmt->var_.get())) {
      continue;
    }
    auto output = As<TileType>(stmt->var_->GetType());
    INTERNAL_CHECK_SPAN(output, stmt->span_) << "Internal error: software pipeline reuse requires a tile";
    INTERNAL_CHECK_SPAN(output->memref_.has_value(), stmt->span_)
        << "Internal error: software pipeline reuse requires an allocated tile";
    const auto* output_base = (*output->memref_)->base_.get();
    if (declared_bases.count(output_base) || facts.partial_view_bases.count(output_base)) continue;
    const auto& entry = OpRegistry::GetInstance().GetEntry(call->op_->name_);
    if (!entry.IsInplaceSafe() || entry.GetCrossCoreRole()) continue;
    auto def = lifetimes.var_liveness.find(stmt->var_.get());
    if (def == lifetimes.var_liveness.end()) continue;
    auto resolved = [&](const VarPtr& var) {
      auto it = storage.find(facts.Root(var.get()));
      return it == storage.end() ? var : it->second;
    };
    std::set<const Var*> forbidden;
    for (const auto arg : entry.ForbidOutputAliasArgs()) {
      if (arg >= call->args_.size()) continue;
      auto var = AsVarLike(call->args_[arg]);
      if (!var) continue;
      auto ref = GetTypeMemRef(resolved(var)->GetType());
      if (ref && *ref) forbidden.insert((*ref)->base_.get());
    }
    for (size_t i = 0; i < call->args_.size(); ++i) {
      if (entry.GetArgEffect(i, call->kwargs_) != ArgEffect::Read || entry.IsWorkspaceArg(i)) continue;
      auto input = AsVarLike(call->args_[i]);
      if (!input || !last_use.count(facts.Root(input.get())) ||
          last_use.at(facts.Root(input.get())) != def->second.first) {
        continue;
      }
      auto source = resolved(input);
      auto tile = As<TileType>(source->GetType());
      if (!tile || !tile->memref_) continue;
      const auto& ref = *tile->memref_;
      if (ref->slot_count_ <= 1 || !facts.version_bases.count(ref->base_.get()) ||
          forbidden.count(ref->base_.get()) || output->memory_space_ != tile->memory_space_ ||
          output->dtype_.GetBit() != tile->dtype_.GetBit() ||
          !TileBufSignature::FromTileType(*output).IsFullSlotAlias(TileBufSignature::FromTileType(*tile))) {
        continue;
      }
      // Marked slot regions originate only from pipeline loads, including their
      // tile.create consumer views and forwarded chains. Preserve the target
      // load/TPOP constraint even though those views are not tile.load defs.
      if (hazard.reads_tpop.count(stmt->var_.get())) continue;
      reuse.emplace(stmt->var_, ReusePlacement{source, 0, true});
      storage.emplace(stmt->var_.get(), source);
      break;
    }
  }
  return reuse.empty() ? body : ApplyMemRefSharing(body, reuse, lifetimes.var_sharing_groups);
}

}  // namespace memory_reuse
}  // namespace ir
}  // namespace pypto
