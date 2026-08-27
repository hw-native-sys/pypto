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

#ifndef PYPTO_IR_TRANSFORMS_UTILS_ATTRS_H_
#define PYPTO_IR_TRANSFORMS_UTILS_ATTRS_H_

#include <any>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace pypto {
namespace ir {

/// Private provenance on every compiler-generated ``tile.load(GM -> Mat)``
/// call introduced while bridging a Tensor operand to tile IR.
/// ``InferTileMemorySpace`` consumes this evidence when deciding whether a
/// stationary operand is eligible for loop residency; user-authored tile loads
/// deliberately do not carry it.
inline constexpr const char* kCompilerTensorToTileMatBridgeAttr = "__compiler_tensor_to_tile_mat_bridge";

/// Attribute key for ``pl.pipeline(N, stage=F)`` — appears on ``ForStmt.attrs_``
/// if and only if ``ForStmt.kind_ == ForKind::Pipeline`` (bidirectional invariant
/// enforced by the structural verifier ``PipelineLoopValid``).
///
/// Lifecycle:
///   - User-written ``pl.pipeline(stage=F)``           → attr = F (any F ≥ 1)
///   - After ``LowerPipelineLoops`` (factor > 1 path)  → attr = 1 (post-lowering marker)
///   - After ``CanonicalizeIOOrder``                   → attr stripped, kind demoted
///
/// ``LowerPipelineLoops`` triggers on attr > 1; attr == 1 is a no-op trigger
/// (loop is left intact for ``CanonicalizeIOOrder`` to reorder and demote).
inline constexpr const char* kPipelineStagesAttr = "pipeline_stages";

/// Optional ``bool`` policy attr on a ``ForKind::Pipeline`` ``ForStmt``: when
/// ``false``, ``CanonicalizeIOOrder`` keeps store-like ops in the *compute*
/// stage tier instead of floating them to the bottom ``Store`` tier.
///
/// Rationale: the default (absent ⇒ ``true``) floats all sibling-iteration
/// stores below all compute, which keeps both iterations' *output* tiles
/// co-live — a ping-pong on the output buffer. For the full-K M/N matmul
/// pipeline each iteration writes a *different, large* L0C result, so output
/// ping-pong would force two L0C buffers co-live (``2·m·n·bytes_c``) while the
/// tile chooser budgets only one (``double_buffer_c == false``) — an L0C
/// overflow at allocation. Setting this ``false`` yields the one-accumulator
/// schedule ``extract_i, extract_{i+1}, matmul_i, store_i, matmul_{i+1}, …``:
/// the moving-operand extract is still double-buffered (Load tier, hoisted),
/// but ``store_i`` drains before ``matmul_{i+1}`` overwrites the single L0C
/// accumulator. Consumed (stripped) by ``CanonicalizeIOOrder`` alongside
/// ``pipeline_stages``.
inline constexpr const char* kPipelineOverlapStoresAttr = "pipeline_overlap_stores";

/// Optional ``bool`` policy attr on a ``ForKind::Pipeline`` ``ForStmt`` (absent ⇒
/// ``false``): when ``true``, ``CanonicalizeIOOrder`` floats the Acc-draining ops
/// into a tier *above all compute* in the loop body, so every sibling-iteration
/// drain sorts after every matmul — ``matmul_i, matmul_{i+1}, drain_i, drain_{i+1}``
/// instead of ``matmul_i, drain_i, matmul_{i+1}, drain_{i+1}``. For a source
/// pipeline deeper than two, this ordering repeats in depth-two chunks
/// (``MMSS MMSS ...``), so operand prefetch depth remains user-selected while
/// L0C membership still rotates over two stage residues in each fully
/// replicated group. The drain op is ``tile.store`` on the direct-store
/// (Acc→GM) path and ``tile.assemble`` on the Mat-scratch (Acc→Mat) path.
///
/// This is a *stronger* float than ``pipeline_overlap_stores`` (which only orders
/// store-after-compute *within* a stage — the compute/store tier is shared and
/// sorted by stage, so a stage-i store still precedes the stage-{i+1} matmul).
/// It keeps the two iterations' L0C accumulators genuinely co-live, which is the
/// dbC=2 (double-buffered L0C) ping-pong: overlapping their live ranges forces any
/// correct allocator to give them distinct L0C offsets, so tile i's FIXPIPE drain
/// overlaps tile i+1's MAD. Under ``memory_planner=PTOAS``, InitMemRef keeps the
/// co-live buffers distinct and ptoas places them. Under the PyPTO planner,
/// ``LowerPipelineLoops`` adds a depth-2 pipeline membership and MemoryReuse
/// preserves the pair. ``AutoTileMatmulL0`` sets the attr either when the chooser
/// picked ``double_buffer_c`` (with the accumulator budgeted at L0C/2), or when it
/// recognizes a user-authored pipeline containing one canonical directly drained
/// L0 matmul whose path-specific trip-count/Acc-size gate is profitable and whose
/// conservative whole-function Acc footprint still fits after adding the extra
/// slot. Direct-to-GM ``tile.store`` and Acc-to-Mat ``tile.assemble`` have
/// separate conservative admission thresholds. Consumed (stripped) by
/// ``CanonicalizeIOOrder`` alongside ``pipeline_stages`` and
/// ``pipeline_overlap_stores``.
inline constexpr const char* kPipelineDoubleBufferCAttr = "pipeline_double_buffer_c";

/// Attribute key marking a tile-producing ``Call`` with the pipeline-stage
/// membership(s) of the tile it defines. ``LowerPipelineLoops`` sets it when it
/// replicates a ``pl.pipeline`` body: every clone of a replicated region is one
/// pipeline *stage*, and the clones must occupy *distinct* physical buffers so
/// the event-based scheduler can overlap stage k of iteration i+1 with stage
/// k+1 of iteration i (the ping-pong that pipelining exists to expose).
///
/// ``MemoryReuse`` reads this attr and refuses to coalesce two tiles that share
/// a common pipeline *group* with *different* stage indices **when at least one
/// of them is a load buffer** — making stage separation an explicit reuse
/// constraint rather than a fragile side effect of ``CanonicalizeIOOrder``
/// statement clustering (which only induces separation when the dependency graph
/// happens to let it cluster sibling-clone loads). The constraint is role-aware:
/// only load buffers need per-stage privacy (so iteration i+1's prefetch overlaps
/// iteration i's compute); compute intermediates of different stages may still
/// coalesce, because forbidding *all* cross-stage reuse (depth = F) overflows the
/// on-chip budget on real kernels (e.g. stage=4 RMSNorm). The L0 matmul spaces
/// (Left/Right/Acc/Bias/LeftScale/RightScale) are exempt entirely — they are
/// matmul-managed and capacity-bound.
///
/// Value encoding (``std::string`` — round-trip-safe via the existing
/// python-printer / ast-parser string-attr codec, with no integer-width
/// ambiguity): semicolon-separated ``"group:stage"`` pairs, e.g. ``"0:1"`` or
/// ``"3:0;0:1"``. A tile carries one pair per enclosing replicated region, so
/// nested same-core pipelines (e.g. an L1→L0 pipeline inside a GM→L1 pipeline)
/// record both memberships and stay separated at every level.
inline constexpr const char* kPipelineMembershipAttr = "pipeline_membership";

/// Append a ``group:stage`` membership pair to a ``pipeline_membership`` string,
/// preserving any memberships already present (an inner-loop tag survives when
/// an enclosing loop re-tags the same tile).
inline std::string AppendPipelineMembership(const std::string& packed, int32_t group, int32_t stage) {
  std::string pair = std::to_string(group) + ":" + std::to_string(stage);
  return packed.empty() ? pair : packed + ";" + pair;
}

/// Parse a ``pipeline_membership`` string into ``(group, stage)`` pairs.
///
/// Non-throwing: a token that is not exactly ``<int>:<int>`` is skipped rather
/// than aborting. The strings this pass emits are always well-formed, but the
/// attr can be re-attached from a hand-written ``attrs={...}`` on round-trip, so
/// a malformed value degrades gracefully instead of terminating the compiler
/// with an uncaught ``std::stol`` exception.
inline std::vector<std::pair<int32_t, int32_t>> ParsePipelineMembership(const std::string& packed) {
  std::vector<std::pair<int32_t, int32_t>> out;
  auto try_parse_int = [](const std::string& s, int32_t* out_val) -> bool {
    try {
      size_t consumed = 0;
      int64_t v = std::stol(s, &consumed);
      if (consumed != s.size()) return false;  // reject trailing garbage (e.g. "12abc")
      *out_val = static_cast<int32_t>(v);
      return true;
    } catch (const std::exception&) {
      return false;  // empty / non-numeric / out-of-range
    }
  };
  size_t i = 0;
  while (i < packed.size()) {
    size_t semi = packed.find(';', i);
    std::string tok = packed.substr(i, semi == std::string::npos ? std::string::npos : semi - i);
    size_t colon = tok.find(':');
    int32_t g = 0;
    int32_t s = 0;
    if (colon != std::string::npos && try_parse_int(tok.substr(0, colon), &g) &&
        try_parse_int(tok.substr(colon + 1), &s)) {
      out.emplace_back(g, s);
    }
    if (semi == std::string::npos) break;
    i = semi + 1;
  }
  return out;
}

/// True when two pre-parsed ``pipeline_membership`` lists conflict: they share a
/// common group id with *different* stage indices. Such tiles belong to the same
/// replicated region but to clones meant to run concurrently, so they must not
/// share a buffer. Takes pre-parsed vectors (parsed once in ComputeLifetimes) so
/// the O(N²) reuse packer never re-parses strings. O(A·B) over the (tiny —
/// bounded by pipeline nesting depth) member lists.
inline bool PipelineMembershipsConflict(const std::vector<std::pair<int32_t, int32_t>>& pa,
                                        const std::vector<std::pair<int32_t, int32_t>>& pb) {
  for (const auto& [ga, sa] : pa) {
    for (const auto& [gb, sb] : pb) {
      if (ga == gb && sa != sb) return true;
    }
  }
  return false;
}

/// Return a copy of `attrs` with any entry matching `key` removed. The order of
/// the remaining entries is preserved.
inline std::vector<std::pair<std::string, std::any>> StripAttr(
    const std::vector<std::pair<std::string, std::any>>& attrs, std::string_view key) {
  std::vector<std::pair<std::string, std::any>> out;
  out.reserve(attrs.size());
  for (const auto& [k, v] : attrs) {
    if (k == key) continue;
    out.emplace_back(k, v);
  }
  return out;
}

/// ``bool`` attr on a MANUAL ``RuntimeScopeStmt`` marking it as a scope that the
/// compiler synthesised (``AutoDeriveTaskDependencies`` / ``MaterializeRuntimeScopes``)
/// rather than one the user wrote with ``pl.manual_scope()``. Structural analyses
/// peek through such a scope as if it were AUTO (see ``transform_utils::UnwrapAutoScope``).
inline constexpr const char* kAttrCompilerAutoManualScopeCandidate = "__compiler_auto_manual_scope_candidate";

// ---------------------------------------------------------------------------
// ForStmt iter_arg carry classification (produced by ``ClassifyIterArgCarry``)
// ---------------------------------------------------------------------------
//
// ``ClassifyIterArgCarry`` stamps one ``bool`` attr per iter_arg naming its
// lowering (trivial alias vs. materialised rebind carry), plus an optional
// ``int`` attr sizing a TaskId array-carry. Keys are index-suffixed because
// ``ForStmt::attrs_`` is a flat string→scalar map whose printer/parser codec
// only round-trips scalar values.
//
//   attrs={"iter_arg_rebind_0": True, "iter_arg_array_size_0": 4}
//
// The rebind attr is stamped for **every** iter_arg (even when false) so its
// presence proves the pass ran; the array-size attr is stamped only when
// positive. See ``docs/en/dev/passes/47-classify_iter_arg_carry.md``.

/// Prefix of the per-iter_arg ``bool`` "needs a materialised carry" attr.
inline constexpr const char* kIterArgRebindAttrPrefix = "iter_arg_rebind_";
/// Prefix of the per-iter_arg ``int`` TaskId array-carry extent attr.
inline constexpr const char* kIterArgArraySizeAttrPrefix = "iter_arg_array_size_";

inline std::string IterArgRebindAttrKey(size_t idx) {
  return std::string(kIterArgRebindAttrPrefix) + std::to_string(idx);
}

inline std::string IterArgArraySizeAttrKey(size_t idx) {
  return std::string(kIterArgArraySizeAttrPrefix) + std::to_string(idx);
}

// ---------------------------------------------------------------------------
// Region placement carrier (``LowerAutoVectorSplit`` -> ``ExpandMixedKernel``)
// ---------------------------------------------------------------------------
//
// ``pl.split_aiv`` opens an explicit AIV region as a first-class
// ``SplitAivScopeStmt``. ``LowerAutoVectorSplit`` (pass 20) lowers each region
// and ERASES the wrapper, so by the time ``ExpandMixedKernel`` (pass 21)
// partitions the function into an AIC and an AIV lane the region node is gone
// and nothing records that the author pinned those statements to the vector
// lane. Without that record pass 21 duplicates every SHARED statement onto BOTH
// lanes — which for a side effect that must not run on a second core, such as
// ``pld.system.notify``, is wrong. The hazard is not double-counting but
// PREMATURE RELEASE FROM THE WRONG LANE: the cube copy can publish the signal
// before the vector lane's TPUT has landed the data that signal releases, so
// the peer reads stale bytes. A ``NotifyOp::kSet`` fires that race as readily
// as an atomic-add.
//
// Pass 20 therefore stamps ``attrs["core_placement"] = "aiv"`` on the region
// calls whose lane the region DECIDES, and ``ClassifyCallAffinity`` reads it as
// the placement authority. This is a plain string attr, exactly like the
// per-op ``split`` ints and the function-level ``split_aiv_region_validated``
// flag the same pass already stamps, so it needs no new IR concept and keeps
// pass 21's "no live SplitAivScopeStmt survives" invariant intact.
//
// The attr asserts a placement, so it is written only where the region is what
// settles one: a call that STATES its own lane (`tile.create`, a
// `core_type=`-dispatched barrier) or whose lane its memory spec already fixes
// (any ordinary vector op is VECTOR; the `aiv_shard` / `aic_gather` boundary is
// MIXED because it really does run on both lanes) is placed without it. See
// ``RegionPlacementStamper`` in lower_auto_vector_split_pass.cpp for the full
// rule; in practice a mixed comm kernel gains exactly one of these, on the
// notify.
//
// LIFECYCLE: strictly the pass 20 -> pass 21 window. ``ExpandMixedKernel``
// strips the attr from every function it emits once it has consumed it, so no
// downstream pass, printed dump, ``.pto`` round-trip or structural comparison
// ever sees it. (``Call::attrs_`` is a reflection ``UsualField`` and the python
// printer serialises attrs open-world, so an un-stripped stamp WOULD leak into
// both.) Same shape as ``kPipelineStagesAttr``, which lives from
// ``LowerPipelineLoops`` until ``CanonicalizeIOOrder`` strips it.
inline constexpr const char* kCorePlacementAttr = "core_placement";

/// The only value ``kCorePlacementAttr`` currently takes: "this call was
/// written inside a ``pl.split_aiv`` region, so the author placed it on the
/// vector lane". A string (rather than a bool) leaves room for a cube-side
/// placement authority without a second key, and round-trips through the
/// existing string attr codec.
inline constexpr const char* kCorePlacementAiv = "aiv";

}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORMS_UTILS_ATTRS_H_
