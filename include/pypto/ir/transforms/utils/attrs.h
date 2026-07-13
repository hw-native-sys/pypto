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
/// instead of ``matmul_i, drain_i, matmul_{i+1}, drain_{i+1}``. The drain op is
/// ``tile.store`` on the direct-store (Acc→GM) path and ``tile.assemble`` on the
/// Mat-scratch (Acc→Mat) path.
///
/// This is a *stronger* float than ``pipeline_overlap_stores`` (which only orders
/// store-after-compute *within* a stage — the compute/store tier is shared and
/// sorted by stage, so a stage-i store still precedes the stage-{i+1} matmul).
/// It keeps the two iterations' L0C accumulators genuinely co-live, which is the
/// dbC=2 (double-buffered L0C) ping-pong: overlapping their live ranges forces any
/// correct allocator to give them distinct L0C offsets, so tile i's FIXPIPE drain
/// overlaps tile i+1's MAD.  The co-live pair only survives under
/// ``memory_planner=PTOAS``, which skips MemoryReuse (whose opportunistic reuse
/// over-coalesces the pair into one buffer); InitMemRef then keeps the two buffers
/// distinct and ptoas places them.  AutoTileMatmulL0 sets it only when the chooser
/// picked ``double_buffer_c`` (ptoas planner + accumulator budgeted at L0C/2);
/// under the pypto planner it stays absent (⇒ ``false``).  Consumed (stripped) by
/// ``CanonicalizeIOOrder`` alongside ``pipeline_stages`` and ``pipeline_overlap_stores``.
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
/// (Left/Right/Acc/Bias) are exempt entirely — they are matmul-managed and
/// capacity-bound.
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
// positive. See ``docs/en/dev/passes/42-classify_iter_arg_carry.md``.

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

/// ``bool`` attr on a ``tensor.create`` ``Call`` marking it as a scope-local
/// allocation that codegen must hoist out of its enclosing ``pl.manual_scope``
/// (``RuntimeScopeStmt`` with ``manual_ == true``) into the enclosing scope.
///
/// Stamped by ``HoistScopeLocalAllocs`` on exactly the creates that sit directly
/// in a manual-scope body (not nested in a for/if within it) and whose result
/// shape does not reference any Var defined inside that body — i.e. the buffer
/// is enclosing-scope-valid. A manual_scope is a scheduling region, not a
/// storage scope: a buffer it allocates may be read by a task placed AFTER the
/// block, so its C++ declaration must live one level out to stay in scope at the
/// after-scope reader (issue #1697). Orchestration codegen reads this attr
/// instead of re-deriving the hoist set from emit-time indent arithmetic.
inline constexpr const char* kAttrHoistableAlloc = "hoistable_alloc";

}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORMS_UTILS_ATTRS_H_
