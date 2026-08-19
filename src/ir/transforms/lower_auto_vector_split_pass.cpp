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

// LowerAutoVectorSplit (RFC #1300 staged convergence)
// ===================================================
//
// Converts an AUTO ``pl.split`` mixed InCore function into the EXPLICIT
// ``split_aiv`` form *before* ExpandMixedKernel, so that ExpandMixedKernel's
// op-driven boundary arm folds tile.aiv_shard / tile.aic_gather into
// split-stamped tpush/tpop uniformly — the same path hand-authored explicit
// kernels take. Once that conversion happens, the downstream SplitVectorKernel
// no longer needs to halve the body: it sees the ``split_aiv`` marker and only
// stamps attributes (its "already explicit" arm).
//
// This is the LIVE auto-split lowering path: it always runs in the pipeline,
// immediately before ExpandMixedKernel. After it runs, every split function
// reaches SplitVectorKernel already ``split_aiv``-marked, so SplitVectorKernel's
// former per-op halving driver is no longer needed (it was deleted once this
// pass became unconditional — the halving machinery now lives only in
// split_axis_utils, shared by this pass).
//
// Algorithm (per mixed InCore function carrying a function-level split mode M,
// M != None, that is not already ``split_aiv``):
//   1. Per-statement affinity via core_affinity::ClassifyCallAffinity.
//   2. Find C<->V boundaries: a C/V-crossing tile.move (ClassifyMoveDirection).
//   3. C->V boundary: replace with tile.aiv_shard(full_cube_tile, split=int(M))
//      -> HALF; seed the shard result into tile_vars like tpop_from_aic.
//   4. V->C boundary: insert tile.aic_gather(half_vector_tile, split=int(M))
//      -> FULL, then keep the original cube placement move on the full tile.
//   5. Halve ONLY the vector sub-region (AFFINITY GATE): a tile-producing op is
//      halved iff it is VECTOR-affine. CUBE-affine ops (matmul operands, the
//      cube result before the C->V boundary) stay FULL. We assert no CUBE op was
//      halved.
//   6. Inject get_subblock_idx + stamp split + split_aiv so StampTfreeSplit /
//      codegen / the AivSplitVerifier read it.

#include <algorithm>
#include <any>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/core/logging.h"
#include "pypto/ir/core_affinity_kind.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/structural_comparison.h"
#include "pypto/ir/transforms/utils/attrs.h"
#include "pypto/ir/transforms/utils/core_affinity.h"
#include "pypto/ir/transforms/utils/deep_clone_utils.h"
#include "pypto/ir/transforms/utils/loop_state_repair.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/transforms/utils/split_axis_utils.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/transforms/utils/var_collectors.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

namespace {

using core_affinity::ClassifyCallAffinity;
using core_affinity::ClassifyMoveDirection;
using core_affinity::CombineAffinity;
using core_affinity::CoreAffinity;
using core_affinity::CVDirection;
using split_axis::InjectSubblockIdx;
using split_axis::InjectSubblockIdxIntoStmts;
using split_axis::ProcessStmts;
using split_axis::SplitDimension;
using split_axis::TileInfo;

CallPtr AsCall(const ExprPtr& expr) { return std::dynamic_pointer_cast<const Call>(expr); }

// Defined below LowerStmts; forward-declared so the LowerStmts SplitAivScopeStmt
// arm (explicit per-region lowering) can run them on the lowered region body.
void CheckNoCubeTileHalved(const std::vector<StmtPtr>& stmts,
                           const std::unordered_map<const Var*, TileInfo>& halved, bool& cube_halved);
void ValidateTransposeSplitHazard(const std::vector<StmtPtr>& stmts, int split_dim, const Span& region_span);
void ValidateMixedExplicitRegion(const std::vector<StmtPtr>& stmts, const Span& region_span);

// Make a split-kwarg call. On tile.aiv_shard / tile.aic_gather the split int
// attr is the authored SplitMode, NOT the pto-isa split code; ExpandMixedKernel
// derives the code from it (see split_axis::ShardSplitCode / GatherSplitCode).
CallPtr MakeReshapeOpCall(const std::string& op_name, const ExprPtr& source, int split_mode, const Span& span,
                          const ExprPtr& lane_stride = nullptr) {
  std::vector<std::pair<std::string, std::any>> kwargs{{"split", std::any(split_mode)}};
  // Only a rebalanced body stamps the stride; the default box partition leaves
  // the attr absent so every non-ragged kernel's IR is unchanged.
  if (auto stride = std::dynamic_pointer_cast<const ConstInt>(lane_stride)) {
    kwargs.emplace_back("lane_stride", std::any(static_cast<int>(stride->value_)));
  }
  return OpRegistry::GetInstance().Create(op_name, {source}, kwargs, span);
}

// Whether a region body already carries a user-authored explicit boundary op
// (tile.aiv_shard / tile.aic_gather). When it does, the body is in EXPLICIT
// half-width form already: the user manually sharded the cube tile and wrote the
// vector compute on the per-lane half. Re-running the affinity-gated halving over
// such a body would double-shard (a downstream Acc->Vec move would be misread as
// a fresh C->V boundary and rewritten to a second aiv_shard) and inject a
// duplicate subblock index. So the region path passes these bodies through
// unchanged (scope wrapper dropped); ExpandMixedKernel folds the explicit
// boundary into tpush/tpop exactly as for a hand-authored split_aiv kernel.
class ExplicitSplitBoundaryFinder : public IRVisitor {
 public:
  bool found_ = false;

 protected:
  void VisitExpr_(const CallPtr& op) override {
    if (op && op->op_ && (IsOp(op, "tile.aiv_shard") || IsOp(op, "tile.aic_gather"))) found_ = true;
    IRVisitor::VisitExpr_(op);
  }
};

bool RegionBodyHasExplicitBoundary(const StmtPtr& body) {
  if (!body) return false;
  ExplicitSplitBoundaryFinder finder;
  finder.VisitStmt(body);
  return finder.found_;
}

// Stamp ``attrs["core_placement"] = "aiv"`` on every leaf Call of a region body.
//
// This pass ERASES the SplitAivScopeStmt wrapper, so without a carrier the next
// pass (ExpandMixedKernel) cannot tell a statement the author placed inside a
// region from one written at top level — and it duplicates every SHARED
// statement onto BOTH lanes, double-firing a non-idempotent side effect such as
// pld.system.notify. The stamp is that carrier; ``ClassifyCallAffinity`` reads
// it as the placement authority. See kCorePlacementAttr (attrs.h) for the
// pass 20 -> pass 21 lifetime, and ExpandMixedKernel for where it is stripped.
//
// WHAT GETS STAMPED. The attr asserts a placement — "this call runs on the AIV
// lane" — so it is written exactly where the region is what DECIDES that, and
// nowhere else. The walk visits every call in the region; a call is stamped
// when both hold:
//
//   * it does not STATE its own lane (core_affinity::HasStatedLane): a
//     `tile.create` shared by policy so both lanes can declare the buffer, or a
//     `system.syncall(core_type="mix")` that rendezvouses both cores, is placed
//     by its own declaration. Region membership does not outrank that, so
//     recording a contrary placement on it would be a false claim.
//   * its intrinsic affinity is SHARED, i.e. nothing about the op, its kwargs
//     or its operand memory spaces already fixes a lane. This is the case the
//     carrier exists for: SHARED is what ExpandMixedKernel duplicates onto both
//     lanes, and `pld.system.notify` — core-agnostic by ISA, hence SHARED — is
//     the op whose double-fire started all this.
//   * duplication would actually be WRONG for it (IsNoDuplicateCall).
//
// SHARED ALONE IS NOT ENOUGH, and getting this wrong is a miscompile rather
// than a missed optimisation. Pinning a call to AIV does not merely *place* it
// — it REMOVES it from the cube lane. For a duplicate-safe SHARED op that is a
// silent semantic change: `pld.system.wait` inside a region would stop blocking
// the cube core, so the matmul races ahead of the peer's data it was waiting
// for. The same reasoning covers any SHARED op that defines a value a cube
// statement outside the region consumes — pinning it would leave the AIC body
// referencing a variable it no longer defines. So the stamp is written only for
// the ops whose registration says duplication changes program meaning.
//
// The three intrinsic answers left out are left out because the region does not
// decide them, not to save space:
//
//   * VECTOR is already the AIV lane; the stamp would restate what the op's own
//     memory spec says.
//   * CUBE inside a region is an authoring error that check (a) reports; a
//     stamp cannot fix it and ClassifyCallAffinity declines to override it.
//   * MIXED IS the cross-core transfer — `tile.aiv_shard` / `tile.aic_gather`
//     and a C/V-crossing `tile.move` lower to a tpush on one lane plus a tpop
//     on the other, so they genuinely run on both and "aiv" would be false of
//     them. (The printed low-level boundary form also accepts no kwarg beyond
//     `split=`, so stamping one would break the print -> parse round-trip.)
//
// Consequently a region's ordinary vector compute is untouched, and a mixed
// comm kernel gains exactly one attr — on the notify.
//
// Being an IRMutator, this descends into for / if / while / seq bodies, so a
// comm op nested in a loop inside the region is stamped like any other. The
// re-stamp guard keeps it idempotent, which is what makes NESTED regions work:
// an inner region is lowered (and stamped) by the recursive LowerStmts call
// before the outer arm stamps its whole lowered body, and a duplicate key would
// violate the attrs unique-key invariant.
//
// Only ``Call`` is stamped, never ``Submit``: submits are task launches inside
// a pl.manual_scope, which is Orchestration-level, whereas SplitAivScopeStmt
// lives in InCore function bodies — the two cannot co-occur. (ExpandMixedKernel
// likewise classifies affinity from Calls only.)
class RegionPlacementStamper : public IRMutator {
 protected:
  ExprPtr VisitExpr_(const CallPtr& op) override {
    auto mutated = IRMutator::VisitExpr_(op);
    auto call = As<Call>(mutated);
    if (!call || call->HasAttr(kCorePlacementAttr)) return mutated;
    if (core_affinity::HasStatedLane(call)) return mutated;
    if (core_affinity::ClassifyIntrinsicCallAffinity(call) != CoreAffinity::SHARED) return mutated;
    // ...and only when duplication would actually be WRONG for this call site.
    // See the "SHARED alone is not enough" note above: pinning a duplicate-safe
    // SHARED op removes it from the cube lane, which is a miscompile for any op
    // whose presence there is load-bearing (pld.system.wait being the case that
    // caught this).
    if (!core_affinity::IsNoDuplicateCall(call)) return mutated;
    auto attrs = call->attrs_;
    attrs.emplace_back(kCorePlacementAttr, std::any(std::string(kCorePlacementAiv)));
    return std::make_shared<Call>(call->op_, call->args_, call->kwargs_, std::move(attrs), call->GetType(),
                                  call->span_);
  }
};

std::vector<StmtPtr> StampRegionPlacement(const std::vector<StmtPtr>& stmts) {
  RegionPlacementStamper stamper;
  std::vector<StmtPtr> stamped;
  stamped.reserve(stmts.size());
  for (const auto& stmt : stmts) {
    INTERNAL_CHECK(stmt) << "Internal error: null statement in a pl.split_aiv region body";
    stamped.push_back(stamper.VisitStmt(stmt));
  }
  return stamped;
}

// Collect every variable name (DEF and referenced) in a function body so the
// per-region subblock-index injection reserves against them. Threaded through
// the explicit-region walk and grown after each region mints its index, so
// sibling regions get unique names (subblock_idx, subblock_idx_0, ...) instead
// of all colliding on the same "subblock_idx" (an empty reservation set let two
// sibling regions mint identical names, breaking SSA after lowering).
std::unordered_set<std::string> CollectBodyVarNames(const StmtPtr& body) {
  std::unordered_set<std::string> names;
  if (!body) return names;
  var_collectors::VarDefUseCollector collector;
  collector.VisitStmt(body);
  for (const auto* v : collector.GetAllVarRefs()) names.insert(v->name_hint_);
  return names;
}

// Affinity-gated lowering of a flat statement list.
//
// tile_vars / var_replacements thread the per-var halved-extent tracking and the
// old->new var rebind exactly like split_axis::ProcessStmts, so a single final
// Substitute over the rebuilt body re-localizes downstream offsets. The
// cube-operand integrity check is a separate post-lowering walk
// (CheckNoCubeTileHalved) so it observes the FINAL stmts regardless of how a
// tile was routed.
std::vector<StmtPtr> LowerStmts(const std::vector<StmtPtr>& stmts, SplitMode mode, int split_dim,
                                std::unordered_map<const Var*, TileInfo>& tile_vars,
                                const ExprPtr& subblock_idx,
                                std::unordered_map<const Var*, VarPtr>& var_replacements,
                                std::unordered_set<std::string>& used_names,
                                const ExprPtr& lane_stride = nullptr) {
  std::vector<StmtPtr> result;
  result.reserve(stmts.size());

  for (const auto& stmt : stmts) {
    // --- Explicit split_aiv region (nested or top-level): lower in place. ---
    // The region carries its OWN mode (reg->split_); region-local tile_vars /
    // var_replacements maps keep a halved var from leaking into a sibling region
    // or an out-of-region full-width op. After lowering, the scope wrapper is
    // dropped and its scope-free body spliced in.
    if (auto reg = As<SplitAivScopeStmt>(stmt)) {
      SplitMode rmode = reg->split_;

      auto region_stmts = transform_utils::FlattenToStmts(reg->body_);
      // Empty region (DCE-emptied, or a ``pass``-only body whose sole binding was
      // dropped): drop the scope wrapper and emit nothing — a no-op, not a crash.
      if (region_stmts.empty()) {
        continue;
      }

      // TASK-PARALLEL form (SplitMode::None): both AIV lanes run the FULL body for
      // disjoint work the author dispatches via aiv_id. No split axis, so no
      // halving and no offset localization. (This must branch BEFORE
      // SplitDimension(rmode) below, which rejects None.)
      //
      // tile.aiv_shard / tile.aic_gather ARE allowed here, and mean the one thing
      // that still applies without a split axis: this value crosses the AIC/AIV
      // boundary. Their split=0 deduction is shape-preserving (cross_core.cpp), so
      // nothing is halved or re-joined and no re-halving guard is needed — the
      // body is spliced through exactly as a boundary-free one is, and
      // ExpandMixedKernel folds the op into a split=0 tpush/tpop pair, the same
      // pair an implicit crossing produces. The AivSplitValid verifier is what
      // makes writing them mandatory (checks (f)/(g)) rather than optional.
      //
      // ValidateMixedExplicitRegion is deliberately NOT run for this mode: it
      // rejects a body that mixes half-width boundary ops with full-width vector
      // ops, and in a task-parallel region EVERYTHING is full width, so the mix it
      // describes does not exist.
      if (rmode == SplitMode::None) {
        // Pass the body through UNCHANGED except for the placement stamp,
        // dropping the scope wrapper. The body already opens with
        // aiv_id = get_subblock_idx() (the author's lane index, used for
        // disjoint dispatch). No halving and no per-lane offset localization
        // happen here, so there is no second internal subblock_idx to inject —
        // both AIV lanes run the full body verbatim. This is the arm the comm
        // kernels use: 'for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.NONE)'
        // is how an author pins a notify to the vector lane.
        for (auto& s : StampRegionPlacement(region_stmts)) result.push_back(s);
        continue;
      }

      int rdim = SplitDimension(rmode);

      // EXPLICIT boundary form (user wrote tile.aiv_shard / tile.aic_gather
      // inside the region): the body is already half-width and carries its own
      // lane index. Drop the scope wrapper and splice the body unchanged — no
      // re-halving, no duplicate subblock_idx. Still run the per-region transpose
      // hazard check so a transpose that swaps the split axis is rejected with a
      // region-scoped diagnostic. ExpandMixedKernel folds the explicit boundary
      // into tpush/tpop just as for a hand-authored split_aiv kernel.
      if (RegionBodyHasExplicitBoundary(reg->body_)) {
        ValidateMixedExplicitRegion(region_stmts, reg->span_);
        ValidateTransposeSplitHazard(region_stmts, rdim, reg->span_);
        // The body is already half-width, but the boundary op's split-axis valid
        // extent is still the deducer's lane-agnostic ceil-div guess. This region
        // is where it can be repaired: the author's own aiv_id is in scope, so
        // the lane's true extent is materializable (see
        // split_axis::LocalizeExplicitBoundaryValid). A fully-valid split axis is
        // returned untouched, so the common case is a no-op walk.
        auto localized = split_axis::LocalizeExplicitBoundaryValid(region_stmts, rdim, reg->span_);
        // Still region-placed, so pass 21 must be told.
        for (auto& s : StampRegionPlacement(localized)) result.push_back(s);
        continue;
      }

      std::unordered_map<const Var*, TileInfo> r_tile_vars;
      std::unordered_map<const Var*, VarPtr> r_var_repl;
      // Reserve the injected per-region index against every name visible in the
      // enclosing function body (threaded via ``used_names``) plus the region's
      // own bindings, so sibling regions get unique names. Grow ``used_names``
      // with the freshly minted name afterwards so the next sibling skips it.
      auto inj = InjectSubblockIdxIntoStmts(region_stmts, used_names);
      used_names = inj.used_names;
      auto lowered =
          LowerStmts(inj.body_stmts, rmode, rdim, r_tile_vars, inj.subblock_idx_expr, r_var_repl, used_names);

      // Per-region cube-operand backstop and transpose-hazard check, using THIS
      // region's split_dim and span so diagnostics point at the region.
      bool cube_halved = false;
      CheckNoCubeTileHalved(lowered, r_tile_vars, cube_halved);
      INTERNAL_CHECK_SPAN(!cube_halved, reg->span_)
          << "Internal error: LowerAutoVectorSplit halved a CUBE-affinity op inside a pl.split_aiv "
             "region — the vector-sub-region affinity gate leaked into a cube operand.";
      ValidateTransposeSplitHazard(lowered, rdim, reg->span_);

      StmtPtr region_body =
          (lowered.size() == 1) ? lowered[0] : std::make_shared<SeqStmts>(lowered, reg->span_);
      if (!r_var_repl.empty()) {
        region_body = transform_utils::Substitute(region_body, r_var_repl);
      }
      // Stamp LAST, on the final statements: the halving machinery rewrote and
      // replaced calls above (boundary moves became aiv_shard / aic_gather,
      // shapes and offsets were localized), so stamping earlier would mark
      // calls that no longer exist and miss the ones that replaced them.
      for (auto& s : StampRegionPlacement(transform_utils::FlattenToStmts(region_body))) {
        result.push_back(s);
      }
      continue;
    }

    // --- Boundary tile.move: rewrite to aiv_shard (C->V) / aic_gather (V->C). ---
    if (auto assign = std::dynamic_pointer_cast<const AssignStmt>(stmt)) {
      if (auto call = AsCall(assign->value_)) {
        CVDirection dir = ClassifyMoveDirection(call);

        if (dir == CVDirection::CUBE_TO_VECTOR) {
          // C->V: full cube tile -> HALF vector tile via aiv_shard. The source
          // (matmul/Acc result) stays FULL; only the result is halved and tracked.
          INTERNAL_CHECK_SPAN(!call->args_.empty(), call->span_)
              << "Internal error: C->V boundary tile.move must carry a source tile";
          // The boundary op carries the authored MODE; ExpandMixedKernel derives
          // the pto-isa split code (including the odd ones) from it and the
          // cube tile's extents when it mints the tpush / tpop pair. When the
          // body was rebalanced onto this boundary's valid region, the stride
          // rides along so that derivation sees the same partition.
          auto shard = MakeReshapeOpCall("tile.aiv_shard", call->args_[0], static_cast<int>(mode),
                                         call->span_, lane_stride);
          // Result: the op's deduced HALF type. The split deducer leaves the memory
          // space null, and OpRegistry::Create fills it from tile.aiv_shard's
          // set_output_memory declaration — Vec, the CONSUMING (vector) lane, which
          // is also the C->V move's destination. Going through Create is what keeps
          // this AUTO path's IR identical to the explicit pl.aiv_shard form lowered
          // by ConvertTensorToTileOps: both get the space from the same declaration.
          // The deducer can only ceil-halve the split-axis valid extent, because an
          // op's type function does not know the lane. Here subblock_idx IS in
          // scope, so give the result the lane's true extent — the same repair
          // the explicit-region arm applies through
          // LocalizeExplicitBoundaryValid. A fully-valid split axis is returned
          // unchanged, so the common case keeps the deducer's exact type.
          auto half_type = split_axis::LocalizeShardValidForLane(shard->GetType(), call->args_[0]->GetType(),
                                                                 split_dim, subblock_idx, lane_stride);
          auto new_var = std::make_shared<Var>(assign->var_->name_hint_, half_type, assign->var_->span_);
          auto shard_typed =
              std::make_shared<Call>(shard->op_, shard->args_, shard->kwargs_, half_type, shard->span_);
          if (auto tt = std::dynamic_pointer_cast<const TileType>(call->GetType());
              tt && split_dim < static_cast<int>(tt->shape_.size())) {
            // Record split_dim too: the V->C arm now gathers along the tracked
            // dim, so a C->V shard result fed straight into a V->C boundary must
            // carry the real dim (LEFT_RIGHT => 1), not the default 0.
            TileInfo info{split_axis::ComputeHalfDimSize(tt->shape_[split_dim]), split_dim};
            tile_vars[assign->var_.get()] = info;
            tile_vars[new_var.get()] = info;
          }
          var_replacements[assign->var_.get()] = new_var;
          result.push_back(std::make_shared<AssignStmt>(new_var, shard_typed, assign->span_));
          continue;
        }

        if (dir == CVDirection::VECTOR_TO_CUBE) {
          // V->C: HALF vector tile -> FULL via aic_gather, then keep the original
          // cube-placement move on the gathered FULL tile. The gather result is
          // full (un-tracked); the cube placement move and matmul stay full.
          INTERNAL_CHECK_SPAN(!call->args_.empty(), call->span_)
              << "Internal error: V->C boundary tile.move must carry a source tile";
          // The vector lane works on per-lane HALVES, so the boundary source has
          // already been halved by the affinity gate (it is sequenced before this
          // move). Resolve it to the halved var so aic_gather doubles HALF -> FULL;
          // using the original full-typed reference would over-double to 2x FULL.
          auto src = call->args_[0];
          auto src_var = AsVarLike(src);
          auto tracked = src_var ? tile_vars.find(src_var.get()) : tile_vars.end();
          // Precondition, not a guarantee: a Vec param moved straight to cube, or a
          // singleton split dim the affinity gate preserves, reaches here un-halved.
          // Reject rather than emit a doubled operand under a FULL-typed move.
          CHECK_SPAN(tracked != tile_vars.end(), call->span_)
              << "LowerAutoVectorSplit: the V->C boundary tile.move here carries a full-width "
              << "vector operand" << (src_var ? " '" + src_var->name_hint_ + "'" : "")
              << ". tile.aic_gather reassembles the two AIV lanes' per-lane halves into the full "
              << "tile the cube expects, so its operand must be a value the split halving "
              << "produced (a tile.load / tile.slice / elementwise result inside the vector "
              << "sub-region). An un-halved value has no half to gather — either derive the "
              << "per-lane half first (load or slice the value inside the split function) and "
              << "move that to the cube side, or, if the split axis is a singleton that cannot "
              << "be halved, keep the value on the vector side.";
          if (auto it = var_replacements.find(src_var.get()); it != var_replacements.end()) {
            src = it->second;
          }
          // Gather along the OPERAND's own split axis, not the function/region mode:
          // a tile.reshape can migrate the split axis (the rms_norm [N,1]<->[1,N]
          // column reshape moves it from dim 0 to dim 1), and TileInfo::split_dim
          // tracks where it actually ended up. Doubling the function axis instead
          // would reassemble the wrong dimension — for a [1,8] operand under UP_DOWN
          // it yields [2,8] where the cube-placement move expects [1,16].
          const int tracked_split_dim = tracked->second.split_dim;
          CHECK_SPAN(tracked_split_dim == 0 || tracked_split_dim == 1, call->span_)
              << "LowerAutoVectorSplit: the V->C boundary operand"
              << (src_var ? " '" + src_var->name_hint_ + "'" : "") << " carries its split on dim "
              << tracked_split_dim
              << ", but tile.aic_gather reassembles a 2D tile along dim 0 or dim 1 only. Cross to "
              << "the cube side before the op that moved the split axis there.";
          // The gather carries the authored MODE for the axis it re-joins (dim 0
          // -> UP_DOWN, dim 1 -> LEFT_RIGHT); its transport code (always an even
          // one — the V->C direction has no odd form) is derived downstream by
          // ExpandMixedKernel.
          const int gather_split =
              static_cast<int>(tracked_split_dim == 0 ? SplitMode::UpDown : SplitMode::LeftRight);
          auto gather = MakeReshapeOpCall("tile.aic_gather", src, gather_split, call->span_);
          // Gather result: full shape, with the memory space OpRegistry::Create
          // fills in from tile.aic_gather's set_output_memory declaration — Mat, the
          // CONSUMING (cube) lane, which is the space ExpandMixedKernel pops the
          // V->C boundary into. Same single source of truth as the shard above.
          auto gather_type = gather->GetType();
          auto gather_typed =
              std::make_shared<Call>(gather->op_, gather->args_, gather->kwargs_, gather_type, gather->span_);
          // Name the gathered FULL tile with the cube-destination's "_mat" suffix:
          // ExpandMixedKernel folds this gather into the AIC-side V->C boundary and
          // names the synthesized tpop after this var. The standalone split_aiv
          // move-boundary path names that tpop BuildBoundaryTpopName(AIC, dest) =
          // "<dest>_mat", so matching it here keeps both paths' .pto byte-identical.
          auto full_mat_var =
              std::make_shared<Var>(assign->var_->name_hint_ + "_mat", gather_type, assign->span_);
          result.push_back(std::make_shared<AssignStmt>(full_mat_var, gather_typed, assign->span_));
          // Original cube placement move, now on the FULL gathered tile. Keeping the
          // move's original result type is only valid if the gather reassembled back
          // to exactly that shape — now a genuine invariant, since the gather doubles
          // the operand's own tracked split axis and both user-facing preconditions
          // (operand halved; split dim gatherable) are enforced above. Compare shapes
          // only: the gather type deliberately drops layout.
          auto gathered_tile = As<TileType>(gather_type);
          auto move_tile = As<TileType>(call->GetType());
          INTERNAL_CHECK_SPAN(gathered_tile && move_tile, call->span_)
              << "Internal error: a V->C boundary tile.move and its aic_gather must both be tiles";
          INTERNAL_CHECK_SPAN(gathered_tile->shape_.size() == move_tile->shape_.size(), call->span_)
              << "Internal error: aic_gather result rank " << gathered_tile->shape_.size()
              << " does not match the cube-placement move's rank " << move_tile->shape_.size();
          for (size_t i = 0; i < gathered_tile->shape_.size(); ++i) {
            INTERNAL_CHECK_SPAN(structural_equal(gathered_tile->shape_[i], move_tile->shape_[i]), call->span_)
                << "Internal error: aic_gather result dim " << i
                << " does not match the cube-placement move's result type; the V->C boundary "
                << "would emit a tile.move whose result shape contradicts its operand";
          }
          std::vector<ExprPtr> move_args = call->args_;
          move_args[0] = full_mat_var;
          auto new_move = std::make_shared<Call>(call->op_, std::move(move_args), call->kwargs_,
                                                 call->GetType(), call->span_);
          result.push_back(std::make_shared<AssignStmt>(assign->var_, new_move, assign->span_));
          continue;
        }
      }
    }

    // --- Affinity gate: only halve VECTOR-affine leaf stmts. ---
    CallPtr leaf_call;
    if (auto assign = std::dynamic_pointer_cast<const AssignStmt>(stmt)) {
      leaf_call = AsCall(assign->value_);
    } else if (auto eval = std::dynamic_pointer_cast<const EvalStmt>(stmt)) {
      leaf_call = AsCall(eval->expr_);
    }

    if (leaf_call && leaf_call->op_) {
      CoreAffinity aff = ClassifyCallAffinity(leaf_call);
      if (aff == CoreAffinity::VECTOR) {
        // Route this single vector stmt through the shared halving machinery.
        auto lowered = ProcessStmts({stmt}, mode, split_dim, tile_vars, /*is_aiv=*/true, subblock_idx,
                                    var_replacements, lane_stride);
        for (auto& s : lowered) result.push_back(s);
        continue;
      }
      if (aff == CoreAffinity::CUBE) {
        // Affinity gate: CUBE ops are passed through FULL — never routed to the
        // halving machinery. The post-lowering CheckNoCubeTileHalved walk
        // verifies that no cube operand or result was shrunk (see LowerFunction).
        result.push_back(stmt);
        continue;
      }
    }

    // --- Compound stmts: recurse into the body for vector content. ---
    if (auto for_stmt = std::dynamic_pointer_cast<const ForStmt>(stmt)) {
      auto body = transform_utils::FlattenToStmts(for_stmt->body_);
      auto new_body = LowerStmts(body, mode, split_dim, tile_vars, subblock_idx, var_replacements, used_names,
                                 lane_stride);
      auto new_for = MutableCopy(for_stmt);
      new_for->body_ = loop_repair::MakeBody(new_body, for_stmt->span_);
      result.push_back(new_for);
      continue;
    }
    if (auto if_stmt = std::dynamic_pointer_cast<const IfStmt>(stmt)) {
      auto then_body = transform_utils::FlattenToStmts(if_stmt->then_body_);
      auto new_then = LowerStmts(then_body, mode, split_dim, tile_vars, subblock_idx, var_replacements,
                                 used_names, lane_stride);
      std::optional<StmtPtr> new_else;
      if (if_stmt->else_body_.has_value()) {
        auto else_body = transform_utils::FlattenToStmts(*if_stmt->else_body_);
        auto new_else_stmts = LowerStmts(else_body, mode, split_dim, tile_vars, subblock_idx,
                                         var_replacements, used_names, lane_stride);
        new_else = loop_repair::MakeBody(new_else_stmts, if_stmt->span_);
      }
      auto new_if = MutableCopy(if_stmt);
      new_if->then_body_ = loop_repair::MakeBody(new_then, if_stmt->span_);
      new_if->else_body_ = new_else;
      result.push_back(new_if);
      continue;
    }
    if (auto while_stmt = std::dynamic_pointer_cast<const WhileStmt>(stmt)) {
      auto body = transform_utils::FlattenToStmts(while_stmt->body_);
      auto new_body = LowerStmts(body, mode, split_dim, tile_vars, subblock_idx, var_replacements, used_names,
                                 lane_stride);
      auto new_while = MutableCopy(while_stmt);
      new_while->body_ = loop_repair::MakeBody(new_body, while_stmt->span_);
      result.push_back(new_while);
      continue;
    }

    // SHARED leaf / ReturnStmt / anything else: pass through unchanged.
    result.push_back(stmt);
  }

  return result;
}

// Post-lowering cube-tile integrity walk (O(N) over the rebuilt body).
//
// EFFECTIVE backstop for the affinity gate: a CUBE-affine op must consume — and
// produce — only FULL tiles. ``halved`` is the split-tracking set (every var the
// gate partitioned along the split axis, keyed by both its original and its
// rebuilt pointer; see split_axis::ProcessStmts). For every CUBE-affine leaf
// call we assert that neither its result var nor any of its tile operands is in
// ``halved``. If the vector sub-region gate ever leaked a shrunk tile into a
// cube operand (e.g. a cube op mis-routed through the halving machinery, which
// inserts its result into ``halved``), this fires.
//
// This replaces the prior output-only guard that sat INSIDE the non-halving cube
// branch: there the cube result var was never inserted into the tracking set, so
// the check could never observe a halved tile (theatrical). Re-deriving affinity
// over the FINAL stmts decouples the check from the routing decision, so it
// genuinely trips whenever a cube tile was halved, regardless of how.
void CheckNoCubeTileHalved(const std::vector<StmtPtr>& stmts,
                           const std::unordered_map<const Var*, TileInfo>& halved, bool& cube_halved) {
  for (const auto& stmt : stmts) {
    CallPtr leaf;
    VarPtr def_var;
    if (auto assign = std::dynamic_pointer_cast<const AssignStmt>(stmt)) {
      leaf = AsCall(assign->value_);
      def_var = assign->var_;
    } else if (auto eval = std::dynamic_pointer_cast<const EvalStmt>(stmt)) {
      leaf = AsCall(eval->expr_);
    }
    if (leaf && leaf->op_ && ClassifyCallAffinity(leaf) == CoreAffinity::CUBE) {
      if (def_var && halved.count(def_var.get()) != 0) cube_halved = true;
      for (const auto& arg : leaf->args_) {
        if (auto v = AsVarLike(arg)) {
          if (halved.count(v.get()) != 0) cube_halved = true;
        }
      }
    }

    // Recurse into compound stmts (loops, conditionals, nested seqs).
    if (auto for_stmt = std::dynamic_pointer_cast<const ForStmt>(stmt)) {
      CheckNoCubeTileHalved(transform_utils::FlattenToStmts(for_stmt->body_), halved, cube_halved);
    } else if (auto if_stmt = std::dynamic_pointer_cast<const IfStmt>(stmt)) {
      CheckNoCubeTileHalved(transform_utils::FlattenToStmts(if_stmt->then_body_), halved, cube_halved);
      if (if_stmt->else_body_.has_value()) {
        CheckNoCubeTileHalved(transform_utils::FlattenToStmts(*if_stmt->else_body_), halved, cube_halved);
      }
    } else if (auto seq = std::dynamic_pointer_cast<const SeqStmts>(stmt)) {
      CheckNoCubeTileHalved(seq->stmts_, halved, cube_halved);
    }
  }
}

// Per-region transpose-split hazard check (user-facing limitation). A
// tile.transpose that swaps the split axis migrates the per-lane data to the
// other dimension and cannot be split correctly; reject it with an actionable
// error pointing at the region. Shares the detector with ExpandMixedKernel's
// AUTO whole-function check (split_axis::FindTransposeSplitHazard).
void ValidateTransposeSplitHazard(const std::vector<StmtPtr>& stmts, int split_dim, const Span& region_span) {
  auto body = std::make_shared<SeqStmts>(stmts, region_span);
  auto hazard = split_axis::FindTransposeSplitHazard(body, split_dim);
  if (hazard.call) {
    const char* mode_name = (split_dim == 0) ? "UP_DOWN" : "LEFT_RIGHT";
    std::string where = hazard.result_name.empty() ? std::string() : " (result '" + hazard.result_name + "')";
    CHECK_SPAN(false, hazard.call->span_)
        << "LowerAutoVectorSplit: a pl.split_aiv(" << mode_name << ") region contains a tile.transpose"
        << where << " that swaps the split axis (dim " << split_dim
        << "). The transpose moves the per-lane split data to the other dimension, so the region cannot "
           "be split correctly. Fix it one of two ways: (1) remove the transpose, e.g. replace a "
           "transpose-then-row-index with a direct column slice such as pre[:, h:h+1]; or (2) move the "
           "transpose outside the pl.split_aiv region.";
  }
}

// Mutable state shared across the whole region walk, so propagation follows
// program order. Bundled (rather than threaded as three out-params) to match
// split_axis::SubblockInjectionResult, which this file already unpacks.
struct HalfWidthScan {
  std::unordered_set<const Var*> half_tiles;    // in the aiv_shard half-width dataflow
  std::unordered_set<const Var*> lane_scalars;  // derived from the region's lane index
  std::vector<std::string> full_width_vec_ops;  // offenders to report
};

// True iff any Var / IterArg referenced anywhere in ``exprs`` is lane-derived.
// One collector over all of them: the walk covers each whole expression tree, so
// a lane reference nested inside a MakeTuple offset list or an arithmetic
// sub-expression is found. VarDefUseCollector overrides both the Var and the
// IterArg handler (see var_collectors.h) — a loop-carried lane offset is a
// reference too.
bool ReferencesLaneIndex(const std::vector<ExprPtr>& exprs,
                         const std::unordered_set<const Var*>& lane_scalars) {
  if (lane_scalars.empty()) return false;
  var_collectors::VarDefUseCollector collector;
  for (const auto& e : exprs) {
    if (e) collector.VisitExpr(e);
  }
  for (const auto* v : collector.var_uses) {
    if (lane_scalars.count(v) != 0) return true;
  }
  return false;
}

// The positional args that carry an op's ADDRESS — the base offset selecting
// which window of the source this call reads. Empty for anything that is not an
// addressing op.
//
// Only these args are consulted for a lane reference. A lane-derived scalar
// anywhere ELSE in an addressing op — a shape, a valid_shape — does not localize
// the window: ``tile.load(data, [0, 0], [64, 128], valid_shape=[aiv_id + 1,
// 128])`` mentions aiv_id, yet both lanes still read the same base rows. Scanning
// every arg would admit it and then trust its consumers as half-width.
std::vector<ExprPtr> AddressArgs(const CallPtr& call) {
  const auto& args = call->args_;
  auto at = [&args](size_t i) -> ExprPtr { return i < args.size() ? args[i] : nullptr; };
  if (IsOp(call, "tile.load")) return {at(1)};            // (tensor, offsets, shapes, valid_shape)
  if (IsOp(call, "tile.slice")) return {at(2)};           // (input, shape, offset, valid_shape, drop_dims)
  if (IsOp(call, "tile.extract")) return {at(1), at(2)};  // (src, index_row, index_col, shape)
  // (dst, src, dst_offset, src_offset, shapes[, valid_shape]) — DPS: the op reads
  // a GM row window at ``src_offset`` and writes it into its own accumulator at
  // ``dst_offset``. Only ``src_offset`` is the READ window, so only it localizes:
  // a lane-derived src_offset means the two lanes gather DIFFERENT rows (the
  // per-lane scattered gather). A lane-derived dst_offset with a lane-invariant
  // src is the opposite shape — both lanes fetch the same rows into different
  // slots of a FULL-width accumulator — which is exactly what this scan must
  // still reject.
  if (IsOp(call, "tile.gather_row")) return {at(3)};
  return {};
}

// Track tiles that are part of the half-width boundary dataflow: results of
// tile.aiv_shard, plus results of VECTOR-affine ops that consume such a half
// tile. Any VECTOR-affine op consuming NONE of them operates on full-width data
// — exactly what the implicit affinity gate would have halved. Records the names
// of such full-width vector ops in a single ordered walk.
void ScanRegionHalfWidth(const std::vector<StmtPtr>& stmts, HalfWidthScan& scan) {
  for (const auto& stmt : stmts) {
    CallPtr leaf;
    VarPtr def_var;
    auto assign = std::dynamic_pointer_cast<const AssignStmt>(stmt);
    if (assign) {
      leaf = AsCall(assign->value_);
      def_var = assign->var_;
    } else if (auto eval = std::dynamic_pointer_cast<const EvalStmt>(stmt)) {
      leaf = AsCall(eval->expr_);
    }

    // --- Lane-index dataflow (SCALARS) --------------------------------------
    // SEED: the region's own lane index, bound by ``aiv_id =
    // tile.get_subblock_idx()`` (the parser emits it as the region body's first
    // statement — ast_parser.py::_emit_split_aiv_region). Matched by OP, not by
    // position, so a re-injected or reordered binding is still found. The parser
    // emits it unconditionally, so the set is normally non-empty from statement
    // one; it stays empty only if the binding is absent (e.g. DCE stripped an
    // unused aiv_id, or hand-built IR omits it), in which case nothing is
    // admitted below and the guard behaves exactly as before.
    // PROPAGATE: any scalar bound from an expression referencing a lane-derived
    // scalar (``kv_lane0 = kv0 + aiv_id * 64``). Program order + SSA put every
    // def before its uses, so one forward pass reaches the fixpoint. A lane value
    // carried through a loop iter_arg is NOT propagated (the walk does not visit
    // loop phi defs) — conservative: it rejects rather than wrongly admits.
    if (def_var && As<ScalarType>(def_var->GetType()) &&
        (IsOp(leaf, "tile.get_subblock_idx") ||
         // ``assign &&`` is defensive: def_var is currently only set in the
         // AssignStmt arm above, so def_var non-null already implies it.
         (assign && ReferencesLaneIndex({assign->value_}, scan.lane_scalars)))) {
      scan.lane_scalars.insert(def_var.get());
    }

    if (leaf && leaf->op_) {
      // aiv_shard produces a HALF tile (the C->V boundary); seed the dataflow.
      if (IsOp(leaf, "tile.aiv_shard")) {
        if (def_var) scan.half_tiles.insert(def_var.get());
        continue;
      }
      // aic_gather doubles HALF -> FULL (the V->C boundary back to cube); its
      // result leaves the half-width dataflow, so it is not tracked.
      if (IsOp(leaf, "tile.aic_gather")) {
        continue;
      }
      // Pure generators are lane-invariant ROOTS: the result is a function of the
      // op's ATTRIBUTES ONLY -- they read no tile and no memory, so there is no
      // "which half do I read?" question for them to get wrong, and replicating
      // one on both lanes is correct by construction at whatever extent the
      // author wrote.
      //
      // Deliberately NOT inserted into half_tiles: being lane-invariant makes a
      // generator SAFE, it does not make it HALF-WIDTH. Admitting it would let a
      // full-width generator vouch for its consumers and silently suppress a real
      // rejection -- e.g. `z = tile.full([128,128]); y = tile.add(z, z);
      // tile.store(y, [0,0], out, atomic)`, where both lanes would atomically add
      // the same full tile (double-counted result). Leaving it NEUTRAL keeps each
      // consumer judged on its own merits.
      //
      // (tile.create is listed for category completeness; it also classifies
      // SHARED, so the VECTOR arm below would never report it anyway. For
      // tile.random, replication means both lanes draw the SAME stream from the
      // same key/counter -- identical to what the implicit path produces, since
      // halving shrinks the shape, not the stream. Per-lane independence is the
      // author's job: vary the counter with aiv_id.)
      if (IsOp(leaf, "tile.full") || IsOp(leaf, "tile.create") || IsOp(leaf, "tile.ci") ||
          IsOp(leaf, "tile.random")) {
        continue;
      }
      // Dataflow propagation over tile-producing ops. An op that consumes a half
      // tile STAYS in the half-width dataflow regardless of its affinity
      // classification -- crucially this includes a Vec->Vec tile.move between the
      // shard and the compute, which classifies MIXED/SHARED (not VECTOR). Only a
      // VECTOR-affine tile op that consumes NONE of the half tiles is genuinely
      // full-width: that is exactly what the implicit affinity gate would halve,
      // and what the explicit-passthrough path would leave un-localized (both AIV
      // lanes computing the full tile). A scalar-producing VECTOR op (e.g.
      // tile.get_subblock_idx) is not a tile op, so it never flags.
      if (std::dynamic_pointer_cast<const TileType>(leaf->GetType()) != nullptr) {
        bool consumes_half = false;
        for (const auto& arg : leaf->args_) {
          if (auto v = AsVarLike(arg)) {
            if (scan.half_tiles.count(v.get()) != 0) {
              consumes_half = true;
              break;
            }
          }
        }
        // Stays in the half-width dataflow when it either consumes a half tile,
        // or is AUTHOR-LOCALIZED: an op whose ADDRESS args reference the region's
        // lane index, so the author explicitly wrote it per-lane — e.g. a GM load
        // at ``[kv0 + aiv_id * 64, 0]``. That is the same trust the pass already
        // extends to tile.store, whose lane-dependent offset it never checks (a
        // store returns a TensorType, so it never reaches this branch).
        //
        // Localization is trusted only via AddressArgs, and so only on addressing
        // ops. On any other op a lane-derived scalar is just an operand and proves
        // nothing about width; without that restriction
        // ``tile.set_validshape(full_width_tile, 1, valid_aiv)`` would launder a
        // full-width tile into the half-width dataflow and silence every
        // downstream check.
        if (consumes_half ||
            (!scan.lane_scalars.empty() && ReferencesLaneIndex(AddressArgs(leaf), scan.lane_scalars))) {
          if (def_var) scan.half_tiles.insert(def_var.get());
        } else if (ClassifyCallAffinity(leaf) == CoreAffinity::VECTOR) {
          scan.full_width_vec_ops.push_back(leaf->op_->name_);
        }
        continue;
      }
    }
    if (auto for_stmt = std::dynamic_pointer_cast<const ForStmt>(stmt)) {
      ScanRegionHalfWidth(transform_utils::FlattenToStmts(for_stmt->body_), scan);
    } else if (auto if_stmt = std::dynamic_pointer_cast<const IfStmt>(stmt)) {
      ScanRegionHalfWidth(transform_utils::FlattenToStmts(if_stmt->then_body_), scan);
      if (if_stmt->else_body_.has_value()) {
        ScanRegionHalfWidth(transform_utils::FlattenToStmts(*if_stmt->else_body_), scan);
      }
    } else if (auto while_stmt = std::dynamic_pointer_cast<const WhileStmt>(stmt)) {
      ScanRegionHalfWidth(transform_utils::FlattenToStmts(while_stmt->body_), scan);
    } else if (auto seq = std::dynamic_pointer_cast<const SeqStmts>(stmt)) {
      ScanRegionHalfWidth(seq->stmts_, scan);
    }
  }
}

// Reject a region that MIXES explicit half-width boundary ops (tile.aiv_shard /
// tile.aic_gather) with a plain full-width VECTOR-affine op the implicit
// affinity-gated path would otherwise halve (user-facing limitation). The
// explicit boundary keeps the whole region in half-width form and the
// passthrough path splices the body UNCHANGED, so a full-width vector op would be
// left un-localized and BOTH AIV lanes would compute the full tile (a silent
// miscompile). A purely-explicit region — every vector op derived from the
// aiv_shard result — passes through unchanged.
void ValidateMixedExplicitRegion(const std::vector<StmtPtr>& stmts, const Span& region_span) {
  HalfWidthScan scan;
  ScanRegionHalfWidth(stmts, scan);
  if (scan.full_width_vec_ops.empty()) return;

  std::string ops;
  for (size_t i = 0; i < scan.full_width_vec_ops.size(); ++i) {
    if (i != 0) ops += ", ";
    ops += scan.full_width_vec_ops[i];
  }
  CHECK_SPAN(false, region_span)
      << "LowerAutoVectorSplit: a pl.split_aiv region mixes explicit "
         "tile.aiv_shard/tile.aic_gather boundary ops with plain full-width vector op(s) ["
      << ops
      << "] that operate outside the per-lane half-width dataflow. The explicit boundary keeps the "
         "region in half-width form, so these full-width ops would be left un-localized and both AIV "
         "lanes would compute the full tile. Fix it one of three ways: (1) derive the op from the "
         "tile.aiv_shard result; (2) localize it yourself with the region's lane index, e.g. load at "
         "'base + aiv_id * HALF' at the half extent — an op whose READ address references aiv_id is "
         "per-lane by construction and is accepted. The lane reference must land in that read "
         "address (the offset selecting which window the op reads), not in a shape, a valid_shape, "
         "or a destination slot; or (3) remove the explicit tile.aiv_shard/tile.aic_gather and let "
         "the implicit affinity-gated path halve the region.";
}

// Top-level walk for the explicit ``SplitAivScopeStmt`` path. Statements OUTSIDE
// any region are emitted FULL-WIDTH (passed through unchanged); the LowerStmts
// SplitAivScopeStmt arm lowers each region's vector compute with region-local
// maps. Recurses into for/if/seq so a region nested in a loop or conditional is
// found and lowered while its surrounding full-width compute is preserved.
std::vector<StmtPtr> LowerExplicitRegions(const std::vector<StmtPtr>& stmts,
                                          std::unordered_set<std::string>& used_names) {
  std::vector<StmtPtr> result;
  result.reserve(stmts.size());
  for (const auto& stmt : stmts) {
    if (As<SplitAivScopeStmt>(stmt)) {
      // Delegate to the LowerStmts region arm; it reads the region's own mode, so
      // the placeholder mode/dim args are ignored. Region-local maps live inside
      // the arm, so nothing leaks to the surrounding full-width context. The
      // shared ``used_names`` is grown by each region's index injection so
      // sibling regions get unique subblock indices.
      std::unordered_map<const Var*, TileInfo> ignored_tile_vars;
      std::unordered_map<const Var*, VarPtr> ignored_var_repl;
      auto lowered = LowerStmts({stmt}, SplitMode::None, /*split_dim=*/0, ignored_tile_vars,
                                /*subblock_idx=*/nullptr, ignored_var_repl, used_names);
      for (auto& s : lowered) result.push_back(s);
      continue;
    }
    if (auto for_stmt = std::dynamic_pointer_cast<const ForStmt>(stmt)) {
      auto new_body = LowerExplicitRegions(transform_utils::FlattenToStmts(for_stmt->body_), used_names);
      auto new_for = MutableCopy(for_stmt);
      new_for->body_ = loop_repair::MakeBody(new_body, for_stmt->span_);
      result.push_back(new_for);
      continue;
    }
    // A SplitAivScopeStmt may also nest inside a while body; mirror the ForStmt
    // arm so the region is lowered + erased rather than surviving to the codegen
    // guard (which rejects any live SplitAivScopeStmt).
    if (auto while_stmt = std::dynamic_pointer_cast<const WhileStmt>(stmt)) {
      auto new_body = LowerExplicitRegions(transform_utils::FlattenToStmts(while_stmt->body_), used_names);
      auto new_while = MutableCopy(while_stmt);
      new_while->body_ = loop_repair::MakeBody(new_body, while_stmt->span_);
      result.push_back(new_while);
      continue;
    }
    if (auto if_stmt = std::dynamic_pointer_cast<const IfStmt>(stmt)) {
      auto new_then = LowerExplicitRegions(transform_utils::FlattenToStmts(if_stmt->then_body_), used_names);
      std::optional<StmtPtr> new_else;
      if (if_stmt->else_body_.has_value()) {
        auto new_else_stmts =
            LowerExplicitRegions(transform_utils::FlattenToStmts(*if_stmt->else_body_), used_names);
        new_else = loop_repair::MakeBody(new_else_stmts, if_stmt->span_);
      }
      auto new_if = MutableCopy(if_stmt);
      new_if->then_body_ = loop_repair::MakeBody(new_then, if_stmt->span_);
      new_if->else_body_ = new_else;
      result.push_back(new_if);
      continue;
    }
    if (auto seq = std::dynamic_pointer_cast<const SeqStmts>(stmt)) {
      result.push_back(std::make_shared<SeqStmts>(LowerExplicitRegions(seq->stmts_, used_names), seq->span_));
      continue;
    }
    // Out-of-region statement: full width, unchanged.
    result.push_back(stmt);
  }
  return result;
}

// Find the first live ``SplitAivScopeStmt`` in a body (O(N) single walk), so a
// caller can point a diagnostic at the author's ``for aiv_id in
// pl.split_aiv(...)`` line via the returned node's span. Unlike the hand-rolled
// walks above this is an IRVisitor, so it descends into ScopeStmt bodies too —
// that asymmetry is exactly what the post-lowering guard below exists to catch.
class SplitAivScopeFinder : public IRVisitor {
 public:
  SplitAivScopeStmtPtr found_;

 protected:
  void VisitStmt_(const SplitAivScopeStmtPtr& op) override {
    if (!found_) found_ = op;
  }
};

SplitAivScopeStmtPtr FindFirstSplitAivScope(const StmtPtr& body) {
  if (!body) return nullptr;
  SplitAivScopeFinder finder;
  finder.VisitStmt(body);
  return finder.found_;
}

bool BodyContainsSplitAivScope(const StmtPtr& body) { return FindFirstSplitAivScope(body) != nullptr; }

// Find the first ``ScopeStmt`` of ANY kind in a body, for the AUTO path's guard
// below. Like the finder above this is an IRVisitor, so it reaches scopes the
// hand-rolled lowering walks do not — which is the whole point: it must find
// exactly what those walks would silently step over.
class AnyScopeFinder : public IRVisitor {
 public:
  ScopeStmtPtr found_;

 protected:
  void VisitStmt_(const InCoreScopeStmtPtr& op) override { Record(op); }
  void VisitStmt_(const ClusterScopeStmtPtr& op) override { Record(op); }
  void VisitStmt_(const HierarchyScopeStmtPtr& op) override { Record(op); }
  void VisitStmt_(const SpmdScopeStmtPtr& op) override { Record(op); }
  void VisitStmt_(const SplitAivScopeStmtPtr& op) override { Record(op); }
  void VisitStmt_(const RuntimeScopeStmtPtr& op) override { Record(op); }
  void VisitStmt_(const CommDomainScopeStmtPtr& op) override { Record(op); }

 private:
  template <typename T>
  void Record(const std::shared_ptr<const T>& op) {
    if (!found_) found_ = op;
    IRVisitor::VisitStmt(op->body_);
  }
};

ScopeStmtPtr FindFirstScope(const StmtPtr& body) {
  if (!body) return nullptr;
  AnyScopeFinder finder;
  finder.VisitStmt(body);
  return finder.found_;
}

// Lower an InCore function that carries explicit ``SplitAivScopeStmt`` regions:
// halve only the vector compute inside each region (region-local), leave
// out-of-region compute full-width, drop each scope wrapper, and stamp
// ``split_aiv`` (idempotent — already bridged at OutlineIncoreScopes) plus
// ``split_aiv_region_validated`` (signals ExpandMixedKernel (pass 21) to skip its func-mode check).
//
// Region lowering deliberately does NOT cross a ``ScopeStmt``: a scope carries
// outlining and name-visibility semantics that region-local halving must not
// reach through. So every region must already be scope-free by the time this
// runs, and the guard below enforces that instead of letting an unreachable one
// ride through — unlowered and un-validated, yet stamped "region validated".
FunctionPtr LowerExplicitRegionFunction(const FunctionPtr& func) {
  auto stmts = transform_utils::FlattenToStmts(func->body_);
  // Seed the reservation set with every name visible in the function body (params
  // + all def/use names) so each region's injected ``subblock_idx`` is unique
  // both against existing bindings and against sibling regions' indices.
  std::unordered_set<std::string> used_names = CollectBodyVarNames(func->body_);
  for (const auto& p : func->params_) used_names.insert(p->name_hint_);
  auto new_stmts = LowerExplicitRegions(stmts, used_names);
  StmtPtr new_body =
      (new_stmts.size() == 1) ? new_stmts[0] : std::make_shared<SeqStmts>(new_stmts, func->span_);

  // Every region must have been consumed. A surviving one means it sat behind a
  // scope the lowering walk does not enter: OutlineIncoreScopes outlines scopes
  // only out of Opaque / Orchestration functions, so an author-written scope
  // inside a function declared ``pl.FunctionType.InCore`` reaches this pass
  // intact. Reject it here, pointing at the region; passing it through would
  // skip every region guard and then fail far downstream as an internal
  // assertion in PTO codegen.
  //
  // Span safety: the check fails exactly when ``survivor`` is non-null, so the
  // dereference in the span argument is only evaluated when it is valid.
  auto survivor = FindFirstSplitAivScope(new_body);
  CHECK_SPAN(!survivor, survivor->span_)
      << "LowerAutoVectorSplit: this pl.split_aiv region is nested inside a scope and cannot be "
         "lowered — region lowering does not cross a scope boundary. The scope may be one you "
         "wrote ('with pl.at(...)') or the InCore scope the parser adds around a top-level "
         "pl.split_aiv. OutlineIncoreScopes only outlines scopes out of Opaque / Orchestration "
         "functions, so a scope inside a function declared pl.FunctionType.InCore reaches this "
         "pass intact. Fix it one of two ways: (1) declare the enclosing function with plain "
         "@pl.function / @pl.jit so the scope is outlined before this pass runs; or (2) move the "
         "pl.split_aiv region out of the enclosing scope.";

  // ... and no scope of any other kind may remain either. The check above proves
  // no region sat BEHIND a scope; this one covers the mirror case — a scope
  // nested INSIDE a region body. The region itself is consumed there, so the
  // check above passes, yet the inner walks (``LowerStmts``,
  // ``CheckNoCubeTileHalved``, ``ScanRegionHalfWidth``) step over the scope
  // rather than entering it, and the vector ops inside it are spliced out
  // FULL-WIDTH — both AIV lanes computing the whole tile, with no diagnostic.
  //
  // By this point a region-bearing InCore function should hold no scope at all:
  // OutlineIncoreScopes lifts every scope it can see into its own function
  // (a ``with pl.at(...)`` inside a region becomes its own ``*_incore_0``), and
  // AivSplitValid check (h) rejects authoring a region in an InCore function the
  // outliner did not produce. So this is unreachable from the DSL and guards IR
  // that never went through pass 8 — hand-built, or a deserialized ``.pto``.
  // Cheap to state, and the alternative failure is silent rather than loud.
  auto scope_survivor = FindFirstScope(new_body);
  CHECK_SPAN(!scope_survivor, scope_survivor->span_)
      << "LowerAutoVectorSplit: a scope survives inside a pl.split_aiv region body. Region "
         "lowering does not cross a scope boundary, so the vector ops inside this scope would be "
         "emitted full-width and BOTH AIV lanes would compute the whole tile. Every scope must "
         "already be outlined by the time this pass runs — declare the enclosing function with "
         "plain @pl.function / @pl.jit (Opaque) so OutlineIncoreScopes lifts it, or drop the "
         "scope from inside the region.";

  auto [cloned_body, clone_map_unused] = DeepClone(new_body);
  (void)clone_map_unused;

  // Earned only now: the guard above proves every region was actually lowered,
  // so ``split_aiv_region_validated`` is a true claim and pass 21
  // (ExpandMixedKernel) may skip its own single-func-mode check on its strength.
  auto attrs = func->attrs_;
  attrs.erase(std::remove_if(attrs.begin(), attrs.end(),
                             [](const auto& kv) {
                               return kv.first == kAttrSplitAiv || kv.first == kAttrSplitAivRegionValidated;
                             }),
              attrs.end());
  attrs.emplace_back(kAttrSplitAiv, true);
  attrs.emplace_back(kAttrSplitAivRegionValidated, true);

  auto new_func = MutableCopy(func);
  new_func->body_ = cloned_body;
  new_func->attrs_ = std::move(attrs);
  return new_func;
}

std::vector<std::pair<std::string, std::any>> WithSplitAivAttrs(const FunctionPtr& func, SplitMode mode) {
  auto attrs = func->attrs_;
  attrs.erase(std::remove_if(attrs.begin(), attrs.end(),
                             [](const auto& kv) {
                               return kv.first == "split" || kv.first == kAttrSplitAiv ||
                                      kv.first == kAttrDualAivDispatch;
                             }),
              attrs.end());
  attrs.emplace_back("split", static_cast<int>(mode));
  attrs.emplace_back(kAttrSplitAiv, true);
  return attrs;
}

FunctionPtr LowerFunction(const FunctionPtr& func, SplitMode mode) {
  int split_dim = SplitDimension(mode);

  // Inject get_subblock_idx at the top (is_aiv=true => a binding is prepended).
  auto injected = InjectSubblockIdx(func, /*is_aiv=*/true);

  std::unordered_map<const Var*, TileInfo> tile_vars;
  std::unordered_map<const Var*, VarPtr> var_replacements;
  // The AUTO whole-function path carries no SplitAivScopeStmt regions, so the
  // region arm is never reached; this set is only a placeholder for the shared
  // LowerStmts signature, seeded with the names InjectSubblockIdx already
  // reserved.
  std::unordered_set<std::string> used_names = injected.used_names;

  // Partition the split axis by the boundary's VALID region when the body is a
  // single ragged crossing (see split_axis::ResolveLaneStride); null keeps the
  // universal box partition.
  auto lane_stride = split_axis::ResolveLaneStride(injected.body_stmts, split_dim);

  auto new_stmts = LowerStmts(injected.body_stmts, mode, split_dim, tile_vars, injected.subblock_idx_expr,
                              var_replacements, used_names, lane_stride);

  // Effective cube-operand backstop: re-walk the rebuilt body and assert no
  // CUBE-affine op operates on a halved tile (see CheckNoCubeTileHalved).
  bool cube_halved = false;
  CheckNoCubeTileHalved(new_stmts, tile_vars, cube_halved);

  INTERNAL_CHECK_SPAN(!cube_halved, func->span_)
      << "Internal error: LowerAutoVectorSplit halved a CUBE-affinity op in '" << func->name_
      << "' — the vector-sub-region affinity gate leaked into a cube operand.";

  StmtPtr new_body =
      (new_stmts.size() == 1) ? new_stmts[0] : std::make_shared<SeqStmts>(new_stmts, func->span_);
  if (!var_replacements.empty()) {
    new_body = transform_utils::Substitute(new_body, var_replacements);
  }
  auto [cloned_body, clone_map_unused] = DeepClone(new_body);
  (void)clone_map_unused;

  auto new_func = MutableCopy(func);
  new_func->body_ = cloned_body;
  new_func->attrs_ = WithSplitAivAttrs(func, mode);
  return new_func;
}

bool IsAlreadyExplicitSplitAiv(const FunctionPtr& func) {
  return func->HasAttr(kAttrSplitAiv) && func->GetAttr<bool>(kAttrSplitAiv, false);
}

// Roll up the cross-core affinity of a statement list, mirroring
// ExpandMixedKernel's AnalyzeStmtsAffinity (combined == MIXED <=> the function
// spans both cube and vector). The tpop-result downgrade that AnalyzeStmtAffinity
// applies is intentionally omitted: it is irrelevant here because (a) tpops are
// inserted by ExpandMixedKernel, which runs AFTER this pass, and (b) the only
// functions carrying aiv_shard/aic_gather (the other tpop-like ops) are already
// explicit split_aiv and filtered out by IsAlreadyExplicitSplitAiv before this
// is reached. So over the inputs this pass actually sees, the roll-up matches
// ExpandMixedKernel's is_mixed decision exactly.
CoreAffinity RollupAffinity(const std::vector<StmtPtr>& stmts) {
  CoreAffinity combined = CoreAffinity::SHARED;
  for (const auto& stmt : stmts) {
    CoreAffinity result = CoreAffinity::SHARED;
    if (auto assign = std::dynamic_pointer_cast<const AssignStmt>(stmt)) {
      if (auto call = AsCall(assign->value_)) result = ClassifyCallAffinity(call);
    } else if (auto eval = std::dynamic_pointer_cast<const EvalStmt>(stmt)) {
      if (auto call = AsCall(eval->expr_)) result = ClassifyCallAffinity(call);
    } else if (auto for_stmt = std::dynamic_pointer_cast<const ForStmt>(stmt)) {
      result = RollupAffinity(transform_utils::FlattenToStmts(for_stmt->body_));
    } else if (auto if_stmt = std::dynamic_pointer_cast<const IfStmt>(stmt)) {
      result = RollupAffinity(transform_utils::FlattenToStmts(if_stmt->then_body_));
      if (if_stmt->else_body_.has_value()) {
        result =
            CombineAffinity(result, RollupAffinity(transform_utils::FlattenToStmts(*if_stmt->else_body_)));
      }
    } else if (auto while_stmt = std::dynamic_pointer_cast<const WhileStmt>(stmt)) {
      result = RollupAffinity(transform_utils::FlattenToStmts(while_stmt->body_));
    } else if (auto scope = std::dynamic_pointer_cast<const ScopeStmt>(stmt)) {
      // A scope's affinity IS its body's — the compute inside it is still this
      // function's compute. Without this arm a scope fell to the SHARED default
      // below, so `IsMixedCubeVector` reported false for any scope-bodied
      // function and the AUTO arm skipped it *silently*. Classifying correctly
      // is only half the fix: lowering still must not cross a scope boundary,
      // so a function that now classifies MIXED is rejected by the guard in the
      // pass rather than half-lowered.
      result = RollupAffinity(transform_utils::FlattenToStmts(scope->body_));
    } else if (auto seq = std::dynamic_pointer_cast<const SeqStmts>(stmt)) {
      result = RollupAffinity(seq->stmts_);
    }
    combined = CombineAffinity(combined, result);
  }
  return combined;
}

// A function needs the cube<->vector boundary convergence iff it is genuinely
// mixed. A PURE-vector pl.split function (e.g. an elementwise op split across the
// two AIV lanes) has no boundary: ExpandMixedKernel converts it to a plain AIV
// function and STRIPS its split attr, so stamping split_aiv + halving it here
// would desync (split_aiv survives, split is stripped) and trip SplitVectorKernel.
// Leave such functions untouched -- they keep their prior (un-split) behavior.
bool IsMixedCubeVector(const FunctionPtr& func) {
  if (!func->body_) return false;
  return RollupAffinity(transform_utils::FlattenToStmts(func->body_)) == CoreAffinity::MIXED;
}

}  // namespace

namespace pass {

Pass LowerAutoVectorSplit() {
  auto pass_func = [](const ProgramPtr& program) -> ProgramPtr {
    std::vector<FunctionPtr> new_functions;
    bool changed = false;
    new_functions.reserve(program->functions_.size());

    for (const auto& [gvar, func] : program->functions_) {
      auto mode = func->GetSplitMode();
      const bool is_incore = (func->func_type_ == FunctionType::InCore);
      // EXPLICIT region path: an InCore function whose body still carries one or
      // more SplitAivScopeStmt regions (preserved through OutlineIncoreScopes).
      // Each region carries its own mode, so this is checked before the AUTO path
      // and handles the multi-mode case the single func-level mode cannot.
      if (is_incore && BodyContainsSplitAivScope(func->body_)) {
        new_functions.push_back(LowerExplicitRegionFunction(func));
        changed = true;
        continue;
      }
      // AUTO whole-function path: lower genuinely mixed (cube<->vector)
      // functions. Pure-vector pl.split functions have no boundary to converge;
      // ExpandMixedKernel strips their split, so marking them split_aiv here
      // would desync.
      if (is_incore && mode.has_value() && mode.value() != SplitMode::None &&
          !IsAlreadyExplicitSplitAiv(func) && IsMixedCubeVector(func)) {
        // Same rule as the explicit region path: the halving walks recurse into
        // for / while / if / seq but deliberately NOT into a ScopeStmt, whose
        // outlining and name-visibility semantics whole-function halving must
        // not reach through. So a mixed AUTO function whose body still carries a
        // scope cannot be lowered, and saying so is the only safe answer:
        // passing it through leaves the kernel silently un-split, while lowering
        // it would stamp `split_aiv` on a body whose in-scope ops were never
        // halved. Normally unreachable — the function-level `split` attr this
        // path keys on is written by OutlineIncoreScopes, which consumes the
        // scope in the same step — but nothing enforces that, which is exactly
        // why it is checked rather than assumed.
        //
        // Span safety: the check fails exactly when `scope` is non-null, so the
        // dereference in the span argument only runs when it is valid.
        auto scope = FindFirstScope(func->body_);
        CHECK_SPAN(!scope, scope->span_)
            << "LowerAutoVectorSplit: function '" << func->name_
            << "' declares an AUTO split (optimizations=[pl.split(...)]) and is a mixed "
               "cube/vector kernel, but its body still contains a scope — whole-function "
               "halving does not cross a scope boundary, so the split cannot be applied. "
               "Declare the enclosing function with plain @pl.function / @pl.jit (Opaque) so "
               "OutlineIncoreScopes lifts the scope before this pass runs, or move the split "
               "declaration onto the scope itself.";
        new_functions.push_back(LowerFunction(func, mode.value()));
        changed = true;
      } else {
        new_functions.push_back(func);
      }
    }

    if (!changed) return program;
    return std::make_shared<Program>(new_functions, program->name_, program->span_);
  };

  return CreateProgramPass(pass_func, "LowerAutoVectorSplit", kLowerAutoVectorSplitProperties);
}

}  // namespace pass
}  // namespace ir
}  // namespace pypto
