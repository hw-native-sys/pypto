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
#include "pypto/ir/memory_space.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
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

constexpr const char* kDualAivDispatchAttr = "dual_aiv_dispatch";
constexpr const char* kSplitAivAttr = "split_aiv";
// Stamped by the explicit-region path so ExpandMixedKernel (pass 21) skips its
// single-func-mode transpose-hazard check (validated per-region here instead).
constexpr const char* kSplitAivRegionValidatedAttr = "split_aiv_region_validated";

CallPtr AsCall(const ExprPtr& expr) { return std::dynamic_pointer_cast<const Call>(expr); }

// Defined below LowerStmts; forward-declared so the LowerStmts SplitAivScopeStmt
// arm (explicit per-region lowering) can run them on the lowered region body.
void CheckNoCubeTileHalved(const std::vector<StmtPtr>& stmts,
                           const std::unordered_map<const Var*, TileInfo>& halved, bool& cube_halved);
void ValidateTransposeSplitHazard(const std::vector<StmtPtr>& stmts, int split_dim, const Span& region_span);
void ValidateMixedExplicitRegion(const std::vector<StmtPtr>& stmts, const Span& region_span);

// Half of a split-axis physical extent: ConstInt even -> value/2, dynamic ->
// floordiv(dim, 2). Mirrors split_axis::ComputeHalfDimSize (anonymous in
// split_axis_utils.cpp) for the tracked TileInfo extent.
ExprPtr HalfDimExtent(const ExprPtr& dim_size) {
  if (auto ci = std::dynamic_pointer_cast<const ConstInt>(dim_size)) {
    return std::make_shared<ConstInt>(ci->value_ / 2, ci->dtype(), ci->span_);
  }
  auto two = std::make_shared<ConstInt>(2, GetScalarDtype(dim_size), dim_size->span_);
  return MakeFloorDiv(dim_size, two, dim_size->span_);
}

// Re-attach a memory space to a split-reshape op's deduced result type. The
// aiv_shard / aic_gather deducer correctly halves/doubles the split-axis shape
// and valid_shape but drops the memory space (see DeduceSplitReshape); the
// boundary's target memory (Vec for the shard/gather result) is restored here.
TypePtr ReshapeTypeWithMemory(const TypePtr& deduced_type, const std::optional<MemorySpace>& mem) {
  auto tt = std::dynamic_pointer_cast<const TileType>(deduced_type);
  if (!tt) return deduced_type;
  return std::make_shared<TileType>(tt->shape_, tt->dtype_, tt->memref_, tt->tile_view_, mem);
}

std::optional<MemorySpace> TileMemory(const TypePtr& type) {
  if (auto tt = std::dynamic_pointer_cast<const TileType>(type)) return tt->memory_space_;
  return std::nullopt;
}

// Make a split-kwarg call (split int attr is the SplitMode int encoding).
CallPtr MakeReshapeOpCall(const std::string& op_name, const ExprPtr& source, int split_int,
                          const Span& span) {
  std::vector<std::pair<std::string, std::any>> kwargs{{"split", std::any(split_int)}};
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
std::vector<StmtPtr> LowerStmts(const std::vector<StmtPtr>& stmts, SplitMode mode, int split_int,
                                int split_dim, std::unordered_map<const Var*, TileInfo>& tile_vars,
                                const ExprPtr& subblock_idx,
                                std::unordered_map<const Var*, VarPtr>& var_replacements,
                                std::unordered_set<std::string>& used_names) {
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
      // halving, no offset localization, no aiv_shard/aic_gather. Bind aiv_id and
      // splice the body through unchanged, then drop the scope wrapper. (This must
      // branch BEFORE SplitDimension(rmode) below, which rejects None. A
      // shard/gather op inside a None region is caught by the AivSplitValid
      // verifier — there is no split axis for it to mark.)
      if (rmode == SplitMode::None) {
        // A boundary op needs a split axis to mark; a task-parallel region has
        // none. The AivSplitValid verifier rejects this with a user diagnostic,
        // but guard here too so a verification-off build fails loudly rather than
        // miscompiling (full tile silently passed where a half is expected).
        CHECK_SPAN(!RegionBodyHasExplicitBoundary(reg->body_), reg->span_)
            << "pl.split_aiv(mode=pl.SplitMode.NONE) region must not contain tile.aiv_shard / "
               "tile.aic_gather: a task-parallel region has no split axis to shard / gather. Use "
               "mode=pl.SplitMode.UP_DOWN / LEFT_RIGHT for data-parallel halving.";
        // Pass the body through UNCHANGED, dropping only the scope wrapper. The
        // body already opens with aiv_id = get_subblock_idx() (the author's lane
        // index, used for disjoint dispatch). No halving and no per-lane offset
        // localization happen here, so there is no second internal subblock_idx
        // to inject — both AIV lanes run the full body verbatim.
        for (auto& s : region_stmts) result.push_back(s);
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
        for (auto& s : region_stmts) result.push_back(s);
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
      auto lowered = LowerStmts(inj.body_stmts, rmode, static_cast<int>(rmode), rdim, r_tile_vars,
                                inj.subblock_idx_expr, r_var_repl, used_names);

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
      for (auto& s : transform_utils::FlattenToStmts(region_body)) result.push_back(s);
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
          auto shard = MakeReshapeOpCall("tile.aiv_shard", call->args_[0], split_int, call->span_);
          // Result: the op's deduced HALF type, with the boundary target memory
          // (the move's destination memory, e.g. Vec) re-attached.
          auto half_type = ReshapeTypeWithMemory(shard->GetType(), TileMemory(call->GetType()));
          auto new_var = std::make_shared<Var>(assign->var_->name_hint_, half_type, assign->var_->span_);
          auto shard_typed =
              std::make_shared<Call>(shard->op_, shard->args_, shard->kwargs_, half_type, shard->span_);
          if (auto tt = std::dynamic_pointer_cast<const TileType>(call->GetType());
              tt && split_dim < static_cast<int>(tt->shape_.size())) {
            TileInfo info{HalfDimExtent(tt->shape_[split_dim])};
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
          if (auto src_var = AsVarLike(src)) {
            auto it = var_replacements.find(src_var.get());
            if (it != var_replacements.end()) src = it->second;
          }
          auto gather = MakeReshapeOpCall("tile.aic_gather", src, split_int, call->span_);
          // Gather result: full shape, Vec memory (inherit input side).
          auto src_tt = std::dynamic_pointer_cast<const TileType>(src->GetType());
          auto gather_tt = std::dynamic_pointer_cast<const TileType>(gather->GetType());
          TypePtr gather_type = gather->GetType();
          if (src_tt && gather_tt) {
            gather_type = std::make_shared<TileType>(gather_tt->shape_, gather_tt->dtype_, gather_tt->memref_,
                                                     gather_tt->tile_view_, src_tt->memory_space_);
          }
          auto gather_typed =
              std::make_shared<Call>(gather->op_, gather->args_, gather->kwargs_, gather_type, gather->span_);
          // Name the gathered FULL tile with the cube-destination's "_mat" suffix:
          // ExpandMixedKernel folds this gather into the AIC-side V->C boundary and
          // names the synthesized tpop after this var. The standalone split_aiv
          // move-boundary path names that tpop BuildBoundaryTpopName(AIC, dest) =
          // "<dest>_mat", so matching it here keeps both paths' .pto byte-identical.
          auto full_vec_var =
              std::make_shared<Var>(assign->var_->name_hint_ + "_mat", gather_type, assign->span_);
          result.push_back(std::make_shared<AssignStmt>(full_vec_var, gather_typed, assign->span_));
          // Original cube placement move, now on the FULL gathered tile.
          std::vector<ExprPtr> move_args = call->args_;
          move_args[0] = full_vec_var;
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
        auto lowered = ProcessStmts({stmt}, mode, split_int, split_dim, tile_vars, /*is_aiv=*/true,
                                    subblock_idx, var_replacements);
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
      auto new_body =
          LowerStmts(body, mode, split_int, split_dim, tile_vars, subblock_idx, var_replacements, used_names);
      auto new_for = MutableCopy(for_stmt);
      new_for->body_ = loop_repair::MakeBody(new_body, for_stmt->span_);
      result.push_back(new_for);
      continue;
    }
    if (auto if_stmt = std::dynamic_pointer_cast<const IfStmt>(stmt)) {
      auto then_body = transform_utils::FlattenToStmts(if_stmt->then_body_);
      auto new_then = LowerStmts(then_body, mode, split_int, split_dim, tile_vars, subblock_idx,
                                 var_replacements, used_names);
      std::optional<StmtPtr> new_else;
      if (if_stmt->else_body_.has_value()) {
        auto else_body = transform_utils::FlattenToStmts(*if_stmt->else_body_);
        auto new_else_stmts = LowerStmts(else_body, mode, split_int, split_dim, tile_vars, subblock_idx,
                                         var_replacements, used_names);
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
      auto new_body =
          LowerStmts(body, mode, split_int, split_dim, tile_vars, subblock_idx, var_replacements, used_names);
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

// Track tiles that are part of the half-width boundary dataflow: results of
// tile.aiv_shard, plus results of VECTOR-affine ops that consume such a half
// tile. Any VECTOR-affine op consuming NONE of them operates on full-width data
// — exactly what the implicit affinity gate would have halved. Records the names
// of such full-width vector ops (a single ordered walk; ``half_tiles`` is shared
// across recursion so the propagation follows program order).
void ScanRegionHalfWidth(const std::vector<StmtPtr>& stmts, std::unordered_set<const Var*>& half_tiles,
                         std::vector<std::string>& full_width_vec_ops) {
  for (const auto& stmt : stmts) {
    CallPtr leaf;
    VarPtr def_var;
    if (auto assign = std::dynamic_pointer_cast<const AssignStmt>(stmt)) {
      leaf = AsCall(assign->value_);
      def_var = assign->var_;
    } else if (auto eval = std::dynamic_pointer_cast<const EvalStmt>(stmt)) {
      leaf = AsCall(eval->expr_);
    }
    if (leaf && leaf->op_) {
      // aiv_shard produces a HALF tile (the C->V boundary); seed the dataflow.
      if (IsOp(leaf, "tile.aiv_shard")) {
        if (def_var) half_tiles.insert(def_var.get());
        continue;
      }
      // aic_gather doubles HALF -> FULL (the V->C boundary back to cube); its
      // result leaves the half-width dataflow, so it is not tracked.
      if (IsOp(leaf, "tile.aic_gather")) {
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
            if (half_tiles.count(v.get()) != 0) {
              consumes_half = true;
              break;
            }
          }
        }
        if (consumes_half) {
          if (def_var) half_tiles.insert(def_var.get());  // stays in the half-width dataflow
        } else if (ClassifyCallAffinity(leaf) == CoreAffinity::VECTOR) {
          full_width_vec_ops.push_back(leaf->op_->name_);
        }
        continue;
      }
    }
    if (auto for_stmt = std::dynamic_pointer_cast<const ForStmt>(stmt)) {
      ScanRegionHalfWidth(transform_utils::FlattenToStmts(for_stmt->body_), half_tiles, full_width_vec_ops);
    } else if (auto if_stmt = std::dynamic_pointer_cast<const IfStmt>(stmt)) {
      ScanRegionHalfWidth(transform_utils::FlattenToStmts(if_stmt->then_body_), half_tiles,
                          full_width_vec_ops);
      if (if_stmt->else_body_.has_value()) {
        ScanRegionHalfWidth(transform_utils::FlattenToStmts(*if_stmt->else_body_), half_tiles,
                            full_width_vec_ops);
      }
    } else if (auto while_stmt = std::dynamic_pointer_cast<const WhileStmt>(stmt)) {
      ScanRegionHalfWidth(transform_utils::FlattenToStmts(while_stmt->body_), half_tiles, full_width_vec_ops);
    } else if (auto seq = std::dynamic_pointer_cast<const SeqStmts>(stmt)) {
      ScanRegionHalfWidth(seq->stmts_, half_tiles, full_width_vec_ops);
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
  std::unordered_set<const Var*> half_tiles;
  std::vector<std::string> full_width_vec_ops;
  ScanRegionHalfWidth(stmts, half_tiles, full_width_vec_ops);
  if (full_width_vec_ops.empty()) return;

  std::string ops;
  for (size_t i = 0; i < full_width_vec_ops.size(); ++i) {
    if (i != 0) ops += ", ";
    ops += full_width_vec_ops[i];
  }
  CHECK_SPAN(false, region_span)
      << "LowerAutoVectorSplit: a pl.split_aiv region mixes explicit "
         "tile.aiv_shard/tile.aic_gather boundary ops with plain full-width vector op(s) ["
      << ops
      << "] that operate outside the per-lane half-width dataflow. The explicit boundary keeps the "
         "region in half-width form, so these full-width ops would be left un-localized and both AIV "
         "lanes would compute the full tile. Fix it one of two ways: (1) author the whole region in "
         "half-width form — derive every vector op from the tile.aiv_shard result; or (2) remove the "
         "explicit tile.aiv_shard/tile.aic_gather and let the implicit affinity-gated path halve the "
         "region.";
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
      auto lowered = LowerStmts({stmt}, SplitMode::None, /*split_int=*/0, /*split_dim=*/0, ignored_tile_vars,
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

// Detect any live ``SplitAivScopeStmt`` in a body (O(N) single walk).
class HasSplitAivScopeFinder : public IRVisitor {
 public:
  bool found_ = false;

 protected:
  void VisitStmt_(const SplitAivScopeStmtPtr& op) override { found_ = true; }
};

bool BodyContainsSplitAivScope(const StmtPtr& body) {
  if (!body) return false;
  HasSplitAivScopeFinder finder;
  finder.VisitStmt(body);
  return finder.found_;
}

// Lower an InCore function that carries explicit ``SplitAivScopeStmt`` regions:
// halve only the vector compute inside each region (region-local), leave
// out-of-region compute full-width, drop each scope wrapper, and stamp
// ``split_aiv`` (idempotent — already bridged at OutlineIncoreScopes) plus
// ``split_aiv_region_validated`` (signals pass 21 to skip its func-mode check).
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
  auto [cloned_body, clone_map_unused] = DeepClone(new_body);
  (void)clone_map_unused;

  auto attrs = func->attrs_;
  attrs.erase(std::remove_if(attrs.begin(), attrs.end(),
                             [](const auto& kv) {
                               return kv.first == kSplitAivAttr || kv.first == kSplitAivRegionValidatedAttr;
                             }),
              attrs.end());
  attrs.emplace_back(kSplitAivAttr, true);
  attrs.emplace_back(kSplitAivRegionValidatedAttr, true);

  auto new_func = MutableCopy(func);
  new_func->body_ = cloned_body;
  new_func->attrs_ = std::move(attrs);
  return new_func;
}

std::vector<std::pair<std::string, std::any>> WithSplitAivAttrs(const FunctionPtr& func, SplitMode mode) {
  auto attrs = func->attrs_;
  attrs.erase(std::remove_if(attrs.begin(), attrs.end(),
                             [](const auto& kv) {
                               return kv.first == "split" || kv.first == kSplitAivAttr ||
                                      kv.first == kDualAivDispatchAttr;
                             }),
              attrs.end());
  attrs.emplace_back("split", static_cast<int>(mode));
  attrs.emplace_back(kSplitAivAttr, true);
  return attrs;
}

FunctionPtr LowerFunction(const FunctionPtr& func, SplitMode mode) {
  int split_int = static_cast<int>(mode);
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

  auto new_stmts = LowerStmts(injected.body_stmts, mode, split_int, split_dim, tile_vars,
                              injected.subblock_idx_expr, var_replacements, used_names);

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
  return func->HasAttr(kSplitAivAttr) && func->GetAttr<bool>(kSplitAivAttr, false);
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
      // AUTO whole-function path (unchanged): lower genuinely mixed
      // (cube<->vector) functions. Pure-vector pl.split functions have no boundary
      // to converge; ExpandMixedKernel strips their split, so marking them
      // split_aiv here would desync.
      if (is_incore && mode.has_value() && mode.value() != SplitMode::None &&
          !IsAlreadyExplicitSplitAiv(func) && IsMixedCubeVector(func)) {
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
