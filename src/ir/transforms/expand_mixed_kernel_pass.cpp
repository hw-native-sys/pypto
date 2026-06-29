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
#include <any>
#include <cstddef>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/backend/common/backend_handler.h"
#include "pypto/core/any_cast.h"
#include "pypto/core/logging.h"
#include "pypto/ir/core_affinity_kind.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/tile_view_semantics.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_context.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/core_affinity.h"
#include "pypto/ir/transforms/utils/core_side_ops.h"
#include "pypto/ir/transforms/utils/cross_core_pipe.h"
#include "pypto/ir/transforms/utils/dead_code_elimination.h"
#include "pypto/ir/transforms/utils/deep_clone_utils.h"
#include "pypto/ir/transforms/utils/loop_state_repair.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/transforms/utils/op_predicates.h"
#include "pypto/ir/transforms/utils/return_lineage_utils.h"
#include "pypto/ir/transforms/utils/scope_outline_utils.h"
#include "pypto/ir/transforms/utils/split_axis_utils.h"
#include "pypto/ir/transforms/utils/tpop_tfree_finalizer.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/transforms/utils/var_collectors.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

namespace {

constexpr const char* kDualAivDispatchAttr = "dual_aiv_dispatch";

using core_affinity::ClassifyCallAffinity;
using core_affinity::ClassifyMoveDirection;
using core_affinity::CombineAffinity;
using core_affinity::CoreAffinity;
using core_affinity::CoreSide;
using core_affinity::CVBoundaryMove;
using core_affinity::CVDirection;
using cross_core_pipe::BuildAutomaticPipeSetup;
using cross_core_pipe::PrependPipeSetup;
using loop_repair::BuildDefMap;
using loop_repair::FinalizeSplitCoreBody;
using loop_repair::MakeBody;
using tpop_tfree::FinalizeTpopTfrees;

// ============================================================================
// Flatten body helper
// ============================================================================

// Use the shared utility; local alias preserves call sites.
const auto& FlattenBody = transform_utils::FlattenToStmts;

// ============================================================================
// Explicit split-reshape op helpers (tile.aiv_shard / tile.aic_gather)
// ============================================================================
//
// These ops are folded into ExpandMixedKernel's cross-core boundary machinery:
// aiv_shard (cube -> vector, full -> half) becomes a CUBE_TO_VECTOR boundary,
// aic_gather (vector -> cube, half -> full) a VECTOR_TO_CUBE boundary. The
// direction is authoritative by op name — both ops keep the input's memory
// space (set_output_memory_inherit_input), so it cannot be derived from memory.

const std::string* GetSplitReshapeOpName(const CallPtr& call) {
  if (!call) return nullptr;
  auto op = std::dynamic_pointer_cast<const Op>(call->op_);
  if (!op) return nullptr;
  if (IsOp(op, "tile.aiv_shard") || IsOp(op, "tile.aic_gather")) return &op->name_;
  return nullptr;
}

bool IsSplitReshapeOp(const CallPtr& call) { return GetSplitReshapeOpName(call) != nullptr; }

CVDirection SplitReshapeDirection(const std::string& op_name) {
  // aiv_shard pushes from the cube lane into the vector lane; aic_gather is its
  // inverse (vector lane pushes into the cube lane). Route the literal through the
  // registry getter (GetOp throws on a typo) and match by canonical name, mirroring
  // the IsOp convention for sites that hold only the op name string.
  return (op_name == OpRegistry::GetInstance().GetOp("tile.aiv_shard")->name_) ? CVDirection::CUBE_TO_VECTOR
                                                                               : CVDirection::VECTOR_TO_CUBE;
}

/// Verify no explicit split-reshape op (tile.aiv_shard / tile.aic_gather)
/// survives in a finalized AIC/AIV body — they must all be folded into
/// cross-core tpush/tpop boundaries.
void AssertNoSplitReshapeSurvives(const std::vector<StmtPtr>& stmts, const std::string& func_name) {
  std::function<void(const std::vector<StmtPtr>&)> walk = [&](const std::vector<StmtPtr>& ss) {
    for (const auto& stmt : ss) {
      CallPtr call;
      if (auto assign = std::dynamic_pointer_cast<const AssignStmt>(stmt)) {
        call = std::dynamic_pointer_cast<const Call>(assign->value_);
      } else if (auto eval = std::dynamic_pointer_cast<const EvalStmt>(stmt)) {
        call = std::dynamic_pointer_cast<const Call>(eval->expr_);
      }
      if (call) {
        INTERNAL_CHECK_SPAN(!IsSplitReshapeOp(call), call->span_)
            << "Internal error: explicit split-reshape op survived ExpandMixedKernel in '" << func_name
            << "' — boundary folding failed to recognise it.";
      }
      if (auto for_stmt = std::dynamic_pointer_cast<const ForStmt>(stmt)) {
        walk(FlattenBody(for_stmt->body_));
      } else if (auto if_stmt = std::dynamic_pointer_cast<const IfStmt>(stmt)) {
        walk(FlattenBody(if_stmt->then_body_));
        if (if_stmt->else_body_.has_value()) walk(FlattenBody(if_stmt->else_body_.value()));
      } else if (auto while_stmt = std::dynamic_pointer_cast<const WhileStmt>(stmt)) {
        walk(FlattenBody(while_stmt->body_));
      }
    }
  };
  walk(stmts);
}

// True iff `stmt` binds `var = tile.get_subblock_idx()`. The auto pl.split path
// (LowerAutoVectorSplit) injects this binding at the top of the mixed body BEFORE
// ExpandMixedKernel, so it surfaces as the first AIV-lane statement here. The
// standalone split_aiv path (SplitVectorKernel) injects the identical binding at
// the very top of the AIV body AFTER the pipe setup already exists — i.e. ABOVE
// it. To keep the two paths byte-identical we must hoist this binding above the
// prepended pipe setup here too (see PrependPipeSetupKeepingSubblockIdx).
bool IsGetSubblockIdxBinding(const StmtPtr& stmt) {
  auto assign = std::dynamic_pointer_cast<const AssignStmt>(stmt);
  if (!assign) return false;
  auto call = std::dynamic_pointer_cast<const Call>(assign->value_);
  return IsOp(call, "tile.get_subblock_idx");
}

// Like PrependPipeSetup, but if `body` begins with a get_subblock_idx binding the
// pipe setup is inserted directly AFTER it, so the subblock-index binding stays at
// the absolute top of the lane body. This matches the standalone split_aiv
// (SplitVectorKernel) placement exactly, keeping both paths' .pto byte-identical.
std::vector<StmtPtr> PrependPipeSetupKeepingSubblockIdx(const std::vector<StmtPtr>& prologue,
                                                        const std::vector<StmtPtr>& body) {
  if (prologue.empty() || body.empty() || !IsGetSubblockIdxBinding(body.front())) {
    return PrependPipeSetup(prologue, body);
  }
  std::vector<StmtPtr> result;
  result.reserve(prologue.size() + body.size());
  result.push_back(body.front());
  result.insert(result.end(), prologue.begin(), prologue.end());
  result.insert(result.end(), body.begin() + 1, body.end());
  return result;
}

// ============================================================================
// Recursive Affinity Analysis
// ============================================================================

using TpopDefs = std::unordered_map<const Var*, CallPtr>;

/// Collect tpop (tile.tpop_from_aiv / tile.tpop_from_aic) result Vars and their defining Calls.
/// Used to avoid misclassifying tile.move from a tpop result as a cross-core boundary —
/// the data already crossed via the tpop, so the move is internal to the consuming core.
TpopDefs CollectTpopDefs(const std::vector<StmtPtr>& stmts) {
  TpopDefs result;
  std::function<void(const std::vector<StmtPtr>&)> walk = [&](const std::vector<StmtPtr>& ss) {
    for (const auto& stmt : ss) {
      if (auto assign = std::dynamic_pointer_cast<const AssignStmt>(stmt)) {
        if (auto call = std::dynamic_pointer_cast<const Call>(assign->value_)) {
          // Treat explicit split-reshape results like tpop results: a follow-on
          // tile.move that places an aic_gather result for the cube (Vec -> Left)
          // must NOT be re-detected as a second cross-core boundary — the data
          // already crosses via the gather's own tpush/tpop, so the move is just
          // internal cube placement on the consuming lane.
          if (op_predicates::IsTPop(call) || IsSplitReshapeOp(call)) {
            result[assign->var_.get()] = call;
          }
        }
      }
      if (auto for_stmt = std::dynamic_pointer_cast<const ForStmt>(stmt)) {
        walk(FlattenBody(for_stmt->body_));
      } else if (auto if_stmt = std::dynamic_pointer_cast<const IfStmt>(stmt)) {
        walk(FlattenBody(if_stmt->then_body_));
        if (if_stmt->else_body_.has_value()) {
          walk(FlattenBody(if_stmt->else_body_.value()));
        }
      } else if (auto while_stmt = std::dynamic_pointer_cast<const WhileStmt>(stmt)) {
        walk(FlattenBody(while_stmt->body_));
      }
    }
  };
  walk(stmts);
  return result;
}

// Forward declare
CoreAffinity AnalyzeStmtAffinity(const StmtPtr& stmt, std::unordered_map<const Stmt*, CoreAffinity>& stmt_map,
                                 std::unordered_map<const Var*, CoreAffinity>& var_affinity,
                                 const TpopDefs& tpop_defs);

CoreAffinity AnalyzeStmtsAffinity(const std::vector<StmtPtr>& stmts,
                                  std::unordered_map<const Stmt*, CoreAffinity>& stmt_map,
                                  std::unordered_map<const Var*, CoreAffinity>& var_affinity,
                                  const TpopDefs& tpop_defs = {}) {
  CoreAffinity combined = CoreAffinity::SHARED;
  for (const auto& stmt : stmts) {
    combined = CombineAffinity(combined, AnalyzeStmtAffinity(stmt, stmt_map, var_affinity, tpop_defs));
  }
  return combined;
}

CoreAffinity AnalyzeStmtAffinity(const StmtPtr& stmt, std::unordered_map<const Stmt*, CoreAffinity>& stmt_map,
                                 std::unordered_map<const Var*, CoreAffinity>& var_affinity,
                                 const TpopDefs& tpop_defs) {
  CoreAffinity result = CoreAffinity::SHARED;

  if (auto assign = std::dynamic_pointer_cast<const AssignStmt>(stmt)) {
    auto call = std::dynamic_pointer_cast<const Call>(assign->value_);
    if (call) {
      result = ClassifyCallAffinity(call);
      // tile.move from a tpop result is not a cross-core boundary: the data already
      // arrived via tpop, so the move is just internal data placement on the consuming
      // core. ClassifyCallAffinity returns MIXED for *any* C/V-crossing tile.move,
      // which for a leaf call can only be a boundary move; downgrade to the consuming
      // core's affinity when the source is a tpop result.
      if (result == CoreAffinity::MIXED && !call->args_.empty()) {
        if (auto src_var = std::dynamic_pointer_cast<const Var>(call->args_[0])) {
          if (tpop_defs.count(src_var.get()) > 0) {
            for (const auto& [key, value] : call->kwargs_) {
              if (key == "target_memory") {
                auto target = AnyCast<MemorySpace>(value, "target_memory");
                result = core_affinity::IsCubeMemorySpace(target) ? CoreAffinity::CUBE : CoreAffinity::VECTOR;
                break;
              }
            }
          }
        }
      }
    }
    var_affinity[assign->var_.get()] = result;
  } else if (auto eval = std::dynamic_pointer_cast<const EvalStmt>(stmt)) {
    auto call = std::dynamic_pointer_cast<const Call>(eval->expr_);
    if (call) result = ClassifyCallAffinity(call);
  } else if (auto for_stmt = std::dynamic_pointer_cast<const ForStmt>(stmt)) {
    result = AnalyzeStmtsAffinity(FlattenBody(for_stmt->body_), stmt_map, var_affinity, tpop_defs);
  } else if (auto if_stmt = std::dynamic_pointer_cast<const IfStmt>(stmt)) {
    result = AnalyzeStmtsAffinity(FlattenBody(if_stmt->then_body_), stmt_map, var_affinity, tpop_defs);
    const auto& else_body = if_stmt->else_body_;
    if (else_body.has_value()) {
      result = CombineAffinity(
          result, AnalyzeStmtsAffinity(FlattenBody(*else_body), stmt_map, var_affinity, tpop_defs));
    }
  } else if (auto while_stmt = std::dynamic_pointer_cast<const WhileStmt>(stmt)) {
    result = AnalyzeStmtsAffinity(FlattenBody(while_stmt->body_), stmt_map, var_affinity, tpop_defs);
  } else if (auto seq = std::dynamic_pointer_cast<const SeqStmts>(stmt)) {
    result = AnalyzeStmtsAffinity(seq->stmts_, stmt_map, var_affinity, tpop_defs);
  }

  stmt_map[stmt.get()] = result;
  return result;
}

// ============================================================================
// CV Boundary Move Collection
// ============================================================================

/// Collect all CV boundary tile.move statements recursively.
/// Also looks up whether the source tile is defined by a tpop call, so that kwargs
/// (e.g., split) can be propagated to the replacement tpop.
void CollectCVBoundaryMoves(const std::vector<StmtPtr>& stmts,
                            std::map<const Stmt*, CVBoundaryMove>& boundary_moves,
                            const TpopDefs& tpop_defs) {
  for (const auto& stmt : stmts) {
    if (auto assign = std::dynamic_pointer_cast<const AssignStmt>(stmt)) {
      auto call = std::dynamic_pointer_cast<const Call>(assign->value_);
      // Explicit split-reshape ops are op-driven boundaries. The direction is
      // authoritative by op name (both sides keep the input memory), the
      // half/full shape lives in the op result type, and the split axis is
      // stamped onto the generated tpush/tpop.
      if (const std::string* reshape_name = GetSplitReshapeOpName(call)) {
        INTERNAL_CHECK_SPAN(call->args_.size() == 1, call->span_)
            << "Internal error: " << *reshape_name << " must carry exactly one tile argument, got "
            << call->args_.size();
        boundary_moves[stmt.get()] = CVBoundaryMove{SplitReshapeDirection(*reshape_name),
                                                    assign->var_,
                                                    call->args_[0],
                                                    call->GetType(),
                                                    /*op_driven=*/true,
                                                    call->GetKwarg<int>("split", 0)};
      } else if (call) {
        auto dir = ClassifyMoveDirection(call);
        if (dir != CVDirection::NONE) {
          INTERNAL_CHECK_SPAN(!call->args_.empty(), call->span_)
              << "Internal error: tile.move must have at least one argument";
          // tile.move whose source comes from a tpop is not a genuine cross-core
          // boundary — the data already crossed via the tpop; this move is just
          // internal data placement on the consuming core. Skip recording so
          // BuildCoreBody's boundary-membership check treats it as a plain move.
          if (auto source_var = std::dynamic_pointer_cast<const Var>(call->args_[0])) {
            if (tpop_defs.count(source_var.get()) > 0) continue;
          }
          boundary_moves[stmt.get()] = CVBoundaryMove{dir, assign->var_, call->args_[0], call->GetType()};
        }
      }
    }

    // Recurse into compound statements
    if (auto for_stmt = std::dynamic_pointer_cast<const ForStmt>(stmt)) {
      CollectCVBoundaryMoves(FlattenBody(for_stmt->body_), boundary_moves, tpop_defs);
    } else if (auto if_stmt = std::dynamic_pointer_cast<const IfStmt>(stmt)) {
      CollectCVBoundaryMoves(FlattenBody(if_stmt->then_body_), boundary_moves, tpop_defs);
      const auto& else_body = if_stmt->else_body_;
      if (else_body.has_value()) {
        CollectCVBoundaryMoves(FlattenBody(*else_body), boundary_moves, tpop_defs);
      }
    } else if (auto while_stmt = std::dynamic_pointer_cast<const WhileStmt>(stmt)) {
      CollectCVBoundaryMoves(FlattenBody(while_stmt->body_), boundary_moves, tpop_defs);
    }
  }
}

// ============================================================================
// TPUSH / TPOP creation helpers
// ============================================================================

std::vector<std::pair<std::string, std::any>> MakeSplitKwargs(int split = 0) {
  return {{"split", std::any(split)}};
}

CallPtr CreateTpush(const std::string& op_name, const ExprPtr& tile, const Span& span, int split = 0) {
  return OpRegistry::GetInstance().Create(op_name, {tile}, MakeSplitKwargs(split), span);
}

CallPtr CreateTpop(const std::string& op_name, const TypePtr& result_type, const Span& span,
                   const std::vector<std::pair<std::string, std::any>>& kwargs = {}) {
  auto op = OpRegistry::GetInstance().GetOp(op_name);
  auto effective_kwargs = kwargs.empty() ? MakeSplitKwargs() : kwargs;
  return std::make_shared<Call>(op, std::vector<ExprPtr>{}, std::move(effective_kwargs), result_type, span);
}

CallPtr CreateMove(const ExprPtr& tile, MemorySpace target_memory, const TypePtr& result_type,
                   const Span& span) {
  auto op = OpRegistry::GetInstance().GetOp("tile.move");

  std::vector<std::pair<std::string, std::any>> kwargs{{"target_memory", std::any(target_memory)}};
  if (auto tt = std::dynamic_pointer_cast<const TileType>(result_type); tt) {
    const TileView eff = tile_view_semantics::GetEffectiveTileView(*tt);
    kwargs.emplace_back("blayout", std::any(eff.blayout));
    kwargs.emplace_back("slayout", std::any(eff.slayout));
  }
  return std::make_shared<Call>(op, std::vector<ExprPtr>{tile}, std::move(kwargs), result_type, span);
}

// ============================================================================
// Parameterized Core Body Builder (shared by AIC and AIV)
// ============================================================================

MemorySpace GetBoundaryTpopMemory(CoreSide side) {
  return (side == CoreSide::AIC) ? MemorySpace::Mat : MemorySpace::Vec;
}

TypePtr BuildBoundaryTpopType(CoreSide side, const TypePtr& original_type) {
  auto tt = std::dynamic_pointer_cast<const TileType>(original_type);
  INTERNAL_CHECK(tt) << "Boundary-generated tpop requires TileType result";
  return std::make_shared<TileType>(tt->shape_, tt->dtype_, std::nullopt, std::nullopt,
                                    GetBoundaryTpopMemory(side));
}

bool NeedsPostTpopMove(CoreSide side, const TileType& dest_type) {
  INTERNAL_CHECK(dest_type.memory_space_.has_value())
      << "Boundary move destination must have inferred memory_space before ExpandMixedKernel";
  return dest_type.memory_space_.value() != GetBoundaryTpopMemory(side);
}

std::string BuildBoundaryTpopName(CoreSide side, const std::string& dest_name) {
  return dest_name + ((side == CoreSide::AIC) ? "_mat" : "_vec");
}

/// Determine the fractal TileView for cross-core data transfer based on the
/// boundary move destination memory space.
///
/// Computes the cross-core transfer view by delegating to the active
/// BackendHandler. See BackendHandler::BuildCrossCoreTransferView and the
/// per-backend implementations in src/backend/910B/backend_910b_handler.cpp
/// and src/backend/950/backend_950_handler.cpp for the layout rules.
TileView BuildCrossCoreTransferView(MemorySpace dest_ms, const TileView& original_view) {
  return PassContext::Current()->GetBackendHandler()->BuildCrossCoreTransferView(dest_ms, original_view);
}

// ============================================================================
// GM-Mediated Cross-Lane Dependency Detection (issue #1433)
// ============================================================================
//
// A tile.store to a GM tensor on one lane (e.g. AIC) followed by a tile.load
// from the same GM tensor on the other lane (e.g. AIV) is a genuine cross-core
// data dependency, but neither op is a tile.move, so CollectCVBoundaryMoves
// never records it. Without a fence the two split kernels race on the shared
// GM region (see issue #1433). We detect such pairs and emit a pure-
// synchronisation tpush/tpop handshake: the data still flows through GM
// unchanged, while the consumer's tpop blocks until the producer's tpush
// (sequenced after the store) completes, establishing the missing
// happens-before edge. The popped tile is a fence token, freed immediately by
// FinalizeTpopTfrees. BuildAutomaticPipeSetup then injects the pipe setup just
// as it does for tile.move boundaries.

struct GmSyncPush {
  CoreSide producer_side;
  ExprPtr push_tile;
};

struct GmSyncPop {
  CoreSide consumer_side;
  TypePtr pop_tile_type;
  std::string token_name;
};

/// Resolve a tensor SSA Var to its origin (the parameter / create result it
/// derives from) by following tile.store result -> store destination chains.
/// A tile.store's result is a fresh SSA version of the destination tensor, so
/// the loaded version and the stored version may differ; resolving to a common
/// origin lets us match them.
const Var* ResolveTensorOrigin(const Var* var,
                               const std::unordered_map<const Var*, const Var*>& store_result_to_dest) {
  std::unordered_set<const Var*> seen;
  while (var) {
    auto it = store_result_to_dest.find(var);
    if (it == store_result_to_dest.end()) break;
    if (!seen.insert(var).second) break;  // guard against cycles
    var = it->second;
  }
  return var;
}

/// Detect GM-mediated cross-lane store/load pairs and populate the sync maps.
///
/// Conservative for deadlock-freedom: a handshake is emitted only when, for a
/// given GM tensor origin, there is exactly one producer-lane store and an
/// opposite-lane load that appears after it and either (a) lives in the *same
/// structural body* (identified by a per-body id assigned during the walk), or
/// (b) is nested in a loop/branch under the store's body. In case (a) the tpop
/// is emitted at the load; in case (b) it is hoisted before the store-body-level
/// compound that encloses the load. Either way the tpush and tpop execute the
/// same number of times (the store body's trip count), so the ring buffer cannot
/// deadlock. Pairs split across sibling/disjoint bodies are left untouched.
void CollectGmCrossLaneSyncs(const std::vector<StmtPtr>& stmts,
                             const std::unordered_map<const Stmt*, CoreAffinity>& stmt_map,
                             std::map<const Stmt*, GmSyncPush>& gm_sync_pushes,
                             std::map<const Stmt*, std::vector<GmSyncPop>>& gm_sync_pops) {
  // Pass 1: build tensor origin chains from tile.store results.
  std::unordered_map<const Var*, const Var*> store_result_to_dest;
  std::function<void(const std::vector<StmtPtr>&)> build_origins = [&](const std::vector<StmtPtr>& ss) {
    for (const auto& stmt : ss) {
      if (auto assign = std::dynamic_pointer_cast<const AssignStmt>(stmt)) {
        auto call = std::dynamic_pointer_cast<const Call>(assign->value_);
        if (IsOp(call, "tile.store") && call->args_.size() >= 3) {
          if (auto dest = std::dynamic_pointer_cast<const Var>(call->args_[2])) {
            store_result_to_dest[assign->var_.get()] = dest.get();
          }
        }
      }
      if (auto for_stmt = std::dynamic_pointer_cast<const ForStmt>(stmt)) {
        build_origins(FlattenBody(for_stmt->body_));
      } else if (auto if_stmt = std::dynamic_pointer_cast<const IfStmt>(stmt)) {
        build_origins(FlattenBody(if_stmt->then_body_));
        if (if_stmt->else_body_.has_value()) build_origins(FlattenBody(*if_stmt->else_body_));
      } else if (auto while_stmt = std::dynamic_pointer_cast<const WhileStmt>(stmt)) {
        build_origins(FlattenBody(while_stmt->body_));
      }
    }
  };
  build_origins(stmts);

  // Pass 2: collect cross-lane store/load records, tagged with a structural
  // body id and a program-order index.
  struct AccessRec {
    const Stmt* stmt;
    const Var* origin;
    CoreSide side;
    CallPtr call;
    int body_id;
    size_t order;
  };
  std::vector<AccessRec> stores;
  std::vector<AccessRec> loads;
  size_t order_counter = 0;
  int next_body_id = 1;  // 0 == function top level
  // body_id -> (enclosing compound stmt, parent body_id). Lets a consumer load
  // nested under the producer store's body be fenced by hoisting the tpop to the
  // loop/branch that sits in the store's body (see the C2V pairing below).
  std::unordered_map<int, std::pair<const Stmt*, int>> body_info;
  std::function<void(const std::vector<StmtPtr>&, int)> collect = [&](const std::vector<StmtPtr>& ss,
                                                                      int body_id) {
    for (const auto& stmt : ss) {
      auto aff_it = stmt_map.find(stmt.get());
      CoreAffinity aff = (aff_it != stmt_map.end()) ? aff_it->second : CoreAffinity::SHARED;
      if (auto assign = std::dynamic_pointer_cast<const AssignStmt>(stmt)) {
        auto call = std::dynamic_pointer_cast<const Call>(assign->value_);
        const bool single_lane = (aff == CoreAffinity::CUBE || aff == CoreAffinity::VECTOR);
        if (single_lane && IsOp(call, "tile.store") && call->args_.size() >= 3) {
          if (auto dest = std::dynamic_pointer_cast<const Var>(call->args_[2])) {
            const CoreSide side = (aff == CoreAffinity::CUBE) ? CoreSide::AIC : CoreSide::AIV;
            stores.push_back({stmt.get(), ResolveTensorOrigin(dest.get(), store_result_to_dest), side, call,
                              body_id, order_counter++});
          }
        } else if (single_lane && IsOp(call, "tile.load") && !call->args_.empty()) {
          if (auto src = std::dynamic_pointer_cast<const Var>(call->args_[0])) {
            const CoreSide side = (aff == CoreAffinity::CUBE) ? CoreSide::AIC : CoreSide::AIV;
            loads.push_back({stmt.get(), ResolveTensorOrigin(src.get(), store_result_to_dest), side, call,
                             body_id, order_counter++});
          }
        }
      }
      if (auto for_stmt = std::dynamic_pointer_cast<const ForStmt>(stmt)) {
        int child = next_body_id++;
        body_info[child] = {stmt.get(), body_id};
        collect(FlattenBody(for_stmt->body_), child);
      } else if (auto if_stmt = std::dynamic_pointer_cast<const IfStmt>(stmt)) {
        int then_id = next_body_id++;
        body_info[then_id] = {stmt.get(), body_id};
        collect(FlattenBody(if_stmt->then_body_), then_id);
        if (if_stmt->else_body_.has_value()) {
          int else_id = next_body_id++;
          body_info[else_id] = {stmt.get(), body_id};
          collect(FlattenBody(*if_stmt->else_body_), else_id);
        }
      } else if (auto while_stmt = std::dynamic_pointer_cast<const WhileStmt>(stmt)) {
        int child = next_body_id++;
        body_info[child] = {stmt.get(), body_id};
        collect(FlattenBody(while_stmt->body_), child);
      }
    }
  };
  collect(stmts, 0);

  // Pass 3: pair a unique producer store with the first matching consumer load.
  // Bucket stores and loads by origin first so the matching stays linear in the
  // number of GM accesses (rather than rescanning all stores/loads per store).
  // Track origins in store-traversal insertion order so iteration is
  // deterministic without iterating the pointer-keyed unordered_map (which
  // yields pointer-order traversal and unstable downstream assignments).
  std::unordered_map<const Var*, std::vector<const AccessRec*>> stores_by_origin;
  std::unordered_map<const Var*, std::vector<const AccessRec*>> loads_by_origin;
  std::vector<const Var*> ordered_origins;
  std::unordered_set<const Var*> seen_origins;
  for (const auto& store : stores) {
    if (!store.origin) continue;
    if (seen_origins.insert(store.origin).second) ordered_origins.push_back(store.origin);
    stores_by_origin[store.origin].push_back(&store);
  }
  for (const auto& load : loads) {
    if (load.origin) loads_by_origin[load.origin].push_back(&load);
  }

  // Is `anc` equal to, or an ancestor body of, `desc`?
  auto is_ancestor_body = [&](int anc, int desc) -> bool {
    int cur = desc;
    std::unordered_set<int> guard;
    while (cur != anc) {
      auto it = body_info.find(cur);
      if (it == body_info.end()) return false;  // reached the top without hitting anc
      if (!guard.insert(cur).second) return false;
      cur = it->second.second;  // parent body
    }
    return true;
  };
  // The compound stmt living in body `anc` that encloses body `desc`, so the
  // consumer fence can be hoisted before it. Returns nullptr if `desc` is not
  // nested under `anc`.
  auto enclosing_stmt_in_body = [&](int anc, int desc) -> const Stmt* {
    int cur = desc;
    std::unordered_set<int> guard;
    while (true) {
      auto it = body_info.find(cur);
      if (it == body_info.end()) return nullptr;
      if (!guard.insert(cur).second) return nullptr;
      if (it->second.second == anc) return it->second.first;  // enclosing stmt at anc level
      cur = it->second.second;
    }
  };

  for (const Var* origin : ordered_origins) {
    const auto& origin_stores = stores_by_origin.at(origin);
    if (origin_stores.size() != 1) continue;  // require a unique producer store
    const AccessRec& store = *origin_stores.front();

    // Restrict to the cube-store -> vector-load (C2V) direction. The producer
    // store is on AIC, the consumer load on AIV: the AIC tpush sends the source
    // tile raw, exactly as the normal boundary C2V push does on both backends
    // (no fractal adaptation), and the AIV tpop lands in Vec where the transfer
    // view is "preserve original". The reverse V2C direction would need the V->C
    // fractal adaptation (tile.move to NZ/ZN before tpush_to_aic, fractal-typed
    // tpop_from_aiv) that the boundary path applies; emitting a raw-tile sync
    // there would violate the cross-core transport contract, so we leave V2C
    // GM exchanges unfenced rather than emit unadapted transport.
    if (store.side != CoreSide::AIC) continue;

    const AccessRec* chosen = nullptr;
    auto loads_it = loads_by_origin.find(origin);
    if (loads_it == loads_by_origin.end()) continue;
    for (const AccessRec* load : loads_it->second) {
      if (load->side != CoreSide::AIV) continue;  // C2V only: consumer on the vector lane
      // The consumer load must share the store's body (a 1:1 fence) or be nested
      // in a loop/branch under it. In the nested case the tpop is hoisted to the
      // store-body-level compound below, so tpush and the hoisted tpop still run
      // the same number of times. Loads in a sibling/disjoint body are left
      // unfenced — their trip count vs. the store's is unproven (deadlock risk).
      if (!is_ancestor_body(store.body_id, load->body_id)) continue;
      if (load->order <= store.order) continue;  // load must follow the store
      if (chosen == nullptr || load->order < chosen->order) chosen = load;
    }
    if (chosen == nullptr) continue;

    // Where the consumer fence tpop is emitted: at the load itself when it
    // shares the store's body, otherwise hoisted before the store-body-level
    // compound that encloses it (so it runs once per producer store, not once
    // per loop iteration).
    const Stmt* pop_stmt = chosen->stmt;
    if (chosen->body_id != store.body_id) {
      pop_stmt = enclosing_stmt_in_body(store.body_id, chosen->body_id);
      if (pop_stmt == nullptr) continue;  // defensive: not actually nested under the store
    }

    auto src_tile_type = std::dynamic_pointer_cast<const TileType>(store.call->args_[0]->GetType());
    if (!src_tile_type) continue;  // need a TileType for cross-core slot sizing

    const CoreSide consumer_side = chosen->side;
    const MemorySpace consumer_mem = GetBoundaryTpopMemory(consumer_side);
    auto pop_tile_type = std::make_shared<TileType>(src_tile_type->shape_, src_tile_type->dtype_,
                                                    std::nullopt, std::nullopt, consumer_mem);
    std::string token_name = origin->name_hint_ + "_gm_sync";

    gm_sync_pushes[store.stmt] = GmSyncPush{store.side, store.call->args_[0]};
    // A vector per stmt: several GM dependencies can hoist to the same enclosing
    // stmt (e.g. two scratch loads in one outer loop) — each needs its own fence.
    gm_sync_pops[pop_stmt].push_back(GmSyncPop{consumer_side, pop_tile_type, std::move(token_name)});
  }
}

/// Build the body for one core side (AIC or AIV), filtering statements by affinity
/// and replacing CV boundary moves with TPUSH/TPOP ops.
/// tpop_var_remap collects dest_var and (when source_tile is a Var) source_tile pointers
/// -> clean-typed new_var mappings, so downstream references to either the pre-move or
/// post-move variable can be updated via pointer-based substitution.
/// superseded_tpop_vars tracks source Vars whose defining tpop is superseded by a
/// boundary-generated tpop, so the original tpop can be eliminated.
std::vector<StmtPtr> BuildCoreBody(CoreSide side, const std::vector<StmtPtr>& stmts,
                                   const std::unordered_map<const Stmt*, CoreAffinity>& stmt_map,
                                   const std::map<const Stmt*, CVBoundaryMove>& boundary_moves,
                                   std::unordered_map<const Var*, VarPtr>& tpop_var_remap,
                                   std::unordered_set<const Var*>& superseded_tpop_vars,
                                   const std::map<const Stmt*, GmSyncPush>& gm_sync_pushes,
                                   const std::map<const Stmt*, std::vector<GmSyncPop>>& gm_sync_pops) {
  const auto* handler = PassContext::Current()->GetBackendHandler();
  // AIC keeps CUBE, skips VECTOR; AIV keeps VECTOR, skips CUBE
  CoreAffinity keep_affinity = (side == CoreSide::AIC) ? CoreAffinity::CUBE : CoreAffinity::VECTOR;
  CoreAffinity skip_affinity = (side == CoreSide::AIC) ? CoreAffinity::VECTOR : CoreAffinity::CUBE;

  // For boundary moves: the "push" side sends data, the "pop" side receives it.
  // AIC: C→V = push to AIV, V→C = pop from AIV
  // AIV: C→V = pop from AIC, V→C = push to AIC
  std::string push_op = core_side_ops::TPushOp(side);
  std::string pop_op = core_side_ops::TPopOp(side);
  CVDirection push_direction = core_side_ops::PushDirection(side);

  std::vector<StmtPtr> result;

  for (const auto& stmt : stmts) {
    auto it = stmt_map.find(stmt.get());
    CoreAffinity affinity = (it != stmt_map.end()) ? it->second : CoreAffinity::SHARED;

    // Leaf boundary move — emit tpush/tpop. boundary_moves is the authoritative
    // signal for "this stmt needs cross-core split"; the stmt's CoreAffinity is
    // MIXED here, but we handle it specially before the generic MIXED arm below.
    auto bm_it = boundary_moves.find(stmt.get());
    if (bm_it != boundary_moves.end()) {
      {
        const auto& bm = bm_it->second;
        // The cross-core transfer memory FOR THIS SIDE: AIC drains into Mat,
        // AIV into Vec. Op-driven boundaries (aiv_shard / aic_gather) keep the
        // input memory on both sides, so the fractal view and tpop type must
        // key off this transfer memory rather than the op's inherited result
        // memory (which is Vec on both lanes).
        const MemorySpace xfer_ms = GetBoundaryTpopMemory(side);
        const int op_split = bm.op_driven ? bm.split : 0;
        if (bm.direction == push_direction) {
          ExprPtr push_source = bm.source_tile;
          // AIV V->C push: insert tile.move (tmov) to adapt the source into
          // the required fractal layout before tpush.
          // On Ascend950: Left -> NZ, Right -> ZN.
          // On Ascend910B: don't need to adapt layout! push/pop will be ub -> gm -> mat, ub -> gm can
          // directly use nd
          if (side == CoreSide::AIV && handler->RequiresVtoCFractalAdapt()) {
            auto src_type = std::dynamic_pointer_cast<const TileType>(bm.source_tile->GetType());
            INTERNAL_CHECK_SPAN(src_type, stmt->span_) << "V->C tpush source must have TileType";
            // For op-driven boundaries the cube-side transfer memory is Mat
            // (aic_gather gathers into Mat); the fractal adapter keys off it and
            // off the half source's own view. For tile.move boundaries the
            // destination is the cube tile itself (Mat/Left/Right), so key off
            // its memory and view as before.
            TileView fractal_view;
            if (bm.op_driven) {
              fractal_view = BuildCrossCoreTransferView(
                  GetBoundaryTpopMemory(CoreSide::AIC),  // cube-side transfer memory (Mat)
                  tile_view_semantics::GetEffectiveTileView(*src_type));
            } else {
              auto push_dest_type = std::dynamic_pointer_cast<const TileType>(bm.dest_var->GetType());
              INTERNAL_CHECK_SPAN(push_dest_type && push_dest_type->memory_space_.has_value(), stmt->span_)
                  << "Boundary move destination must have TileType and MemSpace";
              fractal_view = BuildCrossCoreTransferView(
                  push_dest_type->memory_space_.value(),  // NOLINT(bugprone-unchecked-optional-access)
                  tile_view_semantics::GetEffectiveTileView(*push_dest_type));
            }

            auto tmov_type = std::make_shared<TileType>(src_type->shape_, src_type->dtype_, std::nullopt,
                                                        fractal_view, MemorySpace::Vec);
            std::string src_name = "tile";
            if (auto sv = std::dynamic_pointer_cast<const Var>(bm.source_tile)) {
              src_name = sv->name_hint_;
            }
            bool is_nz = (fractal_view.blayout == TileLayout::col_major);
            auto tmov_var = std::make_shared<Var>(src_name + (is_nz ? "_nz" : "_zn"), tmov_type, stmt->span_);
            auto tmov_call = CreateMove(bm.source_tile, MemorySpace::Vec, tmov_type, stmt->span_);
            result.push_back(std::make_shared<AssignStmt>(tmov_var, tmov_call, stmt->span_));
            push_source = tmov_var;
          }
          result.push_back(std::make_shared<EvalStmt>(
              CreateTpush(push_op, push_source, stmt->span_, op_split), stmt->span_));
        } else {
          // Op-driven pop: the half/full shape comes from the op result type and
          // the memory from this side's transfer memory; the explicit follow-on
          // tile.move owns any cube placement, so no synthesized post-tpop move.
          // tile.move boundaries instead carry their destination memory on
          // dest_var and may need a post-tpop move to that memory.
          TypePtr shape_source = bm.op_driven ? bm.result_type : bm.dest_var->GetType();
          auto shape_tt = std::dynamic_pointer_cast<const TileType>(shape_source);
          INTERNAL_CHECK_SPAN(shape_tt, stmt->span_) << "Boundary pop requires a TileType result/destination";
          MemorySpace view_ms;
          bool needs_post_move = false;
          if (bm.op_driven) {
            view_ms = xfer_ms;
          } else {
            INTERNAL_CHECK_SPAN(shape_tt->memory_space_.has_value(), stmt->span_)
                << "Boundary move destination must have TileType and MemSpace";
            view_ms = shape_tt->memory_space_.value();  // NOLINT(bugprone-unchecked-optional-access)
            needs_post_move = NeedsPostTpopMove(side, *shape_tt);
          }
          auto tpop_type = BuildBoundaryTpopType(side, shape_source);
          // Consumer-side transfer view. For op-driven boundaries the cross-core
          // data lands in a FRESH transfer tile of this side's memory (Vec on
          // AIV, Mat on AIC) — exactly like a plain move-boundary destination —
          // so the layout must be the implicit view of the transfer memory, NOT
          // the split-reshape op result's input-inherited (cube) layout. This is
          // what SplitVectorKernel's RebuildTpopWithHalvedShape preserves on the
          // OFF path; keying off the op result here would print a divergent
          // col_major/row_major Vec tpop. The op deducer's reshaped valid_shape
          // is kept so per-lane valid extents survive.
          TileView boundary_view;
          if (bm.op_driven) {
            boundary_view = tile_view_semantics::GetImplicitTileView(shape_tt->shape_, view_ms);
            boundary_view.valid_shape = tile_view_semantics::GetEffectiveTileView(*shape_tt).valid_shape;
          } else {
            boundary_view = tile_view_semantics::GetEffectiveTileView(*shape_tt);
          }
          auto fractal_view = BuildCrossCoreTransferView(view_ms, boundary_view);
          std::string tpop_name = needs_post_move ? BuildBoundaryTpopName(side, bm.dest_var->name_hint_)
                                                  : bm.dest_var->name_hint_;
          auto tt = std::dynamic_pointer_cast<const TileType>(tpop_type);
          auto tpop_result_type = std::make_shared<TileType>(tt->shape_, tt->dtype_, std::nullopt,
                                                             fractal_view, tt->memory_space_);
          auto tpop_var = std::make_shared<Var>(tpop_name, tpop_result_type, stmt->span_);
          if (!needs_post_move) {
            tpop_var_remap[bm.dest_var.get()] = tpop_var;
          }
          // Op-driven boundaries reuse the op's source on the producer lane only;
          // do NOT alias the source Var to the tpop result on the consumer lane
          // (the source is the cross-lane half/full, not a renamed copy).
          if (!bm.op_driven) {
            if (auto source_var = std::dynamic_pointer_cast<const Var>(bm.source_tile)) {
              tpop_var_remap[source_var.get()] = tpop_var;
            }
          }
          tpop_var_remap[tpop_var.get()] = tpop_var;
          // tile.move boundary tpops carry no split kwarg here (assigned later by
          // SplitVectorKernel); op-driven tpops stamp the op's split now.
          auto pop_kwargs =
              bm.op_driven ? MakeSplitKwargs(op_split) : std::vector<std::pair<std::string, std::any>>{};
          result.push_back(std::make_shared<AssignStmt>(
              tpop_var, CreateTpop(pop_op, tpop_result_type, stmt->span_, pop_kwargs), stmt->span_));
          if (needs_post_move) {
            auto target_memory = shape_tt->memory_space_;
            INTERNAL_CHECK_SPAN(target_memory.has_value(), stmt->span_)
                << "Boundary move destination must have memory_space before post-tpop move emission";
            result.push_back(std::make_shared<AssignStmt>(
                bm.dest_var, CreateMove(tpop_var, *target_memory, bm.dest_var->GetType(), stmt->span_),
                stmt->span_));
          }
        }
        continue;
      }
    }

    if (affinity == skip_affinity) continue;

    // Skip original tpop statements whose results are superseded by boundary-generated tpops
    if (auto assign = std::dynamic_pointer_cast<const AssignStmt>(stmt)) {
      if (superseded_tpop_vars.count(assign->var_.get()) > 0) continue;
    }

    // GM cross-lane sync (issue #1433): on the consumer lane, emit a fence tpop
    // just before the keyed stmt. That stmt is the load itself when producer and
    // consumer share a body, or the loop/branch enclosing the load (so the fence
    // is hoisted out of the consumer loop and runs once per producer store). The
    // popped tile is unused; FinalizeTpopTfrees frees it right after. Several GM
    // dependencies may key on the same enclosing stmt, so emit every fence in the
    // vector (insertion order is preserved for a deterministic dump).
    if (auto pop_it = gm_sync_pops.find(stmt.get()); pop_it != gm_sync_pops.end()) {
      for (const auto& info : pop_it->second) {
        if (side != info.consumer_side) continue;
        auto pop_var = std::make_shared<Var>(info.token_name, info.pop_tile_type, stmt->span_);
        result.push_back(std::make_shared<AssignStmt>(
            pop_var, CreateTpop(pop_op, info.pop_tile_type, stmt->span_, /*kwargs=*/{}), stmt->span_));
      }
    }

    if (affinity == keep_affinity || affinity == CoreAffinity::SHARED) {
      result.push_back(stmt);
      // GM cross-lane sync: on the producer lane, emit the matching tpush just
      // after the store that writes the shared GM tensor.
      if (auto push_it = gm_sync_pushes.find(stmt.get());
          push_it != gm_sync_pushes.end() && side == push_it->second.producer_side) {
        result.push_back(std::make_shared<EvalStmt>(
            CreateTpush(push_op, push_it->second.push_tile, stmt->span_), stmt->span_));
      }
    } else if (affinity == CoreAffinity::MIXED) {
      // Recurse into compound statements, building pruned copies
      if (auto for_stmt = std::dynamic_pointer_cast<const ForStmt>(stmt)) {
        auto new_body = BuildCoreBody(side, FlattenBody(for_stmt->body_), stmt_map, boundary_moves,
                                      tpop_var_remap, superseded_tpop_vars, gm_sync_pushes, gm_sync_pops);
        auto new_for = MutableCopy(for_stmt);
        new_for->body_ = MakeBody(new_body, for_stmt->span_);
        result.push_back(new_for);
      } else if (auto if_stmt = std::dynamic_pointer_cast<const IfStmt>(stmt)) {
        auto new_then = BuildCoreBody(side, FlattenBody(if_stmt->then_body_), stmt_map, boundary_moves,
                                      tpop_var_remap, superseded_tpop_vars, gm_sync_pushes, gm_sync_pops);
        std::optional<StmtPtr> new_else;
        const auto& else_body = if_stmt->else_body_;
        if (else_body.has_value()) {
          auto new_else_stmts =
              BuildCoreBody(side, FlattenBody(*else_body), stmt_map, boundary_moves, tpop_var_remap,
                            superseded_tpop_vars, gm_sync_pushes, gm_sync_pops);
          new_else = MakeBody(new_else_stmts, if_stmt->span_);
        }
        auto new_if = MutableCopy(if_stmt);
        new_if->then_body_ = MakeBody(new_then, if_stmt->span_);
        new_if->else_body_ = new_else;
        result.push_back(new_if);
      } else if (auto while_stmt = std::dynamic_pointer_cast<const WhileStmt>(stmt)) {
        auto new_body = BuildCoreBody(side, FlattenBody(while_stmt->body_), stmt_map, boundary_moves,
                                      tpop_var_remap, superseded_tpop_vars, gm_sync_pushes, gm_sync_pops);
        auto new_while = MutableCopy(while_stmt);
        new_while->body_ = MakeBody(new_body, while_stmt->span_);
        result.push_back(new_while);
      } else {
        result.push_back(stmt);  // Unknown compound, include as-is
      }
    }
  }

  return result;
}

// ============================================================================
// Main Expansion Logic
// ============================================================================

struct ExpandedKernel {
  FunctionPtr aic_func;
  FunctionPtr aiv_func;
  std::optional<FunctionPtr> group_func;  // nullopt when existing Group caller will be rewritten
};

ExpandedKernel ExpandMixedFunction(const FunctionPtr& func, bool create_group = true) {
  // A tile.transpose that swaps the split axis cannot be split correctly:
  // SplitVectorKernel halves the original split axis, but the transpose moves
  // that data to the other dimension, mis-typing the result. Reject the split
  // request with an actionable error rather than silently miscompiling — the
  // user controls this perf decision (drop the split, or remove the transpose).
  //
  // Explicit ``pl.split_aiv`` regions are validated per-region by
  // LowerAutoVectorSplit (pass 21), where each region's mode is unambiguous; skip
  // the single-func-mode check for them. A multi-mode function carries no single
  // ``func->GetSplitMode()`` and this whole-function check would mis-check the
  // other region's axis (critique #2).
  if (!func->HasAttr("split_aiv_region_validated")) {
    if (auto mode = func->GetSplitMode(); mode.has_value() && *mode != SplitMode::None) {
      int split_dim = (*mode == SplitMode::UpDown) ? 0 : 1;
      auto hazard = split_axis::FindTransposeSplitHazard(func->body_, split_dim);
      if (hazard.call) {
        const char* mode_name = (*mode == SplitMode::UpDown) ? "UP_DOWN" : "LEFT_RIGHT";
        std::string where =
            hazard.result_name.empty() ? std::string() : " (result '" + hazard.result_name + "')";
        CHECK_SPAN(false, hazard.call->span_)
            << "ExpandMixedKernel: kernel '" << func->name_ << "' requests pl.split(" << mode_name
            << ") but contains a tile.transpose" << where << " that swaps the split axis (dim " << split_dim
            << "). SplitVectorKernel halves the split axis while the transpose moves that data to the "
               "other dimension, so the split cannot be applied correctly. Fix it one of two ways: "
               "(1) drop the split — set attrs={\"split\": pl.SplitMode.NONE} (or remove the pl.split "
               "optimization); or (2) eliminate this transpose, e.g. replace a transpose-then-row-index "
               "with a direct column slice such as pre[:, h:h+1].";
      }
    }
  }

  const bool needs_dual_aiv_dispatch =
      PassContext::Current()->GetBackendHandler()->RequiresNoSplitDualAivDispatch() &&
      (!func->GetSplitMode().has_value() || *func->GetSplitMode() == SplitMode::None);

  auto stmts = FlattenBody(func->body_);

  // Pre-scan for tpop result vars and their defining calls. Needed for
  // (1) affinity analysis — avoid misclassifying tile.move from tpop results
  // as cross-core boundaries, and (2) boundary-move kwarg propagation — carry
  // split=1 etc. from the original tpop onto its boundary-generated replacement.
  auto tpop_defs = CollectTpopDefs(stmts);

  // Recursive affinity analysis (descends into ForStmt/IfStmt/WhileStmt)
  std::unordered_map<const Stmt*, CoreAffinity> stmt_map;
  std::unordered_map<const Var*, CoreAffinity> var_affinity;
  AnalyzeStmtsAffinity(stmts, stmt_map, var_affinity, tpop_defs);

  std::map<const Stmt*, CVBoundaryMove> boundary_moves;
  CollectCVBoundaryMoves(stmts, boundary_moves, tpop_defs);

  // Detect GM-mediated cross-lane store/load dependencies (issue #1433) that
  // CollectCVBoundaryMoves misses, and schedule a tpush/tpop fence for each.
  std::map<const Stmt*, GmSyncPush> gm_sync_pushes;
  std::map<const Stmt*, std::vector<GmSyncPop>> gm_sync_pops;
  CollectGmCrossLaneSyncs(stmts, stmt_map, gm_sync_pushes, gm_sync_pops);

  // Build definition map from original body for init value fixup (#533)
  std::unordered_map<const Var*, StmtPtr> original_def_map;
  BuildDefMap(stmts, original_def_map);

  // Boundary-generated tpops never reuse the source-tpop Var (CollectCVBoundaryMoves
  // skips moves whose source comes from a tpop), so no original-tpop statements need
  // to be suppressed. Keep the set empty so BuildCoreBody's tpop-superseded check is
  // a no-op in this code path; the structure is preserved in case future work
  // reintroduces source-tpop reuse.
  std::unordered_set<const Var*> superseded_tpop_vars;

  // Build AIC body (recursive — handles MIXED compound stmts)
  std::unordered_map<const Var*, VarPtr> aic_tpop_remap;
  auto aic_stmts = BuildCoreBody(CoreSide::AIC, stmts, stmt_map, boundary_moves, aic_tpop_remap,
                                 superseded_tpop_vars, gm_sync_pushes, gm_sync_pops);

  // Remove ReturnStmt from AIC (AIC doesn't return values)
  std::vector<StmtPtr> aic_stmts_no_return;
  for (const auto& s : aic_stmts) {
    if (!std::dynamic_pointer_cast<const ReturnStmt>(s)) {
      aic_stmts_no_return.push_back(s);
    }
  }
  // tpop_var_remap keys are originally-defined Vars whose producers got
  // replaced on this side by a boundary tpop. A later DeepClone seeded with
  // `tpop_var_remap` substitutes them with valid in-scope Vars, so they are
  // not truly dangling. Pass the keys to FinalizeSplitCoreBody so its
  // StripDanglingIfReturnVars phase doesn't drop IfStmt return_vars whose
  // yields reference those Vars.
  auto remap_keys = [](const std::unordered_map<const Var*, VarPtr>& m) {
    std::unordered_set<const Var*> keys;
    keys.reserve(m.size());
    for (const auto& kv : m) keys.insert(kv.first);
    return keys;
  };
  auto aic_final = FinalizeTpopTfrees(
      FinalizeSplitCoreBody(aic_stmts_no_return, original_def_map, remap_keys(aic_tpop_remap)), CoreSide::AIC,
      aic_tpop_remap);

  // Build AIV body (recursive — handles MIXED compound stmts)
  std::unordered_map<const Var*, VarPtr> aiv_tpop_remap;
  auto aiv_stmts = BuildCoreBody(CoreSide::AIV, stmts, stmt_map, boundary_moves, aiv_tpop_remap,
                                 superseded_tpop_vars, gm_sync_pushes, gm_sync_pops);
  auto aiv_final =
      FinalizeTpopTfrees(FinalizeSplitCoreBody(aiv_stmts, original_def_map, remap_keys(aiv_tpop_remap)),
                         CoreSide::AIV, aiv_tpop_remap);

  // Every explicit split-reshape op must have been folded into a cross-core
  // tpush/tpop boundary on both lanes. A survivor means the boundary machinery
  // failed to recognise it — fail loudly here, not later in codegen.
  AssertNoSplitReshapeSurvives(aic_final, func->name_ + "_aic");
  AssertNoSplitReshapeSurvives(aiv_final, func->name_ + "_aiv");

  const std::string aic_name = func->name_ + "_aic";
  const std::string aiv_name = func->name_ + "_aiv";
  // Cross-core ring depth from pl.split(mode, slot_num=N), propagated as a
  // function attr by the outliner. nullopt keeps the hardcoded 8/4 default.
  std::optional<int> slot_num_override =
      func->HasAttr("slot_num") ? std::optional<int>(func->GetAttr<int>("slot_num", 0)) : std::nullopt;
  auto automatic_pipe_setup = BuildAutomaticPipeSetup(func->name_, aic_name, aiv_name, aic_final, aiv_final,
                                                      slot_num_override, func->span_);
  aic_final = PrependPipeSetup(automatic_pipe_setup.aic_stmts, aic_final);
  // Keep a leading get_subblock_idx binding (injected by the auto pl.split
  // LowerAutoVectorSplit path) above the pipe setup, matching the standalone
  // split_aiv SplitVectorKernel placement for byte-identical output.
  aiv_final = PrependPipeSetupKeepingSubblockIdx(automatic_pipe_setup.aiv_stmts, aiv_final);

  // Helper to create fresh params and build a DeepClone var_map
  auto make_param_map = [&]() {
    std::unordered_map<const Var*, ExprPtr> param_map;
    std::vector<VarPtr> fresh_params;
    for (const auto& var : func->params_) {
      auto fresh = std::make_shared<Var>(var->name_hint_, var->GetType(), func->span_);
      fresh_params.push_back(fresh);
      param_map[var.get()] = fresh;
    }
    return std::make_pair(fresh_params, param_map);
  };

  // Helper to pre-seed tpop var remappings into a DeepClone map.
  // Maps both the original dest_var and the clean_var itself to prevent
  // DeepClone from creating yet another fresh copy at the AssignStmt DefField.
  auto seed_tpop_remap = [](std::unordered_map<const Var*, ExprPtr>& clone_map,
                            const std::unordered_map<const Var*, VarPtr>& tpop_remap) {
    for (const auto& [orig_ptr, tpop_var] : tpop_remap) {
      clone_map[orig_ptr] = tpop_var;
      // Also seed the tpop_var itself to prevent DeepClone from re-cloning it
      // when it encounters it as an AssignStmt LHS (DefField).
      clone_map[tpop_var.get()] = tpop_var;
    }
  };

  // Create AIC function with deep clone (fresh Vars for all params and locals)
  auto [aic_params, aic_map] = make_param_map();
  seed_tpop_remap(aic_map, aic_tpop_remap);
  auto [aic_cloned_body, aic_clone_map_unused] = DeepClone(MakeBody(aic_final, func->span_), aic_map);
  (void)aic_clone_map_unused;
  auto aic_func = std::make_shared<Function>(aic_name, aic_params, func->param_directions_,
                                             std::vector<TypePtr>{}, aic_cloned_body, func->span_,
                                             FunctionType::AIC, std::nullopt, std::nullopt, func->attrs_);

  // Create AIV function with deep clone (fresh Vars for all params and locals,
  // ensuring no shared Var pointers with AIC for structural equality)
  auto [aiv_params, aiv_map] = make_param_map();
  seed_tpop_remap(aiv_map, aiv_tpop_remap);

  // Map dangling tile.store result vars to the store destination's fresh param.
  // When a tile.store is on the AIC side, its result var is stripped from the AIV body,
  // but the var may still be referenced (e.g., in a ReturnStmt). Remap those dangling
  // references to the fresh parameter corresponding to the store's output tensor.
  {
    // Collect all vars defined in the AIV body
    outline_utils::VarDefUseCollector aiv_def_collector;
    auto aiv_body_stmt = MakeBody(aiv_final, func->span_);
    aiv_def_collector.VisitStmt(aiv_body_stmt);

    // Scan original body recursively for tile.store AssignStmts
    std::vector<std::shared_ptr<const AssignStmt>> original_assigns;
    dce::CollectAllAssignStmts(stmts, original_assigns);
    for (const auto& assign : original_assigns) {
      auto call = std::dynamic_pointer_cast<const Call>(assign->value_);
      if (!call || !call->op_) continue;
      auto opnode = std::dynamic_pointer_cast<const Op>(call->op_);
      if (!opnode || !IsOp(opnode, "tile.store")) continue;
      if (call->args_.size() < 3) continue;

      // Check if the result var is NOT defined in the AIV body
      const Var* result_ptr = assign->var_.get();
      if (aiv_def_collector.var_defs.count(result_ptr)) continue;

      // Map the dangling result var to the fresh param for the store destination
      auto dest_var = std::dynamic_pointer_cast<const Var>(call->args_[2]);
      if (!dest_var) continue;

      // Find the fresh param that corresponds to the destination tensor
      auto param_it = aiv_map.find(dest_var.get());
      if (param_it != aiv_map.end()) {
        aiv_map[result_ptr] = param_it->second;
      }
    }

    // Also remap any undefined SSA versions of output parameters that survive in
    // the AIV body after AIC-side stores inside nested control flow are stripped.
    // Build an origin map: for each body Var, trace back to the function parameter
    // it was derived from. Propagates through assignments, iter_args, and return_vars.
    std::unordered_map<const Var*, const Var*> origin_map;
    // Seed with param -> param identity
    for (const auto& param : func->params_) {
      origin_map[param.get()] = param.get();
    }

    // Helper to propagate origin from a Var expression
    auto propagate_from_expr = [&](const Var* dest, const ExprPtr& src_expr) {
      if (auto src_var = std::dynamic_pointer_cast<const Var>(src_expr)) {
        auto it = origin_map.find(src_var.get());
        if (it != origin_map.end()) {
          origin_map[dest] = it->second;
        }
      }
    };

    // Recursive walk to propagate origins through assignments, iter_args, and return_vars
    std::function<void(const std::vector<StmtPtr>&)> walk_origins;
    walk_origins = [&](const std::vector<StmtPtr>& ss) {
      for (const auto& stmt : ss) {
        if (auto assign = std::dynamic_pointer_cast<const AssignStmt>(stmt)) {
          const Var* lhs = assign->var_.get();
          if (auto call = std::dynamic_pointer_cast<const Call>(assign->value_)) {
            auto opnode = std::dynamic_pointer_cast<const Op>(call->op_);
            if (opnode && IsOp(opnode, "tile.store") && call->args_.size() >= 3) {
              propagate_from_expr(lhs, call->args_[2]);
              continue;
            }
          }
          propagate_from_expr(lhs, assign->value_);
        } else if (auto for_stmt = std::dynamic_pointer_cast<const ForStmt>(stmt)) {
          // Propagate: iter_arg -> origin of init_value
          for (const auto& ia : for_stmt->iter_args_) {
            propagate_from_expr(ia.get(), ia->initValue_);
          }
          walk_origins(FlattenBody(for_stmt->body_));
          // Propagate: return_var[i] -> origin of iter_arg[i]
          for (size_t i = 0; i < for_stmt->return_vars_.size() && i < for_stmt->iter_args_.size(); ++i) {
            auto ia_it = origin_map.find(for_stmt->iter_args_[i].get());
            if (ia_it != origin_map.end()) {
              origin_map[for_stmt->return_vars_[i].get()] = ia_it->second;
            }
          }
        } else if (auto while_stmt = std::dynamic_pointer_cast<const WhileStmt>(stmt)) {
          for (const auto& ia : while_stmt->iter_args_) {
            propagate_from_expr(ia.get(), ia->initValue_);
          }
          walk_origins(FlattenBody(while_stmt->body_));
          for (size_t i = 0; i < while_stmt->return_vars_.size() && i < while_stmt->iter_args_.size(); ++i) {
            auto ia_it = origin_map.find(while_stmt->iter_args_[i].get());
            if (ia_it != origin_map.end()) {
              origin_map[while_stmt->return_vars_[i].get()] = ia_it->second;
            }
          }
        } else if (auto if_stmt = std::dynamic_pointer_cast<const IfStmt>(stmt)) {
          walk_origins(FlattenBody(if_stmt->then_body_));
          if (if_stmt->else_body_.has_value()) {
            walk_origins(FlattenBody(if_stmt->else_body_.value()));
          }
        } else if (auto seq = std::dynamic_pointer_cast<const SeqStmts>(stmt)) {
          walk_origins(seq->stmts_);
        }
      }
    };
    walk_origins(stmts);

    outline_utils::VarDefUseCollector aiv_ref_collector;
    aiv_ref_collector.VisitStmt(aiv_body_stmt);
    auto aiv_all_refs = var_collectors::GetSortedVarRefs(aiv_ref_collector.GetAllVarRefs());
    for (const Var* ref_ptr : aiv_all_refs) {
      if (!ref_ptr || aiv_def_collector.var_defs.count(ref_ptr) || aiv_map.count(ref_ptr)) {
        continue;
      }
      // Find the origin parameter for this dangling reference
      auto origin_it = origin_map.find(ref_ptr);
      if (origin_it == origin_map.end()) continue;
      const Var* origin_param = origin_it->second;
      // Verify the origin is an Out/InOut parameter
      for (size_t idx = 0; idx < func->params_.size() && idx < func->param_directions_.size(); ++idx) {
        if (func->param_directions_[idx] == ParamDirection::In) continue;
        if (func->params_[idx].get() != origin_param) continue;
        auto param_it = aiv_map.find(origin_param);
        if (param_it != aiv_map.end()) {
          aiv_map[ref_ptr] = param_it->second;
        }
        break;
      }
    }
  }

  auto [aiv_cloned_body, aiv_clone_map_unused] = DeepClone(MakeBody(aiv_final, func->span_), aiv_map);
  (void)aiv_clone_map_unused;
  auto aiv_attrs = func->attrs_;
  if (needs_dual_aiv_dispatch) {
    aiv_attrs.erase(std::remove_if(aiv_attrs.begin(), aiv_attrs.end(),
                                   [](const auto& kv) { return kv.first == kDualAivDispatchAttr; }),
                    aiv_attrs.end());
    aiv_attrs.emplace_back(kDualAivDispatchAttr, true);
  }
  auto aiv_func = std::make_shared<Function>(aiv_name, aiv_params, func->param_directions_,
                                             func->return_types_, aiv_cloned_body, func->span_,
                                             FunctionType::AIV, std::nullopt, std::nullopt, aiv_attrs);

  if (!create_group) {
    return {aic_func, aiv_func, std::nullopt};
  }

  // Create Group function: calls AIC then AIV, returns AIV result
  std::string group_name = func->name_;  // Group replaces the original

  // Create fresh parameters for the group function
  auto [group_params, group_map_unused] = make_param_map();
  (void)group_map_unused;

  // Build call args from group params
  std::vector<ExprPtr> call_args(group_params.begin(), group_params.end());

  // AIC call (no return value)
  auto aic_gvar = std::make_shared<GlobalVar>(aic_name);
  auto aic_call = std::make_shared<Call>(aic_gvar, call_args, func->span_);
  auto aic_eval = std::make_shared<EvalStmt>(aic_call, func->span_);

  // AIV call (returns result)
  auto aiv_gvar = std::make_shared<GlobalVar>(aiv_name);
  TypePtr aiv_return_type;
  if (func->return_types_.size() == 1) {
    aiv_return_type = func->return_types_[0];
  } else if (func->return_types_.size() > 1) {
    aiv_return_type = std::make_shared<TupleType>(func->return_types_);
  }

  CallPtr aiv_call;
  if (aiv_return_type) {
    aiv_call = std::make_shared<Call>(aiv_gvar, call_args, aiv_return_type, func->span_);
  } else {
    aiv_call = std::make_shared<Call>(aiv_gvar, call_args, func->span_);
  }

  // Build group body
  std::vector<StmtPtr> group_stmts;
  group_stmts.push_back(aic_eval);

  if (func->return_types_.empty()) {
    group_stmts.push_back(std::make_shared<EvalStmt>(aiv_call, func->span_));
  } else {
    // Return the group params the kernel's returns write through (group_params
    // is positionally 1:1 with func->params_). Explicit param returns keep the
    // return->param mapping a pointer-identity lookup for orchestration
    // codegen (#1702). Fall back to returning the AIV result only when some
    // position is not a param writeback.
    auto returned_idxs = return_lineage::ReturnedParamIndices(func, nullptr);
    bool all_param_returns = returned_idxs.size() == func->return_types_.size();
    for (const auto& idx : returned_idxs) {
      all_param_returns = all_param_returns && idx.has_value() && idx.value() < group_params.size();
    }
    if (all_param_returns) {
      group_stmts.push_back(std::make_shared<EvalStmt>(aiv_call, func->span_));
      std::vector<ExprPtr> return_exprs;
      return_exprs.reserve(returned_idxs.size());
      for (const auto& idx : returned_idxs) {
        // all_param_returns guarantees has_value().
        return_exprs.push_back(group_params[idx.value()]);  // NOLINT(bugprone-unchecked-optional-access)
      }
      group_stmts.push_back(std::make_shared<ReturnStmt>(return_exprs, func->span_));
    } else {
      auto result_var = std::make_shared<Var>("result", aiv_return_type, func->span_);
      group_stmts.push_back(std::make_shared<AssignStmt>(result_var, aiv_call, func->span_));
      std::vector<ExprPtr> return_exprs = {result_var};
      group_stmts.push_back(std::make_shared<ReturnStmt>(return_exprs, func->span_));
    }
  }

  auto group_body = SeqStmts::Flatten(std::move(group_stmts), func->span_);
  auto group_func = std::make_shared<Function>(group_name, group_params, func->param_directions_,
                                               func->return_types_, group_body, func->span_,
                                               FunctionType::Group, std::nullopt, std::nullopt, func->attrs_);

  return {aic_func, aiv_func, group_func};
}

// ============================================================================
// Rewrite existing Group callers to replace InCore calls with AIC+AIV
// ============================================================================

/// Rewrite a Group function's body, replacing calls to `incore_name` with
/// an EvalStmt(Call(aic_name)) + AssignStmt/EvalStmt(Call(aiv_name)).
FunctionPtr RewriteGroupCaller(const FunctionPtr& group_func, const std::string& incore_name,
                               const std::string& aic_name, const std::string& aiv_name) {
  auto stmts = FlattenBody(group_func->body_);
  std::vector<StmtPtr> new_stmts;

  for (const auto& stmt : stmts) {
    // Extract the Call targeting incore_name (from AssignStmt or EvalStmt)
    CallPtr call;
    auto assign = std::dynamic_pointer_cast<const AssignStmt>(stmt);
    if (assign) {
      call = std::dynamic_pointer_cast<const Call>(assign->value_);
    } else if (auto eval = std::dynamic_pointer_cast<const EvalStmt>(stmt)) {
      call = std::dynamic_pointer_cast<const Call>(eval->expr_);
    }

    if (call) {
      auto gv = std::dynamic_pointer_cast<const GlobalVar>(call->op_);
      if (gv && gv->name_ == incore_name) {
        // Emit AIC call (always fire-and-forget). Original's leading_comments
        // attach here — AIC is the semantic front of the split pair.
        // Carry kwargs_ + attrs_ (e.g. kAttrDumpVars) onto both lanes: the
        // split reuses call->args_ unchanged, so dump/dep Var references stay
        // valid, and orchestration codegen matches them per-arg by identity.
        auto aic_call = std::make_shared<Call>(std::make_shared<GlobalVar>(aic_name), call->args_,
                                               call->kwargs_, call->attrs_, GetUnknownType(), stmt->span_);
        new_stmts.push_back(std::make_shared<EvalStmt>(aic_call, stmt->span_, stmt->leading_comments_));

        // Emit AIV call: AssignStmt preserves return value, EvalStmt for void.
        // AIV is a continuation of the same logical op, so no comments attach.
        if (assign) {
          auto aiv_call = std::make_shared<Call>(std::make_shared<GlobalVar>(aiv_name), call->args_,
                                                 call->kwargs_, call->attrs_, call->GetType(), stmt->span_);
          new_stmts.push_back(std::make_shared<AssignStmt>(assign->var_, aiv_call, stmt->span_));
        } else {
          auto aiv_call = std::make_shared<Call>(std::make_shared<GlobalVar>(aiv_name), call->args_,
                                                 call->kwargs_, call->attrs_, GetUnknownType(), stmt->span_);
          new_stmts.push_back(std::make_shared<EvalStmt>(aiv_call, stmt->span_));
        }
        continue;
      }
    }

    new_stmts.push_back(stmt);
  }

  auto new_body = SeqStmts::Flatten(std::move(new_stmts), group_func->span_);
  auto result = std::make_shared<Function>(group_func->name_, group_func->params_,
                                           group_func->param_directions_, group_func->return_types_, new_body,
                                           group_func->span_, group_func->func_type_, group_func->level_,
                                           group_func->role_, group_func->attrs_);
  return result;
}

/// Check if a function body contains a call to a given function name.
bool FunctionCallsFunction(const FunctionPtr& func, const std::string& callee_name) {
  auto stmts = FlattenBody(func->body_);
  for (const auto& stmt : stmts) {
    if (auto call = transform_utils::GetCallFromStmt(stmt)) {
      auto gv = std::dynamic_pointer_cast<const GlobalVar>(call->op_);
      if (gv && gv->name_ == callee_name) return true;
    }
  }
  return false;
}

}  // namespace

namespace pass {

Pass ExpandMixedKernel() {
  auto pass_func = [](const ProgramPtr& program) -> ProgramPtr {
    // Phase 1: Pre-scan — find InCore functions that have existing callers.
    std::unordered_set<std::string> incore_names;
    for (const auto& [gvar, func] : program->functions_) {
      if (func->func_type_ == FunctionType::InCore) {
        incore_names.insert(func->name_);
      }
    }

    // Map InCore name -> callers that can be rewritten in place.
    std::unordered_set<std::string> incore_with_group_caller;
    // Map InCore name -> callers that still need the original function name to remain callable.
    std::unordered_set<std::string> incore_with_preserved_name_caller;
    for (const auto& [gvar, func] : program->functions_) {
      for (const auto& name : incore_names) {
        if (!FunctionCallsFunction(func, name)) {
          continue;
        }
        if (func->func_type_ == FunctionType::Group) {
          incore_with_group_caller.insert(name);
        } else {
          incore_with_preserved_name_caller.insert(name);
        }
      }
    }

    // Phase 2: Expand InCore functions, collect rewrite info
    struct RewriteInfo {
      std::string aic_name;
      std::string aiv_name;
    };
    std::unordered_map<std::string, RewriteInfo> rewrite_map;
    std::vector<FunctionPtr> new_functions;

    for (const auto& [gvar, func] : program->functions_) {
      if (func->func_type_ != FunctionType::InCore) {
        new_functions.push_back(func);
        continue;
      }

      // Check if function is mixed (recursive analysis detects ops inside loops/conditionals)
      auto stmts = FlattenBody(func->body_);
      auto tpop_defs = CollectTpopDefs(stmts);
      std::unordered_map<const Stmt*, CoreAffinity> stmt_map;
      std::unordered_map<const Var*, CoreAffinity> var_affinity;
      auto combined = AnalyzeStmtsAffinity(stmts, stmt_map, var_affinity, tpop_defs);

      // A function is mixed if combined affinity says so. Leaf boundary moves
      // (tile.move across the C/V divide) classify as MIXED via ClassifyCallAffinity,
      // so the roll-up captures them without a separate enum value.
      bool is_mixed = (combined == CoreAffinity::MIXED);

      if (!is_mixed) {
        // Not mixed — convert InCore to the corresponding AIC or AIV type
        FunctionType new_type = (combined == CoreAffinity::CUBE) ? FunctionType::AIC : FunctionType::AIV;
        // Clear split mode — pure AIC/AIV functions don't need vector splitting
        auto attrs = func->attrs_;
        attrs.erase(
            std::remove_if(attrs.begin(), attrs.end(), [](const auto& kv) { return kv.first == "split"; }),
            attrs.end());
        auto converted = MutableCopy(func);
        converted->func_type_ = new_type;
        converted->level_ = FunctionTypeToLevel(new_type);
        converted->role_ = Role::SubWorker;
        converted->attrs_ = attrs;
        new_functions.push_back(converted);
        continue;
      }
      // Expand mixed kernel.
      // Existing Group callers can be rewritten in place, but any non-Group
      // caller (for example a standalone Spmd wrapper) still needs the
      // original function name to resolve to a callable wrapper.
      bool has_group_caller = incore_with_group_caller.count(func->name_) > 0;
      bool needs_preserved_name = incore_with_preserved_name_caller.count(func->name_) > 0;
      auto expanded = ExpandMixedFunction(func, /*create_group=*/needs_preserved_name || !has_group_caller);

      new_functions.push_back(expanded.aic_func);
      new_functions.push_back(expanded.aiv_func);
      if (expanded.group_func.has_value()) {
        new_functions.push_back(expanded.group_func.value());
      }

      if (has_group_caller) {
        rewrite_map[func->name_] = {expanded.aic_func->name_, expanded.aiv_func->name_};
      }
    }

    // Phase 3: Rewrite existing Group callers to call AIC+AIV directly
    for (auto& func : new_functions) {
      if (func->func_type_ != FunctionType::Group) continue;
      for (const auto& [incore_name, info] : rewrite_map) {
        if (FunctionCallsFunction(func, incore_name)) {
          func = RewriteGroupCaller(func, incore_name, info.aic_name, info.aiv_name);
        }
      }
    }

    return std::make_shared<Program>(new_functions, program->name_, program->span_);
  };

  return CreateProgramPass(pass_func, "ExpandMixedKernel", kExpandMixedKernelProperties);
}

}  // namespace pass

}  // namespace ir
}  // namespace pypto
