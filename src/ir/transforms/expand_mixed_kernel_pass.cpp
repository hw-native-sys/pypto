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
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/backend/common/backend.h"
#include "pypto/backend/common/backend_config.h"
#include "pypto/core/any_cast.h"
#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/core_affinity.h"
#include "pypto/ir/transforms/utils/cross_core_pipe.h"
#include "pypto/ir/transforms/utils/dead_code_elimination.h"
#include "pypto/ir/transforms/utils/deep_clone_utils.h"
#include "pypto/ir/transforms/utils/loop_state_repair.h"
#include "pypto/ir/transforms/utils/scope_outline_utils.h"
#include "pypto/ir/transforms/utils/tpop_chain_normalizer.h"
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
using core_affinity::CoreSide;
using core_affinity::CVBoundaryMove;
using core_affinity::CVDirection;
using cross_core_pipe::BuildAutomaticPipeSetup;
using cross_core_pipe::PrependPipeSetup;
using loop_repair::BuildDefMap;
using loop_repair::FinalizeSplitCoreBody;
using loop_repair::MakeBody;
using tpop_chain::NormalizeTpopChains;

// ============================================================================
// Flatten body helper
// ============================================================================

// Use the shared utility; local alias preserves call sites.
const auto& FlattenBody = transform_utils::FlattenToStmts;

// ============================================================================
// Recursive Affinity Analysis
// ============================================================================

/// Collect Vars defined by tpop operations (tile.tpop_from_aiv / tile.tpop_from_aic).
/// Used to avoid misclassifying tile.move from a tpop result as a cross-core boundary.
std::unordered_set<const Var*> CollectTpopVars(const std::vector<StmtPtr>& stmts) {
  std::unordered_set<const Var*> result;
  for (const auto& stmt : stmts) {
    if (auto assign = std::dynamic_pointer_cast<const AssignStmt>(stmt)) {
      if (auto call = std::dynamic_pointer_cast<const Call>(assign->value_)) {
        if (auto op = std::dynamic_pointer_cast<const Op>(call->op_)) {
          if (op->name_ == "tile.tpop_from_aiv" || op->name_ == "tile.tpop_from_aic") {
            result.insert(assign->var_.get());
          }
        }
      }
    }
    // Recurse into compound statements
    if (auto for_stmt = std::dynamic_pointer_cast<const ForStmt>(stmt)) {
      auto inner = CollectTpopVars(FlattenBody(for_stmt->body_));
      result.insert(inner.begin(), inner.end());
    } else if (auto if_stmt = std::dynamic_pointer_cast<const IfStmt>(stmt)) {
      auto inner = CollectTpopVars(FlattenBody(if_stmt->then_body_));
      result.insert(inner.begin(), inner.end());
      if (if_stmt->else_body_.has_value()) {
        auto inner2 = CollectTpopVars(FlattenBody(if_stmt->else_body_.value()));
        result.insert(inner2.begin(), inner2.end());
      }
    } else if (auto while_stmt = std::dynamic_pointer_cast<const WhileStmt>(stmt)) {
      auto inner = CollectTpopVars(FlattenBody(while_stmt->body_));
      result.insert(inner.begin(), inner.end());
    }
  }
  return result;
}

// Forward declare
CoreAffinity AnalyzeStmtAffinity(const StmtPtr& stmt, std::unordered_map<const Stmt*, CoreAffinity>& stmt_map,
                                 std::unordered_map<const Var*, CoreAffinity>& var_affinity,
                                 const std::unordered_set<const Var*>& tpop_vars);

CoreAffinity AnalyzeStmtsAffinity(const std::vector<StmtPtr>& stmts,
                                  std::unordered_map<const Stmt*, CoreAffinity>& stmt_map,
                                  std::unordered_map<const Var*, CoreAffinity>& var_affinity,
                                  const std::unordered_set<const Var*>& tpop_vars = {}) {
  CoreAffinity combined = CoreAffinity::SHARED;
  for (const auto& stmt : stmts) {
    combined = CombineAffinity(combined, AnalyzeStmtAffinity(stmt, stmt_map, var_affinity, tpop_vars));
  }
  return combined;
}

CoreAffinity AnalyzeStmtAffinity(const StmtPtr& stmt, std::unordered_map<const Stmt*, CoreAffinity>& stmt_map,
                                 std::unordered_map<const Var*, CoreAffinity>& var_affinity,
                                 const std::unordered_set<const Var*>& tpop_vars) {
  CoreAffinity result = CoreAffinity::SHARED;

  if (auto assign = std::dynamic_pointer_cast<const AssignStmt>(stmt)) {
    auto call = std::dynamic_pointer_cast<const Call>(assign->value_);
    if (call) {
      result = ClassifyCallAffinity(call);
      // tile.move from a tpop result is not a cross-core boundary: the data already
      // arrived via tpop, so the move is just internal data placement on the consuming core.
      if (result == CoreAffinity::BOUNDARY && !call->args_.empty()) {
        if (auto src_var = std::dynamic_pointer_cast<const Var>(call->args_[0])) {
          if (tpop_vars.count(src_var.get()) > 0) {
            // Reclassify by target memory space (the consuming core's side)
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
    result = AnalyzeStmtsAffinity(FlattenBody(for_stmt->body_), stmt_map, var_affinity, tpop_vars);
  } else if (auto if_stmt = std::dynamic_pointer_cast<const IfStmt>(stmt)) {
    result = AnalyzeStmtsAffinity(FlattenBody(if_stmt->then_body_), stmt_map, var_affinity, tpop_vars);
    if (if_stmt->else_body_.has_value()) {
      result = CombineAffinity(result, AnalyzeStmtsAffinity(FlattenBody(if_stmt->else_body_.value()),
                                                            stmt_map, var_affinity, tpop_vars));
    }
  } else if (auto while_stmt = std::dynamic_pointer_cast<const WhileStmt>(stmt)) {
    result = AnalyzeStmtsAffinity(FlattenBody(while_stmt->body_), stmt_map, var_affinity, tpop_vars);
  } else if (auto seq = std::dynamic_pointer_cast<const SeqStmts>(stmt)) {
    result = AnalyzeStmtsAffinity(seq->stmts_, stmt_map, var_affinity, tpop_vars);
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
                            const std::unordered_map<const Var*, CallPtr>& var_to_tpop) {
  for (const auto& stmt : stmts) {
    if (auto assign = std::dynamic_pointer_cast<const AssignStmt>(stmt)) {
      auto call = std::dynamic_pointer_cast<const Call>(assign->value_);
      if (call) {
        auto dir = ClassifyMoveDirection(call);
        if (dir != CVDirection::NONE) {
          INTERNAL_CHECK(!call->args_.empty()) << "Internal error: tile.move must have at least one argument";
          // Look up whether the source tile was produced by a tpop call
          CallPtr source_tpop;
          if (auto source_var = std::dynamic_pointer_cast<const Var>(call->args_[0])) {
            auto it = var_to_tpop.find(source_var.get());
            if (it != var_to_tpop.end()) {
              source_tpop = it->second;
            }
          }
          boundary_moves[stmt.get()] =
              CVBoundaryMove{dir, assign->var_, call->args_[0], call->GetType(), source_tpop};
        }
      }
    }

    // Recurse into compound statements
    if (auto for_stmt = std::dynamic_pointer_cast<const ForStmt>(stmt)) {
      CollectCVBoundaryMoves(FlattenBody(for_stmt->body_), boundary_moves, var_to_tpop);
    } else if (auto if_stmt = std::dynamic_pointer_cast<const IfStmt>(stmt)) {
      CollectCVBoundaryMoves(FlattenBody(if_stmt->then_body_), boundary_moves, var_to_tpop);
      if (if_stmt->else_body_.has_value()) {
        CollectCVBoundaryMoves(FlattenBody(if_stmt->else_body_.value()), boundary_moves, var_to_tpop);
      }
    } else if (auto while_stmt = std::dynamic_pointer_cast<const WhileStmt>(stmt)) {
      CollectCVBoundaryMoves(FlattenBody(while_stmt->body_), boundary_moves, var_to_tpop);
    }
  }
}

// ============================================================================
// TPUSH / TPOP creation helpers
// ============================================================================

std::vector<std::pair<std::string, std::any>> MakeSplitKwargs() { return {{"split", std::any(0)}}; }

CallPtr CreateTpush(const std::string& op_name, const ExprPtr& tile, const Span& span) {
  return OpRegistry::GetInstance().Create(op_name, {tile}, MakeSplitKwargs(), span);
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
  if (auto tt = std::dynamic_pointer_cast<const TileType>(result_type); tt && tt->tile_view_.has_value()) {
    kwargs.emplace_back("blayout", std::any(tt->tile_view_->blayout));
    kwargs.emplace_back("slayout", std::any(tt->tile_view_->slayout));
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
/// Non-transpose 950: Left -> NZ, Right -> ZN, Mat/Vec -> preserve original.
TileView BuildCrossCoreTransferView(MemorySpace dest_ms, const TileView& original_view) {
  INTERNAL_CHECK(backend::GetBackendType() == backend::BackendType::Ascend950)
      << "BuildCrossCoreTransferView is only supported on Ascend950 backend";

  TileView result = original_view;
  switch (dest_ms) {
    case MemorySpace::Left:
      result.blayout = TileLayout::col_major;
      result.slayout = TileLayout::row_major;
      return result;
    case MemorySpace::Right:
      result.blayout = TileLayout::row_major;
      result.slayout = TileLayout::col_major;
      return result;
    case MemorySpace::Mat:
    case MemorySpace::Vec:
      return original_view;
    default:
      INTERNAL_UNREACHABLE << "cross-core move destination must be Vec, Mat, Left, or Right, got "
                           << static_cast<int>(dest_ms);
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
                                   std::unordered_set<const Var*>& superseded_tpop_vars) {
  // AIC keeps CUBE, skips VECTOR; AIV keeps VECTOR, skips CUBE
  CoreAffinity keep_affinity = (side == CoreSide::AIC) ? CoreAffinity::CUBE : CoreAffinity::VECTOR;
  CoreAffinity skip_affinity = (side == CoreSide::AIC) ? CoreAffinity::VECTOR : CoreAffinity::CUBE;

  // For boundary moves: the "push" side sends data, the "pop" side receives it.
  // AIC: C→V = push to AIV, V→C = pop from AIV
  // AIV: C→V = pop from AIC, V→C = push to AIC
  std::string push_op = (side == CoreSide::AIC) ? "tile.tpush_to_aiv" : "tile.tpush_to_aic";
  std::string pop_op = (side == CoreSide::AIC) ? "tile.tpop_from_aiv" : "tile.tpop_from_aic";
  // AIC pushes on C→V and pops on V→C; AIV is the reverse
  CVDirection push_direction =
      (side == CoreSide::AIC) ? CVDirection::CUBE_TO_VECTOR : CVDirection::VECTOR_TO_CUBE;

  std::vector<StmtPtr> result;

  for (const auto& stmt : stmts) {
    auto it = stmt_map.find(stmt.get());
    CoreAffinity affinity = (it != stmt_map.end()) ? it->second : CoreAffinity::SHARED;

    if (affinity == CoreAffinity::BOUNDARY) {
      auto bm_it = boundary_moves.find(stmt.get());
      if (bm_it != boundary_moves.end()) {
        // Leaf boundary move — emit tpush/tpop
        const auto& bm = bm_it->second;
        if (bm.direction == push_direction) {
          ExprPtr push_source = bm.source_tile;
          // AIV V->C push: insert tile.move (tmov) to adapt fractal layout before tpush
          if (side == CoreSide::AIV) {
            auto push_dest_type = std::dynamic_pointer_cast<const TileType>(bm.dest_var->GetType());
            INTERNAL_CHECK(push_dest_type && push_dest_type->memory_space_.has_value() &&
                           push_dest_type->tile_view_.has_value())
                << "Boundary move destination must have TileType, MemSpace and TileView";

            // NOLINT: optional checked by INTERNAL_CHECK above
            auto fractal_view = BuildCrossCoreTransferView(
                push_dest_type->memory_space_.value(),  // NOLINT(bugprone-unchecked-optional-access)
                push_dest_type->tile_view_.value());    // NOLINT(bugprone-unchecked-optional-access)

            auto src_type = std::dynamic_pointer_cast<const TileType>(bm.source_tile->GetType());
            INTERNAL_CHECK(src_type) << "V->C tpush source must have TileType";
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
          result.push_back(
              std::make_shared<EvalStmt>(CreateTpush(push_op, push_source, stmt->span_), stmt->span_));
        } else {
          auto dest_tile_type = std::dynamic_pointer_cast<const TileType>(bm.dest_var->GetType());
          INTERNAL_CHECK(dest_tile_type && dest_tile_type->memory_space_.has_value() &&
                         dest_tile_type->tile_view_.has_value())
              << "Boundary move destination must have TileType, MemSpace and TileView";
          auto tpop_type = BuildBoundaryTpopType(side, bm.dest_var->GetType());
          bool needs_post_move = NeedsPostTpopMove(side, *dest_tile_type);
          std::string tpop_name = needs_post_move ? BuildBoundaryTpopName(side, bm.dest_var->name_hint_)
                                                  : bm.dest_var->name_hint_;
          // Build tpop result type: with fractal TileView for boundary
          // NOLINT: optional checked by INTERNAL_CHECK above
          auto fractal_view = BuildCrossCoreTransferView(
              dest_tile_type->memory_space_.value(),  // NOLINT(bugprone-unchecked-optional-access)
              dest_tile_type->tile_view_.value());    // NOLINT(bugprone-unchecked-optional-access)
          auto tt = std::dynamic_pointer_cast<const TileType>(tpop_type);
          auto tpop_result_type = std::make_shared<TileType>(tt->shape_, tt->dtype_, std::nullopt,
                                                             fractal_view, tt->memory_space_);
          auto tpop_var = std::make_shared<Var>(tpop_name, tpop_result_type, stmt->span_);
          if (!needs_post_move) {
            tpop_var_remap[bm.dest_var.get()] = tpop_var;
          }
          if (auto source_var = std::dynamic_pointer_cast<const Var>(bm.source_tile)) {
            tpop_var_remap[source_var.get()] = tpop_var;
          }
          tpop_var_remap[tpop_var.get()] = tpop_var;
          // Propagate kwargs from the original tpop (e.g., split=1) if available
          std::vector<std::pair<std::string, std::any>> kwargs;
          if (bm.source_tpop_call) {
            kwargs = bm.source_tpop_call->kwargs_;
          }
          result.push_back(std::make_shared<AssignStmt>(
              tpop_var, CreateTpop(pop_op, tpop_result_type, stmt->span_, kwargs), stmt->span_));
          if (needs_post_move) {
            auto target_memory = dest_tile_type->memory_space_;
            INTERNAL_CHECK(target_memory.has_value())
                << "Boundary move destination must have memory_space before post-tpop move emission";
            result.push_back(std::make_shared<AssignStmt>(
                bm.dest_var, CreateMove(tpop_var, *target_memory, bm.dest_var->GetType(), stmt->span_),
                stmt->span_));
          }
        }
        continue;
      }
      // Compound stmt whose children are all BOUNDARY — recurse like MIXED
      affinity = CoreAffinity::MIXED;
    }

    if (affinity == skip_affinity) continue;

    // Skip original tpop statements whose results are superseded by boundary-generated tpops
    if (auto assign = std::dynamic_pointer_cast<const AssignStmt>(stmt)) {
      if (superseded_tpop_vars.count(assign->var_.get()) > 0) continue;
    }

    if (affinity == keep_affinity || affinity == CoreAffinity::SHARED) {
      result.push_back(stmt);
    } else if (affinity == CoreAffinity::MIXED) {
      // Recurse into compound statements, building pruned copies
      if (auto for_stmt = std::dynamic_pointer_cast<const ForStmt>(stmt)) {
        auto new_body = BuildCoreBody(side, FlattenBody(for_stmt->body_), stmt_map, boundary_moves,
                                      tpop_var_remap, superseded_tpop_vars);
        result.push_back(std::make_shared<ForStmt>(
            for_stmt->loop_var_, for_stmt->start_, for_stmt->stop_, for_stmt->step_, for_stmt->iter_args_,
            MakeBody(new_body, for_stmt->span_), for_stmt->return_vars_, for_stmt->span_, for_stmt->kind_,
            for_stmt->chunk_size_, for_stmt->chunk_policy_, for_stmt->loop_origin_));
      } else if (auto if_stmt = std::dynamic_pointer_cast<const IfStmt>(stmt)) {
        auto new_then = BuildCoreBody(side, FlattenBody(if_stmt->then_body_), stmt_map, boundary_moves,
                                      tpop_var_remap, superseded_tpop_vars);
        std::optional<StmtPtr> new_else;
        if (if_stmt->else_body_.has_value()) {
          auto new_else_stmts = BuildCoreBody(side, FlattenBody(if_stmt->else_body_.value()), stmt_map,
                                              boundary_moves, tpop_var_remap, superseded_tpop_vars);
          new_else = MakeBody(new_else_stmts, if_stmt->span_);
        }
        result.push_back(std::make_shared<IfStmt>(if_stmt->condition_, MakeBody(new_then, if_stmt->span_),
                                                  new_else, if_stmt->return_vars_, if_stmt->span_));
      } else if (auto while_stmt = std::dynamic_pointer_cast<const WhileStmt>(stmt)) {
        auto new_body = BuildCoreBody(side, FlattenBody(while_stmt->body_), stmt_map, boundary_moves,
                                      tpop_var_remap, superseded_tpop_vars);
        result.push_back(std::make_shared<WhileStmt>(while_stmt->condition_, while_stmt->iter_args_,
                                                     MakeBody(new_body, while_stmt->span_),
                                                     while_stmt->return_vars_, while_stmt->span_));
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
  auto stmts = FlattenBody(func->body_);

  // Pre-scan for tpop result vars (needed by affinity analysis to avoid
  // misclassifying tile.move from tpop results as cross-core boundaries)
  auto tpop_vars = CollectTpopVars(stmts);

  // Recursive affinity analysis (descends into ForStmt/IfStmt/WhileStmt)
  std::unordered_map<const Stmt*, CoreAffinity> stmt_map;
  std::unordered_map<const Var*, CoreAffinity> var_affinity;
  AnalyzeStmtsAffinity(stmts, stmt_map, var_affinity, tpop_vars);

  // Collect CV boundary moves from explicit tile.move ops.
  // First, build a map from Var -> defining tpop Call so boundary moves can
  // propagate kwargs (e.g., split) from the original tpop to the replacement.
  // Uses the already-collected tpop_vars set and rescans for the full Call objects.
  std::unordered_map<const Var*, CallPtr> var_to_tpop;
  auto collect_var_to_tpop = [&](const std::vector<StmtPtr>& ss, auto&& self) -> void {
    for (const auto& stmt : ss) {
      if (auto assign = std::dynamic_pointer_cast<const AssignStmt>(stmt)) {
        if (auto call = std::dynamic_pointer_cast<const Call>(assign->value_)) {
          if (auto op = std::dynamic_pointer_cast<const Op>(call->op_)) {
            if (op->name_ == "tile.tpop_from_aiv" || op->name_ == "tile.tpop_from_aic") {
              var_to_tpop[assign->var_.get()] = call;
            }
          }
        }
      }
      if (auto for_stmt = std::dynamic_pointer_cast<const ForStmt>(stmt)) {
        self(FlattenBody(for_stmt->body_), self);
      } else if (auto if_stmt = std::dynamic_pointer_cast<const IfStmt>(stmt)) {
        self(FlattenBody(if_stmt->then_body_), self);
        if (if_stmt->else_body_.has_value()) {
          self(FlattenBody(if_stmt->else_body_.value()), self);
        }
      } else if (auto while_stmt = std::dynamic_pointer_cast<const WhileStmt>(stmt)) {
        self(FlattenBody(while_stmt->body_), self);
      }
    }
  };
  collect_var_to_tpop(stmts, collect_var_to_tpop);
  std::map<const Stmt*, CVBoundaryMove> boundary_moves;
  CollectCVBoundaryMoves(stmts, boundary_moves, var_to_tpop);

  // Build definition map from original body for init value fixup (#533)
  std::unordered_map<const Var*, StmtPtr> original_def_map;
  BuildDefMap(stmts, original_def_map);

  // Precompute the set of source Vars whose original tpop is superseded by a
  // boundary-generated tpop.  This must be done before BuildCoreBody because
  // the original tpop statement appears before the boundary tile.move in
  // program order — computing it lazily inside the loop would be too late.
  std::unordered_set<const Var*> superseded_tpop_vars;
  for (const auto& [stmt_ptr, bm] : boundary_moves) {
    if (bm.source_tpop_call) {
      if (auto source_var = std::dynamic_pointer_cast<const Var>(bm.source_tile)) {
        superseded_tpop_vars.insert(source_var.get());
      }
    }
  }

  // Build AIC body (recursive — handles MIXED compound stmts)
  std::unordered_map<const Var*, VarPtr> aic_tpop_remap;
  auto aic_stmts =
      BuildCoreBody(CoreSide::AIC, stmts, stmt_map, boundary_moves, aic_tpop_remap, superseded_tpop_vars);

  // Remove ReturnStmt from AIC (AIC doesn't return values)
  std::vector<StmtPtr> aic_stmts_no_return;
  for (const auto& s : aic_stmts) {
    if (!std::dynamic_pointer_cast<const ReturnStmt>(s)) {
      aic_stmts_no_return.push_back(s);
    }
  }
  auto aic_final = NormalizeTpopChains(FinalizeSplitCoreBody(aic_stmts_no_return, original_def_map),
                                       CoreSide::AIC, aic_tpop_remap);

  // Build AIV body (recursive — handles MIXED compound stmts)
  std::unordered_map<const Var*, VarPtr> aiv_tpop_remap;
  auto aiv_stmts =
      BuildCoreBody(CoreSide::AIV, stmts, stmt_map, boundary_moves, aiv_tpop_remap, superseded_tpop_vars);
  auto aiv_final =
      NormalizeTpopChains(FinalizeSplitCoreBody(aiv_stmts, original_def_map), CoreSide::AIV, aiv_tpop_remap);

  const std::string aic_name = func->name_ + "_aic";
  const std::string aiv_name = func->name_ + "_aiv";
  auto automatic_pipe_setup =
      BuildAutomaticPipeSetup(func->name_, aic_name, aiv_name, aic_final, aiv_final, func->span_);
  aic_final = PrependPipeSetup(automatic_pipe_setup.aic_stmts, aic_final);
  aiv_final = PrependPipeSetup(automatic_pipe_setup.aiv_stmts, aiv_final);

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
      if (!opnode || opnode->name_ != "tile.store") continue;
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
            if (opnode && opnode->name_ == "tile.store" && call->args_.size() >= 3) {
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
  auto aiv_func = std::make_shared<Function>(aiv_name, aiv_params, func->param_directions_,
                                             func->return_types_, aiv_cloned_body, func->span_,
                                             FunctionType::AIV, std::nullopt, std::nullopt, func->attrs_);

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

  auto aiv_call = aiv_return_type ? std::make_shared<Call>(aiv_gvar, call_args, aiv_return_type, func->span_)
                                  : std::make_shared<Call>(aiv_gvar, call_args, func->span_);

  // Build group body
  std::vector<StmtPtr> group_stmts;
  group_stmts.push_back(aic_eval);

  if (func->return_types_.empty()) {
    group_stmts.push_back(std::make_shared<EvalStmt>(aiv_call, func->span_));
  } else {
    // Assign AIV result and return it
    auto result_var = std::make_shared<Var>("result", aiv_return_type, func->span_);
    group_stmts.push_back(std::make_shared<AssignStmt>(result_var, aiv_call, func->span_));
    std::vector<ExprPtr> return_exprs = {result_var};
    group_stmts.push_back(std::make_shared<ReturnStmt>(return_exprs, func->span_));
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
        // Emit AIC call (always fire-and-forget)
        auto aic_call =
            std::make_shared<Call>(std::make_shared<GlobalVar>(aic_name), call->args_, stmt->span_);
        new_stmts.push_back(std::make_shared<EvalStmt>(aic_call, stmt->span_));

        // Emit AIV call: AssignStmt preserves return value, EvalStmt for void
        if (assign) {
          auto aiv_call = std::make_shared<Call>(std::make_shared<GlobalVar>(aiv_name), call->args_,
                                                 call->GetType(), stmt->span_);
          new_stmts.push_back(std::make_shared<AssignStmt>(assign->var_, aiv_call, stmt->span_));
        } else {
          auto aiv_call =
              std::make_shared<Call>(std::make_shared<GlobalVar>(aiv_name), call->args_, stmt->span_);
          new_stmts.push_back(std::make_shared<EvalStmt>(aiv_call, stmt->span_));
        }
        continue;
      }
    }

    new_stmts.push_back(stmt);
  }

  auto new_body = SeqStmts::Flatten(std::move(new_stmts), group_func->span_);
  auto result =
      std::make_shared<Function>(group_func->name_, group_func->params_, group_func->param_directions_,
                                 group_func->return_types_, new_body, group_func->span_, FunctionType::Group,
                                 group_func->level_, group_func->role_, group_func->attrs_);
  return result;
}

/// Check if a Group function body contains a call to a given function name.
bool GroupCallsFunction(const FunctionPtr& group_func, const std::string& callee_name) {
  auto stmts = FlattenBody(group_func->body_);
  for (const auto& stmt : stmts) {
    CallPtr call;
    if (auto assign = std::dynamic_pointer_cast<const AssignStmt>(stmt)) {
      call = std::dynamic_pointer_cast<const Call>(assign->value_);
    } else if (auto eval = std::dynamic_pointer_cast<const EvalStmt>(stmt)) {
      call = std::dynamic_pointer_cast<const Call>(eval->expr_);
    }
    if (call) {
      auto gv = std::dynamic_pointer_cast<const GlobalVar>(call->op_);
      if (gv && gv->name_ == callee_name) return true;
    }
  }
  return false;
}

// ============================================================================
// GM Slot Buffer Injection (a2a3 only)
// ============================================================================

/// Well-known parameter name for the GM slot buffer.
constexpr const char* kGMPipeBufferName = "__gm_pipe_buffer";

/// Check if a statement list contains aic_initialize_pipe or aiv_initialize_pipe.
bool HasInitializePipeOps(const std::vector<StmtPtr>& stmts) {
  for (const auto& stmt : stmts) {
    CallPtr call;
    if (auto assign = std::dynamic_pointer_cast<const AssignStmt>(stmt)) {
      call = std::dynamic_pointer_cast<const Call>(assign->value_);
    } else if (auto eval = std::dynamic_pointer_cast<const EvalStmt>(stmt)) {
      call = std::dynamic_pointer_cast<const Call>(eval->expr_);
    }
    if (call) {
      auto op = std::dynamic_pointer_cast<const Op>(call->op_);
      if (op && (op->name_ == "system.aic_initialize_pipe" || op->name_ == "system.aiv_initialize_pipe")) {
        return true;
      }
    }
    if (auto for_stmt = std::dynamic_pointer_cast<const ForStmt>(stmt)) {
      if (HasInitializePipeOps(FlattenBody(for_stmt->body_))) return true;
    } else if (auto if_stmt = std::dynamic_pointer_cast<const IfStmt>(stmt)) {
      if (HasInitializePipeOps(FlattenBody(if_stmt->then_body_))) return true;
      if (if_stmt->else_body_.has_value()) {
        if (HasInitializePipeOps(FlattenBody(if_stmt->else_body_.value()))) return true;
      }
    } else if (auto while_stmt = std::dynamic_pointer_cast<const WhileStmt>(stmt)) {
      if (HasInitializePipeOps(FlattenBody(while_stmt->body_))) return true;
    }
  }
  return false;
}

bool HasGMPipeBufferParam(const FunctionPtr& func) {
  for (const auto& param : func->params_) {
    if (param->name_hint_ == kGMPipeBufferName) return true;
  }
  return false;
}

void BuildCallGraphFromFunctions(const std::vector<FunctionPtr>& functions,
                                 std::unordered_map<std::string, std::unordered_set<std::string>>& callers,
                                 std::unordered_map<std::string, std::unordered_set<std::string>>& callees) {
  std::unordered_set<std::string> func_names;
  for (const auto& func : functions) func_names.insert(func->name_);
  for (const auto& func : functions) {
    std::function<void(const std::vector<StmtPtr>&)> walk = [&](const std::vector<StmtPtr>& stmts) {
      for (const auto& stmt : stmts) {
        CallPtr call;
        if (auto assign = std::dynamic_pointer_cast<const AssignStmt>(stmt)) {
          call = std::dynamic_pointer_cast<const Call>(assign->value_);
        } else if (auto eval = std::dynamic_pointer_cast<const EvalStmt>(stmt)) {
          call = std::dynamic_pointer_cast<const Call>(eval->expr_);
        }
        if (call) {
          auto gv = std::dynamic_pointer_cast<const GlobalVar>(call->op_);
          if (gv && func_names.count(gv->name_)) {
            callees[func->name_].insert(gv->name_);
            callers[gv->name_].insert(func->name_);
          }
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
    if (func->body_) walk(FlattenBody(func->body_));
  }
}

FunctionPtr AddGMSlotBufferParam(const FunctionPtr& func) {
  auto gm_type =
      std::make_shared<TensorType>(std::vector<int64_t>{1}, DataType::FP32, std::nullopt, std::nullopt);
  auto gm_var = std::make_shared<Var>(kGMPipeBufferName, gm_type, func->span_);
  auto new_params = func->params_;
  new_params.push_back(gm_var);
  auto new_directions = func->param_directions_;
  new_directions.push_back(ParamDirection::In);
  return std::make_shared<Function>(func->name_, new_params, new_directions, func->return_types_, func->body_,
                                    func->span_, func->func_type_, func->level_, func->role_, func->attrs_);
}

StmtPtr RewriteCallsForGMBuffer(const StmtPtr& body, const std::unordered_set<std::string>& modified_funcs,
                                const VarPtr& gm_param) {
  auto stmts = FlattenBody(body);
  std::vector<StmtPtr> new_stmts;
  bool any_changed = false;
  for (const auto& stmt : stmts) {
    auto try_rewrite = [&](const CallPtr& call) -> CallPtr {
      if (!call) return nullptr;
      auto gv = std::dynamic_pointer_cast<const GlobalVar>(call->op_);
      if (!gv || !modified_funcs.count(gv->name_)) return nullptr;
      std::vector<ExprPtr> new_args = call->args_;
      new_args.push_back(gm_param);
      return call->GetType() ? std::make_shared<Call>(call->op_, new_args, call->GetType(), call->span_)
                             : std::make_shared<Call>(call->op_, new_args, call->span_);
    };
    if (auto assign = std::dynamic_pointer_cast<const AssignStmt>(stmt)) {
      if (auto rw = try_rewrite(std::dynamic_pointer_cast<const Call>(assign->value_))) {
        new_stmts.push_back(std::make_shared<AssignStmt>(assign->var_, rw, assign->span_));
        any_changed = true;
        continue;
      }
    } else if (auto eval = std::dynamic_pointer_cast<const EvalStmt>(stmt)) {
      if (auto rw = try_rewrite(std::dynamic_pointer_cast<const Call>(eval->expr_))) {
        new_stmts.push_back(std::make_shared<EvalStmt>(rw, eval->span_));
        any_changed = true;
        continue;
      }
    }
    if (auto for_stmt = std::dynamic_pointer_cast<const ForStmt>(stmt)) {
      auto nb = RewriteCallsForGMBuffer(for_stmt->body_, modified_funcs, gm_param);
      if (nb != for_stmt->body_) {
        new_stmts.push_back(std::make_shared<ForStmt>(
            for_stmt->loop_var_, for_stmt->start_, for_stmt->stop_, for_stmt->step_, for_stmt->iter_args_, nb,
            for_stmt->return_vars_, for_stmt->span_, for_stmt->kind_, for_stmt->chunk_size_,
            for_stmt->chunk_policy_, for_stmt->loop_origin_));
        any_changed = true;
      } else {
        new_stmts.push_back(stmt);
      }
    } else if (auto if_stmt = std::dynamic_pointer_cast<const IfStmt>(stmt)) {
      auto nt = RewriteCallsForGMBuffer(if_stmt->then_body_, modified_funcs, gm_param);
      std::optional<StmtPtr> ne;
      if (if_stmt->else_body_.has_value()) {
        ne = RewriteCallsForGMBuffer(if_stmt->else_body_.value(), modified_funcs, gm_param);
      }
      bool body_changed = (nt != if_stmt->then_body_);
      if (!body_changed && ne.has_value() && if_stmt->else_body_.has_value()) {
        body_changed = (ne.value() != if_stmt->else_body_.value());
      }
      if (body_changed) {
        new_stmts.push_back(
            std::make_shared<IfStmt>(if_stmt->condition_, nt, ne, if_stmt->return_vars_, if_stmt->span_));
        any_changed = true;
      } else {
        new_stmts.push_back(stmt);
      }
    } else if (auto while_stmt = std::dynamic_pointer_cast<const WhileStmt>(stmt)) {
      auto nb = RewriteCallsForGMBuffer(while_stmt->body_, modified_funcs, gm_param);
      if (nb != while_stmt->body_) {
        new_stmts.push_back(std::make_shared<WhileStmt>(while_stmt->condition_, while_stmt->iter_args_, nb,
                                                        while_stmt->return_vars_, while_stmt->span_));
        any_changed = true;
      } else {
        new_stmts.push_back(stmt);
      }
    } else {
      new_stmts.push_back(stmt);
    }
  }
  if (!any_changed) return body;
  return SeqStmts::Flatten(std::move(new_stmts), body->span_);
}

/// Inject __gm_pipe_buffer into pipe-using functions and propagate through callers.
void InjectGMSlotBufferInPlace(std::vector<FunctionPtr>& functions) {
  if (!backend::BackendConfig::IsConfigured() ||
      backend::GetBackendType() != backend::BackendType::Ascend910B) {
    return;
  }

  std::unordered_map<std::string, std::unordered_set<std::string>> callers, callees;
  BuildCallGraphFromFunctions(functions, callers, callees);

  std::unordered_set<std::string> pipe_funcs;
  for (const auto& func : functions) {
    if (!HasGMPipeBufferParam(func) && func->body_ && HasInitializePipeOps(FlattenBody(func->body_))) {
      pipe_funcs.insert(func->name_);
    }
  }
  if (pipe_funcs.empty()) return;

  // Propagate upward
  std::unordered_set<std::string> needs_gm = pipe_funcs;
  std::vector<std::string> worklist(pipe_funcs.begin(), pipe_funcs.end());
  while (!worklist.empty()) {
    std::string name = worklist.back();
    worklist.pop_back();
    auto it = callers.find(name);
    if (it == callers.end()) continue;
    for (const auto& c : it->second) {
      if (needs_gm.insert(c).second) worklist.push_back(c);
    }
  }

  // Add params then rewrite call sites
  for (auto& func : functions) {
    if (needs_gm.count(func->name_) && !HasGMPipeBufferParam(func)) {
      func = AddGMSlotBufferParam(func);
    }
  }

  for (auto& func : functions) {
    if (!needs_gm.count(func->name_)) continue;

    VarPtr gm_param;
    for (const auto& p : func->params_) {
      if (p->name_hint_ == kGMPipeBufferName) {
        gm_param = p;
        break;
      }
    }
    INTERNAL_CHECK(gm_param) << "Internal error: " << func->name_ << " should have " << kGMPipeBufferName;

    std::unordered_set<std::string> mod_callees;
    auto ci = callees.find(func->name_);
    if (ci != callees.end()) {
      for (const auto& c : ci->second) {
        if (needs_gm.count(c)) mod_callees.insert(c);
      }
    }

    if (!mod_callees.empty()) {
      auto nb = RewriteCallsForGMBuffer(func->body_, mod_callees, gm_param);
      func = std::make_shared<Function>(func->name_, func->params_, func->param_directions_,
                                        func->return_types_, nb, func->span_, func->func_type_, func->level_,
                                        func->role_, func->attrs_);
    }
  }
}

}  // namespace

namespace pass {

Pass ExpandMixedKernel() {
  auto pass_func = [](const ProgramPtr& program) -> ProgramPtr {
    // Phase 1: Pre-scan — find InCore functions that have existing Group callers
    std::unordered_set<std::string> incore_names;
    for (const auto& [gvar, func] : program->functions_) {
      if (func->func_type_ == FunctionType::InCore) {
        incore_names.insert(func->name_);
      }
    }

    // Map InCore name -> set of Group function names that call it
    std::unordered_set<std::string> incore_with_group_caller;
    for (const auto& [gvar, func] : program->functions_) {
      if (func->func_type_ != FunctionType::Group) continue;
      for (const auto& name : incore_names) {
        if (GroupCallsFunction(func, name)) {
          incore_with_group_caller.insert(name);
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
      auto tpop_vars = CollectTpopVars(stmts);
      std::unordered_map<const Stmt*, CoreAffinity> stmt_map;
      std::unordered_map<const Var*, CoreAffinity> var_affinity;
      auto combined = AnalyzeStmtsAffinity(stmts, stmt_map, var_affinity, tpop_vars);

      // A function is mixed if the combined affinity is MIXED or BOUNDARY
      // (both imply cube+vector presence). Pure CUBE or pure VECTOR are not mixed.
      bool is_mixed = (combined == CoreAffinity::MIXED || combined == CoreAffinity::BOUNDARY);

      if (!is_mixed) {
        // Not mixed — convert InCore to the corresponding AIC or AIV type
        FunctionType new_type = (combined == CoreAffinity::CUBE) ? FunctionType::AIC : FunctionType::AIV;
        // Clear split mode — pure AIC/AIV functions don't need vector splitting
        auto attrs = func->attrs_;
        attrs.erase(
            std::remove_if(attrs.begin(), attrs.end(), [](const auto& kv) { return kv.first == "split"; }),
            attrs.end());
        auto converted = std::make_shared<Function>(func->name_, func->params_, func->param_directions_,
                                                    func->return_types_, func->body_, func->span_, new_type,
                                                    std::nullopt, std::nullopt, attrs);
        new_functions.push_back(converted);
        continue;
      }
      // Warn if split unset or is NONE — don't supported by pto-isa now.
      auto split_mode = func->GetSplitMode();
      if (!split_mode.has_value() || *split_mode == SplitMode::None) {
        LOG_ERROR << "Mixed kernel '" << func->name_ << "' use none split mode not supported by isa now; "
                  << "consider using split=pl.SplitMode.UP_DOWN on its auto_incore scope";
      }

      // Expand mixed kernel — skip Group wrapper if an existing Group caller exists
      bool has_group_caller = incore_with_group_caller.count(func->name_) > 0;
      auto expanded = ExpandMixedFunction(func, /*create_group=*/!has_group_caller);

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
        if (GroupCallsFunction(func, incore_name)) {
          func = RewriteGroupCaller(func, incore_name, info.aic_name, info.aiv_name);
        }
      }
    }

    // Phase 4: Inject GM slot buffer params for a2a3 cross-core pipe communication.
    // On Ascend910B, cross-core pipes require a GM slot buffer as intermediary.
    InjectGMSlotBufferInPlace(new_functions);

    return std::make_shared<Program>(new_functions, program->name_, program->span_);
  };

  return CreateProgramPass(pass_func, "ExpandMixedKernel", kExpandMixedKernelProperties);
}

}  // namespace pass

}  // namespace ir
}  // namespace pypto
