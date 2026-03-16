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

#include <any>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/core/any_cast.h"
#include "pypto/core/error.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/deep_clone_utils.h"
#include "pypto/ir/transforms/utils/scope_outline_utils.h"
#include "pypto/ir/type.h"
#include "pypto/ir/verifier/verifier.h"

namespace pypto {
namespace ir {

namespace {

// ============================================================================
// Core Affinity Classification
// ============================================================================

enum class CoreAffinity { CUBE, VECTOR, SHARED, MIXED, BOUNDARY };

CoreAffinity CombineAffinity(CoreAffinity a, CoreAffinity b) {
  if (a == b) return a;
  if (a == CoreAffinity::SHARED) return b;
  if (b == CoreAffinity::SHARED) return a;
  return CoreAffinity::MIXED;
}

bool IsCubeOp(const std::string& name) {
  static const std::unordered_set<std::string> cube_ops = {
      "tile.matmul",   "tile.matmul_acc", "tile.matmul_bias", "tile.gemv",
      "tile.gemv_acc", "tile.gemv_bias",  "tile.batch_matmul"};
  return cube_ops.count(name) > 0;
}

bool IsCubeMemorySpace(MemorySpace ms) { return ms != MemorySpace::DDR && ms != MemorySpace::Vec; }

/// Get target_memory from the first tile-typed argument of a Call.
/// Returns nullopt if no tile-typed Var argument is found.
std::optional<MemorySpace> GetFirstTileArgMemory(const CallPtr& call) {
  for (const auto& arg : call->args_) {
    if (auto var = std::dynamic_pointer_cast<const Var>(arg)) {
      if (auto tile_type = std::dynamic_pointer_cast<const TileType>(var->GetType())) {
        return tile_type->memory_space_;
      }
    }
  }
  return std::nullopt;
}

// ============================================================================
// CV Boundary Move Detection
// ============================================================================

enum class CVDirection { NONE, CUBE_TO_VECTOR, VECTOR_TO_CUBE };

/// Classify whether a Call is a CV-boundary tile.move.
/// Returns CUBE_TO_VECTOR if source is cube memory and target is vector memory.
/// Returns VECTOR_TO_CUBE if source is vector memory and target is cube memory.
/// Returns NONE for non-tile.move calls or same-side moves.
CVDirection ClassifyMoveDirection(const CallPtr& call) {
  if (!call || !call->op_) return CVDirection::NONE;

  auto op = std::dynamic_pointer_cast<const Op>(call->op_);
  if (!op || op->name_ != "tile.move") return CVDirection::NONE;

  auto src_memory = GetFirstTileArgMemory(call);
  if (!src_memory.has_value()) return CVDirection::NONE;

  // target_memory kwarg is always present on tile.move (ensured by InferTileTargetMemory)
  std::optional<MemorySpace> target_memory;
  for (const auto& [key, value] : call->kwargs_) {
    if (key == "target_memory") {
      target_memory = AnyCast<MemorySpace>(value, "target_memory");
      break;
    }
  }
  INTERNAL_CHECK(target_memory.has_value()) << "Internal error: tile.move missing target_memory kwarg";

  bool src_cube = IsCubeMemorySpace(src_memory.value());
  bool tgt_cube = IsCubeMemorySpace(target_memory.value());
  if (src_cube && !tgt_cube) return CVDirection::CUBE_TO_VECTOR;
  if (!src_cube && tgt_cube) return CVDirection::VECTOR_TO_CUBE;
  return CVDirection::NONE;
}

// ============================================================================
// Core Affinity Classification (call-level)
// ============================================================================

CoreAffinity ClassifyCallAffinity(const CallPtr& call) {
  if (!call || !call->op_) return CoreAffinity::SHARED;

  // GlobalVar call (function call) is SHARED
  if (std::dynamic_pointer_cast<const GlobalVar>(call->op_)) {
    return CoreAffinity::SHARED;
  }

  auto op = std::dynamic_pointer_cast<const Op>(call->op_);
  if (!op) return CoreAffinity::SHARED;

  const auto& name = op->name_;

  // Cube ops (matmul, gemv, etc.)
  if (IsCubeOp(name)) return CoreAffinity::CUBE;

  // tile.move: CV boundary moves get BOUNDARY affinity;
  // non-boundary moves are classified by source tile memory space.
  if (name == "tile.move") {
    auto dir = ClassifyMoveDirection(call);
    if (dir != CVDirection::NONE) return CoreAffinity::BOUNDARY;
    auto ms = GetFirstTileArgMemory(call);
    if (ms.has_value() && IsCubeMemorySpace(ms.value())) return CoreAffinity::CUBE;
    return CoreAffinity::VECTOR;
  }

  // tile.store, tile.reshape: classified by source tile memory space.
  static const std::unordered_set<std::string> tile_arg_classified_ops = {"tile.store", "tile.reshape"};
  if (tile_arg_classified_ops.count(name)) {
    auto ms = GetFirstTileArgMemory(call);
    if (ms.has_value() && IsCubeMemorySpace(ms.value())) return CoreAffinity::CUBE;
    return CoreAffinity::VECTOR;
  }

  // tile.load: classify by target_memory kwarg (no tile input to inspect)
  if (name == "tile.load") {
    for (const auto& [key, value] : call->kwargs_) {
      if (key == "target_memory") {
        return IsCubeMemorySpace(AnyCast<MemorySpace>(value, "target_memory")) ? CoreAffinity::CUBE
                                                                               : CoreAffinity::VECTOR;
      }
    }
    return CoreAffinity::VECTOR;  // default target_memory is Vec
  }

  // Other tile.* ops are vector
  if (name.substr(0, 5) == "tile.") return CoreAffinity::VECTOR;

  // Cross-core ops: AIV-side ops are VECTOR, AIC-side ops are CUBE.
  // reserve_buffer and import_peer_buffer are SHARED (used by both sides).
  static const std::unordered_set<std::string> vector_cross_core_ops = {
      "system.aiv_initialize_pipe", "system.tpush_to_aic", "system.tfree_to_aic"};
  if (vector_cross_core_ops.count(name)) return CoreAffinity::VECTOR;

  static const std::unordered_set<std::string> cube_cross_core_ops = {
      "system.aic_initialize_pipe", "system.tfree_to_aiv", "system.tpush_to_aiv"};
  if (cube_cross_core_ops.count(name)) return CoreAffinity::CUBE;

  return CoreAffinity::SHARED;
}

// ============================================================================
// Flatten body / make body helpers
// ============================================================================

std::vector<StmtPtr> FlattenBody(const StmtPtr& body) {
  if (auto seq = std::dynamic_pointer_cast<const SeqStmts>(body)) {
    return seq->stmts_;
  }
  return {body};
}

StmtPtr MakeBody(const std::vector<StmtPtr>& stmts, const Span& span) {
  if (stmts.empty()) return std::make_shared<SeqStmts>(std::vector<StmtPtr>{}, span);
  if (stmts.size() == 1) return stmts[0];
  return std::make_shared<SeqStmts>(stmts, span);
}

// ============================================================================
// Recursive Affinity Analysis
// ============================================================================

// Forward declare
CoreAffinity AnalyzeStmtAffinity(const StmtPtr& stmt, std::unordered_map<const Stmt*, CoreAffinity>& stmt_map,
                                 std::unordered_map<std::string, CoreAffinity>& var_affinity);

CoreAffinity AnalyzeStmtsAffinity(const std::vector<StmtPtr>& stmts,
                                  std::unordered_map<const Stmt*, CoreAffinity>& stmt_map,
                                  std::unordered_map<std::string, CoreAffinity>& var_affinity) {
  CoreAffinity combined = CoreAffinity::SHARED;
  for (const auto& stmt : stmts) {
    combined = CombineAffinity(combined, AnalyzeStmtAffinity(stmt, stmt_map, var_affinity));
  }
  return combined;
}

CoreAffinity AnalyzeStmtAffinity(const StmtPtr& stmt, std::unordered_map<const Stmt*, CoreAffinity>& stmt_map,
                                 std::unordered_map<std::string, CoreAffinity>& var_affinity) {
  CoreAffinity result = CoreAffinity::SHARED;

  if (auto assign = std::dynamic_pointer_cast<const AssignStmt>(stmt)) {
    auto call = std::dynamic_pointer_cast<const Call>(assign->value_);
    if (call) result = ClassifyCallAffinity(call);
    var_affinity[assign->var_->name_] = result;
  } else if (auto eval = std::dynamic_pointer_cast<const EvalStmt>(stmt)) {
    auto call = std::dynamic_pointer_cast<const Call>(eval->expr_);
    if (call) result = ClassifyCallAffinity(call);
  } else if (auto for_stmt = std::dynamic_pointer_cast<const ForStmt>(stmt)) {
    result = AnalyzeStmtsAffinity(FlattenBody(for_stmt->body_), stmt_map, var_affinity);
  } else if (auto if_stmt = std::dynamic_pointer_cast<const IfStmt>(stmt)) {
    result = AnalyzeStmtsAffinity(FlattenBody(if_stmt->then_body_), stmt_map, var_affinity);
    if (if_stmt->else_body_.has_value()) {
      result = CombineAffinity(
          result, AnalyzeStmtsAffinity(FlattenBody(if_stmt->else_body_.value()), stmt_map, var_affinity));
    }
  } else if (auto while_stmt = std::dynamic_pointer_cast<const WhileStmt>(stmt)) {
    result = AnalyzeStmtsAffinity(FlattenBody(while_stmt->body_), stmt_map, var_affinity);
  } else if (auto seq = std::dynamic_pointer_cast<const SeqStmts>(stmt)) {
    result = AnalyzeStmtsAffinity(seq->stmts_, stmt_map, var_affinity);
  }

  stmt_map[stmt.get()] = result;
  return result;
}

// ============================================================================
// CV Boundary Move Collection
// ============================================================================

/// Information about a single CV boundary tile.move found in the IR.
struct CVBoundaryMove {
  CVDirection direction;
  VarPtr dest_var;      // AssignStmt LHS
  ExprPtr source_tile;  // First arg of tile.move
  TypePtr result_type;  // Return type of the tile.move call
};

/// Collect all CV boundary tile.move statements recursively.
void CollectCVBoundaryMoves(const std::vector<StmtPtr>& stmts,
                            std::unordered_map<const Stmt*, CVBoundaryMove>& boundary_moves) {
  for (const auto& stmt : stmts) {
    if (auto assign = std::dynamic_pointer_cast<const AssignStmt>(stmt)) {
      auto call = std::dynamic_pointer_cast<const Call>(assign->value_);
      if (call) {
        auto dir = ClassifyMoveDirection(call);
        if (dir != CVDirection::NONE) {
          INTERNAL_CHECK(!call->args_.empty()) << "Internal error: tile.move must have at least one argument";
          boundary_moves[stmt.get()] = CVBoundaryMove{dir, assign->var_, call->args_[0], call->GetType()};
        }
      }
    }

    // Recurse into compound statements
    if (auto for_stmt = std::dynamic_pointer_cast<const ForStmt>(stmt)) {
      CollectCVBoundaryMoves(FlattenBody(for_stmt->body_), boundary_moves);
    } else if (auto if_stmt = std::dynamic_pointer_cast<const IfStmt>(stmt)) {
      CollectCVBoundaryMoves(FlattenBody(if_stmt->then_body_), boundary_moves);
      if (if_stmt->else_body_.has_value()) {
        CollectCVBoundaryMoves(FlattenBody(if_stmt->else_body_.value()), boundary_moves);
      }
    } else if (auto while_stmt = std::dynamic_pointer_cast<const WhileStmt>(stmt)) {
      CollectCVBoundaryMoves(FlattenBody(while_stmt->body_), boundary_moves);
    }
  }
}

// ============================================================================
// TPUSH / TPOP creation helpers
// ============================================================================

std::vector<std::pair<std::string, std::any>> MakeAivIdxKwargs() { return {{"aiv_idx", std::any(0)}}; }

CallPtr CreateTpush(const std::string& op_name, const ExprPtr& tile, const Span& span) {
  return OpRegistry::GetInstance().Create(op_name, {tile}, MakeAivIdxKwargs(), span);
}

/// Build a clean TileType with only shape, dtype, and memory_space (no TileView/memref).
/// tpop results should be expressible in the Python DSL without requiring TileView metadata.
TypePtr CleanTileType(const TypePtr& tile_type) {
  auto tt = std::dynamic_pointer_cast<const TileType>(tile_type);
  if (!tt) return tile_type;
  return std::make_shared<TileType>(tt->shape_, tt->dtype_, std::nullopt, std::nullopt, tt->memory_space_);
}

CallPtr CreateTpop(const std::string& op_name, const TypePtr& tile_type, const Span& span) {
  auto op = OpRegistry::GetInstance().GetOp(op_name);
  return std::make_shared<Call>(op, std::vector<ExprPtr>{}, MakeAivIdxKwargs(), CleanTileType(tile_type),
                                span);
}

// ============================================================================
// Recursive Dead Code Elimination
// ============================================================================

/// Extract the Op name from an AssignStmt or EvalStmt containing a Call.
/// Returns empty string if the statement doesn't match this pattern.
std::string GetStmtOpName(const StmtPtr& stmt) {
  CallPtr call;
  if (auto assign = std::dynamic_pointer_cast<const AssignStmt>(stmt)) {
    call = std::dynamic_pointer_cast<const Call>(assign->value_);
  } else if (auto eval = std::dynamic_pointer_cast<const EvalStmt>(stmt)) {
    call = std::dynamic_pointer_cast<const Call>(eval->expr_);
  }
  if (call && call->op_) {
    if (auto op = std::dynamic_pointer_cast<const Op>(call->op_)) {
      return op->name_;
    }
  }
  return "";
}

bool IsSideEffectOp(const StmtPtr& stmt) {
  static const std::unordered_set<std::string> side_effect_ops = {
      "system.tpush_to_aiv",  "system.tpush_to_aic", "system.tpop_from_aic",
      "system.tpop_from_aiv", "tile.store",          "tile.assemble"};
  return side_effect_ops.count(GetStmtOpName(stmt)) > 0;
}

void CollectAllAssignStmts(const std::vector<StmtPtr>& stmts,
                           std::vector<std::shared_ptr<const AssignStmt>>& assigns) {
  for (const auto& stmt : stmts) {
    if (auto assign = std::dynamic_pointer_cast<const AssignStmt>(stmt)) {
      assigns.push_back(assign);
    }
    if (auto for_stmt = std::dynamic_pointer_cast<const ForStmt>(stmt)) {
      CollectAllAssignStmts(FlattenBody(for_stmt->body_), assigns);
    } else if (auto if_stmt = std::dynamic_pointer_cast<const IfStmt>(stmt)) {
      CollectAllAssignStmts(FlattenBody(if_stmt->then_body_), assigns);
      if (if_stmt->else_body_.has_value()) {
        CollectAllAssignStmts(FlattenBody(if_stmt->else_body_.value()), assigns);
      }
    } else if (auto while_stmt = std::dynamic_pointer_cast<const WhileStmt>(stmt)) {
      CollectAllAssignStmts(FlattenBody(while_stmt->body_), assigns);
    }
  }
}

void FindLiveRootsRecursive(const std::vector<StmtPtr>& stmts, std::unordered_set<std::string>& live) {
  for (const auto& stmt : stmts) {
    if (std::dynamic_pointer_cast<const ReturnStmt>(stmt) ||
        std::dynamic_pointer_cast<const YieldStmt>(stmt) || IsSideEffectOp(stmt)) {
      outline_utils::VarRefCollector refs;
      refs.VisitStmt(stmt);
      live.insert(refs.var_refs.begin(), refs.var_refs.end());
      // Mark LHS of side-effect assignments as live for downstream propagation
      if (auto assign = std::dynamic_pointer_cast<const AssignStmt>(stmt)) {
        live.insert(assign->var_->name_);
      }
    }
    // Collect variable refs from control expressions and iter_args init values
    auto collect_iter_arg_refs = [&](const auto& loop_stmt) {
      for (const auto& iter_arg : loop_stmt->iter_args_) {
        outline_utils::VarRefCollector refs;
        refs.VisitExpr(iter_arg->initValue_);
        live.insert(refs.var_refs.begin(), refs.var_refs.end());
      }
    };
    auto collect_expr_refs = [&](const ExprPtr& expr) {
      outline_utils::VarRefCollector refs;
      refs.VisitExpr(expr);
      live.insert(refs.var_refs.begin(), refs.var_refs.end());
    };

    if (auto for_stmt = std::dynamic_pointer_cast<const ForStmt>(stmt)) {
      collect_expr_refs(for_stmt->start_);
      collect_expr_refs(for_stmt->stop_);
      collect_expr_refs(for_stmt->step_);
      collect_iter_arg_refs(for_stmt);
      FindLiveRootsRecursive(FlattenBody(for_stmt->body_), live);
    } else if (auto if_stmt = std::dynamic_pointer_cast<const IfStmt>(stmt)) {
      collect_expr_refs(if_stmt->condition_);
      FindLiveRootsRecursive(FlattenBody(if_stmt->then_body_), live);
      if (if_stmt->else_body_.has_value()) {
        FindLiveRootsRecursive(FlattenBody(if_stmt->else_body_.value()), live);
      }
    } else if (auto while_stmt = std::dynamic_pointer_cast<const WhileStmt>(stmt)) {
      collect_expr_refs(while_stmt->condition_);
      collect_iter_arg_refs(while_stmt);
      FindLiveRootsRecursive(FlattenBody(while_stmt->body_), live);
    }
  }
}

std::vector<StmtPtr> FilterDeadCode(const std::vector<StmtPtr>& stmts,
                                    const std::unordered_set<std::string>& live) {
  std::vector<StmtPtr> result;
  for (const auto& stmt : stmts) {
    if (auto assign = std::dynamic_pointer_cast<const AssignStmt>(stmt)) {
      if (live.count(assign->var_->name_) || IsSideEffectOp(stmt)) {
        result.push_back(stmt);
      }
    } else if (auto for_stmt = std::dynamic_pointer_cast<const ForStmt>(stmt)) {
      auto filtered = FilterDeadCode(FlattenBody(for_stmt->body_), live);
      result.push_back(std::make_shared<ForStmt>(
          for_stmt->loop_var_, for_stmt->start_, for_stmt->stop_, for_stmt->step_, for_stmt->iter_args_,
          MakeBody(filtered, for_stmt->span_), for_stmt->return_vars_, for_stmt->span_, for_stmt->kind_,
          for_stmt->chunk_size_, for_stmt->chunk_policy_, for_stmt->loop_origin_));
    } else if (auto if_stmt = std::dynamic_pointer_cast<const IfStmt>(stmt)) {
      auto filtered_then = FilterDeadCode(FlattenBody(if_stmt->then_body_), live);
      std::optional<StmtPtr> filtered_else;
      if (if_stmt->else_body_.has_value()) {
        auto fe = FilterDeadCode(FlattenBody(if_stmt->else_body_.value()), live);
        filtered_else = MakeBody(fe, if_stmt->span_);
      }
      result.push_back(std::make_shared<IfStmt>(if_stmt->condition_, MakeBody(filtered_then, if_stmt->span_),
                                                filtered_else, if_stmt->return_vars_, if_stmt->span_));
    } else if (auto while_stmt = std::dynamic_pointer_cast<const WhileStmt>(stmt)) {
      auto filtered = FilterDeadCode(FlattenBody(while_stmt->body_), live);
      result.push_back(std::make_shared<WhileStmt>(while_stmt->condition_, while_stmt->iter_args_,
                                                   MakeBody(filtered, while_stmt->span_),
                                                   while_stmt->return_vars_, while_stmt->span_));
    } else {
      // ReturnStmt, EvalStmt (side-effect), etc. — always keep
      result.push_back(stmt);
    }
  }
  return result;
}

std::vector<StmtPtr> EliminateDeadCode(const std::vector<StmtPtr>& stmts) {
  std::unordered_set<std::string> live;

  // Find initial live set from returns and side-effect ops at all nesting levels
  FindLiveRootsRecursive(stmts, live);

  // Collect all assignments at all nesting levels for backward propagation
  std::vector<std::shared_ptr<const AssignStmt>> all_assigns;
  CollectAllAssignStmts(stmts, all_assigns);

  // Backward pass: propagate liveness
  bool changed = true;
  while (changed) {
    changed = false;
    for (auto it = all_assigns.rbegin(); it != all_assigns.rend(); ++it) {
      if (!live.count((*it)->var_->name_)) continue;

      outline_utils::VarRefCollector refs;
      refs.VisitExpr((*it)->value_);
      for (const auto& ref : refs.var_refs) {
        if (!live.count(ref)) {
          live.insert(ref);
          changed = true;
        }
      }
    }
  }

  return FilterDeadCode(stmts, live);
}

// ============================================================================
// Parameterized Core Body Builder (shared by AIC and AIV)
// ============================================================================

enum class CoreSide { AIC, AIV };

/// Build the body for one core side (AIC or AIV), filtering statements by affinity
/// and replacing CV boundary moves with TPUSH/TPOP ops.
/// tpop_var_remap collects original dest_var pointer -> clean-typed new_var mappings
/// for tpop dest vars, so downstream references can be updated via pointer-based substitution.
std::vector<StmtPtr> BuildCoreBody(CoreSide side, const std::vector<StmtPtr>& stmts,
                                   const std::unordered_map<const Stmt*, CoreAffinity>& stmt_map,
                                   const std::unordered_map<const Stmt*, CVBoundaryMove>& boundary_moves,
                                   std::unordered_map<const Var*, VarPtr>& tpop_var_remap) {
  // AIC keeps CUBE, skips VECTOR; AIV keeps VECTOR, skips CUBE
  CoreAffinity keep_affinity = (side == CoreSide::AIC) ? CoreAffinity::CUBE : CoreAffinity::VECTOR;
  CoreAffinity skip_affinity = (side == CoreSide::AIC) ? CoreAffinity::VECTOR : CoreAffinity::CUBE;

  // For boundary moves: the "push" side sends data, the "pop" side receives it.
  // AIC: C→V = push to AIV, V→C = pop from AIV
  // AIV: C→V = pop from AIC, V→C = push to AIC
  std::string push_op = (side == CoreSide::AIC) ? "system.tpush_to_aiv" : "system.tpush_to_aic";
  std::string pop_op = (side == CoreSide::AIC) ? "system.tpop_from_aiv" : "system.tpop_from_aic";
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
          result.push_back(
              std::make_shared<EvalStmt>(CreateTpush(push_op, bm.source_tile, stmt->span_), stmt->span_));
        } else {
          // Use the dest_var's type (which has memory_space from infer_tile_memory_space)
          // as the source, then strip TileView so the type is DSL-expressible.
          auto clean_type = CleanTileType(bm.dest_var->GetType());
          auto clean_var = std::make_shared<Var>(bm.dest_var->name_, clean_type, stmt->span_);
          tpop_var_remap[bm.dest_var.get()] = clean_var;
          result.push_back(std::make_shared<AssignStmt>(
              clean_var, CreateTpop(pop_op, clean_type, stmt->span_), stmt->span_));
        }
        continue;
      }
      // Compound stmt whose children are all BOUNDARY — recurse like MIXED
      affinity = CoreAffinity::MIXED;
    }

    if (affinity == skip_affinity) continue;

    if (affinity == keep_affinity || affinity == CoreAffinity::SHARED) {
      result.push_back(stmt);
    } else if (affinity == CoreAffinity::MIXED) {
      // Recurse into compound statements, building pruned copies
      if (auto for_stmt = std::dynamic_pointer_cast<const ForStmt>(stmt)) {
        auto new_body =
            BuildCoreBody(side, FlattenBody(for_stmt->body_), stmt_map, boundary_moves, tpop_var_remap);
        result.push_back(std::make_shared<ForStmt>(
            for_stmt->loop_var_, for_stmt->start_, for_stmt->stop_, for_stmt->step_, for_stmt->iter_args_,
            MakeBody(new_body, for_stmt->span_), for_stmt->return_vars_, for_stmt->span_, for_stmt->kind_,
            for_stmt->chunk_size_, for_stmt->chunk_policy_, for_stmt->loop_origin_));
      } else if (auto if_stmt = std::dynamic_pointer_cast<const IfStmt>(stmt)) {
        auto new_then =
            BuildCoreBody(side, FlattenBody(if_stmt->then_body_), stmt_map, boundary_moves, tpop_var_remap);
        std::optional<StmtPtr> new_else;
        if (if_stmt->else_body_.has_value()) {
          auto new_else_stmts = BuildCoreBody(side, FlattenBody(if_stmt->else_body_.value()), stmt_map,
                                              boundary_moves, tpop_var_remap);
          new_else = MakeBody(new_else_stmts, if_stmt->span_);
        }
        result.push_back(std::make_shared<IfStmt>(if_stmt->condition_, MakeBody(new_then, if_stmt->span_),
                                                  new_else, if_stmt->return_vars_, if_stmt->span_));
      } else if (auto while_stmt = std::dynamic_pointer_cast<const WhileStmt>(stmt)) {
        auto new_body =
            BuildCoreBody(side, FlattenBody(while_stmt->body_), stmt_map, boundary_moves, tpop_var_remap);
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

  // Recursive affinity analysis (descends into ForStmt/IfStmt/WhileStmt)
  std::unordered_map<const Stmt*, CoreAffinity> stmt_map;
  std::unordered_map<std::string, CoreAffinity> var_affinity;
  AnalyzeStmtsAffinity(stmts, stmt_map, var_affinity);

  // Collect CV boundary moves from explicit tile.move ops
  std::unordered_map<const Stmt*, CVBoundaryMove> boundary_moves;
  CollectCVBoundaryMoves(stmts, boundary_moves);

  // Build AIC body (recursive — handles MIXED compound stmts)
  std::unordered_map<const Var*, VarPtr> aic_tpop_remap;
  auto aic_stmts = BuildCoreBody(CoreSide::AIC, stmts, stmt_map, boundary_moves, aic_tpop_remap);

  // Remove ReturnStmt from AIC (AIC doesn't return values)
  std::vector<StmtPtr> aic_stmts_no_return;
  for (const auto& s : aic_stmts) {
    if (!std::dynamic_pointer_cast<const ReturnStmt>(s)) {
      aic_stmts_no_return.push_back(s);
    }
  }
  // DCE on AIC (recursive)
  auto aic_final = EliminateDeadCode(aic_stmts_no_return);

  // Build AIV body (recursive — handles MIXED compound stmts)
  std::unordered_map<const Var*, VarPtr> aiv_tpop_remap;
  auto aiv_stmts = BuildCoreBody(CoreSide::AIV, stmts, stmt_map, boundary_moves, aiv_tpop_remap);
  // DCE on AIV (recursive)
  auto aiv_final = EliminateDeadCode(aiv_stmts);

  // Helper to create fresh params and build a DeepClone var_map
  auto make_param_map = [&]() {
    std::unordered_map<const Var*, ExprPtr> param_map;
    std::vector<VarPtr> fresh_params;
    for (const auto& var : func->params_) {
      auto fresh = std::make_shared<Var>(var->name_, var->GetType(), func->span_);
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
  std::string aic_name = func->name_ + "_aic";
  auto [aic_params, aic_map] = make_param_map();
  seed_tpop_remap(aic_map, aic_tpop_remap);
  auto [aic_cloned_body, aic_clone_map_unused] = DeepClone(MakeBody(aic_final, func->span_), aic_map);
  (void)aic_clone_map_unused;
  auto aic_func =
      std::make_shared<Function>(aic_name, aic_params, func->param_directions_, std::vector<TypePtr>{},
                                 aic_cloned_body, func->span_, FunctionType::AIC);

  // Create AIV function with deep clone (fresh Vars for all params and locals,
  // ensuring no shared Var pointers with AIC for structural equality)
  std::string aiv_name = func->name_ + "_aiv";
  auto [aiv_params, aiv_map] = make_param_map();
  seed_tpop_remap(aiv_map, aiv_tpop_remap);
  auto [aiv_cloned_body, aiv_clone_map_unused] = DeepClone(MakeBody(aiv_final, func->span_), aiv_map);
  (void)aiv_clone_map_unused;
  auto aiv_func =
      std::make_shared<Function>(aiv_name, aiv_params, func->param_directions_, func->return_types_,
                                 aiv_cloned_body, func->span_, FunctionType::AIV);

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

  auto group_body = std::make_shared<SeqStmts>(group_stmts, func->span_);
  auto group_func =
      std::make_shared<Function>(group_name, group_params, func->param_directions_, func->return_types_,
                                 group_body, func->span_, FunctionType::Group);

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

  auto new_body = std::make_shared<SeqStmts>(new_stmts, group_func->span_);
  return std::make_shared<Function>(group_func->name_, group_func->params_, group_func->param_directions_,
                                    group_func->return_types_, new_body, group_func->span_,
                                    FunctionType::Group);
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
      std::unordered_map<const Stmt*, CoreAffinity> stmt_map;
      std::unordered_map<std::string, CoreAffinity> var_affinity;
      auto combined = AnalyzeStmtsAffinity(stmts, stmt_map, var_affinity);

      // A function is mixed if the combined affinity is MIXED or BOUNDARY
      // (both imply cube+vector presence). Pure CUBE or pure VECTOR are not mixed.
      bool is_mixed = (combined == CoreAffinity::MIXED || combined == CoreAffinity::BOUNDARY);

      if (!is_mixed) {
        // Not mixed — convert InCore to the corresponding AIC or AIV type
        FunctionType new_type = (combined == CoreAffinity::CUBE) ? FunctionType::AIC : FunctionType::AIV;
        auto converted = std::make_shared<Function>(func->name_, func->params_, func->param_directions_,
                                                    func->return_types_, func->body_, func->span_, new_type);
        new_functions.push_back(converted);
        continue;
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

    return std::make_shared<Program>(new_functions, program->name_, program->span_);
  };

  return CreateProgramPass(pass_func, "ExpandMixedKernel", kExpandMixedKernelProperties);
}

}  // namespace pass

// ============================================================================
// MixedKernelExpanded property verifier
// ============================================================================

namespace {

class MixedKernelExpandedVerifier : public IRVisitor {
 public:
  explicit MixedKernelExpandedVerifier(std::vector<Diagnostic>& diagnostics, std::string func_name)
      : diagnostics_(diagnostics), func_name_(std::move(func_name)) {}

  void VisitExpr_(const CallPtr& op) override {
    if (!op || !op->op_) {
      IRVisitor::VisitExpr_(op);
      return;
    }
    auto affinity = ClassifyCallAffinity(op);
    if (affinity == CoreAffinity::CUBE) {
      has_cube_ = true;
    } else if (affinity == CoreAffinity::VECTOR) {
      has_vector_ = true;
    } else if (affinity == CoreAffinity::BOUNDARY) {
      has_cube_ = true;
      has_vector_ = true;
    }
    IRVisitor::VisitExpr_(op);
  }

  void CheckResult() {
    if (has_cube_ && has_vector_) {
      diagnostics_.emplace_back(DiagnosticSeverity::Error, "MixedKernelExpanded", 0,
                                "InCore function '" + func_name_ +
                                    "' contains both Cube and Vector tile ops (should have been expanded)",
                                Span::unknown());
    }
  }

 private:
  std::vector<Diagnostic>& diagnostics_;
  std::string func_name_;
  bool has_cube_ = false;
  bool has_vector_ = false;
};

}  // namespace

class MixedKernelExpandedPropertyVerifierImpl : public PropertyVerifier {
 public:
  [[nodiscard]] std::string GetName() const override { return "MixedKernelExpanded"; }

  void Verify(const ProgramPtr& program, std::vector<Diagnostic>& diagnostics) override {
    if (!program) return;
    for (const auto& [gv, func] : program->functions_) {
      if (!func || !func->body_) continue;
      // Only check InCore functions (AIC/AIV are already split)
      if (func->func_type_ != FunctionType::InCore) continue;
      MixedKernelExpandedVerifier verifier(diagnostics, func->name_);
      verifier.VisitStmt(func->body_);
      verifier.CheckResult();
    }
  }
};

PropertyVerifierPtr CreateMixedKernelExpandedPropertyVerifier() {
  return std::make_shared<MixedKernelExpandedPropertyVerifierImpl>();
}

}  // namespace ir
}  // namespace pypto
