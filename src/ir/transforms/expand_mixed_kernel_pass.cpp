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

#include "pypto/core/any_cast.h"
#include "pypto/core/error.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/deep_clone_utils.h"
#include "pypto/ir/transforms/utils/scope_outline_utils.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
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

  // Cross-core tile ops: must be checked before the generic tile.* branch.
  // AIV-side ops are VECTOR, AIC-side ops are CUBE.
  static const std::unordered_set<std::string> vector_cross_core_ops = {
      "system.aiv_initialize_pipe", "system.tpush_to_aic", "system.tfree_to_aic", "tile.tpush_to_aic",
      "tile.tpop_from_aic"};
  if (vector_cross_core_ops.count(name)) return CoreAffinity::VECTOR;

  static const std::unordered_set<std::string> cube_cross_core_ops = {
      "system.aic_initialize_pipe", "system.tfree_to_aiv", "system.tpush_to_aiv", "tile.tpush_to_aiv",
      "tile.tpop_from_aiv"};
  if (cube_cross_core_ops.count(name)) return CoreAffinity::CUBE;

  // Other tile.* ops are vector
  if (name.substr(0, 5) == "tile.") return CoreAffinity::VECTOR;

  return CoreAffinity::SHARED;
}

// ============================================================================
// Flatten body / make body helpers
// ============================================================================

// Use the shared utility; local alias preserves call sites.
const auto& FlattenBody = transform_utils::FlattenToStmts;

StmtPtr MakeBody(const std::vector<StmtPtr>& stmts, const Span& span) {
  if (stmts.empty()) return std::make_shared<SeqStmts>(std::vector<StmtPtr>{}, span);
  if (stmts.size() == 1) return stmts[0];
  return std::make_shared<SeqStmts>(stmts, span);
}

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
                result = IsCubeMemorySpace(target) ? CoreAffinity::CUBE : CoreAffinity::VECTOR;
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

/// Information about a single CV boundary tile.move found in the IR.
struct CVBoundaryMove {
  CVDirection direction;
  VarPtr dest_var;      // AssignStmt LHS
  ExprPtr source_tile;  // First arg of tile.move
  TypePtr result_type;  // Return type of the tile.move call
  /// Original tpop Call that defines source_tile (nullptr if source is not a tpop result).
  /// Used to propagate kwargs (e.g., split) from the user-written tpop to the replacement tpop.
  CallPtr source_tpop_call;
};

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

/// Build a clean TileType with only shape, dtype, and memory_space (no TileView/memref).
/// tpop results should be expressible in the Python DSL without requiring TileView metadata.
TypePtr CleanTileType(const TypePtr& tile_type) {
  auto tt = std::dynamic_pointer_cast<const TileType>(tile_type);
  if (!tt) return tile_type;
  return std::make_shared<TileType>(tt->shape_, tt->dtype_, std::nullopt, std::nullopt, tt->memory_space_);
}

CallPtr CreateTpop(const std::string& op_name, const TypePtr& tile_type, const Span& span,
                   const std::vector<std::pair<std::string, std::any>>& kwargs = {}) {
  auto op = OpRegistry::GetInstance().GetOp(op_name);
  auto effective_kwargs = kwargs.empty() ? MakeSplitKwargs() : kwargs;
  return std::make_shared<Call>(op, std::vector<ExprPtr>{}, std::move(effective_kwargs),
                                CleanTileType(tile_type), span);
}

CallPtr CreateMove(const ExprPtr& tile, MemorySpace target_memory, const TypePtr& result_type,
                   const Span& span) {
  auto op = OpRegistry::GetInstance().GetOp("tile.move");
  std::vector<std::pair<std::string, std::any>> kwargs{{"target_memory", std::any(target_memory)}};
  return std::make_shared<Call>(op, std::vector<ExprPtr>{tile}, std::move(kwargs), result_type, span);
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
  static const std::unordered_set<std::string> side_effect_ops = {"tile.tpush_to_aiv",  "tile.tpush_to_aic",
                                                                  "tile.tpop_from_aic", "tile.tpop_from_aiv",
                                                                  "tile.store",         "tile.assemble"};
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

void FindLiveRootsRecursive(const std::vector<StmtPtr>& stmts, std::unordered_set<const Var*>& live) {
  for (const auto& stmt : stmts) {
    if (std::dynamic_pointer_cast<const ReturnStmt>(stmt) ||
        std::dynamic_pointer_cast<const YieldStmt>(stmt) || IsSideEffectOp(stmt)) {
      outline_utils::VarRefCollector refs;
      refs.VisitStmt(stmt);
      live.insert(refs.var_refs.begin(), refs.var_refs.end());
      // Mark LHS of side-effect assignments as live for downstream propagation
      if (auto assign = std::dynamic_pointer_cast<const AssignStmt>(stmt)) {
        live.insert(assign->var_.get());
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
                                    const std::unordered_set<const Var*>& live) {
  std::vector<StmtPtr> result;
  for (const auto& stmt : stmts) {
    if (auto assign = std::dynamic_pointer_cast<const AssignStmt>(stmt)) {
      if (live.count(assign->var_.get()) || IsSideEffectOp(stmt)) {
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
  std::unordered_set<const Var*> live;

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
      if (!live.count((*it)->var_.get())) continue;

      outline_utils::VarRefCollector refs;
      refs.VisitExpr((*it)->value_);
      for (const Var* ref : refs.var_refs) {
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
// Loop Iter_arg Cleanup (strip dead, fix dangling references)
// ============================================================================

// --- Reconstruction helpers (reduce repetitive loop/if rebuilds) ---

/// Rebuild a ForStmt with a new body, preserving all other fields.
StmtPtr RebuildForStmt(const std::shared_ptr<const ForStmt>& f, const StmtPtr& new_body) {
  return std::make_shared<ForStmt>(f->loop_var_, f->start_, f->stop_, f->step_, f->iter_args_, new_body,
                                   f->return_vars_, f->span_, f->kind_, f->chunk_size_, f->chunk_policy_,
                                   f->loop_origin_);
}

/// Rebuild a ForStmt with new iter_args, body, and return_vars.
StmtPtr RebuildForStmt(const std::shared_ptr<const ForStmt>& f, const std::vector<IterArgPtr>& iter_args,
                       const StmtPtr& new_body, const std::vector<VarPtr>& return_vars) {
  return std::make_shared<ForStmt>(f->loop_var_, f->start_, f->stop_, f->step_, iter_args, new_body,
                                   return_vars, f->span_, f->kind_, f->chunk_size_, f->chunk_policy_,
                                   f->loop_origin_);
}

/// Rebuild a WhileStmt with a new body, preserving all other fields.
StmtPtr RebuildWhileStmt(const std::shared_ptr<const WhileStmt>& w, const StmtPtr& new_body) {
  return std::make_shared<WhileStmt>(w->condition_, w->iter_args_, new_body, w->return_vars_, w->span_);
}

/// Rebuild a WhileStmt with new iter_args, body, and return_vars.
StmtPtr RebuildWhileStmt(const std::shared_ptr<const WhileStmt>& w, const std::vector<IterArgPtr>& iter_args,
                         const StmtPtr& new_body, const std::vector<VarPtr>& return_vars) {
  return std::make_shared<WhileStmt>(w->condition_, iter_args, new_body, return_vars, w->span_);
}

/// Rebuild an IfStmt with new then/else bodies, preserving condition and return_vars.
StmtPtr RebuildIfStmt(const std::shared_ptr<const IfStmt>& s, const std::vector<StmtPtr>& new_then,
                      const std::optional<std::vector<StmtPtr>>& new_else_stmts) {
  std::optional<StmtPtr> new_else;
  if (new_else_stmts.has_value()) {
    new_else = MakeBody(new_else_stmts.value(), s->span_);
  }
  return std::make_shared<IfStmt>(s->condition_, MakeBody(new_then, s->span_), new_else, s->return_vars_,
                                  s->span_);
}

/// Process an IfStmt's else branch: flatten, apply transform, return processed stmts.
/// Returns nullopt if the IfStmt has no else branch.
template <typename Fn>
std::optional<std::vector<StmtPtr>> ProcessElseBranch(const std::shared_ptr<const IfStmt>& if_stmt,
                                                      Fn&& transform) {
  if (!if_stmt->else_body_.has_value()) return std::nullopt;
  return transform(FlattenBody(if_stmt->else_body_.value()));
}

/// Apply a function to the last statement of a compound stmt tree (IfStmt/SeqStmts).
/// The transform function may return nullptr to signal removal of the last statement.
/// Used by FilterYieldStmt (nullable) and FixDanglingYieldStmt (always non-null).
template <typename Fn>
StmtPtr TransformLastStmt(const StmtPtr& stmt, Fn&& transform) {
  // Helper: apply transform to the back of a statement list, removing if null
  auto apply_to_back = [&](std::vector<StmtPtr>& stmts) {
    if (stmts.empty()) return;
    auto result = TransformLastStmt(stmts.back(), transform);
    if (result) {
      stmts.back() = result;
    } else {
      stmts.pop_back();
    }
  };

  if (auto if_stmt = std::dynamic_pointer_cast<const IfStmt>(stmt)) {
    auto then_stmts = FlattenBody(if_stmt->then_body_);
    apply_to_back(then_stmts);
    auto else_stmts = ProcessElseBranch(if_stmt, [&](std::vector<StmtPtr> es) {
      apply_to_back(es);
      return es;
    });
    return RebuildIfStmt(if_stmt, then_stmts, else_stmts);
  }

  if (auto seq = std::dynamic_pointer_cast<const SeqStmts>(stmt)) {
    auto seq_stmts = seq->stmts_;
    apply_to_back(seq_stmts);
    return MakeBody(seq_stmts, seq->span_);
  }

  return transform(stmt);
}

// --- Variable reference collection helpers ---

/// Collect variable references from body statements, skipping top-level YieldStmt.
/// This determines which iter_arg variables are actually used by computation
/// statements, excluding the yield that merely carries values between iterations.
/// NOTE: Yields nested inside IfStmt/SeqStmts are still visited (conservative --
/// may keep iter_args alive that are only referenced in conditional yields).
/// This is safe: over-conservative keeps correctness, and such patterns are rare.
void CollectBodyRefsSkippingYield(const std::vector<StmtPtr>& stmts, std::unordered_set<const Var*>& refs) {
  for (const auto& stmt : stmts) {
    if (std::dynamic_pointer_cast<const YieldStmt>(stmt)) continue;
    outline_utils::VarRefCollector collector;
    collector.VisitStmt(stmt);
    refs.insert(collector.var_refs.begin(), collector.var_refs.end());
  }
}

/// Recursively filter YieldStmt values by kept_indices.
/// Returns nullptr if all yield values are stripped and the yield should be removed.
StmtPtr FilterYieldStmt(const StmtPtr& stmt, const std::vector<size_t>& kept_indices) {
  return TransformLastStmt(stmt, [&](const StmtPtr& s) -> StmtPtr {
    auto yield_stmt = std::dynamic_pointer_cast<const YieldStmt>(s);
    if (!yield_stmt) return s;
    if (kept_indices.empty()) return nullptr;
    std::vector<ExprPtr> new_values;
    for (size_t idx : kept_indices) {
      INTERNAL_CHECK(idx < yield_stmt->value_.size())
          << "Internal error: yield index " << idx << " out of range " << yield_stmt->value_.size();
      new_values.push_back(yield_stmt->value_[idx]);
    }
    return std::make_shared<YieldStmt>(new_values, yield_stmt->span_);
  });
}

/// Rebuild a For or While loop with the given iter_args, body, and return_vars.
StmtPtr RebuildLoop(const std::shared_ptr<const ForStmt>& for_stmt,
                    const std::shared_ptr<const WhileStmt>& while_stmt,
                    const std::vector<IterArgPtr>& iter_args, const StmtPtr& new_body,
                    const std::vector<VarPtr>& return_vars) {
  if (for_stmt) return RebuildForStmt(for_stmt, iter_args, new_body, return_vars);
  return RebuildWhileStmt(while_stmt, iter_args, new_body, return_vars);
}

/// Strip dead iter_args from ForStmt/WhileStmt in the given statement list.
/// An iter_arg is dead if:
///   1. Its variable name is not referenced by any non-yield statement in the loop body, AND
///   2. Its corresponding return_var is not referenced by any statement after the loop.
/// Dead iter_args, their return_vars, and corresponding yield values are removed.
std::vector<StmtPtr> StripDeadIterArgs(const std::vector<StmtPtr>& stmts) {
  // Precompute suffix reference sets (back-to-front) to avoid O(N^2) re-scanning.
  // suffix_refs[i] contains all variable references from stmts[i+1..end].
  std::vector<std::unordered_set<const Var*>> suffix_refs(stmts.size());
  for (size_t i = stmts.size(); i-- > 0;) {
    if (i + 1 < stmts.size()) {
      suffix_refs[i] = suffix_refs[i + 1];
    }
    outline_utils::VarRefCollector collector;
    collector.VisitStmt(stmts[i]);
    suffix_refs[i].insert(collector.var_refs.begin(), collector.var_refs.end());
  }

  std::vector<StmtPtr> result;

  for (size_t idx = 0; idx < stmts.size(); ++idx) {
    auto for_stmt = std::dynamic_pointer_cast<const ForStmt>(stmts[idx]);
    auto while_stmt = std::dynamic_pointer_cast<const WhileStmt>(stmts[idx]);

    // For non-loop statements, recurse into compound children
    if (!for_stmt && !while_stmt) {
      if (auto if_stmt = std::dynamic_pointer_cast<const IfStmt>(stmts[idx])) {
        auto new_then = StripDeadIterArgs(FlattenBody(if_stmt->then_body_));
        auto new_else =
            ProcessElseBranch(if_stmt, [](const std::vector<StmtPtr>& es) { return StripDeadIterArgs(es); });
        result.push_back(RebuildIfStmt(if_stmt, new_then, new_else));
      } else {
        result.push_back(stmts[idx]);
      }
      continue;
    }

    const auto& iter_args = for_stmt ? for_stmt->iter_args_ : while_stmt->iter_args_;
    const auto& return_vars = for_stmt ? for_stmt->return_vars_ : while_stmt->return_vars_;
    const auto& body = for_stmt ? for_stmt->body_ : while_stmt->body_;
    const auto& span = for_stmt ? for_stmt->span_ : while_stmt->span_;

    // Recursively process the loop body first (bottom-up)
    auto processed_body = StripDeadIterArgs(FlattenBody(body));

    // No iter_args — just rebuild with processed body
    if (iter_args.empty()) {
      result.push_back(
          RebuildLoop(for_stmt, while_stmt, iter_args, MakeBody(processed_body, span), return_vars));
      continue;
    }

    // Collect var refs from processed loop body, excluding top-level YieldStmt
    std::unordered_set<const Var*> body_refs;
    CollectBodyRefsSkippingYield(processed_body, body_refs);

    // O(1) lookup into precomputed suffix refs for statements after this loop
    static const std::unordered_set<const Var*> kEmptyRefs;
    const auto& after_refs = (idx + 1 < stmts.size()) ? suffix_refs[idx + 1] : kEmptyRefs;

    // Determine which iter_args are live
    std::vector<size_t> kept_indices;
    for (size_t i = 0; i < iter_args.size(); ++i) {
      bool used_in_body = body_refs.count(iter_args[i].get()) > 0;
      bool return_var_used = i < return_vars.size() && after_refs.count(return_vars[i].get()) > 0;
      if (used_in_body || return_var_used) {
        kept_indices.push_back(i);
      }
    }

    // Filter iter_args and return_vars by kept_indices (identity when all alive)
    std::vector<IterArgPtr> new_iter_args;
    std::vector<VarPtr> new_return_vars;
    for (size_t i : kept_indices) {
      new_iter_args.push_back(iter_args[i]);
      if (i < return_vars.size()) {
        new_return_vars.push_back(return_vars[i]);
      }
    }

    // Filter YieldStmt values when some iter_args were stripped
    if (kept_indices.size() < iter_args.size() && !processed_body.empty()) {
      auto filtered_last = FilterYieldStmt(processed_body.back(), kept_indices);
      if (filtered_last) {
        processed_body.back() = filtered_last;
      } else {
        processed_body.pop_back();
      }
    }

    result.push_back(
        RebuildLoop(for_stmt, while_stmt, new_iter_args, MakeBody(processed_body, span), new_return_vars));
  }

  return result;
}

/// Build a map from variable name to its defining AssignStmt in the original body.
/// Only collects top-level assignments (init values are typically at the top level).
void BuildDefMap(const std::vector<StmtPtr>& stmts, std::unordered_map<const Var*, StmtPtr>& def_map) {
  for (const auto& stmt : stmts) {
    if (auto assign = std::dynamic_pointer_cast<const AssignStmt>(stmt)) {
      def_map[assign->var_.get()] = stmt;
    }
  }
}

/// Recursively collect the definition chain for a variable from the def_map.
/// Returns definitions in dependency order (dependencies first).
void PullDefinitionChain(const Var* var_ptr, const std::unordered_map<const Var*, StmtPtr>& def_map,
                         const std::unordered_set<const Var*>& already_defined,
                         std::unordered_set<const Var*>& pulled, std::vector<StmtPtr>& out) {
  if (pulled.count(var_ptr) || already_defined.count(var_ptr)) return;
  auto it = def_map.find(var_ptr);
  if (it == def_map.end()) return;

  pulled.insert(var_ptr);

  // Recursively pull dependencies first
  auto assign = std::dynamic_pointer_cast<const AssignStmt>(it->second);
  if (assign) {
    outline_utils::VarRefCollector refs;
    refs.VisitExpr(assign->value_);
    for (const Var* dep : refs.var_refs) {
      PullDefinitionChain(dep, def_map, already_defined, pulled, out);
    }
  }

  out.push_back(it->second);
}

/// Fix alive iter_args whose init values reference undefined variables.
/// Pulls the missing definitions from the original (pre-split) body.
/// Uses a prefix-only `defined_so_far` set (variables defined before the current
/// statement) to avoid treating non-dominating definitions as available.
std::vector<StmtPtr> FixupIterArgInitValues(const std::vector<StmtPtr>& stmts,
                                            const std::unordered_map<const Var*, StmtPtr>& original_def_map) {
  auto recurse = [&](const std::vector<StmtPtr>& s) { return FixupIterArgInitValues(s, original_def_map); };

  // Build prefix-only defined set: track definitions as we scan statements in order.
  std::unordered_set<const Var*> defined_so_far;
  std::vector<StmtPtr> result;
  std::unordered_set<const Var*> pulled;

  for (const auto& stmt : stmts) {
    auto for_stmt = std::dynamic_pointer_cast<const ForStmt>(stmt);
    auto while_stmt = std::dynamic_pointer_cast<const WhileStmt>(stmt);

    // Pull missing init value definitions for loop iter_args
    const std::vector<IterArgPtr>* iter_args_ptr = nullptr;
    if (for_stmt) {
      iter_args_ptr = &for_stmt->iter_args_;
    } else if (while_stmt) {
      iter_args_ptr = &while_stmt->iter_args_;
    }
    if (iter_args_ptr && !iter_args_ptr->empty()) {
      std::vector<StmtPtr> missing_defs;
      for (const auto& iter_arg : *iter_args_ptr) {
        outline_utils::VarRefCollector refs;
        refs.VisitExpr(iter_arg->initValue_);
        for (const Var* ref : refs.var_refs) {
          if (!defined_so_far.count(ref) && !pulled.count(ref)) {
            PullDefinitionChain(ref, original_def_map, defined_so_far, pulled, missing_defs);
          }
        }
      }
      // Add pulled definitions to defined_so_far so later statements see them
      for (const auto& def : missing_defs) {
        if (auto assign = std::dynamic_pointer_cast<const AssignStmt>(def)) {
          defined_so_far.insert(assign->var_.get());
        }
      }
      result.insert(result.end(), missing_defs.begin(), missing_defs.end());
    }

    // Track definitions from the current statement before recursing
    outline_utils::VarDefCollector stmt_defs;
    stmt_defs.VisitStmt(stmt);
    defined_so_far.insert(stmt_defs.var_defs.begin(), stmt_defs.var_defs.end());

    // Recurse into compound statements
    if (for_stmt) {
      auto new_body = recurse(FlattenBody(for_stmt->body_));
      result.push_back(RebuildForStmt(for_stmt, MakeBody(new_body, for_stmt->span_)));
    } else if (auto if_stmt = std::dynamic_pointer_cast<const IfStmt>(stmt)) {
      auto new_then = recurse(FlattenBody(if_stmt->then_body_));
      auto new_else = ProcessElseBranch(if_stmt, [&](const std::vector<StmtPtr>& es) { return recurse(es); });
      result.push_back(RebuildIfStmt(if_stmt, new_then, new_else));
    } else if (while_stmt) {
      auto new_body = recurse(FlattenBody(while_stmt->body_));
      result.push_back(RebuildWhileStmt(while_stmt, MakeBody(new_body, while_stmt->span_)));
    } else {
      result.push_back(stmt);
    }
  }

  return result;
}

/// Recursively fix dangling YieldStmt values by replacing them with the iter_arg
/// (identity yield -- preserves previous iteration's value on this core side).
StmtPtr FixDanglingYieldStmt(const StmtPtr& stmt, const std::vector<IterArgPtr>& iter_args,
                             const std::unordered_set<const Var*>& defined_vars) {
  return TransformLastStmt(stmt, [&](const StmtPtr& s) -> StmtPtr {
    auto yield_stmt = std::dynamic_pointer_cast<const YieldStmt>(s);
    if (!yield_stmt) return s;

    std::vector<ExprPtr> new_values;
    for (size_t i = 0; i < yield_stmt->value_.size(); ++i) {
      outline_utils::VarRefCollector refs;
      refs.VisitExpr(yield_stmt->value_[i]);
      bool has_undefined = std::any_of(refs.var_refs.begin(), refs.var_refs.end(),
                                       [&](const Var* ref) { return !defined_vars.count(ref); });
      if (has_undefined && i < iter_args.size()) {
        new_values.push_back(iter_args[i]);
      } else {
        new_values.push_back(yield_stmt->value_[i]);
      }
    }
    return std::make_shared<YieldStmt>(new_values, yield_stmt->span_);
  });
}

/// Fix dangling yields that appear anywhere in a loop body statement list.
/// This covers both the loop's final YieldStmt and nested IfStmt branch yields
/// that carry loop state forward through a conditional.
std::vector<StmtPtr> FixDanglingLoopBodyYields(const std::vector<StmtPtr>& stmts,
                                               const std::vector<IterArgPtr>& iter_args,
                                               const std::unordered_set<const Var*>& defined_vars) {
  std::vector<StmtPtr> result;
  result.reserve(stmts.size());
  for (const auto& stmt : stmts) {
    result.push_back(FixDanglingYieldStmt(stmt, iter_args, defined_vars));
  }
  return result;
}

/// Fix dangling yield values in ForStmt/WhileStmt bodies.
/// When a yield value references an undefined variable (its definition was stripped
/// during core body splitting), replace it with the corresponding iter_arg variable
/// (identity yield -- preserves the value from the previous iteration on this core side).
/// Uses a prefix-only `defined_so_far` set to avoid treating non-dominating definitions
/// (defined later in the same scope) as available.
std::vector<StmtPtr> FixupDanglingYieldValues(const std::vector<StmtPtr>& stmts) {
  // Build prefix-only defined set: track definitions as we scan statements in order.
  std::unordered_set<const Var*> defined_so_far;

  std::vector<StmtPtr> result;
  for (const auto& stmt : stmts) {
    auto for_stmt = std::dynamic_pointer_cast<const ForStmt>(stmt);
    auto while_stmt = std::dynamic_pointer_cast<const WhileStmt>(stmt);

    if ((for_stmt && !for_stmt->iter_args_.empty()) || (while_stmt && !while_stmt->iter_args_.empty())) {
      const auto& iter_args = for_stmt ? for_stmt->iter_args_ : while_stmt->iter_args_;
      const auto& body = for_stmt ? for_stmt->body_ : while_stmt->body_;

      // Collect defined vars within the loop body + preceding scope
      outline_utils::VarDefCollector body_def_collector;
      body_def_collector.VisitStmt(body);
      auto all_defined = defined_so_far;
      all_defined.insert(body_def_collector.var_defs.begin(), body_def_collector.var_defs.end());

      // Recursively process body, then fix dangling yields in the loop tail and
      // any nested conditional branches carrying loop state.
      auto body_stmts = FixupDanglingYieldValues(FlattenBody(body));
      body_stmts = FixDanglingLoopBodyYields(body_stmts, iter_args, all_defined);

      const auto& span = for_stmt ? for_stmt->span_ : while_stmt->span_;
      if (for_stmt) {
        result.push_back(RebuildForStmt(for_stmt, MakeBody(body_stmts, span)));
      } else {
        result.push_back(RebuildWhileStmt(while_stmt, MakeBody(body_stmts, span)));
      }
    } else if (auto if_stmt = std::dynamic_pointer_cast<const IfStmt>(stmt)) {
      auto new_then = FixupDanglingYieldValues(FlattenBody(if_stmt->then_body_));
      auto new_else = ProcessElseBranch(
          if_stmt, [](const std::vector<StmtPtr>& es) { return FixupDanglingYieldValues(es); });
      result.push_back(RebuildIfStmt(if_stmt, new_then, new_else));
    } else {
      result.push_back(stmt);
    }

    // Track definitions from the current statement for subsequent iterations
    outline_utils::VarDefCollector stmt_defs;
    stmt_defs.VisitStmt(stmt);
    defined_so_far.insert(stmt_defs.var_defs.begin(), stmt_defs.var_defs.end());
  }

  return result;
}

/// Final cleanup after one core side has been split out of the mixed body.
///
/// The split step intentionally preserves SHARED control-flow structure on both
/// sides. That can leave loop-carried state temporarily inconsistent until we
/// repair it in order:
///   1. Strip iter_args whose carried values are dead on this side.
///   2. Pull back any now-missing init-value definitions for surviving iter_args.
///   3. Rewrite dangling yields to identity yields when a branch-local value was
///      stripped on this side.
///   4. Run DCE once the loop state is structurally valid again.
///   5. Normalize loop-carried state one more time, because DCE may remove a
///      SHARED-only post-loop use that temporarily kept an iter_arg alive.
///   6. Run DCE again to clean up init-value chains exposed by the second strip.
std::vector<StmtPtr> FinalizeSplitCoreBody(const std::vector<StmtPtr>& stmts,
                                           const std::unordered_map<const Var*, StmtPtr>& original_def_map) {
  auto repaired = StripDeadIterArgs(stmts);
  repaired = FixupIterArgInitValues(repaired, original_def_map);
  repaired = FixupDanglingYieldValues(repaired);
  repaired = EliminateDeadCode(repaired);
  repaired = StripDeadIterArgs(repaired);
  return EliminateDeadCode(repaired);
}

// ============================================================================
// Parameterized Core Body Builder (shared by AIC and AIV)
// ============================================================================

enum class CoreSide { AIC, AIV };

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
          result.push_back(
              std::make_shared<EvalStmt>(CreateTpush(push_op, bm.source_tile, stmt->span_), stmt->span_));
        } else {
          auto dest_tile_type = std::dynamic_pointer_cast<const TileType>(bm.dest_var->GetType());
          INTERNAL_CHECK(dest_tile_type) << "Boundary move destination must have TileType";
          auto tpop_type = BuildBoundaryTpopType(side, bm.dest_var->GetType());
          bool needs_post_move = NeedsPostTpopMove(side, *dest_tile_type);
          std::string tpop_name = needs_post_move ? BuildBoundaryTpopName(side, bm.dest_var->name_hint_)
                                                  : bm.dest_var->name_hint_;
          auto tpop_var = std::make_shared<Var>(tpop_name, CleanTileType(tpop_type), stmt->span_);
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
              tpop_var, CreateTpop(pop_op, tpop_type, stmt->span_, kwargs), stmt->span_));
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
  auto aic_final = FinalizeSplitCoreBody(aic_stmts_no_return, original_def_map);

  // Build AIV body (recursive — handles MIXED compound stmts)
  std::unordered_map<const Var*, VarPtr> aiv_tpop_remap;
  auto aiv_stmts =
      BuildCoreBody(CoreSide::AIV, stmts, stmt_map, boundary_moves, aiv_tpop_remap, superseded_tpop_vars);
  auto aiv_final = FinalizeSplitCoreBody(aiv_stmts, original_def_map);

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

  // Map dangling tile.store result vars to the store destination's fresh param.
  // When a tile.store is on the AIC side, its result var is stripped from the AIV body,
  // but the var may still be referenced (e.g., in a ReturnStmt). Remap those dangling
  // references to the fresh parameter corresponding to the store's output tensor.
  {
    // Collect all vars defined in the AIV body
    outline_utils::VarDefCollector aiv_def_collector;
    auto aiv_body_stmt = MakeBody(aiv_final, func->span_);
    aiv_def_collector.VisitStmt(aiv_body_stmt);

    // Scan original body recursively for tile.store AssignStmts
    std::vector<std::shared_ptr<const AssignStmt>> original_assigns;
    CollectAllAssignStmts(stmts, original_assigns);
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

    outline_utils::VarRefCollector aiv_ref_collector;
    aiv_ref_collector.VisitStmt(aiv_body_stmt);
    for (const Var* ref_ptr : aiv_ref_collector.var_refs) {
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

class TpopMemoryVerifier : public IRVisitor {
 public:
  TpopMemoryVerifier(std::vector<Diagnostic>& diagnostics, std::string func_name, FunctionType func_type)
      : diagnostics_(diagnostics), func_name_(std::move(func_name)), func_type_(func_type) {}

  void VisitStmt_(const AssignStmtPtr& op) override {
    if (!op) return;
    auto call = std::dynamic_pointer_cast<const Call>(op->value_);
    auto ir_op = call ? std::dynamic_pointer_cast<const Op>(call->op_) : nullptr;
    if (!ir_op) {
      IRVisitor::VisitStmt_(op);
      return;
    }

    std::optional<MemorySpace> expected_memory;
    if (func_type_ == FunctionType::AIC && ir_op->name_ == "tile.tpop_from_aiv") {
      expected_memory = MemorySpace::Mat;
    } else if (func_type_ == FunctionType::AIV && ir_op->name_ == "tile.tpop_from_aic") {
      expected_memory = MemorySpace::Vec;
    }

    if (expected_memory.has_value()) {
      auto tile_type = std::dynamic_pointer_cast<const TileType>(op->var_->GetType());
      bool valid = tile_type && tile_type->memory_space_.has_value() &&
                   tile_type->memory_space_.value() == expected_memory.value();
      if (!valid) {
        std::string func_kind = (func_type_ == FunctionType::AIC) ? "AIC" : "AIV";
        std::string actual_memory = (tile_type && tile_type->memory_space_.has_value())
                                        ? MemorySpaceToString(tile_type->memory_space_.value())
                                        : "unset";
        diagnostics_.emplace_back(
            DiagnosticSeverity::Error, "MixedKernelExpanded", 0,
            func_kind + " function '" + func_name_ + "' requires " + ir_op->name_ +
                " result in MemorySpace::" + MemorySpaceToString(expected_memory.value()) +
                ", got MemorySpace::" + actual_memory,
            op->span_);
      }
    }

    IRVisitor::VisitStmt_(op);
  }

 private:
  std::vector<Diagnostic>& diagnostics_;
  std::string func_name_;
  FunctionType func_type_;
};

}  // namespace

class MixedKernelExpandedPropertyVerifierImpl : public PropertyVerifier {
 public:
  [[nodiscard]] std::string GetName() const override { return "MixedKernelExpanded"; }

  void Verify(const ProgramPtr& program, std::vector<Diagnostic>& diagnostics) override {
    if (!program) return;
    for (const auto& [gv, func] : program->functions_) {
      if (!func || !func->body_) continue;
      if (func->func_type_ == FunctionType::InCore) {
        MixedKernelExpandedVerifier verifier(diagnostics, func->name_);
        verifier.VisitStmt(func->body_);
        verifier.CheckResult();
        continue;
      }
      if (func->func_type_ == FunctionType::AIC || func->func_type_ == FunctionType::AIV) {
        TpopMemoryVerifier verifier(diagnostics, func->name_, func->func_type_);
        verifier.VisitStmt(func->body_);
      }
    }
  }
};

PropertyVerifierPtr CreateMixedKernelExpandedPropertyVerifier() {
  return std::make_shared<MixedKernelExpandedPropertyVerifierImpl>();
}

}  // namespace ir
}  // namespace pypto
