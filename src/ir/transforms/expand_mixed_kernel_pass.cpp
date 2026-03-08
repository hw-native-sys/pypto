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
#include "pypto/ir/transforms/utils/scope_outline_utils.h"
#include "pypto/ir/type.h"
#include "pypto/ir/verifier/verifier.h"

namespace pypto {
namespace ir {

namespace {

// ============================================================================
// Core Affinity Classification
// ============================================================================

enum class CoreAffinity { CUBE, VECTOR, SHARED, MIXED };

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

CoreAffinity ClassifyCallAffinity(const CallPtr& call) {
  if (!call || !call->op_) return CoreAffinity::SHARED;

  // GlobalVar call (function call) is SHARED
  if (std::dynamic_pointer_cast<const GlobalVar>(call->op_)) {
    return CoreAffinity::SHARED;
  }

  auto op = std::dynamic_pointer_cast<const Op>(call->op_);
  if (!op) return CoreAffinity::SHARED;

  const auto& name = op->name_;

  // Cube ops
  if (IsCubeOp(name)) return CoreAffinity::CUBE;

  // tile.* ops that are not cube are vector
  if (name.substr(0, 5) == "tile.") return CoreAffinity::VECTOR;

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
// Recursive Boundary Analysis
// ============================================================================

struct BoundaryInfo {
  // Variables flowing from VECTOR->CUBE (tpush_to_aic / tpop_from_aiv)
  std::unordered_set<std::string> v2c_vars;
  // Variables flowing from CUBE->VECTOR (tpush_to_aiv / tpop_from_aic)
  std::unordered_set<std::string> c2v_vars;
};

void AnalyzeBoundaries(const std::vector<StmtPtr>& stmts,
                       const std::unordered_map<const Stmt*, CoreAffinity>& stmt_map,
                       const std::unordered_map<std::string, CoreAffinity>& var_affinity,
                       const std::unordered_set<std::string>& param_names, BoundaryInfo& boundaries) {
  for (const auto& stmt : stmts) {
    auto it = stmt_map.find(stmt.get());
    CoreAffinity affinity = (it != stmt_map.end()) ? it->second : CoreAffinity::SHARED;

    // Check leaf CUBE/VECTOR stmts for cross-affinity variable references
    if (affinity == CoreAffinity::CUBE || affinity == CoreAffinity::VECTOR) {
      outline_utils::VarRefCollector ref_collector;
      ref_collector.VisitStmt(stmt);

      for (const auto& ref_name : ref_collector.var_refs) {
        if (param_names.count(ref_name)) continue;

        auto def_it = var_affinity.find(ref_name);
        if (def_it == var_affinity.end()) continue;
        auto def_aff = def_it->second;
        if (def_aff == CoreAffinity::SHARED) continue;

        if (def_aff == CoreAffinity::VECTOR && affinity == CoreAffinity::CUBE) {
          boundaries.v2c_vars.insert(ref_name);
        } else if (def_aff == CoreAffinity::CUBE && affinity == CoreAffinity::VECTOR) {
          boundaries.c2v_vars.insert(ref_name);
        }
      }
    }

    // Recurse into compound statements to find boundaries at nested levels
    if (auto for_stmt = std::dynamic_pointer_cast<const ForStmt>(stmt)) {
      AnalyzeBoundaries(FlattenBody(for_stmt->body_), stmt_map, var_affinity, param_names, boundaries);
    } else if (auto if_stmt = std::dynamic_pointer_cast<const IfStmt>(stmt)) {
      AnalyzeBoundaries(FlattenBody(if_stmt->then_body_), stmt_map, var_affinity, param_names, boundaries);
      if (if_stmt->else_body_.has_value()) {
        AnalyzeBoundaries(FlattenBody(if_stmt->else_body_.value()), stmt_map, var_affinity, param_names,
                          boundaries);
      }
    } else if (auto while_stmt = std::dynamic_pointer_cast<const WhileStmt>(stmt)) {
      AnalyzeBoundaries(FlattenBody(while_stmt->body_), stmt_map, var_affinity, param_names, boundaries);
    }
  }
}

// ============================================================================
// TPUSH / TPOP creation helpers
// ============================================================================

CallPtr CreateTpushToAiv(const ExprPtr& tile, const Span& span) {
  std::vector<std::pair<std::string, std::any>> kwargs;
  kwargs.emplace_back("aiv_idx", 0);
  return OpRegistry::GetInstance().Create("system.tpush_to_aiv", {tile}, kwargs, span);
}

CallPtr CreateTpushToAic(const ExprPtr& tile, const Span& span) {
  std::vector<std::pair<std::string, std::any>> kwargs;
  kwargs.emplace_back("aiv_idx", 0);
  return OpRegistry::GetInstance().Create("system.tpush_to_aic", {tile}, kwargs, span);
}

CallPtr CreateTpopFromAiv(const TypePtr& tile_type, const Span& span) {
  std::vector<std::pair<std::string, std::any>> kwargs;
  kwargs.emplace_back("aiv_idx", 0);
  auto op = OpRegistry::GetInstance().GetOp("system.tpop_from_aiv");
  return std::make_shared<Call>(op, std::vector<ExprPtr>{}, std::move(kwargs), tile_type, span);
}

CallPtr CreateTpopFromAic(const TypePtr& tile_type, const Span& span) {
  std::vector<std::pair<std::string, std::any>> kwargs;
  kwargs.emplace_back("aiv_idx", 0);
  auto op = OpRegistry::GetInstance().GetOp("system.tpop_from_aic");
  return std::make_shared<Call>(op, std::vector<ExprPtr>{}, std::move(kwargs), tile_type, span);
}

// ============================================================================
// Recursive Dead Code Elimination
// ============================================================================

bool IsSideEffectOp(const StmtPtr& stmt) {
  auto get_op_name = [](const StmtPtr& s) -> std::string {
    CallPtr call;
    if (auto assign = std::dynamic_pointer_cast<const AssignStmt>(s)) {
      call = std::dynamic_pointer_cast<const Call>(assign->value_);
    } else if (auto eval = std::dynamic_pointer_cast<const EvalStmt>(s)) {
      call = std::dynamic_pointer_cast<const Call>(eval->expr_);
    }
    if (call && call->op_) {
      if (auto op = std::dynamic_pointer_cast<const Op>(call->op_)) {
        return op->name_;
      }
    }
    return "";
  };

  auto name = get_op_name(stmt);
  // tpush ops, tile.store, tile.assemble are side-effecting
  return name == "system.tpush_to_aiv" || name == "system.tpush_to_aic" || name == "tile.store" ||
         name == "tile.assemble";
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
    if (std::dynamic_pointer_cast<const ReturnStmt>(stmt) || IsSideEffectOp(stmt)) {
      outline_utils::VarRefCollector refs;
      refs.VisitStmt(stmt);
      live.insert(refs.var_refs.begin(), refs.var_refs.end());
    }
    if (auto for_stmt = std::dynamic_pointer_cast<const ForStmt>(stmt)) {
      FindLiveRootsRecursive(FlattenBody(for_stmt->body_), live);
    } else if (auto if_stmt = std::dynamic_pointer_cast<const IfStmt>(stmt)) {
      FindLiveRootsRecursive(FlattenBody(if_stmt->then_body_), live);
      if (if_stmt->else_body_.has_value()) {
        FindLiveRootsRecursive(FlattenBody(if_stmt->else_body_.value()), live);
      }
    } else if (auto while_stmt = std::dynamic_pointer_cast<const WhileStmt>(stmt)) {
      FindLiveRootsRecursive(FlattenBody(while_stmt->body_), live);
    }
  }
}

std::vector<StmtPtr> FilterDeadCode(const std::vector<StmtPtr>& stmts,
                                    const std::unordered_set<std::string>& live) {
  std::vector<StmtPtr> result;
  for (const auto& stmt : stmts) {
    if (auto assign = std::dynamic_pointer_cast<const AssignStmt>(stmt)) {
      if (live.count(assign->var_->name_)) {
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
// Recursive AIC / AIV Body Builders
// ============================================================================

std::vector<StmtPtr> BuildAICBody(const std::vector<StmtPtr>& stmts,
                                  const std::unordered_map<const Stmt*, CoreAffinity>& stmt_map,
                                  const BoundaryInfo& boundaries,
                                  const std::unordered_map<std::string, VarPtr>& var_objects,
                                  std::unordered_set<std::string>& v2c_popped,
                                  std::unordered_set<std::string>& c2v_pushed) {
  std::vector<StmtPtr> aic_stmts;

  for (const auto& stmt : stmts) {
    auto it = stmt_map.find(stmt.get());
    CoreAffinity affinity = (it != stmt_map.end()) ? it->second : CoreAffinity::SHARED;

    if (affinity == CoreAffinity::VECTOR) {
      // Skip VECTOR statements in AIC
      continue;
    }

    if (affinity == CoreAffinity::CUBE) {
      // Insert tpop_from_aiv before first use of V2C vars
      outline_utils::VarRefCollector refs;
      refs.VisitStmt(stmt);
      for (const auto& ref : refs.var_refs) {
        if (boundaries.v2c_vars.count(ref) && !v2c_popped.count(ref)) {
          v2c_popped.insert(ref);
          auto var_it = var_objects.find(ref);
          INTERNAL_CHECK(var_it != var_objects.end()) << "Internal error: var " << ref << " not found";
          aic_stmts.push_back(std::make_shared<AssignStmt>(
              var_it->second, CreateTpopFromAiv(var_it->second->GetType(), stmt->span_), stmt->span_));
        }
      }

      // Keep CUBE stmt
      aic_stmts.push_back(stmt);

      // Insert tpush_to_aiv after CUBE stmt that defines C2V boundary var
      if (auto assign = std::dynamic_pointer_cast<const AssignStmt>(stmt)) {
        if (boundaries.c2v_vars.count(assign->var_->name_) && !c2v_pushed.count(assign->var_->name_)) {
          c2v_pushed.insert(assign->var_->name_);
          aic_stmts.push_back(
              std::make_shared<EvalStmt>(CreateTpushToAiv(assign->var_, stmt->span_), stmt->span_));
        }
      }
    } else if (affinity == CoreAffinity::MIXED) {
      // Recurse into compound statements, building pruned copies
      if (auto for_stmt = std::dynamic_pointer_cast<const ForStmt>(stmt)) {
        auto body_stmts = FlattenBody(for_stmt->body_);
        auto new_body = BuildAICBody(body_stmts, stmt_map, boundaries, var_objects, v2c_popped, c2v_pushed);
        aic_stmts.push_back(std::make_shared<ForStmt>(
            for_stmt->loop_var_, for_stmt->start_, for_stmt->stop_, for_stmt->step_, for_stmt->iter_args_,
            MakeBody(new_body, for_stmt->span_), for_stmt->return_vars_, for_stmt->span_, for_stmt->kind_,
            for_stmt->chunk_size_, for_stmt->chunk_policy_, for_stmt->loop_origin_));
      } else if (auto if_stmt = std::dynamic_pointer_cast<const IfStmt>(stmt)) {
        auto new_then = BuildAICBody(FlattenBody(if_stmt->then_body_), stmt_map, boundaries, var_objects,
                                     v2c_popped, c2v_pushed);
        std::optional<StmtPtr> new_else;
        if (if_stmt->else_body_.has_value()) {
          auto new_else_stmts = BuildAICBody(FlattenBody(if_stmt->else_body_.value()), stmt_map, boundaries,
                                             var_objects, v2c_popped, c2v_pushed);
          new_else = MakeBody(new_else_stmts, if_stmt->span_);
        }
        aic_stmts.push_back(std::make_shared<IfStmt>(if_stmt->condition_, MakeBody(new_then, if_stmt->span_),
                                                     new_else, if_stmt->return_vars_, if_stmt->span_));
      } else if (auto while_stmt = std::dynamic_pointer_cast<const WhileStmt>(stmt)) {
        auto body_stmts = FlattenBody(while_stmt->body_);
        auto new_body = BuildAICBody(body_stmts, stmt_map, boundaries, var_objects, v2c_popped, c2v_pushed);
        aic_stmts.push_back(std::make_shared<WhileStmt>(while_stmt->condition_, while_stmt->iter_args_,
                                                        MakeBody(new_body, while_stmt->span_),
                                                        while_stmt->return_vars_, while_stmt->span_));
      } else {
        aic_stmts.push_back(stmt);  // Unknown compound, include as-is
      }
    } else {
      // SHARED — include as-is
      aic_stmts.push_back(stmt);
    }
  }

  return aic_stmts;
}

std::vector<StmtPtr> BuildAIVBody(const std::vector<StmtPtr>& stmts,
                                  const std::unordered_map<const Stmt*, CoreAffinity>& stmt_map,
                                  const BoundaryInfo& boundaries,
                                  const std::unordered_map<std::string, VarPtr>& var_objects,
                                  std::unordered_set<std::string>& v2c_pushed,
                                  std::unordered_set<std::string>& c2v_popped) {
  std::vector<StmtPtr> aiv_stmts;

  for (const auto& stmt : stmts) {
    auto it = stmt_map.find(stmt.get());
    CoreAffinity affinity = (it != stmt_map.end()) ? it->second : CoreAffinity::SHARED;

    if (affinity == CoreAffinity::CUBE) {
      // Skip CUBE statements in AIV
      continue;
    }

    if (affinity == CoreAffinity::VECTOR) {
      // Insert tpop_from_aic before first use of C2V vars
      outline_utils::VarRefCollector refs;
      refs.VisitStmt(stmt);
      for (const auto& ref : refs.var_refs) {
        if (boundaries.c2v_vars.count(ref) && !c2v_popped.count(ref)) {
          c2v_popped.insert(ref);
          auto var_it = var_objects.find(ref);
          INTERNAL_CHECK(var_it != var_objects.end()) << "Internal error: var " << ref << " not found";
          aiv_stmts.push_back(std::make_shared<AssignStmt>(
              var_it->second, CreateTpopFromAic(var_it->second->GetType(), stmt->span_), stmt->span_));
        }
      }

      // Keep VECTOR stmt
      aiv_stmts.push_back(stmt);

      // Insert tpush_to_aic after VECTOR stmt that defines V2C boundary var
      if (auto assign = std::dynamic_pointer_cast<const AssignStmt>(stmt)) {
        if (boundaries.v2c_vars.count(assign->var_->name_) && !v2c_pushed.count(assign->var_->name_)) {
          v2c_pushed.insert(assign->var_->name_);
          aiv_stmts.push_back(
              std::make_shared<EvalStmt>(CreateTpushToAic(assign->var_, stmt->span_), stmt->span_));
        }
      }
    } else if (affinity == CoreAffinity::MIXED) {
      // Recurse into compound statements, building pruned copies
      if (auto for_stmt = std::dynamic_pointer_cast<const ForStmt>(stmt)) {
        auto body_stmts = FlattenBody(for_stmt->body_);
        auto new_body = BuildAIVBody(body_stmts, stmt_map, boundaries, var_objects, v2c_pushed, c2v_popped);
        aiv_stmts.push_back(std::make_shared<ForStmt>(
            for_stmt->loop_var_, for_stmt->start_, for_stmt->stop_, for_stmt->step_, for_stmt->iter_args_,
            MakeBody(new_body, for_stmt->span_), for_stmt->return_vars_, for_stmt->span_, for_stmt->kind_,
            for_stmt->chunk_size_, for_stmt->chunk_policy_, for_stmt->loop_origin_));
      } else if (auto if_stmt = std::dynamic_pointer_cast<const IfStmt>(stmt)) {
        auto new_then = BuildAIVBody(FlattenBody(if_stmt->then_body_), stmt_map, boundaries, var_objects,
                                     v2c_pushed, c2v_popped);
        std::optional<StmtPtr> new_else;
        if (if_stmt->else_body_.has_value()) {
          auto new_else_stmts = BuildAIVBody(FlattenBody(if_stmt->else_body_.value()), stmt_map, boundaries,
                                             var_objects, v2c_pushed, c2v_popped);
          new_else = MakeBody(new_else_stmts, if_stmt->span_);
        }
        aiv_stmts.push_back(std::make_shared<IfStmt>(if_stmt->condition_, MakeBody(new_then, if_stmt->span_),
                                                     new_else, if_stmt->return_vars_, if_stmt->span_));
      } else if (auto while_stmt = std::dynamic_pointer_cast<const WhileStmt>(stmt)) {
        auto body_stmts = FlattenBody(while_stmt->body_);
        auto new_body = BuildAIVBody(body_stmts, stmt_map, boundaries, var_objects, v2c_pushed, c2v_popped);
        aiv_stmts.push_back(std::make_shared<WhileStmt>(while_stmt->condition_, while_stmt->iter_args_,
                                                        MakeBody(new_body, while_stmt->span_),
                                                        while_stmt->return_vars_, while_stmt->span_));
      } else {
        aiv_stmts.push_back(stmt);  // Unknown compound, include as-is
      }
    } else {
      // SHARED — include as-is
      aiv_stmts.push_back(stmt);
    }
  }

  return aiv_stmts;
}

// ============================================================================
// Main Expansion Logic
// ============================================================================

struct ExpandedKernel {
  FunctionPtr aic_func;
  FunctionPtr aiv_func;
  FunctionPtr group_func;
};

ExpandedKernel ExpandMixedFunction(const FunctionPtr& func) {
  auto stmts = FlattenBody(func->body_);

  // Build symbol table (VarCollector already recurses into nested structures)
  outline_utils::VarCollector var_collector;
  for (const auto& var : func->params_) {
    var_collector.var_types[var->name_] = var->GetType();
    var_collector.var_objects[var->name_] = var;
  }
  var_collector.VisitStmt(func->body_);

  // Build param name set
  std::unordered_set<std::string> param_names;
  for (const auto& var : func->params_) {
    param_names.insert(var->name_);
  }

  // Recursive affinity analysis (descends into ForStmt/IfStmt/WhileStmt)
  std::unordered_map<const Stmt*, CoreAffinity> stmt_map;
  std::unordered_map<std::string, CoreAffinity> var_affinity;
  AnalyzeStmtsAffinity(stmts, stmt_map, var_affinity);

  // Recursive boundary analysis
  BoundaryInfo boundaries;
  AnalyzeBoundaries(stmts, stmt_map, var_affinity, param_names, boundaries);

  // Build AIC body (recursive — handles MIXED compound stmts)
  std::unordered_set<std::string> aic_v2c_popped, aic_c2v_pushed;
  auto aic_stmts =
      BuildAICBody(stmts, stmt_map, boundaries, var_collector.var_objects, aic_v2c_popped, aic_c2v_pushed);

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
  std::unordered_set<std::string> aiv_v2c_pushed, aiv_c2v_popped;
  auto aiv_stmts =
      BuildAIVBody(stmts, stmt_map, boundaries, var_collector.var_objects, aiv_v2c_pushed, aiv_c2v_popped);
  // DCE on AIV (recursive)
  auto aiv_final = EliminateDeadCode(aiv_stmts);

  // Create AIC function (same params, no return)
  std::string aic_name = func->name_ + "_aic";
  auto aic_func =
      std::make_shared<Function>(aic_name, func->params_, func->param_directions_, std::vector<TypePtr>{},
                                 MakeBody(aic_final, func->span_), func->span_, FunctionType::AIC);

  // Create AIV function (same params, same return)
  std::string aiv_name = func->name_ + "_aiv";
  auto aiv_func =
      std::make_shared<Function>(aiv_name, func->params_, func->param_directions_, func->return_types_,
                                 MakeBody(aiv_final, func->span_), func->span_, FunctionType::AIV);

  // Create Group function: calls AIC then AIV, returns AIV result
  std::string group_name = func->name_;  // Group replaces the original

  // Create fresh parameters for the group function
  std::vector<VarPtr> group_params;
  for (const auto& var : func->params_) {
    group_params.push_back(std::make_shared<Var>(var->name_, var->GetType(), func->span_));
  }

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

  std::shared_ptr<Call> aiv_call;
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
// Call Site Updater: replace calls to InCore with calls to Group
// ============================================================================

// No explicit call site update needed — the Group function keeps the same name
// as the original InCore function. The program map is keyed by name via
// GlobalVar, so the replacement happens automatically when we insert the Group
// function with the original name.

}  // namespace

namespace pass {

Pass ExpandMixedKernel() {
  auto pass_func = [](const ProgramPtr& program) -> ProgramPtr {
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

      bool has_cube = (combined == CoreAffinity::CUBE || combined == CoreAffinity::MIXED);
      bool has_vector = (combined == CoreAffinity::VECTOR || combined == CoreAffinity::MIXED);

      if (!has_cube || !has_vector) {
        // Not mixed — pass through unchanged
        new_functions.push_back(func);
        continue;
      }

      // Expand mixed kernel
      auto expanded = ExpandMixedFunction(func);
      // Add AIC and AIV before the Group function (inner functions first)
      new_functions.push_back(expanded.aic_func);
      new_functions.push_back(expanded.aiv_func);
      new_functions.push_back(expanded.group_func);
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
    auto opnode = std::dynamic_pointer_cast<const Op>(op->op_);
    if (!opnode) {
      IRVisitor::VisitExpr_(op);
      return;
    }
    if (IsCubeOp(opnode->name_)) {
      has_cube_ = true;
    } else if (opnode->name_.substr(0, 5) == "tile.") {
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
