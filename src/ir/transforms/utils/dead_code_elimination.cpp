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

#include "pypto/ir/transforms/utils/dead_code_elimination.h"

#include <memory>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

#include "pypto/ir/expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/utils/loop_state_repair.h"
#include "pypto/ir/transforms/utils/scope_outline_utils.h"
#include "pypto/ir/transforms/utils/transform_utils.h"

namespace pypto {
namespace ir {
namespace dce {

const auto& FlattenBody = transform_utils::FlattenToStmts;

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
  static const std::unordered_set<std::string> side_effect_ops = {"tile.tpush_to_aiv",
                                                                  "tile.tpush_to_aic",
                                                                  "tile.tpop_from_aic",
                                                                  "tile.tpop_from_aiv",
                                                                  "tile.store",
                                                                  "tile.assemble",
                                                                  "system.tfree_to_aic",
                                                                  "system.tfree_to_aiv",
                                                                  "system.reserve_buffer",
                                                                  "system.import_peer_buffer",
                                                                  "system.aic_initialize_pipe",
                                                                  "system.aiv_initialize_pipe"};
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

namespace {

using loop_repair::MakeBody;

void FindLiveRootsRecursive(const std::vector<StmtPtr>& stmts, std::unordered_set<const Var*>& live) {
  for (const auto& stmt : stmts) {
    if (std::dynamic_pointer_cast<const ReturnStmt>(stmt) ||
        std::dynamic_pointer_cast<const YieldStmt>(stmt) || IsSideEffectOp(stmt)) {
      outline_utils::VarRefCollector refs;
      refs.VisitStmt(stmt);
      live.insert(refs.var_refs.begin(), refs.var_refs.end());
      if (auto assign = std::dynamic_pointer_cast<const AssignStmt>(stmt)) {
        live.insert(assign->var_.get());
      }
    }
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
      result.push_back(stmt);
    }
  }
  return result;
}

}  // namespace

std::vector<StmtPtr> EliminateDeadCode(const std::vector<StmtPtr>& stmts) {
  std::unordered_set<const Var*> live;
  FindLiveRootsRecursive(stmts, live);

  std::vector<std::shared_ptr<const AssignStmt>> all_assigns;
  CollectAllAssignStmts(stmts, all_assigns);

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

}  // namespace dce
}  // namespace ir
}  // namespace pypto
