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

#include "pypto/ir/transforms/utils/normalize_stmt_structure.h"

#include <memory>
#include <vector>

#include "pypto/core/logging.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"

namespace pypto::ir {

/**
 * @brief Mutator that normalizes statement structure
 *
 * This mutator ensures:
 * 1. Function/IfStmt/ForStmt body are SeqStmts
 * 2. Consecutive AssignStmt/EvalStmt in SeqStmts are wrapped in OpStmts
 */
class NormalizeStmtStructureMutator : public IRMutator {
 public:
  NormalizeStmtStructureMutator() = default;

  // Override statement visitors
  StmtPtr VisitStmt_(const SeqStmtsPtr& op) override;
  StmtPtr VisitStmt_(const IfStmtPtr& op) override;
  StmtPtr VisitStmt_(const ForStmtPtr& op) override;

 private:
  /**
   * @brief Normalize a body statement to ensure it's a SeqStmts
   *
   * If body is not a SeqStmts, wraps it in SeqStmts.
   * Then normalizes the SeqStmts content (wrapping AssignStmt/EvalStmt in OpStmts).
   *
   * @param body Input body statement
   * @return Normalized SeqStmts
   */
  SeqStmtsPtr NormalizeBody(const StmtPtr& body);

  /**
   * @brief Check if a statement is AssignStmt or EvalStmt
   */
  [[nodiscard]] bool IsOpStmt(const StmtPtr& stmt) const {
    return As<AssignStmt>(stmt) || As<EvalStmt>(stmt);
  }
};

SeqStmtsPtr NormalizeStmtStructureMutator::NormalizeBody(const StmtPtr& body) {
  // First, recursively visit the body
  auto visited_body = VisitStmt(body);

  // If it's already a SeqStmts, it will be normalized by VisitStmt_(SeqStmtsPtr)
  if (auto seq = As<SeqStmts>(visited_body)) {
    return seq;
  }

  // Otherwise, wrap in SeqStmts
  std::vector<StmtPtr> stmts;

  // If it's AssignStmt or EvalStmt, wrap in OpStmts first
  if (IsOpStmt(visited_body)) {
    auto op_stmts = std::make_shared<const OpStmts>(std::vector<StmtPtr>{visited_body}, visited_body->span_);
    stmts.push_back(op_stmts);
  } else {
    stmts.push_back(visited_body);
  }

  return std::make_shared<const SeqStmts>(stmts, visited_body->span_);
}

StmtPtr NormalizeStmtStructureMutator::VisitStmt_(const SeqStmtsPtr& op) {
  std::vector<StmtPtr> new_stmts;
  std::vector<StmtPtr> current_group;  // Group of consecutive AssignStmt/EvalStmt
  bool changed = false;

  for (const auto& stmt : op->stmts_) {
    // Recursively visit the statement
    auto new_stmt = VisitStmt(stmt);
    if (new_stmt.get() != stmt.get()) {
      changed = true;
    }

    // Check if this is an AssignStmt or EvalStmt
    if (IsOpStmt(new_stmt)) {
      // Add to current group
      current_group.push_back(new_stmt);
    } else {
      // Flush current group if non-empty
      if (!current_group.empty()) {
        auto op_stmts = std::make_shared<const OpStmts>(current_group, current_group[0]->span_);
        new_stmts.push_back(op_stmts);
        current_group.clear();
        changed = true;
      }

      // Add the non-op statement
      new_stmts.push_back(new_stmt);
    }
  }

  // Flush remaining group
  if (!current_group.empty()) {
    auto op_stmts = std::make_shared<const OpStmts>(current_group, current_group[0]->span_);
    new_stmts.push_back(op_stmts);
    changed = true;
  }

  // Copy-on-write: only create new node if changed
  if (changed) {
    return std::make_shared<const SeqStmts>(new_stmts, op->span_);
  }
  return op;
}

StmtPtr NormalizeStmtStructureMutator::VisitStmt_(const IfStmtPtr& op) {
  // Normalize then branch
  auto new_then = NormalizeBody(op->then_body_);

  // Normalize else branch if present
  std::optional<StmtPtr> new_else;
  bool else_changed = false;
  if (op->else_body_.has_value()) {
    auto normalized_else = NormalizeBody(op->else_body_.value());
    new_else = normalized_else;
    else_changed = (normalized_else.get() != op->else_body_.value().get());
  }

  // Check if anything changed
  bool changed = (new_then.get() != op->then_body_.get()) || else_changed;

  if (changed) {
    // Visit condition (shouldn't change for normalization, but call for consistency)
    auto new_condition = VisitExpr(op->condition_);
    return std::make_shared<IfStmt>(new_condition, new_then, new_else, op->return_vars_, op->span_);
  }
  return op;
}

StmtPtr NormalizeStmtStructureMutator::VisitStmt_(const ForStmtPtr& op) {
  // Normalize body
  auto new_body = NormalizeBody(op->body_);

  // Check if body changed
  if (new_body.get() != op->body_.get()) {
    // Visit range expressions (shouldn't change for normalization, but call for consistency)
    auto new_start = VisitExpr(op->start_);
    auto new_stop = VisitExpr(op->stop_);
    auto new_step = VisitExpr(op->step_);

    return std::make_shared<ForStmt>(op->loop_var_, new_start, new_stop, new_step, op->iter_args_, new_body,
                                     op->return_vars_, op->span_, op->kind_);
  }
  return op;
}

// Public API
FunctionPtr NormalizeStmtStructure(const FunctionPtr& func) {
  INTERNAL_CHECK(func) << "NormalizeStmtStructure cannot run on null function";

  NormalizeStmtStructureMutator mutator;
  auto new_body = mutator.VisitStmt(func->body_);

  // Ensure function body is SeqStmts
  SeqStmtsPtr normalized_body;
  if (auto seq = As<SeqStmts>(new_body)) {
    normalized_body = seq;
  } else {
    // Wrap in SeqStmts
    std::vector<StmtPtr> stmts;
    if (As<AssignStmt>(new_body) || As<EvalStmt>(new_body)) {
      auto op_stmts = std::make_shared<const OpStmts>(std::vector<StmtPtr>{new_body}, new_body->span_);
      stmts.push_back(op_stmts);
    } else {
      stmts.push_back(new_body);
    }
    normalized_body = std::make_shared<const SeqStmts>(stmts, new_body->span_);
  }

  return std::make_shared<Function>(func->name_, func->params_, func->return_types_, normalized_body,
                                    func->span_, func->func_type_);
}

}  // namespace pypto::ir
