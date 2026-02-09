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

#include "pypto/ir/transforms/utils/flatten_single_stmt.h"

#include <memory>
#include <vector>

#include "pypto/core/logging.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"

namespace pypto::ir {

/**
 * @brief Mutator that recursively flattens single-statement blocks
 *
 * This mutator simplifies IR by removing unnecessary nesting:
 * - SeqStmts([single_stmt]) => single_stmt
 * - OpStmts([single_stmt]) => single_stmt
 * - Applied recursively
 */
class FlattenSingleStmtMutator : public IRMutator {
 public:
  FlattenSingleStmtMutator() = default;

  // Override statement visitors
  StmtPtr VisitStmt_(const SeqStmtsPtr& op) override;
  StmtPtr VisitStmt_(const OpStmtsPtr& op) override;
  StmtPtr VisitStmt_(const IfStmtPtr& op) override;
  StmtPtr VisitStmt_(const ForStmtPtr& op) override;
};

StmtPtr FlattenSingleStmtMutator::VisitStmt_(const SeqStmtsPtr& op) {
  // First, recursively visit all child statements
  std::vector<StmtPtr> new_stmts;
  bool changed = false;

  for (const auto& stmt : op->stmts_) {
    auto new_stmt = VisitStmt(stmt);
    new_stmts.push_back(new_stmt);
    if (new_stmt.get() != stmt.get()) {
      changed = true;
    }
  }

  // If only one statement, flatten
  if (new_stmts.size() == 1) {
    return new_stmts[0];
  }

  // Copy-on-write: only create new node if changed
  if (changed) {
    return std::make_shared<const SeqStmts>(new_stmts, op->span_);
  }
  return op;
}

StmtPtr FlattenSingleStmtMutator::VisitStmt_(const OpStmtsPtr& op) {
  // First, recursively visit all child statements
  std::vector<StmtPtr> new_stmts;
  bool changed = false;

  for (const auto& stmt : op->stmts_) {
    auto new_stmt = VisitStmt(stmt);
    new_stmts.push_back(new_stmt);
    if (new_stmt.get() != stmt.get()) {
      changed = true;
    }
  }

  // If only one statement, flatten
  if (new_stmts.size() == 1) {
    return new_stmts[0];
  }

  // Copy-on-write: only create new node if changed
  if (changed) {
    return std::make_shared<const OpStmts>(new_stmts, op->span_);
  }
  return op;
}

StmtPtr FlattenSingleStmtMutator::VisitStmt_(const IfStmtPtr& op) {
  // Visit condition
  auto new_condition = VisitExpr(op->condition_);

  // Visit then branch
  auto new_then = VisitStmt(op->then_body_);

  // Visit else branch if present
  std::optional<StmtPtr> new_else;
  bool else_changed = false;
  if (op->else_body_.has_value()) {
    auto visited_else = VisitStmt(op->else_body_.value());
    new_else = visited_else;
    else_changed = (visited_else.get() != op->else_body_.value().get());
  }

  // Check if anything changed
  bool changed = (new_condition.get() != op->condition_.get()) || (new_then.get() != op->then_body_.get()) ||
                 else_changed;

  if (changed) {
    return std::make_shared<IfStmt>(new_condition, new_then, new_else, op->return_vars_, op->span_);
  }
  return op;
}

StmtPtr FlattenSingleStmtMutator::VisitStmt_(const ForStmtPtr& op) {
  // Visit range expressions
  auto new_start = VisitExpr(op->start_);
  auto new_stop = VisitExpr(op->stop_);
  auto new_step = VisitExpr(op->step_);

  // Visit body
  auto new_body = VisitStmt(op->body_);

  // Check if anything changed
  bool changed = (new_start.get() != op->start_.get()) || (new_stop.get() != op->stop_.get()) ||
                 (new_step.get() != op->step_.get()) || (new_body.get() != op->body_.get());

  if (changed) {
    return std::make_shared<ForStmt>(op->loop_var_, new_start, new_stop, new_step, op->iter_args_, new_body,
                                     op->return_vars_, op->span_);
  }
  return op;
}

// Public API
FunctionPtr FlattenSingleStmt(const FunctionPtr& func) {
  INTERNAL_CHECK(func) << "FlattenSingleStmt cannot run on null function";

  FlattenSingleStmtMutator mutator;
  auto new_body = mutator.VisitStmt(func->body_);

  // Check if body changed
  if (new_body.get() != func->body_.get()) {
    return std::make_shared<Function>(func->name_, func->params_, func->return_types_, new_body, func->span_,
                                      func->func_type_);
  }
  return func;
}

}  // namespace pypto::ir
