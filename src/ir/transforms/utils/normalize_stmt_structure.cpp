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
#include <optional>
#include <utility>
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
 * 1. Nested SeqStmts are flattened
 * 2. Single-child SeqStmts are unwrapped
 */
class NormalizeStmtStructureMutator : public IRMutator {
 public:
  NormalizeStmtStructureMutator() = default;

  // Override statement visitors
  StmtPtr VisitStmt_(const SeqStmtsPtr& op) override;
  StmtPtr VisitStmt_(const IfStmtPtr& op) override;
  StmtPtr VisitStmt_(const ForStmtPtr& op) override;
  StmtPtr VisitStmt_(const WhileStmtPtr& op) override;

 private:
  /**
   * @brief Normalize a body statement
   *
   * Normalizes the body content. Single-child SeqStmts are unwrapped to avoid
   * redundant nesting.
   *
   * @param body Input body statement
   * @return Normalized statement
   */
  StmtPtr NormalizeBody(const StmtPtr& body);
};

StmtPtr NormalizeStmtStructureMutator::NormalizeBody(const StmtPtr& body) {
  // First, recursively visit the body
  auto visited_body = VisitStmt(body);

  // If it's already a SeqStmts, it will be normalized by VisitStmt_(SeqStmtsPtr)
  // (which also unwraps single-child SeqStmts)
  return visited_body;
}

StmtPtr NormalizeStmtStructureMutator::VisitStmt_(const SeqStmtsPtr& op) {
  std::vector<StmtPtr> new_stmts;
  bool changed = false;

  for (const auto& stmt : op->stmts_) {
    auto new_stmt = VisitStmt(stmt);
    if (new_stmt.get() != stmt.get()) {
      changed = true;
    }

    // Flatten nested SeqStmts by absorbing children
    if (auto nested_seq = As<SeqStmts>(new_stmt)) {
      changed = true;
      for (const auto& inner : nested_seq->stmts_) {
        new_stmts.push_back(inner);
      }
    } else {
      new_stmts.push_back(new_stmt);
    }
  }

  // Unwrap single-child even if content didn't change
  if (new_stmts.size() == 1) {
    return new_stmts[0];
  }
  // Copy-on-write: only create new node if changed
  if (!changed) {
    return op;
  }
  return SeqStmts::Flatten(std::move(new_stmts), op->span_);
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
                                     op->return_vars_, op->span_, op->kind_, op->chunk_size_,
                                     op->chunk_policy_, op->loop_origin_);
  }
  return op;
}

StmtPtr NormalizeStmtStructureMutator::VisitStmt_(const WhileStmtPtr& op) {
  // Normalize body
  auto new_body = NormalizeBody(op->body_);

  // Check if body changed
  if (new_body.get() != op->body_.get()) {
    // Visit condition (shouldn't change for normalization, but call for consistency)
    auto new_condition = VisitExpr(op->condition_);

    return std::make_shared<WhileStmt>(new_condition, op->iter_args_, new_body, op->return_vars_, op->span_);
  }
  return op;
}

// Public API
FunctionPtr NormalizeStmtStructure(const FunctionPtr& func) {
  INTERNAL_CHECK(func) << "NormalizeStmtStructure cannot run on null function";

  NormalizeStmtStructureMutator mutator;
  auto new_body = mutator.VisitStmt(func->body_);

  return std::make_shared<Function>(func->name_, func->params_, func->param_directions_, func->return_types_,
                                    new_body, func->span_, func->func_type_, func->level_, func->role_);
}

}  // namespace pypto::ir
