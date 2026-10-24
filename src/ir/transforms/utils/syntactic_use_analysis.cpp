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

#include "pypto/ir/transforms/utils/syntactic_use_analysis.h"

namespace pypto {
namespace ir {

void SyntacticUseAnalysis::VisitVarLike_(const VarPtr& op) {
  if (!op) return;
  ++read_counts[op.get()];
  IRVisitor::VisitVarLike_(op);
}

void SyntacticUseAnalysis::VisitExpr_(const IterArgPtr& op) {
  if (op) VisitVarLike_(op);
}

void SyntacticUseAnalysis::VisitStmt_(const AssignStmtPtr& op) {
  if (!op) return;
  definitions.emplace(op->var_.get(), op);
  if (op->value_) VisitExpr(op->value_);
}

void SyntacticUseAnalysis::VisitStmt_(const IfStmtPtr& op) {
  if (!op) return;
  if (op->condition_) VisitExpr(op->condition_);
  if (op->then_body_) VisitStmt(op->then_body_);
  if (op->else_body_.has_value() && *op->else_body_) VisitStmt(*op->else_body_);
}

void SyntacticUseAnalysis::VisitStmt_(const ForStmtPtr& op) {
  if (!op) return;
  if (op->start_) VisitExpr(op->start_);
  if (op->stop_) VisitExpr(op->stop_);
  if (op->step_) VisitExpr(op->step_);
  for (const auto& iter_arg : op->iter_args_) {
    if (iter_arg && iter_arg->initValue_) VisitExpr(iter_arg->initValue_);
  }
  if (op->body_) VisitStmt(op->body_);
}

void SyntacticUseAnalysis::VisitStmt_(const WhileStmtPtr& op) {
  if (!op) return;
  if (op->condition_) VisitExpr(op->condition_);
  for (const auto& iter_arg : op->iter_args_) {
    if (iter_arg && iter_arg->initValue_) VisitExpr(iter_arg->initValue_);
  }
  if (op->body_) VisitStmt(op->body_);
}

void SyntacticUseAnalysis::VisitStmt_(const SeqStmtsPtr& op) {
  for (size_t i = 0; i < op->stmts_.size(); ++i) {
    const auto& stmt = op->stmts_[i];
    positions.emplace(stmt.get(), Position{op.get(), i});
    VisitStmt(stmt);
  }
}

}  // namespace ir
}  // namespace pypto
