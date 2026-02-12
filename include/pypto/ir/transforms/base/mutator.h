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

#ifndef PYPTO_IR_TRANSFORMS_BASE_MUTATOR_H_
#define PYPTO_IR_TRANSFORMS_BASE_MUTATOR_H_

#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/functor.h"

namespace pypto {
namespace ir {

/**
 * @brief IR mutator for immutable transformations
 *
 * Provides default implementations that recursively transform the IR tree.
 * Returns new ExprPtr or StmtPtr for transformed IR nodes, respecting immutability.
 * Uses copy-on-write: if children are unchanged, returns the original shared_ptr.
 */
class IRMutator : public ExprFunctor<ExprPtr>, public StmtFunctor<StmtPtr> {
 public:
  ~IRMutator() override = default;

  // Override base class methods
  ExprPtr VisitExpr(const ExprPtr& expr) override;
  StmtPtr VisitStmt(const StmtPtr& stmt) override;

 protected:
  // Leaf nodes - return as-is by default
  ExprPtr VisitExpr_(const VarPtr& op) override;
  ExprPtr VisitExpr_(const IterArgPtr& op) override;
  ExprPtr VisitExpr_(const MemRefPtr& op) override;
  ExprPtr VisitExpr_(const ConstIntPtr& op) override;
  ExprPtr VisitExpr_(const ConstFloatPtr& op) override;
  ExprPtr VisitExpr_(const ConstBoolPtr& op) override;
  ExprPtr VisitExpr_(const CallPtr& op) override;
  ExprPtr VisitExpr_(const MakeTuplePtr& op) override;
  ExprPtr VisitExpr_(const TupleGetItemExprPtr& op) override;

  // Binary operations - reconstruct with mutated children
  ExprPtr VisitExpr_(const AddPtr& op) override;
  ExprPtr VisitExpr_(const SubPtr& op) override;
  ExprPtr VisitExpr_(const MulPtr& op) override;
  ExprPtr VisitExpr_(const FloorDivPtr& op) override;
  ExprPtr VisitExpr_(const FloorModPtr& op) override;
  ExprPtr VisitExpr_(const FloatDivPtr& op) override;
  ExprPtr VisitExpr_(const MinPtr& op) override;
  ExprPtr VisitExpr_(const MaxPtr& op) override;
  ExprPtr VisitExpr_(const PowPtr& op) override;
  ExprPtr VisitExpr_(const EqPtr& op) override;
  ExprPtr VisitExpr_(const NePtr& op) override;
  ExprPtr VisitExpr_(const LtPtr& op) override;
  ExprPtr VisitExpr_(const LePtr& op) override;
  ExprPtr VisitExpr_(const GtPtr& op) override;
  ExprPtr VisitExpr_(const GePtr& op) override;
  ExprPtr VisitExpr_(const AndPtr& op) override;
  ExprPtr VisitExpr_(const OrPtr& op) override;
  ExprPtr VisitExpr_(const XorPtr& op) override;
  ExprPtr VisitExpr_(const BitAndPtr& op) override;
  ExprPtr VisitExpr_(const BitOrPtr& op) override;
  ExprPtr VisitExpr_(const BitXorPtr& op) override;
  ExprPtr VisitExpr_(const BitShiftLeftPtr& op) override;
  ExprPtr VisitExpr_(const BitShiftRightPtr& op) override;

  // Unary operations - reconstruct with mutated operand
  ExprPtr VisitExpr_(const AbsPtr& op) override;
  ExprPtr VisitExpr_(const NegPtr& op) override;
  ExprPtr VisitExpr_(const NotPtr& op) override;
  ExprPtr VisitExpr_(const BitNotPtr& op) override;
  ExprPtr VisitExpr_(const CastPtr& op) override;

  // Statement types
  StmtPtr VisitStmt_(const AssignStmtPtr& op) override;
  StmtPtr VisitStmt_(const IfStmtPtr& op) override;
  StmtPtr VisitStmt_(const YieldStmtPtr& op) override;
  StmtPtr VisitStmt_(const ReturnStmtPtr& op) override;
  StmtPtr VisitStmt_(const ForStmtPtr& op) override;
  StmtPtr VisitStmt_(const WhileStmtPtr& op) override;
  StmtPtr VisitStmt_(const ScopeStmtPtr& op) override;
  StmtPtr VisitStmt_(const SeqStmtsPtr& op) override;
  StmtPtr VisitStmt_(const OpStmtsPtr& op) override;
  StmtPtr VisitStmt_(const EvalStmtPtr& op) override;
  StmtPtr VisitStmt_(const StmtPtr& op) override;
};

}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORMS_BASE_MUTATOR_H_
