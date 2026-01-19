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

#ifndef PYPTO_IR_TRANSFORM_BASE_VISITOR_H_
#define PYPTO_IR_TRANSFORM_BASE_VISITOR_H_

#include "pypto/ir/stmt.h"
#include "pypto/ir/transform/base/functor.h"

namespace pypto {
namespace ir {

/**
 * @brief Read-only IR visitor for both expressions and statements
 *
 * Provides default implementations that recursively traverse the IR tree.
 * Subclasses can override specific VisitExpr_ or VisitStmt_ methods to implement custom behavior.
 * All methods don't modify the visited IR nodes.
 */
class IRVisitor : public IRFunctor<void> {
 public:
  ~IRVisitor() override = default;

  void VisitExpr(const ExprPtr& expr) override;
  void VisitStmt(const StmtPtr& stmt) override;

 protected:
  // Leaf nodes - no children to visit
  void VisitExpr_(const VarPtr& op) override;
  void VisitExpr_(const IterArgPtr& op) override;
  void VisitExpr_(const ConstIntPtr& op) override;
  void VisitExpr_(const ConstFloatPtr& op) override;
  void VisitExpr_(const ConstBoolPtr& op) override;
  void VisitExpr_(const CallPtr& op) override;
  void VisitExpr_(const TupleGetItemExprPtr& op) override;

  // Binary operations - visit left and right children
  void VisitExpr_(const AddPtr& op) override;
  void VisitExpr_(const SubPtr& op) override;
  void VisitExpr_(const MulPtr& op) override;
  void VisitExpr_(const FloorDivPtr& op) override;
  void VisitExpr_(const FloorModPtr& op) override;
  void VisitExpr_(const FloatDivPtr& op) override;
  void VisitExpr_(const MinPtr& op) override;
  void VisitExpr_(const MaxPtr& op) override;
  void VisitExpr_(const PowPtr& op) override;
  void VisitExpr_(const EqPtr& op) override;
  void VisitExpr_(const NePtr& op) override;
  void VisitExpr_(const LtPtr& op) override;
  void VisitExpr_(const LePtr& op) override;
  void VisitExpr_(const GtPtr& op) override;
  void VisitExpr_(const GePtr& op) override;
  void VisitExpr_(const AndPtr& op) override;
  void VisitExpr_(const OrPtr& op) override;
  void VisitExpr_(const XorPtr& op) override;
  void VisitExpr_(const BitAndPtr& op) override;
  void VisitExpr_(const BitOrPtr& op) override;
  void VisitExpr_(const BitXorPtr& op) override;
  void VisitExpr_(const BitShiftLeftPtr& op) override;
  void VisitExpr_(const BitShiftRightPtr& op) override;

  // Unary operations - visit operand
  void VisitExpr_(const AbsPtr& op) override;
  void VisitExpr_(const NegPtr& op) override;
  void VisitExpr_(const NotPtr& op) override;
  void VisitExpr_(const BitNotPtr& op) override;
  void VisitExpr_(const CastPtr& op) override;

  // Statement types
  void VisitStmt_(const AssignStmtPtr& op) override;
  void VisitStmt_(const IfStmtPtr& op) override;
  void VisitStmt_(const YieldStmtPtr& op) override;
  void VisitStmt_(const ReturnStmtPtr& op) override;
  void VisitStmt_(const ForStmtPtr& op) override;
  void VisitStmt_(const SeqStmtsPtr& op) override;
  void VisitStmt_(const OpStmtsPtr& op) override;
  void VisitStmt_(const StmtPtr& op) override;

 private:
  /**
   * @brief Helper to visit both children of a binary expression
   */
  void VisitBinaryOp_(const BinaryExprPtr& op);

  /**
   * @brief Helper to visit the operand of a unary expression
   */
  void VisitUnaryOp_(const UnaryExprPtr& op);
};

}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORM_BASE_VISITOR_H_
