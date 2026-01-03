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

#include "pypto/ir/transform/visitor.h"

#include "pypto/ir/scalar_expr.h"

namespace pypto {
namespace ir {

void ExprVisitor::VisitExpr(const ExprPtr& expr) { ExprFunctor::VisitExpr(expr); }

// Leaf nodes - no children to visit
void ExprVisitor::VisitExpr_(const VarPtr& op) {
  // Leaf node, no children to visit
}

void ExprVisitor::VisitExpr_(const ConstIntPtr& op) {
  // Leaf node, no children to visit
}

void ExprVisitor::VisitExpr_(const CallPtr& op) {
  // Visit all arguments
  for (const auto& arg : op->args_) {
    VisitExpr(arg);
  }
}

// Helper methods
void ExprVisitor::VisitBinaryOp_(const BinaryExprPtr& op) {
  VisitExpr(op->left_);
  VisitExpr(op->right_);
}

void ExprVisitor::VisitUnaryOp_(const UnaryExprPtr& op) { VisitExpr(op->operand_); }

// Binary operations - all use the same pattern
void ExprVisitor::VisitExpr_(const AddPtr& op) { VisitBinaryOp_(op); }
void ExprVisitor::VisitExpr_(const SubPtr& op) { VisitBinaryOp_(op); }
void ExprVisitor::VisitExpr_(const MulPtr& op) { VisitBinaryOp_(op); }
void ExprVisitor::VisitExpr_(const FloorDivPtr& op) { VisitBinaryOp_(op); }
void ExprVisitor::VisitExpr_(const FloorModPtr& op) { VisitBinaryOp_(op); }
void ExprVisitor::VisitExpr_(const FloatDivPtr& op) { VisitBinaryOp_(op); }
void ExprVisitor::VisitExpr_(const MinPtr& op) { VisitBinaryOp_(op); }
void ExprVisitor::VisitExpr_(const MaxPtr& op) { VisitBinaryOp_(op); }
void ExprVisitor::VisitExpr_(const PowPtr& op) { VisitBinaryOp_(op); }
void ExprVisitor::VisitExpr_(const EqPtr& op) { VisitBinaryOp_(op); }
void ExprVisitor::VisitExpr_(const NePtr& op) { VisitBinaryOp_(op); }
void ExprVisitor::VisitExpr_(const LtPtr& op) { VisitBinaryOp_(op); }
void ExprVisitor::VisitExpr_(const LePtr& op) { VisitBinaryOp_(op); }
void ExprVisitor::VisitExpr_(const GtPtr& op) { VisitBinaryOp_(op); }
void ExprVisitor::VisitExpr_(const GePtr& op) { VisitBinaryOp_(op); }
void ExprVisitor::VisitExpr_(const AndPtr& op) { VisitBinaryOp_(op); }
void ExprVisitor::VisitExpr_(const OrPtr& op) { VisitBinaryOp_(op); }
void ExprVisitor::VisitExpr_(const XorPtr& op) { VisitBinaryOp_(op); }
void ExprVisitor::VisitExpr_(const BitAndPtr& op) { VisitBinaryOp_(op); }
void ExprVisitor::VisitExpr_(const BitOrPtr& op) { VisitBinaryOp_(op); }
void ExprVisitor::VisitExpr_(const BitXorPtr& op) { VisitBinaryOp_(op); }
void ExprVisitor::VisitExpr_(const BitShiftLeftPtr& op) { VisitBinaryOp_(op); }
void ExprVisitor::VisitExpr_(const BitShiftRightPtr& op) { VisitBinaryOp_(op); }

// Unary operations - all use the same pattern
void ExprVisitor::VisitExpr_(const AbsPtr& op) { VisitUnaryOp_(op); }
void ExprVisitor::VisitExpr_(const NegPtr& op) { VisitUnaryOp_(op); }
void ExprVisitor::VisitExpr_(const NotPtr& op) { VisitUnaryOp_(op); }
void ExprVisitor::VisitExpr_(const BitNotPtr& op) { VisitUnaryOp_(op); }

}  // namespace ir
}  // namespace pypto
