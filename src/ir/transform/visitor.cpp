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

#include "pypto/ir/transform/base/visitor.h"

#include "pypto/core/logging.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/tensor_expr.h"

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
  for (size_t i = 0; i < op->args_.size(); ++i) {
    INTERNAL_CHECK(op->args_[i]) << "Call has null argument at index " << i;
    VisitExpr(op->args_[i]);
  }
}

// Macro to generate binary visitor with null checks
#define DEFINE_BINARY_VISITOR(OpType)                                \
  void ExprVisitor::VisitExpr_(const OpType##Ptr& op) {              \
    INTERNAL_CHECK(op->left_) << #OpType " has null left operand";   \
    INTERNAL_CHECK(op->right_) << #OpType " has null right operand"; \
    VisitExpr(op->left_);                                            \
    VisitExpr(op->right_);                                           \
  }

// Binary operations
DEFINE_BINARY_VISITOR(Add)
DEFINE_BINARY_VISITOR(Sub)
DEFINE_BINARY_VISITOR(Mul)
DEFINE_BINARY_VISITOR(FloorDiv)
DEFINE_BINARY_VISITOR(FloorMod)
DEFINE_BINARY_VISITOR(FloatDiv)
DEFINE_BINARY_VISITOR(Min)
DEFINE_BINARY_VISITOR(Max)
DEFINE_BINARY_VISITOR(Pow)
DEFINE_BINARY_VISITOR(Eq)
DEFINE_BINARY_VISITOR(Ne)
DEFINE_BINARY_VISITOR(Lt)
DEFINE_BINARY_VISITOR(Le)
DEFINE_BINARY_VISITOR(Gt)
DEFINE_BINARY_VISITOR(Ge)
DEFINE_BINARY_VISITOR(And)
DEFINE_BINARY_VISITOR(Or)
DEFINE_BINARY_VISITOR(Xor)
DEFINE_BINARY_VISITOR(BitAnd)
DEFINE_BINARY_VISITOR(BitOr)
DEFINE_BINARY_VISITOR(BitXor)
DEFINE_BINARY_VISITOR(BitShiftLeft)
DEFINE_BINARY_VISITOR(BitShiftRight)

#undef DEFINE_BINARY_VISITOR

// Macro to generate unary visitor with null checks
#define DEFINE_UNARY_VISITOR(OpType)                             \
  void ExprVisitor::VisitExpr_(const OpType##Ptr& op) {          \
    INTERNAL_CHECK(op->operand_) << #OpType " has null operand"; \
    VisitExpr(op->operand_);                                     \
  }

// Unary operations
DEFINE_UNARY_VISITOR(Abs)
DEFINE_UNARY_VISITOR(Neg)
DEFINE_UNARY_VISITOR(Not)
DEFINE_UNARY_VISITOR(BitNot)

#undef DEFINE_UNARY_VISITOR

// Tensor expressions
void ExprVisitor::VisitExpr_(const TensorVarPtr& op) {
  // Leaf node, but need to visit shape expressions
  for (const auto& dim : op->shape_) {
    VisitExpr(dim);
  }
}

}  // namespace ir
}  // namespace pypto
