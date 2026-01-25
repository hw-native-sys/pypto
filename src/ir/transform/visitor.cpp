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
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

void IRVisitor::VisitExpr(const ExprPtr& expr) { ExprFunctor<void>::VisitExpr(expr); }

void IRVisitor::VisitStmt(const StmtPtr& stmt) { StmtFunctor<void>::VisitStmt(stmt); }

// Leaf nodes - no children to visit
void IRVisitor::VisitExpr_(const VarPtr& op) {
  // Visit type if it's a TensorType (to visit shape expressions)
  if (auto tensor_type = As<TensorType>(op->GetType())) {
    for (const auto& dim : tensor_type->shape_) {
      VisitExpr(dim);
    }
  }
}

void IRVisitor::VisitExpr_(const IterArgPtr& op) {
  // Visit initValue as Expr
  INTERNAL_CHECK(op->initValue_) << "IterArg has null initValue";
  VisitExpr(op->initValue_);
  // Also visit type if it's a TensorType (inherited from Var)
  if (auto tensor_type = As<TensorType>(op->GetType())) {
    for (const auto& dim : tensor_type->shape_) {
      VisitExpr(dim);
    }
  }
}

void IRVisitor::VisitExpr_(const ConstIntPtr& op) {
  // Leaf node, no children to visit
}

void IRVisitor::VisitExpr_(const ConstFloatPtr& op) {
  // Leaf node, no children to visit
}

void IRVisitor::VisitExpr_(const ConstBoolPtr& op) {
  // Leaf node, no children to visit
}

void IRVisitor::VisitExpr_(const CallPtr& op) {
  // Visit all arguments
  for (size_t i = 0; i < op->args_.size(); ++i) {
    INTERNAL_CHECK(op->args_[i]) << "Call has null argument at index " << i;
    VisitExpr(op->args_[i]);
  }
}

void IRVisitor::VisitExpr_(const TupleGetItemExprPtr& op) {
  // Visit the tuple expression
  INTERNAL_CHECK(op->tuple_) << "TupleGetItemExpr has null tuple";
  VisitExpr(op->tuple_);
}

// Macro to generate binary visitor with null checks
#define DEFINE_BINARY_VISITOR(OpType)                                \
  void IRVisitor::VisitExpr_(const OpType##Ptr& op) {                \
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
  void IRVisitor::VisitExpr_(const OpType##Ptr& op) {            \
    INTERNAL_CHECK(op->operand_) << #OpType " has null operand"; \
    VisitExpr(op->operand_);                                     \
  }

// Unary operations
DEFINE_UNARY_VISITOR(Abs)
DEFINE_UNARY_VISITOR(Neg)
DEFINE_UNARY_VISITOR(Not)
DEFINE_UNARY_VISITOR(BitNot)
DEFINE_UNARY_VISITOR(Cast)

#undef DEFINE_UNARY_VISITOR

// Statement types
void IRVisitor::VisitStmt_(const AssignStmtPtr& op) {
  INTERNAL_CHECK(op->var_) << "AssignStmt has null var";
  INTERNAL_CHECK(op->value_) << "AssignStmt has null value";
  VisitExpr(op->var_);
  VisitExpr(op->value_);
}

void IRVisitor::VisitStmt_(const IfStmtPtr& op) {
  INTERNAL_CHECK(op->condition_) << "IfStmt has null condition";
  VisitExpr(op->condition_);
  INTERNAL_CHECK(op->then_body_) << "IfStmt has null then_body";
  VisitStmt(op->then_body_);
  if (op->else_body_.has_value()) {
    INTERNAL_CHECK(*op->else_body_) << "IfStmt has null else_body";
    VisitStmt(*op->else_body_);
  }
  for (size_t i = 0; i < op->return_vars_.size(); ++i) {
    INTERNAL_CHECK(op->return_vars_[i]) << "IfStmt has null return_vars at index " << i;
    VisitExpr(op->return_vars_[i]);
  }
}

void IRVisitor::VisitStmt_(const YieldStmtPtr& op) {
  for (size_t i = 0; i < op->value_.size(); ++i) {
    INTERNAL_CHECK(op->value_[i]) << "YieldStmt has null value at index " << i;
    VisitExpr(op->value_[i]);
  }
}

void IRVisitor::VisitStmt_(const ReturnStmtPtr& op) {
  for (size_t i = 0; i < op->value_.size(); ++i) {
    INTERNAL_CHECK(op->value_[i]) << "ReturnStmt has null value at index " << i;
    VisitExpr(op->value_[i]);
  }
}

void IRVisitor::VisitStmt_(const ForStmtPtr& op) {
  INTERNAL_CHECK(op->loop_var_) << "ForStmt has null loop_var";
  INTERNAL_CHECK(op->start_) << "ForStmt has null start";
  INTERNAL_CHECK(op->stop_) << "ForStmt has null stop";
  INTERNAL_CHECK(op->step_) << "ForStmt has null step";
  VisitExpr(op->loop_var_);
  VisitExpr(op->start_);
  VisitExpr(op->stop_);
  VisitExpr(op->step_);
  for (size_t i = 0; i < op->iter_args_.size(); ++i) {
    INTERNAL_CHECK(op->iter_args_[i]) << "ForStmt has null iter_args at index " << i;
    VisitExpr(op->iter_args_[i]);
  }
  INTERNAL_CHECK(op->body_) << "ForStmt has null body";
  VisitStmt(op->body_);
  for (size_t i = 0; i < op->return_vars_.size(); ++i) {
    INTERNAL_CHECK(op->return_vars_[i]) << "ForStmt has null return_vars at index " << i;
    VisitExpr(op->return_vars_[i]);
  }
}

void IRVisitor::VisitStmt_(const SeqStmtsPtr& op) {
  for (size_t i = 0; i < op->stmts_.size(); ++i) {
    INTERNAL_CHECK(op->stmts_[i]) << "SeqStmts has null statement at index " << i;
    VisitStmt(op->stmts_[i]);
  }
}

void IRVisitor::VisitStmt_(const OpStmtsPtr& op) {
  for (size_t i = 0; i < op->stmts_.size(); ++i) {
    INTERNAL_CHECK(op->stmts_[i]) << "OpStmts has null assignment statement at index " << i;
    VisitStmt(op->stmts_[i]);
  }
}

void IRVisitor::VisitStmt_(const EvalStmtPtr& op) {
  INTERNAL_CHECK(op->expr_) << "EvalStmt has null expr";
  VisitExpr(op->expr_);
}

void IRVisitor::VisitStmt_(const StmtPtr& op) {
  // Base Stmt has no children to visit
}

}  // namespace ir
}  // namespace pypto
