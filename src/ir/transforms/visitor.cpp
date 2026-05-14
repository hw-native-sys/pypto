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

#include "pypto/ir/transforms/base/visitor.h"

#include <any>
#include <cstddef>
#include <vector>

#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/functor.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

// Top-level entry points
void IRVisitor::VisitProgram(const ProgramPtr& program) {
  for (auto& [gv, func] : program->functions_) {
    VisitFunction(func);
  }
}

void IRVisitor::VisitFunction(const FunctionPtr& func) {
  for (auto& param : func->params_) {
    VisitExpr(param);
  }
  if (func->body_) {
    VisitStmt(func->body_);
  }
}

void IRVisitor::VisitExpr(const ExprPtr& expr) { ExprFunctor<void>::VisitExpr(expr); }

void IRVisitor::VisitStmt(const StmtPtr& stmt) { StmtFunctor<void>::VisitStmt(stmt); }

void IRVisitor::VisitVarLike_(const VarPtr& op) {
  if (auto tensor_type = As<TensorType>(op->GetType())) {
    for (const auto& dim : tensor_type->shape_) {
      VisitExpr(dim);
    }
  }
}

void IRVisitor::VisitExpr_(const VarPtr& op) { VisitVarLike_(op); }

void IRVisitor::VisitExpr_(const IterArgPtr& op) {
  VisitVarLike_(op);
  INTERNAL_CHECK_SPAN(op->initValue_, op->span_) << "IterArg has null initValue";
  VisitExpr(op->initValue_);
}

void IRVisitor::VisitExpr_(const MemRefPtr& op) {}

void IRVisitor::VisitExpr_(const ConstIntPtr& op) {}

void IRVisitor::VisitExpr_(const ConstFloatPtr& op) {}

void IRVisitor::VisitExpr_(const ConstBoolPtr& op) {}

void IRVisitor::VisitExpr_(const CallPtr& op) {
  for (size_t i = 0; i < op->args_.size(); ++i) {
    INTERNAL_CHECK_SPAN(op->args_[i], op->span_) << "Call has null argument at index " << i;
    VisitExpr(op->args_[i]);
  }
  // Var-typed attr ``manual_dep_edges`` references Vars defined elsewhere
  // in the IR. Treat them as real uses so analyses such as the unused-variable
  // check don't flag a Var that is referenced only via ``deps=[tid]``.
  for (const auto& [k, v] : op->attrs_) {
    if (k != kAttrManualDepEdges) continue;
    const auto* edges = std::any_cast<std::vector<VarPtr>>(&v);
    if (!edges) continue;
    for (const auto& e : *edges) {
      if (e) VisitExpr(e);
    }
  }
}

void IRVisitor::VisitExpr_(const MakeTuplePtr& op) {
  for (size_t i = 0; i < op->elements_.size(); ++i) {
    INTERNAL_CHECK_SPAN(op->elements_[i], op->span_) << "MakeTuple has null element at index " << i;
    VisitExpr(op->elements_[i]);
  }
}

void IRVisitor::VisitExpr_(const TupleGetItemExprPtr& op) {
  INTERNAL_CHECK_SPAN(op->tuple_, op->span_) << "TupleGetItemExpr has null tuple";
  VisitExpr(op->tuple_);
}

void IRVisitor::VisitBinaryExpr_(const BinaryExprPtr& op) {
  INTERNAL_CHECK_SPAN(op->left_, op->span_) << "BinaryExpr has null left operand";
  INTERNAL_CHECK_SPAN(op->right_, op->span_) << "BinaryExpr has null right operand";
  VisitExpr(op->left_);
  VisitExpr(op->right_);
}

void IRVisitor::VisitUnaryExpr_(const UnaryExprPtr& op) {
  INTERNAL_CHECK_SPAN(op->operand_, op->span_) << "UnaryExpr has null operand";
  VisitExpr(op->operand_);
}

#define DEFINE_BINARY_VISITOR(OpType) \
  void IRVisitor::VisitExpr_(const OpType##Ptr& op) { VisitBinaryExpr_(op); }

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

#define DEFINE_UNARY_VISITOR(OpType) \
  void IRVisitor::VisitExpr_(const OpType##Ptr& op) { VisitUnaryExpr_(op); }

DEFINE_UNARY_VISITOR(Abs)
DEFINE_UNARY_VISITOR(Neg)
DEFINE_UNARY_VISITOR(Not)
DEFINE_UNARY_VISITOR(BitNot)
DEFINE_UNARY_VISITOR(Cast)

#undef DEFINE_UNARY_VISITOR

void IRVisitor::VisitStmt_(const AssignStmtPtr& op) {
  INTERNAL_CHECK_SPAN(op->var_, op->span_) << "AssignStmt has null var";
  INTERNAL_CHECK_SPAN(op->value_, op->span_) << "AssignStmt has null value";
  VisitExpr(op->var_);
  VisitExpr(op->value_);
}

void IRVisitor::VisitStmt_(const IfStmtPtr& op) {
  INTERNAL_CHECK_SPAN(op->condition_, op->span_) << "IfStmt has null condition";
  VisitExpr(op->condition_);
  INTERNAL_CHECK_SPAN(op->then_body_, op->span_) << "IfStmt has null then_body";
  VisitStmt(op->then_body_);
  if (op->else_body_.has_value()) {
    INTERNAL_CHECK_SPAN(*op->else_body_, op->span_) << "IfStmt has null else_body";
    VisitStmt(*op->else_body_);
  }
  for (size_t i = 0; i < op->return_vars_.size(); ++i) {
    INTERNAL_CHECK_SPAN(op->return_vars_[i], op->span_) << "IfStmt has null return_vars at index " << i;
    VisitExpr(op->return_vars_[i]);
  }
}

void IRVisitor::VisitStmt_(const YieldStmtPtr& op) {
  for (size_t i = 0; i < op->value_.size(); ++i) {
    INTERNAL_CHECK_SPAN(op->value_[i], op->span_) << "YieldStmt has null value at index " << i;
    VisitExpr(op->value_[i]);
  }
}

void IRVisitor::VisitStmt_(const ReturnStmtPtr& op) {
  for (size_t i = 0; i < op->value_.size(); ++i) {
    INTERNAL_CHECK_SPAN(op->value_[i], op->span_) << "ReturnStmt has null value at index " << i;
    VisitExpr(op->value_[i]);
  }
}

void IRVisitor::VisitStmt_(const ForStmtPtr& op) {
  INTERNAL_CHECK_SPAN(op->loop_var_, op->span_) << "ForStmt has null loop_var";
  INTERNAL_CHECK_SPAN(op->start_, op->span_) << "ForStmt has null start";
  INTERNAL_CHECK_SPAN(op->stop_, op->span_) << "ForStmt has null stop";
  INTERNAL_CHECK_SPAN(op->step_, op->span_) << "ForStmt has null step";
  VisitExpr(op->loop_var_);
  VisitExpr(op->start_);
  VisitExpr(op->stop_);
  VisitExpr(op->step_);
  for (size_t i = 0; i < op->iter_args_.size(); ++i) {
    INTERNAL_CHECK_SPAN(op->iter_args_[i], op->span_) << "ForStmt has null iter_args at index " << i;
    VisitExpr(op->iter_args_[i]);
  }
  INTERNAL_CHECK_SPAN(op->body_, op->span_) << "ForStmt has null body";
  VisitStmt(op->body_);
  for (size_t i = 0; i < op->return_vars_.size(); ++i) {
    INTERNAL_CHECK_SPAN(op->return_vars_[i], op->span_) << "ForStmt has null return_vars at index " << i;
    VisitExpr(op->return_vars_[i]);
  }
}

void IRVisitor::VisitStmt_(const WhileStmtPtr& op) {
  INTERNAL_CHECK_SPAN(op->condition_, op->span_) << "WhileStmt has null condition";
  VisitExpr(op->condition_);
  for (size_t i = 0; i < op->iter_args_.size(); ++i) {
    INTERNAL_CHECK_SPAN(op->iter_args_[i], op->span_) << "WhileStmt has null iter_args at index " << i;
    VisitExpr(op->iter_args_[i]);
  }
  INTERNAL_CHECK_SPAN(op->body_, op->span_) << "WhileStmt has null body";
  VisitStmt(op->body_);
  for (size_t i = 0; i < op->return_vars_.size(); ++i) {
    INTERNAL_CHECK_SPAN(op->return_vars_[i], op->span_) << "WhileStmt has null return_vars at index " << i;
    VisitExpr(op->return_vars_[i]);
  }
}

void IRVisitor::VisitStmt_(const InCoreScopeStmtPtr& op) {
  INTERNAL_CHECK_SPAN(op->body_, op->span_) << "InCoreScopeStmt has null body";
  VisitStmt(op->body_);
}

void IRVisitor::VisitStmt_(const AutoInCoreScopeStmtPtr& op) {
  INTERNAL_CHECK_SPAN(op->body_, op->span_) << "AutoInCoreScopeStmt has null body";
  VisitStmt(op->body_);
}

void IRVisitor::VisitStmt_(const ClusterScopeStmtPtr& op) {
  INTERNAL_CHECK_SPAN(op->body_, op->span_) << "ClusterScopeStmt has null body";
  VisitStmt(op->body_);
}

void IRVisitor::VisitStmt_(const HierarchyScopeStmtPtr& op) {
  INTERNAL_CHECK_SPAN(op->body_, op->span_) << "HierarchyScopeStmt has null body";
  VisitStmt(op->body_);
}

void IRVisitor::VisitStmt_(const SpmdScopeStmtPtr& op) {
  INTERNAL_CHECK_SPAN(op->core_num_, op->span_) << "SpmdScopeStmt has null core_num";
  VisitExpr(op->core_num_);
  INTERNAL_CHECK_SPAN(op->body_, op->span_) << "SpmdScopeStmt has null body";
  VisitStmt(op->body_);
}

void IRVisitor::VisitStmt_(const RuntimeScopeStmtPtr& op) {
  INTERNAL_CHECK_SPAN(op->body_, op->span_) << "RuntimeScopeStmt has null body";
  VisitStmt(op->body_);
}

void IRVisitor::VisitStmt_(const SeqStmtsPtr& op) {
  for (size_t i = 0; i < op->stmts_.size(); ++i) {
    INTERNAL_CHECK_SPAN(op->stmts_[i], op->span_) << "SeqStmts has null statement at index " << i;
    VisitStmt(op->stmts_[i]);
  }
}

void IRVisitor::VisitStmt_(const EvalStmtPtr& op) {
  INTERNAL_CHECK_SPAN(op->expr_, op->span_) << "EvalStmt has null expr";
  VisitExpr(op->expr_);
}

void IRVisitor::VisitStmt_(const BreakStmtPtr& op) {}

void IRVisitor::VisitStmt_(const ContinueStmtPtr& op) {}

void IRVisitor::VisitStmt_(const InlineStmtPtr& op) {}

void IRVisitor::VisitStmt_(const StmtPtr& op) {}

}  // namespace ir
}  // namespace pypto
