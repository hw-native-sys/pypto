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

#include "pypto/ir/transform/base/mutator.h"

#include <memory>
#include <utility>
#include <vector>

#include "pypto/core/logging.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

ExprPtr IRMutator::VisitExpr(const ExprPtr& expr) {
  // Call the base class VisitExpr which returns ExprPtr
  return ExprFunctor<ExprPtr>::VisitExpr(expr);
}

StmtPtr IRMutator::VisitStmt(const StmtPtr& stmt) {
  // Call the base class VisitStmt which returns StmtPtr
  return StmtFunctor<StmtPtr>::VisitStmt(stmt);
}

// Leaf nodes - return original shared_ptr (immutable)
ExprPtr IRMutator::VisitExpr_(const VarPtr& op) {
  // Var is immutable, return original
  return op;
}

ExprPtr IRMutator::VisitExpr_(const IterArgPtr& op) {
  // Visit initValue as Expr and value as Var
  INTERNAL_CHECK(op->initValue_) << "IterArg has null initValue";
  INTERNAL_CHECK(op->value_) << "IterArg has null value";
  auto new_init_value = ExprFunctor<ExprPtr>::VisitExpr(op->initValue_);
  INTERNAL_CHECK(new_init_value) << "IterArg initValue mutated to null";
  auto new_value_expr = ExprFunctor<ExprPtr>::VisitExpr(op->value_);
  INTERNAL_CHECK(new_value_expr) << "IterArg value mutated to null";
  auto new_value = std::dynamic_pointer_cast<const Var>(new_value_expr);
  INTERNAL_CHECK(new_value) << "IterArg value is not a Var after mutation";

  // Copy-on-write: only create new node if children changed
  if (new_init_value.get() != op->initValue_.get() || new_value.get() != op->value_.get()) {
    return std::make_shared<const IterArg>(op->name_, op->GetType(), std::move(new_init_value),
                                           std::move(new_value), op->span_);
  } else {
    return op;
  }
}

ExprPtr IRMutator::VisitExpr_(const ConstIntPtr& op) {
  // ConstInt is immutable, return original
  return op;
}

ExprPtr IRMutator::VisitExpr_(const CallPtr& op) {
  // Visit all arguments
  std::vector<ExprPtr> new_args;
  bool changed = false;
  new_args.reserve(op->args_.size());

  for (size_t i = 0; i < op->args_.size(); ++i) {
    INTERNAL_CHECK(op->args_[i]) << "Call has null argument at index " << i;
    auto new_arg = ExprFunctor<ExprPtr>::VisitExpr(op->args_[i]);
    INTERNAL_CHECK(new_arg) << "Call argument at index " << i << " mutated to null";
    new_args.push_back(new_arg);
    if (new_arg.get() != op->args_[i].get()) {
      changed = true;
    }
  }

  // Copy-on-write: only create new node if arguments changed
  if (changed) {
    return std::make_shared<const Call>(op->op_, std::move(new_args), op->span_);
  } else {
    return op;
  }
}

// Macro to generate binary operation mutators with copy-on-write
#define DEFINE_BINARY_MUTATOR(OpType)                                                              \
  ExprPtr IRMutator::VisitExpr_(const OpType##Ptr& op) {                                           \
    INTERNAL_CHECK(op->left_) << #OpType " has null left operand";                                 \
    INTERNAL_CHECK(op->right_) << #OpType " has null right operand";                               \
    auto new_left = ExprFunctor<ExprPtr>::VisitExpr(op->left_);                                    \
    auto new_right = ExprFunctor<ExprPtr>::VisitExpr(op->right_);                                  \
    INTERNAL_CHECK(new_left) << #OpType " left operand mutated to null";                           \
    INTERNAL_CHECK(new_right) << #OpType " right operand mutated to null";                         \
    if (new_left.get() != op->left_.get() || new_right.get() != op->right_.get()) {                \
      return std::make_shared<const OpType>(std::move(new_left), std::move(new_right), op->dtype_, \
                                            op->span_);                                            \
    } else {                                                                                       \
      return op;                                                                                   \
    }                                                                                              \
  }

// Binary operations
DEFINE_BINARY_MUTATOR(Add)
DEFINE_BINARY_MUTATOR(Sub)
DEFINE_BINARY_MUTATOR(Mul)
DEFINE_BINARY_MUTATOR(FloorDiv)
DEFINE_BINARY_MUTATOR(FloorMod)
DEFINE_BINARY_MUTATOR(FloatDiv)
DEFINE_BINARY_MUTATOR(Min)
DEFINE_BINARY_MUTATOR(Max)
DEFINE_BINARY_MUTATOR(Pow)
DEFINE_BINARY_MUTATOR(Eq)
DEFINE_BINARY_MUTATOR(Ne)
DEFINE_BINARY_MUTATOR(Lt)
DEFINE_BINARY_MUTATOR(Le)
DEFINE_BINARY_MUTATOR(Gt)
DEFINE_BINARY_MUTATOR(Ge)
DEFINE_BINARY_MUTATOR(And)
DEFINE_BINARY_MUTATOR(Or)
DEFINE_BINARY_MUTATOR(Xor)
DEFINE_BINARY_MUTATOR(BitAnd)
DEFINE_BINARY_MUTATOR(BitOr)
DEFINE_BINARY_MUTATOR(BitXor)
DEFINE_BINARY_MUTATOR(BitShiftLeft)
DEFINE_BINARY_MUTATOR(BitShiftRight)

#undef DEFINE_BINARY_MUTATOR

// Macro to generate unary operation mutators with copy-on-write
#define DEFINE_UNARY_MUTATOR(OpType)                                                        \
  ExprPtr IRMutator::VisitExpr_(const OpType##Ptr& op) {                                    \
    INTERNAL_CHECK(op->operand_) << #OpType " has null operand";                            \
    auto new_operand = ExprFunctor<ExprPtr>::VisitExpr(op->operand_);                       \
    INTERNAL_CHECK(new_operand) << #OpType " operand mutated to null";                      \
    if (new_operand.get() != op->operand_.get()) {                                          \
      return std::make_shared<const OpType>(std::move(new_operand), op->dtype_, op->span_); \
    } else {                                                                                \
      return op;                                                                            \
    }                                                                                       \
  }

// Unary operations
DEFINE_UNARY_MUTATOR(Abs)
DEFINE_UNARY_MUTATOR(Neg)
DEFINE_UNARY_MUTATOR(Not)
DEFINE_UNARY_MUTATOR(BitNot)

#undef DEFINE_UNARY_MUTATOR

// Statement types
StmtPtr IRMutator::VisitStmt_(const AssignStmtPtr& op) {
  INTERNAL_CHECK(op->var_) << "AssignStmt has null var";
  INTERNAL_CHECK(op->value_) << "AssignStmt has null value";
  auto new_var_expr = ExprFunctor<ExprPtr>::VisitExpr(op->var_);
  auto new_value = ExprFunctor<ExprPtr>::VisitExpr(op->value_);
  INTERNAL_CHECK(new_var_expr) << "AssignStmt var mutated to null";
  INTERNAL_CHECK(new_value) << "AssignStmt value mutated to null";
  // Cast new_var from ExprPtr to VarPtr (required by AssignStmt constructor)
  auto new_var = std::dynamic_pointer_cast<const Var>(new_var_expr);
  INTERNAL_CHECK(new_var) << "AssignStmt var is not a Var after mutation";
  if (new_var.get() != op->var_.get() || new_value.get() != op->value_.get()) {
    return std::make_shared<const AssignStmt>(std::move(new_var), std::move(new_value), op->span_);
  } else {
    return op;
  }
}

StmtPtr IRMutator::VisitStmt_(const IfStmtPtr& op) {
  INTERNAL_CHECK(op->condition_) << "IfStmt has null condition";
  auto new_condition = ExprFunctor<ExprPtr>::VisitExpr(op->condition_);
  INTERNAL_CHECK(new_condition) << "IfStmt condition mutated to null";

  INTERNAL_CHECK(op->then_body_) << "IfStmt has null then_body";
  auto new_then_body = StmtFunctor<StmtPtr>::VisitStmt(op->then_body_);
  INTERNAL_CHECK(new_then_body) << "IfStmt then_body mutated to null";
  bool then_changed = (new_then_body.get() != op->then_body_.get());

  std::optional<StmtPtr> new_else_body;
  bool else_changed = false;
  if (op->else_body_.has_value()) {
    INTERNAL_CHECK(*op->else_body_) << "IfStmt has null else_body";
    auto new_stmt = StmtFunctor<StmtPtr>::VisitStmt(*op->else_body_);
    INTERNAL_CHECK(new_stmt) << "IfStmt else_body mutated to null";
    new_else_body = new_stmt;
    if (new_stmt.get() != op->else_body_->get()) {
      else_changed = true;
    }
  }

  std::vector<VarPtr> new_return_vars;
  bool return_vars_changed = false;
  new_return_vars.reserve(op->return_vars_.size());
  for (size_t i = 0; i < op->return_vars_.size(); ++i) {
    INTERNAL_CHECK(op->return_vars_[i]) << "IfStmt has null return_vars at index " << i;
    auto new_var_expr = ExprFunctor<ExprPtr>::VisitExpr(op->return_vars_[i]);
    INTERNAL_CHECK(new_var_expr) << "IfStmt return_vars at index " << i << " mutated to null";
    // Cast new_var from ExprPtr to VarPtr (required by IfStmt constructor)
    auto new_var = std::dynamic_pointer_cast<const Var>(new_var_expr);
    INTERNAL_CHECK(new_var) << "IfStmt return_vars at index " << i << " is not a Var after mutation";
    new_return_vars.push_back(new_var);
    if (new_var.get() != op->return_vars_[i].get()) {
      return_vars_changed = true;
    }
  }

  if (new_condition.get() != op->condition_.get() || then_changed || else_changed || return_vars_changed) {
    if (new_else_body.has_value()) {
      return std::make_shared<const IfStmt>(std::move(new_condition), std::move(new_then_body),
                                            *new_else_body, std::move(new_return_vars), op->span_);
    } else {
      return std::make_shared<const IfStmt>(std::move(new_condition), std::move(new_then_body), std::nullopt,
                                            std::move(new_return_vars), op->span_);
    }
  } else {
    return op;
  }
}

StmtPtr IRMutator::VisitStmt_(const YieldStmtPtr& op) {
  std::vector<VarPtr> new_value;
  bool changed = false;
  new_value.reserve(op->value_.size());

  for (size_t i = 0; i < op->value_.size(); ++i) {
    INTERNAL_CHECK(op->value_[i]) << "YieldStmt has null value at index " << i;
    auto new_var_expr = ExprFunctor<ExprPtr>::VisitExpr(op->value_[i]);
    INTERNAL_CHECK(new_var_expr) << "YieldStmt value at index " << i << " mutated to null";
    // Cast new_var from ExprPtr to VarPtr (required by YieldStmt constructor)
    auto new_var = std::dynamic_pointer_cast<const Var>(new_var_expr);
    INTERNAL_CHECK(new_var) << "YieldStmt value at index " << i << " is not a Var after mutation";
    new_value.push_back(new_var);
    if (new_var.get() != op->value_[i].get()) {
      changed = true;
    }
  }

  if (changed) {
    return std::make_shared<const YieldStmt>(std::move(new_value), op->span_);
  } else {
    return op;
  }
}

StmtPtr IRMutator::VisitStmt_(const ForStmtPtr& op) {
  INTERNAL_CHECK(op->loop_var_) << "ForStmt has null loop_var";
  INTERNAL_CHECK(op->start_) << "ForStmt has null start";
  INTERNAL_CHECK(op->stop_) << "ForStmt has null stop";
  INTERNAL_CHECK(op->step_) << "ForStmt has null step";
  auto new_loop_var_expr = ExprFunctor<ExprPtr>::VisitExpr(op->loop_var_);
  INTERNAL_CHECK(new_loop_var_expr) << "ForStmt loop_var mutated to null";
  auto new_loop_var = std::dynamic_pointer_cast<const Var>(new_loop_var_expr);
  INTERNAL_CHECK(new_loop_var) << "ForStmt loop_var is not a Var after mutation";

  auto new_start = ExprFunctor<ExprPtr>::VisitExpr(op->start_);
  INTERNAL_CHECK(new_start) << "ForStmt start mutated to null";

  auto new_stop = ExprFunctor<ExprPtr>::VisitExpr(op->stop_);
  INTERNAL_CHECK(new_stop) << "ForStmt stop mutated to null";

  auto new_step = ExprFunctor<ExprPtr>::VisitExpr(op->step_);
  INTERNAL_CHECK(new_step) << "ForStmt step mutated to null";

  std::vector<IterArgPtr> new_iter_args;
  bool iter_args_changed = false;
  new_iter_args.reserve(op->iter_args_.size());
  for (size_t i = 0; i < op->iter_args_.size(); ++i) {
    INTERNAL_CHECK(op->iter_args_[i]) << "ForStmt has null iter_args at index " << i;
    auto new_iter_arg_expr = ExprFunctor<ExprPtr>::VisitExpr(op->iter_args_[i]);
    INTERNAL_CHECK(new_iter_arg_expr) << "ForStmt iter_args at index " << i << " mutated to null";
    auto new_iter_arg = std::dynamic_pointer_cast<const IterArg>(new_iter_arg_expr);
    INTERNAL_CHECK(new_iter_arg) << "ForStmt iter_args at index " << i << " is not an IterArg after mutation";
    new_iter_args.push_back(new_iter_arg);
    if (new_iter_arg.get() != op->iter_args_[i].get()) {
      iter_args_changed = true;
    }
  }

  INTERNAL_CHECK(op->body_) << "ForStmt has null body";
  auto new_body = StmtFunctor<StmtPtr>::VisitStmt(op->body_);
  INTERNAL_CHECK(new_body) << "ForStmt body mutated to null";
  bool body_changed = (new_body.get() != op->body_.get());

  std::vector<VarPtr> new_return_vars;
  bool return_vars_changed = false;
  new_return_vars.reserve(op->return_vars_.size());
  for (size_t i = 0; i < op->return_vars_.size(); ++i) {
    INTERNAL_CHECK(op->return_vars_[i]) << "ForStmt has null return_vars at index " << i;
    auto new_var_expr = ExprFunctor<ExprPtr>::VisitExpr(op->return_vars_[i]);
    INTERNAL_CHECK(new_var_expr) << "ForStmt return_vars at index " << i << " mutated to null";
    // Cast new_var from ExprPtr to VarPtr (required by ForStmt constructor)
    auto new_var = std::dynamic_pointer_cast<const Var>(new_var_expr);
    INTERNAL_CHECK(new_var) << "ForStmt return_vars at index " << i << " is not a Var after mutation";
    new_return_vars.push_back(new_var);
    if (new_var.get() != op->return_vars_[i].get()) {
      return_vars_changed = true;
    }
  }

  if (new_loop_var.get() != op->loop_var_.get() || new_start.get() != op->start_.get() ||
      new_stop.get() != op->stop_.get() || new_step.get() != op->step_.get() || iter_args_changed ||
      body_changed || return_vars_changed) {
    return std::make_shared<const ForStmt>(std::move(new_loop_var), std::move(new_start), std::move(new_stop),
                                           std::move(new_step), std::move(new_iter_args), std::move(new_body),
                                           std::move(new_return_vars), op->span_);
  } else {
    return op;
  }
}

StmtPtr IRMutator::VisitStmt_(const SeqStmtsPtr& op) {
  std::vector<StmtPtr> new_stmts;
  bool changed = false;
  new_stmts.reserve(op->stmts_.size());
  for (size_t i = 0; i < op->stmts_.size(); ++i) {
    INTERNAL_CHECK(op->stmts_[i]) << "SeqStmts has null statement at index " << i;
    auto new_stmt = StmtFunctor<StmtPtr>::VisitStmt(op->stmts_[i]);
    INTERNAL_CHECK(new_stmt) << "SeqStmts statement at index " << i << " mutated to null";
    new_stmts.push_back(new_stmt);
    if (new_stmt.get() != op->stmts_[i].get()) {
      changed = true;
    }
  }

  if (changed) {
    return std::make_shared<const SeqStmts>(std::move(new_stmts), op->span_);
  } else {
    return op;
  }
}

StmtPtr IRMutator::VisitStmt_(const OpStmtsPtr& op) {
  std::vector<AssignStmtPtr> new_stmts;
  bool changed = false;
  new_stmts.reserve(op->stmts_.size());
  for (size_t i = 0; i < op->stmts_.size(); ++i) {
    INTERNAL_CHECK(op->stmts_[i]) << "OpStmts has null assignment statement at index " << i;
    auto new_stmt = StmtFunctor<StmtPtr>::VisitStmt(op->stmts_[i]);
    INTERNAL_CHECK(new_stmt) << "OpStmts assignment statement at index " << i << " mutated to null";
    // Cast to AssignStmtPtr (required by OpStmts constructor)
    auto new_assign_stmt = std::dynamic_pointer_cast<const AssignStmt>(new_stmt);
    INTERNAL_CHECK(new_assign_stmt) << "OpStmts statement at index " << i
                                    << " is not an AssignStmt after mutation";
    new_stmts.push_back(new_assign_stmt);
    if (new_assign_stmt.get() != op->stmts_[i].get()) {
      changed = true;
    }
  }

  if (changed) {
    return std::make_shared<const OpStmts>(std::move(new_stmts), op->span_);
  } else {
    return op;
  }
}

StmtPtr IRMutator::VisitStmt_(const StmtPtr& op) {
  // Base Stmt is immutable, return original
  return op;
}

}  // namespace ir
}  // namespace pypto
