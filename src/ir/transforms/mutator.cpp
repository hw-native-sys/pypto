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

#include "pypto/ir/transforms/base/mutator.h"

#include <cstddef>
#include <map>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/core.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memref.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/functor.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

namespace {

/// Reconstruct a binary expression with new children, preserving the concrete type.
/// All binary ops share the constructor signature (ExprPtr, ExprPtr, DataType, Span).
ExprPtr ReconstructBinaryExpr(ObjectKind kind, ExprPtr left, ExprPtr right, DataType dtype,
                              const Span& span) {
  switch (kind) {
    case ObjectKind::Add:
      return std::make_shared<const Add>(std::move(left), std::move(right), dtype, span);
    case ObjectKind::Sub:
      return std::make_shared<const Sub>(std::move(left), std::move(right), dtype, span);
    case ObjectKind::Mul:
      return std::make_shared<const Mul>(std::move(left), std::move(right), dtype, span);
    case ObjectKind::FloorDiv:
      return std::make_shared<const FloorDiv>(std::move(left), std::move(right), dtype, span);
    case ObjectKind::FloorMod:
      return std::make_shared<const FloorMod>(std::move(left), std::move(right), dtype, span);
    case ObjectKind::FloatDiv:
      return std::make_shared<const FloatDiv>(std::move(left), std::move(right), dtype, span);
    case ObjectKind::Min:
      return std::make_shared<const Min>(std::move(left), std::move(right), dtype, span);
    case ObjectKind::Max:
      return std::make_shared<const Max>(std::move(left), std::move(right), dtype, span);
    case ObjectKind::Pow:
      return std::make_shared<const Pow>(std::move(left), std::move(right), dtype, span);
    case ObjectKind::Eq:
      return std::make_shared<const Eq>(std::move(left), std::move(right), dtype, span);
    case ObjectKind::Ne:
      return std::make_shared<const Ne>(std::move(left), std::move(right), dtype, span);
    case ObjectKind::Lt:
      return std::make_shared<const Lt>(std::move(left), std::move(right), dtype, span);
    case ObjectKind::Le:
      return std::make_shared<const Le>(std::move(left), std::move(right), dtype, span);
    case ObjectKind::Gt:
      return std::make_shared<const Gt>(std::move(left), std::move(right), dtype, span);
    case ObjectKind::Ge:
      return std::make_shared<const Ge>(std::move(left), std::move(right), dtype, span);
    case ObjectKind::And:
      return std::make_shared<const And>(std::move(left), std::move(right), dtype, span);
    case ObjectKind::Or:
      return std::make_shared<const Or>(std::move(left), std::move(right), dtype, span);
    case ObjectKind::Xor:
      return std::make_shared<const Xor>(std::move(left), std::move(right), dtype, span);
    case ObjectKind::BitAnd:
      return std::make_shared<const BitAnd>(std::move(left), std::move(right), dtype, span);
    case ObjectKind::BitOr:
      return std::make_shared<const BitOr>(std::move(left), std::move(right), dtype, span);
    case ObjectKind::BitXor:
      return std::make_shared<const BitXor>(std::move(left), std::move(right), dtype, span);
    case ObjectKind::BitShiftLeft:
      return std::make_shared<const BitShiftLeft>(std::move(left), std::move(right), dtype, span);
    case ObjectKind::BitShiftRight:
      return std::make_shared<const BitShiftRight>(std::move(left), std::move(right), dtype, span);
    default:
      INTERNAL_CHECK_SPAN(false, span) << "Unknown binary expression kind in ReconstructBinaryExpr";
  }
}

/// Reconstruct a unary expression with a new operand, preserving the concrete type.
/// All unary ops share the constructor signature (ExprPtr, DataType, Span).
ExprPtr ReconstructUnaryExpr(ObjectKind kind, ExprPtr operand, DataType dtype, const Span& span) {
  switch (kind) {
    case ObjectKind::Abs:
      return std::make_shared<const Abs>(std::move(operand), dtype, span);
    case ObjectKind::Neg:
      return std::make_shared<const Neg>(std::move(operand), dtype, span);
    case ObjectKind::Not:
      return std::make_shared<const Not>(std::move(operand), dtype, span);
    case ObjectKind::BitNot:
      return std::make_shared<const BitNot>(std::move(operand), dtype, span);
    case ObjectKind::Cast:
      return std::make_shared<const Cast>(std::move(operand), dtype, span);
    default:
      INTERNAL_CHECK_SPAN(false, span) << "Unknown unary expression kind in ReconstructUnaryExpr";
  }
}

}  // namespace

// Top-level entry points
ProgramPtr IRMutator::VisitProgram(const ProgramPtr& program) {
  std::map<GlobalVarPtr, FunctionPtr, GlobalVarPtrLess> new_functions;
  bool changed = false;
  for (auto& [gv, func] : program->functions_) {
    auto new_func = VisitFunction(func);
    new_functions.emplace(gv, new_func);
    if (new_func.get() != func.get()) {
      changed = true;
    }
  }
  if (!changed) {
    return program;
  }
  return std::make_shared<const Program>(std::move(new_functions), program->name_, program->span_);
}

FunctionPtr IRMutator::VisitFunction(const FunctionPtr& func) {
  auto new_body = VisitStmt(func->body_);
  if (new_body.get() == func->body_.get()) {
    return func;
  }
  return std::make_shared<const Function>(func->name_, func->params_, func->param_directions_,
                                          func->return_types_, std::move(new_body), func->span_,
                                          func->func_type_, func->level_, func->role_, func->attrs_);
}

ExprPtr IRMutator::VisitExpr(const ExprPtr& expr) { return ExprFunctor<ExprPtr>::VisitExpr(expr); }

StmtPtr IRMutator::VisitStmt(const StmtPtr& stmt) { return StmtFunctor<StmtPtr>::VisitStmt(stmt); }

ExprPtr IRMutator::VisitExpr_(const VarPtr& op) {
  auto it = var_remap_.find(op.get());
  if (it != var_remap_.end()) {
    return it->second;
  }
  return op;
}

ExprPtr IRMutator::VisitExpr_(const IterArgPtr& op) {
  auto it = var_remap_.find(op.get());
  if (it != var_remap_.end()) {
    return it->second;
  }
  INTERNAL_CHECK_SPAN(op->initValue_, op->span_) << "IterArg has null initValue";
  auto new_init_value = ExprFunctor<ExprPtr>::VisitExpr(op->initValue_);
  INTERNAL_CHECK_SPAN(new_init_value, op->span_) << "IterArg initValue mutated to null";
  if (new_init_value.get() != op->initValue_.get()) {
    return std::make_shared<const IterArg>(op->name_hint_, op->GetType(), std::move(new_init_value),
                                           op->span_);
  }
  return op;
}

ExprPtr IRMutator::VisitExpr_(const MemRefPtr& op) { return op; }

ExprPtr IRMutator::VisitExpr_(const ConstIntPtr& op) { return op; }

ExprPtr IRMutator::VisitExpr_(const ConstFloatPtr& op) { return op; }

ExprPtr IRMutator::VisitExpr_(const ConstBoolPtr& op) { return op; }

ExprPtr IRMutator::VisitExpr_(const CallPtr& op) {
  std::vector<ExprPtr> new_args;
  bool changed = false;
  new_args.reserve(op->args_.size());

  for (size_t i = 0; i < op->args_.size(); ++i) {
    INTERNAL_CHECK_SPAN(op->args_[i], op->span_) << "Call has null argument at index " << i;
    auto new_arg = ExprFunctor<ExprPtr>::VisitExpr(op->args_[i]);
    INTERNAL_CHECK_SPAN(new_arg, op->span_) << "Call argument at index " << i << " mutated to null";
    new_args.push_back(new_arg);
    if (new_arg.get() != op->args_[i].get()) {
      changed = true;
    }
  }

  if (changed) {
    return std::make_shared<const Call>(op->op_, std::move(new_args), op->kwargs_, op->GetType(), op->span_);
  }
  return op;
}

ExprPtr IRMutator::VisitExpr_(const MakeTuplePtr& op) {
  std::vector<ExprPtr> new_elements;
  new_elements.reserve(op->elements_.size());
  bool changed = false;

  for (const auto& elem : op->elements_) {
    INTERNAL_CHECK_SPAN(elem, op->span_) << "MakeTuple has null element";
    auto new_elem = ExprFunctor<ExprPtr>::VisitExpr(elem);
    INTERNAL_CHECK_SPAN(new_elem, op->span_) << "MakeTuple element mutated to null";
    new_elements.push_back(new_elem);
    if (new_elem.get() != elem.get()) {
      changed = true;
    }
  }

  if (changed) {
    return std::make_shared<const MakeTuple>(std::move(new_elements), op->span_);
  }
  return op;
}

ExprPtr IRMutator::VisitExpr_(const TupleGetItemExprPtr& op) {
  INTERNAL_CHECK_SPAN(op->tuple_, op->span_) << "TupleGetItemExpr has null tuple";
  auto new_tuple = ExprFunctor<ExprPtr>::VisitExpr(op->tuple_);
  INTERNAL_CHECK_SPAN(new_tuple, op->span_) << "TupleGetItemExpr tuple mutated to null";

  if (new_tuple.get() != op->tuple_.get()) {
    return std::make_shared<const TupleGetItemExpr>(new_tuple, op->index_, op->span_);
  }
  return op;
}

ExprPtr IRMutator::VisitBinaryExpr_(const BinaryExprPtr& op) {
  INTERNAL_CHECK_SPAN(op->left_, op->span_) << "BinaryExpr has null left operand";
  INTERNAL_CHECK_SPAN(op->right_, op->span_) << "BinaryExpr has null right operand";
  auto new_left = ExprFunctor<ExprPtr>::VisitExpr(op->left_);
  auto new_right = ExprFunctor<ExprPtr>::VisitExpr(op->right_);
  INTERNAL_CHECK_SPAN(new_left, op->span_) << "BinaryExpr left operand mutated to null";
  INTERNAL_CHECK_SPAN(new_right, op->span_) << "BinaryExpr right operand mutated to null";
  if (new_left.get() != op->left_.get() || new_right.get() != op->right_.get()) {
    auto scalar_type = As<ScalarType>(op->GetType());
    INTERNAL_CHECK_SPAN(scalar_type, op->span_) << "BinaryExpr has null type";
    return ReconstructBinaryExpr(op->GetKind(), std::move(new_left), std::move(new_right),
                                 scalar_type->dtype_, op->span_);
  }
  return op;
}

ExprPtr IRMutator::VisitUnaryExpr_(const UnaryExprPtr& op) {
  INTERNAL_CHECK_SPAN(op->operand_, op->span_) << "UnaryExpr has null operand";
  auto new_operand = ExprFunctor<ExprPtr>::VisitExpr(op->operand_);
  INTERNAL_CHECK_SPAN(new_operand, op->span_) << "UnaryExpr operand mutated to null";
  if (new_operand.get() != op->operand_.get()) {
    auto scalar_type = As<ScalarType>(op->GetType());
    INTERNAL_CHECK_SPAN(scalar_type, op->span_) << "UnaryExpr has null type";
    return ReconstructUnaryExpr(op->GetKind(), std::move(new_operand), scalar_type->dtype_, op->span_);
  }
  return op;
}

#define DEFINE_BINARY_MUTATOR(OpType) \
  ExprPtr IRMutator::VisitExpr_(const OpType##Ptr& op) { return VisitBinaryExpr_(op); }

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

#define DEFINE_UNARY_MUTATOR(OpType) \
  ExprPtr IRMutator::VisitExpr_(const OpType##Ptr& op) { return VisitUnaryExpr_(op); }

DEFINE_UNARY_MUTATOR(Abs)
DEFINE_UNARY_MUTATOR(Neg)
DEFINE_UNARY_MUTATOR(Not)
DEFINE_UNARY_MUTATOR(BitNot)
DEFINE_UNARY_MUTATOR(Cast)

#undef DEFINE_UNARY_MUTATOR

StmtPtr IRMutator::VisitStmt_(const AssignStmtPtr& op) {
  INTERNAL_CHECK_SPAN(op->var_, op->span_) << "AssignStmt has null var";
  INTERNAL_CHECK_SPAN(op->value_, op->span_) << "AssignStmt has null value";
  auto new_var_expr = ExprFunctor<ExprPtr>::VisitExpr(op->var_);
  auto new_value = ExprFunctor<ExprPtr>::VisitExpr(op->value_);
  INTERNAL_CHECK_SPAN(new_var_expr, op->span_) << "AssignStmt var mutated to null";
  INTERNAL_CHECK_SPAN(new_value, op->span_) << "AssignStmt value mutated to null";
  // As<Var> uses exact kind match, so also try As<MemRef> (MemRef inherits from Var)
  auto new_var = As<Var>(new_var_expr);
  if (!new_var) {
    auto memref = As<MemRef>(new_var_expr);
    if (memref) {
      new_var = std::static_pointer_cast<const Var>(memref);
    }
  }
  INTERNAL_CHECK_SPAN(new_var, op->span_) << "AssignStmt var is not a Var after mutation";
  if (new_var.get() != op->var_.get() || new_value.get() != op->value_.get()) {
    auto result = MutableCopy(op);
    result->var_ = std::move(new_var);
    result->value_ = std::move(new_value);
    return result;
  }
  return op;
}

StmtPtr IRMutator::VisitStmt_(const IfStmtPtr& op) {
  INTERNAL_CHECK_SPAN(op->condition_, op->span_) << "IfStmt has null condition";
  auto new_condition = ExprFunctor<ExprPtr>::VisitExpr(op->condition_);
  INTERNAL_CHECK_SPAN(new_condition, op->span_) << "IfStmt condition mutated to null";

  INTERNAL_CHECK_SPAN(op->then_body_, op->span_) << "IfStmt has null then_body";
  auto new_then_body = StmtFunctor<StmtPtr>::VisitStmt(op->then_body_);
  INTERNAL_CHECK_SPAN(new_then_body, op->span_) << "IfStmt then_body mutated to null";
  bool then_changed = (new_then_body.get() != op->then_body_.get());

  std::optional<StmtPtr> new_else_body;
  bool else_changed = false;
  if (op->else_body_.has_value()) {
    INTERNAL_CHECK_SPAN(*op->else_body_, op->span_) << "IfStmt has null else_body";
    auto new_stmt = StmtFunctor<StmtPtr>::VisitStmt(*op->else_body_);
    INTERNAL_CHECK_SPAN(new_stmt, op->span_) << "IfStmt else_body mutated to null";
    new_else_body = new_stmt;
    if (new_stmt.get() != op->else_body_->get()) {
      else_changed = true;
    }
  }

  std::vector<VarPtr> new_return_vars;
  bool return_vars_changed = false;
  new_return_vars.reserve(op->return_vars_.size());
  for (size_t i = 0; i < op->return_vars_.size(); ++i) {
    INTERNAL_CHECK_SPAN(op->return_vars_[i], op->span_) << "IfStmt has null return_vars at index " << i;
    auto new_var_expr = ExprFunctor<ExprPtr>::VisitExpr(op->return_vars_[i]);
    INTERNAL_CHECK_SPAN(new_var_expr, op->span_) << "IfStmt return_vars at index " << i << " mutated to null";
    auto new_var = As<Var>(new_var_expr);
    INTERNAL_CHECK_SPAN(new_var, op->span_)
        << "IfStmt return_vars at index " << i << " is not a Var after mutation";
    new_return_vars.push_back(new_var);
    if (new_var.get() != op->return_vars_[i].get()) {
      return_vars_changed = true;
    }
  }

  if (new_condition.get() != op->condition_.get() || then_changed || else_changed || return_vars_changed) {
    auto result = MutableCopy(op);
    result->condition_ = std::move(new_condition);
    result->then_body_ = std::move(new_then_body);
    result->else_body_ = new_else_body;
    result->return_vars_ = std::move(new_return_vars);
    return result;
  }
  return op;
}

StmtPtr IRMutator::VisitStmt_(const YieldStmtPtr& op) {
  std::vector<ExprPtr> new_value;
  bool changed = false;
  new_value.reserve(op->value_.size());

  for (size_t i = 0; i < op->value_.size(); ++i) {
    INTERNAL_CHECK_SPAN(op->value_[i], op->span_) << "YieldStmt has null value at index " << i;
    auto new_expr = ExprFunctor<ExprPtr>::VisitExpr(op->value_[i]);
    INTERNAL_CHECK_SPAN(new_expr, op->span_) << "YieldStmt value at index " << i << " mutated to null";
    new_value.push_back(new_expr);
    if (new_expr.get() != op->value_[i].get()) {
      changed = true;
    }
  }

  if (changed) {
    auto result = MutableCopy(op);
    result->value_ = std::move(new_value);
    return result;
  }
  return op;
}

StmtPtr IRMutator::VisitStmt_(const ReturnStmtPtr& op) {
  std::vector<ExprPtr> new_value;
  bool changed = false;
  new_value.reserve(op->value_.size());

  for (size_t i = 0; i < op->value_.size(); ++i) {
    INTERNAL_CHECK_SPAN(op->value_[i], op->span_) << "ReturnStmt has null value at index " << i;
    auto new_expr = ExprFunctor<ExprPtr>::VisitExpr(op->value_[i]);
    INTERNAL_CHECK_SPAN(new_expr, op->span_) << "ReturnStmt value at index " << i << " mutated to null";
    new_value.push_back(new_expr);
    if (new_expr.get() != op->value_[i].get()) {
      changed = true;
    }
  }

  if (changed) {
    auto result = MutableCopy(op);
    result->value_ = std::move(new_value);
    return result;
  }
  return op;
}

StmtPtr IRMutator::VisitStmt_(const ForStmtPtr& op) {
  INTERNAL_CHECK_SPAN(op->loop_var_, op->span_) << "ForStmt has null loop_var";
  INTERNAL_CHECK_SPAN(op->start_, op->span_) << "ForStmt has null start";
  INTERNAL_CHECK_SPAN(op->stop_, op->span_) << "ForStmt has null stop";
  INTERNAL_CHECK_SPAN(op->step_, op->span_) << "ForStmt has null step";
  auto new_loop_var_expr = ExprFunctor<ExprPtr>::VisitExpr(op->loop_var_);
  INTERNAL_CHECK_SPAN(new_loop_var_expr, op->span_) << "ForStmt loop_var mutated to null";
  auto new_loop_var = As<Var>(new_loop_var_expr);
  INTERNAL_CHECK_SPAN(new_loop_var, op->span_) << "ForStmt loop_var is not a Var after mutation";

  auto new_start = ExprFunctor<ExprPtr>::VisitExpr(op->start_);
  INTERNAL_CHECK_SPAN(new_start, op->span_) << "ForStmt start mutated to null";

  auto new_stop = ExprFunctor<ExprPtr>::VisitExpr(op->stop_);
  INTERNAL_CHECK_SPAN(new_stop, op->span_) << "ForStmt stop mutated to null";

  auto new_step = ExprFunctor<ExprPtr>::VisitExpr(op->step_);
  INTERNAL_CHECK_SPAN(new_step, op->span_) << "ForStmt step mutated to null";

  std::vector<IterArgPtr> new_iter_args;
  bool iter_args_changed = false;
  new_iter_args.reserve(op->iter_args_.size());
  for (size_t i = 0; i < op->iter_args_.size(); ++i) {
    INTERNAL_CHECK_SPAN(op->iter_args_[i], op->span_) << "ForStmt has null iter_args at index " << i;
    auto new_iter_arg_expr = ExprFunctor<ExprPtr>::VisitExpr(op->iter_args_[i]);
    INTERNAL_CHECK_SPAN(new_iter_arg_expr, op->span_)
        << "ForStmt iter_args at index " << i << " mutated to null";
    auto new_iter_arg = As<IterArg>(std::static_pointer_cast<const IRNode>(new_iter_arg_expr));
    INTERNAL_CHECK_SPAN(new_iter_arg, op->span_)
        << "ForStmt iter_args at index " << i << " is not an IterArg after mutation";
    new_iter_args.push_back(new_iter_arg);
    if (new_iter_arg.get() != op->iter_args_[i].get()) {
      iter_args_changed = true;
    }
  }

  // Register old→new IterArg mappings so body references are substituted
  for (size_t i = 0; i < op->iter_args_.size(); ++i) {
    if (new_iter_args[i].get() != op->iter_args_[i].get()) {
      var_remap_[op->iter_args_[i].get()] = new_iter_args[i];
    }
  }

  INTERNAL_CHECK_SPAN(op->body_, op->span_) << "ForStmt has null body";
  auto new_body = StmtFunctor<StmtPtr>::VisitStmt(op->body_);
  INTERNAL_CHECK_SPAN(new_body, op->span_) << "ForStmt body mutated to null";
  bool body_changed = (new_body.get() != op->body_.get());

  // Clean up IterArg remappings.
  // Safe to clean before visiting return_vars: return_vars are separate Var objects,
  // not references to IterArgs, so they don't need the remapping.
  for (const auto& old_iter_arg : op->iter_args_) {
    var_remap_.erase(old_iter_arg.get());
  }

  std::vector<VarPtr> new_return_vars;
  bool return_vars_changed = false;
  new_return_vars.reserve(op->return_vars_.size());
  for (size_t i = 0; i < op->return_vars_.size(); ++i) {
    INTERNAL_CHECK_SPAN(op->return_vars_[i], op->span_) << "ForStmt has null return_vars at index " << i;
    auto new_var_expr = ExprFunctor<ExprPtr>::VisitExpr(op->return_vars_[i]);
    INTERNAL_CHECK_SPAN(new_var_expr, op->span_)
        << "ForStmt return_vars at index " << i << " mutated to null";
    auto new_var = As<Var>(new_var_expr);
    INTERNAL_CHECK_SPAN(new_var, op->span_)
        << "ForStmt return_vars at index " << i << " is not a Var after mutation";
    new_return_vars.push_back(new_var);
    if (new_var.get() != op->return_vars_[i].get()) {
      return_vars_changed = true;
    }
  }

  std::optional<ChunkConfig> new_chunk_config = op->chunk_config_;
  bool chunk_config_changed = false;
  if (op->chunk_config_.has_value()) {
    auto new_cs = ExprFunctor<ExprPtr>::VisitExpr(op->chunk_config_->size);
    INTERNAL_CHECK_SPAN(new_cs, op->span_) << "ForStmt chunk_size mutated to null";
    if (new_cs.get() != op->chunk_config_->size.get()) {
      new_chunk_config = ChunkConfig{new_cs, op->chunk_config_->policy};
      chunk_config_changed = true;
    }
  }

  if (new_loop_var.get() != op->loop_var_.get() || new_start.get() != op->start_.get() ||
      new_stop.get() != op->stop_.get() || new_step.get() != op->step_.get() || iter_args_changed ||
      body_changed || return_vars_changed || chunk_config_changed) {
    auto result = MutableCopy(op);
    result->loop_var_ = std::move(new_loop_var);
    result->start_ = std::move(new_start);
    result->stop_ = std::move(new_stop);
    result->step_ = std::move(new_step);
    result->iter_args_ = std::move(new_iter_args);
    result->body_ = std::move(new_body);
    result->return_vars_ = std::move(new_return_vars);
    result->chunk_config_ = std::move(new_chunk_config);
    return result;
  }
  return op;
}

StmtPtr IRMutator::VisitStmt_(const WhileStmtPtr& op) {
  // Visit iter_args first (definitions), before condition and body (uses).
  std::vector<IterArgPtr> new_iter_args;
  bool iter_args_changed = false;
  new_iter_args.reserve(op->iter_args_.size());
  for (size_t i = 0; i < op->iter_args_.size(); ++i) {
    INTERNAL_CHECK_SPAN(op->iter_args_[i], op->span_) << "WhileStmt has null iter_args at index " << i;
    auto new_iter_arg_expr = ExprFunctor<ExprPtr>::VisitExpr(op->iter_args_[i]);
    INTERNAL_CHECK_SPAN(new_iter_arg_expr, op->span_)
        << "WhileStmt iter_args at index " << i << " mutated to null";
    auto new_iter_arg = As<IterArg>(std::static_pointer_cast<const IRNode>(new_iter_arg_expr));
    INTERNAL_CHECK_SPAN(new_iter_arg, op->span_)
        << "WhileStmt iter_args at index " << i << " is not an IterArg after mutation";
    new_iter_args.push_back(new_iter_arg);
    if (new_iter_arg.get() != op->iter_args_[i].get()) {
      iter_args_changed = true;
    }
  }

  // Register old→new IterArg mappings so condition and body references are substituted
  for (size_t i = 0; i < op->iter_args_.size(); ++i) {
    if (new_iter_args[i].get() != op->iter_args_[i].get()) {
      var_remap_[op->iter_args_[i].get()] = new_iter_args[i];
    }
  }

  INTERNAL_CHECK_SPAN(op->condition_, op->span_) << "WhileStmt has null condition";
  auto new_condition = ExprFunctor<ExprPtr>::VisitExpr(op->condition_);
  INTERNAL_CHECK_SPAN(new_condition, op->span_) << "WhileStmt condition mutated to null";
  bool condition_changed = (new_condition.get() != op->condition_.get());

  INTERNAL_CHECK_SPAN(op->body_, op->span_) << "WhileStmt has null body";
  auto new_body = StmtFunctor<StmtPtr>::VisitStmt(op->body_);
  INTERNAL_CHECK_SPAN(new_body, op->span_) << "WhileStmt body mutated to null";
  bool body_changed = (new_body.get() != op->body_.get());

  // Clean up IterArg remappings.
  // Safe to clean before visiting return_vars: return_vars are separate Var objects,
  // not references to IterArgs, so they don't need the remapping.
  for (const auto& old_iter_arg : op->iter_args_) {
    var_remap_.erase(old_iter_arg.get());
  }

  std::vector<VarPtr> new_return_vars;
  bool return_vars_changed = false;
  new_return_vars.reserve(op->return_vars_.size());
  for (size_t i = 0; i < op->return_vars_.size(); ++i) {
    INTERNAL_CHECK_SPAN(op->return_vars_[i], op->span_) << "WhileStmt has null return_vars at index " << i;
    auto new_var_expr = ExprFunctor<ExprPtr>::VisitExpr(op->return_vars_[i]);
    INTERNAL_CHECK_SPAN(new_var_expr, op->span_)
        << "WhileStmt return_vars at index " << i << " mutated to null";
    auto new_var = As<Var>(new_var_expr);
    INTERNAL_CHECK_SPAN(new_var, op->span_)
        << "WhileStmt return_vars at index " << i << " is not a Var after mutation";
    new_return_vars.push_back(new_var);
    if (new_var.get() != op->return_vars_[i].get()) {
      return_vars_changed = true;
    }
  }

  if (condition_changed || iter_args_changed || body_changed || return_vars_changed) {
    auto result = MutableCopy(op);
    result->condition_ = std::move(new_condition);
    result->iter_args_ = std::move(new_iter_args);
    result->body_ = std::move(new_body);
    result->return_vars_ = std::move(new_return_vars);
    return result;
  }
  return op;
}

StmtPtr IRMutator::VisitStmt_(const ScopeStmtPtr& op) {
  INTERNAL_CHECK_SPAN(op->body_, op->span_) << "ScopeStmt has null body";
  auto new_body = StmtFunctor<StmtPtr>::VisitStmt(op->body_);
  INTERNAL_CHECK_SPAN(new_body, op->span_) << "ScopeStmt body mutated to null";
  if (new_body.get() != op->body_.get()) {
    auto result = MutableCopy(op);
    result->body_ = std::move(new_body);
    return result;
  }
  return op;
}

StmtPtr IRMutator::VisitStmt_(const SeqStmtsPtr& op) {
  std::vector<StmtPtr> new_stmts;
  bool changed = false;
  new_stmts.reserve(op->stmts_.size());
  for (size_t i = 0; i < op->stmts_.size(); ++i) {
    INTERNAL_CHECK_SPAN(op->stmts_[i], op->span_) << "SeqStmts has null statement at index " << i;
    auto new_stmt = StmtFunctor<StmtPtr>::VisitStmt(op->stmts_[i]);
    INTERNAL_CHECK_SPAN(new_stmt, op->span_) << "SeqStmts statement at index " << i << " mutated to null";
    new_stmts.push_back(new_stmt);
    if (new_stmt.get() != op->stmts_[i].get()) {
      changed = true;
    }
  }

  if (changed) {
    return SeqStmts::Flatten(std::move(new_stmts), op->span_);
  }
  return op;
}

StmtPtr IRMutator::VisitStmt_(const EvalStmtPtr& op) {
  INTERNAL_CHECK_SPAN(op->expr_, op->span_) << "EvalStmt has null expr";
  auto new_expr = ExprFunctor<ExprPtr>::VisitExpr(op->expr_);
  INTERNAL_CHECK_SPAN(new_expr, op->span_) << "EvalStmt expr mutated to null";

  if (new_expr.get() != op->expr_.get()) {
    auto result = MutableCopy(op);
    result->expr_ = std::move(new_expr);
    return result;
  }
  return op;
}

StmtPtr IRMutator::VisitStmt_(const BreakStmtPtr& op) { return op; }

StmtPtr IRMutator::VisitStmt_(const ContinueStmtPtr& op) { return op; }

StmtPtr IRMutator::VisitStmt_(const StmtPtr& op) { return op; }

}  // namespace ir
}  // namespace pypto
