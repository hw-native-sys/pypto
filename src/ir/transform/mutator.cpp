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

#include "pypto/ir/transform/mutator.h"

#include <memory>
#include <utility>
#include <vector>

namespace pypto {
namespace ir {

// Store original shared_ptr in thread-local storage for copy-on-write optimization
thread_local const ExprPtr* g_current_expr = nullptr;

ExprPtr ExprMutator::VisitExpr(const ExprPtr& expr) {
  // Store the original shared_ptr for potential reuse
  const ExprPtr* prev = g_current_expr;
  g_current_expr = &expr;
  ExprPtr result = ExprFunctor::VisitExpr(expr);
  g_current_expr = prev;
  return result;
}

// Leaf nodes - return original shared_ptr (immutable)
ExprPtr ExprMutator::VisitExpr_(const VarPtr& op) {
  // Var is immutable, return original
  return *g_current_expr;
}

ExprPtr ExprMutator::VisitExpr_(const ConstIntPtr& op) {
  // ConstInt is immutable, return original
  return *g_current_expr;
}

ExprPtr ExprMutator::VisitExpr_(const CallPtr& op) {
  // Visit all arguments
  std::vector<ExprPtr> new_args;
  bool changed = false;
  new_args.reserve(op->args_.size());

  for (const auto& arg : op->args_) {
    ExprPtr new_arg = VisitExpr(arg);
    new_args.push_back(new_arg);
    if (new_arg.get() != arg.get()) {
      changed = true;
    }
  }

  // Copy-on-write: only create new node if arguments changed
  if (changed) {
    return std::make_shared<const Call>(op->op_, std::move(new_args), op->span);
  } else {
    return *g_current_expr;
  }
}

// Macro to generate binary operation mutators with copy-on-write
#define DEFINE_BINARY_MUTATOR(OpType)                                                             \
  ExprPtr ExprMutator::VisitExpr_(const OpType##Ptr& op) {                                        \
    ExprPtr new_left = VisitExpr(op->left_);                                                      \
    ExprPtr new_right = VisitExpr(op->right_);                                                    \
    if (new_left.get() != op->left_.get() || new_right.get() != op->right_.get()) {               \
      return std::make_shared<const OpType>(std::move(new_left), std::move(new_right), op->span); \
    } else {                                                                                      \
      return *g_current_expr;                                                                     \
    }                                                                                             \
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
#define DEFINE_UNARY_MUTATOR(OpType)                                           \
  ExprPtr ExprMutator::VisitExpr_(const OpType##Ptr& op) {                     \
    ExprPtr new_operand = VisitExpr(op->operand_);                             \
    if (new_operand.get() != op->operand_.get()) {                             \
      return std::make_shared<const OpType>(std::move(new_operand), op->span); \
    } else {                                                                   \
      return *g_current_expr;                                                  \
    }                                                                          \
  }

// Unary operations
DEFINE_UNARY_MUTATOR(Abs)
DEFINE_UNARY_MUTATOR(Neg)
DEFINE_UNARY_MUTATOR(Not)
DEFINE_UNARY_MUTATOR(BitNot)

#undef DEFINE_UNARY_MUTATOR

}  // namespace ir
}  // namespace pypto
