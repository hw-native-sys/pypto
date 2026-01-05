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

namespace pypto {
namespace ir {

ExprPtr ExprMutator::VisitExpr(const ExprPtr& expr) {
  // Store the original shared_ptr for potential reuse
  return ExprFunctor::VisitExpr(expr);
}

// Leaf nodes - return original shared_ptr (immutable)
ExprPtr ExprMutator::VisitExpr_(const VarPtr& op) {
  // Var is immutable, return original
  return op;
}

ExprPtr ExprMutator::VisitExpr_(const ConstIntPtr& op) {
  // ConstInt is immutable, return original
  return op;
}

ExprPtr ExprMutator::VisitExpr_(const CallPtr& op) {
  // Visit all arguments
  std::vector<ScalarExprPtr> new_args;
  bool changed = false;
  new_args.reserve(op->args_.size());

  for (size_t i = 0; i < op->args_.size(); ++i) {
    INTERNAL_CHECK(op->args_[i]) << "Call has null argument at index " << i;
    auto new_arg = std::dynamic_pointer_cast<const ScalarExpr>(VisitExpr(op->args_[i]));
    INTERNAL_CHECK(new_arg) << "Call argument at index " << i << " mutated to non-ScalarExpr or null";
    new_args.push_back(new_arg);
    if (new_arg.get() != op->args_[i].get()) {
      changed = true;
    }
  }

  // Copy-on-write: only create new node if arguments changed
  if (changed) {
    return std::make_shared<const Call>(op->op_, std::move(new_args), op->dtype_, op->span_);
  } else {
    return op;
  }
}

// Macro to generate binary operation mutators with copy-on-write
#define DEFINE_BINARY_MUTATOR(OpType)                                                              \
  ExprPtr ExprMutator::VisitExpr_(const OpType##Ptr& op) {                                         \
    INTERNAL_CHECK(op->left_) << #OpType " has null left operand";                                 \
    INTERNAL_CHECK(op->right_) << #OpType " has null right operand";                               \
    auto new_left = std::dynamic_pointer_cast<const ScalarExpr>(VisitExpr(op->left_));             \
    auto new_right = std::dynamic_pointer_cast<const ScalarExpr>(VisitExpr(op->right_));           \
    INTERNAL_CHECK(new_left) << #OpType " left operand mutated to non-ScalarExpr or null";         \
    INTERNAL_CHECK(new_right) << #OpType " right operand mutated to non-ScalarExpr or null";       \
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
#define DEFINE_UNARY_MUTATOR(OpType)                                                         \
  ExprPtr ExprMutator::VisitExpr_(const OpType##Ptr& op) {                                   \
    INTERNAL_CHECK(op->operand_) << #OpType " has null operand";                             \
    auto new_operand = std::dynamic_pointer_cast<const ScalarExpr>(VisitExpr(op->operand_)); \
    INTERNAL_CHECK(new_operand) << #OpType " operand mutated to non-ScalarExpr or null";     \
    if (new_operand.get() != op->operand_.get()) {                                           \
      return std::make_shared<const OpType>(std::move(new_operand), op->dtype_, op->span_);  \
    } else {                                                                                 \
      return op;                                                                             \
    }                                                                                        \
  }

// Unary operations
DEFINE_UNARY_MUTATOR(Abs)
DEFINE_UNARY_MUTATOR(Neg)
DEFINE_UNARY_MUTATOR(Not)
DEFINE_UNARY_MUTATOR(BitNot)

#undef DEFINE_UNARY_MUTATOR

}  // namespace ir
}  // namespace pypto
