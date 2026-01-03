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

#ifndef PYPTO_IR_TRANSFORM_FUNCTOR_H_
#define PYPTO_IR_TRANSFORM_FUNCTOR_H_

#include <utility>

#include "pypto/core/error.h"
#include "pypto/ir/scalar_expr.h"

namespace pypto {
namespace ir {

/**
 * @brief Base template for expression functors
 *
 * Provides a visitor-like interface for operating on IR expressions.
 * Subclasses implement specific operations by overriding VisitExpr_ methods.
 *
 * @tparam R Return type of the visit operations
 * @tparam Args Additional arguments passed to visit methods
 */
template <typename R, typename... Args>
class ExprFunctor {
 public:
  virtual ~ExprFunctor() = default;

  /**
   * @brief Dispatcher for expression types
   *
   * Uses dynamic_cast to determine concrete type and dispatch to appropriate handler.
   *
   * @param expr Expression pointer (non-null)
   * @param args Additional arguments
   * @return Result of visiting the expression
   */
  virtual R VisitExpr(const ExprPtr& expr, Args... args);

 protected:
  // Leaf nodes
  virtual R VisitExpr_(const VarPtr& op, Args... args) = 0;
  virtual R VisitExpr_(const ConstIntPtr& op, Args... args) = 0;
  virtual R VisitExpr_(const CallPtr& op, Args... args) = 0;

  // Binary operations (22 types)
  virtual R VisitExpr_(const AddPtr& op, Args... args) = 0;
  virtual R VisitExpr_(const SubPtr& op, Args... args) = 0;
  virtual R VisitExpr_(const MulPtr& op, Args... args) = 0;
  virtual R VisitExpr_(const FloorDivPtr& op, Args... args) = 0;
  virtual R VisitExpr_(const FloorModPtr& op, Args... args) = 0;
  virtual R VisitExpr_(const FloatDivPtr& op, Args... args) = 0;
  virtual R VisitExpr_(const MinPtr& op, Args... args) = 0;
  virtual R VisitExpr_(const MaxPtr& op, Args... args) = 0;
  virtual R VisitExpr_(const PowPtr& op, Args... args) = 0;
  virtual R VisitExpr_(const EqPtr& op, Args... args) = 0;
  virtual R VisitExpr_(const NePtr& op, Args... args) = 0;
  virtual R VisitExpr_(const LtPtr& op, Args... args) = 0;
  virtual R VisitExpr_(const LePtr& op, Args... args) = 0;
  virtual R VisitExpr_(const GtPtr& op, Args... args) = 0;
  virtual R VisitExpr_(const GePtr& op, Args... args) = 0;
  virtual R VisitExpr_(const AndPtr& op, Args... args) = 0;
  virtual R VisitExpr_(const OrPtr& op, Args... args) = 0;
  virtual R VisitExpr_(const XorPtr& op, Args... args) = 0;
  virtual R VisitExpr_(const BitAndPtr& op, Args... args) = 0;
  virtual R VisitExpr_(const BitOrPtr& op, Args... args) = 0;
  virtual R VisitExpr_(const BitXorPtr& op, Args... args) = 0;
  virtual R VisitExpr_(const BitShiftLeftPtr& op, Args... args) = 0;
  virtual R VisitExpr_(const BitShiftRightPtr& op, Args... args) = 0;

  // Unary operations (4 types)
  virtual R VisitExpr_(const AbsPtr& op, Args... args) = 0;
  virtual R VisitExpr_(const NegPtr& op, Args... args) = 0;
  virtual R VisitExpr_(const NotPtr& op, Args... args) = 0;
  virtual R VisitExpr_(const BitNotPtr& op, Args... args) = 0;
};

// Macro to dispatch based on expression type
#define EXPR_FUNCTOR_DISPATCH(OpType)                            \
  if (auto op = std::dynamic_pointer_cast<const OpType>(expr)) { \
    return VisitExpr_(op, std::forward<Args>(args)...);          \
  }

template <typename R, typename... Args>
R ExprFunctor<R, Args...>::VisitExpr(const ExprPtr& expr, Args... args) {
  // Leaf nodes
  EXPR_FUNCTOR_DISPATCH(Var);
  EXPR_FUNCTOR_DISPATCH(ConstInt);
  EXPR_FUNCTOR_DISPATCH(Call);

  // Binary operations
  EXPR_FUNCTOR_DISPATCH(Add);
  EXPR_FUNCTOR_DISPATCH(Sub);
  EXPR_FUNCTOR_DISPATCH(Mul);
  EXPR_FUNCTOR_DISPATCH(FloorDiv);
  EXPR_FUNCTOR_DISPATCH(FloorMod);
  EXPR_FUNCTOR_DISPATCH(FloatDiv);
  EXPR_FUNCTOR_DISPATCH(Min);
  EXPR_FUNCTOR_DISPATCH(Max);
  EXPR_FUNCTOR_DISPATCH(Pow);
  EXPR_FUNCTOR_DISPATCH(Eq);
  EXPR_FUNCTOR_DISPATCH(Ne);
  EXPR_FUNCTOR_DISPATCH(Lt);
  EXPR_FUNCTOR_DISPATCH(Le);
  EXPR_FUNCTOR_DISPATCH(Gt);
  EXPR_FUNCTOR_DISPATCH(Ge);
  EXPR_FUNCTOR_DISPATCH(And);
  EXPR_FUNCTOR_DISPATCH(Or);
  EXPR_FUNCTOR_DISPATCH(Xor);
  EXPR_FUNCTOR_DISPATCH(BitAnd);
  EXPR_FUNCTOR_DISPATCH(BitOr);
  EXPR_FUNCTOR_DISPATCH(BitXor);
  EXPR_FUNCTOR_DISPATCH(BitShiftLeft);
  EXPR_FUNCTOR_DISPATCH(BitShiftRight);

  // Unary operations
  EXPR_FUNCTOR_DISPATCH(Abs);
  EXPR_FUNCTOR_DISPATCH(Neg);
  EXPR_FUNCTOR_DISPATCH(Not);
  EXPR_FUNCTOR_DISPATCH(BitNot);

  // Should never reach here if all types are handled
  throw pypto::TypeError("Unknown expression type in ExprFunctor::VisitExpr");
}

#undef EXPR_FUNCTOR_DISPATCH

}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORM_FUNCTOR_H_
