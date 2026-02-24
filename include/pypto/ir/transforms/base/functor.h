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

#ifndef PYPTO_IR_TRANSFORMS_BASE_FUNCTOR_H_
#define PYPTO_IR_TRANSFORMS_BASE_FUNCTOR_H_

#include <utility>

#include "pypto/core/error.h"
#include "pypto/ir/core.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memref.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"

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
  virtual R VisitExpr_(const IterArgPtr& op, Args... args) = 0;
  virtual R VisitExpr_(const MemRefPtr& op, Args... args) = 0;
  virtual R VisitExpr_(const ConstIntPtr& op, Args... args) = 0;
  virtual R VisitExpr_(const ConstFloatPtr& op, Args... args) = 0;
  virtual R VisitExpr_(const ConstBoolPtr& op, Args... args) = 0;
  virtual R VisitExpr_(const CallPtr& op, Args... args) = 0;
  virtual R VisitExpr_(const MakeTuplePtr& op, Args... args) = 0;
  virtual R VisitExpr_(const TupleGetItemExprPtr& op, Args... args) = 0;

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

  // Unary operations (5 types)
  virtual R VisitExpr_(const AbsPtr& op, Args... args) = 0;
  virtual R VisitExpr_(const NegPtr& op, Args... args) = 0;
  virtual R VisitExpr_(const NotPtr& op, Args... args) = 0;
  virtual R VisitExpr_(const BitNotPtr& op, Args... args) = 0;
  virtual R VisitExpr_(const CastPtr& op, Args... args) = 0;
};

// Macro to dispatch based on expression type
#define EXPR_FUNCTOR_DISPATCH(OpType)                   \
  if (auto op = As<OpType>(expr)) {                     \
    return VisitExpr_(op, std::forward<Args>(args)...); \
  }

template <typename R, typename... Args>
R ExprFunctor<R, Args...>::VisitExpr(const ExprPtr& expr, Args... args) {
  // Leaf nodes
  // Note: IterArg and MemRef must be checked before Var since they inherit from Var
  EXPR_FUNCTOR_DISPATCH(IterArg);
  EXPR_FUNCTOR_DISPATCH(MemRef);
  EXPR_FUNCTOR_DISPATCH(Var);
  EXPR_FUNCTOR_DISPATCH(ConstInt);
  EXPR_FUNCTOR_DISPATCH(ConstFloat);
  EXPR_FUNCTOR_DISPATCH(ConstBool);
  EXPR_FUNCTOR_DISPATCH(Call);
  EXPR_FUNCTOR_DISPATCH(MakeTuple);
  EXPR_FUNCTOR_DISPATCH(TupleGetItemExpr);

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
  EXPR_FUNCTOR_DISPATCH(Cast);

  // Should never reach here if all types are handled
  throw pypto::TypeError("Unknown expression type in ExprFunctor::VisitExpr");
}

#undef EXPR_FUNCTOR_DISPATCH

/**
 * @brief Base template for statement functors
 *
 * Provides a visitor-like interface for operating on IR statements.
 * Subclasses implement specific operations by overriding VisitStmt_ methods.
 *
 * @tparam R Return type of the visit operations
 * @tparam Args Additional arguments passed to visit methods
 */
template <typename R, typename... Args>
class StmtFunctor {
 public:
  virtual ~StmtFunctor() = default;

  /**
   * @brief Dispatcher for statement types
   *
   * Uses dynamic_cast to determine concrete type and dispatch to appropriate handler.
   *
   * @param stmt Statement pointer (non-null)
   * @param args Additional arguments
   * @return Result of visiting the statement
   */
  virtual R VisitStmt(const StmtPtr& stmt, Args... args);

 protected:
  // Statement types
  virtual R VisitStmt_(const AssignStmtPtr& op, Args... args) = 0;
  virtual R VisitStmt_(const IfStmtPtr& op, Args... args) = 0;
  virtual R VisitStmt_(const YieldStmtPtr& op, Args... args) = 0;
  virtual R VisitStmt_(const ReturnStmtPtr& op, Args... args) = 0;
  virtual R VisitStmt_(const ForStmtPtr& op, Args... args) = 0;
  virtual R VisitStmt_(const WhileStmtPtr& op, Args... args) = 0;
  virtual R VisitStmt_(const ScopeStmtPtr& op, Args... args) = 0;
  virtual R VisitStmt_(const SeqStmtsPtr& op, Args... args) = 0;
  virtual R VisitStmt_(const OpStmtsPtr& op, Args... args) = 0;
  virtual R VisitStmt_(const EvalStmtPtr& op, Args... args) = 0;
  virtual R VisitStmt_(const BreakStmtPtr& op, Args... args) = 0;
  virtual R VisitStmt_(const ContinueStmtPtr& op, Args... args) = 0;
  virtual R VisitStmt_(const StmtPtr& op, Args... args) = 0;
};

// Macro to dispatch based on statement type
#define STMT_FUNCTOR_DISPATCH(OpType)                   \
  if (auto op = As<OpType>(stmt)) {                     \
    return VisitStmt_(op, std::forward<Args>(args)...); \
  }

template <typename R, typename... Args>
R StmtFunctor<R, Args...>::VisitStmt(const StmtPtr& stmt, Args... args) {
  // Dispatch to concrete statement types
  STMT_FUNCTOR_DISPATCH(AssignStmt);
  STMT_FUNCTOR_DISPATCH(IfStmt);
  STMT_FUNCTOR_DISPATCH(YieldStmt);
  STMT_FUNCTOR_DISPATCH(ReturnStmt);
  STMT_FUNCTOR_DISPATCH(ForStmt);
  STMT_FUNCTOR_DISPATCH(WhileStmt);
  STMT_FUNCTOR_DISPATCH(ScopeStmt);
  STMT_FUNCTOR_DISPATCH(SeqStmts);
  STMT_FUNCTOR_DISPATCH(OpStmts);
  STMT_FUNCTOR_DISPATCH(EvalStmt);
  STMT_FUNCTOR_DISPATCH(BreakStmt);
  STMT_FUNCTOR_DISPATCH(ContinueStmt);

  // Should never reach here if all types are handled
  throw pypto::TypeError("Unknown statement type in StmtFunctor::VisitStmt");
}

#undef STMT_FUNCTOR_DISPATCH

/**
 * @brief Unified functor for both expressions and statements
 *
 * Combines ExprFunctor and StmtFunctor to provide a unified interface
 * for visiting both expression and statement IR nodes.
 *
 * @tparam R Return type of the visit operations
 * @tparam Args Additional arguments passed to visit methods
 */
template <typename R, typename... Args>
class IRFunctor : public ExprFunctor<R, Args...>, public StmtFunctor<R, Args...> {
 public:
  virtual ~IRFunctor() = default;

  /**
   * @brief Dispatcher for IR node types (Expr or Stmt)
   *
   * Determines whether the node is an Expr or Stmt and dispatches accordingly.
   *
   * @param node IR node pointer (non-null)
   * @param args Additional arguments
   * @return Result of visiting the IR node
   */
  R VisitIRNode(const IRNodePtr& node, Args... args) {
    if (auto expr = As<Expr>(node)) {
      return ExprFunctor<R, Args...>::VisitExpr(expr, std::forward<Args>(args)...);
    } else if (auto stmt = As<Stmt>(node)) {
      return StmtFunctor<R, Args...>::VisitStmt(stmt, std::forward<Args>(args)...);
    }
    throw pypto::TypeError("Unknown IR node type in IRFunctor::VisitIRNode");
  }
};

}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORMS_BASE_FUNCTOR_H_
