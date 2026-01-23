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

#ifndef PYPTO_IR_KIND_TRAITS_H_
#define PYPTO_IR_KIND_TRAITS_H_

#include <memory>

#include "pypto/ir/core.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

// Macro to define KindTrait specialization
#define DEFINE_KIND_TRAIT(TypeName, KindValue)    \
  template <>                                     \
  struct KindTrait<TypeName> {                    \
    static constexpr IRNodeKind kind = KindValue; \
  };

// KindTrait specializations for all concrete IR node types
// These enable compile-time type-to-Kind mapping for IsA<T>() and As<T>()

// Expression types
DEFINE_KIND_TRAIT(Var, IRNodeKind::Var)
DEFINE_KIND_TRAIT(IterArg, IRNodeKind::IterArg)
DEFINE_KIND_TRAIT(Call, IRNodeKind::Call)
DEFINE_KIND_TRAIT(TupleGetItemExpr, IRNodeKind::TupleGetItemExpr)
DEFINE_KIND_TRAIT(ConstInt, IRNodeKind::ConstInt)
DEFINE_KIND_TRAIT(ConstFloat, IRNodeKind::ConstFloat)
DEFINE_KIND_TRAIT(ConstBool, IRNodeKind::ConstBool)

// Binary expression types
DEFINE_KIND_TRAIT(Add, IRNodeKind::Add)
DEFINE_KIND_TRAIT(Sub, IRNodeKind::Sub)
DEFINE_KIND_TRAIT(Mul, IRNodeKind::Mul)
DEFINE_KIND_TRAIT(FloorDiv, IRNodeKind::FloorDiv)
DEFINE_KIND_TRAIT(FloorMod, IRNodeKind::FloorMod)
DEFINE_KIND_TRAIT(FloatDiv, IRNodeKind::FloatDiv)
DEFINE_KIND_TRAIT(Min, IRNodeKind::Min)
DEFINE_KIND_TRAIT(Max, IRNodeKind::Max)
DEFINE_KIND_TRAIT(Pow, IRNodeKind::Pow)
DEFINE_KIND_TRAIT(Eq, IRNodeKind::Eq)
DEFINE_KIND_TRAIT(Ne, IRNodeKind::Ne)
DEFINE_KIND_TRAIT(Lt, IRNodeKind::Lt)
DEFINE_KIND_TRAIT(Le, IRNodeKind::Le)
DEFINE_KIND_TRAIT(Gt, IRNodeKind::Gt)
DEFINE_KIND_TRAIT(Ge, IRNodeKind::Ge)
DEFINE_KIND_TRAIT(And, IRNodeKind::And)
DEFINE_KIND_TRAIT(Or, IRNodeKind::Or)
DEFINE_KIND_TRAIT(Xor, IRNodeKind::Xor)
DEFINE_KIND_TRAIT(BitAnd, IRNodeKind::BitAnd)
DEFINE_KIND_TRAIT(BitOr, IRNodeKind::BitOr)
DEFINE_KIND_TRAIT(BitXor, IRNodeKind::BitXor)
DEFINE_KIND_TRAIT(BitShiftLeft, IRNodeKind::BitShiftLeft)
DEFINE_KIND_TRAIT(BitShiftRight, IRNodeKind::BitShiftRight)

// Unary expression types
DEFINE_KIND_TRAIT(Abs, IRNodeKind::Abs)
DEFINE_KIND_TRAIT(Neg, IRNodeKind::Neg)
DEFINE_KIND_TRAIT(Not, IRNodeKind::Not)
DEFINE_KIND_TRAIT(BitNot, IRNodeKind::BitNot)
DEFINE_KIND_TRAIT(Cast, IRNodeKind::Cast)

// Statement types
DEFINE_KIND_TRAIT(AssignStmt, IRNodeKind::AssignStmt)
DEFINE_KIND_TRAIT(IfStmt, IRNodeKind::IfStmt)
DEFINE_KIND_TRAIT(YieldStmt, IRNodeKind::YieldStmt)
DEFINE_KIND_TRAIT(ReturnStmt, IRNodeKind::ReturnStmt)
DEFINE_KIND_TRAIT(ForStmt, IRNodeKind::ForStmt)
DEFINE_KIND_TRAIT(SeqStmts, IRNodeKind::SeqStmts)
DEFINE_KIND_TRAIT(OpStmts, IRNodeKind::OpStmts)
DEFINE_KIND_TRAIT(EvalStmt, IRNodeKind::EvalStmt)

// Type types
DEFINE_KIND_TRAIT(UnknownType, IRNodeKind::UnknownType)
DEFINE_KIND_TRAIT(ScalarType, IRNodeKind::ScalarType)
DEFINE_KIND_TRAIT(ShapedType, IRNodeKind::ShapedType)
DEFINE_KIND_TRAIT(TensorType, IRNodeKind::TensorType)
DEFINE_KIND_TRAIT(TileType, IRNodeKind::TileType)
DEFINE_KIND_TRAIT(TupleType, IRNodeKind::TupleType)

// Other IR node types
DEFINE_KIND_TRAIT(Function, IRNodeKind::Function)
DEFINE_KIND_TRAIT(Program, IRNodeKind::Program)

#undef DEFINE_KIND_TRAIT

// Convenience overloads for typed pointers (ExprPtr, StmtPtr, TypePtr)
template <typename T>
inline bool IsA(const ExprPtr& expr) {
  return IsA<T>(std::static_pointer_cast<const IRNode>(expr));
}

template <typename T>
inline std::shared_ptr<const T> As(const ExprPtr& expr) {
  return As<T>(std::static_pointer_cast<const IRNode>(expr));
}

template <typename T>
inline bool IsA(const StmtPtr& stmt) {
  return IsA<T>(std::static_pointer_cast<const IRNode>(stmt));
}

template <typename T>
inline std::shared_ptr<const T> As(const StmtPtr& stmt) {
  return As<T>(std::static_pointer_cast<const IRNode>(stmt));
}

// Type does not inherit from IRNode, so we need different overloads
template <typename T>
inline bool IsA(const TypePtr& type) {
  return type && type->GetKind() == KindTrait<T>::kind;
}

template <typename T>
inline std::shared_ptr<const T> As(const TypePtr& type) {
  if (IsA<T>(type)) {
    return std::static_pointer_cast<const T>(type);
  }
  return nullptr;
}

}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_KIND_TRAITS_H_
