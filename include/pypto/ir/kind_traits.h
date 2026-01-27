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
    static constexpr ObjectKind kind = KindValue; \
  };

// KindTrait specializations for all concrete IR node types
// These enable compile-time type-to-Kind mapping for IsA<T>() and As<T>()

// Expression types
DEFINE_KIND_TRAIT(Var, ObjectKind::Var)
DEFINE_KIND_TRAIT(IterArg, ObjectKind::IterArg)
DEFINE_KIND_TRAIT(Call, ObjectKind::Call)
DEFINE_KIND_TRAIT(TupleGetItemExpr, ObjectKind::TupleGetItemExpr)
DEFINE_KIND_TRAIT(ConstInt, ObjectKind::ConstInt)
DEFINE_KIND_TRAIT(ConstFloat, ObjectKind::ConstFloat)
DEFINE_KIND_TRAIT(ConstBool, ObjectKind::ConstBool)

// Binary expression types
DEFINE_KIND_TRAIT(Add, ObjectKind::Add)
DEFINE_KIND_TRAIT(Sub, ObjectKind::Sub)
DEFINE_KIND_TRAIT(Mul, ObjectKind::Mul)
DEFINE_KIND_TRAIT(FloorDiv, ObjectKind::FloorDiv)
DEFINE_KIND_TRAIT(FloorMod, ObjectKind::FloorMod)
DEFINE_KIND_TRAIT(FloatDiv, ObjectKind::FloatDiv)
DEFINE_KIND_TRAIT(Min, ObjectKind::Min)
DEFINE_KIND_TRAIT(Max, ObjectKind::Max)
DEFINE_KIND_TRAIT(Pow, ObjectKind::Pow)
DEFINE_KIND_TRAIT(Eq, ObjectKind::Eq)
DEFINE_KIND_TRAIT(Ne, ObjectKind::Ne)
DEFINE_KIND_TRAIT(Lt, ObjectKind::Lt)
DEFINE_KIND_TRAIT(Le, ObjectKind::Le)
DEFINE_KIND_TRAIT(Gt, ObjectKind::Gt)
DEFINE_KIND_TRAIT(Ge, ObjectKind::Ge)
DEFINE_KIND_TRAIT(And, ObjectKind::And)
DEFINE_KIND_TRAIT(Or, ObjectKind::Or)
DEFINE_KIND_TRAIT(Xor, ObjectKind::Xor)
DEFINE_KIND_TRAIT(BitAnd, ObjectKind::BitAnd)
DEFINE_KIND_TRAIT(BitOr, ObjectKind::BitOr)
DEFINE_KIND_TRAIT(BitXor, ObjectKind::BitXor)
DEFINE_KIND_TRAIT(BitShiftLeft, ObjectKind::BitShiftLeft)
DEFINE_KIND_TRAIT(BitShiftRight, ObjectKind::BitShiftRight)

// Unary expression types
DEFINE_KIND_TRAIT(Abs, ObjectKind::Abs)
DEFINE_KIND_TRAIT(Neg, ObjectKind::Neg)
DEFINE_KIND_TRAIT(Not, ObjectKind::Not)
DEFINE_KIND_TRAIT(BitNot, ObjectKind::BitNot)
DEFINE_KIND_TRAIT(Cast, ObjectKind::Cast)

// Statement types
DEFINE_KIND_TRAIT(AssignStmt, ObjectKind::AssignStmt)
DEFINE_KIND_TRAIT(IfStmt, ObjectKind::IfStmt)
DEFINE_KIND_TRAIT(YieldStmt, ObjectKind::YieldStmt)
DEFINE_KIND_TRAIT(ReturnStmt, ObjectKind::ReturnStmt)
DEFINE_KIND_TRAIT(ForStmt, ObjectKind::ForStmt)
DEFINE_KIND_TRAIT(SeqStmts, ObjectKind::SeqStmts)
DEFINE_KIND_TRAIT(OpStmts, ObjectKind::OpStmts)
DEFINE_KIND_TRAIT(EvalStmt, ObjectKind::EvalStmt)

// Type types
DEFINE_KIND_TRAIT(UnknownType, ObjectKind::UnknownType)
DEFINE_KIND_TRAIT(ScalarType, ObjectKind::ScalarType)
// ShapedType is both a concrete type and a base class - handled separately below
DEFINE_KIND_TRAIT(TensorType, ObjectKind::TensorType)
DEFINE_KIND_TRAIT(TileType, ObjectKind::TileType)
DEFINE_KIND_TRAIT(TupleType, ObjectKind::TupleType)

// Other IR node types
DEFINE_KIND_TRAIT(Function, ObjectKind::Function)
DEFINE_KIND_TRAIT(Program, ObjectKind::Program)

// Op kinds
DEFINE_KIND_TRAIT(Op, ObjectKind::Op)
DEFINE_KIND_TRAIT(GlobalVar, ObjectKind::GlobalVar)

#undef DEFINE_KIND_TRAIT

// KindTrait specializations for abstract base classes
// These enable IsA<T>() and As<T>() for base class types

// Stmt base class - matches any statement kind
template <>
struct KindTrait<Stmt> {
  static constexpr ObjectKind kinds[] = {ObjectKind::AssignStmt, ObjectKind::IfStmt,  ObjectKind::YieldStmt,
                                         ObjectKind::ReturnStmt, ObjectKind::ForStmt, ObjectKind::SeqStmts,
                                         ObjectKind::OpStmts,    ObjectKind::EvalStmt};
  static constexpr size_t count = 8;
};

// Expr base class - matches any expression kind
template <>
struct KindTrait<Expr> {
  static constexpr ObjectKind kinds[] = {
      // Direct expression types
      ObjectKind::Var, ObjectKind::IterArg, ObjectKind::Call, ObjectKind::TupleGetItemExpr,
      ObjectKind::ConstInt, ObjectKind::ConstFloat, ObjectKind::ConstBool,
      // Binary expressions (22 kinds)
      ObjectKind::Add, ObjectKind::Sub, ObjectKind::Mul, ObjectKind::FloorDiv, ObjectKind::FloorMod,
      ObjectKind::FloatDiv, ObjectKind::Min, ObjectKind::Max, ObjectKind::Pow, ObjectKind::Eq, ObjectKind::Ne,
      ObjectKind::Lt, ObjectKind::Le, ObjectKind::Gt, ObjectKind::Ge, ObjectKind::And, ObjectKind::Or,
      ObjectKind::Xor, ObjectKind::BitAnd, ObjectKind::BitOr, ObjectKind::BitXor, ObjectKind::BitShiftLeft,
      ObjectKind::BitShiftRight,
      // Unary expressions (5 kinds)
      ObjectKind::Abs, ObjectKind::Neg, ObjectKind::Not, ObjectKind::BitNot, ObjectKind::Cast};
  static constexpr size_t count = 34;
};

// BinaryExpr base class - matches any binary expression kind
template <>
struct KindTrait<BinaryExpr> {
  static constexpr ObjectKind kinds[] = {
      ObjectKind::Add,      ObjectKind::Sub,          ObjectKind::Mul,          ObjectKind::FloorDiv,
      ObjectKind::FloorMod, ObjectKind::FloatDiv,     ObjectKind::Min,          ObjectKind::Max,
      ObjectKind::Pow,      ObjectKind::Eq,           ObjectKind::Ne,           ObjectKind::Lt,
      ObjectKind::Le,       ObjectKind::Gt,           ObjectKind::Ge,           ObjectKind::And,
      ObjectKind::Or,       ObjectKind::Xor,          ObjectKind::BitAnd,       ObjectKind::BitOr,
      ObjectKind::BitXor,   ObjectKind::BitShiftLeft, ObjectKind::BitShiftRight};
  static constexpr size_t count = sizeof(kinds) / sizeof(ObjectKind);
};

// UnaryExpr base class - matches any unary expression kind
template <>
struct KindTrait<UnaryExpr> {
  static constexpr ObjectKind kinds[] = {ObjectKind::Abs, ObjectKind::Neg, ObjectKind::Not,
                                         ObjectKind::BitNot, ObjectKind::Cast};
  static constexpr size_t count = sizeof(kinds) / sizeof(ObjectKind);
};

// Type base class - matches any type kind
template <>
struct KindTrait<Type> {
  static constexpr ObjectKind kinds[] = {ObjectKind::UnknownType, ObjectKind::ScalarType,
                                         ObjectKind::ShapedType,  ObjectKind::TensorType,
                                         ObjectKind::TileType,    ObjectKind::TupleType};
  static constexpr size_t count = sizeof(kinds) / sizeof(ObjectKind);
};

// ShapedType can be used as both a concrete type and a base class
// It matches itself, TensorType, and TileType
template <>
struct KindTrait<ShapedType> {
  // For base class matching: includes ShapedType, TensorType, TileType
  static constexpr ObjectKind kinds[] = {ObjectKind::ShapedType, ObjectKind::TensorType,
                                         ObjectKind::TileType};
  static constexpr size_t count = sizeof(kinds) / sizeof(ObjectKind);
};

}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_KIND_TRAITS_H_
