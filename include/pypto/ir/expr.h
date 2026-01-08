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

#ifndef PYPTO_IR_EXPR_H_
#define PYPTO_IR_EXPR_H_

#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "pypto/ir/core.h"
#include "pypto/ir/reflection/field_traits.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

/**
 * @brief Base class for all expressions in the IR
 *
 * This is the root base class for all expression types (scalar, tensor, etc).
 * Expressions represent computations that produce values.
 * All expressions are immutable.
 */
class Expr : public IRNode {
 protected:
  TypePtr type_;  // Type of the expression result

 public:
  /**
   * @brief Create an expression
   *
   * @param span Source location
   * @param type Type of the expression result (defaults to UnknownType)
   */
  explicit Expr(Span s, TypePtr type = GetUnknownType()) : IRNode(std::move(s)), type_(std::move(type)) {}
  ~Expr() override = default;

  /**
   * @brief Get the type name of this expression
   *
   * @return Human-readable type name (e.g., "ScalarExpr", "Var", "Call")
   */
  [[nodiscard]] std::string TypeName() const override { return "Expr"; }

  /**
   * @brief Get the type of this expression
   *
   * @return Type pointer of the expression result
   */
  [[nodiscard]] const TypePtr& GetType() const { return type_; }

  static constexpr auto GetFieldDescriptors() {
    return std::tuple_cat(IRNode::GetFieldDescriptors(),
                          std::make_tuple(reflection::UsualField(&Expr::type_, "type")));
  }
};

using ExprPtr = std::shared_ptr<const Expr>;

/**
 * @brief Base class for operations/functions
 *
 * Represents callable operations in the IR.
 */
class Op {
 public:
  std::string name_;

  explicit Op(std::string name) : name_(std::move(name)) {}
  virtual ~Op() = default;
};

using OpPtr = std::shared_ptr<const Op>;

/**
 * @brief Variable reference expression
 *
 * Represents a reference to a named variable.
 * Can represent both scalar and tensor variables based on its type.
 */
class Var : public Expr {
 public:
  std::string name_;

  /**
   * @brief Create a variable reference
   *
   * @param name Variable name
   * @param type Type of the variable (ScalarType or TensorType)
   * @param span Source location
   * @return Shared pointer to const Var expression
   */
  Var(std::string name, TypePtr type, Span span)
      : Expr(std::move(span), std::move(type)), name_(std::move(name)) {}

  [[nodiscard]] std::string TypeName() const override { return "Var"; }

  /**
   * @brief Get field descriptors for reflection-based visitation
   *
   * @return Tuple of field descriptors (name_ as USUAL field, type_ is in Expr)
   */
  static constexpr auto GetFieldDescriptors() {
    return std::tuple_cat(Expr::GetFieldDescriptors(),
                          std::make_tuple(reflection::IgnoreField(&Var::name_, "name")));
  }
};

using VarPtr = std::shared_ptr<const Var>;

/**
 * @brief Function call expression
 *
 * Represents a function call with an operation and arguments.
 * Can accept any Expr as arguments, not just scalar expressions.
 */
class Call : public Expr {
 public:
  OpPtr op_;                   // Operation/function
  std::vector<ExprPtr> args_;  // Arguments

  /**
   * @brief Create a function call expression
   *
   * @param op Operation/function to call
   * @param args List of argument expressions
   * @param span Source location
   */
  Call(OpPtr op, std::vector<ExprPtr> args, Span span)
      : Expr(std::move(span)), op_(std::move(op)), args_(std::move(args)) {}

  [[nodiscard]] std::string TypeName() const override { return "Call"; }

  /**
   * @brief Get field descriptors for reflection-based visitation
   *
   * @return Tuple of field descriptors (op and args as USUAL fields)
   */
  static constexpr auto GetFieldDescriptors() {
    return std::tuple_cat(Expr::GetFieldDescriptors(),
                          std::make_tuple(reflection::UsualField(&Call::op_, "op"),
                                          reflection::UsualField(&Call::args_, "args")));
  }
};

using CallPtr = std::shared_ptr<const Call>;

}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_EXPR_H_
