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

#ifndef PYPTO_IR_STMT_H_
#define PYPTO_IR_STMT_H_

#include <memory>
#include <string>
#include <tuple>
#include <utility>

#include "pypto/ir/core.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/reflection/field_traits.h"

namespace pypto {
namespace ir {

/**
 * @brief Base class for all statements in the IR
 *
 * Statements represent operations that perform side effects or control flow.
 * All statements are immutable.
 */
class Stmt : public IRNode {
 public:
  /**
   * @brief Create a statement
   *
   * @param span Source location
   */
  explicit Stmt(Span s) : IRNode(std::move(s)) {}
  ~Stmt() override = default;

  /**
   * @brief Get the type name of this statement
   *
   * @return Human-readable type name (e.g., "Stmt", "Assign", "Return")
   */
  [[nodiscard]] std::string TypeName() const override { return "Stmt"; }

  static constexpr auto GetFieldDescriptors() { return IRNode::GetFieldDescriptors(); }
};

using StmtPtr = std::shared_ptr<const Stmt>;

/**
 * @brief Assignment statement
 *
 * Represents an assignment operation: var = value
 * where var is a variable and value is an expression.
 */
class AssignStmt : public Stmt {
 public:
  VarPtr var_;     // Variable
  ExprPtr value_;  // Expression

  /**
   * @brief Create an assignment statement
   *
   * @param var Variable
   * @param value Expression
   * @param span Source location
   */
  AssignStmt(VarPtr var, ExprPtr value, Span span)
      : Stmt(std::move(span)), var_(std::move(var)), value_(std::move(value)) {}

  [[nodiscard]] std::string TypeName() const override { return "AssignStmt"; }

  /**
   * @brief Get field descriptors for reflection-based visitation
   *
   * @return Tuple of field descriptors (var and value as DEF and USUAL fields)
   */
  static constexpr auto GetFieldDescriptors() {
    return std::tuple_cat(Stmt::GetFieldDescriptors(),
                          std::make_tuple(reflection::DefField(&AssignStmt::var_, "var"),
                                          reflection::UsualField(&AssignStmt::value_, "value")));
  }
};

using AssignStmtPtr = std::shared_ptr<const AssignStmt>;

}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_STMT_H_
