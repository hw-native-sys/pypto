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
#include <vector>

#include "pypto/ir/core.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/reflection/field_traits.h"

namespace pypto {
namespace ir {

// Forward declarations for friend classes
class IRVisitor;
class IRMutator;
class IRPrinter;

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

/**
 * @brief Conditional statement
 *
 * Represents an if-else statement: if condition then then_body else else_body
 * where condition is an expression and then_body/else_body are statement lists.
 */
class IfStmt : public Stmt {
 public:
  /**
   * @brief Create a conditional statement
   *
   * @param condition Condition expression
   * @param then_body Then branch statements
   * @param else_body Else branch statements (can be empty)
   * @param span Source location
   */
  IfStmt(ExprPtr condition, std::vector<StmtPtr> then_body, std::vector<StmtPtr> else_body, Span span)
      : Stmt(std::move(span)),
        condition_(std::move(condition)),
        then_body_(std::move(then_body)),
        else_body_(std::move(else_body)) {}

  [[nodiscard]] std::string TypeName() const override { return "IfStmt"; }

  /**
   * @brief Get field descriptors for reflection-based visitation
   *
   * @return Tuple of field descriptors (condition, then_body, else_body as USUAL fields)
   */
  static constexpr auto GetFieldDescriptors() {
    return std::tuple_cat(Stmt::GetFieldDescriptors(),
                          std::make_tuple(reflection::UsualField(&IfStmt::condition_, "condition"),
                                          reflection::UsualField(&IfStmt::then_body_, "then_body"),
                                          reflection::UsualField(&IfStmt::else_body_, "else_body")));
  }

 public:
  ExprPtr condition_;               // Condition expression
  std::vector<StmtPtr> then_body_;  // Then branch statements
  std::vector<StmtPtr> else_body_;  // Else branch statements (can be empty)
};

using IfStmtPtr = std::shared_ptr<const IfStmt>;

/**
 * @brief Yield statement
 *
 * Represents a yield operation: yield value
 * where value is a list of variables to yield.
 */
class YieldStmt : public Stmt {
 public:
  /**
   * @brief Create a yield statement
   *
   * @param value List of variables to yield (can be empty)
   * @param span Source location
   */
  YieldStmt(std::vector<VarPtr> value, Span span) : Stmt(std::move(span)), value_(std::move(value)) {}

  /**
   * @brief Create a yield statement without values
   *
   * @param span Source location
   */
  explicit YieldStmt(Span span) : Stmt(std::move(span)), value_() {}

  [[nodiscard]] std::string TypeName() const override { return "YieldStmt"; }

  /**
   * @brief Get field descriptors for reflection-based visitation
   *
   * @return Tuple of field descriptors (value as USUAL field)
   */
  static constexpr auto GetFieldDescriptors() {
    return std::tuple_cat(Stmt::GetFieldDescriptors(),
                          std::make_tuple(reflection::UsualField(&YieldStmt::value_, "value")));
  }

 public:
  std::vector<VarPtr> value_;  // List of variables to yield (can be empty)
};

using YieldStmtPtr = std::shared_ptr<const YieldStmt>;

/**
 * @brief For loop statement
 *
 * Represents a for loop: for loop_var in range(start, stop, step): body
 * where loop_var is the loop variable, start/stop/step are expressions,
 * and body is a list of statements.
 */
class ForStmt : public Stmt {
 public:
  /**
   * @brief Create a for loop statement
   *
   * @param loop_var Loop variable
   * @param start Start value expression
   * @param stop Stop value expression
   * @param step Step value expression
   * @param body Loop body statements
   * @param span Source location
   */
  ForStmt(VarPtr loop_var, ExprPtr start, ExprPtr stop, ExprPtr step, std::vector<StmtPtr> body, Span span)
      : Stmt(std::move(span)),
        loop_var_(std::move(loop_var)),
        start_(std::move(start)),
        stop_(std::move(stop)),
        step_(std::move(step)),
        body_(std::move(body)) {}

  [[nodiscard]] std::string TypeName() const override { return "ForStmt"; }

  /**
   * @brief Get field descriptors for reflection-based visitation
   *
   * @return Tuple of field descriptors (loop_var as DEF field, others as USUAL fields)
   */
  static constexpr auto GetFieldDescriptors() {
    return std::tuple_cat(Stmt::GetFieldDescriptors(),
                          std::make_tuple(reflection::DefField(&ForStmt::loop_var_, "loop_var"),
                                          reflection::UsualField(&ForStmt::start_, "start"),
                                          reflection::UsualField(&ForStmt::stop_, "stop"),
                                          reflection::UsualField(&ForStmt::step_, "step"),
                                          reflection::UsualField(&ForStmt::body_, "body")));
  }

 public:
  VarPtr loop_var_;            // Loop variable
  ExprPtr start_;              // Start value expression
  ExprPtr stop_;               // Stop value expression
  ExprPtr step_;               // Step value expression
  std::vector<StmtPtr> body_;  // Loop body statements
};

using ForStmtPtr = std::shared_ptr<const ForStmt>;

/**
 * @brief Operation statements
 *
 * Represents a sequence of statements: stmt1; stmt2; ... stmtN
 * where stmts is a list of statements.
 */
class OpStmts : public Stmt {
 public:
  /**
   * @brief Create an operation statements
   *
   * @param stmts List of statements
   * @param span Source location
   */
  OpStmts(std::vector<StmtPtr> stmts, Span span) : Stmt(std::move(span)), stmts_(std::move(stmts)) {}

  [[nodiscard]] std::string TypeName() const override { return "OpStmts"; }

  /**
   * @brief Get field descriptors for reflection-based visitation
   *
   * @return Tuple of field descriptors (stmts as USUAL field)
   */
  static constexpr auto GetFieldDescriptors() {
    return std::tuple_cat(Stmt::GetFieldDescriptors(),
                          std::make_tuple(reflection::UsualField(&OpStmts::stmts_, "stmts")));
  }

 public:
  std::vector<StmtPtr> stmts_;  // List of statements
};

using OpStmtsPtr = std::shared_ptr<const OpStmts>;

}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_STMT_H_
