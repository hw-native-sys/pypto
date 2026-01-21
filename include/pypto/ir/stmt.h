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
#include <optional>
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
 * where condition is an expression and then_body/else_body is statement.
 */
class IfStmt : public Stmt {
 public:
  /**
   * @brief Create a conditional statement with then and else branches
   *
   * @param condition Condition expression
   * @param then_body Then branch statement
   * @param else_body Else branch statement (can be optional)
   * @param return_vars Return variables (can be empty)
   * @param span Source location
   */
  IfStmt(ExprPtr condition, StmtPtr then_body, std::optional<StmtPtr> else_body,
         std::vector<VarPtr> return_vars, Span span)
      : Stmt(std::move(span)),
        condition_(std::move(condition)),
        then_body_(std::move(then_body)),
        else_body_(std::move(else_body)),
        return_vars_(std::move(return_vars)) {}

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
                                          reflection::UsualField(&IfStmt::else_body_, "else_body"),
                                          reflection::DefField(&IfStmt::return_vars_, "return_vars")));
  }

 public:
  ExprPtr condition_;                 // Condition expression
  StmtPtr then_body_;                 // Then branch statement
  std::optional<StmtPtr> else_body_;  // Else branch statement (optional)
  std::vector<VarPtr> return_vars_;   // Return variables (can be empty)
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
  YieldStmt(std::vector<ExprPtr> value, Span span) : Stmt(std::move(span)), value_(std::move(value)) {}

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
  std::vector<ExprPtr> value_;  // List of expressions to yield
};

using YieldStmtPtr = std::shared_ptr<const YieldStmt>;

/**
 * @brief Return statement
 *
 * Represents a return operation: return value
 * where value is a list of expressions to return.
 */
class ReturnStmt : public Stmt {
 public:
  /**
   * @brief Create a return statement
   *
   * @param value List of expressions to return (can be empty)
   * @param span Source location
   */
  ReturnStmt(std::vector<ExprPtr> value, Span span) : Stmt(std::move(span)), value_(std::move(value)) {}

  /**
   * @brief Create a return statement without values
   *
   * @param span Source location
   */
  explicit ReturnStmt(Span span) : Stmt(std::move(span)), value_() {}

  [[nodiscard]] std::string TypeName() const override { return "ReturnStmt"; }

  /**
   * @brief Get field descriptors for reflection-based visitation
   *
   * @return Tuple of field descriptors (value as USUAL field)
   */
  static constexpr auto GetFieldDescriptors() {
    return std::tuple_cat(Stmt::GetFieldDescriptors(),
                          std::make_tuple(reflection::UsualField(&ReturnStmt::value_, "value")));
  }

 public:
  std::vector<ExprPtr> value_;  // List of expressions to return
};

using ReturnStmtPtr = std::shared_ptr<const ReturnStmt>;

/**
 * @brief For loop statement
 *
 * Represents a for loop with optional loop-carried values (SSA-style iteration).
 *
 * **Basic loop:** for loop_var in range(start, stop, step): body
 *
 * **Loop with iteration arguments:**
 * for loop_var, (iter_arg1, iter_arg2) in pl.range(start, stop, step, init_values=[...]):
 *     iter_arg1, iter_arg2 = pl.yield_(new_val1, new_val2)
 * return_var1 = iter_arg1
 * return_var2 = iter_arg2
 *
 * **Key Relationships:**
 * - iter_args: IterArg variables scoped to loop body, carry values between iterations
 * - return_vars: Var variables that capture final iteration values, accessible after loop
 * - Number of iter_args must equal number of return_vars
 * - Number of yielded values must equal number of iter_args
 * - IterArgs cannot be directly accessed outside the loop; use return_vars instead
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
   * @param iter_args Iteration arguments (loop-carried values, scoped to loop body)
   * @param body Loop body statement (must yield values matching iter_args if non-empty)
   * @param return_vars Return variables (capture final values, accessible after loop)
   * @param span Source location
   */
  ForStmt(VarPtr loop_var, ExprPtr start, ExprPtr stop, ExprPtr step, std::vector<IterArgPtr> iter_args,
          StmtPtr body, std::vector<VarPtr> return_vars, Span span)
      : Stmt(std::move(span)),
        loop_var_(std::move(loop_var)),
        start_(std::move(start)),
        stop_(std::move(stop)),
        step_(std::move(step)),
        iter_args_(std::move(iter_args)),
        body_(std::move(body)),
        return_vars_(std::move(return_vars)) {}

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
                                          reflection::DefField(&ForStmt::iter_args_, "iter_args"),
                                          reflection::UsualField(&ForStmt::body_, "body"),
                                          reflection::DefField(&ForStmt::return_vars_, "return_vars")));
  }

 public:
  VarPtr loop_var_;                    // Loop variable (e.g., i in "for i in range(...)")
  ExprPtr start_;                      // Start value expression
  ExprPtr stop_;                       // Stop value expression
  ExprPtr step_;                       // Step value expression
  std::vector<IterArgPtr> iter_args_;  // Loop-carried values (scoped to loop body)
  StmtPtr body_;                       // Loop body statement (must yield if iter_args non-empty)
  std::vector<VarPtr> return_vars_;    // Variables capturing final iteration values (accessible after loop)
};

using ForStmtPtr = std::shared_ptr<const ForStmt>;

/**
 * @brief Sequence of statements
 *
 * Represents a sequence of statements: stmt1; stmt2; ... stmtN
 * where stmts is a list of statements.
 */
class SeqStmts : public Stmt {
 public:
  /**
   * @brief Create a sequence of statements
   *
   * @param stmts List of statements
   * @param span Source location
   */
  SeqStmts(std::vector<StmtPtr> stmts, Span span) : Stmt(std::move(span)), stmts_(std::move(stmts)) {}

  [[nodiscard]] std::string TypeName() const override { return "SeqStmts"; }

  /**
   * @brief Get field descriptors for reflection-based visitation
   *
   * @return Tuple of field descriptors (stmts as USUAL field)
   */
  static constexpr auto GetFieldDescriptors() {
    return std::tuple_cat(Stmt::GetFieldDescriptors(),
                          std::make_tuple(reflection::UsualField(&SeqStmts::stmts_, "stmts")));
  }

 public:
  std::vector<StmtPtr> stmts_;  // List of statements
};

using SeqStmtsPtr = std::shared_ptr<const SeqStmts>;

/**
 * @brief Operation statements
 *
 * Represents a sequence of assignment statements: assign1; assign2; ... assignN
 * where stmts is a list of assignment statements.
 */
class OpStmts : public Stmt {
 public:
  /**
   * @brief Create an operation statements
   *
   * @param stmts List of assignment statements
   * @param span Source location
   */
  OpStmts(std::vector<AssignStmtPtr> stmts, Span span) : Stmt(std::move(span)), stmts_(std::move(stmts)) {}

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
  std::vector<AssignStmtPtr> stmts_;  // List of assignment statements
};

using OpStmtsPtr = std::shared_ptr<const OpStmts>;

/**
 * @brief Evaluation statement
 *
 * Represents an expression executed as a statement: expr
 * where expr is an expression (typically a Call).
 * This is used for expressions that have side effects but no return value
 * (or return value is ignored).
 */
class EvalStmt : public Stmt {
 public:
  /**
   * @brief Create an evaluation statement
   *
   * @param expr Expression to execute
   * @param span Source location
   */
  EvalStmt(ExprPtr expr, Span span) : Stmt(std::move(span)), expr_(std::move(expr)) {}

  [[nodiscard]] std::string TypeName() const override { return "EvalStmt"; }

  /**
   * @brief Get field descriptors for reflection-based visitation
   *
   * @return Tuple of field descriptors (expr as USUAL field)
   */
  static constexpr auto GetFieldDescriptors() {
    return std::tuple_cat(Stmt::GetFieldDescriptors(),
                          std::make_tuple(reflection::UsualField(&EvalStmt::expr_, "expr")));
  }

 public:
  ExprPtr expr_;  // Expression
};

using EvalStmtPtr = std::shared_ptr<const EvalStmt>;

}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_STMT_H_
