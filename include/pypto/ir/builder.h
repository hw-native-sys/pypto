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

#ifndef PYPTO_IR_BUILDER_H_
#define PYPTO_IR_BUILDER_H_

#include <memory>
#include <stack>
#include <string>
#include <utility>
#include <vector>

#include "pypto/ir/core.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

// Forward declarations
class BuildContext;
class FunctionContext;
class ForLoopContext;
class IfStmtContext;

/**
 * @brief IR Builder for incremental IR construction with context management
 *
 * The IRBuilder provides a stateful API for building IR incrementally using
 * Begin/End patterns (C++) or context managers (Python). It maintains a
 * context stack to track nested scopes and validates proper construction.
 *
 * Key features:
 * - Stack-based context management
 * - All methods accept explicit Span parameters
 * - Validates proper nesting and construction
 * - Supports functions, for loops, and if statements
 *
 * Example usage (C++):
 * @code
 * IRBuilder builder;
 * auto span = Span(__FILE__, __LINE__, 0);
 * builder.BeginFunction("my_func", span);
 * auto x = builder.FuncArg("x", ScalarType::Create(DataType::INT64), span);
 * builder.ReturnType(ScalarType::Create(DataType::INT64));
 * // ... build body ...
 * auto func = builder.EndFunction(span);
 * @endcode
 */
class IRBuilder {
 public:
  IRBuilder();
  ~IRBuilder();

  // Disable copying and moving since we have unique_ptr members
  IRBuilder(const IRBuilder&) = delete;
  IRBuilder& operator=(const IRBuilder&) = delete;
  IRBuilder(IRBuilder&&) = delete;
  IRBuilder& operator=(IRBuilder&&) = delete;

  // ========== Function Building ==========

  /**
   * @brief Begin building a function
   *
   * Creates a new function context and pushes it onto the context stack.
   * Must be closed with EndFunction().
   *
   * @param name Function name
   * @param span Source location for function definition
   * @throws RuntimeError if already inside a function (no nested functions allowed)
   */
  void BeginFunction(const std::string& name, const Span& span);

  /**
   * @brief Add a function parameter
   *
   * Must be called within a function context (after BeginFunction).
   *
   * @param name Parameter name
   * @param type Parameter type
   * @param span Source location for parameter
   * @return Variable representing the parameter
   * @throws RuntimeError if not inside a function context
   */
  VarPtr FuncArg(const std::string& name, const TypePtr& type, const Span& span);

  /**
   * @brief Add a return type to the current function
   *
   * Can be called multiple times to add multiple return types.
   *
   * @param type Return type
   * @throws RuntimeError if not inside a function context
   */
  void ReturnType(const TypePtr& type);

  /**
   * @brief End building a function
   *
   * Finalizes the function and pops the function context from the stack.
   *
   * @param end_span Source location for end of function
   * @return The built function
   * @throws RuntimeError if not inside a function context
   */
  FunctionPtr EndFunction(const Span& end_span);

  // ========== For Loop Building ==========

  /**
   * @brief Begin building a for loop
   *
   * Creates a new for loop context and pushes it onto the context stack.
   * Must be closed with EndForLoop().
   *
   * @param loop_var Loop variable
   * @param start Start value expression
   * @param stop Stop value expression
   * @param step Step value expression
   * @param span Source location for loop definition
   * @throws RuntimeError if not inside a function or another loop
   */
  void BeginForLoop(const VarPtr& loop_var, const ExprPtr& start, const ExprPtr& stop, const ExprPtr& step,
                    const Span& span);

  /**
   * @brief Add an iteration argument to the current for loop
   *
   * Iteration arguments are loop-carried values (SSA-style).
   *
   * @param iter_arg Iteration argument with initial value
   * @throws RuntimeError if not inside a for loop context
   */
  void AddIterArg(const IterArgPtr& iter_arg);

  /**
   * @brief Add a return variable to the current for loop
   *
   * Return variables capture the final values of iteration arguments.
   * The number of return variables must match the number of iteration arguments.
   *
   * @param var Return variable
   * @throws RuntimeError if not inside a for loop context
   */
  void AddReturnVar(const VarPtr& var);

  /**
   * @brief End building a for loop
   *
   * Finalizes the loop and pops the loop context from the stack.
   *
   * @param end_span Source location for end of loop
   * @return The built for statement
   * @throws RuntimeError if not inside a for loop context
   * @throws RuntimeError if number of return variables doesn't match iteration arguments
   */
  StmtPtr EndForLoop(const Span& end_span);

  // ========== If Statement Building ==========

  /**
   * @brief Begin building an if statement
   *
   * Creates a new if context and pushes it onto the context stack.
   * Must be closed with EndIf().
   *
   * @param condition Condition expression
   * @param span Source location for if statement
   * @throws RuntimeError if not inside a function or loop
   */
  void BeginIf(const ExprPtr& condition, const Span& span);

  /**
   * @brief Begin the else branch of the current if statement
   *
   * Must be called after building the then branch and before EndIf().
   *
   * @param span Source location for else keyword
   * @throws RuntimeError if not inside an if context
   * @throws RuntimeError if else branch already begun
   */
  void BeginElse(const Span& span);

  /**
   * @brief Add a return variable to the current if statement
   *
   * Return variables are used for SSA phi nodes when if has return values.
   *
   * @param var Return variable
   * @throws RuntimeError if not inside an if context
   */
  void AddIfReturnVar(const VarPtr& var);

  /**
   * @brief End building an if statement
   *
   * Finalizes the if statement and pops the context from the stack.
   *
   * @param end_span Source location for end of if
   * @return The built if statement
   * @throws RuntimeError if not inside an if context
   */
  StmtPtr EndIf(const Span& end_span);

  // ========== Statement Recording ==========

  /**
   * @brief Emit a statement in the current context
   *
   * Adds a statement to the current context's statement list.
   *
   * @param stmt Statement to emit
   * @throws RuntimeError if not inside a valid context for emitting statements
   */
  void Emit(const StmtPtr& stmt);

  /**
   * @brief Create an assignment statement and emit it
   *
   * Convenience method that creates an assignment and emits it.
   *
   * @param var Variable to assign to
   * @param value Expression value
   * @param span Source location for assignment
   * @return The created assignment statement
   * @throws RuntimeError if not inside a valid context
   */
  AssignStmtPtr Assign(const VarPtr& var, const ExprPtr& value, const Span& span);

  /**
   * @brief Create a variable (does not emit)
   *
   * Helper to create a variable. User must create assignment separately.
   *
   * @param name Variable name
   * @param type Variable type
   * @param span Source location
   * @return The created variable
   */
  VarPtr Var(const std::string& name, const TypePtr& type, const Span& span);

  /**
   * @brief Create a return statement and emit it
   *
   * Convenience method that creates a return statement and emits it.
   *
   * @param values List of expressions to return (can be empty)
   * @param span Source location for return statement
   * @return The created return statement
   * @throws RuntimeError if not inside a valid context
   */
  ReturnStmtPtr Return(const std::vector<ExprPtr>& values, const Span& span);

  /**
   * @brief Create a return statement without values and emit it
   *
   * Convenience method that creates an empty return statement and emits it.
   *
   * @param span Source location for return statement
   * @return The created return statement
   * @throws RuntimeError if not inside a valid context
   */
  ReturnStmtPtr Return(const Span& span);

  // ========== Context State Queries ==========

  /**
   * @brief Get the current context
   *
   * @return Pointer to current context, or nullptr if no context
   */
  BuildContext* CurrentContext();

  /**
   * @brief Check if currently inside a function
   *
   * @return true if inside a function context
   */
  bool InFunction() const;

  /**
   * @brief Check if currently inside a for loop
   *
   * @return true if inside a for loop context
   */
  bool InLoop() const;

  /**
   * @brief Check if currently inside an if statement
   *
   * @return true if inside an if statement context
   */
  bool InIf() const;

 private:
  std::vector<std::unique_ptr<BuildContext>> context_stack_;

  // Helper to get current context with type checking
  template <typename T>
  T* GetCurrentContextAs();

  // Helper to validate we're in the right context
  void ValidateInFunction(const std::string& operation);
  void ValidateInLoop(const std::string& operation);
  void ValidateInIf(const std::string& operation);
};

/**
 * @brief Base class for build contexts
 *
 * Each context type (function, loop, if) maintains state for building
 * that construct incrementally.
 */
class BuildContext {
 public:
  enum class Type { FUNCTION, FOR_LOOP, IF_STMT };

  explicit BuildContext(Type type, Span span) : type_(type), begin_span_(std::move(span)) {}
  virtual ~BuildContext() = default;

  Type GetType() const { return type_; }
  const Span& GetBeginSpan() const { return begin_span_; }

  // Accumulate statements in this context
  void AddStmt(const StmtPtr& stmt) { stmts_.push_back(stmt); }
  const std::vector<StmtPtr>& GetStmts() const { return stmts_; }

 protected:
  Type type_;
  Span begin_span_;
  std::vector<StmtPtr> stmts_;
};

/**
 * @brief Context for building a function
 */
class FunctionContext : public BuildContext {
 public:
  FunctionContext(std::string name, Span span)
      : BuildContext(Type::FUNCTION, std::move(span)), name_(std::move(name)) {}

  void AddParam(const VarPtr& param) { params_.push_back(param); }
  void AddReturnType(const TypePtr& type) { return_types_.push_back(type); }

  const std::string& GetName() const { return name_; }
  const std::vector<VarPtr>& GetParams() const { return params_; }
  const std::vector<TypePtr>& GetReturnTypes() const { return return_types_; }

 private:
  std::string name_;
  std::vector<VarPtr> params_;
  std::vector<TypePtr> return_types_;
};

/**
 * @brief Context for building a for loop
 */
class ForLoopContext : public BuildContext {
 public:
  ForLoopContext(VarPtr loop_var, ExprPtr start, ExprPtr stop, ExprPtr step, Span span)
      : BuildContext(Type::FOR_LOOP, std::move(span)),
        loop_var_(std::move(loop_var)),
        start_(std::move(start)),
        stop_(std::move(stop)),
        step_(std::move(step)) {}

  void AddIterArg(const IterArgPtr& iter_arg) { iter_args_.push_back(iter_arg); }
  void AddReturnVar(const VarPtr& var) { return_vars_.push_back(var); }

  const VarPtr& GetLoopVar() const { return loop_var_; }
  const ExprPtr& GetStart() const { return start_; }
  const ExprPtr& GetStop() const { return stop_; }
  const ExprPtr& GetStep() const { return step_; }
  const std::vector<IterArgPtr>& GetIterArgs() const { return iter_args_; }
  const std::vector<VarPtr>& GetReturnVars() const { return return_vars_; }

 private:
  VarPtr loop_var_;
  ExprPtr start_;
  ExprPtr stop_;
  ExprPtr step_;
  std::vector<IterArgPtr> iter_args_;
  std::vector<VarPtr> return_vars_;
};

/**
 * @brief Context for building an if statement
 */
class IfStmtContext : public BuildContext {
 public:
  IfStmtContext(ExprPtr condition, Span span)
      : BuildContext(Type::IF_STMT, std::move(span)),
        condition_(std::move(condition)),
        in_else_branch_(false) {}

  void BeginElseBranch() {
    in_else_branch_ = true;
    else_stmts_.clear();
  }

  void AddElseStmt(const StmtPtr& stmt) { else_stmts_.push_back(stmt); }
  void AddReturnVar(const VarPtr& var) { return_vars_.push_back(var); }

  const ExprPtr& GetCondition() const { return condition_; }
  bool InElseBranch() const { return in_else_branch_; }
  const std::vector<StmtPtr>& GetElseStmts() const { return else_stmts_; }
  const std::vector<VarPtr>& GetReturnVars() const { return return_vars_; }

 private:
  ExprPtr condition_;
  bool in_else_branch_;
  std::vector<StmtPtr> else_stmts_;
  std::vector<VarPtr> return_vars_;
};

}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_BUILDER_H_
