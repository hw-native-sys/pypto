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

#include <any>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/error.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/program.h"
#include "pypto/ir/span.h"
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
  ~IRBuilder() = default;

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
   * @param type Function type (default: Opaque)
   * @param level Hierarchy level (default: nullopt — unspecified)
   * @param role Function role (default: nullopt)
   * @throws RuntimeError if already inside a function (no nested functions allowed)
   */
  void BeginFunction(const std::string& name, const Span& span, FunctionType type = FunctionType::Opaque,
                     std::optional<Level> level = std::nullopt, std::optional<Role> role = std::nullopt,
                     std::vector<std::pair<std::string, std::any>> attrs = {});

  /**
   * @brief Add a function parameter
   *
   * Must be called within a function context (after BeginFunction).
   *
   * @param name Parameter name
   * @param type Parameter type
   * @param span Source location for parameter
   * @param direction Parameter direction (default: In)
   * @return Variable representing the parameter
   * @throws RuntimeError if not inside a function context
   */
  VarPtr FuncArg(const std::string& name, const TypePtr& type, const Span& span,
                 ParamDirection direction = ParamDirection::In);

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
   * @param kind Loop kind (Sequential or Parallel, default: Sequential)
   * @throws RuntimeError if not inside a function or another loop
   */
  void BeginForLoop(const VarPtr& loop_var, const ExprPtr& start, const ExprPtr& stop, const ExprPtr& step,
                    const Span& span, ForKind kind = ForKind::Sequential,
                    std::optional<ChunkConfig> chunk_config = std::nullopt,
                    std::vector<std::pair<std::string, std::any>> attrs = {});

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

  // ========== While Loop Building ==========

  /**
   * @brief Begin building a while loop
   *
   * Creates a new while loop context and pushes it onto the context stack.
   * Must be closed with EndWhileLoop().
   *
   * @param condition Condition expression
   * @param span Source location for loop definition
   * @throws RuntimeError if not inside a function or another loop
   */
  void BeginWhileLoop(const ExprPtr& condition, const Span& span);

  /**
   * @brief Add an iteration argument to the current while loop
   *
   * Iteration arguments are loop-carried values (SSA-style).
   *
   * @param iter_arg Iteration argument with initial value
   * @throws RuntimeError if not inside a while loop context
   */
  void AddWhileIterArg(const IterArgPtr& iter_arg);

  /**
   * @brief Add a return variable to the current while loop
   *
   * Return variables capture the final values of iteration arguments.
   * The number of return variables must match the number of iteration arguments.
   *
   * @param var Return variable
   * @throws RuntimeError if not inside a while loop context
   */
  void AddWhileReturnVar(const VarPtr& var);

  /**
   * @brief Set the condition for the current while loop
   *
   * Used to update the loop condition after setting up iter_args. This allows
   * the condition to reference iter_arg variables that are defined in the loop.
   *
   * @param condition New condition expression
   * @throws RuntimeError if not inside a while loop context
   */
  void SetWhileLoopCondition(const ExprPtr& condition);

  /**
   * @brief End building a while loop
   *
   * Finalizes the loop and pops the loop context from the stack.
   *
   * @param end_span Source location for end of loop
   * @return The built while statement
   * @throws RuntimeError if not inside a while loop context
   * @throws RuntimeError if number of return variables doesn't match iteration arguments
   */
  StmtPtr EndWhileLoop(const Span& end_span);

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

  // ========== Scope Building ==========

  /**
   * @brief Begin building a scope statement
   *
   * Creates a new scope context and pushes it onto the context stack.
   * Must be closed with EndScope().
   *
   * @param scope_kind The kind of scope (e.g., InCore)
   * @param span Source location for scope statement
   * @param level Hierarchy level (for Hierarchy scopes)
   * @param role Function role (for Hierarchy scopes)
   * @throws RuntimeError if not inside a function or loop
   */
  void BeginScope(ScopeKind scope_kind, const Span& span, std::optional<Level> level = std::nullopt,
                  std::optional<Role> role = std::nullopt, std::optional<SplitMode> split = std::nullopt,
                  std::string name_hint = "", ExprPtr core_num = nullptr,
                  std::optional<bool> sync_start = std::nullopt, std::optional<bool> manual = std::nullopt);

  /**
   * @brief End building a scope statement
   *
   * Finalizes the scope statement and pops the context from the stack.
   *
   * @param end_span Source location for end of scope
   * @return The built scope statement
   * @throws RuntimeError if not inside a scope context
   */
  StmtPtr EndScope(const Span& end_span);

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
   * @brief Push leading comments onto the pending stack.
   *
   * The DSL parser calls this before dispatching to a ``parse_*`` helper.
   * The first stmt emitted in the same context as the push (e.g. the outer
   * compound stmt) absorbs the queued comments through its ctor path. If
   * the helper emits nothing (e.g. ``pl.static_assert``), call
   * ``PopPendingLeadingComments`` afterward to recover the unconsumed entry.
   *
   * The stack lets nested ``parse_statement`` calls (parent + body stmts)
   * keep their pending entries independent: the parent's queue waits on the
   * outer context while inner-stmt queues consume in the inner body
   * context. Without a stack, the outer entry would be clobbered when the
   * body parse sets its own pending.
   *
   * @param comments Comment lines (without leading '#')
   */
  void PushPendingLeadingComments(std::vector<std::string> comments);

  /**
   * @brief Pop the top pending entry, returning whatever stayed unconsumed.
   *
   * The parser pairs one pop with every push. If the dispatched helper
   * emitted a matching stmt, the popped entry is empty. Otherwise the
   * parser re-queues the returned comments onto the next line so they
   * land on the next source stmt instead of being silently dropped.
   */
  [[nodiscard]] std::vector<std::string> PopPendingLeadingComments();

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
  [[nodiscard]] bool InFunction() const;

  /**
   * @brief Check if currently inside a for loop
   *
   * @return true if inside a for loop context
   */
  [[nodiscard]] bool InLoop() const;

  /**
   * @brief Check if currently inside an if statement
   *
   * @return true if inside an if statement context
   */
  [[nodiscard]] bool InIf() const;

  /**
   * @brief Check if currently inside a while loop
   *
   * @return true if inside a while loop context
   */
  [[nodiscard]] bool InWhileLoop() const;

  // ========== Program Building ==========

  /**
   * @brief Begin building a program
   *
   * Creates a new program context and pushes it onto the context stack.
   * Must be closed with EndProgram().
   *
   * @param name Program name
   * @param span Source location for program definition
   * @throws RuntimeError if already inside another program
   */
  void BeginProgram(const std::string& name, const Span& span);

  /**
   * @brief Declare a function in the current program
   *
   * Creates a GlobalVar for the function that can be used in Call expressions
   * before the function is fully built. This enables cross-function calls.
   *
   * @param func_name Function name to declare
   * @return GlobalVar that can be used in Call expressions
   * @throws RuntimeError if not inside a program context
   */
  GlobalVarPtr DeclareFunction(const std::string& func_name);

  /**
   * @brief Get a GlobalVar for a declared function
   *
   * Retrieves a GlobalVar that was previously declared with DeclareFunction.
   *
   * @param func_name Function name
   * @return GlobalVar for the function
   * @throws RuntimeError if not inside a program context or function not declared
   */
  GlobalVarPtr GetGlobalVar(const std::string& func_name);

  /**
   * @brief Add a completed function to the current program
   *
   * The function must have been previously declared with DeclareFunction.
   *
   * @param func Completed function to add
   * @throws RuntimeError if not inside a program context
   */
  void AddFunction(const FunctionPtr& func);

  /**
   * @brief End building a program
   *
   * Finalizes the program and pops the program context from the stack.
   *
   * @param end_span Source location for end of program
   * @return The built program
   * @throws RuntimeError if not inside a program context
   */
  ProgramPtr EndProgram(const Span& end_span);

  /**
   * @brief Check if currently inside a program
   *
   * @return true if inside a program context
   */
  [[nodiscard]] bool InProgram() const;

  /**
   * @brief Get return types for a function by its GlobalVar
   *
   * Returns the return types for a function if it has been added to the program.
   * Returns empty vector if not inside a program or function not yet added.
   *
   * @param gvar GlobalVar for the function
   * @return Vector of return types
   */
  [[nodiscard]] std::vector<TypePtr> GetFunctionReturnTypes(const GlobalVarPtr& gvar) const;

 private:
  std::vector<std::unique_ptr<BuildContext>> context_stack_;

  // Stack of pending leading-comment entries. The parser pushes one entry
  // per ``parse_statement`` call (before dispatching) and pops it afterward.
  // Each entry is bound to the context that was current at push time — Emit
  // consumes only when the top-of-stack entry's target matches the context
  // receiving the new stmt. This lets nested parse_statement calls coexist:
  // outer push waits on the outer (FunctionContext etc.) for the compound
  // stmt, inner push waits on the inner (ForContext etc.) for the body
  // stmt. Without the target binding, inner-body emits would steal the
  // outer queue and leave the compound stmt without its comments.
  struct PendingLeadingCommentsEntry {
    std::vector<std::string> comments;
    BuildContext* target;
  };
  std::vector<PendingLeadingCommentsEntry> pending_leading_stack_;

  // If the current context matches pending_target_context_, apply the queued
  // leading comments to `stmt`. Called from Emit() and from every compound-stmt
  // exit path (EndForLoop / EndWhileLoop / EndIf / EndScope) before appending
  // the synthesized stmt to the outer context.
  void ApplyPendingLeadingComments(const StmtPtr& stmt);

  // Helper to get current context with type checking
  template <typename T>
  T* GetCurrentContextAs();

  // Helper to validate we're in the right context
  void ValidateInFunction(const std::string& operation);
  void ValidateInLoop(const std::string& operation);
  void ValidateInIf(const std::string& operation);
  void ValidateInWhileLoop(const std::string& operation);
  void ValidateInProgram(const std::string& operation);
};

/**
 * @brief Base class for build contexts
 *
 * Each context type (function, loop, if) maintains state for building
 * that construct incrementally.
 */
class BuildContext {
 public:
  enum class Type { FUNCTION, FOR_LOOP, WHILE_LOOP, IF_STMT, SCOPE, PROGRAM };

  explicit BuildContext(Type type, Span span) : type_(type), begin_span_(std::move(span)) {}
  virtual ~BuildContext() = default;

  [[nodiscard]] Type GetType() const { return type_; }
  [[nodiscard]] const Span& GetBeginSpan() const { return begin_span_; }

  // Accumulate statements in this context
  virtual void AddStmt(const StmtPtr& stmt) = 0;
  [[nodiscard]] const std::vector<StmtPtr>& GetStmts() const { return stmts_; }

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
  FunctionContext(std::string name, Span span, FunctionType func_type = FunctionType::Opaque,
                  std::optional<Level> level = std::nullopt, std::optional<Role> role = std::nullopt,
                  std::vector<std::pair<std::string, std::any>> attrs = {})
      : BuildContext(Type::FUNCTION, std::move(span)),
        name_(std::move(name)),
        func_type_(func_type),
        level_(level),
        role_(role),
        attrs_(std::move(attrs)) {}

  void AddParam(const VarPtr& param, ParamDirection direction = ParamDirection::In) {
    params_.push_back(param);
    param_directions_.push_back(direction);
  }
  void AddReturnType(const TypePtr& type) { return_types_.push_back(type); }

  void AddStmt(const StmtPtr& stmt) override { stmts_.push_back(stmt); }
  [[nodiscard]] const std::string& GetName() const { return name_; }
  [[nodiscard]] const std::vector<VarPtr>& GetParams() const { return params_; }
  [[nodiscard]] const std::vector<ParamDirection>& GetParamDirections() const { return param_directions_; }
  [[nodiscard]] const std::vector<TypePtr>& GetReturnTypes() const { return return_types_; }
  [[nodiscard]] FunctionType GetFuncType() const { return func_type_; }
  [[nodiscard]] std::optional<Level> GetLevel() const { return level_; }
  [[nodiscard]] std::optional<Role> GetRole() const { return role_; }
  [[nodiscard]] const std::vector<std::pair<std::string, std::any>>& GetAttrs() const { return attrs_; }

 private:
  std::string name_;
  FunctionType func_type_;
  std::optional<Level> level_;
  std::optional<Role> role_;
  std::vector<std::pair<std::string, std::any>> attrs_;
  std::vector<VarPtr> params_;
  std::vector<ParamDirection> param_directions_;
  std::vector<TypePtr> return_types_;
};

/**
 * @brief Context for building a for loop
 */
class ForLoopContext : public BuildContext {
 public:
  ForLoopContext(VarPtr loop_var, ExprPtr start, ExprPtr stop, ExprPtr step, Span span,
                 ForKind kind = ForKind::Sequential, std::optional<ChunkConfig> chunk_config = std::nullopt,
                 std::vector<std::pair<std::string, std::any>> attrs = {})
      : BuildContext(Type::FOR_LOOP, std::move(span)),
        loop_var_(std::move(loop_var)),
        start_(std::move(start)),
        stop_(std::move(stop)),
        step_(std::move(step)),
        kind_(kind),
        chunk_config_(std::move(chunk_config)),
        attrs_(std::move(attrs)) {}

  void AddIterArg(const IterArgPtr& iter_arg) { iter_args_.push_back(iter_arg); }
  void AddReturnVar(const VarPtr& var) { return_vars_.push_back(var); }

  void AddStmt(const StmtPtr& stmt) override { stmts_.push_back(stmt); }
  [[nodiscard]] const VarPtr& GetLoopVar() const { return loop_var_; }
  [[nodiscard]] const ExprPtr& GetStart() const { return start_; }
  [[nodiscard]] const ExprPtr& GetStop() const { return stop_; }
  [[nodiscard]] const ExprPtr& GetStep() const { return step_; }
  [[nodiscard]] const std::vector<IterArgPtr>& GetIterArgs() const { return iter_args_; }
  [[nodiscard]] const std::vector<VarPtr>& GetReturnVars() const { return return_vars_; }
  [[nodiscard]] ForKind GetKind() const { return kind_; }
  [[nodiscard]] const std::optional<ChunkConfig>& GetChunkConfig() const { return chunk_config_; }
  [[nodiscard]] const std::vector<std::pair<std::string, std::any>>& GetAttrs() const { return attrs_; }

 private:
  VarPtr loop_var_;
  ExprPtr start_;
  ExprPtr stop_;
  ExprPtr step_;
  ForKind kind_;
  std::optional<ChunkConfig> chunk_config_;
  std::vector<std::pair<std::string, std::any>> attrs_;
  std::vector<IterArgPtr> iter_args_;
  std::vector<VarPtr> return_vars_;
};

/**
 * @brief Context for building a while loop
 */
class WhileLoopContext : public BuildContext {
 public:
  WhileLoopContext(ExprPtr condition, Span span)
      : BuildContext(Type::WHILE_LOOP, std::move(span)), condition_(std::move(condition)) {}

  void AddIterArg(const IterArgPtr& iter_arg) { iter_args_.push_back(iter_arg); }
  void AddReturnVar(const VarPtr& var) { return_vars_.push_back(var); }
  void SetCondition(const ExprPtr& condition) { condition_ = condition; }

  void AddStmt(const StmtPtr& stmt) override { stmts_.push_back(stmt); }
  [[nodiscard]] const ExprPtr& GetCondition() const { return condition_; }
  [[nodiscard]] const std::vector<IterArgPtr>& GetIterArgs() const { return iter_args_; }
  [[nodiscard]] const std::vector<VarPtr>& GetReturnVars() const { return return_vars_; }

 private:
  ExprPtr condition_;
  std::vector<IterArgPtr> iter_args_;
  std::vector<VarPtr> return_vars_;
};

/**
 * @brief Context for building an if statement
 */
class IfStmtContext : public BuildContext {
 public:
  IfStmtContext(ExprPtr condition, Span span)
      : BuildContext(Type::IF_STMT, std::move(span)), condition_(std::move(condition)) {}

  void BeginElseBranch() {
    in_else_branch_ = true;
    else_stmts_.clear();
  }

  void AddReturnVar(const VarPtr& var) { return_vars_.push_back(var); }

  void AddStmt(const StmtPtr& stmt) override { (in_else_branch_ ? else_stmts_ : stmts_).push_back(stmt); }
  [[nodiscard]] const ExprPtr& GetCondition() const { return condition_; }
  [[nodiscard]] bool InElseBranch() const { return in_else_branch_; }
  [[nodiscard]] const std::vector<StmtPtr>& GetElseStmts() const { return else_stmts_; }
  [[nodiscard]] const std::vector<VarPtr>& GetReturnVars() const { return return_vars_; }

 private:
  ExprPtr condition_;
  bool in_else_branch_ = false;
  std::vector<StmtPtr> else_stmts_;
  std::vector<VarPtr> return_vars_;
};

/**
 * @brief Context for building a scope statement
 */
class ScopeContext : public BuildContext {
 public:
  ScopeContext(ScopeKind scope_kind, Span span, std::optional<Level> level = std::nullopt,
               std::optional<Role> role = std::nullopt, std::optional<SplitMode> split = std::nullopt,
               std::string name_hint = "", ExprPtr core_num = nullptr,
               std::optional<bool> sync_start = std::nullopt, std::optional<bool> manual = std::nullopt)
      : BuildContext(Type::SCOPE, std::move(span)),
        scope_kind_(scope_kind),
        level_(level),
        role_(role),
        split_(split),
        name_hint_(std::move(name_hint)),
        core_num_(std::move(core_num)),
        sync_start_(sync_start),
        manual_(manual) {}

  void AddStmt(const StmtPtr& stmt) override { stmts_.push_back(stmt); }

  [[nodiscard]] ScopeKind GetScopeKind() const { return scope_kind_; }
  [[nodiscard]] std::optional<Level> GetLevel() const { return level_; }
  [[nodiscard]] std::optional<Role> GetRole() const { return role_; }
  [[nodiscard]] std::optional<SplitMode> GetSplit() const { return split_; }
  [[nodiscard]] const std::string& GetNameHint() const { return name_hint_; }
  [[nodiscard]] const ExprPtr& GetCoreNum() const { return core_num_; }
  [[nodiscard]] std::optional<bool> GetSyncStart() const { return sync_start_; }
  [[nodiscard]] std::optional<bool> GetManual() const { return manual_; }
  [[nodiscard]] const std::vector<StmtPtr>& GetStmts() const { return stmts_; }

 private:
  ScopeKind scope_kind_;
  std::optional<Level> level_;
  std::optional<Role> role_;
  std::optional<SplitMode> split_;
  std::string name_hint_;
  ExprPtr core_num_;
  std::optional<bool> sync_start_;
  std::optional<bool> manual_;
  std::vector<StmtPtr> stmts_;
};

/**
 * @brief Context for building a program
 */
class ProgramContext : public BuildContext {
 public:
  ProgramContext(std::string name, Span span)
      : BuildContext(Type::PROGRAM, std::move(span)), name_(std::move(name)) {}

  /**
   * @brief Declare a function and get its GlobalVar
   *
   * @param func_name Function name to declare
   * @return GlobalVar for the function
   */
  GlobalVarPtr DeclareFunction(const std::string& func_name);

  /**
   * @brief Get a GlobalVar for a declared function
   *
   * @param func_name Function name
   * @return GlobalVar for the function, or nullptr if not found
   */
  [[nodiscard]] GlobalVarPtr GetGlobalVar(const std::string& func_name) const;

  /**
   * @brief Add a function to the program
   *
   * @param func Function to add
   */
  void AddFunction(const FunctionPtr& func);

  /**
   * @brief Get the program name
   *
   * @return Program name
   */
  [[nodiscard]] const std::string& GetName() const { return name_; }

  /**
   * @brief Get all functions in the program
   *
   * @return Vector of functions
   */
  [[nodiscard]] const std::vector<FunctionPtr>& GetFunctions() const { return functions_; }

  /**
   * @brief Get all GlobalVars in the program
   *
   * @return Map of function names to GlobalVars
   */
  [[nodiscard]] const std::map<std::string, GlobalVarPtr>& GetGlobalVars() const { return global_vars_; }

  /**
   * @brief Get return types for a function by its GlobalVar
   *
   * @param gvar GlobalVar for the function
   * @return Vector of return types, or empty vector if function not yet added
   */
  [[nodiscard]] std::vector<TypePtr> GetReturnTypes(const GlobalVarPtr& gvar) const;

  // ProgramContext doesn't accumulate statements
  void AddStmt(const StmtPtr& stmt) override {
    throw pypto::InternalError("Cannot add statements directly to program context");
  }

 private:
  std::string name_;
  std::vector<FunctionPtr> functions_;
  std::map<std::string, GlobalVarPtr> global_vars_;           // Track GlobalVars for cross-function calls
  std::map<std::string, std::vector<TypePtr>> return_types_;  // Track return types for each function
};

}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_BUILDER_H_
