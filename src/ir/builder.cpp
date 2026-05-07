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

#include "pypto/ir/builder.h"

#include <any>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/error.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/program.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

// ========== IRBuilder Implementation ==========

IRBuilder::IRBuilder() = default;

// ========== Function Building ==========

void IRBuilder::BeginFunction(const std::string& name, const Span& span, FunctionType type,
                              std::optional<Level> level, std::optional<Role> role,
                              std::vector<std::pair<std::string, std::any>> attrs) {
  if (InFunction()) {
    throw pypto::RuntimeError("Cannot begin function '" + name + "': already inside function '" +
                              static_cast<FunctionContext*>(CurrentContext())->GetName() + "' at " +
                              CurrentContext()->GetBeginSpan().to_string());
  }

  context_stack_.push_back(
      std::make_unique<FunctionContext>(name, span, type, level, role, std::move(attrs)));
}

VarPtr IRBuilder::FuncArg(const std::string& name, const TypePtr& type, const Span& span,
                          ParamDirection direction) {
  ValidateInFunction("FuncArg");

  auto var = std::make_shared<ir::Var>(name, type, span);
  static_cast<FunctionContext*>(CurrentContext())->AddParam(var, direction);
  return var;
}

void IRBuilder::ReturnType(const TypePtr& type) {
  ValidateInFunction("ReturnType");
  static_cast<FunctionContext*>(CurrentContext())->AddReturnType(type);
}

FunctionPtr IRBuilder::EndFunction(const Span& end_span) {
  ValidateInFunction("EndFunction");

  auto* func_ctx = static_cast<FunctionContext*>(CurrentContext());

  // Build body from accumulated statements
  const auto& stmts = func_ctx->GetStmts();
  StmtPtr body = (stmts.size() == 1) ? stmts[0] : std::make_shared<SeqStmts>(stmts, end_span);

  // Combine begin and end spans
  const Span& begin_span = func_ctx->GetBeginSpan();
  Span combined_span(begin_span.filename_, begin_span.begin_line_, begin_span.begin_column_,
                     end_span.begin_line_, end_span.begin_column_);

  // Create function
  auto func =
      std::make_shared<Function>(func_ctx->GetName(), func_ctx->GetParams(), func_ctx->GetParamDirections(),
                                 func_ctx->GetReturnTypes(), body, combined_span, func_ctx->GetFuncType(),
                                 func_ctx->GetLevel(), func_ctx->GetRole(), func_ctx->GetAttrs());

  // Pop context
  context_stack_.pop_back();

  return func;
}

// ========== For Loop Building ==========

void IRBuilder::BeginForLoop(const VarPtr& loop_var, const ExprPtr& start, const ExprPtr& stop,
                             const ExprPtr& step, const Span& span, ForKind kind,
                             std::optional<ChunkConfig> chunk_config,
                             std::vector<std::pair<std::string, std::any>> attrs) {
  if (context_stack_.empty()) {
    throw pypto::RuntimeError("Cannot begin for loop: not inside a function or another valid context at " +
                              span.to_string());
  }

  context_stack_.push_back(std::make_unique<ForLoopContext>(loop_var, start, stop, step, span, kind,
                                                            std::move(chunk_config), std::move(attrs)));
}

void IRBuilder::AddIterArg(const IterArgPtr& iter_arg) {
  ValidateInLoop("AddIterArg");
  static_cast<ForLoopContext*>(CurrentContext())->AddIterArg(iter_arg);
}

void IRBuilder::AddReturnVar(const VarPtr& var) {
  ValidateInLoop("AddReturnVar");
  static_cast<ForLoopContext*>(CurrentContext())->AddReturnVar(var);
}

StmtPtr IRBuilder::EndForLoop(const Span& end_span) {
  ValidateInLoop("EndForLoop");

  auto* loop_ctx = static_cast<ForLoopContext*>(CurrentContext());

  // Validate iter_args and return_vars match
  if (loop_ctx->GetIterArgs().size() != loop_ctx->GetReturnVars().size()) {
    // Pop context before throwing to maintain stack consistency
    context_stack_.pop_back();

    std::ostringstream oss;
    oss << "For loop has " << loop_ctx->GetIterArgs().size() << " iteration arguments but "
        << loop_ctx->GetReturnVars().size() << " return variables. They must match.";
    throw pypto::RuntimeError(oss.str());
  }

  // Build body from accumulated statements
  const auto& stmts = loop_ctx->GetStmts();
  StmtPtr body = (stmts.size() == 1) ? stmts[0] : std::make_shared<SeqStmts>(stmts, end_span);

  // Combine begin and end spans
  const Span& begin_span = loop_ctx->GetBeginSpan();
  Span combined_span(begin_span.filename_, begin_span.begin_line_, begin_span.begin_column_,
                     end_span.begin_line_, end_span.begin_column_);

  // Create for statement
  auto for_stmt = std::make_shared<ForStmt>(loop_ctx->GetLoopVar(), loop_ctx->GetStart(), loop_ctx->GetStop(),
                                            loop_ctx->GetStep(), loop_ctx->GetIterArgs(), body,
                                            loop_ctx->GetReturnVars(), combined_span, loop_ctx->GetKind(),
                                            loop_ctx->GetChunkConfig(), loop_ctx->GetAttrs());

  // Pop context
  context_stack_.pop_back();

  // Emit to parent context if it exists
  if (!context_stack_.empty()) {
    ApplyPendingLeadingComments(for_stmt);
    CurrentContext()->AddStmt(for_stmt);
  }

  return for_stmt;
}

// ========== While Loop Building ==========

void IRBuilder::BeginWhileLoop(const ExprPtr& condition, const Span& span) {
  if (context_stack_.empty()) {
    throw pypto::RuntimeError("Cannot begin while loop: not inside a function or another valid context at " +
                              span.to_string());
  }

  context_stack_.push_back(std::make_unique<WhileLoopContext>(condition, span));
}

void IRBuilder::AddWhileIterArg(const IterArgPtr& iter_arg) {
  ValidateInWhileLoop("AddWhileIterArg");
  static_cast<WhileLoopContext*>(CurrentContext())->AddIterArg(iter_arg);
}

void IRBuilder::AddWhileReturnVar(const VarPtr& var) {
  ValidateInWhileLoop("AddWhileReturnVar");
  static_cast<WhileLoopContext*>(CurrentContext())->AddReturnVar(var);
}

void IRBuilder::SetWhileLoopCondition(const ExprPtr& condition) {
  ValidateInWhileLoop("SetWhileLoopCondition");
  static_cast<WhileLoopContext*>(CurrentContext())->SetCondition(condition);
}

StmtPtr IRBuilder::EndWhileLoop(const Span& end_span) {
  ValidateInWhileLoop("EndWhileLoop");

  auto* loop_ctx = static_cast<WhileLoopContext*>(CurrentContext());

  // Validate iter_args and return_vars match
  if (loop_ctx->GetIterArgs().size() != loop_ctx->GetReturnVars().size()) {
    // Pop context before throwing to maintain stack consistency
    context_stack_.pop_back();

    std::ostringstream oss;
    oss << "While loop has " << loop_ctx->GetIterArgs().size() << " iteration arguments but "
        << loop_ctx->GetReturnVars().size() << " return variables. They must match.";
    throw pypto::RuntimeError(oss.str());
  }

  // Build body from accumulated statements
  const auto& stmts = loop_ctx->GetStmts();
  StmtPtr body = (stmts.size() == 1) ? stmts[0] : std::make_shared<SeqStmts>(stmts, end_span);

  // Combine begin and end spans
  const Span& begin_span = loop_ctx->GetBeginSpan();
  Span combined_span(begin_span.filename_, begin_span.begin_line_, begin_span.begin_column_,
                     end_span.begin_line_, end_span.begin_column_);

  // Create while statement
  auto while_stmt = std::make_shared<WhileStmt>(loop_ctx->GetCondition(), loop_ctx->GetIterArgs(), body,
                                                loop_ctx->GetReturnVars(), combined_span);

  // Pop context
  context_stack_.pop_back();

  // Emit to parent context if it exists
  if (!context_stack_.empty()) {
    ApplyPendingLeadingComments(while_stmt);
    CurrentContext()->AddStmt(while_stmt);
  }

  return while_stmt;
}

// ========== If Statement Building ==========

void IRBuilder::BeginIf(const ExprPtr& condition, const Span& span) {
  CHECK(!context_stack_.empty())
      << "Cannot begin if statement: not inside a function or another valid context at " << span.to_string();
  context_stack_.push_back(std::make_unique<IfStmtContext>(condition, span));
}

void IRBuilder::BeginElse(const Span& span) {
  ValidateInIf("BeginElse");

  auto* if_ctx = static_cast<IfStmtContext*>(CurrentContext());
  CHECK(!if_ctx->InElseBranch()) << "Cannot begin else branch: already in else branch at "
                                 << span.to_string();

  if_ctx->BeginElseBranch();
}

void IRBuilder::AddIfReturnVar(const VarPtr& var) {
  ValidateInIf("AddIfReturnVar");
  static_cast<IfStmtContext*>(CurrentContext())->AddReturnVar(var);
}

StmtPtr IRBuilder::EndIf(const Span& end_span) {
  ValidateInIf("EndIf");

  auto* if_ctx = static_cast<IfStmtContext*>(CurrentContext());

  // Build then body
  const auto& then_stmts = if_ctx->GetStmts();
  StmtPtr then_body =
      (then_stmts.size() == 1) ? then_stmts[0] : std::make_shared<SeqStmts>(then_stmts, end_span);

  // Build else body (optional)
  std::optional<StmtPtr> else_body;
  if (if_ctx->InElseBranch()) {
    const auto& else_stmts = if_ctx->GetElseStmts();
    if (!else_stmts.empty()) {
      else_body = (else_stmts.size() == 1) ? else_stmts[0] : std::make_shared<SeqStmts>(else_stmts, end_span);
    }
  }

  // Combine begin and end spans
  const Span& begin_span = if_ctx->GetBeginSpan();
  Span combined_span(begin_span.filename_, begin_span.begin_line_, begin_span.begin_column_,
                     end_span.begin_line_, end_span.begin_column_);

  // Create if statement
  auto if_stmt = std::make_shared<IfStmt>(if_ctx->GetCondition(), then_body, else_body,
                                          if_ctx->GetReturnVars(), combined_span);

  // Pop context
  context_stack_.pop_back();

  // Emit to parent context if it exists
  if (!context_stack_.empty()) {
    ApplyPendingLeadingComments(if_stmt);
    CurrentContext()->AddStmt(if_stmt);
  }

  return if_stmt;
}

// ========== Scope Building ==========

void IRBuilder::BeginScope(ScopeKind scope_kind, const Span& span, std::optional<Level> level,
                           std::optional<Role> role, std::optional<SplitMode> split, std::string name_hint,
                           ExprPtr core_num, std::optional<bool> sync_start, std::optional<bool> manual) {
  CHECK(!context_stack_.empty()) << "Cannot begin scope: not inside a function or another valid context at "
                                 << span.to_string();
  CHECK(scope_kind != ScopeKind::Hierarchy || level.has_value())
      << "Hierarchy scope requires a level at " << span.to_string();
  CHECK(scope_kind != ScopeKind::Runtime || manual.has_value())
      << "Runtime scope requires manual flag at " << span.to_string();
  context_stack_.push_back(std::make_unique<ScopeContext>(
      scope_kind, span, level, role, split, std::move(name_hint), std::move(core_num), sync_start, manual));
}

StmtPtr IRBuilder::EndScope(const Span& end_span) {
  CHECK(!context_stack_.empty() && CurrentContext()->GetType() == BuildContext::Type::SCOPE)
      << "Cannot end scope: not inside a scope context at " << end_span.to_string();

  auto* scope_ctx = static_cast<ScopeContext*>(CurrentContext());

  // Build body
  const auto& stmts = scope_ctx->GetStmts();
  StmtPtr body = (stmts.size() == 1) ? stmts[0] : std::make_shared<SeqStmts>(stmts, end_span);

  // Combine spans
  const Span& begin_span = scope_ctx->GetBeginSpan();
  Span combined_span(begin_span.filename_, begin_span.begin_line_, begin_span.begin_column_,
                     end_span.begin_line_, end_span.begin_column_);

  auto scope_kind = scope_ctx->GetScopeKind();
  auto level = scope_ctx->GetLevel();
  auto role = scope_ctx->GetRole();
  auto split = scope_ctx->GetSplit();
  auto name_hint = scope_ctx->GetNameHint();
  auto core_num = scope_ctx->GetCoreNum();
  auto sync_start = scope_ctx->GetSyncStart();
  auto manual = scope_ctx->GetManual();

  // Create scope statement before popping context so that if construction throws
  // (e.g. validation CHECK fails) the builder state stays consistent.
  // Dispatch on scope_kind to the matching derived class (issue #1047).
  ScopeStmtPtr scope_stmt;
  switch (scope_kind) {
    case ScopeKind::InCore:
      scope_stmt = std::make_shared<const InCoreScopeStmt>(split, std::move(name_hint), body, combined_span);
      break;
    case ScopeKind::AutoInCore:
      scope_stmt =
          std::make_shared<const AutoInCoreScopeStmt>(split, std::move(name_hint), body, combined_span);
      break;
    case ScopeKind::Cluster:
      scope_stmt = std::make_shared<const ClusterScopeStmt>(std::move(name_hint), body, combined_span);
      break;
    case ScopeKind::Hierarchy:
      CHECK(level.has_value()) << "Hierarchy scope requires a level";
      scope_stmt =
          std::make_shared<const HierarchyScopeStmt>(*level, role, std::move(name_hint), body, combined_span);
      break;
    case ScopeKind::Spmd:
      CHECK(core_num != nullptr) << "Spmd scope requires core_num";
      scope_stmt = std::make_shared<const SpmdScopeStmt>(core_num, sync_start.value_or(false),
                                                         std::move(name_hint), body, combined_span);
      break;
    case ScopeKind::Runtime:
      CHECK(manual.has_value()) << "Runtime scope requires manual flag";
      scope_stmt =
          std::make_shared<const RuntimeScopeStmt>(*manual, std::move(name_hint), body, combined_span);
      break;
  }
  // Safety net: every ScopeKind value above must populate scope_stmt. The switch has
  // no default so adding a new ScopeKind without a case here will trip -Wswitch-enum;
  // this assertion guards the runtime path in case someone bypasses that warning.
  INTERNAL_CHECK(scope_stmt != nullptr) << "Unhandled ScopeKind in EndScope";
  context_stack_.pop_back();

  // Emit to parent context if it exists
  if (!context_stack_.empty()) {
    ApplyPendingLeadingComments(scope_stmt);
    CurrentContext()->AddStmt(scope_stmt);
  }

  return scope_stmt;
}

// ========== Program Building ==========
void IRBuilder::BeginProgram(const std::string& name, const Span& span) {
  if (InProgram()) {
    throw pypto::RuntimeError("Cannot begin program '" + name + "': already inside program '" +
                              static_cast<ProgramContext*>(CurrentContext())->GetName() + "' at " +
                              CurrentContext()->GetBeginSpan().to_string());
  }

  context_stack_.push_back(std::make_unique<ProgramContext>(name, span));
}

GlobalVarPtr IRBuilder::DeclareFunction(const std::string& func_name) {
  ValidateInProgram("DeclareFunction");
  return static_cast<ProgramContext*>(CurrentContext())->DeclareFunction(func_name);
}

GlobalVarPtr IRBuilder::GetGlobalVar(const std::string& func_name) {
  ValidateInProgram("GetGlobalVar");
  auto gvar = static_cast<ProgramContext*>(CurrentContext())->GetGlobalVar(func_name);
  if (!gvar) {
    throw pypto::RuntimeError("Function '" + func_name + "' not declared in current program");
  }
  return gvar;
}

void IRBuilder::AddFunction(const FunctionPtr& func) {
  ValidateInProgram("AddFunction");
  static_cast<ProgramContext*>(CurrentContext())->AddFunction(func);
}

ProgramPtr IRBuilder::EndProgram(const Span& end_span) {
  ValidateInProgram("EndProgram");

  auto* prog_ctx = static_cast<ProgramContext*>(CurrentContext());

  // Combine begin and end spans
  const Span& begin_span = prog_ctx->GetBeginSpan();
  Span combined_span(begin_span.filename_, begin_span.begin_line_, begin_span.begin_column_,
                     end_span.begin_line_, end_span.begin_column_);

  // Create program from functions vector
  auto program = std::make_shared<Program>(prog_ctx->GetFunctions(), prog_ctx->GetName(), combined_span);

  // Pop context
  context_stack_.pop_back();

  return program;
}

bool IRBuilder::InProgram() const {
  for (const auto& ctx : context_stack_) {
    if (ctx->GetType() == BuildContext::Type::PROGRAM) {
      return true;
    }
  }
  return false;
}

std::vector<TypePtr> IRBuilder::GetFunctionReturnTypes(const GlobalVarPtr& gvar) const {
  // Find the program context in the stack
  for (const auto& ctx : context_stack_) {
    if (ctx->GetType() == BuildContext::Type::PROGRAM) {
      auto* prog_ctx = static_cast<const ProgramContext*>(ctx.get());
      return prog_ctx->GetReturnTypes(gvar);
    }
  }
  return {};
}

// ========== Statement Recording ==========

void IRBuilder::Emit(const StmtPtr& stmt) {
  if (context_stack_.empty()) {
    throw pypto::RuntimeError("Cannot emit statement: not inside any context");
  }
  ApplyPendingLeadingComments(stmt);
  CurrentContext()->AddStmt(stmt);
}

void IRBuilder::ApplyPendingLeadingComments(const StmtPtr& stmt) {
  if (pending_leading_stack_.empty() || context_stack_.empty()) return;
  auto& top = pending_leading_stack_.back();
  if (top.comments.empty() || top.target != CurrentContext()) return;
  AttachLeadingComments(stmt, std::move(top.comments));
  top.comments.clear();  // entry stays on the stack; Pop clears it
}

void IRBuilder::PushPendingLeadingComments(std::vector<std::string> comments) {
  BuildContext* target = context_stack_.empty() ? nullptr : context_stack_.back().get();
  pending_leading_stack_.push_back({std::move(comments), target});
}

std::vector<std::string> IRBuilder::PopPendingLeadingComments() {
  INTERNAL_CHECK(!pending_leading_stack_.empty())
      << "PopPendingLeadingComments called without a matching Push";
  auto comments = std::move(pending_leading_stack_.back().comments);
  pending_leading_stack_.pop_back();
  return comments;
}

AssignStmtPtr IRBuilder::Assign(const VarPtr& var, const ExprPtr& value, const Span& span) {
  auto assign = std::make_shared<AssignStmt>(var, value, span);
  Emit(assign);
  return assign;
}

VarPtr IRBuilder::Var(const std::string& name, const TypePtr& type, const Span& span) {
  return std::make_shared<ir::Var>(name, type, span);
}

ReturnStmtPtr IRBuilder::Return(const std::vector<ExprPtr>& values, const Span& span) {
  auto return_stmt = std::make_shared<ReturnStmt>(values, span);
  Emit(return_stmt);
  return return_stmt;
}

ReturnStmtPtr IRBuilder::Return(const Span& span) {
  auto return_stmt = std::make_shared<ReturnStmt>(span);
  Emit(return_stmt);
  return return_stmt;
}

// ========== Context State Queries ==========

BuildContext* IRBuilder::CurrentContext() {
  if (context_stack_.empty()) {
    return nullptr;
  }
  return context_stack_.back().get();
}

bool IRBuilder::InFunction() const {
  for (const auto& ctx : context_stack_) {
    if (ctx->GetType() == BuildContext::Type::FUNCTION) {
      return true;
    }
  }
  return false;
}

bool IRBuilder::InLoop() const {
  if (context_stack_.empty()) {
    return false;
  }
  return context_stack_.back()->GetType() == BuildContext::Type::FOR_LOOP;
}

bool IRBuilder::InIf() const {
  if (context_stack_.empty()) {
    return false;
  }
  return context_stack_.back()->GetType() == BuildContext::Type::IF_STMT;
}

bool IRBuilder::InWhileLoop() const {
  if (context_stack_.empty()) {
    return false;
  }
  return context_stack_.back()->GetType() == BuildContext::Type::WHILE_LOOP;
}

// ========== Private Helpers ==========

template <typename T>
T* IRBuilder::GetCurrentContextAs() {
  auto* ctx = CurrentContext();
  if (!ctx) {
    return nullptr;
  }
  return dynamic_cast<T*>(ctx);
}

void IRBuilder::ValidateInFunction(const std::string& operation) {
  CHECK(InFunction()) << operation << " can only be called inside a function context";
  CHECK(CurrentContext()->GetType() == BuildContext::Type::FUNCTION)
      << operation << " must be called directly in function context, not nested";
}

void IRBuilder::ValidateInLoop(const std::string& operation) {
  CHECK(InLoop()) << operation << " can only be called inside a for loop context";
}

void IRBuilder::ValidateInIf(const std::string& operation) {
  CHECK(InIf()) << operation << " can only be called inside an if statement context";
}

void IRBuilder::ValidateInWhileLoop(const std::string& operation) {
  CHECK(InWhileLoop()) << operation << " can only be called inside a while loop context";
}

void IRBuilder::ValidateInProgram(const std::string& operation) {
  CHECK(InProgram()) << operation << " can only be called inside a program context";
}

// ========== ProgramContext Implementation ==========

GlobalVarPtr ProgramContext::DeclareFunction(const std::string& func_name) {
  // Check if already declared
  auto it = global_vars_.find(func_name);
  if (it != global_vars_.end()) {
    return it->second;
  }

  // Create new GlobalVar
  auto gvar = std::make_shared<GlobalVar>(func_name);
  global_vars_[func_name] = gvar;
  return gvar;
}

GlobalVarPtr ProgramContext::GetGlobalVar(const std::string& func_name) const {
  auto it = global_vars_.find(func_name);
  if (it != global_vars_.end()) {
    return it->second;
  }
  return nullptr;
}

void ProgramContext::AddFunction(const FunctionPtr& func) {
  INTERNAL_CHECK(func) << "Cannot add null function to program";

  // Verify function was declared (if not, declare it automatically)
  auto it = global_vars_.find(func->name_);
  if (it == global_vars_.end()) {
    // Function wasn't declared, declare it now for convenience
    DeclareFunction(func->name_);
  }

  // Store return types for this function
  return_types_[func->name_] = func->return_types_;

  functions_.push_back(func);
}

std::vector<TypePtr> ProgramContext::GetReturnTypes(const GlobalVarPtr& gvar) const {
  auto it = return_types_.find(gvar->name_);
  if (it != return_types_.end()) {
    return it->second;
  }
  return {};
}

}  // namespace ir
}  // namespace pypto
