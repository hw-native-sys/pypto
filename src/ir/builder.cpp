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

#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "pypto/core/error.h"

namespace pypto {
namespace ir {

// ========== IRBuilder Implementation ==========

IRBuilder::IRBuilder() {}

IRBuilder::~IRBuilder() {}

// ========== Function Building ==========

void IRBuilder::BeginFunction(const std::string& name, const Span& span) {
  if (InFunction()) {
    throw pypto::RuntimeError("Cannot begin function '" + name + "': already inside function '" +
                              static_cast<FunctionContext*>(CurrentContext())->GetName() + "' at " +
                              CurrentContext()->GetBeginSpan().to_string());
  }

  context_stack_.push_back(std::make_unique<FunctionContext>(name, span));
}

VarPtr IRBuilder::FuncArg(const std::string& name, const TypePtr& type, const Span& span) {
  ValidateInFunction("FuncArg");

  auto var = std::make_shared<ir::Var>(name, type, span);
  static_cast<FunctionContext*>(CurrentContext())->AddParam(var);
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
  StmtPtr body;
  const auto& stmts = func_ctx->GetStmts();
  if (stmts.empty()) {
    // Empty body - create empty SeqStmts
    body = std::make_shared<SeqStmts>(std::vector<StmtPtr>(), end_span);
  } else if (stmts.size() == 1) {
    body = stmts[0];
  } else {
    body = std::make_shared<SeqStmts>(stmts, end_span);
  }

  // Combine begin and end spans
  const Span& begin_span = func_ctx->GetBeginSpan();
  Span combined_span(begin_span.filename_, begin_span.begin_line_, begin_span.begin_column_,
                     end_span.begin_line_, end_span.begin_column_);

  // Create function
  auto func = std::make_shared<Function>(func_ctx->GetName(), func_ctx->GetParams(),
                                         func_ctx->GetReturnTypes(), body, combined_span);

  // Pop context
  context_stack_.pop_back();

  return func;
}

// ========== For Loop Building ==========

void IRBuilder::BeginForLoop(const VarPtr& loop_var, const ExprPtr& start, const ExprPtr& stop,
                             const ExprPtr& step, const Span& span) {
  if (context_stack_.empty()) {
    throw pypto::RuntimeError("Cannot begin for loop: not inside a function or another valid context at " +
                              span.to_string());
  }

  context_stack_.push_back(std::make_unique<ForLoopContext>(loop_var, start, stop, step, span));
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
  StmtPtr body;
  const auto& stmts = loop_ctx->GetStmts();
  if (stmts.empty()) {
    body = std::make_shared<SeqStmts>(std::vector<StmtPtr>(), end_span);
  } else if (stmts.size() == 1) {
    body = stmts[0];
  } else {
    body = std::make_shared<SeqStmts>(stmts, end_span);
  }

  // Combine begin and end spans
  const Span& begin_span = loop_ctx->GetBeginSpan();
  Span combined_span(begin_span.filename_, begin_span.begin_line_, begin_span.begin_column_,
                     end_span.begin_line_, end_span.begin_column_);

  // Create for statement
  auto for_stmt = std::make_shared<ForStmt>(loop_ctx->GetLoopVar(), loop_ctx->GetStart(), loop_ctx->GetStop(),
                                            loop_ctx->GetStep(), loop_ctx->GetIterArgs(), body,
                                            loop_ctx->GetReturnVars(), combined_span);

  // Pop context
  context_stack_.pop_back();

  // Emit to parent context if it exists
  if (!context_stack_.empty()) {
    CurrentContext()->AddStmt(for_stmt);
  }

  return for_stmt;
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
  StmtPtr then_body;
  const auto& then_stmts = if_ctx->GetStmts();
  if (then_stmts.empty()) {
    then_body = std::make_shared<SeqStmts>(std::vector<StmtPtr>(), end_span);
  } else if (then_stmts.size() == 1) {
    then_body = then_stmts[0];
  } else {
    then_body = std::make_shared<SeqStmts>(then_stmts, end_span);
  }

  // Build else body (optional)
  std::optional<StmtPtr> else_body;
  if (if_ctx->InElseBranch()) {
    const auto& else_stmts = if_ctx->GetElseStmts();
    if (!else_stmts.empty()) {
      if (else_stmts.size() == 1) {
        else_body = else_stmts[0];
      } else {
        else_body = std::make_shared<SeqStmts>(else_stmts, end_span);
      }
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
    CurrentContext()->AddStmt(if_stmt);
  }

  return if_stmt;
}

// ========== Statement Recording ==========

void IRBuilder::Emit(const StmtPtr& stmt) {
  if (context_stack_.empty()) {
    throw pypto::RuntimeError("Cannot emit statement: not inside any context");
  }

  auto* ctx = CurrentContext();

  // If in if statement and in else branch, add to else statements
  if (ctx->GetType() == BuildContext::Type::IF_STMT) {
    auto* if_ctx = static_cast<IfStmtContext*>(ctx);
    if (if_ctx->InElseBranch()) {
      if_ctx->AddElseStmt(stmt);
      return;
    }
  }

  // Otherwise add to main statement list
  ctx->AddStmt(stmt);
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

}  // namespace ir
}  // namespace pypto
