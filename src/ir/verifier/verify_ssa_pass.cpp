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

#include <cstddef>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/core/error.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/program.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/printer.h"
#include "pypto/ir/type.h"
#include "pypto/ir/verifier/verification_error.h"
#include "pypto/ir/verifier/verifier.h"

namespace pypto {
namespace ir {

namespace ssa {
std::string ErrorTypeToString(ErrorType type) {
  switch (type) {
    case ErrorType::MULTIPLE_ASSIGNMENT:
      return "MULTIPLE_ASSIGNMENT";
    case ErrorType::NAME_SHADOWING:
      return "NAME_SHADOWING";
    case ErrorType::MISSING_YIELD:
      return "MISSING_YIELD";
    case ErrorType::ITER_ARGS_RETURN_VARS_MISMATCH:
      return "ITER_ARGS_RETURN_VARS_MISMATCH";
    case ErrorType::YIELD_COUNT_MISMATCH:
      return "YIELD_COUNT_MISMATCH";
    case ErrorType::SCOPE_VIOLATION:
      return "SCOPE_VIOLATION";
    case ErrorType::MISPLACED_YIELD:
      return "MISPLACED_YIELD";
    default:
      return "UNKNOWN";
  }
}
}  // namespace ssa

namespace {
/**
 * @brief Helper visitor class for SSA verification
 *
 * Traverses the IR tree and collects SSA violations including:
 * - Multiple assignments to the same Var pointer
 * - Scope violations (variable used outside defining scope)
 * - Cardinality mismatches (iter_args vs return_vars, yield counts)
 * - Missing yield statements
 */
class SSAVerifier : public IRVisitor {
 public:
  SSAVerifier(std::vector<Diagnostic>& diagnostics, std::string func_name, FunctionPtr func)
      : diagnostics_(diagnostics), func_name_(std::move(func_name)), func_(std::move(func)) {}

  void VisitVarLike_(const VarPtr& op) override;
  void VisitStmt_(const AssignStmtPtr& op) override;
  void VisitStmt_(const ForStmtPtr& op) override;
  void VisitStmt_(const WhileStmtPtr& op) override;
  void VisitStmt_(const IfStmtPtr& op) override;

  /**
   * @brief Register function parameters and their type-embedded vars in the outermost scope
   *
   * This includes dynamic shape variables referenced in parameter TensorType shapes
   * (e.g., pl.Tensor[[M, N], pl.FP32] where M, N are pl.dynamic vars).
   */
  void RegisterParams(const std::vector<VarPtr>& params) {
    for (const auto& param : params) {
      if (param) {
        DefineVar(param);
        // Also register any Var references embedded in the parameter type
        // (e.g., dynamic shape variables in TensorType shapes)
        RegisterTypeVars(param->GetType());
      }
    }
  }

  /**
   * @brief Register Var references from return types (dynamic shape vars)
   */
  void RegisterReturnTypeVars(const std::vector<TypePtr>& types) {
    for (const auto& type : types) {
      RegisterTypeVars(type);
    }
  }

  [[nodiscard]] const std::vector<Diagnostic>& GetDiagnostics() const { return diagnostics_; }

 private:
  std::vector<Diagnostic>& diagnostics_;
  std::string func_name_;
  FunctionPtr func_;
  mutable std::string cached_func_str_;
  std::unordered_map<const Var*, int> var_assignment_count_;

  /// Scope stack: each entry is the set of Var pointers defined in that scope
  std::vector<std::unordered_set<const Var*>> scope_stack_ = {{}};

  /**
   * @brief Register Var references found in a type (e.g., dynamic shape vars in TensorType)
   */
  void RegisterTypeVars(const TypePtr& type) {
    if (!type) return;
    if (auto tensor_type = As<TensorType>(type)) {
      for (const auto& dim : tensor_type->shape_) {
        if (auto var = As<Var>(dim)) {
          DefineVar(var);
        }
      }
    }
  }

  /**
   * @brief Define a variable in the current (innermost) scope
   */
  void DefineVar(const VarPtr& var) {
    if (!var || scope_stack_.empty()) return;
    scope_stack_.back().insert(var.get());
  }

  /**
   * @brief Check if a variable is visible in the current scope stack
   */
  bool IsVarInScope(const Var* var_ptr) const {
    for (auto it = scope_stack_.rbegin(); it != scope_stack_.rend(); ++it) {
      if (it->count(var_ptr)) return true;
    }
    return false;
  }

  /**
   * @brief Push a new scope
   */
  void EnterScope() { scope_stack_.emplace_back(); }

  /**
   * @brief Pop the current scope
   */
  void ExitScope() {
    if (scope_stack_.size() > 1) {
      scope_stack_.pop_back();
    }
  }

  /**
   * @brief Check if a variable has been assigned multiple times
   */
  void CheckVariableAssignment(const VarPtr& var);

  /**
   * @brief Record an error
   */
  void RecordError(ssa::ErrorType type, const std::string& message, const Span& span);

  /**
   * @brief Get the last statement in a statement block (recursive for SeqStmts)
   */
  StmtPtr GetLastStmt(const StmtPtr& stmt);

  /**
   * @brief Record a MISPLACED_YIELD if any YieldStmt appears in `body` other
   * than as the trailing statement. The diagnostic span points at the
   * offending YieldStmt itself. Caller is responsible for the trailing check;
   * this helper covers the "no mid-body yield" half.
   */
  void CheckNoMidBodyYield(const std::string& scope_kind, const StmtPtr& body);

  /**
   * @brief Verify iter_args/return_vars cardinality and yield constraints for a loop statement.
   *
   * Shared logic for ForStmt and WhileStmt verification.
   */
  void VerifyLoopIterArgsAndYield(const std::string& stmt_kind, size_t iter_args_size,
                                  size_t return_vars_size, const StmtPtr& body, const Span& span);

  /**
   * @brief Verify IfStmt specific constraints
   */
  void VerifyIfStmt(const IfStmtPtr& if_stmt);
};

void SSAVerifier::VisitVarLike_(const VarPtr& op) {
  if (!op) return;
  // Check that the variable is visible in the current scope
  if (!IsVarInScope(op.get())) {
    std::ostringstream msg;
    msg << "Variable '" << op->name_hint_ << "' used outside its defining scope";
    RecordError(ssa::ErrorType::SCOPE_VIOLATION, msg.str(), op->span_);
  }
  // Call base implementation to visit type shape expressions
  IRVisitor::VisitVarLike_(op);
}

void SSAVerifier::CheckVariableAssignment(const VarPtr& var) {
  if (!var) return;

  const Var* key = var.get();
  var_assignment_count_[key]++;

  if (var_assignment_count_[key] > 1) {
    std::ostringstream msg;
    msg << "Variable '" << var->name_hint_ << "' is assigned more than once (" << var_assignment_count_[key]
        << " times), violating SSA form";
    RecordError(ssa::ErrorType::MULTIPLE_ASSIGNMENT, msg.str(), var->span_);
  }
}

void SSAVerifier::RecordError(ssa::ErrorType type, const std::string& message, const Span& span) {
  std::ostringstream full_msg;
  full_msg << message << "\n  In function '" << func_name_ << "'";
  if (func_) {
    if (cached_func_str_.empty()) {
      cached_func_str_ = PythonPrint(func_);
    }
    full_msg << ":\n" << cached_func_str_;
  }
  diagnostics_.emplace_back(DiagnosticSeverity::Error, "SSAVerify", static_cast<int>(type), full_msg.str(),
                            span);
}

StmtPtr SSAVerifier::GetLastStmt(const StmtPtr& stmt) {
  if (!stmt) return nullptr;

  // If it's a SeqStmts, recursively get the last statement
  if (auto seq = As<SeqStmts>(stmt)) {
    if (!seq->stmts_.empty()) {
      return GetLastStmt(seq->stmts_.back());
    }
  }

  return stmt;
}

void SSAVerifier::CheckNoMidBodyYield(const std::string& scope_kind, const StmtPtr& body) {
  auto seq = As<SeqStmts>(body);
  if (!seq) return;
  for (size_t i = 0; i + 1 < seq->stmts_.size(); ++i) {
    if (As<YieldStmt>(seq->stmts_[i])) {
      RecordError(ssa::ErrorType::MISPLACED_YIELD,
                  scope_kind +
                      " body has YieldStmt before the terminating position; "
                      "YieldStmt must be the last statement in its scope",
                  seq->stmts_[i]->span_);
      return;
    }
  }
}

void SSAVerifier::VerifyLoopIterArgsAndYield(const std::string& stmt_kind, size_t iter_args_size,
                                             size_t return_vars_size, const StmtPtr& body, const Span& span) {
  // Cardinality check: iter_args.size() == return_vars.size()
  if (iter_args_size != return_vars_size) {
    std::ostringstream msg;
    msg << stmt_kind << " iter_args count (" << iter_args_size << ") != return_vars count ("
        << return_vars_size << ")";
    RecordError(ssa::ErrorType::ITER_ARGS_RETURN_VARS_MISMATCH, msg.str(), span);
  }

  // Check: If iter_args is not empty, body must end with YieldStmt and no
  // YieldStmt may appear mid-body (the trailing yield is the scope's
  // terminator). When iter_args is empty the scope produces no values, so
  // the mid-body check does not apply (matches the pre-SSA bare-yield case).
  if (iter_args_size > 0) {
    StmtPtr last_stmt = GetLastStmt(body);
    if (!last_stmt || !As<YieldStmt>(last_stmt)) {
      RecordError(ssa::ErrorType::MISSING_YIELD,
                  stmt_kind + " with iter_args must have YieldStmt as last statement in body", span);
    } else {
      auto yield = As<YieldStmt>(last_stmt);
      if (yield->value_.size() != iter_args_size) {
        std::ostringstream msg;
        msg << stmt_kind << " YieldStmt value count (" << yield->value_.size() << ") != iter_args count ("
            << iter_args_size << ")";
        RecordError(ssa::ErrorType::YIELD_COUNT_MISMATCH, msg.str(), span);
      } else {
        // Trailing yield is sound; check no earlier yield sits mid-body.
        // Skipping when trailing already failed avoids cascading errors.
        CheckNoMidBodyYield(stmt_kind, body);
      }
    }
  }
}

void SSAVerifier::VerifyIfStmt(const IfStmtPtr& if_stmt) {
  if (!if_stmt) return;

  // Check only if return_vars is not empty
  if (if_stmt->return_vars_.empty()) {
    return;
  }

  // Check 1: else_body must exist
  if (!if_stmt->else_body_.has_value()) {
    RecordError(ssa::ErrorType::MISSING_YIELD, "IfStmt with return_vars must have else branch",
                if_stmt->span_);
    return;
  }

  // Check 2: Both then_body and else_body must end with YieldStmt
  StmtPtr then_last = GetLastStmt(if_stmt->then_body_);
  StmtPtr else_last = GetLastStmt(if_stmt->else_body_.value());

  auto then_yield = As<YieldStmt>(then_last);
  auto else_yield = As<YieldStmt>(else_last);

  if (!then_yield) {
    RecordError(ssa::ErrorType::MISSING_YIELD,
                "IfStmt then branch must end with YieldStmt when return_vars exist", if_stmt->span_);
  } else if (then_yield->value_.size() != if_stmt->return_vars_.size()) {
    std::ostringstream msg;
    msg << "IfStmt then-branch YieldStmt value count (" << then_yield->value_.size()
        << ") != return_vars count (" << if_stmt->return_vars_.size() << ")";
    RecordError(ssa::ErrorType::YIELD_COUNT_MISMATCH, msg.str(), if_stmt->span_);
  } else {
    // Trailing yield is sound; check no earlier yield sits mid-body. Skipping
    // when trailing already failed avoids cascading errors with redundant
    // function dumps.
    CheckNoMidBodyYield("IfStmt then-branch", if_stmt->then_body_);
  }

  if (!else_yield) {
    RecordError(ssa::ErrorType::MISSING_YIELD,
                "IfStmt else branch must end with YieldStmt when return_vars exist", if_stmt->span_);
  } else if (else_yield->value_.size() != if_stmt->return_vars_.size()) {
    std::ostringstream msg;
    msg << "IfStmt else-branch YieldStmt value count (" << else_yield->value_.size()
        << ") != return_vars count (" << if_stmt->return_vars_.size() << ")";
    RecordError(ssa::ErrorType::YIELD_COUNT_MISMATCH, msg.str(), if_stmt->span_);
  } else {
    CheckNoMidBodyYield("IfStmt else-branch", if_stmt->else_body_.value());
  }
}

void SSAVerifier::VisitStmt_(const AssignStmtPtr& op) {
  if (!op || !op->var_) return;

  // Visit RHS first (uses current scope)
  if (op->value_) VisitExpr(op->value_);

  // Check for multiple assignments
  CheckVariableAssignment(op->var_);

  // Define the LHS variable in current scope
  DefineVar(op->var_);

  // Register type-embedded Var references (e.g., dynamic shape vars in TensorType)
  // as defined in the current scope. These are external/global references (like
  // pl.dynamic vars) that may not be function parameters but appear in types
  // propagated from other functions (e.g., InCore return types used in Orchestration).
  RegisterTypeVars(op->var_->GetType());
}

void SSAVerifier::VisitStmt_(const ForStmtPtr& op) {
  if (!op) return;

  // return_vars are visible after the loop, but not while evaluating the loop
  // header or body.
  for (const auto& return_var : op->return_vars_) {
    if (return_var) {
      CheckVariableAssignment(return_var);
    }
  }

  // Visit start, stop, step, and iter_args' initValue in current (outer) scope
  if (op->start_) VisitExpr(op->start_);
  if (op->stop_) VisitExpr(op->stop_);
  if (op->step_) VisitExpr(op->step_);

  for (const auto& iter_arg : op->iter_args_) {
    if (iter_arg && iter_arg->initValue_) {
      VisitExpr(iter_arg->initValue_);
    }
  }

  // Enter body scope for loop_var and iter_args
  EnterScope();

  // Define loop_var in body scope
  DefineVar(op->loop_var_);

  // Define iter_args in body scope
  for (const auto& iter_arg : op->iter_args_) {
    if (iter_arg) {
      DefineVar(iter_arg);
    }
  }

  // Visit loop body
  if (op->body_) {
    VisitStmt(op->body_);
  }

  ExitScope();

  VerifyLoopIterArgsAndYield("ForStmt", op->iter_args_.size(), op->return_vars_.size(), op->body_, op->span_);

  // return_vars become visible only after the loop exits
  for (const auto& return_var : op->return_vars_) {
    DefineVar(return_var);
  }
}

void SSAVerifier::VisitStmt_(const WhileStmtPtr& op) {
  if (!op) return;

  // return_vars are visible after the loop, but not while evaluating the loop
  // header, condition, or body.
  for (const auto& return_var : op->return_vars_) {
    if (return_var) {
      CheckVariableAssignment(return_var);
    }
  }

  // Visit iter_args' initValue in current (outer) scope
  for (const auto& iter_arg : op->iter_args_) {
    if (iter_arg && iter_arg->initValue_) {
      VisitExpr(iter_arg->initValue_);
    }
  }

  // Enter body scope
  EnterScope();

  // Define iter_args in body scope
  for (const auto& iter_arg : op->iter_args_) {
    if (iter_arg) {
      DefineVar(iter_arg);
    }
  }

  // Visit condition in body scope (it references iter_args)
  if (op->condition_) {
    VisitExpr(op->condition_);
  }

  // Visit loop body
  if (op->body_) {
    VisitStmt(op->body_);
  }

  ExitScope();

  VerifyLoopIterArgsAndYield("WhileStmt", op->iter_args_.size(), op->return_vars_.size(), op->body_,
                             op->span_);

  // return_vars become visible only after the loop exits
  for (const auto& return_var : op->return_vars_) {
    DefineVar(return_var);
  }
}

void SSAVerifier::VisitStmt_(const IfStmtPtr& op) {
  if (!op) return;

  // return_vars are visible after the if, but not in the condition or branches.
  for (const auto& return_var : op->return_vars_) {
    if (return_var) {
      CheckVariableAssignment(return_var);
    }
  }

  // Visit condition in current scope
  if (op->condition_) {
    VisitExpr(op->condition_);
  }

  // Visit then branch in its own scope
  EnterScope();
  if (op->then_body_) {
    VisitStmt(op->then_body_);
  }
  ExitScope();

  // Visit else branch in its own scope (if exists)
  if (op->else_body_.has_value() && op->else_body_.value()) {
    EnterScope();
    VisitStmt(op->else_body_.value());
    ExitScope();
  }

  // Verify IfStmt specific constraints
  VerifyIfStmt(op);

  // return_vars become visible only after the if exits
  for (const auto& return_var : op->return_vars_) {
    DefineVar(return_var);
  }
}

}  // namespace

/**
 * @brief SSA property verifier for use with PropertyVerifierRegistry
 */
class SSAPropertyVerifierImpl : public PropertyVerifier {
 public:
  [[nodiscard]] std::string GetName() const override { return "SSAVerify"; }

  void Verify(const ProgramPtr& program, std::vector<Diagnostic>& diagnostics) override {
    if (!program) {
      return;
    }

    for (const auto& [global_var, func] : program->functions_) {
      if (!func) {
        continue;
      }

      // Create verifier and run verification per function
      SSAVerifier verifier(diagnostics, func->name_, func);

      // Register function parameters (and their type-embedded vars) in the outermost scope
      verifier.RegisterParams(func->params_);

      // Register Var references from return types (dynamic shape vars)
      verifier.RegisterReturnTypeVars(func->return_types_);

      if (func->body_) {
        verifier.VisitStmt(func->body_);
      }
    }
  }
};

PropertyVerifierPtr CreateSSAPropertyVerifier() { return std::make_shared<SSAPropertyVerifierImpl>(); }

}  // namespace ir
}  // namespace pypto
