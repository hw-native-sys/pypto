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

#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "pypto/core/error.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/program.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/verification_error.h"
#include "pypto/ir/transforms/verifier.h"

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
    default:
      return "UNKNOWN";
  }
}
}  // namespace ssa

namespace {
/**
 * @brief Helper visitor class for SSA verification
 *
 * Traverses the IR tree and collects SSA violations
 */
class SSAVerifier : public IRVisitor {
 public:
  explicit SSAVerifier(std::vector<Diagnostic>& diagnostics) : diagnostics_(diagnostics) {}

  void VisitStmt_(const AssignStmtPtr& op) override;
  void VisitStmt_(const ForStmtPtr& op) override;
  void VisitStmt_(const WhileStmtPtr& op) override;
  void VisitStmt_(const IfStmtPtr& op) override;

  [[nodiscard]] const std::vector<Diagnostic>& GetDiagnostics() const { return diagnostics_; }

 private:
  std::vector<Diagnostic>& diagnostics_;
  std::unordered_map<std::string, int> var_assignment_count_;

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
   * @brief Verify ForStmt specific constraints
   */
  void VerifyForStmt(const ForStmtPtr& for_stmt);

  /**
   * @brief Verify WhileStmt specific constraints
   */
  void VerifyWhileStmt(const WhileStmtPtr& while_stmt);

  /**
   * @brief Verify IfStmt specific constraints
   */
  void VerifyIfStmt(const IfStmtPtr& if_stmt);
};

void SSAVerifier::CheckVariableAssignment(const VarPtr& var) {
  if (!var) return;

  const std::string& var_name = var->name_;
  var_assignment_count_[var_name]++;

  if (var_assignment_count_[var_name] > 1) {
    std::ostringstream msg;
    msg << "Variable '" << var_name << "' is assigned more than once (" << var_assignment_count_[var_name]
        << " times), violating SSA form";
    RecordError(ssa::ErrorType::MULTIPLE_ASSIGNMENT, msg.str(), var->span_);
  }
}

void SSAVerifier::RecordError(ssa::ErrorType type, const std::string& message, const Span& span) {
  diagnostics_.emplace_back(DiagnosticSeverity::Error, "SSAVerify", static_cast<int>(type), message, span);
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

void SSAVerifier::VerifyForStmt(const ForStmtPtr& for_stmt) {
  if (!for_stmt) return;

  // Check: If iter_args is not empty, body must end with YieldStmt
  if (!for_stmt->iter_args_.empty()) {
    StmtPtr last_stmt = GetLastStmt(for_stmt->body_);
    if (!last_stmt || !As<YieldStmt>(last_stmt)) {
      RecordError(ssa::ErrorType::MISSING_YIELD,
                  "ForStmt with iter_args must have YieldStmt as last statement in body", for_stmt->span_);
    }
  }
}

void SSAVerifier::VerifyWhileStmt(const WhileStmtPtr& while_stmt) {
  if (!while_stmt) return;

  // Check: If iter_args is not empty, body must end with YieldStmt
  if (!while_stmt->iter_args_.empty()) {
    StmtPtr last_stmt = GetLastStmt(while_stmt->body_);
    if (!last_stmt || !As<YieldStmt>(last_stmt)) {
      RecordError(ssa::ErrorType::MISSING_YIELD,
                  "WhileStmt with iter_args must have YieldStmt as last statement in body",
                  while_stmt->span_);
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
  }

  if (!else_yield) {
    RecordError(ssa::ErrorType::MISSING_YIELD,
                "IfStmt else branch must end with YieldStmt when return_vars exist", if_stmt->span_);
  }
}

void SSAVerifier::VisitStmt_(const AssignStmtPtr& op) {
  if (!op || !op->var_) return;

  // Check for multiple assignments
  CheckVariableAssignment(op->var_);

  // Continue with default traversal
  IRVisitor::VisitStmt_(op);
}

void SSAVerifier::VisitStmt_(const ForStmtPtr& op) {
  if (!op) return;

  // Check return_vars for multiple assignments
  for (const auto& return_var : op->return_vars_) {
    if (return_var) {
      CheckVariableAssignment(return_var);
    }
  }

  // Visit start, stop, step, and iter_args' initValue in current scope
  // These are all evaluated in the outer scope before the loop begins
  if (op->start_) VisitExpr(op->start_);
  if (op->stop_) VisitExpr(op->stop_);
  if (op->step_) VisitExpr(op->step_);

  for (const auto& iter_arg : op->iter_args_) {
    if (iter_arg && iter_arg->initValue_) {
      VisitExpr(iter_arg->initValue_);
    }
  }

  // Visit loop body
  if (op->body_) {
    VisitStmt(op->body_);
  }

  // Verify ForStmt specific constraints
  VerifyForStmt(op);
}

void SSAVerifier::VisitStmt_(const WhileStmtPtr& op) {
  if (!op) return;

  // Check return_vars for multiple assignments
  for (const auto& return_var : op->return_vars_) {
    if (return_var) {
      CheckVariableAssignment(return_var);
    }
  }

  // Visit iter_args' initValue in current scope
  // These are all evaluated in the outer scope before the loop begins
  for (const auto& iter_arg : op->iter_args_) {
    if (iter_arg && iter_arg->initValue_) {
      VisitExpr(iter_arg->initValue_);
    }
  }

  // Visit condition (it references iter_args)
  if (op->condition_) {
    VisitExpr(op->condition_);
  }

  // Visit loop body
  if (op->body_) {
    VisitStmt(op->body_);
  }

  // Verify WhileStmt specific constraints
  VerifyWhileStmt(op);
}

void SSAVerifier::VisitStmt_(const IfStmtPtr& op) {
  if (!op) return;

  // Check return_vars for multiple assignments
  for (const auto& return_var : op->return_vars_) {
    if (return_var) {
      CheckVariableAssignment(return_var);
    }
  }

  // Visit condition in current scope
  if (op->condition_) {
    VisitExpr(op->condition_);
  }

  // Visit then branch
  if (op->then_body_) {
    VisitStmt(op->then_body_);
  }

  // Visit else branch (if exists)
  if (op->else_body_.has_value() && op->else_body_.value()) {
    VisitStmt(op->else_body_.value());
  }

  // Verify IfStmt specific constraints
  VerifyIfStmt(op);
}

}  // namespace

/**
 * @brief SSA property verifier for use with IRVerifier
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
      SSAVerifier verifier(diagnostics);

      if (func->body_) {
        verifier.VisitStmt(func->body_);
      }
    }
  }
};

PropertyVerifierPtr CreateSSAPropertyVerifier() { return std::make_shared<SSAPropertyVerifierImpl>(); }

}  // namespace ir
}  // namespace pypto
