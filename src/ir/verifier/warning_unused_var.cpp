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
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/core/error.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/program.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/verifier/verifier.h"
#include "pypto/ir/verifier/warning_verifier_registry.h"

namespace pypto {
namespace ir {

namespace {

/// Warning error codes (1000+ range for warnings)
constexpr int kUnusedVariableCode = 1001;
constexpr int kUnusedControlFlowResultCode = 1002;

// Two-pass approach:
//   Pass 1 (UseCollector): collect all Var pointers read in expressions.
//   Pass 2 (UnusedVarChecker): walk definitions, report any not in the use set.
// Function parameters, loop variables, and iter_args are excluded.

/// Collects Var pointers from expression use-sites only.
/// Overrides statement visitors to skip definition-site vars (AssignStmt::var_,
/// ForStmt::loop_var_, return_vars_, etc.).
class UseCollector : public IRVisitor {
 public:
  std::unordered_set<const Var*> used_vars;

 protected:
  void VisitVarLike_(const VarPtr& op) override {
    if (op) {
      used_vars.insert(op.get());
    }
    // Don't recurse into type shape expressions — they aren't data-flow uses.
  }

  void VisitStmt_(const AssignStmtPtr& op) override {
    if (!op) return;
    if (op->value_) VisitExpr(op->value_);
    // Skip var_ — definition site, not a use
  }

  void VisitStmt_(const ForStmtPtr& op) override {
    if (!op) return;
    if (op->start_) VisitExpr(op->start_);
    if (op->stop_) VisitExpr(op->stop_);
    if (op->step_) VisitExpr(op->step_);
    if (op->chunk_config_.has_value() && op->chunk_config_->size) {
      VisitExpr(op->chunk_config_->size);
    }
    for (const auto& iter_arg : op->iter_args_) {
      if (iter_arg && iter_arg->initValue_) {
        VisitExpr(iter_arg->initValue_);
      }
    }
    if (op->body_) VisitStmt(op->body_);
    // Skip loop_var_, iter_args_ (as vars), return_vars_ — definition sites
  }

  void VisitStmt_(const WhileStmtPtr& op) override {
    if (!op) return;
    for (const auto& iter_arg : op->iter_args_) {
      if (iter_arg && iter_arg->initValue_) {
        VisitExpr(iter_arg->initValue_);
      }
    }
    if (op->condition_) VisitExpr(op->condition_);
    if (op->body_) VisitStmt(op->body_);
    // Skip iter_args_ (as vars), return_vars_ — definition sites
  }

  void VisitStmt_(const IfStmtPtr& op) override {
    if (!op) return;
    if (op->condition_) VisitExpr(op->condition_);
    if (op->then_body_) VisitStmt(op->then_body_);
    if (op->else_body_.has_value() && *op->else_body_) {
      VisitStmt(*op->else_body_);
    }
    // Skip return_vars_ — definition sites
  }
};

/// Walk statement structure only (no expression traversal) to find definitions
/// and report any that don't appear in the uses set.
///
/// When `check_return_vars` is false (UnusedVariable), only AssignStmt definitions
/// are checked.  When true (UnusedControlFlowResult), only for/while/if return_vars
/// are checked.  The two checks are orthogonal and can be enabled independently.
class UnusedVarChecker : public IRVisitor {
 public:
  UnusedVarChecker(const std::unordered_set<const Var*>& used_vars,
                   const std::unordered_set<const Var*>& param_vars, std::vector<Diagnostic>& diagnostics,
                   std::string func_name, bool check_return_vars)
      : used_vars_(used_vars),
        param_vars_(param_vars),
        diagnostics_(diagnostics),
        func_name_(std::move(func_name)),
        check_return_vars_(check_return_vars) {}

 protected:
  // Skip all expression traversal — pass 1 already collected uses.
  void VisitExpr(const ExprPtr& /*expr*/) override {}

  void VisitStmt_(const AssignStmtPtr& op) override {
    if (!op) return;
    if (!check_return_vars_ && op->var_) CheckUnused(op->var_);
  }

  void VisitStmt_(const ForStmtPtr& op) override {
    if (!op) return;
    if (op->body_) VisitStmt(op->body_);
    if (check_return_vars_) {
      for (const auto& rv : op->return_vars_) {
        CheckUnused(rv);
      }
    }
  }

  void VisitStmt_(const WhileStmtPtr& op) override {
    if (!op) return;
    if (op->body_) VisitStmt(op->body_);
    if (check_return_vars_) {
      for (const auto& rv : op->return_vars_) {
        CheckUnused(rv);
      }
    }
  }

  void VisitStmt_(const IfStmtPtr& op) override {
    if (!op) return;
    if (op->then_body_) VisitStmt(op->then_body_);
    if (op->else_body_.has_value() && *op->else_body_) {
      VisitStmt(*op->else_body_);
    }
    if (check_return_vars_) {
      for (const auto& rv : op->return_vars_) {
        CheckUnused(rv);
      }
    }
  }

 private:
  void CheckUnused(const VarPtr& var) {
    if (!var) return;
    if (param_vars_.count(var.get()) > 0) return;
    if (used_vars_.count(var.get()) == 0) {
      const char* rule = check_return_vars_ ? "UnusedControlFlowResultCheck" : "UnusedVariableCheck";
      int code = check_return_vars_ ? kUnusedControlFlowResultCode : kUnusedVariableCode;
      std::ostringstream msg;
      msg << "Unused variable '" << var->name_hint_ << "' in function '" << func_name_ << "'";
      diagnostics_.emplace_back(DiagnosticSeverity::Warning, rule, code, msg.str(), var->span_);
    }
  }

  const std::unordered_set<const Var*>& used_vars_;
  const std::unordered_set<const Var*>& param_vars_;
  std::vector<Diagnostic>& diagnostics_;
  std::string func_name_;
  bool check_return_vars_;
};

/// Shared logic for both verifiers — collects uses once, then runs the checker.
class UnusedVarWarningVerifierBase : public PropertyVerifier {
 public:
  explicit UnusedVarWarningVerifierBase(bool check_return_vars) : check_return_vars_(check_return_vars) {}

  void Verify(const ProgramPtr& program, std::vector<Diagnostic>& diagnostics) override {
    if (!program) return;

    for (const auto& [global_var, func] : program->functions_) {
      if (!func) continue;

      UseCollector collector;
      if (func->body_) collector.VisitStmt(func->body_);

      std::unordered_set<const Var*> param_vars;
      for (const auto& param : func->params_) {
        if (param) param_vars.insert(param.get());
      }

      UnusedVarChecker checker(collector.used_vars, param_vars, diagnostics, func->name_, check_return_vars_);
      if (func->body_) checker.VisitStmt(func->body_);
    }
  }

 private:
  bool check_return_vars_;
};

class UnusedVariableWarningVerifierImpl : public UnusedVarWarningVerifierBase {
 public:
  UnusedVariableWarningVerifierImpl() : UnusedVarWarningVerifierBase(/*check_return_vars=*/false) {}
  [[nodiscard]] std::string GetName() const override { return "UnusedVariableCheck"; }
};

class UnusedControlFlowResultWarningVerifierImpl : public UnusedVarWarningVerifierBase {
 public:
  UnusedControlFlowResultWarningVerifierImpl() : UnusedVarWarningVerifierBase(/*check_return_vars=*/true) {}
  [[nodiscard]] std::string GetName() const override { return "UnusedControlFlowResultCheck"; }
};

}  // namespace

PropertyVerifierPtr CreateUnusedVariableWarningVerifier() {
  return std::make_shared<UnusedVariableWarningVerifierImpl>();
}

PropertyVerifierPtr CreateUnusedControlFlowResultWarningVerifier() {
  return std::make_shared<UnusedControlFlowResultWarningVerifierImpl>();
}

}  // namespace ir
}  // namespace pypto
