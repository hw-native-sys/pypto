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
#include <string>
#include <vector>

#include "pypto/core/error.h"
#include "pypto/ir/function.h"
#include "pypto/ir/program.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/verifier/verifier.h"

namespace pypto {
namespace ir {

namespace {

/**
 * @brief Visitor that detects BreakStmt/ContinueStmt in a function body.
 *
 * Reports a diagnostic error for every BreakStmt or ContinueStmt found.
 * This verifier is used after CtrlFlowTransform to confirm no break/continue remains.
 */
class StructuredCtrlFlowChecker : public IRVisitor {
 public:
  explicit StructuredCtrlFlowChecker(std::vector<Diagnostic>& diagnostics) : diagnostics_(diagnostics) {}

  void VisitStmt_(const BreakStmtPtr& op) override {
    if (!op) return;
    diagnostics_.emplace_back(DiagnosticSeverity::Error, "StructuredCtrlFlow", 0,
                              "BreakStmt found — IR is not in structured control flow form", op->span_);
  }

  void VisitStmt_(const ContinueStmtPtr& op) override {
    if (!op) return;
    diagnostics_.emplace_back(DiagnosticSeverity::Error, "StructuredCtrlFlow", 1,
                              "ContinueStmt found — IR is not in structured control flow form", op->span_);
  }

 private:
  std::vector<Diagnostic>& diagnostics_;
};

}  // namespace

/**
 * @brief StructuredCtrlFlow property verifier for use with IRVerifier
 *
 * Verifies that no BreakStmt or ContinueStmt remains in any function body.
 */
class StructuredCtrlFlowPropertyVerifierImpl : public PropertyVerifier {
 public:
  [[nodiscard]] std::string GetName() const override { return "StructuredCtrlFlow"; }

  void Verify(const ProgramPtr& program, std::vector<Diagnostic>& diagnostics) override {
    if (!program) return;

    for (const auto& [global_var, func] : program->functions_) {
      if (!func || !func->body_) continue;

      // Only verify InCore-type functions (InCore, AIC, AIV).
      // Orchestration/Host functions can use break/continue natively.
      if (!IsInCoreType(func->func_type_)) continue;

      StructuredCtrlFlowChecker checker(diagnostics);
      checker.VisitStmt(func->body_);
    }
  }
};

PropertyVerifierPtr CreateStructuredCtrlFlowPropertyVerifier() {
  return std::make_shared<StructuredCtrlFlowPropertyVerifierImpl>();
}

}  // namespace ir
}  // namespace pypto
