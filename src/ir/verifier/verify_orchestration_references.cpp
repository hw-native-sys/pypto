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
#include <utility>
#include <vector>

#include "pypto/core/error.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/program.h"
#include "pypto/ir/span.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/utils/op_predicates.h"
#include "pypto/ir/verifier/verifier.h"

namespace pypto {
namespace ir {
namespace {

using op_predicates::IsBuiltinOp;

class OrchestrationCallTargetChecker : public IRVisitor {
 public:
  OrchestrationCallTargetChecker(ProgramPtr program, std::vector<Diagnostic>& diagnostics,
                                 std::string func_name)
      : program_(std::move(program)), diagnostics_(diagnostics), func_name_(std::move(func_name)) {}

 protected:
  void VisitExpr_(const CallPtr& call) override {
    IRVisitor::VisitExpr_(call);
    if (!call) return;
    CheckCallee(call->op_, call->span_);
  }

  // Submit is a sibling call-like kind, and IRVisitor dispatches it through a
  // separate handler that does not delegate to the Call path — so overriding
  // only the Call path left every `pl.submit(...)` callee unchecked. That gap
  // matters most for a Graph body, which is predominantly submits.
  void VisitExpr_(const SubmitPtr& submit) override {
    IRVisitor::VisitExpr_(submit);
    if (!submit) return;
    CheckCallee(submit->op_, submit->span_);
  }

 private:
  void CheckCallee(const OpPtr& op, const Span& span) const {
    if (!op) return;
    if (IsBuiltinOp(op->name_)) return;
    if (program_ && program_->GetFunction(op->name_)) return;

    std::ostringstream oss;
    oss << "Function '" << func_name_ << "' references undefined function '" << op->name_
        << "'. The Program must contain every callee referenced from orchestration.";
    diagnostics_.emplace_back(DiagnosticSeverity::Error, "OrchestrationReferencesResolved", 0, oss.str(),
                              span);
  }

  ProgramPtr program_;
  std::vector<Diagnostic>& diagnostics_;
  std::string func_name_;
};

class OrchestrationReferencesResolvedPropertyVerifierImpl : public PropertyVerifier {
 public:
  [[nodiscard]] std::string GetName() const override { return "OrchestrationReferencesResolved"; }

  void Verify(const ProgramPtr& program, std::vector<Diagnostic>& diagnostics) override {
    if (!program) return;
    for (const auto& [gv, func] : program->functions_) {
      if (!func || !func->body_) continue;
      // A Graph body's callees must be present in the Program for the same
      // reason an Orchestration body's must.
      if (!IsOrchestrationLike(func->func_type_)) continue;
      OrchestrationCallTargetChecker checker(program, diagnostics, func->name_);
      checker.VisitStmt(func->body_);
    }
  }
};

}  // namespace

PropertyVerifierPtr CreateOrchestrationReferencesResolvedPropertyVerifier() {
  return std::make_shared<OrchestrationReferencesResolvedPropertyVerifierImpl>();
}

}  // namespace ir
}  // namespace pypto
