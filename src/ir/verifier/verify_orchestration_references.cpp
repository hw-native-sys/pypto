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
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/verifier/verifier.h"

namespace pypto {
namespace ir {
namespace {

/// Local copy of the builtin-op classifier. Verifiers live in the IR layer
/// and must not depend on `pypto::codegen::IsBuiltinOp`. The classification
/// is intentionally a string-prefix check; if a third reader appears,
/// promote it to a shared IR utility.
bool IsBuiltinOpName(const std::string& name) {
  return name.rfind("tile.", 0) == 0 || name.rfind("tensor.", 0) == 0 || name.rfind("system.", 0) == 0 ||
         name.rfind("array.", 0) == 0;
}

class OrchestrationCallTargetChecker : public IRVisitor {
 public:
  OrchestrationCallTargetChecker(ProgramPtr program, std::vector<Diagnostic>& diagnostics,
                                 std::string func_name)
      : program_(std::move(program)), diagnostics_(diagnostics), func_name_(std::move(func_name)) {}

 protected:
  void VisitExpr_(const CallPtr& call) override {
    IRVisitor::VisitExpr_(call);
    if (!call || !call->op_) return;
    if (IsBuiltinOpName(call->op_->name_)) return;
    if (program_ && program_->GetFunction(call->op_->name_)) return;

    std::ostringstream oss;
    oss << "Orchestration function '" << func_name_ << "' references undefined function '" << call->op_->name_
        << "'. The Program must contain every callee referenced from orchestration.";
    diagnostics_.emplace_back(DiagnosticSeverity::Error, "OrchestrationReferencesResolved", 0, oss.str(),
                              call->span_);
  }

 private:
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
      if (func->func_type_ != FunctionType::Orchestration) continue;
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
