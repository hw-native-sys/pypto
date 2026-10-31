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

#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/verifier/verifier.h"

namespace pypto::ir {
namespace {
class ContractChecker : public IRVisitor {
 public:
  explicit ContractChecker(std::vector<Diagnostic>& diagnostics) : diagnostics_(diagnostics) {}
  void VisitStmt_(const ForStmtPtr& op) override {
    if ((op->hardware_loop_policy_ != HardwareLoopPolicy::Auto &&
         op->hardware_loop_policy_ != HardwareLoopPolicy::Require) ||
        (op->hardware_loop_policy_ == HardwareLoopPolicy::Require && op->kind_ != ForKind::Sequential)) {
      diagnostics_.emplace_back(DiagnosticSeverity::Error, "HardwareLoopContractValid", 0,
                                "HardwareLoopPolicy.Require requires Sequential ForKind and a valid policy",
                                op->span_);
    }
    IRVisitor::VisitStmt_(op);
  }

 private:
  std::vector<Diagnostic>& diagnostics_;
};
class ContractVerifier : public PropertyVerifier {
 public:
  [[nodiscard]] std::string GetName() const override { return "HardwareLoopContractValid"; }
  void Verify(const ProgramPtr& program, std::vector<Diagnostic>& diagnostics) override {
    if (!program) return;
    ContractChecker checker(diagnostics);
    checker.VisitProgram(program);
  }
};
}  // namespace
PropertyVerifierPtr CreateHardwareLoopContractPropertyVerifier() {
  return std::make_shared<ContractVerifier>();
}
}  // namespace pypto::ir
