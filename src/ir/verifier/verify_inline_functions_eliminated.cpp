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
#include <unordered_set>
#include <vector>

#include "pypto/core/error.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/program.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/verifier/verifier.h"

namespace pypto {
namespace ir {

namespace {

/**
 * @brief Detects Calls whose callee resolves (by name) to a FunctionType::Inline
 *        function in the program.
 */
class InlineCallVisitor : public IRVisitor {
 public:
  InlineCallVisitor(const std::unordered_set<std::string>& inline_names, std::vector<Diagnostic>& diagnostics)
      : inline_names_(inline_names), diagnostics_(diagnostics) {}

 protected:
  void VisitExpr_(const CallPtr& op) override {
    if (op) {
      if (auto gv = As<GlobalVar>(op->op_); gv && inline_names_.count(gv->name_) > 0) {
        diagnostics_.emplace_back(
            DiagnosticSeverity::Error, "InlineFunctionsEliminated", 1,
            "Call to FunctionType::Inline function '" + gv->name_ + "' survived the InlineFunctions pass",
            op->span_);
      }
    }
    IRVisitor::VisitExpr_(op);
  }

 private:
  const std::unordered_set<std::string>& inline_names_;
  std::vector<Diagnostic>& diagnostics_;
};

class InlineFunctionsEliminatedVerifierImpl : public PropertyVerifier {
 public:
  [[nodiscard]] std::string GetName() const override { return "InlineFunctionsEliminated"; }

  void Verify(const ProgramPtr& program, std::vector<Diagnostic>& diagnostics) override {
    if (!program) return;

    // Pass 1: report any surviving Inline functions and collect their names.
    std::unordered_set<std::string> inline_names;
    for (const auto& [gvar, func] : program->functions_) {
      if (!func) continue;
      if (func->func_type_ == FunctionType::Inline) {
        inline_names.insert(func->name_);
        diagnostics.emplace_back(
            DiagnosticSeverity::Error, "InlineFunctionsEliminated", 0,
            "FunctionType::Inline function '" + func->name_ +
                "' survived the InlineFunctions pass (should have been spliced and removed)",
            func->span_);
      }
    }

    // Pass 2: report any Call resolving to (the names of) those functions.
    // By construction the function is gone from program->functions_, so a
    // surviving Call would dangle — catch it here either way.
    if (inline_names.empty()) return;
    for (const auto& [gvar, func] : program->functions_) {
      if (!func || !func->body_) continue;
      InlineCallVisitor visitor(inline_names, diagnostics);
      visitor.VisitStmt(func->body_);
    }
  }
};

}  // namespace

PropertyVerifierPtr CreateInlineFunctionsEliminatedPropertyVerifier() {
  return std::make_shared<InlineFunctionsEliminatedVerifierImpl>();
}

}  // namespace ir
}  // namespace pypto
