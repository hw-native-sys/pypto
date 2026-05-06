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
 * @brief Detects Calls whose callee GlobalVar resolves to either:
 *        (a) a still-living FunctionType::Inline function (`inline_names`),
 *        or (b) no function at all in the program (`known_names`).
 *
 * Case (b) catches the "InlineFunctions dropped the callee but left the Call
 * dangling" failure mode even when no Inline function survives in the program.
 * Both findings indicate the InlineFunctions pass left work undone.
 */
class InlineCallVisitor : public IRVisitor {
 public:
  InlineCallVisitor(const std::unordered_set<std::string>& inline_names,
                    const std::unordered_set<std::string>& known_names, std::vector<Diagnostic>& diagnostics)
      : inline_names_(inline_names), known_names_(known_names), diagnostics_(diagnostics) {}

 protected:
  void VisitExpr_(const CallPtr& op) override {
    if (op) {
      if (auto gv = As<GlobalVar>(op->op_)) {
        if (inline_names_.count(gv->name_) > 0) {
          diagnostics_.emplace_back(
              DiagnosticSeverity::Error, "InlineFunctionsEliminated", 1,
              "Call to FunctionType::Inline function '" + gv->name_ + "' survived the InlineFunctions pass",
              op->span_);
        } else if (known_names_.count(gv->name_) == 0) {
          // Dangling Call — no function with this name exists in the program.
          // The most likely cause is InlineFunctions dropping the callee
          // without rewriting this site.
          diagnostics_.emplace_back(
              DiagnosticSeverity::Error, "InlineFunctionsEliminated", 2,
              "Dangling Call to function '" + gv->name_ +
                  "' (no such function in program — likely a former Inline callee that wasn't spliced)",
              op->span_);
        }
      }
    }
    IRVisitor::VisitExpr_(op);
  }

 private:
  const std::unordered_set<std::string>& inline_names_;
  const std::unordered_set<std::string>& known_names_;
  std::vector<Diagnostic>& diagnostics_;
};

class InlineFunctionsEliminatedVerifierImpl : public PropertyVerifier {
 public:
  [[nodiscard]] std::string GetName() const override { return "InlineFunctionsEliminated"; }

  void Verify(const ProgramPtr& program, std::vector<Diagnostic>& diagnostics) override {
    if (!program) return;

    // Pass 1: report any surviving Inline functions and collect their names,
    //         plus collect every defined function name in the program.
    std::unordered_set<std::string> inline_names;
    std::unordered_set<std::string> known_names;
    for (const auto& [gvar, func] : program->functions_) {
      if (!func) continue;
      known_names.insert(func->name_);
      if (func->func_type_ == FunctionType::Inline) {
        inline_names.insert(func->name_);
        diagnostics.emplace_back(
            DiagnosticSeverity::Error, "InlineFunctionsEliminated", 0,
            "FunctionType::Inline function '" + func->name_ +
                "' survived the InlineFunctions pass (should have been spliced and removed)",
            func->span_);
      }
    }

    // Pass 2: walk every function body. Flag both live-Inline-callee Calls
    //         (error_code 1) and dangling Calls (error_code 2). Always run —
    //         dangling Calls can outlive the dropped function entirely.
    InlineCallVisitor visitor(inline_names, known_names, diagnostics);
    for (const auto& [gvar, func] : program->functions_) {
      if (!func || !func->body_) continue;
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
