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
#include <string>
#include <vector>

#include "pypto/core/error.h"
#include "pypto/ir/function.h"
#include "pypto/ir/program.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/utils/attrs.h"
#include "pypto/ir/verifier/verifier.h"

namespace pypto {
namespace ir {

namespace {

/// Reports any Orchestration ForStmt whose iter_arg carry plan is missing or
/// malformed. ``ClassifyIterArgCarry`` stamps ``iter_arg_rebind_<i>`` for every
/// slot, so a loop with iter_args and no such attr means the pass never ran —
/// codegen would then silently lower every carry as a trivial alias, dropping
/// the yield-back writes that propagate loop state (issue #1286).
class CarryPlanChecker : public IRVisitor {
 public:
  CarryPlanChecker(std::vector<Diagnostic>& diagnostics, const std::string& func_name)
      : diagnostics_(diagnostics), func_name_(func_name) {}

 protected:
  void VisitStmt_(const ForStmtPtr& op) override {
    for (size_t i = 0; i < op->iter_args_.size(); ++i) {
      if (!op->HasAttr(IterArgRebindAttrKey(i))) {
        diagnostics_.emplace_back(
            DiagnosticSeverity::Error, "IterArgCarryClassified", 0,
            "ForStmt iter_arg " + std::to_string(i) + " in function '" + func_name_ + "' has no attrs[\"" +
                IterArgRebindAttrKey(i) +
                "\"]. ClassifyIterArgCarry must run before orchestration codegen; without it "
                "every loop carry lowers as a trivial alias and rebinds are silently dropped.",
            op->span_);
        continue;
      }
      // A positive array extent only makes sense for a materialised carry.
      if (op->GetAttr<int>(IterArgArraySizeAttrKey(i), 0) > 0 &&
          !op->GetAttr<bool>(IterArgRebindAttrKey(i), false)) {
        diagnostics_.emplace_back(DiagnosticSeverity::Error, "IterArgCarryClassified", 0,
                                  "ForStmt iter_arg " + std::to_string(i) + " in function '" + func_name_ +
                                      "' carries a TaskId array extent but is classified trivial. "
                                      "An array carry requires a materialised (rebind) carry.",
                                  op->span_);
      }
    }
    IRVisitor::VisitStmt_(op);
  }

 private:
  std::vector<Diagnostic>& diagnostics_;
  const std::string& func_name_;
};

class IterArgCarryClassifiedPropertyVerifierImpl : public PropertyVerifier {
 public:
  [[nodiscard]] std::string GetName() const override { return "IterArgCarryClassified"; }

  void Verify(const ProgramPtr& program, std::vector<Diagnostic>& diagnostics) override {
    if (!program) return;
    for (const auto& [gv, func] : program->functions_) {
      if (!func || !func->body_) continue;
      if (func->func_type_ != FunctionType::Orchestration) continue;
      CarryPlanChecker checker(diagnostics, func->name_);
      checker.VisitStmt(func->body_);
    }
  }
};

}  // namespace

PropertyVerifierPtr CreateIterArgCarryClassifiedPropertyVerifier() {
  return std::make_shared<IterArgCarryClassifiedPropertyVerifierImpl>();
}

}  // namespace ir
}  // namespace pypto
