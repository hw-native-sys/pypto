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

#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "pypto/core/error.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/verifier/verifier.h"

namespace pypto {
namespace ir {
namespace {

/// Walks a function body, reporting any ``SpmdScopeStmt`` whose ``core_num_``
/// has not folded to a positive ``ConstInt``. Emitted after Simplify; anything
/// still unresolved by this point indicates the user passed a runtime value
/// (e.g. a function parameter or non-foldable expression) as ``core_num``.
class CoreNumChecker : public IRVisitor {
 public:
  CoreNumChecker(std::vector<Diagnostic>& diagnostics, const std::string& func_name)
      : diagnostics_(diagnostics), func_name_(func_name) {}

  void VisitStmt_(const SpmdScopeStmtPtr& op) override {
    auto ci = As<ConstInt>(op->core_num_);
    if (ci == nullptr) {
      diagnostics_.emplace_back(DiagnosticSeverity::Error, "CoreNumResolved", 0,
                                "pl.spmd core_num in function '" + func_name_ +
                                    "' did not fold to a compile-time integer. core_num must resolve "
                                    "to a positive ConstInt after Simplify — closure-captured Python "
                                    "ints and closure arithmetic are fine, but IR runtime values "
                                    "(e.g. function parameters) are not supported.",
                                op->span_);
    } else if (ci->value_ <= 0) {
      diagnostics_.emplace_back(DiagnosticSeverity::Error, "CoreNumResolved", 1,
                                "pl.spmd core_num in function '" + func_name_ + "' must be positive, got " +
                                    std::to_string(ci->value_),
                                op->span_);
    } else if (ci->value_ > std::numeric_limits<int>::max()) {
      // Downstream orchestration codegen reads core_num as int via Function::GetAttr<int>.
      // Enforce the bound centrally so all consumers can safely narrow int64 → int.
      diagnostics_.emplace_back(DiagnosticSeverity::Error, "CoreNumResolved", 2,
                                "pl.spmd core_num in function '" + func_name_ +
                                    "' exceeds the int32 range; got " + std::to_string(ci->value_) +
                                    ", max is " + std::to_string(std::numeric_limits<int>::max()),
                                op->span_);
    }
    IRVisitor::VisitStmt_(op);
  }

 private:
  std::vector<Diagnostic>& diagnostics_;
  const std::string& func_name_;
};

class CoreNumResolvedPropertyVerifierImpl : public PropertyVerifier {
 public:
  [[nodiscard]] std::string GetName() const override { return "CoreNumResolved"; }

  void Verify(const ProgramPtr& program, std::vector<Diagnostic>& diagnostics) override {
    if (!program) return;
    for (const auto& [gv, func] : program->functions_) {
      if (!func || !func->body_) continue;
      CoreNumChecker checker(diagnostics, func->name_);
      checker.VisitStmt(func->body_);
    }
  }
};

}  // namespace

PropertyVerifierPtr CreateCoreNumResolvedPropertyVerifier() {
  return std::make_shared<CoreNumResolvedPropertyVerifierImpl>();
}

}  // namespace ir
}  // namespace pypto
