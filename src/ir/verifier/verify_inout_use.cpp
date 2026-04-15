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
#include <utility>
#include <vector>

#include "pypto/core/error.h"
#include "pypto/ir/function.h"
#include "pypto/ir/program.h"
#include "pypto/ir/transforms/utils/stmt_dependency_analysis.h"
#include "pypto/ir/verifier/verifier.h"

namespace pypto {
namespace ir {

/**
 * @brief InOutUseValid property verifier (RFC #1026).
 *
 * Walks every function body and applies the InOut-use discipline:
 * a variable passed as an InOut or Out argument to a user-function call is
 * dead for reads from the call onward (CFG order, within the same scope).
 * The walk is shared with `BuildStmtDependencyGraph`'s precondition check —
 * see `CollectInOutUseDisciplineDiagnostics` in stmt_dependency_analysis.cpp
 * for the visitor implementation.
 *
 * Group-typed functions are skipped: they orchestrate parallel AIC/AIV calls
 * that intentionally share the same Out tensor, which the discipline (as
 * stated in RFC #1026) does not yet model. Treating Group bodies as opaque
 * here keeps the verifier useful for the common case while a follow-up
 * extends the rule to parallel-call constructs.
 */
class InOutUseValidPropertyVerifierImpl : public PropertyVerifier {
 public:
  [[nodiscard]] std::string GetName() const override { return "InOutUseValid"; }

  void Verify(const ProgramPtr& program, std::vector<Diagnostic>& diagnostics) override {
    if (!program) return;

    for (const auto& [global_var, func] : program->functions_) {
      if (!func || !func->body_) continue;
      if (func->func_type_ == FunctionType::Group) continue;
      auto func_diags = stmt_dep::CollectInOutUseDisciplineDiagnostics(func->body_, program);
      for (auto& d : func_diags) {
        diagnostics.push_back(std::move(d));
      }
    }
  }
};

PropertyVerifierPtr CreateInOutUseValidPropertyVerifier() {
  return std::make_shared<InOutUseValidPropertyVerifierImpl>();
}

}  // namespace ir
}  // namespace pypto
