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
#include "pypto/ir/verifier/verifier.h"

namespace pypto {
namespace ir {

namespace {

class RuntimeScopesMaterializedPropertyVerifierImpl : public PropertyVerifier {
 public:
  [[nodiscard]] std::string GetName() const override { return "RuntimeScopesMaterialized"; }

  void Verify(const ProgramPtr& program, std::vector<Diagnostic>& diagnostics) override {
    if (!program) return;
    for (const auto& [gv, func] : program->functions_) {
      // Graph bodies are orchestration too: codegen emits SIMPLER_SCOPE only from
      // RuntimeScopeStmt, so a Graph body that skipped MaterializeRuntimeScopes
      // would compile into a scope-less region.
      if (!func || !IsOrchestrationLike(func->func_type_)) continue;
      if (!func->GetAttr<bool>(kAttrAutoScope, true)) continue;

      diagnostics.emplace_back(DiagnosticSeverity::Error, "RuntimeScopesMaterialized", 0,
                               FunctionTypeToString(func->func_type_) + " function '" + func->name_ +
                                   "' still has auto_scope=True. Run MaterializeRuntimeScopes before "
                                   "orchestration codegen — codegen emits SIMPLER_SCOPE only from "
                                   "RuntimeScopeStmt nodes, not from implicit for/if wrapping.",
                               func->span_);
    }
  }
};

}  // namespace

PropertyVerifierPtr CreateRuntimeScopesMaterializedPropertyVerifier() {
  return std::make_shared<RuntimeScopesMaterializedPropertyVerifierImpl>();
}

}  // namespace ir
}  // namespace pypto
