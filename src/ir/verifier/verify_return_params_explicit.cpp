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
#include <unordered_set>
#include <vector>

#include "pypto/core/error.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/program.h"
#include "pypto/ir/transforms/utils/return_lineage_utils.h"
#include "pypto/ir/verifier/verifier.h"

namespace pypto {
namespace ir {

/// Verifies IRProperty::ReturnParamsExplicit: in every InCore/Group/Spmd/Graph/Orchestration
/// function, each tensor return value that is a param writeback references
/// the param by pointer identity instead of an SSA alias of it. Scalar
/// returns and kernel-allocated tensors (untraceable to any param; they only
/// become tail Out params at Submit lowering) are exempt.
class ReturnParamsExplicitVerifierImpl : public PropertyVerifier {
 public:
  [[nodiscard]] std::string GetName() const override { return "ReturnParamsExplicit"; }

  void Verify(const ProgramPtr& program, std::vector<Diagnostic>& diagnostics) override {
    if (!program) return;
    // All queries inspect the same immutable Program. Reuse recursive callee
    // maps across functions, then clear them when this verification ends.
    return_lineage::ScopedTraceCache trace_scope;
    for (const auto& [gv, func] : program->functions_) {
      if (!func || !func->body_) continue;
      // A declaration-only extern has no ReturnStmt, but still must identify
      // its returned parameters. Validate the contract before a wrapper can
      // mistake unresolved extern lineage for a fresh kernel allocation.
      if (func->HasAttr(kAttrExternalSource)) {
        try {
          (void)return_lineage::ExplicitReturnedParamIndices(func);
        } catch (const pypto::Error& error) {
          diagnostics.emplace_back(DiagnosticSeverity::Error, "ReturnParamsExplicit", 0, error.what(),
                                   func->span_);
        }
        continue;
      }
      if (func->HasAttr(kAttrExternalReturnParams)) {
        diagnostics.emplace_back(DiagnosticSeverity::Error, "ReturnParamsExplicit", 0,
                                 "Only external kernels may declare external returned parameters",
                                 func->span_);
      }
      // All normalized tensor callees, including CHIP orchestration, expose
      // return identities before allocation metadata is assigned.
      const bool applies = IsInCoreType(func->func_type_) || func->func_type_ == FunctionType::Group ||
                           func->func_type_ == FunctionType::Spmd ||
                           func->func_type_ == FunctionType::Graph ||
                           func->func_type_ == FunctionType::Orchestration;
      if (!applies) continue;

      bool has_tensor_return = false;
      for (const auto& ty : func->return_types_) {
        if (AsTensorTypeLike(ty)) has_tensor_return = true;
      }
      if (!has_tensor_return) continue;

      const auto returns = return_lineage::CollectReturns(func->body_);
      if (returns.empty()) {
        diagnostics.emplace_back(
            DiagnosticSeverity::Error, "ReturnParamsExplicit", 0,
            "Function '" + func->name_ + "' declares tensor return types but has no ReturnStmt", func->span_);
        continue;
      }

      std::unordered_set<const Var*> param_set;
      for (const auto& p : func->params_) param_set.insert(p.get());

      auto ret_to_param = return_lineage::ReturnedParamIndices(func, program);
      for (const auto& ret : returns) {
        const auto& values = ret->value_;
        for (size_t i = 0; i < values.size(); ++i) {
          if (i < func->return_types_.size() && !AsTensorTypeLike(func->return_types_[i])) continue;
          auto var = AsVarLike(values[i]);
          if (var && param_set.count(var.get())) continue;
          // A position the lineage cannot trace to a param is a fresh
          // kernel-allocated tensor — exempt. Traceable-but-aliased is the
          // violation this property forbids.
          if (i >= ret_to_param.size()) continue;
          const auto& parameter = ret_to_param[i];
          if (!parameter.has_value()) continue;
          std::string name = var ? var->name_hint_ : std::string("<non-var>");
          diagnostics.emplace_back(DiagnosticSeverity::Error, "ReturnParamsExplicit", 0,
                                   "Function '" + func->name_ + "' return position " + std::to_string(i) +
                                       " ('" + name + "') aliases param index " + std::to_string(*parameter) +
                                       "; param-writeback returns must reference the param directly",
                                   ret->span_);
        }
      }
    }
  }
};

PropertyVerifierPtr CreateReturnParamsExplicitPropertyVerifier() {
  return std::make_shared<ReturnParamsExplicitVerifierImpl>();
}

}  // namespace ir
}  // namespace pypto
