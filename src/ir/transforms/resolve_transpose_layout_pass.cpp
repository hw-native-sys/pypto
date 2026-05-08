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
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/program.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

using transform_utils::Substitute;

namespace {

struct TransposeParamInfo {
  size_t param_index;
};

/**
 * Visitor that scans an InCore function body for tile.load calls with
 * transpose=True whose source tensor is a function parameter.
 */
class TransposeLoadScanner : public IRVisitor {
 public:
  explicit TransposeLoadScanner(const std::vector<VarPtr>& params) {
    for (size_t i = 0; i < params.size(); ++i) {
      param_ptr_to_index_[params[i].get()] = i;
    }
  }

  const std::vector<TransposeParamInfo>& GetResults() const { return results_; }

  void VisitExpr_(const CallPtr& call) override {
    if (!call) return;

    if (call->op_->name_ == "tile.load") {
      bool transpose = call->GetKwarg<bool>("transpose", false);
      if (transpose && !call->args_.empty()) {
        auto src_var = As<Var>(call->args_[0]);
        if (src_var) {
          auto it = param_ptr_to_index_.find(src_var.get());
          if (it != param_ptr_to_index_.end()) {
            size_t param_idx = it->second;
            if (visited_params_.count(param_idx) == 0) {
              visited_params_.insert(param_idx);
              results_.push_back({param_idx});
            }
          }
        }
      }
    }

    IRVisitor::VisitExpr_(call);
  }

 private:
  std::unordered_map<const Var*, size_t> param_ptr_to_index_;
  std::unordered_set<size_t> visited_params_;
  std::vector<TransposeParamInfo> results_;
};

// Add DN layout annotation to InCore parameters that have transpose tile.load.
// Shape is preserved (no swap); DN is a codegen hint only.
FunctionPtr TransformIncoreParams(const FunctionPtr& func) {
  TransposeLoadScanner scanner(func->params_);
  scanner.VisitStmt(func->body_);

  const auto& transpose_results = scanner.GetResults();
  std::unordered_set<size_t> needs_dn;
  for (const auto& info : transpose_results) {
    needs_dn.insert(info.param_index);
  }

  if (needs_dn.empty()) {
    return func;
  }

  std::unordered_map<const Var*, VarPtr> substitutions;
  std::vector<VarPtr> new_params = func->params_;

  for (size_t idx : needs_dn) {
    const auto& old_param = func->params_[idx];
    auto old_tensor_type = As<TensorType>(old_param->GetType());
    CHECK(old_tensor_type) << "DN candidate param must be TensorType";

    // Reject ONLY the "tensor.transpose result + tile.load(transpose=True)"
    // combination. tensor.transpose produces a TensorView with both DN layout
    // AND explicit physical strides; combining that with another transpose at
    // load time would double-encode the transpose and emit wrong addresses.
    //
    // Slice-derived inputs (explicit strides + ND layout, attached by
    // OptimizeOrchTensors) still flow through the normal "promote ND → DN,
    // drop strides" path used by matmul B^T patterns like paged_attention.
    if (old_tensor_type->tensor_view_.has_value()) {
      const auto& view = old_tensor_type->tensor_view_.value();
      CHECK(!(view.layout == TensorLayout::DN && !view.stride.empty()))
          << "tile.load(transpose=True) on a tensor.transpose result is not "
             "yet supported (the DN tag and explicit physical strides would "
             "compose as a double transpose). Workaround: do the transpose at "
             "the tile level via tile.load(transpose=True) directly on the "
             "source tensor.";
      if (view.layout == TensorLayout::DN) continue;
    }

    CHECK(old_tensor_type->shape_.size() >= 2)
        << "transpose layout resolution requires at least 2D tensors, got " << old_tensor_type->shape_.size()
        << "D";

    auto new_tensor_type = std::make_shared<TensorType>(
        old_tensor_type->shape_, old_tensor_type->dtype_, old_tensor_type->memref_,
        std::optional<TensorView>(TensorView(std::vector<ExprPtr>{}, TensorLayout::DN)));

    auto new_var = std::make_shared<Var>(old_param->name_hint_, new_tensor_type, old_param->span_);
    new_params[idx] = new_var;
    substitutions[old_param.get()] = new_var;
  }

  if (substitutions.empty()) {
    return func;
  }

  auto new_body = Substitute(func->body_, substitutions);

  auto new_func = MutableCopy(func);
  new_func->params_ = new_params;
  new_func->body_ = new_body;
  return new_func;
}

}  // namespace

namespace pass {

Pass ResolveTransposeLayout() {
  auto pass_func = [](const ProgramPtr& program) -> ProgramPtr {
    bool modified = false;
    std::vector<FunctionPtr> functions;

    for (const auto& [gvar, func] : program->functions_) {
      if (IsInCoreType(func->func_type_)) {
        auto new_func = TransformIncoreParams(func);
        if (new_func != func) {
          modified = true;
        }
        functions.push_back(new_func);
      } else {
        functions.push_back(func);
      }
    }

    if (!modified) {
      return program;
    }

    return std::make_shared<Program>(functions, program->name_, program->span_);
  };

  return CreateProgramPass(pass_func, "ResolveTransposeLayout", kResolveTransposeLayoutProperties);
}

}  // namespace pass

}  // namespace ir
}  // namespace pypto
