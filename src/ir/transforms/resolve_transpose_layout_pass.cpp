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
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/program.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

namespace {

// ---------------------------------------------------------------------------
// Phase 1 helpers: scan InCore functions for tile.load(..., transpose=True)
// ---------------------------------------------------------------------------

struct TransposeParamInfo {
  size_t param_index;
  std::vector<ExprPtr> new_shape;
};

/**
 * Visitor that scans an InCore function body for tile.load calls with
 * transpose=True whose source tensor is a function parameter.
 */
class TransposeLoadScanner : public IRVisitor {
 public:
  explicit TransposeLoadScanner(const std::vector<VarPtr>& params) {
    for (size_t i = 0; i < params.size(); ++i) {
      param_name_to_index_[params[i]->name_] = i;
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
          auto it = param_name_to_index_.find(src_var->name_);
          if (it != param_name_to_index_.end()) {
            size_t param_idx = it->second;
            if (visited_params_.count(param_idx) == 0) {
              visited_params_.insert(param_idx);

              // args_[2] is the shapes tuple (MakeTuple)
              CHECK(call->args_.size() >= 3) << "tile.load must have at least 3 positional args";
              auto shapes_tuple = As<MakeTuple>(call->args_[2]);
              CHECK(shapes_tuple) << "tile.load shapes arg must be a MakeTuple";
              CHECK(shapes_tuple->elements_.size() == 2)
                  << "transpose=true only supports 2D shapes, got " << shapes_tuple->elements_.size();

              results_.push_back({param_idx, shapes_tuple->elements_});
            }
          }
        }
      }
    }

    IRVisitor::VisitExpr_(call);
  }

 private:
  std::unordered_map<std::string, size_t> param_name_to_index_;
  std::unordered_set<size_t> visited_params_;
  std::vector<TransposeParamInfo> results_;
};

// ---------------------------------------------------------------------------
// Var substitution mutator: replaces Var references by name
// ---------------------------------------------------------------------------

class VarSubstitutionMutator : public IRMutator {
 public:
  explicit VarSubstitutionMutator(std::unordered_map<std::string, VarPtr> substitutions)
      : substitutions_(std::move(substitutions)) {}

 protected:
  ExprPtr VisitExpr_(const VarPtr& var) override {
    auto it = substitutions_.find(var->name_);
    if (it != substitutions_.end()) {
      return it->second;
    }
    return var;
  }

 private:
  std::unordered_map<std::string, VarPtr> substitutions_;
};

// ---------------------------------------------------------------------------
// Phase 1: transform InCore function parameters
// ---------------------------------------------------------------------------

struct IncoreTransformResult {
  FunctionPtr func;
  // param_index -> new TensorType (for Phase 2 to propagate to callers)
  std::unordered_map<size_t, std::shared_ptr<const TensorType>> modified_params;
};

IncoreTransformResult TransformIncoreParams(const FunctionPtr& func) {
  TransposeLoadScanner scanner(func->params_);
  scanner.VisitStmt(func->body_);

  const auto& results = scanner.GetResults();
  if (results.empty()) {
    return {func, {}};
  }

  std::unordered_map<size_t, std::shared_ptr<const TensorType>> modified_params;
  std::unordered_map<std::string, VarPtr> substitutions;
  std::vector<VarPtr> new_params = func->params_;

  for (const auto& info : results) {
    const auto& old_param = func->params_[info.param_index];
    auto old_tensor_type = As<TensorType>(old_param->GetType());
    CHECK(old_tensor_type) << "transpose load source param must be TensorType";

    // Skip if already has DN layout
    if (old_tensor_type->tensor_view_.has_value() &&
        old_tensor_type->tensor_view_->layout == TensorLayout::DN) {
      continue;
    }

    auto new_tensor_type =
        std::make_shared<TensorType>(info.new_shape, old_tensor_type->dtype_, old_tensor_type->memref_,
                                     std::optional<TensorView>(TensorView({}, TensorLayout::DN)));

    auto new_var = std::make_shared<Var>(old_param->name_, new_tensor_type, old_param->span_);
    new_params[info.param_index] = new_var;
    substitutions[old_param->name_] = new_var;
    modified_params[info.param_index] = new_tensor_type;
  }

  if (substitutions.empty()) {
    return {func, {}};
  }

  VarSubstitutionMutator mutator(std::move(substitutions));
  auto new_body = mutator.VisitStmt(func->body_);

  auto new_func = std::make_shared<Function>(func->name_, new_params, func->param_directions_,
                                             func->return_types_, new_body, func->span_, func->func_type_);

  return {new_func, std::move(modified_params)};
}

// ---------------------------------------------------------------------------
// Phase 2: propagate type changes to Orchestration / Opaque function callers
// ---------------------------------------------------------------------------

/**
 * Visitor that finds calls to modified InCore functions in the body and
 * collects which caller variables need type updates.
 */
class CallerArgCollector : public IRVisitor {
 public:
  using ModifiedMap =
      std::unordered_map<std::string, std::unordered_map<size_t, std::shared_ptr<const TensorType>>>;

  explicit CallerArgCollector(const ModifiedMap& incore_modifications)
      : incore_modifications_(incore_modifications) {}

  // var_name -> new TensorType (for variables that need updating in the caller)
  const std::unordered_map<std::string, std::shared_ptr<const TensorType>>& GetVarUpdates() const {
    return var_updates_;
  }

  void VisitExpr_(const CallPtr& call) override {
    if (!call) return;

    auto global_var = std::dynamic_pointer_cast<const GlobalVar>(call->op_);
    if (global_var) {
      auto it = incore_modifications_.find(global_var->name_);
      if (it != incore_modifications_.end()) {
        for (const auto& [param_idx, new_type] : it->second) {
          if (param_idx < call->args_.size()) {
            auto arg_var = As<Var>(call->args_[param_idx]);
            if (arg_var && var_updates_.find(arg_var->name_) == var_updates_.end()) {
              var_updates_[arg_var->name_] = new_type;
            }
          }
        }
      }
    }

    IRVisitor::VisitExpr_(call);
  }

 private:
  const ModifiedMap& incore_modifications_;
  std::unordered_map<std::string, std::shared_ptr<const TensorType>> var_updates_;
};

FunctionPtr UpdateCallerFunction(
    const FunctionPtr& func,
    const std::unordered_map<std::string, std::unordered_map<size_t, std::shared_ptr<const TensorType>>>&
        incore_mods) {
  CallerArgCollector collector(incore_mods);
  collector.VisitStmt(func->body_);

  const auto& var_updates = collector.GetVarUpdates();
  if (var_updates.empty()) {
    return func;
  }

  // Build substitution map and update params
  std::unordered_map<std::string, VarPtr> substitutions;
  std::vector<VarPtr> new_params = func->params_;

  for (size_t i = 0; i < func->params_.size(); ++i) {
    auto it = var_updates.find(func->params_[i]->name_);
    if (it != var_updates.end()) {
      auto new_var = std::make_shared<Var>(func->params_[i]->name_, it->second, func->params_[i]->span_);
      new_params[i] = new_var;
      substitutions[func->params_[i]->name_] = new_var;
    }
  }

  if (substitutions.empty()) {
    return func;
  }

  VarSubstitutionMutator mutator(std::move(substitutions));
  auto new_body = mutator.VisitStmt(func->body_);

  return std::make_shared<Function>(func->name_, new_params, func->param_directions_, func->return_types_,
                                    new_body, func->span_, func->func_type_);
}

}  // namespace

namespace pass {

Pass ResolveTransposeLayout() {
  auto pass_func = [](const ProgramPtr& program) -> ProgramPtr {
    // Phase 1: Transform InCore functions -- detect tile.load transpose and update param types
    using ModifiedMap =
        std::unordered_map<std::string, std::unordered_map<size_t, std::shared_ptr<const TensorType>>>;
    ModifiedMap incore_modifications;
    std::vector<FunctionPtr> functions_phase1;

    for (const auto& [gvar, func] : program->functions_) {
      if (IsInCoreType(func->func_type_)) {
        auto result = TransformIncoreParams(func);
        if (!result.modified_params.empty()) {
          incore_modifications[func->name_] = std::move(result.modified_params);
        }
        functions_phase1.push_back(result.func);
      } else {
        functions_phase1.push_back(func);
      }
    }

    if (incore_modifications.empty()) {
      return program;
    }

    // Phase 2: Propagate type changes to Orchestration/Opaque callers
    std::vector<FunctionPtr> functions_phase2;
    for (const auto& func : functions_phase1) {
      if (!IsInCoreType(func->func_type_)) {
        functions_phase2.push_back(UpdateCallerFunction(func, incore_modifications));
      } else {
        functions_phase2.push_back(func);
      }
    }

    return std::make_shared<Program>(functions_phase2, program->name_, program->span_);
  };

  return CreateProgramPass(pass_func, "ResolveTransposeLayout", kResolveTransposeLayoutProperties);
}

}  // namespace pass

}  // namespace ir
}  // namespace pypto
