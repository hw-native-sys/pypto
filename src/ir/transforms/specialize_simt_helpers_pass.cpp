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

#include <any>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <list>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/backend/common/backend.h"
#include "pypto/backend/common/backend_config.h"
#include "pypto/backend/common/backend_handler.h"
#include "pypto/core/error.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/simt.h"
#include "pypto/ir/simt_analysis.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_context.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/structural_comparison.h"
#include "pypto/ir/transforms/utils/deep_clone_utils.h"
#include "pypto/ir/transforms/utils/memref_utils.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/transforms/utils/var_collectors.h"
#include "pypto/ir/type.h"
#include "pypto/ir/verifier/verifier.h"
#include "src/ir/transforms/utils/bounded_kernel_index.h"

namespace pypto::ir {
namespace {

std::string TargetArch() {
  const auto* context = PassContext::Current();
  const auto* target = context ? context->GetBackend() : backend::GetBackend();
  return target->GetHandler()->GetPtoTargetArch();
}

std::vector<int> BlockDims(const SimtLaunchStmtPtr& launch) {
  std::vector<int> result;
  for (const auto& dim : launch->block_dims_) result.push_back(static_cast<int>(As<ConstInt>(dim)->value_));
  return result;
}

/// Construct an allocation-independent signature and its caller-side metadata
/// arguments. Dynamic metadata is passed by value, never captured from another
/// function or specialized on its runtime value. The signature is a Function
/// so structural comparison preserves references to particular scalar params.
class SignatureBuilder {
 public:
  SignatureBuilder(const FunctionPtr& helper, const SimtLaunchStmtPtr& launch) {
    parameters.resize(helper->params_.size());
    for (const auto& parameter : helper->params_) names_.insert(parameter->name_hint_);
    for (size_t i = 0; i < parameters.size(); ++i) {
      const auto& original = helper->params_[i];
      if (!As<ScalarType>(original->GetType())) continue;
      parameters[i] = std::make_shared<Var>(original->name_hint_, original->GetType(), original->span_);
      actual_to_formal_.emplace(launch->args_[i].get(), parameters[i]);
      substitutions.emplace(original.get(), parameters[i]);
      scalar_parameters_.insert(original.get());
    }
    for (size_t i = 0; i < parameters.size(); ++i) {
      if (parameters[i]) continue;
      const auto& original = helper->params_[i];
      auto type = CloneTypeWithMemRefAndRemapExprs(launch->args_[i]->GetType(), std::nullopt,
                                                   [this](const ExprPtr& expr) { return Capture(expr); });
      parameters[i] = std::make_shared<Var>(original->name_hint_, type, original->span_);
      substitutions.emplace(original.get(), parameters[i]);
      BindShape(original->GetType(), type, launch->span_);
    }
    parameters.insert(parameters.end(), metadata_parameters_.begin(), metadata_parameters_.end());
    auto directions = helper->param_directions_;
    directions.resize(parameters.size(), ParamDirection::In);
    signature = std::make_shared<Function>(
        "simt_signature", parameters, std::move(directions), std::vector<TypePtr>{},
        std::make_shared<ReturnStmt>(std::vector<ExprPtr>{}, Span::unknown()), Span::unknown(),
        FunctionType::SIMT);
    weight += parameters.size();
  }

  FunctionPtr signature;
  std::vector<VarPtr> parameters;
  std::vector<ExprPtr> metadata_arguments;
  std::unordered_map<const Var*, ExprPtr> substitutions;
  size_t weight = 1;

 private:
  ExprPtr Capture(const ExprPtr& expr) {
    ++weight;
    if (!expr || As<ConstInt>(expr)) return expr;
    auto found = actual_to_formal_.find(expr.get());
    if (found != actual_to_formal_.end() && structural_equal(found->second->GetType(), expr->GetType())) {
      return found->second;
    }
    CHECK_SPAN(As<ScalarType>(expr->GetType()), expr->span_)
        << "SIMT storage metadata must be a scalar expression";
    std::string name;
    do {
      name = "simt_meta_" + std::to_string(next_name_++);
    } while (!names_.insert(name).second);
    auto formal = std::make_shared<Var>(std::move(name), expr->GetType(), Span::unknown());
    actual_to_formal_[expr.get()] = formal;
    metadata_parameters_.push_back(formal);
    metadata_arguments.push_back(expr);
    return formal;
  }

  void BindShape(const TypePtr& original, const TypePtr& actual, const Span& span) {
    auto before = As<ShapedType>(original);
    auto after = As<ShapedType>(actual);
    INTERNAL_CHECK_SPAN(before && after && before->shape_.size() == after->shape_.size(), span)
        << "Internal error: SIMT specialization requires matching storage ranks";
    for (size_t i = 0; i < before->shape_.size(); ++i) {
      auto symbol = AsVarLike(before->shape_[i]);
      if (!symbol || scalar_parameters_.count(symbol.get()) != 0) continue;
      auto [entry, inserted] = substitutions.emplace(symbol.get(), after->shape_[i]);
      CHECK_SPAN(inserted || structural_equal(entry->second, after->shape_[i]), span)
          << "SIMT helper shape symbol has inconsistent argument bindings";
    }
  }

  size_t next_name_ = 0;
  std::set<std::string> names_;
  std::unordered_map<const Expr*, VarPtr> actual_to_formal_;
  std::unordered_set<const Var*> scalar_parameters_;
  std::vector<VarPtr> metadata_parameters_;
};

struct Variant {
  FunctionPtr original;
  FunctionPtr signature;
  std::vector<int> dims;
  std::string target;
  size_t weight;
  FunctionPtr function;
};

class Specializer : public IRMutator {
 public:
  explicit Specializer(ProgramPtr program) : program_(std::move(program)) {
    for (const auto& [_, function] : program_->functions_) names_.insert(function->name_);
  }

  [[nodiscard]] const std::vector<FunctionPtr>& Functions() const { return functions_; }

 protected:
  StmtPtr VisitStmt_(const SimtLaunchStmtPtr& launch) override {
    const auto resources = InferSimtResources(launch);
    const auto target = TargetArch();
    const auto dims = BlockDims(launch);
    const auto helper = program_->GetFunction(launch->callee_->name_);
    INTERNAL_CHECK_SPAN(helper && helper->func_type_ == FunctionType::SIMT, launch->span_)
        << "Internal error: SIMT launch must reference a verified helper";
    SignatureBuilder signature(helper, launch);

    // Re-running the pass after memory planning must neither specialize on
    // allocation addresses nor append metadata parameters a second time.
    if (helper->GetAttr<std::vector<int>>(kSimtBlockDimsAttr) == dims &&
        helper->GetAttr<std::string>(kSimtTargetAttr) == target &&
        helper->GetAttr<int>(kSimtMaxThreadsAttr) == resources.max_threads &&
        helper->GetAttr<int>(kSimtMaxRegsAttr) == resources.max_regs &&
        signature.metadata_arguments.empty()) {
      auto existing_signature = MutableCopy(signature.signature);
      existing_signature->params_ = helper->params_;
      existing_signature->param_directions_ = helper->param_directions_;
      if (structural_equal(existing_signature, signature.signature)) return launch;
    }

    variants_.push_back({helper, signature.signature, dims, target, signature.weight, nullptr});
    auto& candidate = variants_.back();
    uint64_t hash = structural_hash(candidate.signature);
    auto mix = [&hash](uint64_t value) { hash = (hash ^ value) * UINT64_C(1099511628211); };
    mix(std::hash<const Function*>{}(helper.get()));
    mix(std::hash<std::string>{}(target));
    for (int64_t dim : dims) mix(static_cast<uint64_t>(dim));
    // Use the shared bounded collision index: repeated launches do not cause
    // an unbounded pairwise signature scan, even under hash collisions.
    const auto* variant = index_.FindOrInsert(hash, candidate, [](const Variant& lhs, const Variant& rhs) {
      return lhs.original == rhs.original && lhs.dims == rhs.dims && lhs.target == rhs.target &&
             structural_equal(lhs.signature, rhs.signature);
    });
    if (variant == &candidate) {
      auto cloned = DeepClone(helper->body_, signature.substitutions);
      var_collectors::VarDefUseCollector bindings;
      bindings.VisitStmt(cloned.cloned_body);
      for (const auto& parameter : signature.parameters) bindings.var_defs.insert(parameter.get());
      for (const auto* use : bindings.var_uses) {
        CHECK_SPAN(bindings.var_defs.count(use) != 0, launch->span_)
            << "SIMT helper has unbound metadata; pass it explicitly as a scalar parameter";
      }
      auto function = MutableCopy(helper);
      do {
        function->name_ = helper->name_ + "_simt_" + std::to_string(next_name_++);
      } while (!names_.insert(function->name_).second);
      function->params_ = std::move(signature.parameters);
      function->param_directions_ = signature.signature->param_directions_;
      function->body_ = std::move(cloned.cloned_body);
      SetAttribute(*function, kSimtBlockDimsAttr, dims);
      SetAttribute(*function, kSimtTargetAttr, target);
      SetAttribute(*function, kSimtMaxThreadsAttr, resources.max_threads);
      SetAttribute(*function, kSimtMaxRegsAttr, resources.max_regs);
      candidate.function = function;
      functions_.push_back(std::move(function));
    }
    auto result = MutableCopy(launch);
    result->callee_ = std::make_shared<GlobalVar>(variant->function->name_);
    result->args_.insert(result->args_.end(), signature.metadata_arguments.begin(),
                         signature.metadata_arguments.end());
    return result;
  }

 private:
  static void SetAttribute(Function& function, const std::string& key, std::any value) {
    for (auto& [name, attribute] : function.attrs_) {
      if (name == key) {
        attribute = std::move(value);
        return;
      }
    }
    function.attrs_.emplace_back(key, std::move(value));
  }

  ProgramPtr program_;
  std::set<std::string> names_;
  size_t next_name_ = 0;
  std::list<Variant> variants_;
  deduplication_detail::BoundedIndex<Variant> index_;
  std::vector<FunctionPtr> functions_;
};

ProgramPtr Specialize(const ProgramPtr& program) {
  Specializer specializer(program);
  auto result = MutableCopy(program);
  for (auto& [_, function] : result->functions_) {
    if (function->func_type_ != FunctionType::SIMT) function = specializer.VisitFunction(function);
  }
  for (const auto& helper : specializer.Functions()) {
    result->functions_.emplace(std::make_shared<GlobalVar>(helper->name_), helper);
  }
  return result;
}

class SpecializedVerifier : public PropertyVerifier {
 public:
  [[nodiscard]] std::string GetName() const override { return "SimtSpecialized"; }
  void Verify(const ProgramPtr& program, std::vector<Diagnostic>& diagnostics) override {
    class Checker : public IRVisitor {
     public:
      Checker(const ProgramPtr& program, std::vector<Diagnostic>& diagnostics)
          : program_(program), diagnostics_(diagnostics) {}
      void VisitStmt_(const SimtLaunchStmtPtr& launch) override {
        const auto helper = program_->GetFunction(launch->callee_->name_);
        const auto resources = InferSimtResources(launch);
        if (!helper || helper->GetAttr<std::vector<int>>(kSimtBlockDimsAttr) != BlockDims(launch) ||
            helper->GetAttr<int>(kSimtMaxThreadsAttr) != resources.max_threads ||
            helper->GetAttr<int>(kSimtMaxRegsAttr) != resources.max_regs ||
            helper->GetAttr<std::string>(kSimtTargetAttr) != TargetArch()) {
          diagnostics_.emplace_back(DiagnosticSeverity::Error, "SimtSpecialized", 0,
                                    "SIMT helper resources do not match the launch dimensions and target",
                                    launch->span_);
        }
      }

     private:
      const ProgramPtr& program_;
      std::vector<Diagnostic>& diagnostics_;
    } checker(program, diagnostics);
    checker.VisitProgram(program);
  }
};
}  // namespace

namespace pass {
Pass SpecializeSimtHelpers() {
  return CreateProgramPass(Specialize, "SpecializeSimtHelpers", kSpecializeSimtHelpersProperties);
}
}  // namespace pass

PropertyVerifierPtr CreateSimtSpecializedPropertyVerifier() {
  return std::make_shared<SpecializedVerifier>();
}
}  // namespace pypto::ir
