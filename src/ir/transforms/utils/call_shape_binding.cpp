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

#include "pypto/ir/transforms/utils/call_shape_binding.h"

#include <cstddef>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/structural_comparison.h"
#include "pypto/ir/transforms/utils/signature_symbols.h"
#include "pypto/ir/type.h"

namespace pypto::ir {
namespace {

bool HasLayoutOnlyView(const TensorTypePtr& tensor) {
  return !tensor->tensor_view_ ||
         (tensor->tensor_view_->stride.empty() && tensor->tensor_view_->valid_shape.empty() &&
          tensor->tensor_view_->pad == PadValue::null);
}

// Matching counts pairs, not expanded trees. This caps both shared-DAG work and
// repeated caller constraints; the dedup pass supplies a program-wide budget.
class ShapeMatcher {
 public:
  explicit ShapeMatcher(size_t& budget) : budget_(budget) {}
  CallShapeStatus status = CallShapeStatus::Success;

  bool Match(const ExprPtr& pattern, const ExprPtr& actual, CallShapeMap* bindings, size_t depth = 0) {
    if (depth > 256) {
      status = CallShapeStatus::Unsupported;
      return false;
    }
    if (budget_ == 0) {
      status = CallShapeStatus::BudgetExhausted;
      return false;
    }
    --budget_;
    auto pt = As<ScalarType>(pattern->GetType());
    auto at = As<ScalarType>(actual->GetType());
    if (!pt || !at || pt->dtype_ != at->dtype_) return Conflict();
    if (auto var = AsVarLike(pattern); var && bindings) {
      auto [entry, inserted] = bindings->emplace(var.get(), actual);
      return inserted || Match(entry->second, actual, nullptr, depth + 1);
    }
    if (pattern == actual && !bindings) return true;
    auto& seen = bindings ? matched_ : equal_;
    if (!seen[pattern.get()].insert(actual.get()).second) return true;
    if (pattern->GetKind() != actual->GetKind()) return Conflict();
    if (AsVarLike(pattern)) return Conflict();  // Caller variables retain identity.
    if (auto constant = As<ConstInt>(pattern)) {
      return constant->value_ == As<ConstInt>(actual)->value_ || Conflict();
    }
    if (auto binary = As<BinaryExpr>(pattern)) {
      auto other = As<BinaryExpr>(actual);
      return Match(binary->left_, other->left_, bindings, depth + 1) &&
             Match(binary->right_, other->right_, bindings, depth + 1);
    }
    if (auto unary = As<UnaryExpr>(pattern)) {
      return Match(unary->operand_, As<UnaryExpr>(actual)->operand_, bindings, depth + 1);
    }
    // Only scalar constants reach this fallback. Do not recursively compare
    // arbitrary Call payloads or types outside the accounted shape grammar.
    if (As<ConstFloat>(pattern) || As<ConstBool>(pattern)) {
      return structural_equal(pattern, actual, false) || Conflict();
    }
    status = CallShapeStatus::Unsupported;
    return false;
  }

  bool Type(const TypePtr& pattern, const TypePtr& actual, CallShapeMap* bindings) {
    if (budget_ == 0) {
      status = CallShapeStatus::BudgetExhausted;
      return false;
    }
    --budget_;
    if (pattern->GetKind() != actual->GetKind()) return Conflict();
    if (auto scalar = As<ScalarType>(pattern)) {
      return scalar->dtype_ == As<ScalarType>(actual)->dtype_ || Conflict();
    }
    // Exact kind is intentional: this stage excludes distributed resources.
    auto tensor = As<TensorType>(pattern);
    auto other = As<TensorType>(actual);
    if (!tensor || !other || tensor->memref_ || other->memref_ || !HasLayoutOnlyView(tensor) ||
        !HasLayoutOnlyView(other) || tensor->tensor_view_.has_value() != other->tensor_view_.has_value() ||
        (tensor->tensor_view_ && tensor->tensor_view_->layout != other->tensor_view_->layout) ||
        tensor->dtype_ != other->dtype_ || tensor->shape_.size() != other->shape_.size()) {
      return Conflict();
    }
    for (size_t i = 0; i < tensor->shape_.size(); ++i) {
      if (!Match(tensor->shape_[i], other->shape_[i], bindings)) return false;
    }
    return true;
  }

 private:
  bool Conflict() {
    status = CallShapeStatus::Conflict;
    return false;
  }
  size_t& budget_;
  std::unordered_map<const Expr*, std::unordered_set<const Expr*>> matched_, equal_;
};

}  // namespace

CallShapeBindingPlan::CallShapeBindingPlan(FunctionPtr function) : function_(std::move(function)) {
  // Fixed signature limits also bound per-site reconstruction cost; these are
  // conservative admission limits, not restrictions on valid input programs.
  if (function_->params_.size() > 256 || function_->return_types_.size() > 256) {
    supported_ = false;
    return;
  }
  const auto symbols = ClassifySignatureSymbols(function_);
  if (!symbols.valid_shape.empty() || !symbols.unbound.empty()) supported_ = false;
  required_ = symbols.bound;
  std::unordered_set<const Var*> bound;
  for (const auto& var : symbols.bound) bound.insert(var.get());
  std::unordered_set<const Expr*> seen;
  std::unordered_set<const Type*> types;
  auto inspect = [&](const TypePtr& type) {
    if (!types.insert(type.get()).second) return;
    if (As<ScalarType>(type)) return;
    auto tensor = As<TensorType>(type);
    if (!tensor || tensor->memref_ || !HasLayoutOnlyView(tensor)) {
      supported_ = false;
      return;
    }
    std::vector<ExprPtr> pending(tensor->shape_);
    while (!pending.empty()) {
      auto expr = pending.back();
      pending.pop_back();
      if (!seen.insert(expr.get()).second) continue;
      // Bounds recursion in matching/instantiation, including adversarial DAGs.
      // Large or richer signatures retain their original definitions.
      if (seen.size() > 256) {
        supported_ = false;
        return;
      }
      auto scalar = As<ScalarType>(expr->GetType());
      if (!scalar || (scalar->dtype_ != DataType::INDEX && scalar->dtype_ != DataType::INT64)) {
        supported_ = false;
        return;
      }
      if (auto var = AsVarLike(expr)) {
        if (!bound.count(var.get())) supported_ = false;
      } else if (auto binary = As<BinaryExpr>(expr)) {
        pending.push_back(binary->left_);
        pending.push_back(binary->right_);
      } else if (auto unary = As<UnaryExpr>(expr)) {
        pending.push_back(unary->operand_);
      } else if (!As<ConstInt>(expr)) {
        supported_ = false;
      }
    }
  };
  for (const auto& param : function_->params_) inspect(param->GetType());
  for (const auto& type : function_->return_types_) inspect(type);
}

CallShapeStatus CallShapeBindingPlan::Bind(const std::vector<ExprPtr>& args, CallShapeMap& bindings,
                                           size_t& budget) const {
  if (!supported_) return CallShapeStatus::Unsupported;
  if (args.size() > function_->params_.size()) return CallShapeStatus::Conflict;
  ShapeMatcher matcher(budget);
  CallShapeMap result;
  for (size_t i = 0; i < args.size(); ++i) {
    const auto& param = function_->params_[i];
    if (!matcher.Type(param->GetType(), args[i]->GetType(), &result)) return matcher.status;
    if (As<ScalarType>(param->GetType()) && !matcher.Match(param, args[i], &result)) return matcher.status;
    // A wrapper can acquire private outputs from a richer child signature.
    // Their types may query tensor.dim(param, ...), so preserve tensor actuals
    // as substitution sources too, even when the original signature is simple.
    result.emplace(param.get(), args[i]);
  }
  for (size_t i = args.size(); i < function_->params_.size(); ++i) {
    if (function_->param_directions_[i] != ParamDirection::Out) return CallShapeStatus::Conflict;
  }
  // Out parameters omitted by Submit (or just added by tensor-to-tile) cannot
  // supply their own size. Every symbol must come from an existing actual.
  for (const auto& var : required_) {
    if (budget == 0) return CallShapeStatus::BudgetExhausted;
    --budget;
    if (As<ScalarType>(var->GetType()) && !result.count(var.get())) return CallShapeStatus::MissingSource;
  }
  bindings = std::move(result);
  return CallShapeStatus::Success;
}

bool CallShapeBindingPlan::MatchesResult(const ExprPtr& call, const CallShapeMap& bindings,
                                         size_t& budget) const {
  auto tuple = As<TupleType>(call->GetType());
  std::vector<TypePtr> actual;
  if (As<Submit>(call)) {
    if (function_->return_types_.empty()) {
      if (auto task = As<ScalarType>(call->GetType())) return task->dtype_ == DataType::TASK_ID;
    }
    if (!tuple || tuple->types_.size() != function_->return_types_.size() + 1) return false;
    auto task = As<ScalarType>(tuple->types_.back());
    if (!task || task->dtype_ != DataType::TASK_ID) return false;
    actual.assign(tuple->types_.begin(), tuple->types_.end() - 1);
  } else if (function_->return_types_.empty()) {
    return As<UnknownType>(call->GetType()) != nullptr;
  } else if (function_->return_types_.size() == 1) {
    actual.push_back(call->GetType());
  } else {
    if (!tuple) return false;
    actual = tuple->types_;
  }
  if (actual.size() != function_->return_types_.size()) return false;
  CallShapeInstantiation instantiate(bindings);
  ShapeMatcher matcher(budget);
  for (size_t i = 0; i < actual.size(); ++i) {
    if (!matcher.Type(instantiate.InstantiateType(function_->return_types_[i]), actual[i], nullptr)) {
      return false;
    }
  }
  return true;
}

void CallShapeInstantiation::BindOutput(const VarPtr& output, const ExprPtr& storage) {
  bindings_.emplace(output.get(), storage);
  // Earlier output types may reference this newly materialized parameter.
  // Cached substitutions belong to the preceding environment.
  expressions_.clear();
  types_.clear();
}

ExprPtr CallShapeInstantiation::Replace(const VarPtr& var) const {
  auto found = bindings_.find(var.get());
  return found == bindings_.end() ? var : found->second;
}
ExprPtr CallShapeInstantiation::VisitExpr_(const VarPtr& var) { return Replace(var); }
ExprPtr CallShapeInstantiation::VisitExpr_(const IterArgPtr& var) { return Replace(var); }
ExprPtr CallShapeInstantiation::VisitBinaryExpr_(const BinaryExprPtr& expr) {
  auto found = expressions_.find(expr.get());
  if (found != expressions_.end()) return found->second;
  auto result = IRMutator::VisitBinaryExpr_(expr);
  expressions_.emplace(expr.get(), result);
  return result;
}
ExprPtr CallShapeInstantiation::VisitUnaryExpr_(const UnaryExprPtr& expr) {
  auto found = expressions_.find(expr.get());
  if (found != expressions_.end()) return found->second;
  auto result = IRMutator::VisitUnaryExpr_(expr);
  expressions_.emplace(expr.get(), result);
  return result;
}
TypePtr CallShapeInstantiation::InstantiateType(const TypePtr& type) {
  auto found = types_.find(type.get());
  if (found != types_.end()) return found->second;
  auto result = RemapTypeViaVisitor(type);
  types_.emplace(type.get(), result);
  return result;
}

}  // namespace pypto::ir
