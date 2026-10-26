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

#include <algorithm>
#include <any>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/utils/auto_name_utils.h"
#include "pypto/ir/transforms/utils/call_shape_binding.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/transforms/utils/tile_conversion_utils.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/type.h"
#include "src/ir/transforms/tensor_to_tile/internal.h"
#include "src/ir/transforms/tensor_to_tile/type_propagation.h"

namespace pypto {
namespace ir {
namespace tensor_to_tile {

using tile_conversion_utils::MakeShapeTuple;

std::string MakeOutParamName(size_t index) {
  return auto_name::BuildName("ret" + std::to_string(index), "", "out");
}

namespace {

// ============================================================================
// Output propagation. Every call site receives its own output storage. A
// Group/Spmd exposes that storage as private Out parameters; orchestration
// allocates it before calling the wrapper. Logical returns remain unchanged.
// ============================================================================

// One incremental substitution per call site keeps shape instantiation linear
// in its argument/output types. Rebuilding a full substitution map for every
// shape expression would multiply the argument and output counts.
class CallTypeSubstituter : public IRMutator {
 public:
  void Bind(const VarPtr& param, const ExprPtr& argument) { var_remap_[param.get()] = argument; }
  TypePtr RemapType(const TypePtr& type) { return RemapTypeViaVisitor(type); }
};

class OutputParameterRewriter : public TypePropagatingMutator {
 public:
  OutputParameterRewriter(const AddedOutputMap& added_outputs,
                          const std::unordered_map<std::string, FunctionPtr>& functions,
                          const CallShapePlans& shape_plans, FunctionPtr wrapper = nullptr)
      : added_outputs_(added_outputs),
        functions_(functions),
        wrapper_(std::move(wrapper)),
        shape_plans_(shape_plans) {
    if (wrapper_) {
      params_ = wrapper_->params_;
      directions_ = wrapper_->param_directions_;
      output_start_ = params_.size() - transform_utils::TrailingCommCtxParamCount(wrapper_);
    }
  }

  std::vector<VarPtr> params() const {
    auto params = params_;
    std::rotate(params.begin() + static_cast<std::ptrdiff_t>(output_start_),
                params.begin() + static_cast<std::ptrdiff_t>(wrapper_->params_.size()), params.end());
    return params;
  }
  std::vector<ParamDirection> directions() const {
    auto directions = directions_;
    std::rotate(directions.begin() + static_cast<std::ptrdiff_t>(output_start_),
                directions.begin() + static_cast<std::ptrdiff_t>(wrapper_->params_.size()), directions.end());
    return directions;
  }
  const AddedOutputBindings& outputs() const { return outputs_; }

 protected:
  StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
    RetainVar(op->var_);
    auto value = VisitExpr(op->value_);
    std::vector<StmtPtr> prefix;
    value = RewriteCall(value, prefix);
    auto result = HandlePassThroughAssign(op, value);
    if (prefix.empty()) return result;
    prefix.push_back(result);
    return SeqStmts::Flatten(std::move(prefix), op->span_);
  }

  StmtPtr VisitStmt_(const EvalStmtPtr& op) override {
    auto value = VisitExpr(op->expr_);
    std::vector<StmtPtr> prefix;
    value = RewriteCall(value, prefix);
    if (prefix.empty() && value == op->expr_) return op;
    auto result = MutableCopy(op);
    result->expr_ = value;
    if (prefix.empty()) return result;
    prefix.push_back(result);
    return SeqStmts::Flatten(std::move(prefix), op->span_);
  }

 private:
  ExprPtr MakeStorage(const VarPtr& output, CallTypeSubstituter& arguments, CallShapeInstantiation* shapes,
                      const Span& span, size_t local_index, std::vector<StmtPtr>& prefix) {
    auto type = shapes ? shapes->InstantiateType(output->GetType()) : arguments.RemapType(output->GetType());
    auto tensor = As<TensorType>(type);
    CHECK_SPAN(tensor != nullptr, span)
        << "Cannot materialize output parameter '" << output->name_hint_
        << "': only ordinary GM Tensor outputs can be allocated at this call site";
    if (wrapper_) {
      size_t position = output_start_ + outputs_.size();
      auto param = std::make_shared<Var>(MakeOutParamName(outputs_.size()), type, span);
      params_.push_back(param);
      directions_.push_back(ParamDirection::Out);
      // This can be an internal producer output; it adds no logical wrapper return.
      outputs_.push_back({position});
      return param;
    }
    auto shape = MakeShapeTuple(tensor->shape_, span);
    TensorLayout layout = tensor->tensor_view_ ? tensor->tensor_view_->layout : TensorLayout::ND;
    auto create = OpRegistry::GetInstance().Create("tensor.create", {shape},
                                                   {{"dtype", tensor->dtype_}, {"layout", layout}}, span);
    auto var = std::make_shared<Var>(MakeOutParamName(local_index), create->GetType(), span);
    prefix.push_back(std::make_shared<AssignStmt>(var, create, span));
    return var;
  }

  std::vector<std::pair<std::string, std::any>> RewriteAttrs(
      const std::vector<std::pair<std::string, std::any>>& source, size_t prefix_size, size_t old_size,
      size_t new_size, const Span& span) const {
    auto attrs = source;
    const size_t inserted = new_size - old_size;
    for (auto& [key, value] : attrs) {
      if (key == kAttrArgDirections) {
        auto* dirs = std::any_cast<std::vector<ArgDirection>>(&value);
        INTERNAL_CHECK_SPAN(dirs != nullptr && dirs->size() == old_size, span)
            << "Internal error: malformed arg_directions while materializing kernel outputs";
        dirs->insert(dirs->begin() + static_cast<std::ptrdiff_t>(prefix_size), inserted,
                     ArgDirection::OutputExisting);
      } else if (key == kAttrArgDirectionOverrides) {
        auto* indices = std::any_cast<std::vector<int32_t>>(&value);
        INTERNAL_CHECK_SPAN(indices != nullptr, span)
            << "Internal error: malformed arg_direction_overrides while materializing kernel outputs";
        for (auto& index : *indices) {
          INTERNAL_CHECK_SPAN(index >= 0 && static_cast<size_t>(index) < old_size, span)
              << "Internal error: output argument override is out of range";
          if (static_cast<size_t>(index) >= prefix_size) index += static_cast<int32_t>(inserted);
        }
      }
    }
    return attrs;
  }

  ExprPtr RewriteCall(const ExprPtr& value, std::vector<StmtPtr>& prefix) {
    auto call = As<Call>(value);
    auto submit = As<Submit>(value);
    if (!call && !submit) return value;
    auto op = call ? call->op_ : submit->op_;
    auto global = As<GlobalVar>(op);
    if (!global) return value;
    auto function = functions_.find(global->name_);
    if (function == functions_.end()) return value;
    const auto& callee = function->second;
    auto added = added_outputs_.find(global->name_);
    if (added == added_outputs_.end()) return value;
    const auto& bindings = added->second;
    const auto& old_args = call ? call->args_ : submit->args_;
    const Span& span = value->span_;
    const size_t ctx = transform_utils::TrailingCommCtxParamCount(callee);
    const size_t old_param_count = callee->params_.size() - bindings.size();
    CHECK_SPAN(old_args.size() >= ctx && old_args.size() <= old_param_count, span)
        << "Call to '" << global->name_ << "' has incompatible arguments before output materialization";
    const size_t supplied = old_args.size() - ctx;
    const size_t original_nonctx = old_param_count - ctx;
    CHECK_SPAN(!call || old_args.size() == old_param_count, span)
        << "Call to '" << global->name_ << "' must supply every original parameter";
    // A Group's shared payload must also name outputs omitted by an inner
    // Submit, even when this pass added no new outputs to that callee.
    if (bindings.empty() && (!wrapper_ || supplied == original_nonctx)) return value;

    // Bind the original signature. Added Out parameters must not alter the
    // admission contract or provide the size of their own allocation.
    const auto& plan = shape_plans_.at(global->name_);
    CallShapeMap shape_bindings;
    size_t shape_budget = kCallShapeBudget;
    std::unique_ptr<CallShapeInstantiation> shapes;
    if (plan.Bind(old_args, shape_bindings, shape_budget) == CallShapeStatus::Success) {
      shapes = std::make_unique<CallShapeInstantiation>(std::move(shape_bindings));
    }
    // Richer legacy signatures keep their existing lowering path; dedup retains
    // their symbol-identity guard and never relies on this fallback.

    std::vector<ExprPtr> args(old_args.begin(), old_args.begin() + static_cast<std::ptrdiff_t>(supplied));
    CallTypeSubstituter arguments;
    for (size_t i = 0; i < supplied; ++i) arguments.Bind(callee->params_[i], old_args[i]);
    const size_t new_nonctx = callee->params_.size() - ctx;
    for (size_t i = 0; i < ctx; ++i) {
      arguments.Bind(callee->params_[new_nonctx + i], old_args[supplied + i]);
    }
    size_t local_index = 0;
    // Fill the old omitted-Out gap BEFORE the newly inserted output params.
    // Otherwise the first new output would incorrectly bind an old Out param.
    for (size_t i = supplied; i < original_nonctx; ++i) {
      CHECK_SPAN(callee->param_directions_[i] == ParamDirection::Out, span)
          << "Cannot materialize omitted non-Out parameter " << i << " of '" << global->name_ << "'";
      auto storage = MakeStorage(callee->params_[i], arguments, shapes.get(), span, local_index++, prefix);
      args.push_back(storage);
      arguments.Bind(callee->params_[i], storage);
      if (shapes) shapes->BindOutput(callee->params_[i], storage);
    }
    for (const auto& binding : bindings) {
      INTERNAL_CHECK_SPAN(binding.param_index == args.size() && binding.param_index < new_nonctx, span)
          << "Internal error: added output parameter positions do not precede the CommCtx suffix";
      auto storage = MakeStorage(callee->params_[binding.param_index], arguments, shapes.get(), span,
                                 local_index++, prefix);
      args.push_back(storage);
      arguments.Bind(callee->params_[binding.param_index], storage);
      if (shapes) shapes->BindOutput(callee->params_[binding.param_index], storage);
    }
    args.insert(args.end(), old_args.begin() + static_cast<std::ptrdiff_t>(supplied), old_args.end());
    const auto& attrs = call ? call->attrs_ : submit->attrs_;
    auto new_attrs = RewriteAttrs(attrs, supplied, old_args.size(), args.size(), span);
    // Return types, like newly allocated outputs, belong to this caller's
    // shape variables rather than the callee's scalar parameter identities.
    std::vector<TypePtr> types;
    types.reserve(callee->return_types_.size() + (submit ? 1 : 0));
    for (const auto& type : callee->return_types_) {
      types.push_back(shapes ? shapes->InstantiateType(type) : arguments.RemapType(type));
    }
    if (call) {
      TypePtr result_type = types.empty()       ? GetUnknownType()
                            : types.size() == 1 ? types[0]
                                                : std::make_shared<TupleType>(types);
      return std::make_shared<Call>(call->op_, std::move(args), call->kwargs_, std::move(new_attrs),
                                    result_type, span);
    }
    // Callee return arity is unchanged; retain Submit's extra TaskId position.
    TypePtr result_type = submit->GetType();
    if (auto tuple = As<TupleType>(result_type);
        tuple && tuple->types_.size() == callee->return_types_.size() + 1) {
      types.push_back(tuple->types_.back());
      result_type = std::make_shared<TupleType>(std::move(types));
    }
    return std::make_shared<Submit>(submit->op_, std::move(args), submit->deps_, submit->kwargs_,
                                    std::move(new_attrs), result_type, span, submit->core_num_,
                                    submit->sync_start_, submit->allow_early_resolve_, submit->predicate_);
  }

  const AddedOutputMap& added_outputs_;
  const std::unordered_map<std::string, FunctionPtr>& functions_;
  FunctionPtr wrapper_;
  const CallShapePlans& shape_plans_;
  std::vector<VarPtr> params_;
  std::vector<ParamDirection> directions_;
  AddedOutputBindings outputs_;
  size_t output_start_ = 0;
};

}  // namespace

WrapperTransformResult PropagateOutputsThroughWrapper(
    const FunctionPtr& func, const AddedOutputMap& added_outputs,
    const std::unordered_map<std::string, FunctionPtr>& functions, const CallShapePlans& shape_plans) {
  OutputParameterRewriter mutator(added_outputs, functions, shape_plans, func);
  auto body = mutator.VisitStmt(func->body_);
  if (body == func->body_ && mutator.outputs().empty()) return {func, {}};
  auto result = MutableCopy(func);
  result->params_ = mutator.params();
  result->param_directions_ = mutator.directions();
  result->body_ = body;
  // Internal output buffers never extend the user's logical return tuple.
  return {result, mutator.outputs()};
}

FunctionPtr UpdateCallSites(const FunctionPtr& func, const AddedOutputMap& added_outputs,
                            const std::unordered_map<std::string, FunctionPtr>& functions,
                            const CallShapePlans& shape_plans) {
  OutputParameterRewriter mutator(added_outputs, functions, shape_plans);
  auto body = mutator.VisitStmt(func->body_);
  if (body == func->body_) return func;
  auto result = MutableCopy(func);
  result->body_ = body;
  return result;
}

}  // namespace tensor_to_tile
}  // namespace ir
}  // namespace pypto
