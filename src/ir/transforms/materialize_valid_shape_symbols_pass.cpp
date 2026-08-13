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
#include <map>
#include <memory>
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
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/printer.h"
#include "pypto/ir/type.h"
#include "pypto/ir/type_inference.h"

namespace pypto {
namespace ir {

namespace {

/// A device kernel's valid_shape symbol that nothing in the kernel can bind,
/// together with the declared parameter slots that name it.
///
/// `symbol` is the `pl.dynamic()` Var itself. `DynVar.unwrap()` already builds it
/// as a `Scalar[INDEX]` Var shared by every annotation that mentions it, so
/// appending it to `params_` binds every occurrence at once — no type rewrite.
struct ValidShapeSymbol {
  VarPtr symbol;
  /// (tensor param index, position within that parameter's valid_shape). Every
  /// entry names the symbol *bare*, so the call site reads the actual extent
  /// straight out of the argument's valid_shape at the same position.
  std::vector<std::pair<size_t, size_t>> slots;
};

struct FunctionValidShapePlan {
  std::vector<ValidShapeSymbol> symbols;  ///< deterministic: declaration order
};

/// PTO codegen compiles these once and launches them from orchestration, so a
/// symbol they do not receive as an argument has no value at run time.
[[nodiscard]] bool IsDeviceKernel(const FunctionPtr& func) {
  if (!func) return false;
  switch (func->func_type_) {
    case FunctionType::InCore:
    case FunctionType::AIC:
    case FunctionType::AIV:
    case FunctionType::Spmd:
      return true;
    default:
      return false;
  }
}

/// Collect the Vars a shape/valid_shape expression reads, in first-seen order.
void CollectShapeVars(const ExprPtr& expr, std::unordered_set<const Var*>* seen, std::vector<VarPtr>* out) {
  if (!expr) return;
  if (auto var = AsVarLike(expr)) {
    if (seen->insert(var.get()).second) out->push_back(var);
    return;
  }
  if (auto binary = As<BinaryExpr>(expr)) {
    CollectShapeVars(binary->left_, seen, out);
    CollectShapeVars(binary->right_, seen, out);
    return;
  }
  if (auto unary = As<UnaryExpr>(expr)) {
    CollectShapeVars(unary->operand_, seen, out);
    return;
  }
  if (auto call = As<Call>(expr)) {
    for (const auto& arg : call->args_) CollectShapeVars(arg, seen, out);
    return;
  }
  if (auto tget = As<TupleGetItemExpr>(expr)) {
    CollectShapeVars(tget->tuple_, seen, out);
  }
  // Constants and anything else contribute no symbol.
}

/// Symbols the kernel can already resolve: physical tensor dimensions (the
/// wrapper recovers those from the runtime tensor's `shapes[]`) and its own
/// scalar parameters.
[[nodiscard]] std::unordered_set<const Var*> CollectBoundSymbols(const FunctionPtr& func) {
  std::unordered_set<const Var*> bound;
  for (const auto& param : func->params_) {
    auto tensor_type = AsTensorTypeLike(param->GetType());
    if (!tensor_type) {
      bound.insert(param.get());
      continue;
    }
    std::unordered_set<const Var*> seen;
    std::vector<VarPtr> vars;
    for (const auto& dim : tensor_type->shape_) CollectShapeVars(dim, &seen, &vars);
    for (const auto& var : vars) bound.insert(var.get());
  }
  return bound;
}

/// The valid_shape the author declared on a parameter, or nullptr when the
/// parameter is fully valid. Deliberately NOT GetEffectiveTensorValidShape:
/// that substitutes the physical shape, whose symbols are already bound.
[[nodiscard]] const std::vector<ExprPtr>* GetDeclaredValidShape(const ExprPtr& /*unused*/,
                                                                const TensorTypePtr& tensor_type) {
  if (!tensor_type->tensor_view_.has_value()) return nullptr;
  const auto& valid_shape = tensor_type->tensor_view_->valid_shape;
  if (valid_shape.empty()) return nullptr;
  return &valid_shape;
}

[[nodiscard]] FunctionValidShapePlan BuildPlan(const FunctionPtr& func) {
  FunctionValidShapePlan plan;
  if (!IsDeviceKernel(func)) return plan;

  const auto bound = CollectBoundSymbols(func);

  // Pass 1: every parameter slot that names an unbound symbol *bare* is a place
  // the call site can read the value from.
  std::unordered_map<const Var*, size_t> symbol_index;
  for (size_t i = 0; i < func->params_.size(); ++i) {
    auto tensor_type = AsTensorTypeLike(func->params_[i]->GetType());
    if (!tensor_type) continue;
    const auto* valid_shape = GetDeclaredValidShape(func->params_[i], tensor_type);
    if (valid_shape == nullptr) continue;
    for (size_t d = 0; d < valid_shape->size(); ++d) {
      auto var = AsVarLike((*valid_shape)[d]);
      if (!var || bound.count(var.get()) != 0) continue;
      auto it = symbol_index.find(var.get());
      if (it == symbol_index.end()) {
        symbol_index.emplace(var.get(), plan.symbols.size());
        plan.symbols.push_back(ValidShapeSymbol{var, {{i, d}}});
      } else {
        plan.symbols[it->second].slots.emplace_back(i, d);
      }
    }
  }

  // Pass 2: a symbol reachable only through a compound slot has no bare slot to
  // read, so the call site cannot invert it. Reject rather than emit a kernel
  // whose valid extent is silently wrong.
  for (size_t i = 0; i < func->params_.size(); ++i) {
    auto tensor_type = AsTensorTypeLike(func->params_[i]->GetType());
    if (!tensor_type) continue;
    const auto* valid_shape = GetDeclaredValidShape(func->params_[i], tensor_type);
    if (valid_shape == nullptr) continue;
    for (const auto& dim : *valid_shape) {
      if (AsVarLike(dim)) continue;  // bare symbols handled above
      std::unordered_set<const Var*> seen;
      std::vector<VarPtr> vars;
      CollectShapeVars(dim, &seen, &vars);
      for (const auto& var : vars) {
        if (bound.count(var.get()) != 0) continue;
        CHECK_SPAN(symbol_index.count(var.get()) != 0, func->params_[i]->span_)
            << "MaterializeValidShapeSymbols: parameter '" << func->params_[i]->name_hint_ << "' of kernel '"
            << func->name_ << "' uses symbol '" << var->name_hint_
            << "' only inside the compound valid_shape expression '" << PythonPrint(dim)
            << "', so its value cannot be read from the call site. Declare the symbol on its own in "
               "some parameter's pl.TensorView(valid_shape=...), or pass the extent as a "
               "pl.Scalar[pl.INDEX] parameter and use it in pl.load(..., valid_shape=[...]).";
      }
    }
  }
  return plan;
}

/// Add one `Scalar[INDEX]` parameter per unbindable symbol, in plan order.
///
/// The new parameters go at the FRONT. A symbol is read by the very parameter
/// annotations that name it (`a: Tensor[..., valid_shape=[M, VALID]]`), and the
/// text form declares parameters left to right — appending would print a
/// signature that uses `VALID` before declaring it, which does not re-parse.
/// Signature order is otherwise free: PTOParam dispatches args as
/// [tensors..., scalars...] regardless of it (see PTOCodegen::GenerateFunction).
[[nodiscard]] FunctionPtr ExtendFunctionSignature(const FunctionPtr& func,
                                                  const FunctionValidShapePlan& plan) {
  if (plan.symbols.empty()) return func;
  std::vector<VarPtr> params;
  std::vector<ParamDirection> dirs;
  params.reserve(func->params_.size() + plan.symbols.size());
  dirs.reserve(func->param_directions_.size() + plan.symbols.size());
  for (const auto& symbol : plan.symbols) {
    params.push_back(symbol.symbol);
    dirs.push_back(ParamDirection::In);
  }
  params.insert(params.end(), func->params_.begin(), func->params_.end());
  dirs.insert(dirs.end(), func->param_directions_.begin(), func->param_directions_.end());
  return std::make_shared<Function>(func->name_, std::move(params), std::move(dirs), func->return_types_,
                                    func->body_, func->span_, func->func_type_, func->level_, func->role_,
                                    func->attrs_, func->requires_runtime_binding_);
}

/// Mirror ExtendFunctionSignature's leading insertion on the call's directions.
[[nodiscard]] std::vector<ArgDirection> PrependScalarArgDirections(const std::vector<ArgDirection>& old_dirs,
                                                                   size_t count) {
  std::vector<ArgDirection> dirs(count, ArgDirection::Scalar);
  dirs.insert(dirs.end(), old_dirs.begin(), old_dirs.end());
  return dirs;
}

/// Rewrites every call/submit of a planned kernel to pass the actual extents.
class MaterializeValidShapeSymbolsMutator : public IRMutator {
 public:
  MaterializeValidShapeSymbolsMutator(ProgramPtr program,
                                      const std::unordered_map<std::string, FunctionValidShapePlan>& plans)
      : program_(std::move(program)), plans_(plans) {}

 protected:
  ExprPtr VisitExpr_(const CallPtr& op) override {
    auto base = IRMutator::VisitExpr_(op);
    auto call = As<Call>(base);
    if (!call) return base;
    const auto* plan = LookupPlan(call->op_);
    if (plan == nullptr) return call;

    auto new_args = PrependValidShapeArgs(*plan, call->args_, call->span_);
    auto attrs = call->attrs_;
    if (call->HasArgDirections()) {
      attrs = WithArgDirectionsAttr(
          std::move(attrs), PrependScalarArgDirections(call->GetArgDirections(), plan->symbols.size()));
    }
    return std::make_shared<Call>(call->op_, std::move(new_args), call->kwargs_, std::move(attrs),
                                  call->GetType(), call->span_);
  }

  // Submit is a sibling call-like kind (pl.submit inside pl.manual_scope); a
  // kernel launched there needs the same trailing extents.
  ExprPtr VisitExpr_(const SubmitPtr& op) override {
    auto base = IRMutator::VisitExpr_(op);
    auto submit = As<Submit>(base);
    if (!submit) return base;
    const auto* plan = LookupPlan(submit->op_);
    if (plan == nullptr) return submit;

    auto new_args = PrependValidShapeArgs(*plan, submit->args_, submit->span_);
    auto attrs = submit->attrs_;
    if (submit->HasArgDirections()) {
      attrs = WithArgDirectionsAttr(
          std::move(attrs), PrependScalarArgDirections(submit->GetArgDirections(), plan->symbols.size()));
    }
    return std::make_shared<Submit>(submit->op_, std::move(new_args), submit->deps_, submit->kwargs_,
                                    std::move(attrs), submit->GetType(), submit->span_, submit->core_num_,
                                    submit->sync_start_, submit->allow_early_resolve_, submit->predicate_);
  }

 private:
  [[nodiscard]] const FunctionValidShapePlan* LookupPlan(const OpPtr& op) const {
    auto gvar = As<GlobalVar>(op);
    if (!gvar || !program_) return nullptr;
    auto callee = program_->GetFunction(gvar->name_);
    if (!callee) return nullptr;
    auto it = plans_.find(callee->name_);
    return it == plans_.end() ? nullptr : &it->second;
  }

  /// Read each symbol's value out of the matching actual argument's valid_shape,
  /// and place the values at the front to mirror ExtendFunctionSignature.
  [[nodiscard]] std::vector<ExprPtr> PrependValidShapeArgs(const FunctionValidShapePlan& plan,
                                                           const std::vector<ExprPtr>& old_args,
                                                           const Span& span) const {
    std::vector<ExprPtr> leading;
    leading.reserve(plan.symbols.size());
    for (const auto& symbol : plan.symbols) {
      ExprPtr value = nullptr;
      for (const auto& [param_idx, dim_idx] : symbol.slots) {
        // Submit passes a positional prefix of the callee params, so a slot past
        // the supplied args has no actual to read.
        CHECK_SPAN(param_idx < old_args.size(), span)
            << "MaterializeValidShapeSymbols: call does not supply argument " << param_idx
            << ", which declares valid_shape symbol '" << symbol.symbol->name_hint_ << "'";
        auto tensor_type = AsTensorTypeLike(old_args[param_idx]->GetType());
        CHECK_SPAN(tensor_type, span)
            << "MaterializeValidShapeSymbols: argument " << param_idx << " for valid_shape symbol '"
            << symbol.symbol->name_hint_ << "' must be a tensor, got "
            << old_args[param_idx]->GetType()->TypeName();
        const auto& actual_valid = GetEffectiveTensorValidShape(*tensor_type);
        CHECK_SPAN(dim_idx < actual_valid.size(), span)
            << "MaterializeValidShapeSymbols: argument " << param_idx << " has valid_shape rank "
            << actual_valid.size() << ", too short to supply '" << symbol.symbol->name_hint_
            << "' at position " << dim_idx;
        const ExprPtr& candidate = actual_valid[dim_idx];
        // The same symbol declared in two slots must be given one value; binding
        // it from the first slot alone would silently drop the disagreement.
        CHECK_SPAN(value == nullptr || AreExprsEqual(value, candidate), span)
            << "MaterializeValidShapeSymbols: symbol '" << symbol.symbol->name_hint_
            << "' is declared in more than one parameter's valid_shape, but this call supplies "
               "different extents ('"
            << PythonPrint(value) << "' and '" << PythonPrint(candidate)
            << "'). Pass arguments whose valid_shape agrees at those positions.";
        if (value == nullptr) value = candidate;
      }
      INTERNAL_CHECK_SPAN(value, span)
          << "Internal error: valid_shape symbol '" << symbol.symbol->name_hint_ << "' has no source slot";
      leading.push_back(value);
    }
    leading.insert(leading.end(), old_args.begin(), old_args.end());
    return leading;
  }

  ProgramPtr program_;
  const std::unordered_map<std::string, FunctionValidShapePlan>& plans_;
};

[[nodiscard]] ProgramPtr TransformProgram(const ProgramPtr& program) {
  std::unordered_map<std::string, FunctionValidShapePlan> plans;
  for (const auto& [gvar, func] : program->functions_) {
    auto plan = BuildPlan(func);
    if (!plan.symbols.empty()) plans.emplace(func->name_, std::move(plan));
  }
  if (plans.empty()) return program;

  std::map<GlobalVarPtr, FunctionPtr, GlobalVarPtrLess> extended;
  for (const auto& [gvar, func] : program->functions_) {
    auto it = plans.find(func->name_);
    extended[gvar] = (it == plans.end()) ? func : ExtendFunctionSignature(func, it->second);
  }
  auto with_signatures = std::make_shared<Program>(std::move(extended), program->name_, program->span_);

  MaterializeValidShapeSymbolsMutator mutator(with_signatures, plans);
  return mutator.VisitProgram(with_signatures);
}

}  // namespace

namespace pass {

Pass MaterializeValidShapeSymbols() {
  auto pass_func = [](const ProgramPtr& program) -> ProgramPtr { return TransformProgram(program); };
  return CreateProgramPass(pass_func, "MaterializeValidShapeSymbols",
                           kMaterializeValidShapeSymbolsProperties);
}

}  // namespace pass
}  // namespace ir
}  // namespace pypto
