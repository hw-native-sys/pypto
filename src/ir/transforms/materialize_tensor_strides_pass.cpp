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

/**
 * @file materialize_tensor_strides_pass.cpp
 * @brief MaterializeTensorStrides pass — RFC #1300 §2.4.
 *
 * Walks every TensorType reachable from a Program (function params, return
 * types, body Vars, IterArgs, Call return types, recursively into TupleType)
 * and rewrites any ``view.has_value() && view.stride.empty()`` slot to its
 * packed canonical form per ``BuildLogicalStridesFromLayout``. This is the
 * "codegen entry contract" preparation step: after this pass runs, every
 * TensorType that carries a TensorView has explicit stride matching the
 * layout / shape, which the strict-mode ``TensorViewCanonical`` verifier
 * (also produced by this pass) then enforces.
 *
 * Bare TensorTypes (``!view.has_value()``) are left untouched — they are
 * implicitly ND-packed and the verifier accepts them in both modes.
 *
 * The pass is idempotent: re-running it on already-canonical IR is a no-op.
 */

#include <map>
#include <memory>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pypto/core/logging.h"  // INTERNAL_CHECK; transitively pulls in error.h
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/transforms/utils/tensor_view_semantics.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

namespace {

/// Rewrite a TensorType (or recursively a TupleType containing TensorTypes)
/// so that any ``view.has_value() && view.stride.empty()`` slot is filled
/// with a packed canonical stride per ``BuildLogicalStridesFromLayout``.
///
/// Returns the input TypePtr unchanged when no rewrite is needed (so callers
/// can identity-compare to skip downstream Var/Call rebuilds).
TypePtr MaterializeType(const TypePtr& type) {
  if (!type) return type;

  if (auto tensor_type = As<TensorType>(type)) {
    if (!tensor_type->tensor_view_.has_value()) {
      // Bare tensor — no view to materialize.
      return type;
    }
    const TensorView& view = *tensor_type->tensor_view_;
    if (!view.stride.empty()) {
      // Already explicit.
      return type;
    }
    if (view.layout == TensorLayout::NZ) {
      // NZ on TensorType is rejected by the verifier; do not attempt to
      // materialize because BuildLogicalStridesFromLayout would CHECK-fail.
      return type;
    }
    auto materialized_stride =
        tensor_view_semantics::BuildLogicalStridesFromLayout(tensor_type->shape_, view.layout);
    TensorView new_view(std::move(materialized_stride), view.layout, view.valid_shape);
    return std::make_shared<TensorType>(tensor_type->shape_, tensor_type->dtype_, tensor_type->memref_,
                                        std::make_optional(std::move(new_view)));
  }

  if (auto tuple_type = As<TupleType>(type)) {
    std::vector<TypePtr> new_elems;
    new_elems.reserve(tuple_type->types_.size());
    bool changed = false;
    for (const auto& sub : tuple_type->types_) {
      auto new_sub = MaterializeType(sub);
      if (new_sub.get() != sub.get()) changed = true;
      new_elems.push_back(std::move(new_sub));
    }
    if (!changed) return type;
    return std::make_shared<TupleType>(std::move(new_elems));
  }

  return type;
}

/// Mutator that swaps Vars / Calls whose carried TypePtr changes after
/// ``MaterializeType``. Uses a Var-substitution cache so every reference to
/// a rebuilt Var resolves to the same new Var (consistent with how
/// TileMemorySpaceMutator handles its ``var_cache_``).
class MaterializeTensorStridesMutator : public IRMutator {
 public:
  /// Pre-populate the substitution cache with param/iter-arg Vars whose
  /// types were rebuilt at the function boundary. This way the body's
  /// references to those Vars resolve to the new Vars instead of the old.
  void AddSubstitution(const VarPtr& old_var, const VarPtr& new_var) { var_cache_[old_var] = new_var; }

 protected:
  ExprPtr VisitExpr_(const VarPtr& op) override {
    auto it = var_cache_.find(op);
    if (it != var_cache_.end()) {
      return it->second;
    }
    auto new_type = MaterializeType(op->GetType());
    if (new_type.get() == op->GetType().get()) {
      var_cache_[op] = op;
      return op;
    }
    auto new_var = std::make_shared<Var>(op->name_hint_, std::move(new_type), op->span_);
    var_cache_[op] = new_var;
    return new_var;
  }

  ExprPtr VisitExpr_(const IterArgPtr& op) override {
    auto it = var_cache_.find(op);
    if (it != var_cache_.end()) {
      return it->second;
    }
    auto new_init = IRMutator::VisitExpr(op->initValue_);
    auto new_type = MaterializeType(op->GetType());
    bool init_changed = new_init.get() != op->initValue_.get();
    bool type_changed = new_type.get() != op->GetType().get();
    if (!init_changed && !type_changed) {
      var_cache_[op] = op;
      return op;
    }
    auto new_iter_arg = std::make_shared<IterArg>(op->name_hint_, std::move(new_type), new_init, op->span_);
    var_cache_[op] = new_iter_arg;
    return new_iter_arg;
  }

  ExprPtr VisitExpr_(const CallPtr& op) override {
    // First recurse into args / kwargs via the default IRMutator path, then
    // patch the return type if needed. Rebuild via OpRegistry when possible
    // so deduce-type stays authoritative; fall back to a direct Call ctor
    // for unregistered ops / GlobalVar calls (mirrors TileMemorySpaceMutator).
    std::vector<ExprPtr> new_args;
    new_args.reserve(op->args_.size());
    bool args_changed = false;
    for (const auto& arg : op->args_) {
      auto new_arg = IRMutator::VisitExpr(arg);
      if (new_arg.get() != arg.get()) args_changed = true;
      new_args.push_back(std::move(new_arg));
    }

    auto new_return_type = MaterializeType(op->GetType());
    bool type_changed = new_return_type.get() != op->GetType().get();

    if (!args_changed && !type_changed) return op;

    auto& registry = OpRegistry::GetInstance();
    if (As<GlobalVar>(op->op_) || !registry.IsRegistered(op->op_->name_)) {
      // Direct ctor — must supply a result type. Prefer the materialized one
      // so downstream Vars / Calls see the explicit stride.
      return std::make_shared<Call>(op->op_, std::move(new_args), op->kwargs_, std::move(new_return_type),
                                    op->span_);
    }
    // OpRegistry rebuilds the Call's type via DeduceType. If the deduced type
    // is bare and the materialized type is more specific (carries explicit
    // stride), prefer the materialized one — but in practice most ops produce
    // bare TensorType and the deduction agrees with the input. To avoid
    // surprising layout regressions we just accept whatever the registry
    // returns; if it disagrees with the materialized form, the
    // TensorViewCanonical verifier (strict mode) will surface that as a
    // diagnostic so we know to add an explicit op rebuild here.
    return registry.Create(op->op_->name_, new_args, op->kwargs_, op->span_);
  }

  StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
    // Mirror TileMemorySpaceMutator's pattern: rebuild RHS first, then
    // ensure the LHS Var carries a matching type. If the RHS Call's
    // return type is more specific (materialized) than the LHS Var's
    // current type, sync the Var.
    auto new_var_expr = IRMutator::VisitExpr(op->var_);
    auto new_value = IRMutator::VisitExpr(op->value_);
    auto new_var = As<Var>(new_var_expr);
    // AssignStmt LHS is always a Var by construction; visiting it through the
    // mutator must yield a Var (or a substituted Var via var_cache_). If we
    // ever hit a non-Var here, that's a mutator-correctness bug — surface it
    // immediately rather than passing nullptr to the AssignStmt constructor.
    INTERNAL_CHECK(new_var) << "MaterializeTensorStrides: AssignStmt LHS visited to non-Var "
                            << "expression";

    if (auto new_call = As<Call>(new_value)) {
      auto rhs_type = new_call->GetType();
      if (rhs_type && rhs_type.get() != new_var->GetType().get()) {
        if (auto rhs_tensor = As<TensorType>(rhs_type)) {
          auto lhs_tensor = As<TensorType>(new_var->GetType());
          if (lhs_tensor && rhs_tensor.get() != lhs_tensor.get()) {
            auto synced_var = std::make_shared<Var>(new_var->name_hint_, rhs_type, new_var->span_);
            var_cache_[op->var_] = synced_var;
            new_var = synced_var;
          }
        }
      }
    }

    if (new_var.get() == op->var_.get() && new_value.get() == op->value_.get()) return op;
    return std::make_shared<AssignStmt>(new_var, new_value, op->span_);
  }

 private:
  std::unordered_map<VarPtr, VarPtr> var_cache_;
};

/// Materialize a single function: rebuild params / return_types / body.
/// Returns the input function unchanged when no rewrite is needed.
FunctionPtr TransformFunction(const FunctionPtr& func) {
  // 1) Rewrite param types. New params get fresh Var objects; record old→new
  //    so body references resolve to the rewritten Var.
  bool params_changed = false;
  std::vector<VarPtr> new_params;
  new_params.reserve(func->params_.size());
  std::unordered_map<VarPtr, VarPtr> param_substitutions;
  for (const auto& old_param : func->params_) {
    auto new_type = MaterializeType(old_param->GetType());
    if (new_type.get() == old_param->GetType().get()) {
      new_params.push_back(old_param);
      continue;
    }
    auto new_param = std::make_shared<Var>(old_param->name_hint_, std::move(new_type), old_param->span_);
    new_params.push_back(new_param);
    param_substitutions.emplace(old_param, new_param);
    params_changed = true;
  }

  // 2) Rewrite return_types in place (no Var indirection needed here).
  bool returns_changed = false;
  std::vector<TypePtr> new_return_types;
  new_return_types.reserve(func->return_types_.size());
  for (const auto& rt : func->return_types_) {
    auto new_rt = MaterializeType(rt);
    if (new_rt.get() != rt.get()) returns_changed = true;
    new_return_types.push_back(std::move(new_rt));
  }

  // 3) Walk body. Pre-populate the Var cache with param substitutions so
  //    body references to params resolve to the new Vars.
  MaterializeTensorStridesMutator mutator;
  for (const auto& [old_var, new_var] : param_substitutions) {
    mutator.AddSubstitution(old_var, new_var);
  }
  StmtPtr new_body = func->body_;
  if (func->body_) {
    new_body = mutator.VisitStmt(func->body_);
  }

  bool body_changed = new_body.get() != func->body_.get();
  if (!params_changed && !returns_changed && !body_changed) {
    return func;
  }

  auto new_func = MutableCopy(func);
  if (params_changed) new_func->params_ = std::move(new_params);
  if (returns_changed) new_func->return_types_ = std::move(new_return_types);
  if (body_changed) new_func->body_ = std::move(new_body);
  return new_func;
}

}  // namespace

namespace pass {

Pass MaterializeTensorStrides() {
  auto pass_func = [](const ProgramPtr& program) -> ProgramPtr {
    bool modified = false;
    std::map<GlobalVarPtr, FunctionPtr, GlobalVarPtrLess> new_functions;
    for (const auto& [gvar, func] : program->functions_) {
      auto new_func = TransformFunction(func);
      if (new_func.get() != func.get()) modified = true;
      new_functions[gvar] = std::move(new_func);
    }
    if (!modified) return program;
    return std::make_shared<Program>(std::move(new_functions), program->name_, program->span_);
  };
  return CreateProgramPass(pass_func, "MaterializeTensorStrides", kMaterializeTensorStridesProperties);
}

}  // namespace pass

}  // namespace ir
}  // namespace pypto
