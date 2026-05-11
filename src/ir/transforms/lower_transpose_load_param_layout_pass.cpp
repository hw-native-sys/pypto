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
#include <map>
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
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
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

/// Scans an InCore function body for ``tile.load(param, ..., transpose=True)``
/// where the source tensor is a function parameter.
class TransposeLoadScanner : public IRVisitor {
 public:
  explicit TransposeLoadScanner(const std::vector<VarPtr>& params) {
    for (size_t i = 0; i < params.size(); ++i) {
      param_ptr_to_index_[params[i].get()] = i;
    }
  }

  // Returns the set of param indices that need DN promotion.
  const std::unordered_set<size_t>& GetPromoted() const { return promoted_; }

  // Returns the set of param indices whose `tile.load` calls all carry
  // `transpose=False` (or absent). Used to reject mixed-use parameters.
  const std::unordered_set<size_t>& GetNonTransposedUses() const { return non_transposed_uses_; }

  void VisitExpr_(const CallPtr& call) override {
    if (call && call->op_ && call->op_->name_ == "tile.load" && !call->args_.empty()) {
      auto src_var = As<Var>(call->args_[0]);
      if (src_var) {
        auto it = param_ptr_to_index_.find(src_var.get());
        if (it != param_ptr_to_index_.end()) {
          const size_t param_idx = it->second;
          bool transpose = call->GetKwarg<bool>("transpose", false);
          if (transpose) {
            promoted_.insert(param_idx);
          } else {
            non_transposed_uses_.insert(param_idx);
          }
        }
      }
    }
    IRVisitor::VisitExpr_(call);
  }

 private:
  std::unordered_map<const Var*, size_t> param_ptr_to_index_;
  std::unordered_set<size_t> promoted_;
  std::unordered_set<size_t> non_transposed_uses_;
};

/// Build the canonical TensorType for an InCore parameter that is loaded via
/// ``tile.load(transpose=True)`` (RFC #1300 §3.3 + §4.2):
///   src ``[..., a, b] ND`` ≡ canonical ``[..., b, a] DN``
///
/// The new TensorView carries an empty stride; ``MaterializeTensorStrides``
/// (P6-b) fills it with the packed canonical strides later in the pipeline.
TensorTypePtr PromoteToCanonicalDN(const TensorTypePtr& src) {
  CHECK(src->shape_.size() >= 2)
      << "LowerTransposeLoadParamLayout: parameter must have rank >= 2 to apply DN "
         "canonical form, got "
      << src->shape_.size();
  std::vector<ExprPtr> new_shape = src->shape_;
  std::iter_swap(new_shape.end() - 2, new_shape.end() - 1);
  TensorView dn_view(std::vector<ExprPtr>{}, TensorLayout::DN);
  return std::make_shared<TensorType>(new_shape, src->dtype_, src->memref_,
                                      std::make_optional(std::move(dn_view)));
}

/// Swap the last two elements of a ``MakeTuple`` (offsets / shapes /
/// valid_shapes argument of ``tile.load``).
MakeTuplePtr SwapTrailingPair(const MakeTuplePtr& tuple) {
  INTERNAL_CHECK(tuple) << "Internal error: SwapTrailingPair called with null MakeTuple";
  INTERNAL_CHECK_SPAN(tuple->elements_.size() >= 2, tuple->span_)
      << "LowerTransposeLoadParamLayout: tile.load tuple needs rank >= 2 to swap "
         "trailing pair, got "
      << tuple->elements_.size();
  std::vector<ExprPtr> new_elements = tuple->elements_;
  std::iter_swap(new_elements.end() - 2, new_elements.end() - 1);
  return std::make_shared<MakeTuple>(std::move(new_elements), tuple->span_);
}

/// Rewrite tile.load calls whose first arg is one of the promoted parameters
/// so that:
///   - offsets / shapes / valid_shapes are swapped to canonical coords;
///   - the ``transpose=True`` kwarg is dropped (DN source + Mat target now
///     drives the tile-view swap inside ``DeduceTileLoadType``).
/// All other Calls are passed through unchanged.
class TileLoadBodyRewriter : public IRMutator {
 public:
  explicit TileLoadBodyRewriter(const std::unordered_map<const Var*, VarPtr>& param_subs) {
    for (const auto& [old_ptr, new_var] : param_subs) {
      promoted_param_set_.insert(new_var.get());
    }
  }

  ExprPtr VisitExpr_(const CallPtr& op) override {
    auto base = IRMutator::VisitExpr_(op);
    auto call = std::dynamic_pointer_cast<const Call>(base);
    if (!call || !call->op_ || call->op_->name_ != "tile.load") return base;
    if (call->args_.empty()) return base;

    auto src_var = As<Var>(call->args_[0]);
    if (!src_var || promoted_param_set_.find(src_var.get()) == promoted_param_set_.end()) {
      return base;
    }
    if (!call->GetKwarg<bool>("transpose", false)) return base;

    // tile.load(tensor, offsets, shapes, valid_shapes, ...) — swap the trailing
    // pair of all three tuples so the load is expressed in canonical (DN
    // logical) coordinates that match the promoted parameter's new shape.
    INTERNAL_CHECK_SPAN(call->args_.size() == 4, call->span_)
        << "LowerTransposeLoadParamLayout: expected tile.load to have 4 args, got " << call->args_.size();
    auto offsets = As<MakeTuple>(call->args_[1]);
    auto shapes = As<MakeTuple>(call->args_[2]);
    auto valid_shapes = As<MakeTuple>(call->args_[3]);
    INTERNAL_CHECK_SPAN(offsets && shapes && valid_shapes, call->span_)
        << "LowerTransposeLoadParamLayout: tile.load offsets/shapes/valid_shapes must be MakeTuple";

    std::vector<ExprPtr> new_args = call->args_;
    new_args[1] = SwapTrailingPair(offsets);
    new_args[2] = SwapTrailingPair(shapes);
    new_args[3] = SwapTrailingPair(valid_shapes);

    // Flip transpose=True → transpose=False; the DN-source + Mat-target signal
    // is now carried entirely by the source TensorType's layout tag, but the
    // kwarg slot is kept so print → reparse round-trips faithfully (the
    // tile.load op registers ``transpose`` as a default-false attribute and
    // the parser injects it back on reparse).
    std::vector<std::pair<std::string, std::any>> new_kwargs;
    new_kwargs.reserve(call->kwargs_.size());
    for (const auto& [k, v] : call->kwargs_) {
      if (k == "transpose") {
        new_kwargs.emplace_back(k, std::any(false));
      } else {
        new_kwargs.emplace_back(k, v);
      }
    }

    // Rebuild via OpRegistry so DeduceTileLoadType recomputes the TileType
    // from the new source layout (DN) + swapped shapes.
    return OpRegistry::GetInstance().Create("tile.load", new_args, new_kwargs, call->span_);
  }

 private:
  std::unordered_set<const Var*> promoted_param_set_;
};

/// Result of promoting a single InCore function.
struct PromotionResult {
  FunctionPtr func;
  std::map<size_t, VarPtr> promoted_params;  // param index → new param Var
};

/// Promote an InCore function. Returns the rewritten Function (or the
/// original if no rewrite was needed) and the map of promoted param slots.
/// Throws if any promoted parameter is also loaded without `transpose=True`
/// in the same body (mixed use would corrupt non-transpose loads).
PromotionResult PromoteInCoreFunction(const FunctionPtr& func) {
  TransposeLoadScanner scanner(func->params_);
  scanner.VisitStmt(func->body_);
  const auto& promoted = scanner.GetPromoted();
  const auto& non_transposed = scanner.GetNonTransposedUses();
  if (promoted.empty()) {
    return {func, {}};
  }

  std::unordered_map<const Var*, VarPtr> substitutions;
  std::vector<VarPtr> new_params = func->params_;
  std::map<size_t, VarPtr> promoted_params;

  for (size_t idx : promoted) {
    // Mixed-use rejection: a param promoted from `[a, b]` ND → `[b, a]` DN
    // would invalidate every non-transpose `tile.load(p, ...)` that still
    // expects the original coordinate system.
    CHECK(non_transposed.find(idx) == non_transposed.end())
        << "LowerTransposeLoadParamLayout: parameter at index " << idx
        << " is loaded both with transpose=True and transpose=False — only one "
           "mode is supported per InCore parameter. Split the parameter or unify "
           "the load direction.";

    const auto& old_param = func->params_[idx];
    auto old_tensor_type = As<TensorType>(old_param->GetType());
    CHECK(old_tensor_type) << "LowerTransposeLoadParamLayout: promoted parameter at index " << idx
                           << " must be TensorType";

    // Reject the (DN view + explicit physical stride) combination — these
    // came from `tensor.transpose` and would compose with the load-side
    // transpose to produce a double-encoded transpose.
    if (old_tensor_type->tensor_view_.has_value()) {
      const auto& view = old_tensor_type->tensor_view_.value();
      CHECK(!(view.layout == TensorLayout::DN && !view.stride.empty()))
          << "LowerTransposeLoadParamLayout: tile.load(transpose=True) on a "
             "tensor.transpose result is not supported (the DN tag and explicit "
             "physical strides would compose as a double transpose). Drop one of "
             "the two transpose layers in the source program.";
      // Param already promoted in a prior round (idempotent): skip.
      if (view.layout == TensorLayout::DN) continue;
    }

    auto new_tensor_type = PromoteToCanonicalDN(old_tensor_type);
    auto new_var = std::make_shared<Var>(old_param->name_hint_, new_tensor_type, old_param->span_);
    new_params[idx] = new_var;
    substitutions[old_param.get()] = new_var;
    promoted_params.emplace(idx, new_var);
  }

  if (substitutions.empty()) {
    return {func, {}};
  }

  // 1) Substitute param Vars in the body.
  auto subbed_body = Substitute(func->body_, substitutions);

  // 2) Rewrite each `tile.load(promoted_param, ..., transpose=True)` in the
  //    body — swap offsets / shapes / valid_shapes trailing pair, drop the
  //    transpose kwarg.
  TileLoadBodyRewriter body_rewriter(substitutions);
  auto new_body = body_rewriter.VisitStmt(subbed_body);

  auto new_func = MutableCopy(func);
  new_func->params_ = new_params;
  new_func->body_ = new_body;
  return {new_func, promoted_params};
}

/// Walks every non-InCore function in the program and, for each call site
/// targeting a promoted InCore callee, emits an SSA-form binding for each
/// promoted-slot arg:
///
///   bridged_<param> = tensor.as_layout(<orig_arg>, DN)
///   <orig_lhs> = <callee>(..., bridged_<param>, ...)
///
/// The binding is emitted as a separate ``AssignStmt`` immediately before the
/// call statement (instead of being inlined inside the call's args), which is
/// what downstream orchestration codegen expects — it consumes a ``Var`` or a
/// constant literal per call arg, not a nested ``Call``.
class CallSiteAsLayoutInjector : public IRMutator {
 public:
  explicit CallSiteAsLayoutInjector(const std::map<std::string, std::map<size_t, VarPtr>>& promotions)
      : promotions_(promotions) {}

  StmtPtr VisitStmt_(const SeqStmtsPtr& op) override {
    std::vector<StmtPtr> new_stmts;
    new_stmts.reserve(op->stmts_.size());
    bool any_changed = false;
    for (const auto& stmt : op->stmts_) {
      // Recurse into nested SeqStmts / control-flow first so inner call sites
      // get patched too.
      auto recursed = IRMutator::VisitStmt(stmt);
      bool inserted = false;
      auto patched = MaybeInjectBindings(recursed, new_stmts, &inserted);
      if (inserted || patched.get() != recursed.get() || recursed.get() != stmt.get()) {
        any_changed = true;
      }
      new_stmts.push_back(patched);
    }
    if (!any_changed) return op;
    return SeqStmts::Flatten(std::move(new_stmts), op->span_);
  }

  // Bare (non-SeqStmts) statement bodies — e.g. ``then_body`` of an ``IfStmt``
  // that contains a single ``AssignStmt``. Wrap any injected bindings into
  // a fresh SeqStmts so the resulting body stays a single Stmt.
  StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
    auto recursed = IRMutator::VisitStmt_(op);
    std::vector<StmtPtr> pre;
    bool inserted = false;
    auto patched = MaybeInjectBindings(recursed, pre, &inserted);
    if (!inserted) return patched;
    pre.push_back(patched);
    return SeqStmts::Flatten(std::move(pre), op->span_);
  }

  StmtPtr VisitStmt_(const EvalStmtPtr& op) override {
    auto recursed = IRMutator::VisitStmt_(op);
    std::vector<StmtPtr> pre;
    bool inserted = false;
    auto patched = MaybeInjectBindings(recursed, pre, &inserted);
    if (!inserted) return patched;
    pre.push_back(patched);
    return SeqStmts::Flatten(std::move(pre), op->span_);
  }

  StmtPtr VisitStmt_(const ReturnStmtPtr& op) override {
    auto recursed = IRMutator::VisitStmt_(op);
    std::vector<StmtPtr> pre;
    bool inserted = false;
    auto patched = MaybeInjectBindings(recursed, pre, &inserted);
    if (!inserted) return patched;
    pre.push_back(patched);
    return SeqStmts::Flatten(std::move(pre), op->span_);
  }

 private:
  /// If ``stmt``'s RHS is a Call to a promoted callee, build the binding
  /// AssignStmts (one per promoted slot) and emit them into ``pre``;
  /// rewrite the Call to reference the bound Vars. Returns the (possibly
  /// rewritten) statement and sets ``*inserted = true`` if any bindings
  /// were added.
  StmtPtr MaybeInjectBindings(const StmtPtr& stmt, std::vector<StmtPtr>& pre, bool* inserted) {
    auto extract_call = [](const StmtPtr& s) -> std::pair<CallPtr, VarPtr> {
      if (auto assign = As<AssignStmt>(s)) {
        return {As<Call>(assign->value_), assign->var_};
      }
      if (auto eval = As<EvalStmt>(s)) {
        return {As<Call>(eval->expr_), nullptr};
      }
      if (auto ret = As<ReturnStmt>(s)) {
        if (ret->value_.size() == 1) {
          return {As<Call>(ret->value_[0]), nullptr};
        }
      }
      return {nullptr, nullptr};
    };

    auto [call, lhs_var] = extract_call(stmt);
    if (!call) return stmt;
    auto gv = As<GlobalVar>(call->op_);
    if (!gv) return stmt;
    auto it = promotions_.find(gv->name_);
    if (it == promotions_.end() || it->second.empty()) return stmt;
    const auto& slots = it->second;

    std::vector<ExprPtr> new_args = call->args_;
    bool changed = false;
    for (const auto& [idx, new_param_var] : slots) {
      INTERNAL_CHECK_SPAN(idx < new_args.size(), call->span_)
          << "LowerTransposeLoadParamLayout: promoted param index " << idx << " out of range for call to "
          << gv->name_;
      auto arg = new_args[idx];
      auto arg_tensor = As<TensorType>(arg->GetType());
      if (!arg_tensor) continue;
      // Idempotency: an arg already in DN form needs no bridge.
      if (arg_tensor->tensor_view_.has_value() && arg_tensor->tensor_view_->layout == TensorLayout::DN) {
        continue;
      }
      // Build the bridge: bridged = tensor.as_layout(arg, DN).
      std::vector<std::pair<std::string, std::any>> kwargs = {{"layout", std::any(TensorLayout::DN)}};
      auto bridge_call = OpRegistry::GetInstance().Create("tensor.as_layout", {arg}, kwargs, arg->span_);
      auto bridge_var =
          std::make_shared<Var>(new_param_var->name_hint_ + "_dn_view", bridge_call->GetType(), arg->span_);
      pre.push_back(std::make_shared<AssignStmt>(bridge_var, bridge_call, arg->span_));
      new_args[idx] = bridge_var;
      changed = true;
    }
    if (!changed) return stmt;
    *inserted = true;

    auto new_call = std::make_shared<Call>(call->op_, std::move(new_args), call->kwargs_, call->attrs_,
                                           call->GetType(), call->span_);
    if (auto assign = As<AssignStmt>(stmt)) {
      return std::make_shared<AssignStmt>(assign->var_, new_call, assign->span_);
    }
    if (auto eval = As<EvalStmt>(stmt)) {
      return std::make_shared<EvalStmt>(new_call, eval->span_);
    }
    if (auto ret = As<ReturnStmt>(stmt)) {
      return std::make_shared<ReturnStmt>(std::vector<ExprPtr>{new_call}, ret->span_);
    }
    return stmt;  // unreachable — extract_call only returns non-null for the three above
  }

  const std::map<std::string, std::map<size_t, VarPtr>>& promotions_;
};

}  // namespace

namespace pass {

Pass LowerTransposeLoadParamLayout() {
  auto pass_func = [](const ProgramPtr& program) -> ProgramPtr {
    // Phase 1: rewrite InCore functions and collect promotion info keyed by
    // callee name (callers reference InCore functions through Call->op_'s
    // GlobalVar, which is matched on name_).
    std::map<GlobalVarPtr, FunctionPtr, GlobalVarPtrLess> new_functions;
    std::map<std::string, std::map<size_t, VarPtr>> promotions_by_callee_name;
    bool modified = false;

    for (const auto& [gvar, func] : program->functions_) {
      if (!IsInCoreType(func->func_type_)) {
        new_functions[gvar] = func;
        continue;
      }
      auto result = PromoteInCoreFunction(func);
      new_functions[gvar] = result.func;
      if (result.func.get() != func.get()) modified = true;
      if (!result.promoted_params.empty()) {
        promotions_by_callee_name[gvar->name_] = std::move(result.promoted_params);
      }
    }

    if (promotions_by_callee_name.empty()) {
      return modified ? std::make_shared<Program>(std::move(new_functions), program->name_, program->span_)
                      : program;
    }

    // Phase 2: walk non-InCore functions and inject `tensor.as_layout` at
    // each call site that targets a promoted callee.
    CallSiteAsLayoutInjector injector(promotions_by_callee_name);
    for (auto& [gvar, func] : new_functions) {
      if (IsInCoreType(func->func_type_)) continue;
      if (!func->body_) continue;
      auto new_body = injector.VisitStmt(func->body_);
      if (new_body.get() != func->body_.get()) {
        auto new_func = MutableCopy(func);
        new_func->body_ = new_body;
        new_functions[gvar] = new_func;
        modified = true;
      }
    }

    if (!modified) return program;
    return std::make_shared<Program>(std::move(new_functions), program->name_, program->span_);
  };

  return CreateProgramPass(pass_func, "LowerTransposeLoadParamLayout",
                           kLowerTransposeLoadParamLayoutProperties);
}

}  // namespace pass

}  // namespace ir
}  // namespace pypto
