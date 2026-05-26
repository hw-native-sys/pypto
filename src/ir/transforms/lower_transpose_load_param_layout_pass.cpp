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
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

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

  void VisitExpr_(const CallPtr& call) override {
    if (call && call->op_ && call->op_->name_ == "tile.load" && !call->args_.empty()) {
      auto src_var = As<Var>(call->args_[0]);
      if (src_var) {
        auto it = param_ptr_to_index_.find(src_var.get());
        if (it != param_ptr_to_index_.end() && call->GetKwarg<bool>("transpose", false)) {
          promoted_.insert(it->second);
        }
      }
    }
    IRVisitor::VisitExpr_(call);
  }

 private:
  std::unordered_map<const Var*, size_t> param_ptr_to_index_;
  std::unordered_set<size_t> promoted_;
};

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

/// Rewrite ``tile.load(p, ..., transpose=True)`` calls whose source ``p`` is a
/// promoted InCore parameter to read from the body-local
/// ``p_dn = tensor.as_layout(p, DN)`` binding instead. For each rewritten call:
///   - the source arg is redirected from ``p`` to ``p_dn``;
///   - offsets / shapes / valid_shapes are swapped to canonical coords;
///   - the ``transpose=True`` kwarg is flipped to ``transpose=False`` (the DN
///     source + Mat target now drives the tile-view swap inside
///     ``DeduceTileLoadType``).
/// Loads on the same ``p`` with ``transpose=False`` (or any other use of ``p``)
/// are passed through unchanged so a single param can be loaded under both
/// orientations in the same body — the two orientations end up as two distinct
/// cube loads, one reading ``p`` (ND) and one reading ``p_dn`` (DN).
class TileLoadBodyRewriter : public IRMutator {
 public:
  explicit TileLoadBodyRewriter(const std::unordered_map<const Var*, VarPtr>& param_to_dn_view)
      : param_to_dn_view_(param_to_dn_view) {}

  ExprPtr VisitExpr_(const CallPtr& op) override {
    auto base = IRMutator::VisitExpr_(op);
    auto call = std::dynamic_pointer_cast<const Call>(base);
    if (!call || !call->op_ || call->op_->name_ != "tile.load") return base;
    if (call->args_.empty()) return base;

    auto src_var = As<Var>(call->args_[0]);
    if (!src_var) return base;
    auto it = param_to_dn_view_.find(src_var.get());
    if (it == param_to_dn_view_.end()) return base;
    // Only rewrite the transposed loads; non-transposed loads on the same
    // promoted param keep reading the ND param straight through.
    if (!call->GetKwarg<bool>("transpose", false)) return base;

    // tile.load(tensor, offsets, shapes, valid_shapes, ...) — swap the trailing
    // pair of all three tuples so the load is expressed in canonical (DN
    // logical) coordinates that match the body-local ``b_dn`` view's shape.
    INTERNAL_CHECK_SPAN(call->args_.size() == 4, call->span_)
        << "LowerTransposeLoadParamLayout: expected tile.load to have 4 args, got " << call->args_.size();
    auto offsets = As<MakeTuple>(call->args_[1]);
    auto shapes = As<MakeTuple>(call->args_[2]);
    auto valid_shapes = As<MakeTuple>(call->args_[3]);
    INTERNAL_CHECK_SPAN(offsets && shapes && valid_shapes, call->span_)
        << "LowerTransposeLoadParamLayout: tile.load offsets/shapes/valid_shapes must be MakeTuple";

    std::vector<ExprPtr> new_args = call->args_;
    new_args[0] = it->second;
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
  const std::unordered_map<const Var*, VarPtr>& param_to_dn_view_;
};

/// Rewrite an InCore function: keep params unchanged; prepend
/// ``b_dn = tensor.as_layout(b, layout=DN)`` AssignStmts at the top of the
/// body for every param ``b`` loaded with ``transpose=True``; rewrite each
/// transposed tile.load on those params to read from ``b_dn`` with the
/// trailing pair of offsets/shapes/valid_shapes swapped and ``transpose=True``
/// flipped to ``transpose=False``.
///
/// A param loaded with both ``transpose=True`` and ``transpose=False`` in the
/// same body is supported: the non-transposed loads keep reading the original
/// ND param, while the transposed loads get redirected to ``b_dn``. The two
/// orientations coexist as two distinct cube loads (issue #1532).
///
/// Returns the rewritten Function (or the original if no rewrite was needed).
FunctionPtr LowerInCoreFunction(const FunctionPtr& func) {
  TransposeLoadScanner scanner(func->params_);
  scanner.VisitStmt(func->body_);
  const auto& promoted = scanner.GetPromoted();
  if (promoted.empty()) {
    return func;
  }

  // Build, in deterministic param-index order:
  //   - the prepend AssignStmts (one per promoted param), each of the form
  //     ``b_dn = tensor.as_layout(b, layout=DN)``;
  //   - the ``param -> b_dn`` map used by ``TileLoadBodyRewriter`` to redirect
  //     each transposed tile.load to its body-local DN view.
  std::vector<size_t> sorted_promoted(promoted.begin(), promoted.end());
  std::sort(sorted_promoted.begin(), sorted_promoted.end());

  std::vector<StmtPtr> prepend;
  std::unordered_map<const Var*, VarPtr> param_to_dn_view;

  for (size_t idx : sorted_promoted) {
    const auto& param = func->params_[idx];
    auto param_tensor_type = As<TensorType>(param->GetType());
    CHECK(param_tensor_type) << "LowerTransposeLoadParamLayout: promoted parameter at index " << idx
                             << " must be TensorType";

    // Reject the (DN view + explicit physical stride) combination — these
    // came from `tensor.transpose` and would compose with the load-side
    // transpose to produce a double-encoded transpose.
    if (param_tensor_type->tensor_view_.has_value()) {
      const auto& view = param_tensor_type->tensor_view_.value();
      CHECK(!(view.layout == TensorLayout::DN && !view.stride.empty()))
          << "LowerTransposeLoadParamLayout: tile.load(transpose=True) on a "
             "tensor.transpose result is not supported (the DN tag and explicit "
             "physical strides would compose as a double transpose). Drop one of "
             "the two transpose layers in the source program.";
      // Param already DN-tagged at the boundary (user-written
      // ``pl.Tensor[..., pl.DN]``): the load-side ``transpose=True`` is the
      // user-intended signal that the on-chip tile flips back to row-major
      // Mat orientation. ``DeduceTileLoadType`` already handles this via
      // the (source_is_dn XOR transpose) tile-view logic — adding a bridge
      // and dropping ``transpose=True`` would shift the XOR result and
      // produce the wrong TileType. Skip this param.
      if (view.layout == TensorLayout::DN) continue;
    }

    // Build ``b_dn = tensor.as_layout(b, layout=DN)``. Routing through the
    // OpRegistry::Create path makes ``DeduceTensorAsLayoutType`` compute
    // the post-flip type and inherit ``b``'s MemRef.
    std::vector<std::pair<std::string, std::any>> kwargs = {{"layout", std::any(TensorLayout::DN)}};
    auto bridge_call = OpRegistry::GetInstance().Create("tensor.as_layout", {param}, kwargs, param->span_);
    auto bridge_var =
        std::make_shared<Var>(param->name_hint_ + "_dn_view", bridge_call->GetType(), param->span_);
    prepend.push_back(std::make_shared<AssignStmt>(bridge_var, bridge_call, param->span_));
    param_to_dn_view[param.get()] = bridge_var;
  }

  if (prepend.empty()) {
    return func;
  }

  // Rewrite each ``tile.load(b, ..., transpose=True)`` on a promoted param to
  // ``tile.load(b_dn, ..., transpose=False)`` with the trailing pair of
  // offsets/shapes/valid_shapes swapped. Other uses of ``b`` (including
  // ``tile.load(b, ..., transpose=False)``) are left untouched, so mixed-mode
  // loads on the same param resolve to two coexisting cube loads.
  TileLoadBodyRewriter body_rewriter(param_to_dn_view);
  auto rewritten_body = body_rewriter.VisitStmt(func->body_);

  // Concatenate: new body = SeqStmts([prepend stmts..., rewritten original body]).
  std::vector<StmtPtr> new_body_stmts = std::move(prepend);
  new_body_stmts.push_back(rewritten_body);
  auto new_body = SeqStmts::Flatten(std::move(new_body_stmts), func->body_->span_);

  auto new_func = MutableCopy(func);
  new_func->body_ = new_body;
  return new_func;
}

}  // namespace

namespace pass {

Pass LowerTransposeLoadParamLayout() {
  auto pass_func = [](const ProgramPtr& program) -> ProgramPtr {
    // Rewrite each InCore function: prepend ``b_dn = tensor.as_layout(b, DN)``
    // for every ``transpose=True``-loaded param ``b`` and substitute body uses
    // accordingly. Non-InCore functions (orch callers) are left untouched —
    // they pass their original ND args straight through; the layout
    // reinterpret is now owned by the InCore body it serves.
    std::map<GlobalVarPtr, FunctionPtr, GlobalVarPtrLess> new_functions;
    bool modified = false;

    for (const auto& [gvar, func] : program->functions_) {
      if (!IsInCoreType(func->func_type_)) {
        new_functions[gvar] = func;
        continue;
      }
      auto new_func = LowerInCoreFunction(func);
      new_functions[gvar] = new_func;
      if (new_func.get() != func.get()) modified = true;
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
