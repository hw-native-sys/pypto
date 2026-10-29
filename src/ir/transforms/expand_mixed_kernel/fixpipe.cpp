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

#include "src/ir/transforms/expand_mixed_kernel/fixpipe.h"

#include <cmath>
#include <map>
#include <memory>
#include <unordered_map>
#include <vector>

#include "pypto/backend/common/backend_handler.h"
#include "pypto/ir/cast_saturation.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/tile_view_semantics.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_context.h"

namespace pypto::ir::mixed_kernel {
namespace {

class Consumers : public IRVisitor {
 public:
  std::unordered_map<const Var*, std::vector<AssignStmtPtr>> uses;
  bool explicit_pipe = false;

  AssignStmtPtr SoleUse(const VarPtr& value) const {
    auto found = uses.find(value.get());
    return found != uses.end() && found->second.size() == 1 ? found->second.front() : nullptr;
  }

 protected:
  void VisitStmt_(const AssignStmtPtr& statement) override {
    // Only value uses count. A use in control flow/return is recorded with a
    // null consumer, so it can never masquerade as a foldable cast/ReLU.
    auto previous = current_;
    current_ = statement;
    VisitExpr(statement->value_);
    current_ = previous;
  }
  void VisitExpr_(const VarPtr& value) override { uses[value.get()].push_back(current_); }
  void VisitExpr_(const IterArgPtr& value) override { uses[value.get()].push_back(current_); }
  void VisitExpr_(const CallPtr& call) override {
    if (IsOp(call, "tile.tpush_to_aiv") || IsOp(call, "tile.tpop_from_aic") ||
        IsOp(call, "tile.tpush_to_aic") || IsOp(call, "tile.tpop_from_aiv") ||
        IsOp(call, "system.aic_initialize_pipe") || IsOp(call, "system.aiv_initialize_pipe")) {
      explicit_pipe = true;
    }
    IRVisitor::VisitExpr_(call);
  }

 private:
  AssignStmtPtr current_;
};

bool IsCast(const AssignStmtPtr& statement, DataType source, DataType target, bool narrowing) {
  auto call = statement ? As<Call>(statement->value_) : nullptr;
  if (!call || !IsOp(call, "tile.cast") || call->args_.size() != 1) return false;
  auto from = As<TileType>(call->args_[0]->GetType());
  auto to = As<TileType>(statement->var_->GetType());
  if (!from || !to || from->dtype_ != source || to->dtype_ != target ||
      !tile_view_semantics::ShapeExprListsEquivalent(from->shape_, to->shape_))
    return false;
  // Fixpipe narrows with round-to-nearest-even and has no configurable cast
  // saturation. Preserve every other rounding/saturation request on Vector.
  return !narrowing || (call->GetKwarg<int>("mode", 2) == 1 && !GetSaturationMode(call).has_value());
}

bool IsRelu(const AssignStmtPtr& statement) {
  auto call = statement ? As<Call>(statement->value_) : nullptr;
  if (!call || !IsOp(call, "tile.maximums") || call->args_.size() != 2) return false;
  if (auto scalar = As<ConstFloat>(call->args_[1]))
    return scalar->value_ == 0 && !std::signbit(scalar->value_);
  if (auto scalar = As<ConstInt>(call->args_[1])) return scalar->value_ == 0;
  return false;
}

struct Candidate {
  core_affinity::CVBoundaryMove* boundary;
  AssignStmtPtr cast;
  AssignStmtPtr relu;
  AssignStmtPtr widening;
};

}  // namespace

std::vector<StmtPtr> RouteFixpipeV2C(const std::vector<StmtPtr>& statements) {
  class Router : public IRMutator {
    ExprPtr VisitExpr_(const CallPtr& call) override {
      if (!IsOp(call, "tile.tpush_to_aic") && !IsOp(call, "tile.tpop_from_aiv") &&
          !IsOp(call, "system.tfree_to_aiv"))
        return IRMutator::VisitExpr_(call);
      auto kwargs = call->kwargs_;
      kwargs.emplace_back("id", 1);
      return std::make_shared<Call>(call->op_, call->args_, kwargs, call->GetType(), call->span_);
    }
  } router;
  std::vector<StmtPtr> result;
  for (const auto& statement : statements) result.push_back(router.VisitStmt(statement));
  return result;
}

C2VFixpipeFold PlanC2VFixpipeFold(const FunctionPtr& function, const std::vector<StmtPtr>& statements,
                                  std::map<const Stmt*, core_affinity::CVBoundaryMove>& boundaries) {
  C2VFixpipeFold result;
  const auto* backend = PassContext::Current()->GetBackendHandler();
  if (!backend->SupportsC2VFixpipeEpilogue() ||
      (function->GetSplitMode().has_value() && *function->GetSplitMode() != SplitMode::None))
    return result;
  Consumers consumers;
  for (const auto& statement : statements) consumers.VisitStmt(statement);
  if (consumers.explicit_pipe) return result;
  std::vector<Candidate> candidates;
  bool all_relu = true;
  for (auto& [statement, boundary] : boundaries) {
    if (boundary.direction != core_affinity::CVDirection::CUBE_TO_VECTOR) continue;
    auto source = As<TileType>(boundary.source_tile->GetType());
    if (boundary.op_driven || !source || source->GetMemorySpace() != MemorySpace::Acc ||
        source->dtype_ != DataType::FP32)
      return result;
    auto cast = consumers.SoleUse(boundary.dest_var);
    if (!IsCast(cast, DataType::FP32, DataType::BF16, true)) return result;
    auto target = As<TileType>(cast->var_->GetType());
    auto columns = target->shape_.size() == 2 ? As<ConstInt>(target->shape_[1]) : nullptr;
    const auto view = tile_view_semantics::GetEffectiveTileView(*target);
    // NZ2ND Fixpipe requires a positive 32-byte destination row stride.
    // FP32 transport can accept 8 columns, while a BF16 payload needs 16;
    // declining the fold preserves the existing Vector conversion fallback.
    if (!columns || columns->value_ <= 0 || columns->value_ % 16 != 0 ||
        view.blayout != TileLayout::row_major || view.slayout != TileLayout::none_box)
      return result;
    auto next = consumers.SoleUse(cast->var_);
    AssignStmtPtr widening;
    if (IsCast(next, DataType::BF16, DataType::FP32, false)) {
      widening = next;
      next = consumers.SoleUse(widening->var_);
    }
    auto relu = IsRelu(next) ? next : nullptr;
    all_relu = all_relu && relu != nullptr;
    candidates.push_back({&boundary, cast, relu, widening});
  }
  if (candidates.empty()) return result;
  result.enabled = true;
  for (const auto& candidate : candidates) {
    auto& boundary = *candidate.boundary;
    boundary.dest_var = candidate.cast->var_;
    boundary.result_type = candidate.cast->var_->GetType();
    boundary.fixpipe_bf16 = true;
    boundary.fixpipe_relu = all_relu;
    result.erased.insert(candidate.cast.get());
    if (all_relu && !candidate.widening) {
      boundary.dest_var = candidate.relu->var_;
      boundary.result_type = candidate.relu->var_->GetType();
      result.erased.insert(candidate.relu.get());
    } else if (all_relu) {
      // Keep the existing FP32 consumer ABI, while the FIFO carries the
      // already-clamped BF16 values. The assignment is an ordinary Tile alias.
      result.replacements[candidate.relu.get()] =
          std::make_shared<AssignStmt>(candidate.relu->var_, candidate.widening->var_, candidate.relu->span_);
    }
  }
  return result;
}

}  // namespace pypto::ir::mixed_kernel
