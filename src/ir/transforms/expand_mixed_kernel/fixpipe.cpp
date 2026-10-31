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
#include <cstddef>
#include <map>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "pypto/backend/common/backend_handler.h"
#include "pypto/core/dtype.h"
#include "pypto/ir/cast_saturation.h"
#include "pypto/ir/core_affinity_kind.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/tile_view_semantics.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_context.h"
#include "pypto/ir/transforms/utils/core_affinity.h"
#include "pypto/ir/transforms/utils/op_predicates.h"
#include "pypto/ir/type.h"

namespace pypto::ir::mixed_kernel {
namespace {

class Consumers : public IRVisitor {
 public:
  explicit Consumers(const std::unordered_map<const Stmt*, core_affinity::CoreAffinity>& affinities)
      : affinities_(affinities) {}

  std::unordered_map<const Var*, std::vector<AssignStmtPtr>> uses;
  bool explicit_pipe = false;

  // Propagate local-storage requirements through plain SSA aliases once.
  // Each alias edge is visited at most once, keeping planning linear in IR size.
  void FinalizeLocalStorageUses() {
    std::vector<const Var*> pending(local_storage_.begin(), local_storage_.end());
    pending.reserve(local_storage_.size() + alias_sources_.size());
    for (std::size_t i = 0; i < pending.size(); ++i) {
      auto found = alias_sources_.find(pending[i]);
      if (found != alias_sources_.end() && local_storage_.insert(found->second).second) {
        pending.push_back(found->second);
      }
    }
  }

  [[nodiscard]] bool RequiresLocalStorage(const VarPtr& value) const {
    return local_storage_.count(value.get()) != 0;
  }

  [[nodiscard]] AssignStmtPtr SoleUse(const VarPtr& value) const {
    auto found = uses.find(value.get());
    return found != uses.end() && found->second.size() == 1 ? found->second.front() : nullptr;
  }

 protected:
  void VisitStmt_(const AssignStmtPtr& statement) override {
    // Only value uses count. A use in control flow/return is recorded with a
    // null consumer, so it can never masquerade as a foldable cast/ReLU.
    if (auto source = AsVarLike(statement->value_)) {
      alias_sources_[statement->var_.get()] = source.get();
    }
    auto previous = current_;
    current_ = retained_ ? nullptr : statement;
    VisitExpr(statement->value_);
    current_ = previous;
  }
  // The base IterArg visitor also visits initValue_; overriding only the
  // Var-like hook preserves loop initializer readers in the sole-use analysis.
  void VisitVarLike_(const VarPtr& value) override {
    uses[value.get()].push_back(current_);
    if (auto iter = As<IterArg>(value)) {
      if (auto initial = AsVarLike(iter->initValue_)) {
        local_storage_.insert(initial.get());
      }
    }
  }
  void VisitStmt_(const YieldStmtPtr& statement) override {
    for (const auto& value : statement->value_) {
      if (auto source = AsVarLike(value)) {
        local_storage_.insert(source.get());
      }
    }
    IRVisitor::VisitStmt_(statement);
  }
  void VisitStmt_(const ForStmtPtr& statement) override { VisitCompound(statement); }
  void VisitStmt_(const IfStmtPtr& statement) override { VisitCompound(statement); }
  void VisitStmt_(const WhileStmtPtr& statement) override { VisitCompound(statement); }
  void VisitExpr_(const CallPtr& call) override {
    // PTOAS cannot rebind valid extents or reinterpret a tpop's returned tile
    // as a locally declared allocation. Keep the Vector cast for those uses.
    if (!call->args_.empty() && op_predicates::IsBufferAliasingViewOp(call->op_->name_)) {
      if (auto source = AsVarLike(call->args_[0])) {
        local_storage_.insert(source.get());
      }
    }
    if (IsOp(call, "tile.tpush_to_aiv") || IsOp(call, "tile.tpop_from_aic") ||
        IsOp(call, "tile.tpush_to_aic") || IsOp(call, "tile.tpop_from_aiv") ||
        IsOp(call, "system.aic_initialize_pipe") || IsOp(call, "system.aiv_initialize_pipe")) {
      explicit_pipe = true;
    }
    IRVisitor::VisitExpr_(call);
  }

 private:
  template <typename T>
  void VisitCompound(const T& statement) {
    const bool previous = retained_;
    auto found = affinities_.find(statement.get());
    // BuildCoreBody only descends into MIXED compounds. Uses in a retained
    // Vector/Shared compound must not produce a rewrite plan it cannot apply.
    retained_ =
        retained_ || found == affinities_.end() || found->second != core_affinity::CoreAffinity::MIXED;
    IRVisitor::VisitStmt_(statement);
    retained_ = previous;
  }

  const std::unordered_map<const Stmt*, core_affinity::CoreAffinity>& affinities_;
  bool retained_ = false;
  AssignStmtPtr current_;
  std::unordered_map<const Var*, const Var*> alias_sources_;
  std::unordered_set<const Var*> local_storage_;
};

bool IsCast(const AssignStmtPtr& statement, DataType source, DataType target, bool narrowing) {
  auto call = statement ? As<Call>(statement->value_) : nullptr;
  if (!call || !IsOp(call, "tile.cast") || call->args_.size() != 1) {
    return false;
  }
  auto from = As<TileType>(call->args_[0]->GetType());
  auto to = As<TileType>(statement->var_->GetType());
  if (!from || !to || from->dtype_ != source || to->dtype_ != target ||
      !tile_view_semantics::ShapeExprListsEquivalent(from->shape_, to->shape_)) {
    return false;
  }
  // Fixpipe narrows with round-to-nearest-even and has no configurable cast
  // saturation. Preserve every other rounding/saturation request on Vector.
  return !narrowing || (call->GetKwarg<int>("mode", 2) == 1 && !GetSaturationMode(call).has_value());
}

bool IsRelu(const AssignStmtPtr& statement) {
  auto call = statement ? As<Call>(statement->value_) : nullptr;
  if (!call || !IsOp(call, "tile.maximums") || call->args_.size() != 2) {
    return false;
  }
  if (auto scalar = As<ConstFloat>(call->args_[1])) {
    return scalar->value_ == 0 && !std::signbit(scalar->value_);
  }
  if (auto scalar = As<ConstInt>(call->args_[1])) {
    return scalar->value_ == 0;
  }
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
          !IsOp(call, "system.tfree_to_aiv")) {
        return IRMutator::VisitExpr_(call);
      }
      auto kwargs = call->kwargs_;
      kwargs.emplace_back("id", 1);
      return std::make_shared<Call>(call->op_, call->args_, kwargs, call->GetType(), call->span_);
    }
  } router;
  std::vector<StmtPtr> result;
  result.reserve(statements.size());
  for (const auto& statement : statements) result.push_back(router.VisitStmt(statement));
  return result;
}

C2VFixpipeFold PlanC2VFixpipeFold(
    const FunctionPtr& function, const std::vector<StmtPtr>& statements,
    std::map<const Stmt*, core_affinity::CVBoundaryMove>& boundaries,
    const std::unordered_map<const Stmt*, core_affinity::CoreAffinity>& affinities) {
  C2VFixpipeFold result;
  const auto* backend = PassContext::Current()->GetBackendHandler();
  const auto split_mode = function->GetSplitMode();
  if (!backend->SupportsC2VFixpipeEpilogue() || (split_mode.has_value() && *split_mode != SplitMode::None)) {
    return result;
  }
  Consumers consumers(affinities);
  for (const auto& statement : statements) consumers.VisitStmt(statement);
  if (consumers.explicit_pipe) return result;
  consumers.FinalizeLocalStorageUses();
  std::vector<Candidate> candidates;
  bool all_relu = true;
  for (auto& [statement, boundary] : boundaries) {
    if (boundary.direction != core_affinity::CVDirection::CUBE_TO_VECTOR) continue;
    auto source = As<TileType>(boundary.source_tile->GetType());
    if (boundary.op_driven || !source || source->GetMemorySpace() != MemorySpace::Acc ||
        source->dtype_ != DataType::FP32) {
      return result;
    }
    auto cast = consumers.SoleUse(boundary.dest_var);
    if (!IsCast(cast, DataType::FP32, DataType::BF16, true)) return result;
    auto target = As<TileType>(cast->var_->GetType());
    auto columns = target->shape_.size() == 2 ? As<ConstInt>(target->shape_[1]) : nullptr;
    const auto view = tile_view_semantics::GetEffectiveTileView(*target);
    // NZ2ND Fixpipe requires a positive 32-byte destination row stride.
    // FP32 transport can accept 8 columns, while a BF16 payload needs 16;
    // declining the fold preserves the existing Vector conversion fallback.
    if (!columns || columns->value_ <= 0 || columns->value_ % 16 != 0 ||
        view.blayout != TileLayout::row_major || view.slayout != TileLayout::none_box) {
      return result;
    }
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
  TileTypePtr wire_type;
  for (const auto& candidate : candidates) {
    auto value = all_relu && !candidate.widening ? candidate.relu->var_ : candidate.cast->var_;
    auto type = As<TileType>(value->GetType());
    if (!type || type->shape_.size() != 2) return result;
    const auto view = tile_view_semantics::GetEffectiveTileView(*type);
    // A Fixpipe initializer has one exact pop type. Restrict this optimization
    // to full static tiles; preserve local casts for partial/dynamic views.
    auto rows = As<ConstInt>(type->shape_[0]);
    if (!rows || rows->value_ <= 0 || view.blayout != TileLayout::row_major ||
        view.slayout != TileLayout::none_box || consumers.RequiresLocalStorage(candidate.cast->var_) ||
        consumers.RequiresLocalStorage(value) ||
        !tile_view_semantics::ShapeExprListsEquivalent(view.valid_shape, type->shape_) ||
        (wire_type && (!tile_view_semantics::ShapeExprListsEquivalent(wire_type->shape_, type->shape_) ||
                       tile_view_semantics::GetEffectiveTileView(*wire_type) != view))) {
      return result;
    }
    wire_type = type;
  }
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
