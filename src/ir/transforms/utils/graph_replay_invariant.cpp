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

#include "pypto/ir/transforms/utils/graph_replay_invariant.h"

#include <algorithm>
#include <functional>
#include <iterator>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/structural_comparison.h"
#include "pypto/ir/transforms/utils/alloc_batching.h"
#include "pypto/ir/transforms/utils/auto_name_utils.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/transforms/utils/var_collectors.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {
namespace graph_replay {
namespace {

// This proof cannot succeed with more roots than a Graph can accept as tensor
// arguments. Bound each union to keep the caller scan linear in the IR size.
constexpr size_t kMaxGraphTensorRoots = 128;

void MergeProof(std::optional<std::vector<const Var*>>& lhs,
                const std::optional<std::vector<const Var*>>& rhs) {
  if (!lhs || !rhs) {
    lhs.reset();
    return;
  }
  if (rhs->empty()) return;
  std::vector<const Var*> merged;
  std::set_union(lhs->begin(), lhs->end(), rhs->begin(), rhs->end(), std::back_inserter(merged),
                 std::less<const Var*>{});
  if (merged.size() > kMaxGraphTensorRoots) {
    lhs.reset();
  } else {
    lhs = std::move(merged);
  }
}

}  // namespace

namespace {

/// True when @p op's induction variable takes the same values on every call.
///
/// The same three constants `StaticTripCount` needs, for the same reason: a
/// bound that reads a runtime value makes the sequence — and therefore each
/// iteration's baked-in literal — differ between calls. A zero step is a
/// malformed loop rather than an invariant one, so it is excluded too.
[[nodiscard]] bool HasConstantBounds(const ForStmtPtr& op) {
  auto start = As<ConstInt>(op->start_);
  auto stop = As<ConstInt>(op->stop_);
  auto step = As<ConstInt>(op->step_);
  return start != nullptr && stop != nullptr && step != nullptr && step->value_ != 0;
}

}  // namespace

bool IsTaskIdScalar(const TypePtr& type) {
  auto scalar = As<ScalarType>(type);
  return scalar != nullptr && scalar->dtype_ == DataType::TASK_ID;
}

ReplayInvariantSet::ReplayInvariantSet(const FunctionPtr& graph_func,
                                       const std::unordered_set<const Var*>& proven_fixed_scalar_params) {
  if (!graph_func) return;
  for (const auto& param : graph_func->params_) {
    // A parameter is its own boundary root. Scalar parameters are deliberately
    // *not* seeded into `invariant_` — see the class comment.
    if (param && AsTensorTypeLike(param->GetType())) boundary_tensors_.try_emplace(param.get());
    if (param && As<ScalarType>(param->GetType()) != nullptr && !IsTaskIdScalar(param->GetType()) &&
        proven_fixed_scalar_params.count(param.get()) != 0) {
      invariant_.try_emplace(param.get());
    }
  }
}

void ReplayInvariantSet::SeedBoundaryDimensionSymbols(const FunctionPtr& func) {
  if (!func) return;
  // Match GenerateDynamicDimDefs: a scalar or local declaration occupies its
  // emitted name before tensor parameter extents are considered in order.
  std::unordered_set<std::string> occupied;
  for (const auto& param : func->params_) {
    if (!AsTensorTypeLike(param->GetType())) {
      occupied.insert(auto_name::GetCompatibleBaseName(param->name_hint_));
    }
  }
  var_collectors::VarDefUseCollector definitions;
  if (func->body_) definitions.VisitStmt(func->body_);
  for (const auto* var : definitions.var_defs) {
    occupied.insert(auto_name::GetCompatibleBaseName(var->name_hint_));
  }
  for (const auto& param : func->params_) {
    const auto tensor = AsTensorTypeLike(param->GetType());
    if (!tensor) continue;
    for (const auto& extent : tensor->shape_) {
      const auto symbol = As<Var>(extent);
      if (!symbol) continue;
      const auto name = auto_name::GetCompatibleBaseName(symbol->name_hint_);
      // Even an unselected descriptor claims the first binding of this name.
      // Only the exact Var from a selected descriptor can acquire the proof.
      if (!occupied.insert(name).second) continue;
      auto root = boundary_tensors_.find(param.get());
      if (root != boundary_tensors_.end()) invariant_.try_emplace(symbol.get(), root->second);
    }
  }
}

void ReplayInvariantSet::TrackTensorAlias(const AssignStmtPtr& op) {
  auto var = AsVarLike(op->var_);
  if (!var || !AsTensorTypeLike(var->GetType())) return;
  auto aliased = AsVarLike(op->value_);
  if (aliased) {
    auto it = fixed_local_axes_.find(aliased.get());
    const auto source_type = AsTensorTypeLike(aliased->GetType());
    const auto target_type = AsTensorTypeLike(var->GetType());
    if (it != fixed_local_axes_.end() && source_type && target_type &&
        source_type->shape_.size() == target_type->shape_.size()) {
      std::vector<Proof> axes(it->second.size());
      for (size_t i = 0; i < axes.size(); ++i) {
        if (structural_equal(source_type->shape_[i], target_type->shape_[i])) axes[i] = it->second[i];
      }
      fixed_local_axes_[var.get()] = std::move(axes);
    }
  }
  auto create = As<Call>(op->value_);
  if (create && IsOp(create, "tensor.create") && !alloc_batching::IsInjectedGMPipeCreateVar(var)) {
    const auto type = As<TensorType>(create->GetType());
    // Normalized zero-argument creates allocate from TensorType::shape_ (the
    // orchestration emitter's source of truth). Frontend tuple-shaped creates
    // must agree with that same shape; a forged result type cannot hide data.
    const auto assigned_type = As<TensorType>(var->GetType());
    bool consistent = type && assigned_type && type->shape_.size() == assigned_type->shape_.size();
    if (consistent) {
      for (size_t i = 0; i < type->shape_.size(); ++i) {
        consistent = consistent && structural_equal(type->shape_[i], assigned_type->shape_[i]);
      }
    }
    if (!create->args_.empty()) {
      const auto shape = create->args_.size() == 1 ? As<MakeTuple>(create->args_[0]) : nullptr;
      consistent = consistent && shape && shape->elements_.size() == type->shape_.size();
      if (consistent) {
        for (size_t i = 0; i < type->shape_.size(); ++i) {
          consistent = consistent && structural_equal(shape->elements_[i], type->shape_[i]);
        }
      }
    }
    if (consistent) {
      std::vector<Proof> axes;
      for (const auto& extent : type->shape_) axes.push_back(ProveExpr(extent));
      fixed_local_axes_[var.get()] = std::move(axes);
    }
  }
  if (aliased) {
    auto root = boundary_tensors_.find(aliased.get());
    const auto source_type = AsTensorTypeLike(aliased->GetType());
    const auto target_type = AsTensorTypeLike(var->GetType());
    if (root != boundary_tensors_.end() && source_type && target_type &&
        source_type->shape_.size() == target_type->shape_.size() &&
        std::equal(source_type->shape_.begin(), source_type->shape_.end(), target_type->shape_.begin(),
                   [](const auto& lhs, const auto& rhs) {
                     return structural_equal(lhs, rhs, /*enable_auto_mapping=*/false);
                   })) {
      boundary_tensors_.try_emplace(var.get(), root->second);
      return;
    }
  }

  // A metadata-only view does not read tensor contents. Its dimensions can be
  // recorded only after its boundary source, every shape/offset operand and
  // every resulting extent are proved invariant in SSA definition order.
  // In particular a fixed shape does not bless a data-dependent offset.
  auto call = As<Call>(op->value_);
  if (!call || !(IsOp(call, "tensor.slice") || IsOp(call, "tensor.reshape") || IsOp(call, "tensor.view")) ||
      call->args_.empty())
    return;
  auto source = AsVarLike(call->args_.front());
  auto root = boundary_tensors_.find(source.get());
  if (root == boundary_tensors_.end()) return;
  Proof proof = root->second;
  for (size_t i = 1; i < call->args_.size(); ++i) MergeProof(proof, ProveExpr(call->args_[i]));
  const auto tensor = AsTensorTypeLike(var->GetType());
  for (const auto& extent : tensor->shape_) MergeProof(proof, ProveExpr(extent));
  // A view proof describes shape, not address, contents or descriptor equality.
  if (proof) boundary_tensors_.try_emplace(var.get(), std::move(*proof));
}

void ReplayInvariantSet::Collect(const StmtPtr& body, bool include_constant_trip_indices) {
  if (!body) return;

  class Collector : public IRVisitor {
   public:
    Collector(ReplayInvariantSet* owner, bool include_indices)
        : owner_(owner), include_indices_(include_indices) {}

   protected:
    void VisitStmt_(const ForStmtPtr& op) override {
      // Recorded *before* descending, so the body sees its own induction
      // variable as invariant.
      if (include_indices_ && HasConstantBounds(op) && op->loop_var_)
        owner_->invariant_.try_emplace(op->loop_var_.get());
      IRVisitor::VisitStmt_(op);
    }

    void VisitStmt_(const IfStmtPtr& op) override {
      IRVisitor::VisitStmt_(op);
      // A fixed metadata decision and fixed incoming scalar values select the
      // same value on replay. Runtime scalars and tensor contents stay excluded.
      auto condition = owner_->ProveExpr(op->condition_);
      if (!condition || !op->else_body_.has_value()) return;
      auto then_yield = transform_utils::GetLastYieldStmt(op->then_body_);
      auto else_yield = transform_utils::GetLastYieldStmt(*op->else_body_);
      if (!then_yield || !else_yield || then_yield->value_.size() != op->return_vars_.size() ||
          else_yield->value_.size() != op->return_vars_.size())
        return;
      for (size_t i = 0; i < op->return_vars_.size(); ++i) {
        auto var = AsVarLike(op->return_vars_[i]);
        if (!var || !As<ScalarType>(var->GetType()) || IsTaskIdScalar(var->GetType())) continue;
        auto proof = condition;
        MergeProof(proof, owner_->ProveExpr(then_yield->value_[i]));
        MergeProof(proof, owner_->ProveExpr(else_yield->value_[i]));
        if (proof) owner_->invariant_.try_emplace(var.get(), std::move(*proof));
      }
    }

    void VisitStmt_(const AssignStmtPtr& op) override {
      IRVisitor::VisitStmt_(op);
      owner_->TrackTensorAlias(op);
      auto var = AsVarLike(op->var_);
      if (!var || As<ScalarType>(var->GetType()) == nullptr) return;
      if (auto proof = owner_->ProveExpr(op->value_)) {
        owner_->invariant_.try_emplace(var.get(), std::move(*proof));
      }
    }

   private:
    ReplayInvariantSet* owner_;
    bool include_indices_;
  };

  Collector collector(this, include_constant_trip_indices);
  collector.VisitStmt(body);
}

ReplayInvariantSet::FixedScalarIndex ReplayInvariantSet::IndexFixedScalarsFromCallSites(
    const ProgramPtr& program) {
  if (!program) return {};
  std::unordered_map<std::string, FunctionPtr> graphs;
  for (const auto& [name, func] : program->functions_) {
    if (func && func->func_type_ == FunctionType::Graph) graphs.emplace(func->name_, func);
  }
  if (graphs.empty()) return {};

  // Index sites once, then walk each caller once in SSA definition order.
  // Candidate roots can come from different sites. Every root in a proof must
  // still be present in the actual tensor arguments of the current site.
  struct Site {
    FunctionPtr graph;
    const std::vector<ExprPtr>* args;
  };
  class Calls : public IRVisitor {
   public:
    explicit Calls(const std::unordered_map<std::string, FunctionPtr>& graphs) : graphs_(graphs) {}
    std::vector<Site> sites;

   protected:
    void VisitExpr_(const CallPtr& call) override {
      Record(call->op_, call->args_);
      IRVisitor::VisitExpr_(call);
    }
    void VisitExpr_(const SubmitPtr& submit) override {
      Record(submit->op_, submit->args_);
      IRVisitor::VisitExpr_(submit);
    }

   private:
    void Record(const OpPtr& op, const std::vector<ExprPtr>& args) {
      auto name = As<GlobalVar>(op);
      if (!name) return;
      auto it = graphs_.find(name->name_);
      if (it != graphs_.end()) sites.push_back({it->second, &args});
    }
    const std::unordered_map<std::string, FunctionPtr>& graphs_;
  };

  auto tensor_args = [](const Site& site, const auto& visit) {
    for (size_t i = 0; i < site.args->size() && i < site.graph->params_.size(); ++i) {
      if (!AsTensorTypeLike(site.graph->params_[i]->GetType())) continue;
      auto tensor = AsVarLike((*site.args)[i]);
      if (tensor && AsTensorTypeLike(tensor->GetType())) visit(tensor.get());
    }
  };

  std::unordered_map<const Function*, std::unordered_map<const Var*, bool>> fixed;
  std::unordered_set<const Function*> invalid_arity;
  for (const auto& [name, caller] : program->functions_) {
    if (!caller || !caller->body_) continue;
    Calls calls(graphs);
    calls.VisitStmt(caller->body_);
    if (calls.sites.empty()) continue;

    ReplayInvariantSet proof(nullptr);
    for (const auto& site : calls.sites) {
      tensor_args(site,
                  [&](const Var* tensor) { proof.boundary_tensors_.try_emplace(tensor, Roots{tensor}); });
    }
    proof.SeedBoundaryDimensionSymbols(caller);
    // An induction variable in the caller may change between Graph calls.
    proof.Collect(caller->body_, /*include_constant_trip_indices=*/false);

    for (const auto& site : calls.sites) {
      const auto* graph = site.graph.get();
      if (site.args->size() != graph->params_.size()) {
        invalid_arity.insert(graph);
        continue;
      }
      std::unordered_set<const Var*> actual_tensors;
      tensor_args(site, [&](const Var* tensor) { actual_tensors.insert(tensor); });
      for (size_t i = 0; i < site.graph->params_.size(); ++i) {
        const auto& param = site.graph->params_[i];
        if (!As<ScalarType>(param->GetType()) || IsTaskIdScalar(param->GetType())) continue;
        auto& all_sites = fixed[graph].try_emplace(param.get(), true).first->second;
        if (!all_sites) continue;
        auto roots = proof.ProveExpr((*site.args)[i]);
        all_sites = roots && std::all_of(roots->begin(), roots->end(),
                                         [&](const Var* root) { return actual_tensors.count(root) != 0; });
      }
    }
  }
  FixedScalarIndex result;
  for (const auto& [graph, params] : fixed) {
    if (invalid_arity.count(graph) != 0) continue;
    for (const auto& [param, all_sites] : params) {
      if (all_sites) result[graph].insert(param);
    }
  }
  return result;
}

const std::unordered_set<const Var*>& ReplayInvariantSet::FixedScalarsForGraph(const FixedScalarIndex& index,
                                                                               const FunctionPtr& graph) {
  static const std::unordered_set<const Var*> empty;
  auto it = index.find(graph.get());
  return it == index.end() ? empty : it->second;
}

bool ReplayInvariantSet::IsBoundaryDimRead(const CallPtr& call) const {
  return ProveDimRead(call).has_value();
}

ReplayInvariantSet::Proof ReplayInvariantSet::ProveDimRead(const CallPtr& call) const {
  if (!call || !IsOp(call, "tensor.dim") || call->args_.size() != 2) return std::nullopt;
  auto source = AsVarLike(call->args_[0]);
  // Only a literal axis selects the same extent at every replay.
  const auto axis_expr = As<ConstInt>(call->args_[1]);
  const auto type = source ? AsTensorTypeLike(source->GetType()) : nullptr;
  if (!source || !axis_expr || !type) return std::nullopt;
  const auto rank = static_cast<int64_t>(type->shape_.size());
  const int64_t axis = axis_expr->value_ < 0 ? axis_expr->value_ + rank : axis_expr->value_;
  if (axis < 0 || axis >= rank) return std::nullopt;
  auto root = boundary_tensors_.find(source.get());
  if (root != boundary_tensors_.end()) return root->second;
  auto local = fixed_local_axes_.find(source.get());
  return local == fixed_local_axes_.end() ? std::nullopt : local->second[axis];
}

bool ReplayInvariantSet::IsInvariant(const ExprPtr& expr) const { return ProveExpr(expr).has_value(); }

ReplayInvariantSet::Proof ReplayInvariantSet::ProveExpr(const ExprPtr& expr) const {
  if (!expr) return std::nullopt;

  // Visit all scalar/tuple operands. Calls are opaque except for proved
  // descriptor reads, whose tensor operand is not itself a scalar fact.
  class Checker : public IRVisitor {
   public:
    explicit Checker(const ReplayInvariantSet& owner) : owner_(owner) {}
    Proof proof = Roots{};

   protected:
    void VisitExpr_(const CallPtr& op) override { MergeProof(proof, owner_.ProveDimRead(op)); }
    void VisitExpr_(const SubmitPtr& op) override { proof.reset(); }
    void VisitVarLike_(const VarPtr& op) override {
      auto fact = owner_.invariant_.find(op.get());
      if (fact == owner_.invariant_.end()) {
        proof.reset();
      } else if (proof) {
        MergeProof(proof, fact->second);
      }
    }

   private:
    const ReplayInvariantSet& owner_;
  };
  Checker checker(*this);
  checker.VisitExpr(expr);
  return checker.proof;
}

}  // namespace graph_replay
}  // namespace ir
}  // namespace pypto
