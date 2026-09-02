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

#include "pypto/core/dtype.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {
namespace graph_replay {

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

ReplayInvariantSet::ReplayInvariantSet(const FunctionPtr& graph_func) {
  if (!graph_func) return;
  for (const auto& param : graph_func->params_) {
    // A parameter is its own boundary root. Scalar parameters are deliberately
    // *not* seeded into `invariant_` — see the class comment.
    if (param && As<ScalarType>(param->GetType()) == nullptr) boundary_tensors_.insert(param.get());
  }
}

void ReplayInvariantSet::TrackTensorAlias(const AssignStmtPtr& op) {
  auto var = AsVarLike(op->var_);
  if (!var || As<ScalarType>(var->GetType()) != nullptr) return;
  auto aliased = AsVarLike(op->value_);
  // Only a bare rename. A view is a different tensor with its own shape, and
  // `tensor.dim` of one is not what the boundary signature pins.
  if (aliased && boundary_tensors_.count(aliased.get()) != 0) boundary_tensors_.insert(var.get());
}

void ReplayInvariantSet::Collect(const StmtPtr& body) {
  if (!body) return;

  class Collector : public IRVisitor {
   public:
    explicit Collector(ReplayInvariantSet* owner) : owner_(owner) {}

   protected:
    void VisitStmt_(const ForStmtPtr& op) override {
      // Recorded *before* descending, so the body sees its own induction
      // variable as invariant.
      if (HasConstantBounds(op) && op->loop_var_) owner_->invariant_.insert(op->loop_var_.get());
      IRVisitor::VisitStmt_(op);
    }

    void VisitStmt_(const AssignStmtPtr& op) override {
      IRVisitor::VisitStmt_(op);
      owner_->TrackTensorAlias(op);
      auto var = AsVarLike(op->var_);
      if (!var || As<ScalarType>(var->GetType()) == nullptr) return;
      if (owner_->IsInvariant(op->value_)) owner_->invariant_.insert(var.get());
    }

   private:
    ReplayInvariantSet* owner_;
  };

  Collector collector(this);
  collector.VisitStmt(body);
}

bool ReplayInvariantSet::IsBoundaryDimRead(const CallPtr& call) const {
  if (!call || !IsOp(call, "tensor.dim") || call->args_.size() != 2) return false;
  auto source = AsVarLike(call->args_[0]);
  // A runtime axis would select a different extent per call even though every
  // extent is itself pinned, so only a literal axis qualifies.
  return source != nullptr && boundary_tensors_.count(source.get()) != 0 &&
         As<ConstInt>(call->args_[1]) != nullptr;
}

bool ReplayInvariantSet::IsInvariant(const ExprPtr& expr) const {
  if (!expr) return false;

  /// A generic walk rather than a recursion over node kinds, for the reason
  /// Step A's `IsDerivable` gives: an operand is often a list or a tuple, and
  /// enumerating container kinds by hand silently accepts the ones it misses.
  class Checker : public IRVisitor {
   public:
    explicit Checker(const ReplayInvariantSet& owner) : owner_(owner) {}
    bool invariant = true;

   protected:
    void VisitExpr_(const CallPtr& op) override {
      // A boundary shape read is a leaf, not a call to reject: its operands are
      // a tensor and a literal, neither of which is a scalar value to classify.
      if (owner_.IsBoundaryDimRead(op)) return;
      invariant = false;
    }
    void VisitExpr_(const SubmitPtr& op) override { invariant = false; }
    void VisitVarLike_(const VarPtr& op) override {
      if (owner_.invariant_.count(op.get()) == 0) invariant = false;
    }

   private:
    const ReplayInvariantSet& owner_;
  };

  Checker checker(*this);
  checker.VisitExpr(expr);
  return checker.invariant;
}

}  // namespace graph_replay
}  // namespace ir
}  // namespace pypto
