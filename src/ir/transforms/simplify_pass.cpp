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

/*
 * The arithmetic simplification module takes reference from:
 * - Apache TVM (https://github.com/apache/tvm), Apache License 2.0
 * - MLC-Python (https://github.com/mlc-ai/mlc-python), Apache License 2.0
 */

#include <algorithm>
#include <memory>
#include <optional>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/ir/arith/analyzer.h"
#include "pypto/ir/arith/ir_mutator_with_analyzer.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/dead_code_elimination.h"
#include "pypto/ir/transforms/utils/loop_state_repair.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

namespace {

/// Collects Var pointers whose LHS assignment is NOT safe to bind as a
/// dominating constant. Unsafe if assigned more than once, OR assigned inside
/// a nested scope (IfStmt branch, ForStmt/WhileStmt body) where the
/// assignment may not execute on all paths. This protects pre-SSA callers;
/// in SSA form each Var is single-assigned so the nesting check is redundant
/// but harmless.
class MultiAssignCollector : public IRVisitor {
 public:
  std::unordered_set<const Var*> multi_assigned;

  void VisitStmt_(const AssignStmtPtr& op) override {
    if (nesting_depth_ > 0 || !seen_.insert(op->var_.get()).second) {
      multi_assigned.insert(op->var_.get());
    }
    IRVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const IfStmtPtr& op) override {
    WithNesting([&] { IRVisitor::VisitStmt_(op); });
  }
  void VisitStmt_(const ForStmtPtr& op) override {
    WithNesting([&] { IRVisitor::VisitStmt_(op); });
  }
  void VisitStmt_(const WhileStmtPtr& op) override {
    WithNesting([&] { IRVisitor::VisitStmt_(op); });
  }

 private:
  template <typename F>
  void WithNesting(F&& body) {
    ++nesting_depth_;
    body();
    --nesting_depth_;
  }

  std::unordered_set<const Var*> seen_;
  int nesting_depth_ = 0;
};

class SimplifyMutator : public arith::IRMutatorWithAnalyzer {
 public:
  SimplifyMutator(arith::Analyzer* analyzer, std::unordered_set<const Var*> multi_assigned)
      : IRMutatorWithAnalyzer(analyzer), multi_assigned_(std::move(multi_assigned)) {}

  /// Fold scalar constant bindings at every Var leaf. Reached via the base
  /// IRMutator's qualified ExprFunctor::VisitExpr dispatch when walking Call
  /// args — `analyzer_->Simplify` at the SimplifyExpr level does not recurse
  /// into non-arithmetic Call nodes, so folding must happen at the leaf.
  ExprPtr VisitExpr_(const VarPtr& op) override {
    auto it = var_remap_.find(op.get());
    ExprPtr remapped = (it != var_remap_.end()) ? it->second : op;
    return analyzer_->Simplify(remapped);
  }

  /// Refresh the Call's result type_ so the in-memory IR matches what a
  /// fresh parse would produce (needed for roundtrip structural equality).
  ExprPtr VisitExpr_(const CallPtr& op) override {
    auto base = IRMutator::VisitExpr_(op);
    auto new_type = SimplifyType(base->GetType());
    if (new_type.get() == base->GetType().get()) return base;
    auto call = std::dynamic_pointer_cast<const Call>(base);
    if (!call) return base;
    return std::make_shared<const Call>(call->op_, call->args_, call->kwargs_, call->attrs_, new_type,
                                        call->span_);
  }

  /// Fold arithmetic nodes (Add/Sub/Mul/Div/Min/Max/compare/bitwise/logical)
  /// after children are visited. Needed because Analyzer::Simplify at the
  /// statement-level top does not recurse into non-arithmetic containers
  /// (Call, MakeTuple), so an Add buried inside a shape arg would otherwise
  /// reach downstream with patterns like `K + 0` un-folded.
  ExprPtr VisitBinaryExpr_(const BinaryExprPtr& op) override {
    return analyzer_->Simplify(IRMutator::VisitBinaryExpr_(op));
  }

  ExprPtr VisitUnaryExpr_(const UnaryExprPtr& op) override {
    return analyzer_->Simplify(IRMutator::VisitUnaryExpr_(op));
  }

  StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
    auto new_value = SimplifyExpr(op->value_);
    auto new_var = MaybeRebuildVar(op->var_);
    auto new_type = new_var->GetType();

    // Bind scalar constant assignments so subsequent uses fold to the literal.
    // Pre-SSA: skip if the Var is reassigned — binding the initial value
    // would incorrectly propagate to reads after a reassignment.
    if (As<ScalarType>(new_type) && IsConstScalar(new_value) &&
        multi_assigned_.find(op->var_.get()) == multi_assigned_.end()) {
      analyzer_->Bind(new_var, new_value);
    }

    if (new_value.get() == op->value_.get() && new_var.get() == op->var_.get()) return op;
    auto result = MutableCopy(op);
    result->var_ = new_var;
    result->value_ = new_value;
    return result;
  }

  StmtPtr VisitStmt_(const ForStmtPtr& op) override {
    auto new_start = SimplifyExpr(op->start_);
    auto new_stop = SimplifyExpr(op->stop_);
    auto new_step = SimplifyExpr(op->step_);

    std::optional<ChunkConfig> new_chunk_config = op->chunk_config_;
    bool chunk_config_changed = false;
    if (op->chunk_config_.has_value()) {
      auto new_cs = SimplifyExpr(op->chunk_config_->size);
      if (new_cs.get() != op->chunk_config_->size.get()) {
        new_chunk_config = ChunkConfig{new_cs, op->chunk_config_->policy};
        chunk_config_changed = true;
      }
    }

    // Rebuild iter_args before visiting the body so body references pick up
    // the remapped IterArg identity.
    bool iter_args_changed = false;
    auto new_iter_args = RebuildVec(
        op->iter_args_, [this](const auto& ia) { return MaybeRebuildIterArg(ia); }, &iter_args_changed);

    auto start_ci = As<ConstInt>(new_start);
    auto stop_ci = As<ConstInt>(new_stop);
    bool bound = start_ci && stop_ci && stop_ci->value_ > start_ci->value_;
    if (bound) {
      analyzer_->Bind(op->loop_var_, start_ci->value_, stop_ci->value_);
    }

    auto new_body = VisitStmt(op->body_);

    if (bound) {
      analyzer_->Unbind(op->loop_var_);
    }

    // Rebuild return_vars after the body so folds discovered inside the body
    // are visible in return types.
    bool return_vars_changed = false;
    auto new_return_vars = RebuildVec(
        op->return_vars_, [this](const auto& v) { return MaybeRebuildVar(v); }, &return_vars_changed);

    bool changed = (new_start.get() != op->start_.get()) || (new_stop.get() != op->stop_.get()) ||
                   (new_step.get() != op->step_.get()) || (new_body.get() != op->body_.get()) ||
                   chunk_config_changed || iter_args_changed || return_vars_changed;
    if (!changed) return op;

    auto result = MutableCopy(op);
    result->start_ = new_start;
    result->stop_ = new_stop;
    result->step_ = new_step;
    result->iter_args_ = std::move(new_iter_args);
    result->body_ = new_body;
    result->return_vars_ = std::move(new_return_vars);
    result->chunk_config_ = new_chunk_config;
    return result;
  }

  StmtPtr VisitStmt_(const IfStmtPtr& op) override {
    auto new_condition = SimplifyExpr(op->condition_);

    // Enter constraint scope for then branch (condition is known true).
    StmtPtr new_then;
    {
      auto ctx = analyzer_->GetConstraintContext(new_condition);
      new_then = VisitStmt(op->then_body_);
    }

    // Enter constraint scope for else branch (condition is known false → Not(condition)).
    std::optional<StmtPtr> new_else;
    if (op->else_body_.has_value()) {
      auto ctx = analyzer_->GetConstraintContext(MakeNot(new_condition, new_condition->span_));
      new_else = VisitStmt(*op->else_body_);
    }

    bool changed = (new_condition.get() != op->condition_.get()) ||
                   (new_then.get() != op->then_body_.get()) ||
                   (new_else.has_value() != op->else_body_.has_value()) ||
                   (new_else.has_value() && new_else->get() != op->else_body_->get());
    if (!changed) return op;
    auto result = MutableCopy(op);
    result->condition_ = new_condition;
    result->then_body_ = new_then;
    result->else_body_ = new_else;
    return result;
  }

  StmtPtr VisitStmt_(const WhileStmtPtr& op) override {
    auto new_condition = SimplifyExpr(op->condition_);
    auto new_body = VisitStmt(op->body_);
    bool changed = (new_condition.get() != op->condition_.get()) || (new_body.get() != op->body_.get());
    if (!changed) return op;
    auto result = MutableCopy(op);
    result->condition_ = new_condition;
    result->body_ = new_body;
    return result;
  }

  StmtPtr VisitStmt_(const ReturnStmtPtr& op) override {
    std::vector<ExprPtr> new_values;
    bool changed = false;
    new_values.reserve(op->value_.size());
    for (const auto& val : op->value_) {
      auto new_val = SimplifyExpr(val);
      new_values.push_back(new_val);
      if (new_val.get() != val.get()) changed = true;
    }
    if (!changed) return op;
    auto result = MutableCopy(op);
    result->value_ = std::move(new_values);
    return result;
  }

  StmtPtr VisitStmt_(const YieldStmtPtr& op) override {
    std::vector<ExprPtr> new_values;
    bool changed = false;
    new_values.reserve(op->value_.size());
    for (const auto& val : op->value_) {
      auto new_val = SimplifyExpr(val);
      new_values.push_back(new_val);
      if (new_val.get() != val.get()) changed = true;
    }
    if (!changed) return op;
    auto result = MutableCopy(op);
    result->value_ = std::move(new_values);
    return result;
  }

  StmtPtr VisitStmt_(const EvalStmtPtr& op) override {
    auto new_expr = SimplifyExpr(op->expr_);
    if (new_expr.get() == op->expr_.get()) return op;
    auto result = MutableCopy(op);
    result->expr_ = new_expr;
    return result;
  }

 private:
  /// Compose var-remap (via the base-class `var_remap_`) with analyzer-based
  /// constant folding — the Analyzer only knows about its own bindings and
  /// ignores our Var rebuilds, so remap must run first.
  ExprPtr SimplifyExpr(const ExprPtr& e) {
    if (!e) return e;
    return analyzer_->Simplify(VisitExpr(e));
  }

  std::vector<ExprPtr> SimplifyExprVec(const std::vector<ExprPtr>& vec, bool* changed) {
    return RebuildVec(vec, [this](const ExprPtr& e) { return SimplifyExpr(e); }, changed);
  }

  /// Map @p rebuild over @p vec; sets *changed if any element's identity
  /// differs from the input.
  template <typename Ptr, typename F>
  static std::vector<Ptr> RebuildVec(const std::vector<Ptr>& vec, F&& rebuild, bool* changed) {
    std::vector<Ptr> out;
    out.reserve(vec.size());
    for (const auto& x : vec) {
      auto nx = rebuild(x);
      if (nx.get() != x.get()) *changed = true;
      out.push_back(std::move(nx));
    }
    return out;
  }

  /// Rebuild a TensorType or TileType with every embedded ExprPtr (shape,
  /// stride, valid_shape, start_offset) passed through `SimplifyExpr`.
  /// Returns the original TypePtr if nothing changed.
  TypePtr SimplifyType(const TypePtr& type) {
    if (!type) return type;
    if (auto t = As<TensorType>(type)) {
      bool changed = false;
      auto new_shape = SimplifyExprVec(t->shape_, &changed);
      std::optional<TensorView> new_tv = t->tensor_view_;
      if (t->tensor_view_.has_value()) {
        const auto& tv = *t->tensor_view_;
        bool view_changed = false;
        auto new_stride = SimplifyExprVec(tv.stride, &view_changed);
        auto new_vs = SimplifyExprVec(tv.valid_shape, &view_changed);
        if (view_changed) {
          changed = true;
          new_tv = TensorView(std::move(new_stride), tv.layout, std::move(new_vs), tv.pad);
        }
      }
      if (!changed) return type;
      return std::make_shared<TensorType>(std::move(new_shape), t->dtype_, t->memref_, std::move(new_tv));
    }
    if (auto t = As<TileType>(type)) {
      bool changed = false;
      auto new_shape = SimplifyExprVec(t->shape_, &changed);
      std::optional<TileView> new_tv = t->tile_view_;
      if (t->tile_view_.has_value()) {
        const auto& tv = *t->tile_view_;
        bool view_changed = false;
        auto new_vs = SimplifyExprVec(tv.valid_shape, &view_changed);
        auto new_stride = SimplifyExprVec(tv.stride, &view_changed);
        auto new_offset = tv.start_offset ? SimplifyExpr(tv.start_offset) : tv.start_offset;
        if (new_offset.get() != tv.start_offset.get()) view_changed = true;
        if (view_changed) {
          changed = true;
          TileView ntv = tv;
          ntv.valid_shape = std::move(new_vs);
          ntv.stride = std::move(new_stride);
          ntv.start_offset = std::move(new_offset);
          new_tv = std::move(ntv);
        }
      }
      if (!changed) return type;
      return std::make_shared<TileType>(std::move(new_shape), t->dtype_, t->memref_, std::move(new_tv),
                                        t->memory_space_);
    }
    if (auto t = As<TupleType>(type)) {
      bool changed = false;
      std::vector<TypePtr> new_types;
      new_types.reserve(t->types_.size());
      for (const auto& inner : t->types_) {
        auto new_inner = SimplifyType(inner);
        if (new_inner.get() != inner.get()) changed = true;
        new_types.push_back(std::move(new_inner));
      }
      if (!changed) return type;
      return std::make_shared<TupleType>(std::move(new_types));
    }
    return type;
  }

  static bool IsConstScalar(const ExprPtr& e) {
    return e && (As<ConstInt>(e) || As<ConstFloat>(e) || As<ConstBool>(e));
  }

  /// Rebuild a Var with a simplified type, recording the remap so downstream
  /// VarExpr references pick up the new identity. If the Var was already
  /// rebuilt earlier (e.g., at its defining AssignStmt during the body
  /// traversal), return that existing remap so ForStmt.return_vars_ stays
  /// identical to the body-side definition.
  VarPtr MaybeRebuildVar(const VarPtr& var) {
    if (!var) return var;
    if (auto existing = LookupVarRemap(var.get())) return existing;
    auto new_type = SimplifyType(var->GetType());
    if (new_type.get() == var->GetType().get()) return var;
    auto new_var = std::make_shared<Var>(var->name_hint_, new_type, var->span_);
    var_remap_[var.get()] = new_var;
    return new_var;
  }

  IterArgPtr MaybeRebuildIterArg(const IterArgPtr& ia) {
    if (!ia) return ia;
    auto new_type = SimplifyType(ia->GetType());
    auto new_init = SimplifyExpr(ia->initValue_);
    if (new_type.get() == ia->GetType().get() && new_init.get() == ia->initValue_.get()) return ia;
    auto new_ia = std::make_shared<IterArg>(ia->name_hint_, new_type, new_init, ia->span_);
    var_remap_[ia.get()] = new_ia;
    return new_ia;
  }

  VarPtr LookupVarRemap(const Var* key) {
    auto it = var_remap_.find(key);
    if (it == var_remap_.end()) return nullptr;
    return std::dynamic_pointer_cast<const Var>(it->second);
  }

  std::unordered_set<const Var*> multi_assigned_;
};

FunctionPtr TransformSimplify(const FunctionPtr& func) {
  MultiAssignCollector collector;
  collector.VisitStmt(func->body_);

  auto analyzer = std::make_shared<arith::Analyzer>();
  SimplifyMutator mutator(analyzer.get(), std::move(collector.multi_assigned));
  auto new_body = mutator.VisitStmt(func->body_);

  // Final step: conservative scalar DCE prunes scalar bindings whose only
  // uses were folded out by the mutator above. Call-backed assignments are
  // preserved because the IR has no purity annotations yet — a Call may
  // have observable side effects we cannot reason about.
  auto flat = transform_utils::FlattenToStmts(new_body);
  auto pruned = dce::EliminateDeadScalarAssignments(flat);
  bool dce_changed = pruned.size() != flat.size() ||
                     !std::equal(pruned.begin(), pruned.end(), flat.begin(),
                                 [](const StmtPtr& a, const StmtPtr& b) { return a.get() == b.get(); });
  StmtPtr final_body = dce_changed ? loop_repair::MakeBody(pruned, new_body->span_) : new_body;

  if (final_body.get() == func->body_.get()) return func;
  auto result = MutableCopy(func);
  result->body_ = final_body;
  return result;
}

}  // namespace

namespace pass {

Pass Simplify() { return CreateFunctionPass(TransformSimplify, "Simplify", kSimplifyProperties); }

}  // namespace pass

}  // namespace ir
}  // namespace pypto
