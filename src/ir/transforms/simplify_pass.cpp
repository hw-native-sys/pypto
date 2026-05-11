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
#include <cstddef>
#include <memory>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/core/logging.h"
#include "pypto/ir/arith/analyzer.h"
#include "pypto/ir/arith/ir_mutator_with_analyzer.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/dead_code_elimination.h"
#include "pypto/ir/transforms/utils/deep_clone_utils.h"
#include "pypto/ir/transforms/utils/loop_state_repair.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/transforms/utils/tensor_view_semantics.h"
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

/// Strip the trailing YieldStmt from @p body and return both the stripped
/// body and the yielded values that the caller should bind into its
/// var_remap_ as `return_vars[i] → yielded_values[i]`. Used by control-flow
/// folds (Fold A for IfStmt, Fold B for ForStmt) that lift a kept branch /
/// unrolled body into the parent scope.
///
/// Substituting return_vars via var_remap_ (rather than emitting a literal
/// `AssignStmt(return_var, yielded)`) avoids creating alias assignments that
/// the orchestration codegen's role-aware name disambiguation cannot lower
/// cleanly — e.g. `out__rv_v2 = out__co_l0_rv_v3` printing as
/// `auto out = out;` because both vars share the role-derived base name
/// `out`. The substituted form lets subsequent SimplifyMutator visits
/// (including the function's ReturnStmt) read the yielded value directly.
///
/// Recurses through trailing SeqStmts to find the yield, mirroring
/// `transform_utils::GetLastYieldStmt` — a well-formed control-flow body
/// can place its terminating YieldStmt inside a nested SeqStmts wrapper.
///
/// Pre: @p body is a well-formed control-flow body with `return_vars`
/// non-empty — its (possibly nested) tail is a YieldStmt with
/// `value_.size() == return_vars.size()`.
struct StrippedYield {
  StmtPtr body;                         ///< @p body with the trailing YieldStmt removed.
  std::vector<ExprPtr> yielded_values;  ///< Values from the removed YieldStmt, one per return_var.
};

StrippedYield StripTrailingYield(const StmtPtr& body, size_t return_var_count) {
  if (auto seq = As<SeqStmts>(body)) {
    INTERNAL_CHECK_SPAN(!seq->stmts_.empty(), seq->span_)
        << "Internal error: control-flow body must end with YieldStmt "
           "when return_vars is non-empty";
    auto stripped = seq->stmts_;
    auto inner = StripTrailingYield(stripped.back(), return_var_count);
    if (inner.body) {
      stripped.back() = inner.body;
    } else {
      stripped.pop_back();
    }
    return {loop_repair::MakeBody(stripped, seq->span_), std::move(inner.yielded_values)};
  }
  auto yield_stmt = std::dynamic_pointer_cast<const YieldStmt>(body);
  INTERNAL_CHECK_SPAN(yield_stmt, body->span_)
      << "Internal error: control-flow body tail must be YieldStmt when "
         "return_vars is non-empty";
  INTERNAL_CHECK_SPAN(yield_stmt->value_.size() == return_var_count, yield_stmt->span_)
      << "Internal error: yielded value count " << yield_stmt->value_.size()
      << " does not match return_vars count " << return_var_count;
  return {/*body=*/nullptr, yield_stmt->value_};
}

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
  ///
  /// Also drops identity ``tensor.as_layout`` reinterprets per RFC #1300 §3.3:
  ///   - ``as_layout(x, x.layout)`` → ``x`` (target layout matches source)
  ///
  /// Chain folding (``as_layout(as_layout(x, L1), L2)`` → ``as_layout(x, L2)``)
  /// is intentionally not done at this layer: after SSA conversion the outer
  /// Call references its inner result via a Var binding, not inline, so a
  /// naive pointer inspection cannot see across the binding. A dedicated
  /// SSA-aware chain optimizer can be added if a real pipeline produces such
  /// chains.
  ExprPtr VisitExpr_(const CallPtr& op) override {
    auto base = IRMutator::VisitExpr_(op);
    auto call = std::dynamic_pointer_cast<const Call>(base);
    if (call && call->op_ && call->op_->name_ == "tensor.as_layout") {
      base = SimplifyAsLayout(call);
    }
    auto new_type = SimplifyType(base->GetType());
    if (new_type.get() == base->GetType().get()) return base;
    call = std::dynamic_pointer_cast<const Call>(base);
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

    // Fold B: collapse a *pure* sequential ForStmt when the analyzer can
    // prove the trip count is 0 or 1. "Pure" means no chunk_config, no attrs,
    // and Sequential kind — these forms have no execution-model side effects
    // (no chunk-loop pairing, no Parallel/Unroll/Pipeline scheduling) that a
    // downstream pass might depend on observing as a ForStmt.
    //
    // Trip-count proof:
    //   step >= 1 (forward iteration)
    //   AND (stop <= start            → 0 trips)
    //   OR  (start < stop             AND
    //        stop  <= start + step    → 1 trip)
    //
    // CanProve is used instead of literal-only ConstInt matching so closure-
    // captured trip counts like `for k in pl.range(C, C + 1)` collapse the
    // same way `pl.range(0, 1)` does.
    const bool is_pure_for =
        !op->chunk_config_.has_value() && op->attrs_.empty() && op->kind_ == ForKind::Sequential;
    if (is_pure_for && analyzer_->CanProveGreaterEqual(new_step, 1)) {
      const Span& sp = op->span_;
      int trips = -1;
      if (analyzer_->CanProve(MakeLe(new_stop, new_start, sp))) {
        trips = 0;
      } else if (analyzer_->CanProve(MakeAnd(MakeLt(new_start, new_stop, sp),
                                             MakeLe(new_stop, MakeAdd(new_start, new_step, sp), sp), sp))) {
        trips = 1;
      }

      if (trips == 0) {
        // Loop body never executes: each return_var takes its iter_arg's init value.
        std::vector<StmtPtr> out;
        out.reserve(op->return_vars_.size());
        for (size_t i = 0; i < op->return_vars_.size(); ++i) {
          auto rv = MaybeRebuildVar(op->return_vars_[i]);
          out.push_back(std::make_shared<AssignStmt>(rv, new_iter_args[i]->initValue_, sp));
        }
        return loop_repair::MakeBody(out, sp);
      }
      if (trips == 1) {
        // Exactly one iteration: substitute loop_var → start and each
        // iter_arg → its init value via DeepClone (matches the substitution
        // pattern used by LoopUnrollMutator in unroll_loops_pass.cpp), then
        // re-visit so further folds happen, then rewrite the trailing
        // YieldStmt into AssignStmts on the return_vars.
        //
        // DeepClone is used (rather than a local var_remap_ override + analyzer
        // Bind) so the substitution penetrates Var references buried inside
        // type annotations — e.g. MemRef byte_offsets that mention the loop
        // variable. A var_remap_-only approach misses those uses because
        // SimplifyType / SimplifyExpr go through the analyzer at definition
        // sites only when the type is rebuilt; uses inside nested control-flow
        // bodies or tile-view shapes get printed verbatim.
        std::unordered_map<const Var*, ExprPtr> sub_map;
        sub_map.reserve(1 + op->iter_args_.size());
        sub_map.emplace(op->loop_var_.get(), new_start);
        for (size_t i = 0; i < op->iter_args_.size(); ++i) {
          sub_map.emplace(op->iter_args_[i].get(), new_iter_args[i]->initValue_);
        }
        // clone_def_vars=true is load-bearing: MultiAssignCollector ran on the
        // pre-fold IR and marked any scalar Var defined inside nested for-loop
        // bodies as multi_assigned (the safety check that protects pre-SSA
        // callers). After the loop is unrolled and lifted, those Vars are
        // single-assigned at top-level, but multi_assigned_ still references
        // the original Var pointers — analyzer Bind would be skipped on those.
        // Producing fresh Var clones at every DefField breaks that staleness:
        // the cloned Vars are not in multi_assigned_, so subsequent uses of
        // (e.g.) `cur_offset = 0` propagate via analyzer Bind as expected.
        auto cloned = DeepClone(op->body_, sub_map, /*clone_def_vars=*/true);

        // Snapshot var_remap_ around the cloned-body visit. MaybeRebuildVar
        // inserts entries keyed by the cloned-body's defining-Var raw pointers
        // (the freshly-allocated clones); after this Fold returns, those clones
        // become unreachable as soon as the rebuilt AssignStmts replace them,
        // and the heap addresses can be recycled by a later make_shared<Var>
        // (e.g. another sibling Fold B that DeepClones a different body). A
        // recycled address would silently match the stale entry and substitute
        // the new Var with an unrelated value — producing AssignStmts whose
        // LHS Var has the wrong type for the RHS (observed on qwen3_decode's
        // q_proj/up_proj as a `pto.textract` whose dst aliases the matmul Acc
        // tile of an earlier iteration).
        //
        // The lift's own additions (return_vars[i] → yielded_values[i]) are
        // applied to the restored baseline below, so subsequent siblings that
        // reference this ForStmt's return_vars still substitute correctly.
        auto baseline_remap = var_remap_;

        // Re-visit so any algebraic patterns exposed by the substitution
        // (e.g. `0 + 64 → 64`) fold in this same Simplify run.
        auto unrolled_body = VisitStmt(cloned.cloned_body);

        var_remap_ = std::move(baseline_remap);
        return LiftBodyToReturnVars(unrolled_body, op->return_vars_);
      }
    }

    bool bound = start_ci && stop_ci && stop_ci->value_ > start_ci->value_;
    if (bound) {
      analyzer_->Bind(op->loop_var_, start_ci->value_, stop_ci->value_);
    }

    // Snapshot var_remap_ around the body visit. Nested folds inside the body
    // (Fold A on a nested IfStmt or Fold B on a nested single-trip ForStmt)
    // record remaps from outer Vars to body-internal values; leaking those to
    // siblings of this ForStmt or to the parent scope produces dangling
    // references. The MaybeRebuildIterArg additions made above stay in
    // baseline_remap (they're valid in body and after).
    auto baseline_remap = var_remap_;

    auto new_body = VisitStmt(op->body_);

    if (bound) {
      analyzer_->Unbind(op->loop_var_);
    }

    // Discard body-internal remaps — the For is being rebuilt, so the body's
    // SSA scope is preserved and outside code shouldn't see its private vars.
    var_remap_ = std::move(baseline_remap);

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

  StmtPtr VisitStmt_(const SpmdScopeStmtPtr& op) override {
    // Fold the core_num expression whenever the value is compile-time-derivable.
    // Closure-captured Python ints arrive as ConstInt already, but closure
    // arithmetic (e.g. `core_num=MAX // TILE` where both operands are closure
    // ints) may still need one simplify pass after SSA conversion.
    auto new_core_num = SimplifyExpr(op->core_num_);
    auto new_body = VisitStmt(op->body_);
    if (new_core_num.get() == op->core_num_.get() && new_body.get() == op->body_.get()) return op;
    auto result = MutableCopy(op);
    result->core_num_ = new_core_num;
    result->body_ = new_body;
    return result;
  }

  StmtPtr VisitStmt_(const IfStmtPtr& op) override {
    auto new_condition = SimplifyExpr(op->condition_);

    // Snapshot var_remap_ so each branch's internal remap additions stay
    // scoped to that branch. Without this, a remap recorded inside the
    // then-branch (e.g. by Fold A on a nested IfStmt, or by Fold B on a
    // nested single-trip ForStmt) would still be live when the else-branch
    // is visited — substituting refs in else with values that name vars
    // defined inside the then-branch's lifted body, producing IR with
    // dangling references and type-mismatched assignments.
    //
    // Each branch is processed from the same baseline; its additions are
    // captured separately. When Fold A keeps a branch, that branch's snapshot
    // is adopted (the lifted body's defs become visible in the parent scope).
    // When the IfStmt is rebuilt, both snapshots are discarded — branch
    // bodies remain in their own scopes.
    auto baseline_remap = var_remap_;

    // Enter constraint scope for then branch (condition is known true).
    StmtPtr new_then;
    {
      auto ctx = analyzer_->GetConstraintContext(new_condition);
      new_then = VisitStmt(op->then_body_);
    }
    auto then_remap = std::move(var_remap_);
    var_remap_ = baseline_remap;

    // Enter constraint scope for else branch (condition is known false → Not(condition)).
    std::optional<StmtPtr> new_else;
    if (op->else_body_.has_value()) {
      auto ctx = analyzer_->GetConstraintContext(MakeNot(new_condition, new_condition->span_));
      new_else = VisitStmt(*op->else_body_);
    }
    auto else_remap = std::move(var_remap_);
    var_remap_ = baseline_remap;

    // Fold A: collapse the IfStmt when the analyzer can prove the polarity.
    // True  → keep then_body_; drop else.
    // False → keep else_body_ if present, else replace with empty body
    //         (which is only valid when return_vars_ is empty per IR contract).
    // When return_vars_ is non-empty, the kept branch's trailing YieldStmt is
    // rewritten into AssignStmt(return_vars[i], yielded_value[i]) so SSA
    // references after the IfStmt remain well-defined.
    //
    // CanProve is used instead of an `As<ConstBool>` check because the
    // analyzer's constraint stack (loop-var bounds, scalar bindings, outer
    // if-branch constraints) often establishes the polarity even when
    // SimplifyExpr leaves the condition symbolic.
    const bool always_true = analyzer_->CanProve(new_condition);
    const bool always_false =
        !always_true && analyzer_->CanProve(MakeNot(new_condition, new_condition->span_));
    if (always_true || always_false) {
      StmtPtr kept;
      if (always_true) {
        kept = new_then;
        var_remap_ = std::move(then_remap);  // adopt kept branch's remaps
      } else if (new_else.has_value()) {
        kept = *new_else;
        var_remap_ = std::move(else_remap);
      } else {
        INTERNAL_CHECK_SPAN(op->return_vars_.empty(), op->span_)
            << "Internal error: IfStmt with no else branch must have empty return_vars_";
        return loop_repair::MakeBody({}, op->span_);
      }
      return LiftBodyToReturnVars(kept, op->return_vars_);
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
  /// Identity elimination per RFC #1300 §3.3:
  /// ``as_layout(x, layout=x.layout)`` → ``x``.
  ///
  /// Drops a ``tensor.as_layout`` call when the requested target layout
  /// matches what the source already carries — the call is then a no-op
  /// metadata reinterpret and downstream consumers can use ``src`` directly.
  /// (When layouts differ, ``as_layout`` performs the canonical-pair swap;
  /// such substantive reinterprets are preserved.)
  ///
  /// Chain folding (``as_layout(as_layout(x, L1), L2)`` → ``as_layout(x, L2)``)
  /// is intentionally not implemented here. After SSA the outer Call's arg is
  /// a Var bound to the inner Call (not the inner Call inline), so naive
  /// pointer inspection cannot see across the binding. A dedicated SSA-aware
  /// chain optimizer can be added if real pipelines produce such chains.
  ExprPtr SimplifyAsLayout(const std::shared_ptr<const Call>& call) {
    if (call->args_.size() != 1) return call;
    auto src = call->args_[0];

    auto src_tensor = As<TensorType>(src->GetType());
    auto out_tensor = As<TensorType>(call->GetType());
    if (!src_tensor || !out_tensor) return call;

    // Bare TensorType is implicitly ND-packed.
    TensorLayout src_layout =
        src_tensor->tensor_view_.has_value() ? src_tensor->tensor_view_->layout : TensorLayout::ND;
    TensorLayout target_layout =
        out_tensor->tensor_view_.has_value() ? out_tensor->tensor_view_->layout : TensorLayout::ND;
    if (src_layout == target_layout) {
      return src;
    }
    return call;
  }

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

  /// Lift @p kept_body into the parent scope when a control-flow fold has
  /// chosen it as the surviving branch. Strips the trailing YieldStmt and
  /// records `return_vars[i] → yielded_value[i]` in `var_remap_` so any
  /// downstream uses (subsequent siblings, ReturnStmt) read the yielded
  /// value directly. Shared by Fold A (IfStmt) and the one-trip case of
  /// Fold B (ForStmt).
  ///
  /// Substituting via var_remap_ (rather than emitting `AssignStmt(rv, val)`)
  /// avoids alias assignments that the orchestration codegen lowers
  /// incorrectly when both sides derive from a role-tagged parameter name
  /// (e.g. both vars have base name "out", producing `auto out = out;`).
  StmtPtr LiftBodyToReturnVars(const StmtPtr& kept_body, const std::vector<VarPtr>& return_vars) {
    if (return_vars.empty()) return kept_body;
    auto stripped = StripTrailingYield(kept_body, return_vars.size());
    for (size_t i = 0; i < return_vars.size(); ++i) {
      var_remap_[return_vars[i].get()] = stripped.yielded_values[i];
    }
    return stripped.body ? stripped.body : loop_repair::MakeBody({}, kept_body->span_);
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
