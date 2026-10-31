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
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/transforms/utils/op_predicates.h"
#include "pypto/ir/transforms/utils/spmd_launch_utils.h"
#include "src/ir/transforms/lower_composite/lower_composite_builder.h"
#include "src/ir/transforms/lower_composite/lower_composite_rules.h"

namespace pypto {
namespace ir {

namespace {

using lower_composite::CompositeLoweringFn;
using lower_composite::LookupCompositeRule;
using lower_composite::LoweringBuilder;

// ============================================================================
// LowerCompositeOpsMutator
//
// Generic dispatcher: for every ``var = Call(...)`` AssignStmt (or composite-op
// Call embedded directly in a ReturnStmt), look up a lowering rule via
// ``LookupCompositeRule`` and, if found, replace the statement with a SeqStmts
// containing the rule's primitive decomposition. All other statements pass
// through to the base IRMutator, so the pass is a structural no-op on programs
// that contain no registered composite ops.
//
// The pass is idempotent provided each rule emits only ops that are not
// themselves registered (see lower_composite_rules.cpp).
// ============================================================================
class LowerCompositeOpsMutator : public IRMutator {
 public:
  explicit LowerCompositeOpsMutator(bool skip_managed_collectives = false)
      : skip_managed_collectives_(skip_managed_collectives) {}

  ExprPtr VisitExpr_(const TupleGetItemExprPtr& op) override {
    auto tuple = VisitExpr(op->tuple_);
    // Prefer composite-produced MakeTuples recorded privately — do not rely on
    // global var_remap_ for arbitrary `v = (a, b)` assignments.
    if (auto values = ResolveCompositeTuple(tuple)) {
      INTERNAL_CHECK_SPAN(op->index_ >= 0 && static_cast<size_t>(op->index_) < values->elements_.size(),
                          op->span_)
          << "Tuple index out of range: " << op->index_;
      return values->elements_[static_cast<size_t>(op->index_)];
    }
    if (tuple.get() != op->tuple_.get()) {
      return std::make_shared<TupleGetItemExpr>(tuple, op->index_, op->span_);
    }
    return IRMutator::VisitExpr_(op);
  }

  StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
    auto call = As<Call>(op->value_);
    if (!call) {
      auto visited = IRMutator::VisitStmt_(op);
      auto assign = As<AssignStmt>(visited);
      // Propagate aliases of composite-produced tuples (e.g. alias = pair).
      // Prefer composite_tuples_; if VisitExpr already expanded the RHS to a
      // MakeTuple via var_remap_, also seed var_remap_ for the alias Var so
      // later projections and ConvertToSSA see a concrete tuple.
      if (assign) {
        if (auto mt = ResolveCompositeTuple(assign->value_)) {
          composite_tuples_[op->var_.get()] = mt;
          if (As<MakeTuple>(assign->value_)) {
            var_remap_[op->var_.get()] = assign->value_;
          }
        }
      }
      return visited;
    }
    CompositeLoweringFn rule = LookupRule(call);
    if (!rule) {
      return IRMutator::VisitStmt_(op);
    }

    // Apply var_remap_ (if any) to operand expressions before handing them
    // to the rule.
    std::vector<ExprPtr> visited_args = VisitArgs(call->args_, op->span_);

    LoweringBuilder builder(op->var_->name_hint_, temp_counter_);
    ExprPtr result = rule(call, visited_args, builder);
    if (auto mt = As<MakeTuple>(result)) {
      // Record privately for TupleGetItem folding, and also seed var_remap_ so
      // SSA aliases (`alias = pair`) expand through VisitExpr_(Var) without
      // needing a global "any MakeTuple" remap.
      composite_tuples_[op->var_.get()] = mt;
      var_remap_[op->var_.get()] = result;
    }

    auto stmts = builder.TakeStmts();
    // Bind the final result to the original target Var (preserves uses
    // downstream — original AssignStmt's var keeps its name and identity).
    auto final_assign = MutableCopy(op);
    final_assign->value_ = result;
    stmts.push_back(std::move(final_assign));

    if (stmts.size() == 1) return stmts.front();
    return std::make_shared<SeqStmts>(std::move(stmts), op->span_);
  }

  StmtPtr VisitStmt_(const EvalStmtPtr& op) override {
    auto call = As<Call>(op->expr_);
    CompositeLoweringFn rule = call ? LookupRule(call) : nullptr;
    if (!rule) {
      return IRMutator::VisitStmt_(op);
    }

    std::vector<ExprPtr> visited_args = VisitArgs(call->args_, op->span_);

    LoweringBuilder builder("eval", temp_counter_);
    static_cast<void>(rule(call, visited_args, builder));

    auto stmts = builder.TakeStmts();
    if (stmts.empty()) return op;
    if (stmts.size() == 1) return stmts.front();
    return std::make_shared<SeqStmts>(std::move(stmts), op->span_);
  }

  // In SSA form (which LowerCompositeOps assumes), every Call is bound to an
  // AssignStmt and ReturnStmt::value_ holds only Vars — the override above is
  // the sole rewrite site. Standalone / pre-SSA invocations of the pass can
  // still surface a composite-op Call directly inside ReturnStmt::value_
  // (e.g. ``return pl.tile.sin(x)``); without this override those would slip
  // through unlowered. The override lifts each registered Call into a SeqStmts
  // whose last statement is the (possibly mutated) ReturnStmt referencing
  // fresh result Vars.
  StmtPtr VisitStmt_(const ReturnStmtPtr& op) override {
    std::vector<StmtPtr> prelude;
    std::vector<ExprPtr> new_values;
    new_values.reserve(op->value_.size());
    bool changed = false;

    for (std::size_t i = 0; i < op->value_.size(); ++i) {
      INTERNAL_CHECK_SPAN(op->value_[i], op->span_) << "ReturnStmt has null value at index " << i;
      ExprPtr value = op->value_[i];
      auto call = As<Call>(value);
      CompositeLoweringFn rule = call ? LookupRule(call) : nullptr;
      if (rule) {
        std::vector<ExprPtr> visited_args = VisitArgs(call->args_, op->span_);
        const std::string base = "ret" + std::to_string(i);
        LoweringBuilder builder(base, temp_counter_);
        ExprPtr decomposed = rule(call, visited_args, builder);
        // Bind the decomposed result to a fresh Var so ReturnStmt::value_
        // continues to hold a Var (matches the SSA invariant the rest of the
        // pipeline expects). The Bind appends to the same builder, so a single
        // TakeStmts() drains the rule's prelude + the result binding.
        auto result_var = builder.Bind("result", decomposed, call->span_);
        for (auto& s : builder.TakeStmts()) prelude.push_back(std::move(s));
        new_values.push_back(result_var);
        changed = true;
      } else {
        ExprPtr new_expr = VisitExpr(value);
        INTERNAL_CHECK_SPAN(new_expr, op->span_) << "ReturnStmt value at index " << i << " mutated to null";
        new_values.push_back(new_expr);
        if (new_expr.get() != value.get()) {
          changed = true;
        }
      }
    }

    if (!changed) return op;

    auto new_return = MutableCopy(op);
    new_return->value_ = std::move(new_values);
    if (prelude.empty()) return new_return;
    prelude.push_back(std::move(new_return));
    return std::make_shared<SeqStmts>(std::move(prelude), op->span_);
  }

 private:
  /// True for every ``pld.tensor.*`` cross-rank collective, via the shared
  /// predicate — the CHIP rail's post-condition and the orchestration-reference
  /// verifier must agree with this set, so all three read one list.
  [[nodiscard]] static bool IsTensorCollective(const CallPtr& call) {
    return call && op_predicates::IsManagedTensorCollective(call->op_);
  }

  [[nodiscard]] static bool ShouldSkipManagedCollective(const CallPtr& call) {
    // Managed vs InCore is a function-context property, decided authoritatively
    // by the outer skip_managed_collectives_ flag (set for every Orchestration
    // function), not by arg count or arg[0] type.  Every collective is skipped
    // uniformly here so the flag alone governs which functions defer lowering.
    return IsTensorCollective(call);
  }

  [[nodiscard]] CompositeLoweringFn LookupRule(const CallPtr& call) const {
    if (skip_managed_collectives_ && ShouldSkipManagedCollective(call)) {
      return nullptr;
    }
    return call && call->op_ ? LookupCompositeRule(call->op_->name_) : nullptr;
  }

  std::vector<ExprPtr> VisitArgs(const std::vector<ExprPtr>& args, const Span& span) {
    std::vector<ExprPtr> out;
    out.reserve(args.size());
    for (const auto& arg : args) {
      auto visited = VisitExpr(arg);
      INTERNAL_CHECK_SPAN(visited, span) << "Call argument mutated to null during composite-op lowering";
      out.push_back(std::move(visited));
    }
    return out;
  }

  /// Resolve an expression to a MakeTuple produced by this pass (including
  /// aliases recorded in composite_tuples_). Returns nullptr if not found.
  MakeTuplePtr ResolveCompositeTuple(const ExprPtr& expr) const {
    if (auto mt = As<MakeTuple>(expr)) return mt;
    if (auto var = AsVarLike(expr)) {
      auto it = composite_tuples_.find(var.get());
      if (it != composite_tuples_.end()) return it->second;
    }
    return nullptr;
  }

  std::size_t temp_counter_ = 0;
  bool skip_managed_collectives_{false};
  /// MakeTuples produced by composite lowering rules (and their SSA aliases).
  /// Used only to fold TupleGetItem projections; does not affect global var_remap_.
  std::unordered_map<const Expr*, MakeTuplePtr> composite_tuples_;
};

/// A managed collective is one written in an *orchestration* body — HOST/L3 or
/// CHIP/L2. Neither is tile-lowered, so the composite expansion (which emits
/// tensor-level put/notify/wait) would be illegal there; both defer to their
/// own rail: LowerHostTensorCollectives for HOST, LowerL2TensorCollectives for
/// CHIP. Only an InCore body still expands here.
[[nodiscard]] bool IsOrchestrationBody(const FunctionPtr& func) {
  return func && (func->func_type_ == FunctionType::Orchestration ||
                  (func->role_.has_value() && *func->role_ == Role::Orchestrator));
}

FunctionPtr TransformLowerCompositeOps(const FunctionPtr& func) {
  LowerCompositeOpsMutator mutator(IsOrchestrationBody(func));
  return mutator.VisitFunction(func);
}

/// The first InCore composite collective in `func`'s own body, or null.
CallPtr CompositeInOwnBody(const FunctionPtr& func) {
  if (!func || !func->body_) return nullptr;
  CallSubmitCollector collector;
  collector.VisitStmt(func->body_);
  for (const auto& call : collector.calls) {
    if (op_predicates::IsManagedTensorCollective(call->op_)) return call;
  }
  return nullptr;
}

/// The first InCore composite collective that `func` would expand, in its own
/// body or in a function it calls, or null. Orchestration bodies are skipped:
/// their collectives are managed and lowered by their own rail. Memoized per
/// function; a function already in `memo` (finished, or on the current call
/// path) is not re-entered, so call cycles terminate and each body is walked once.
CallPtr FindCompositeCollective(const FunctionPtr& func, const ProgramPtr& program,
                                std::unordered_map<const Function*, CallPtr>* memo) {
  if (!func || !func->body_ || IsOrchestrationBody(func)) return nullptr;
  if (!memo->try_emplace(func.get(), nullptr).second) return memo->at(func.get());
  CallPtr found = CompositeInOwnBody(func);
  if (!found) {
    for (const auto& callee : DirectCallees(func, program)) {
      found = FindCompositeCollective(callee, program, memo);
      if (found) break;
    }
  }
  memo->at(func.get()) = found;
  return found;
}

/// Reject an InCore composite collective inside a multi-block SPMD launch.
///
/// The composite expansion never reads the block index, so each of N blocks
/// would run the whole collective: N times the transfers, and N times the
/// barrier credits against a wait that expects one, so a barrier can release
/// before every block's data has landed. Multi-core collectives belong to the
/// HOST rail's core_num and the managed CHIP/L2 rail. Only a literal width of 1
/// is accepted; a runtime width cannot be proven to be 1. That exemption covers
/// a collective inside a launched kernel; one in the launch wrapper's own body
/// (or a nested Group/Spmd wrapper it dispatches) is rejected at any width,
/// because codegen emits only the inner call and drops the rest of the body.
void RejectCompositesInSpmdLaunch(const ProgramPtr& program) {
  std::unordered_map<const Function*, CallPtr> memo;
  auto reject_wrapper_local = [](const CallPtr& collective, const FunctionPtr& caller) {
    CHECK_SPAN(false, collective->span_)
        << collective->op_->name_ << " runs directly in the SPMD launch wrapper body (dispatched from '"
        << caller->name_
        << "'), not inside the launched kernel. Orchestration codegen submits only the wrapper's inner "
           "kernel call and does not emit the rest of the body, so this collective would be silently "
           "dropped. Call it outside the SPMD launch.";
  };
  auto reject_in_launch = [&](const CallPtr& collective, const SpmdLaunchSite& site,
                              const ConstIntPtr& width) {
    CHECK_SPAN(false, collective->span_)
        << collective->op_->name_ << " runs inside an SPMD launch of "
        << (width ? std::to_string(width->value_) + " blocks" : std::string("a runtime number of blocks"))
        << " (dispatched from '" << site.caller->name_
        << "'), but InCore collectives are single-block: every block would repeat the whole collective "
           "and the barrier credits would no longer match. Call it outside the SPMD launch, or use a "
           "multi-core path: core_num on the HOST rail, or the managed CHIP/L2 rail "
           "(pld.tensor.all_to_all_v / indexed_all_to_all_v in a CHIP orchestration)";
  };
  auto reject_in_launched_kernel = [&](const FunctionPtr& kernel, const SpmdLaunchSite& site,
                                       const ConstIntPtr& width) {
    auto collective = FindCompositeCollective(kernel, program, &memo);
    if (collective) reject_in_launch(collective, site, width);
  };

  // Wrapper bodies are never emitted: codegen submits only the inner call(s)
  // and drops the rest, so a composite in ANY wrapper body of the chain is
  // silently lost at any width — including one block. A Spmd or Group function
  // is such a wrapper; descend through both, a plain Group included (e.g.
  // `with pl.spmd(...): with pl.cluster():` outlines to Spmd -> Group -> InCore,
  // and that Group is not spmd-unwrapped). `scanned` keeps a shared wrapper to
  // one visit across sites, so each body and callee list is walked once.
  auto is_dispatch_wrapper = [](const FunctionPtr& func) {
    return func && (func->func_type_ == FunctionType::Spmd || func->func_type_ == FunctionType::Group);
  };
  std::unordered_set<const Function*> scanned;
  ForEachSpmdLaunch(program, [&](const SpmdLaunchSite& site) {
    std::vector<FunctionPtr> work;
    if (is_dispatch_wrapper(site.launch_wrapper)) work.push_back(site.launch_wrapper);
    for (const auto& kernel : site.kernels) {
      if (is_dispatch_wrapper(kernel)) work.push_back(kernel);
    }
    while (!work.empty()) {
      FunctionPtr wrapper = work.back();
      work.pop_back();
      if (!wrapper || !scanned.insert(wrapper.get()).second) continue;
      if (auto collective = CompositeInOwnBody(wrapper)) reject_wrapper_local(collective, site.caller);
      for (const auto& callee : DirectCallees(wrapper, program)) {
        if (is_dispatch_wrapper(callee)) work.push_back(callee);
      }
    }
    // A collective dispatched directly by a scope that has not been outlined yet
    // is classified by the carrier outlining would give its body: under an
    // InCore scope that body becomes the launched kernel (kernel rule, one block
    // accepted); otherwise it becomes the launch wrapper (dropped by codegen at
    // any width, one block included).
    if (site.unoutlined && site.dispatch_call &&
        op_predicates::IsManagedTensorCollective(site.dispatch_call->op_)) {
      const ConstIntPtr scope_width = As<ConstInt>(site.core_num);
      const bool single_block = scope_width && scope_width->value_ == 1;
      if (!site.unoutlined_kernel_body) reject_wrapper_local(site.dispatch_call, site.caller);
      if (!single_block) reject_in_launch(site.dispatch_call, site, scope_width);
    }
    // A composite inside an executable kernel is accepted at width 1 because it
    // runs exactly once; a runtime width cannot be proven to be 1.
    auto width = As<ConstInt>(site.core_num);
    if (width && width->value_ == 1) return;
    for (const auto& kernel : site.kernels) reject_in_launched_kernel(kernel, site, width);
  });
}

ProgramPtr TransformLowerCompositeOpsProgram(const ProgramPtr& program) {
  RejectCompositesInSpmdLaunch(program);
  std::vector<FunctionPtr> functions;
  functions.reserve(program->functions_.size());
  for (const auto& [gvar, func] : program->functions_) {
    (void)gvar;
    functions.push_back(TransformLowerCompositeOps(func));
  }
  return std::make_shared<const Program>(functions, program->name_, program->span_);
}

}  // namespace

namespace pass {

Pass LowerCompositeOps() {
  return CreateProgramPass(TransformLowerCompositeOpsProgram, "LowerCompositeOps",
                           kLowerCompositeOpsProperties);
}

}  // namespace pass

}  // namespace ir
}  // namespace pypto
