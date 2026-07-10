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

#include <any>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/core/dtype.h"
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
#include "pypto/ir/transforms/utils/attrs.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/transforms/utils/return_lineage_utils.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {
namespace pass {

namespace {

/// Per-iter_arg carry lowering plan. Serialised onto ``ForStmt::attrs_``.
struct IterArgCarryPlan {
  /// True when the yield value is not in the iter_arg's alias class (or TaskId).
  bool is_rebind = false;
  /// TaskId manual-scope array-carry extent; 0 means scalar/tensor/ArrayType path.
  int64_t array_size = 0;
};

/// AssignStmts and nested ForStmts collected from a loop body (at any depth).
struct BodyAliases {
  std::vector<AssignStmtPtr> assigns;
  std::vector<ForStmtPtr> nested_fors;
};

BodyAliases CollectBodyAliases(const StmtPtr& body) {
  class AliasingNodeCollector : public IRVisitor {
   public:
    BodyAliases result;
    void VisitStmt_(const AssignStmtPtr& a) override {
      result.assigns.push_back(a);
      IRVisitor::VisitStmt_(a);
    }
    void VisitStmt_(const ForStmtPtr& f) override {
      result.nested_fors.push_back(f);
      IRVisitor::VisitStmt_(f);
    }
  };
  AliasingNodeCollector collector;
  collector.VisitStmt(body);
  return collector.result;
}

/// Find a ForStmt within ``body`` whose ``return_vars_`` contains ``target``.
/// Returns nullptr if none. Used to chase Sequential→Parallel array threading.
ForStmtPtr FindForStmtByReturnVar(const StmtPtr& body, const Var* target) {
  class Finder : public IRVisitor {
   public:
    ForStmtPtr result;
    const Var* target = nullptr;
    void VisitStmt_(const ForStmtPtr& f) override {
      if (result) return;
      for (const auto& rv : f->return_vars_) {
        if (rv.get() == target) {
          result = f;
          return;
        }
      }
      IRVisitor::VisitStmt_(f);
    }
  };
  Finder finder;
  finder.target = target;
  finder.VisitStmt(body);
  return finder.result;
}

/// Alias forest over a loop body: maps each Var that is *another name for an
/// existing buffer* to the Var it aliases.
///
/// A Var ``X`` belongs to iter_arg ``A``'s alias class exactly when following
/// ``X → source(X) → …`` lands on ``A``. Four rules produce an edge:
///
///   * ``tensor.assemble``: the result aliases its first arg (the write target).
///   * Output_existing/inout calls: the result aliases the Out/InOut arg the
///     callee actually returns (traced via ``return_lineage``, so a kernel with
///     a GM-scratch Out param does not capture the alias).
///   * ``TupleGetItemExpr``: climb to the tuple-producing call/submit and
///     resolve the corresponding output arg.
///   * Nested ForStmts: a carry threaded through a nested loop re-emerges as
///     the nested loop's return_var, which aliases that loop's init value.
///
/// Every Var is written by at most one AssignStmt (SSA) and each rule yields at
/// most one source, so the edges form a forest and the class query is a memoized
/// chain walk — no fixpoint. (The assemble rule and the call-direction rule
/// cannot both fire: ``tensor.assemble`` is a builtin op, and DeriveCallDirections
/// stamps ``arg_directions`` on non-builtin calls only.)
class AliasForest {
 public:
  AliasForest(const ForStmtPtr& for_stmt, const BodyAliases& body_aliases, const ProgramPtr& program) {
    // The loop's own iter_args are the class roots — never let an edge lead out
    // of one, so ``ClassRoot(iter_arg) == iter_arg`` holds unconditionally.
    for (const auto& iter_arg : for_stmt->iter_args_) {
      roots_.insert(iter_arg.get());
    }

    // Index assignments by produced var so the TupleGetItemExpr rule can climb
    // tuple chains in O(log N) instead of rescanning the body.
    std::unordered_map<const Var*, AssignStmtPtr> var_to_assign;
    var_to_assign.reserve(body_aliases.assigns.size());
    for (const auto& a : body_aliases.assigns) {
      var_to_assign[a->var_.get()] = a;
    }

    for (const auto& assign : body_aliases.assigns) {
      if (auto source = ResolveAssignAliasSource(assign, var_to_assign, program)) {
        AddEdge(assign->var_.get(), source);
      }
    }

    // Nested ForStmts: the parent's carry threaded through a nested loop comes
    // out via the nested loop's return_var.
    //
    // ArrayType iter_args are EXCLUDED from this propagation: unlike TensorType
    // (a pointer-to-buffer alias), an ArrayType iter_arg owns a *fresh* C-stack
    // array at each level. Treating the inner rv as an alias of the outer
    // iter_arg would mis-mark the outer slot as ``is_rebind=false`` (silently
    // dropping the outer's yield-back copy, which is the very mechanism that
    // propagates state across phases in a SEQ x PARALLEL phase fence). The outer
    // carry must be a distinct backing array and the outer yield must emit an
    // explicit array-array copy back into it.
    for (const auto& nf : body_aliases.nested_fors) {
      for (size_t k = 0; k < nf->iter_args_.size() && k < nf->return_vars_.size(); ++k) {
        if (As<ArrayType>(nf->iter_args_[k]->GetType())) continue;
        auto init_var = AsVarLike(nf->iter_args_[k]->initValue_);
        if (!init_var) continue;
        AddEdge(nf->return_vars_[k].get(), init_var.get());
      }
    }
  }

  /// True when ``var`` belongs to ``iter_arg``'s alias class.
  bool InClassOf(const Var* var, const Var* iter_arg) { return ClassRoot(var) == iter_arg; }

 private:
  void AddEdge(const Var* produced, const Var* source) {
    if (!produced || !source || produced == source) return;
    if (roots_.count(produced)) return;
    source_of_.emplace(produced, source);  // first rule wins
  }

  /// Terminal Var of ``var``'s alias chain, memoized. A cycle (impossible under
  /// SSA, but cheap to guard) terminates at its entry node.
  const Var* ClassRoot(const Var* var) {
    if (!var) return nullptr;
    auto memo_it = root_of_.find(var);
    if (memo_it != root_of_.end()) return memo_it->second;

    std::vector<const Var*> chain;
    const Var* cur = var;
    std::unordered_set<const Var*> on_chain;
    while (on_chain.insert(cur).second) {
      auto memo = root_of_.find(cur);
      if (memo != root_of_.end()) break;
      auto next = source_of_.find(cur);
      if (next == source_of_.end()) break;
      chain.push_back(cur);
      cur = next->second;
    }
    auto memo = root_of_.find(cur);
    const Var* root = memo != root_of_.end() ? memo->second : cur;
    root_of_.emplace(cur, root);
    for (const Var* node : chain) root_of_[node] = root;
    return root;
  }

  /// The single alias source ``assign`` establishes for its LHS, or null.
  static const Var* ResolveAssignAliasSource(
      const AssignStmtPtr& assign, const std::unordered_map<const Var*, AssignStmtPtr>& var_to_assign,
      const ProgramPtr& program) {
    // TupleGetItemExpr: climb to the tuple-producing call or submit and resolve
    // the corresponding output arg. Multi-output InCore kernels return tuples;
    // each ``var = ret_tuple[i]`` extract aliases the i-th output-side arg of
    // the call (using the codegen's own indexing). Submit is viewed as a Call
    // via AsCallOrSubmitView, so its output args alias identically.
    if (auto tge = As<TupleGetItemExpr>(assign->value_)) {
      auto tuple_var = AsVarLike(tge->tuple_);
      if (!tuple_var) return nullptr;
      auto it = var_to_assign.find(tuple_var.get());
      if (it == var_to_assign.end()) return nullptr;
      auto tcall = transform_utils::AsCallOrSubmitView(it->second->value_);
      if (!tcall) return nullptr;
      auto tdirs = tcall->GetArgDirections();
      if (tdirs.size() != tcall->args_.size()) return nullptr;
      int64_t out_seen = 0;
      const auto target_idx = static_cast<int64_t>(tge->index_);
      for (size_t a = 0; a < tdirs.size(); ++a) {
        if (tdirs[a] != ArgDirection::OutputExisting && tdirs[a] != ArgDirection::InOut &&
            tdirs[a] != ArgDirection::Output) {
          continue;
        }
        if (out_seen == target_idx) {
          auto out_arg = AsVarLike(tcall->args_[a]);
          return out_arg ? out_arg.get() : nullptr;
        }
        ++out_seen;
      }
      return nullptr;
    }

    auto call = transform_utils::AsCallOrSubmitView(assign->value_);
    if (!call) return nullptr;

    // tensor.assemble: result var aliases its first arg (the target).
    if (IsOp(call, "tensor.assemble") && !call->args_.empty()) {
      auto first_arg = AsVarLike(call->args_[0]);
      return first_arg ? first_arg.get() : nullptr;
    }

    // Calls with output_existing/inout args (e.g. InCore kernels): the result
    // aliases the Out/InOut arg the callee actually returns, mirroring the
    // codegen alias ``const Tensor& result = args[out_idx];``. For kernels with
    // multiple Out params (e.g. real result + GM scratch passed through pl.spmd
    // mixed dispatch), tracing the ReturnStmt back to its Param avoids aliasing
    // the result to an arbitrary scratch tensor.
    auto call_dirs = call->GetArgDirections();
    if (call_dirs.size() != call->args_.size()) return nullptr;
    FunctionPtr callee = program ? program->GetFunction(call->op_->name_) : nullptr;
    std::optional<size_t> returned_idx = return_lineage::ExplicitReturnedParamIndex(callee);
    for (size_t a = 0; a < call_dirs.size(); ++a) {
      if (call_dirs[a] != ArgDirection::OutputExisting && call_dirs[a] != ArgDirection::InOut) continue;
      if (returned_idx.has_value() && a != *returned_idx) continue;
      auto out_arg = AsVarLike(call->args_[a]);
      return out_arg ? out_arg.get() : nullptr;
    }
    return nullptr;
  }

  std::unordered_set<const Var*> roots_;
  std::unordered_map<const Var*, const Var*> source_of_;
  std::unordered_map<const Var*, const Var*> root_of_;
};

bool IsTaskIdScalar(const VarPtr& var) {
  auto sty = As<ScalarType>(var->GetType());
  return sty && sty->dtype_ == DataType::TASK_ID;
}

/// TaskId array-carry extent for iter_arg ``idx`` of ``for_stmt``.
///
///   * Parallel loop: one slot per iteration, so the extent is the const trip
///     count (0 when the bounds are not compile-time constants).
///   * Sequential loop: the carry is threaded through an inner loop; chase the
///     yield value to the inner ForStmt's matching return_var and recurse.
int64_t ResolveArrayCarrySize(const ForStmtPtr& for_stmt, size_t idx) {
  if (idx >= for_stmt->iter_args_.size()) return 0;
  if (!IsTaskIdScalar(for_stmt->iter_args_[idx])) return 0;
  if (for_stmt->kind_ == ForKind::Parallel) {
    return transform_utils::EvalConstTripCount(for_stmt);
  }
  if (for_stmt->kind_ != ForKind::Sequential) return 0;
  auto yield = transform_utils::GetLastYieldStmt(transform_utils::UnwrapAutoScope(for_stmt->body_));
  if (!yield || idx >= yield->value_.size()) return 0;
  auto yield_var = AsVarLike(yield->value_[idx]);
  if (!yield_var) return 0;
  // Search the raw body (not unwrapped): FindForStmtByReturnVar is a visitor
  // that descends through RuntimeScopeStmt nodes, unlike GetLastYieldStmt.
  auto inner = FindForStmtByReturnVar(for_stmt->body_, yield_var.get());
  if (!inner) return 0;
  for (size_t j = 0; j < inner->return_vars_.size(); ++j) {
    if (inner->return_vars_[j].get() == yield_var.get()) return ResolveArrayCarrySize(inner, j);
  }
  return 0;
}

/// Classify every iter_arg of ``for_stmt`` as trivial or rebind and size TaskId
/// array carries. ``manual_scope_depth`` counts the enclosing MANUAL
/// RuntimeScopeStmts — array carries exist only inside a manual scope.
std::vector<IterArgCarryPlan> AnalyzeCarries(const ForStmtPtr& for_stmt, const ProgramPtr& program,
                                             int manual_scope_depth) {
  std::vector<IterArgCarryPlan> plans(for_stmt->iter_args_.size());

  auto yield = transform_utils::GetLastYieldStmt(transform_utils::UnwrapAutoScope(for_stmt->body_));
  if (yield) {
    INTERNAL_CHECK_SPAN(yield->value_.size() == for_stmt->iter_args_.size(), for_stmt->span_)
        << "Internal error: ForStmt yield/iter_args size mismatch";

    AliasForest aliases(for_stmt, CollectBodyAliases(for_stmt->body_), program);
    for (size_t i = 0; i < for_stmt->iter_args_.size(); ++i) {
      auto yield_var = AsVarLike(yield->value_[i]);
      plans[i].is_rebind = !yield_var || !aliases.InClassOf(yield_var.get(), for_stmt->iter_args_[i].get());
      // A TaskId carry is never a trivial alias: the runtime hands back a fresh
      // PTO2TaskId per iteration, so it always needs a materialised carry.
      if (IsTaskIdScalar(for_stmt->iter_args_[i])) plans[i].is_rebind = true;
    }
  }

  if (manual_scope_depth > 0) {
    for (size_t i = 0; i < for_stmt->iter_args_.size(); ++i) {
      if (plans[i].is_rebind) plans[i].array_size = ResolveArrayCarrySize(for_stmt, i);
    }
  }

  if (for_stmt->kind_ == ForKind::Parallel && manual_scope_depth > 0) {
    for (size_t i = 0; i < for_stmt->iter_args_.size(); ++i) {
      if (!plans[i].is_rebind || !IsTaskIdScalar(for_stmt->iter_args_[i])) continue;
      CHECK_SPAN(plans[i].array_size > 0, for_stmt->span_)
          << "manual_scope: pl.parallel loops carrying a manual_scope dep "
          << "(via ``deps=[...]``) must have a statically-known trip count. "
          << "The runtime fence requires a PTO2TaskId[N] array of fixed N. "
          << "Either make the parallel loop's trip count a Python int "
          << "(e.g. ``pl.parallel(4)``) or restructure to put the parallel "
          << "loop inside a const-bounded scope.";
    }
  }

  return plans;
}

/// Replace any previously-stamped carry attrs on ``attrs`` with ``plans``.
/// ``iter_arg_rebind_<i>`` is written for every slot (its presence proves the
/// pass ran); ``iter_arg_array_size_<i>`` only when the extent is positive.
std::vector<std::pair<std::string, std::any>> StampCarryAttrs(
    const std::vector<std::pair<std::string, std::any>>& attrs, const std::vector<IterArgCarryPlan>& plans,
    const Span& span) {
  std::vector<std::pair<std::string, std::any>> out;
  out.reserve(attrs.size() + plans.size());
  for (const auto& kv : attrs) {
    const bool is_carry_attr = kv.first.rfind(kIterArgRebindAttrPrefix, 0) == 0 ||
                               kv.first.rfind(kIterArgArraySizeAttrPrefix, 0) == 0;
    if (!is_carry_attr) out.push_back(kv);
  }
  for (size_t i = 0; i < plans.size(); ++i) {
    out.emplace_back(IterArgRebindAttrKey(i), std::any(plans[i].is_rebind));
    if (plans[i].array_size <= 0) continue;
    INTERNAL_CHECK_SPAN(plans[i].array_size <= std::numeric_limits<int>::max(), span)
        << "Internal error: TaskId array-carry extent " << plans[i].array_size
        << " overflows the int-typed ForStmt attr codec";
    out.emplace_back(IterArgArraySizeAttrKey(i), std::any(static_cast<int>(plans[i].array_size)));
  }
  return out;
}

/// Stamps the carry plan of every ForStmt in an Orchestration function body,
/// tracking the enclosing MANUAL RuntimeScopeStmt depth exactly as the
/// orchestration codegen's ``in_manual_scope_depth_`` counter did.
class IterArgCarryStamper : public IRMutator {
 public:
  explicit IterArgCarryStamper(ProgramPtr program) : program_(std::move(program)) {}

 protected:
  StmtPtr VisitStmt_(const RuntimeScopeStmtPtr& op) override {
    if (op->manual_) ++manual_scope_depth_;
    auto out = IRMutator::VisitStmt_(op);
    if (op->manual_) --manual_scope_depth_;
    return out;
  }

  StmtPtr VisitStmt_(const ForStmtPtr& op) override {
    // Analyze the pre-mutation loop: nested loops only gain attrs, so the
    // structure the plan reads is identical, and Var identity is unambiguous.
    std::vector<IterArgCarryPlan> plans;
    if (!op->iter_args_.empty()) {
      INTERNAL_CHECK_SPAN(op->iter_args_.size() == op->return_vars_.size(), op->span_)
          << "Internal error: ForStmt iter_args/return_vars size mismatch";
      plans = AnalyzeCarries(op, program_, manual_scope_depth_);
    }

    auto base = IRMutator::VisitStmt_(op);
    if (plans.empty()) return base;

    auto for_stmt = As<ForStmt>(base);
    INTERNAL_CHECK_SPAN(for_stmt, op->span_) << "Internal error: ForStmt mutation must yield a ForStmt";
    auto copy = MutableCopy(for_stmt);
    copy->attrs_ = StampCarryAttrs(for_stmt->attrs_, plans, op->span_);
    return copy;
  }

 private:
  ProgramPtr program_;
  int manual_scope_depth_ = 0;
};

}  // namespace

Pass ClassifyIterArgCarry() {
  auto pass_func = [](const ProgramPtr& program) -> ProgramPtr {
    auto new_functions = program->functions_;
    for (auto& [gvar, func] : new_functions) {
      if (!func || !func->body_) continue;
      // Only Orchestration functions carry loop-carried runtime state that the
      // orchestration codegen lowers into carry variables / TaskId arrays.
      if (func->func_type_ != FunctionType::Orchestration) continue;

      IterArgCarryStamper stamper(program);
      auto new_body = stamper.VisitStmt(func->body_);
      if (new_body.get() == func->body_.get()) continue;
      func = std::make_shared<Function>(func->name_, func->params_, func->param_directions_,
                                        func->return_types_, new_body, func->span_, func->func_type_,
                                        func->level_, func->role_, func->attrs_);
    }
    if (new_functions == program->functions_) return program;
    return std::make_shared<Program>(std::move(new_functions), program->name_, program->span_);
  };

  return CreateProgramPass(pass_func, "ClassifyIterArgCarry", kClassifyIterArgCarryProperties);
}

}  // namespace pass
}  // namespace ir
}  // namespace pypto
