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
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/codegen/orchestration/orchestration_analysis.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/program.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"

namespace pypto {
namespace ir {

namespace {

using ::pypto::codegen::IsBuiltinOp;

/// Maximum explicit dep edges per submit, mirroring the runtime's
/// ``PTO2_MAX_EXPLICIT_DEPS = 16`` cap. Exceeding this means codegen would
/// emit an ``Arg::add_dep(...)`` call the runtime rejects; we surface it as
/// an IR-level error instead.
constexpr size_t kManualDepEdgeLimit = 16;

/// Mutator that walks every kernel ``Call`` inside a
/// ``RuntimeScopeStmt(manual=true)`` and writes
/// ``Call.attrs[manual_dep_edges]`` as the union of:
///   1. user-supplied edges (``Call.attrs[user_manual_dep_edges]``)
///   2. data-flow edges: tensor args resolving to a Var produced by a
///      prior ``AssignStmt`` inside the same manual scope, EXCLUDING args
///      whose ``ArgDirection`` is ``NoDep`` (so ``pl.no_dep(x)`` correctly
///      suppresses the auto edge).
class ManualDepMutator : public IRMutator {
 public:
  ManualDepMutator() = default;

  StmtPtr VisitStmt_(const RuntimeScopeStmtPtr& op) override {
    // Save / restore the producer-Var map so nested scopes don't leak
    // producers across manual_scope boundaries. Each manual scope starts
    // fresh; an inner non-manual scope (none expected today) inherits.
    if (op->manual_) {
      auto saved_map = std::move(producer_map_);
      producer_map_.clear();
      ++manual_depth_;

      auto rewritten = IRMutator::VisitStmt_(op);

      --manual_depth_;
      producer_map_ = std::move(saved_map);
      return rewritten;
    }
    return IRMutator::VisitStmt_(op);
  }

  StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
    auto base = IRMutator::VisitStmt_(op);
    auto assign = As<AssignStmt>(base);
    if (!assign || manual_depth_ == 0) return assign;

    // Tuple unpacking creates ``_tuple_tmp = Call(...); a = tuple_get(_tuple_tmp, 0)``.
    // The TupleGetItemExpr LHS is not itself a kernel call — register it as
    // an alias for the underlying tuple's producer so a downstream consumer
    // of ``a`` finds the same task id.
    if (auto get_item = As<TupleGetItemExpr>(assign->value_)) {
      auto tuple_var = AsVarLike(get_item->tuple_);
      if (tuple_var) {
        auto it = producer_map_.find(tuple_var.get());
        if (it != producer_map_.end()) {
          producer_map_[assign->var_.get()] = it->second;
        }
      }
      return assign;
    }

    // Trivial Var-to-Var copy (e.g. SSA pre-image, FlattenCallExpr leftover):
    // forward the producer through the alias.
    if (auto rhs_var = AsVarLike(assign->value_)) {
      auto it = producer_map_.find(rhs_var.get());
      if (it != producer_map_.end()) {
        producer_map_[assign->var_.get()] = it->second;
      }
      return assign;
    }

    auto call = As<Call>(assign->value_);
    if (!call) return assign;
    if (IsBuiltinOp(call->op_->name_)) {
      // Builtin tensor.* / tile.* / system.* ops never get a task; record
      // the LHS as a non-producer (no map entry) so consumers don't add a
      // phantom edge.
      return assign;
    }

    auto new_call = ResolveManualDepsForCall(call);
    // After resolving, register the LHS Var as a producer for downstream
    // siblings — the codegen turns this into ``task_<n>``.
    producer_map_[assign->var_.get()] = assign->var_;
    if (new_call.get() == call.get()) return assign;
    return std::make_shared<AssignStmt>(assign->var_, new_call, assign->span_);
  }

  StmtPtr VisitStmt_(const EvalStmtPtr& op) override {
    auto base = IRMutator::VisitStmt_(op);
    auto eval = As<EvalStmt>(base);
    if (!eval || manual_depth_ == 0) return eval;

    auto call = As<Call>(eval->expr_);
    if (!call || IsBuiltinOp(call->op_->name_)) return eval;

    // EvalStmt-form kernel calls have no LHS; they still need ``add_dep``
    // edges emitted, but they never become producers themselves.
    auto new_call = ResolveManualDepsForCall(call);
    if (new_call.get() == call.get()) return eval;
    return std::make_shared<EvalStmt>(new_call, eval->span_);
  }

 private:
  /// Compute the resolved ``manual_dep_edges`` for ``call`` (user-supplied +
  /// data-flow union, NoDep-aware, capped at the runtime limit) and return a
  /// rewritten Call carrying the attr — or the original Call when no edges.
  CallPtr ResolveManualDepsForCall(const CallPtr& call) {
    std::vector<VarPtr> deps;
    std::unordered_set<const Var*> seen;

    auto append_dep = [&](const VarPtr& var) {
      if (var && seen.insert(var.get()).second) {
        deps.push_back(var);
      }
    };

    // 1. User-supplied edges from kAttrUserManualDepEdges (preserve order).
    for (const auto& [k, v] : call->attrs_) {
      if (k != kAttrUserManualDepEdges) continue;
      const auto* user_deps = std::any_cast<std::vector<VarPtr>>(&v);
      INTERNAL_CHECK_SPAN(user_deps, call->span_)
          << "Internal error: " << kAttrUserManualDepEdges << " attr must hold std::vector<VarPtr>";
      for (const auto& var : *user_deps) {
        if (!var) continue;
        // Accept any Var here; the verifier catches real misuse with a
        // clearer error, and unit tests can hand-build IR fragments without
        // a full producer trace.
        append_dep(var);
      }
      break;
    }

    // 2. Data-flow edges from tensor args, excluding NoDep slots.
    auto arg_dirs = call->GetArgDirections();
    for (size_t i = 0; i < call->args_.size(); ++i) {
      if (i < arg_dirs.size() && arg_dirs[i] == ArgDirection::NoDep) continue;
      auto arg_var = AsVarLike(call->args_[i]);
      if (!arg_var) continue;
      auto it = producer_map_.find(arg_var.get());
      if (it == producer_map_.end()) continue;
      append_dep(it->second);
    }

    // 3. Cap check matching the runtime hard limit.
    INTERNAL_CHECK_SPAN(deps.size() <= kManualDepEdgeLimit, call->span_)
        << "manual_scope: call has " << deps.size() << " dependency edges, exceeds runtime cap of "
        << kManualDepEdgeLimit;

    if (deps.empty()) return call;

    auto new_attrs = WithManualDepEdgesAttr(call->attrs_, std::move(deps));
    return std::make_shared<const Call>(call->op_, call->args_, call->kwargs_, std::move(new_attrs),
                                        call->GetType(), call->span_);
  }

  /// Map from a Var (LHS of an AssignStmt whose RHS is a kernel Call) to the
  /// same Var. The codegen later uses Var emit names to construct
  /// ``task_<name>`` C++ identifiers; we just need the producing Var here.
  std::unordered_map<const Var*, VarPtr> producer_map_;
  int manual_depth_ = 0;
};

}  // namespace

namespace pass {

Pass DeriveManualScopeDeps() {
  auto pass_func = [](const ProgramPtr& program) -> ProgramPtr {
    if (!program) return program;

    auto new_functions = program->functions_;
    for (auto& [gvar, func] : new_functions) {
      if (!func || !func->body_) continue;

      ManualDepMutator mutator;
      auto new_body = mutator.VisitStmt(func->body_);
      if (new_body.get() == func->body_.get()) continue;

      func = std::make_shared<Function>(func->name_, func->params_, func->param_directions_,
                                        func->return_types_, new_body, func->span_, func->func_type_,
                                        func->level_, func->role_, func->attrs_);
    }

    if (new_functions == program->functions_) {
      return program;
    }
    return std::make_shared<Program>(std::move(new_functions), program->name_, program->span_);
  };

  return CreateProgramPass(pass_func, "DeriveManualScopeDeps", kDeriveManualScopeDepsProperties);
}

}  // namespace pass
}  // namespace ir
}  // namespace pypto
