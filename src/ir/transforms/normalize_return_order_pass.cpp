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
#include "pypto/ir/program.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/transforms/utils/return_lineage_utils.h"
#include "pypto/ir/transforms/utils/var_collectors.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {
namespace {

// Sentinel meaning "no matching parameter found".
static constexpr int kNoParam = -1;

// Rewrite tensor return values to the param Var each one writes through
// (pointer identity). Orchestration codegen aliases call results to their
// source args via this mapping; explicit param returns make it a lookup
// instead of a heuristic re-derivation that can mis-bind multi-output
// kernels (#1702, #1573, #1693). Returns nullptr when nothing changes.
FunctionPtr CanonicalizeReturnValues(const FunctionPtr& func,
                                     const std::vector<std::optional<size_t>>& ret_to_param) {
  if (!func || !func->body_) return nullptr;

  if (ret_to_param.empty()) return nullptr;

  // The lineage map already requires consensus across device return paths.
  // Apply that same map to every return site, including nested split bodies.
  class ReturnRewriter : public IRMutator {
   public:
    ReturnRewriter(const FunctionPtr& func, const std::vector<std::optional<size_t>>& ret_to_param)
        : func_(func), ret_to_param_(ret_to_param) {}
    bool changed = false;

   protected:
    StmtPtr VisitStmt_(const ReturnStmtPtr& op) override {
      if (op->value_.empty()) return op;

      // A Group/Spmd wrapper around a multi-result kernel forwards the inner
      // call's whole tuple as ONE return value, while declaring N return
      // positions:  ``result = self.inner(...); return result``.  Expand that
      // into the N params the inner call writes back, so the return is
      // positionally explicit like every other kernel's.
      //
      // Left un-expanded it is invisible to every consumer of the
      // return-position -> param map: the map comes back one-short, callers
      // deem it imprecise and fall back to the legacy tail-alignment
      // heuristic, which shifts each returned element onto the wrong Out/InOut
      // param as soon as the callee writes an Out/InOut param it does not
      // return (an accumulator, ``__gm_pipe_buffer``, an in-place KV cache).
      // That silently binds a consumer task to the wrong tensor operand --
      // #1573 resurfacing on exactly the wrappers its fix left on the
      // heuristic.
      if (op->value_.size() == 1 && ret_to_param_.size() > 1) {
        // Only a function that declares N flat return positions may be given an
        // N-value return. A single ``pl.Tuple[T1, ..., TN]`` return declares ONE
        // (its ``return_types_`` is one TupleType) and stays one value.
        if (func_->return_types_.size() != ret_to_param_.size()) return op;
        std::vector<ExprPtr> expanded;
        expanded.reserve(ret_to_param_.size());
        for (const auto& idx : ret_to_param_) {
          // Every position must be nameable as a param for the expansion to be
          // well-formed; otherwise leave the return alone (status quo ante).
          if (!idx) return op;
          const auto& param = func_->params_[idx.value()];  // NOLINT(bugprone-unchecked-optional-access)
          if (!AsTensorTypeLike(param->GetType())) return op;
          expanded.push_back(param);
        }
        changed = true;
        return std::make_shared<ReturnStmt>(expanded, op->span_);
      }

      if (ret_to_param_.size() != op->value_.size()) return op;
      bool return_changed = false;
      std::vector<ExprPtr> new_values = op->value_;
      for (size_t i = 0; i < new_values.size(); ++i) {
        if (!ret_to_param_[i]) continue;
        const auto& param =
            func_->params_[ret_to_param_[i].value()];  // NOLINT(bugprone-unchecked-optional-access)
        if (!AsTensorTypeLike(param->GetType())) continue;
        if (new_values[i].get() == param.get()) continue;
        new_values[i] = param;
        return_changed = true;
      }
      if (!return_changed) return op;
      changed = true;
      return std::make_shared<ReturnStmt>(new_values, op->span_);
    }

   private:
    const FunctionPtr& func_;
    const std::vector<std::optional<size_t>>& ret_to_param_;
  };

  ReturnRewriter rewriter(func, ret_to_param);
  auto new_body = rewriter.VisitStmt(func->body_);
  if (!rewriter.changed) return nullptr;

  auto new_func = MutableCopy(func);
  new_func->body_ = new_body;
  return new_func;
}

std::vector<int> CollectOutIndices(const FunctionPtr& func) {
  std::vector<int> out_indices;
  for (int i = 0; i < static_cast<int>(func->param_directions_.size()); ++i) {
    if (func->param_directions_[i] == ParamDirection::Out ||
        func->param_directions_[i] == ParamDirection::InOut) {
      out_indices.push_back(i);
    }
  }
  return out_indices;
}

// Compute the permutation that reorders returns so that return[k] corresponds
// to out_indices[k].  Returns an empty vector when no reordering is needed.
//
// permutation[old_index] = new_index
std::vector<int> ComputeReturnPermutation(const FunctionPtr& func,
                                          const std::vector<std::optional<size_t>>& ret_to_param) {
  if (ret_to_param.empty() || ret_to_param.size() != func->return_types_.size()) return {};

  auto out_indices = CollectOutIndices(func);
  if (out_indices.empty()) return {};

  // If there are more Out params than return values the mapping is incomplete
  // (e.g. some outputs are not yet covered by the IR analysis).  Skip reorder
  // to avoid constructing an out-of-bounds permutation.
  if (static_cast<int>(out_indices.size()) > static_cast<int>(ret_to_param.size())) return {};

  // Map param_index -> position in out_indices
  std::unordered_map<int, int> param_to_out_pos;
  for (int k = 0; k < static_cast<int>(out_indices.size()); ++k) {
    param_to_out_pos[out_indices[k]] = k;
  }

  std::vector<int> permutation(ret_to_param.size(), kNoParam);
  bool needs_reorder = false;

  for (int i = 0; i < static_cast<int>(ret_to_param.size()); ++i) {
    const auto& parameter = ret_to_param[i];
    if (!parameter) {
      permutation[i] = i;
      continue;
    }
    auto it = param_to_out_pos.find(static_cast<int>(*parameter));
    if (it == param_to_out_pos.end()) {
      permutation[i] = i;
      continue;
    }
    permutation[i] = it->second;
    if (permutation[i] != i) needs_reorder = true;
  }

  // Guard against non-bijective permutations (duplicate targets or holes).
  // A malformed permutation can later create ReturnStmt entries with null values.
  std::vector<bool> seen(permutation.size(), false);
  for (int i = 0; i < static_cast<int>(permutation.size()); ++i) {
    int target = permutation[i];
    if (target < 0 || target >= static_cast<int>(permutation.size())) {
      return {};
    }
    if (seen[target]) {
      // Collision: two old indices map to the same new index.
      return {};
    }
    seen[target] = true;
  }
  for (bool v : seen) {
    if (!v) {
      // Hole: at least one new index has no source.
      return {};
    }
  }

  if (!needs_reorder) return {};
  return permutation;
}

// Reorder return values and return types of an InCore function according to
// the given permutation.  Returns a new Function with the reordered return.
FunctionPtr ReorderReturns(const FunctionPtr& func, const std::vector<int>& permutation) {
  class ReturnPermuter : public IRMutator {
   public:
    explicit ReturnPermuter(const std::vector<int>& permutation) : permutation_(permutation) {}

   protected:
    StmtPtr VisitStmt_(const ReturnStmtPtr& op) override {
      INTERNAL_CHECK_SPAN(permutation_.size() == op->value_.size(), op->span_)
          << "NormalizeReturnOrder: permutation size mismatch";
      std::vector<ExprPtr> values(op->value_.size());
      for (size_t i = 0; i < permutation_.size(); ++i) values[permutation_[i]] = op->value_[i];
      return std::make_shared<ReturnStmt>(values, op->span_);
    }

   private:
    const std::vector<int>& permutation_;
  };

  std::vector<TypePtr> return_types(permutation.size());
  for (size_t i = 0; i < permutation.size(); ++i) return_types[permutation[i]] = func->return_types_[i];
  auto new_func = MutableCopy(func);
  new_func->return_types_ = std::move(return_types);
  new_func->body_ = ReturnPermuter(permutation).VisitStmt(func->body_);
  return new_func;
}

// Only a direct assignment in a caller visited by the remapper can
// consume a reordered result. A public pass invocation may still contain
// inline calls, so defer their callee's permutation while such uses remain.
class UnremappableCallCollector : public IRVisitor {
 public:
  void VisitFunction(const FunctionPtr& func) override {
    remap_assignments_ = !IsInCoreType(func->func_type_);
    VisitStmt(func->body_);
  }

  std::unordered_set<std::string> callees;

 protected:
  void VisitStmt_(const AssignStmtPtr& op) override {
    const auto* saved = assignment_value_;
    if (remap_assignments_) assignment_value_ = op->value_.get();
    IRVisitor::VisitStmt_(op);
    assignment_value_ = saved;
  }

  void VisitExpr_(const CallPtr& op) override {
    Record(op.get(), op->op_);
    IRVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const SubmitPtr& op) override {
    Record(op.get(), op->op_);
    IRVisitor::VisitExpr_(op);
  }

 private:
  void Record(const Expr* value, const OpPtr& callee) {
    if (value == assignment_value_) return;
    if (const auto global = std::dynamic_pointer_cast<const GlobalVar>(callee)) callees.insert(global->name_);
  }

  const Expr* assignment_value_ = nullptr;
  bool remap_assignments_ = false;
};

// Keep the physical result tuple in the callee's new order. Projections use
// the matching permutation; whole-tuple consumers retain the old logical order.
class TupleIndexPermutationMutator : public IRMutator {
 public:
  TupleIndexPermutationMutator(const std::unordered_map<std::string, std::vector<int>>& permutations,
                               const FunctionPtr& func)
      : permutations_(permutations), flat_returns_(func->return_types_.size() > 1) {
    // Before SSA, one binding can have definitions in multiple branches or
    // loops. Keep those definitions in logical order under their original Var
    // so ConvertToSSA can still discover and merge the shared binding.
    var_collectors::VarDefUseCollector collector;
    collector.VisitStmt(func->body_);
    std::unordered_set<const Var*> defined;
    for (const auto& param : func->params_) defined.insert(param.get());
    for (const auto* var : collector.var_defs_ordered) {
      if (!defined.insert(var).second) rebound_vars_.insert(var);
    }
  }

 protected:
  StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
    auto new_value = VisitExpr(op->value_);
    OpPtr callee;
    const auto call = As<Call>(new_value);
    const auto submit = As<Submit>(new_value);
    if (call) callee = call->op_;
    if (submit) callee = submit->op_;
    const auto global = std::dynamic_pointer_cast<const GlobalVar>(callee);
    const auto found = global ? permutations_.find(global->name_) : permutations_.end();
    if (found != permutations_.end()) {
      const auto& permutation = found->second;
      const auto tuple_type = As<TupleType>(new_value->GetType());
      INTERNAL_CHECK_SPAN(tuple_type && tuple_type->types_.size() >= permutation.size(), op->span_)
          << "NormalizeReturnOrder: call result does not contain all permuted returns";
      // Permute caller-instantiated types, not callee-local shape symbols.
      // Submit's trailing TASK_ID stays outside the permutation.
      auto types = tuple_type->types_;
      for (size_t i = 0; i < permutation.size(); ++i) {
        types[permutation[i]] = tuple_type->types_[i];
      }
      auto type = std::make_shared<TupleType>(types);
      if (call) {
        new_value =
            std::make_shared<Call>(call->op_, call->args_, call->kwargs_, call->attrs_, type, call->span_);
      } else {
        new_value = std::make_shared<Submit>(
            submit->op_, submit->args_, submit->deps_, submit->kwargs_, submit->attrs_, type, submit->span_,
            submit->core_num_, submit->sync_start_, submit->allow_early_resolve_, submit->predicate_);
      }
      auto result = std::make_shared<Var>(op->var_->name_hint_, type, op->var_->span_);
      reordered_tuple_vars_[op->var_.get()] = {result, &permutation, nullptr};
      if (rebound_vars_.count(op->var_.get())) {
        auto logical = VisitExpr(op->var_);
        reordered_tuple_vars_.erase(op->var_.get());
        return SeqStmts::Flatten({std::make_shared<AssignStmt>(result, new_value, op->span_),
                                  std::make_shared<AssignStmt>(op->var_, logical, op->span_)},
                                 op->span_);
      }
      return std::make_shared<AssignStmt>(result, new_value, op->span_);
    }

    // A source-level Var can be rebound before SSA conversion. Its subsequent
    // uses must not retain the permutation of an earlier call assignment.
    reordered_tuple_vars_.erase(op->var_.get());
    if (new_value != op->value_) return std::make_shared<AssignStmt>(op->var_, new_value, op->span_);
    return op;
  }

  ExprPtr VisitExpr_(const TupleGetItemExprPtr& op) override {
    const auto found = reordered_tuple_vars_.find(op->tuple_.get());
    if (found == reordered_tuple_vars_.end()) return IRMutator::VisitExpr_(op);
    const auto& tuple = found->second;
    return std::make_shared<TupleGetItemExpr>(tuple.physical, PermuteIndex(tuple, op->index_), op->span_);
  }

  ExprPtr VisitExpr_(const VarPtr& op) override {
    const auto found = reordered_tuple_vars_.find(op.get());
    if (found == reordered_tuple_vars_.end()) return IRMutator::VisitExpr_(op);
    auto& tuple = found->second;
    if (!tuple.logical) {
      const auto type = As<TupleType>(tuple.physical->GetType());
      std::vector<ExprPtr> elements;
      elements.reserve(type->types_.size());
      for (size_t i = 0; i < type->types_.size(); ++i) {
        elements.push_back(std::make_shared<TupleGetItemExpr>(
            tuple.physical, PermuteIndex(tuple, static_cast<int>(i)), op->span_));
      }
      // Cache one logical view per definition: aliases/forwarding must not
      // repeatedly rebuild the same tuple or evaluate the call a second time.
      tuple.logical = std::make_shared<MakeTuple>(elements, op->span_);
    }
    return tuple.logical;
  }

  StmtPtr VisitStmt_(const ReturnStmtPtr& op) override {
    auto rewritten = IRMutator::VisitStmt_(op);
    const auto returned = As<ReturnStmt>(rewritten);
    if (flat_returns_ && rewritten != op && returned->value_.size() == 1) {
      if (const auto tuple = As<MakeTuple>(returned->value_[0])) {
        return std::make_shared<ReturnStmt>(tuple->elements_, op->span_);
      }
    }
    return rewritten;
  }

 private:
  struct ReorderedTuple {
    VarPtr physical;
    const std::vector<int>* permutation;
    ExprPtr logical;
  };

  static int PermuteIndex(const ReorderedTuple& tuple, int index) {
    return index >= 0 && index < static_cast<int>(tuple.permutation->size()) ? (*tuple.permutation)[index]
                                                                             : index;
  }

  const std::unordered_map<std::string, std::vector<int>>& permutations_;
  bool flat_returns_;
  std::unordered_set<const Var*> rebound_vars_;
  std::unordered_map<const Expr*, ReorderedTuple> reordered_tuple_vars_;
};

}  // namespace

namespace pass {

Pass NormalizeReturnOrder() {
  auto pass_func = [](const ProgramPtr& program) -> ProgramPtr {
    // Step A: Analyze InCore functions and compute permutations.
    std::unordered_map<std::string, std::vector<int>> permutations;
    std::vector<FunctionPtr> functions;
    bool modified = false;
    std::optional<UnremappableCallCollector> unremappable_calls;
    return_lineage::ScopedTraceCache return_cache;

    for (const auto& [gvar, func] : program->functions_) {
      const bool is_wrapper = IsWrapperType(func->func_type_);
      // CHIP Orchestration results are consumed by HOST codegen too. Preserve
      // their parameter identities before InitMemRef gives SSA aliases distinct
      // allocation metadata; only InCore return positions are reordered below.
      if (IsInCoreType(func->func_type_) || is_wrapper || func->func_type_ == FunctionType::Graph ||
          func->func_type_ == FunctionType::Orchestration) {
        // Step A0: make every tensor return an explicit param reference.
        const auto ret_to_param = return_lineage::ReturnedParamIndices(func, program);
        FunctionPtr current = func;
        if (auto canonical = CanonicalizeReturnValues(current, ret_to_param)) {
          current = canonical;
          modified = true;
        }
        std::vector<int> perm;
        // An extern's declared result order is part of its fixed ABI.
        if (IsInCoreType(current->func_type_) && !current->HasAttr(kAttrExternalSource)) {
          perm = ComputeReturnPermutation(current, ret_to_param);
        }
        if (!perm.empty()) {
          if (!unremappable_calls) {
            unremappable_calls.emplace();
            for (const auto& entry : program->functions_) unremappable_calls->VisitFunction(entry.second);
          }
          if (unremappable_calls->callees.count(current->name_)) perm.clear();
        }
        if (!perm.empty()) {
          auto new_func = ReorderReturns(current, perm);
          permutations[current->name_] = std::move(perm);
          functions.push_back(new_func);
          modified = true;
        } else {
          functions.push_back(current);
        }
      } else {
        functions.push_back(func);
      }
    }

    if (!modified) return program;

    // Step B: Update TupleGetItemExpr indices in non-InCore functions.
    std::vector<FunctionPtr> final_functions;
    for (const auto& func : functions) {
      if (!IsInCoreType(func->func_type_)) {
        TupleIndexPermutationMutator mutator(permutations, func);
        auto new_body = mutator.VisitStmt(func->body_);
        if (new_body.get() != func->body_.get()) {
          final_functions.push_back(std::make_shared<Function>(
              func->name_, func->params_, func->param_directions_, func->return_types_, new_body, func->span_,
              func->func_type_, func->level_, func->role_, func->attrs_, func->requires_runtime_binding_,
              func->ir_stage_));
        } else {
          final_functions.push_back(func);
        }
      } else {
        final_functions.push_back(func);
      }
    }

    return std::make_shared<Program>(final_functions, program->name_, program->span_);
  };

  return CreateProgramPass(pass_func, "NormalizeReturnOrder", kNormalizeReturnOrderProperties);
}

}  // namespace pass
}  // namespace ir
}  // namespace pypto
