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

#include <climits>
#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/program.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {
namespace {

// Build a mapping from each return value index to the parameter index it
// corresponds to.  This replicates the analysis that was previously inlined
// in orchestration codegen (BuildReturnToParamMapping).
//
// Traverses the function body (excluding the final ReturnStmt) and builds
// var_to_out_param, a map from Var* to the parameter index of the Out/InOut
// parameter it ultimately originates from.  Handles:
//   - tile.store assignments:    lhs -> param index of store's output arg
//   - Var-to-Var assignments:    lhs -> lookup(rhs) if rhs is already mapped
//   - ForStmt iter_args:         return_var[i] -> lookup(initValue) or find_param
//
// For each ReturnStmt value, the var is looked up in var_to_out_param first,
// then falls back to direct param-identity matching.
std::vector<size_t> BuildReturnToParamMapping(const FunctionPtr& func) {
  std::vector<size_t> mapping;
  if (!func || !func->body_) return mapping;

  auto seq = As<SeqStmts>(func->body_);
  if (!seq || seq->stmts_.empty()) return mapping;
  auto return_stmt = As<ReturnStmt>(seq->stmts_.back());
  if (!return_stmt) return mapping;

  auto find_param_index = [&](const Var* v) -> size_t {
    for (size_t pi = 0; pi < func->params_.size(); ++pi) {
      if (func->params_[pi].get() == v) return pi;
    }
    return SIZE_MAX;
  };

  // Look up v in var_to_out_param; if not found, fall back to direct param match.
  auto lookup = [&](const std::unordered_map<const Var*, size_t>& m, const Var* v) -> size_t {
    auto it = m.find(v);
    return it != m.end() ? it->second : find_param_index(v);
  };

  std::unordered_map<const Var*, size_t> var_to_out_param;
  for (size_t si = 0; si + 1 < seq->stmts_.size(); ++si) {
    if (auto assign = As<AssignStmt>(seq->stmts_[si])) {
      if (!assign->var_) continue;
      if (auto call = As<Call>(assign->value_)) {
        // tile.store(tile, offsets, out_param, ...) → lhs tracks out_param
        if (call->op_ && call->op_->name_ == "tile.store" && call->args_.size() >= 3) {
          if (auto out_param = As<Var>(call->args_[2])) {
            var_to_out_param[assign->var_.get()] = find_param_index(out_param.get());
          }
        }
      } else if (auto src_var = As<Var>(assign->value_)) {
        // Var-to-var assignment: propagate existing mapping
        size_t idx = lookup(var_to_out_param, src_var.get());
        if (idx != SIZE_MAX) {
          var_to_out_param[assign->var_.get()] = idx;
        }
      }
    } else if (auto for_stmt = As<ForStmt>(seq->stmts_[si])) {
      for (size_t ri = 0; ri < for_stmt->return_vars_.size() && ri < for_stmt->iter_args_.size(); ++ri) {
        const auto& iter_arg = for_stmt->iter_args_[ri];
        if (!iter_arg || !iter_arg->initValue_ || !for_stmt->return_vars_[ri]) continue;
        if (auto init_var = As<Var>(iter_arg->initValue_)) {
          size_t idx = lookup(var_to_out_param, init_var.get());
          if (idx != SIZE_MAX) {
            var_to_out_param[for_stmt->return_vars_[ri].get()] = idx;
          }
        }
      }
    }
  }

  for (const auto& ret_expr : return_stmt->value_) {
    auto var = As<Var>(ret_expr);
    if (!var) {
      mapping.push_back(SIZE_MAX);
      continue;
    }
    mapping.push_back(lookup(var_to_out_param, var.get()));
  }
  return mapping;
}

std::vector<size_t> CollectOutIndices(const FunctionPtr& func) {
  std::vector<size_t> out_indices;
  for (size_t i = 0; i < func->param_directions_.size(); ++i) {
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
std::vector<size_t> ComputeReturnPermutation(const FunctionPtr& func) {
  auto ret_to_param = BuildReturnToParamMapping(func);
  if (ret_to_param.empty()) return {};

  auto out_indices = CollectOutIndices(func);
  if (out_indices.empty()) return {};

  // If there are more Out params than return values the mapping is incomplete
  // (e.g. some outputs are not yet covered by the IR analysis).  Skip reorder
  // to avoid constructing an out-of-bounds permutation.
  if (out_indices.size() > ret_to_param.size()) return {};

  // Map param_index -> position in out_indices
  std::unordered_map<size_t, size_t> param_to_out_pos;
  for (size_t k = 0; k < out_indices.size(); ++k) {
    param_to_out_pos[out_indices[k]] = k;
  }

  std::vector<size_t> permutation(ret_to_param.size(), SIZE_MAX);
  bool needs_reorder = false;

  for (size_t i = 0; i < ret_to_param.size(); ++i) {
    if (ret_to_param[i] == SIZE_MAX) {
      permutation[i] = i;
      continue;
    }
    auto it = param_to_out_pos.find(ret_to_param[i]);
    if (it == param_to_out_pos.end()) {
      permutation[i] = i;
      continue;
    }
    permutation[i] = it->second;
    if (permutation[i] != i) needs_reorder = true;
  }

  if (!needs_reorder) return {};
  return permutation;
}

// Reorder return values and return types of an InCore function according to
// the given permutation.  Returns a new Function with the reordered return.
FunctionPtr ReorderReturns(const FunctionPtr& func, const std::vector<size_t>& permutation) {
  auto seq = As<SeqStmts>(func->body_);
  INTERNAL_CHECK(seq && !seq->stmts_.empty()) << "NormalizeReturnOrder: function body has no statements";
  auto return_stmt = As<ReturnStmt>(seq->stmts_.back());
  INTERNAL_CHECK(return_stmt) << "NormalizeReturnOrder: function body has no ReturnStmt";
  INTERNAL_CHECK(permutation.size() == return_stmt->value_.size())
      << "NormalizeReturnOrder: permutation size mismatch";

  std::vector<ExprPtr> new_values(return_stmt->value_.size());
  std::vector<TypePtr> new_return_types(func->return_types_.size());

  for (size_t i = 0; i < permutation.size(); ++i) {
    INTERNAL_CHECK(permutation[i] < new_values.size())
        << "NormalizeReturnOrder: permutation index out of range";
    new_values[permutation[i]] = return_stmt->value_[i];
    if (i < func->return_types_.size() && permutation[i] < new_return_types.size()) {
      new_return_types[permutation[i]] = func->return_types_[i];
    }
  }

  auto new_return = std::make_shared<ReturnStmt>(new_values, return_stmt->span_);
  std::vector<StmtPtr> new_stmts(seq->stmts_.begin(), seq->stmts_.end() - 1);
  new_stmts.push_back(new_return);
  auto new_body = std::make_shared<SeqStmts>(new_stmts, seq->span_);

  return std::make_shared<Function>(func->name_, func->params_, func->param_directions_, new_return_types,
                                    new_body, func->span_, func->func_type_, func->level_, func->role_,
                                    func->attrs_);
}

// Mutator that applies return-order permutations to TupleGetItemExpr indices
// in orchestration / opaque functions that call reordered InCore functions.
class TupleIndexPermutationMutator : public IRMutator {
 public:
  explicit TupleIndexPermutationMutator(
      const std::unordered_map<std::string, std::vector<size_t>>& permutations)
      : permutations_(permutations) {}

 protected:
  StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
    auto new_value = VisitExpr(op->value_);

    if (op->var_) {
      if (auto call = As<Call>(new_value)) {
        // Track call results to reordered functions so we can remap
        // TupleGetItemExpr indices on those results later.
        if (auto global_var = std::dynamic_pointer_cast<const GlobalVar>(call->op_)) {
          auto perm_it = permutations_.find(global_var->name_);
          if (perm_it != permutations_.end() && !perm_it->second.empty()) {
            reordered_tuple_vars_[op->var_.get()] = &perm_it->second;
          } else {
            // Variable reassigned to a non-reordered call: remove stale entry.
            reordered_tuple_vars_.erase(op->var_.get());
          }
        } else {
          reordered_tuple_vars_.erase(op->var_.get());
        }
      } else {
        // Non-call assignment: any previous tuple mapping for this var is stale.
        reordered_tuple_vars_.erase(op->var_.get());
      }
    }

    if (new_value.get() != op->value_.get()) {
      return std::make_shared<AssignStmt>(op->var_, new_value, op->span_);
    }
    return op;
  }

  ExprPtr VisitExpr_(const TupleGetItemExprPtr& op) override {
    auto new_tuple = IRMutator::VisitExpr(op->tuple_);

    int new_index = op->index_;
    // Only consider the transformed tuple node (new_tuple).  If VisitExpr
    // replaced it, any identity-based lookup on op->tuple_ would be stale.
    if (auto var = As<Var>(new_tuple)) {
      auto it = reordered_tuple_vars_.find(var.get());
      if (it != reordered_tuple_vars_.end()) {
        const auto& perm = *it->second;
        if (static_cast<size_t>(op->index_) < perm.size() &&
            perm[static_cast<size_t>(op->index_)] != SIZE_MAX) {
          new_index = static_cast<int>(perm[static_cast<size_t>(op->index_)]);
        }
      }
    }

    if (new_tuple.get() != op->tuple_.get() || new_index != op->index_) {
      return std::make_shared<TupleGetItemExpr>(new_tuple, new_index, op->span_);
    }
    return op;
  }

 private:
  const std::unordered_map<std::string, std::vector<size_t>>& permutations_;
  std::unordered_map<const Var*, const std::vector<size_t>*> reordered_tuple_vars_;
};

}  // namespace

namespace pass {

Pass NormalizeReturnOrder() {
  auto pass_func = [](const ProgramPtr& program) -> ProgramPtr {
    // Step A: Analyze InCore functions and compute permutations.
    std::unordered_map<std::string, std::vector<size_t>> permutations;
    std::vector<FunctionPtr> functions;
    bool modified = false;

    for (const auto& [gvar, func] : program->functions_) {
      if (IsInCoreType(func->func_type_)) {
        auto perm = ComputeReturnPermutation(func);
        if (!perm.empty()) {
          auto new_func = ReorderReturns(func, perm);
          permutations[func->name_] = std::move(perm);
          functions.push_back(new_func);
          modified = true;
        } else {
          functions.push_back(func);
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
        TupleIndexPermutationMutator mutator(permutations);
        auto new_body = mutator.VisitStmt(func->body_);
        if (new_body.get() != func->body_.get()) {
          final_functions.push_back(std::make_shared<Function>(
              func->name_, func->params_, func->param_directions_, func->return_types_, new_body, func->span_,
              func->func_type_, func->level_, func->role_, func->attrs_));
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
