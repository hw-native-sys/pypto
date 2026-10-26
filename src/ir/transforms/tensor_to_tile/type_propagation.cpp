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

#include "src/ir/transforms/tensor_to_tile/type_propagation.h"

#include <cstddef>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {
namespace tensor_to_tile {

using transform_utils::FlattenToStmts;

namespace {

/**
 * @brief Find the YieldStmt in a list of statements and return its value types.
 *
 * Recurses into SeqStmts and ScopeStmt to find yields in nested containers.
 */
std::vector<TypePtr> FindYieldTypes(const std::vector<StmtPtr>& stmts) {
  for (const auto& stmt : stmts) {
    if (auto yield = As<YieldStmt>(stmt)) {
      std::vector<TypePtr> types;
      types.reserve(yield->value_.size());
      for (const auto& val : yield->value_) {
        types.push_back(val->GetType());
      }
      return types;
    }
    if (auto seq = As<SeqStmts>(stmt)) {
      auto found = FindYieldTypes(seq->stmts_);
      if (!found.empty()) return found;
    }
    if (auto scope = As<ScopeStmt>(stmt)) {
      auto body_stmts = FlattenToStmts(scope->body_);
      auto found = FindYieldTypes(body_stmts);
      if (!found.empty()) return found;
    }
  }
  return {};
}

}  // namespace

template <typename ReconstructFn>
StmtPtr TypePropagatingMutator::UpdateLoopReturnVars(const std::vector<IterArgPtr>& new_iter_args,
                                                     const std::vector<VarPtr>& new_return_vars,
                                                     const std::vector<VarPtr>& orig_return_vars,
                                                     ReconstructFn reconstruct, const StmtPtr& original) {
  bool rv_changed = false;
  std::vector<VarPtr> updated_rv;
  updated_rv.reserve(new_return_vars.size());
  for (size_t i = 0; i < new_return_vars.size(); ++i) {
    const auto& rv = new_return_vars[i];
    if (i < new_iter_args.size() && new_iter_args[i]->GetType() != rv->GetType()) {
      auto updated = std::make_shared<Var>(rv->name_hint_, new_iter_args[i]->GetType(), rv->span_);
      // Register mapping for both original and current return_var pointers
      var_remap_[orig_return_vars[i].get()] = updated;
      if (rv.get() != orig_return_vars[i].get()) {
        RetainVar(rv);
        var_remap_[rv.get()] = updated;
      }
      updated_rv.push_back(updated);
      rv_changed = true;
    } else {
      updated_rv.push_back(rv);
    }
  }
  if (!rv_changed) return original;
  return reconstruct(std::move(updated_rv));
}

ExprPtr TypePropagatingMutator::VisitExpr_(const IterArgPtr& op) {
  auto it = var_remap_.find(op.get());
  if (it != var_remap_.end()) return it->second;
  auto new_init = VisitExpr(op->initValue_);
  if (new_init.get() == op->initValue_.get()) return op;
  return std::make_shared<IterArg>(op->name_hint_, new_init->GetType(), new_init, op->span_);
}

StmtPtr TypePropagatingMutator::VisitStmt_(const ForStmtPtr& op) {
  auto result = IRMutator::VisitStmt_(op);
  auto new_for = As<ForStmt>(result);
  if (!new_for) return result;
  return UpdateLoopReturnVars(
      new_for->iter_args_, new_for->return_vars_, op->return_vars_,
      [&](auto new_rv) {
        auto copy = MutableCopy(new_for);
        copy->return_vars_ = std::move(new_rv);
        return copy;
      },
      result);
}

StmtPtr TypePropagatingMutator::VisitStmt_(const WhileStmtPtr& op) {
  auto result = IRMutator::VisitStmt_(op);
  auto new_while = As<WhileStmt>(result);
  if (!new_while) return result;
  return UpdateLoopReturnVars(
      new_while->iter_args_, new_while->return_vars_, op->return_vars_,
      [&](auto new_rv) {
        auto result = MutableCopy(new_while);
        result->return_vars_ = std::move(new_rv);
        return StmtPtr(result);
      },
      result);
}

StmtPtr TypePropagatingMutator::VisitStmt_(const IfStmtPtr& op) {
  auto new_condition = VisitExpr(op->condition_);

  // Save var_remap_ and visit each branch in isolation
  auto saved_remap = var_remap_;
  auto new_then_body = VisitStmt(op->then_body_);

  var_remap_ = saved_remap;
  std::optional<StmtPtr> new_else_body;
  if (op->else_body_.has_value()) {
    new_else_body = VisitStmt(*op->else_body_);
  }
  var_remap_ = saved_remap;

  // Determine yield types from branches to update return_var types
  auto yield_types = FindYieldTypes(FlattenToStmts(new_then_body));
  if (yield_types.empty() && new_else_body.has_value()) {
    yield_types = FindYieldTypes(FlattenToStmts(*new_else_body));
  }

  std::vector<VarPtr> new_return_vars;
  new_return_vars.reserve(op->return_vars_.size());
  for (size_t i = 0; i < op->return_vars_.size(); ++i) {
    const auto& rv = op->return_vars_[i];
    if (i < yield_types.size() && yield_types[i] != rv->GetType()) {
      auto new_rv = std::make_shared<Var>(rv->name_hint_, yield_types[i], rv->span_);
      var_remap_[rv.get()] = new_rv;
      new_return_vars.push_back(new_rv);
    } else {
      new_return_vars.push_back(rv);
    }
  }

  // Copy-on-write: return original when nothing changed
  bool rv_changed = (new_return_vars != op->return_vars_);
  bool else_changed = new_else_body.has_value() != op->else_body_.has_value() ||
                      (new_else_body.has_value() && new_else_body->get() != op->else_body_->get());
  if (!rv_changed && new_condition.get() == op->condition_.get() &&
      new_then_body.get() == op->then_body_.get() && !else_changed) {
    return op;
  }

  auto new_if = MutableCopy(op);
  new_if->condition_ = new_condition;
  new_if->then_body_ = new_then_body;
  new_if->else_body_ = new_else_body;
  new_if->return_vars_ = std::move(new_return_vars);
  return new_if;
}

StmtPtr TypePropagatingMutator::HandlePassThroughAssign(const AssignStmtPtr& op, const ExprPtr& new_value) {
  if (new_value.get() == op->value_.get()) {
    // Assignment is unchanged — clear any stale remap so future uses of this Var*
    // are not rewritten to an older replacement.
    var_remap_.erase(op->var_.get());
    return op;
  }
  if (new_value->GetType() != op->value_->GetType()) {
    auto new_var = std::make_shared<Var>(op->var_->name_hint_, new_value->GetType(), op->var_->span_);
    var_remap_[op->var_.get()] = new_var;
    auto result = MutableCopy(op);
    result->var_ = new_var;
    result->value_ = new_value;
    return result;
  }
  // Value changed but type did not — keep original Var, clear any stale remap.
  var_remap_.erase(op->var_.get());
  auto result = MutableCopy(op);
  result->value_ = new_value;
  return result;
}

void TypePropagatingMutator::RetainVar(const ExprPtr& v) {
  if (v) retained_vars_.push_back(v);
}

}  // namespace tensor_to_tile
}  // namespace ir
}  // namespace pypto
