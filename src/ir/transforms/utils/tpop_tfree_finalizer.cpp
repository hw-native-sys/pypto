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

#include "pypto/ir/transforms/utils/tpop_tfree_finalizer.h"

#include <any>
#include <cstddef>
#include <functional>
#include <limits>
#include <map>
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
#include "pypto/ir/op_registry.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/utils/core_affinity.h"
#include "pypto/ir/transforms/utils/core_side_ops.h"
#include "pypto/ir/transforms/utils/loop_state_repair.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/transforms/utils/op_predicates.h"
#include "pypto/ir/transforms/utils/scope_outline_utils.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/transforms/utils/var_collectors.h"

namespace pypto {
namespace ir {
namespace {

const auto& FlattenBody = transform_utils::FlattenToStmts;
const auto& GetLastYieldStmt = transform_utils::GetLastYieldStmt;

StmtPtr FinalizeNestedTpopTfrees(const StmtPtr& stmt, core_affinity::CoreSide side,
                                 const std::unordered_map<const Var*, VarPtr>& tpop_var_remap);

}  // namespace

namespace tpop_tfree {

std::string GetTfreeOpName(core_affinity::CoreSide side) { return core_side_ops::TFreeOp(side); }

CallPtr CreateTfree(core_affinity::CoreSide side, const ExprPtr& tile, const Span& span,
                    std::optional<int> pipe_id) {
  std::vector<std::pair<std::string, std::any>> kwargs;
  if (pipe_id.has_value()) {
    kwargs.emplace_back("id", std::any(pipe_id.value()));
  }
  return OpRegistry::GetInstance().Create(GetTfreeOpName(side), {tile}, kwargs, span);
}

bool IsTpopAssignStmt(const StmtPtr& stmt, VarPtr* result_var, CallPtr* tpop_call) {
  auto assign = std::dynamic_pointer_cast<const AssignStmt>(stmt);
  if (!assign) return false;
  auto call = std::dynamic_pointer_cast<const Call>(assign->value_);
  if (!op_predicates::IsTPop(call)) return false;
  if (result_var) *result_var = assign->var_;
  if (tpop_call) *tpop_call = call;
  return true;
}

bool IsExpectedTpopOp(const std::string& op_name, FunctionType func_type) {
  // An AIC body expects to see tpop_from_aiv (receiving from its vector peer);
  // an AIV body expects tpop_from_aic (receiving from its cube peer).
  if (func_type == FunctionType::AIC) return op_name == core_side_ops::TPopOp(core_affinity::CoreSide::AIC);
  if (func_type == FunctionType::AIV) return op_name == core_side_ops::TPopOp(core_affinity::CoreSide::AIV);
  return false;
}

bool IsExpectedTpopAssignStmt(const StmtPtr& stmt, FunctionType func_type, VarPtr* result_var) {
  auto assign = std::dynamic_pointer_cast<const AssignStmt>(stmt);
  if (!assign) return false;
  auto call = std::dynamic_pointer_cast<const Call>(assign->value_);
  auto op = call ? std::dynamic_pointer_cast<const Op>(call->op_) : nullptr;
  if (!op || !IsExpectedTpopOp(op->name_, func_type)) return false;
  if (result_var) *result_var = assign->var_;
  return true;
}

bool IsTfreeStmt(const StmtPtr& stmt, VarPtr* tile_var, std::string* op_name) {
  auto eval = std::dynamic_pointer_cast<const EvalStmt>(stmt);
  if (!eval) return false;
  auto call = std::dynamic_pointer_cast<const Call>(eval->expr_);
  if (!op_predicates::IsTFree(call)) return false;
  auto op = std::dynamic_pointer_cast<const Op>(call->op_);
  if (op_name) *op_name = op->name_;
  if (tile_var) *tile_var = !call->args_.empty() ? AsVarLike(call->args_[0]) : nullptr;
  return true;
}

std::unordered_set<const Var*> CollectStmtVarRefs(const StmtPtr& stmt) {
  outline_utils::VarDefUseCollector collector;
  collector.VisitStmt(stmt);
  return collector.GetAllVarRefs();
}

std::unordered_set<const Var*> CollectCallArgVarRefs(const StmtPtr& stmt) {
  CallPtr call;
  if (auto assign = std::dynamic_pointer_cast<const AssignStmt>(stmt)) {
    call = std::dynamic_pointer_cast<const Call>(assign->value_);
  } else if (auto eval = std::dynamic_pointer_cast<const EvalStmt>(stmt)) {
    call = std::dynamic_pointer_cast<const Call>(eval->expr_);
  }
  if (!call) return CollectStmtVarRefs(stmt);

  std::unordered_set<const Var*> refs_set;
  for (const auto& arg : call->args_) {
    outline_utils::VarDefUseCollector collector;
    collector.VisitExpr(arg);
    refs_set.insert(collector.var_uses.begin(), collector.var_uses.end());
  }
  for (const auto& [key, value] : call->kwargs_) {
    (void)key;
    if (value.type() == typeid(ExprPtr)) {
      outline_utils::VarDefUseCollector collector;
      collector.VisitExpr(std::any_cast<ExprPtr>(value));
      refs_set.insert(collector.var_uses.begin(), collector.var_uses.end());
    } else if (value.type() == typeid(VarPtr)) {
      outline_utils::VarDefUseCollector collector;
      collector.VisitExpr(std::any_cast<VarPtr>(value));
      refs_set.insert(collector.var_uses.begin(), collector.var_uses.end());
    } else if (value.type() == typeid(IterArgPtr)) {
      outline_utils::VarDefUseCollector collector;
      collector.VisitExpr(std::any_cast<IterArgPtr>(value));
      refs_set.insert(collector.var_uses.begin(), collector.var_uses.end());
    }
  }
  return refs_set;
}

const Var* CanonicalizeTpopRef(const Var* var, const std::unordered_map<const Var*, VarPtr>& tpop_var_remap) {
  if (!var) return nullptr;
  auto it = tpop_var_remap.find(var);
  return (it != tpop_var_remap.end() && it->second) ? it->second.get() : var;
}

std::vector<StmtPtr> FinalizeTpopTfrees(const std::vector<StmtPtr>& stmts, core_affinity::CoreSide side,
                                        const std::unordered_map<const Var*, VarPtr>& tpop_var_remap) {
  std::vector<StmtPtr> normalized_inputs;
  normalized_inputs.reserve(stmts.size());
  for (const auto& stmt : stmts) {
    normalized_inputs.push_back(FinalizeNestedTpopTfrees(stmt, side, tpop_var_remap));
  }

  std::map<const Var*, TpopLifetime> chains;
  std::vector<const Var*> tpop_order;
  const size_t no_tfree = std::numeric_limits<size_t>::max();
  const std::string expected_tfree = GetTfreeOpName(side);
  using TpopVarRemap = std::unordered_map<const Var*, VarPtr>;
  std::unordered_map<const Var*, VarPtr> local_tpop_var_remap = tpop_var_remap;

  auto canonicalize_var = [&](const VarPtr& var, const TpopVarRemap& remap) -> VarPtr {
    if (!var) {
      return nullptr;
    }
    auto it = remap.find(var.get());
    if (it != remap.end() && it->second) {
      return it->second;
    }
    return var;
  };

  auto try_resolve_canonical_tpop = [&](const ExprPtr& expr, const TpopVarRemap& remap,
                                        VarPtr* canonical_var) -> bool {
    auto src_var = AsVarLike(expr);
    if (!src_var) {
      return false;
    }
    auto canonical = canonicalize_var(src_var, remap);
    if (!canonical || chains.find(canonical.get()) == chains.end()) {
      return false;
    }
    if (canonical_var) {
      *canonical_var = canonical;
    }
    return true;
  };

  std::function<void(const VarPtr&, const ExprPtr&, TpopVarRemap&)> propagate_alias_into;
  std::function<void(const StmtPtr&, TpopVarRemap&)> propagate_stmt_aliases_into;
  std::function<TpopVarRemap(const std::vector<StmtPtr>&, const TpopVarRemap&)> build_body_alias_map;

  propagate_alias_into = [&](const VarPtr& alias_var, const ExprPtr& src_expr, TpopVarRemap& remap) {
    VarPtr canonical_var;
    if (!alias_var || !try_resolve_canonical_tpop(src_expr, remap, &canonical_var)) {
      return;
    }
    remap[alias_var.get()] = canonical_var;
  };

  auto propagate_if_return_aliases_into = [&](const IfStmtPtr& if_stmt, TpopVarRemap& remap) {
    auto then_map = build_body_alias_map(FlattenBody(if_stmt->then_body_), remap);
    auto then_yield = GetLastYieldStmt(if_stmt->then_body_);
    std::optional<TpopVarRemap> else_map;
    YieldStmtPtr else_yield;
    if (if_stmt->else_body_.has_value()) {
      else_map = build_body_alias_map(FlattenBody(if_stmt->else_body_.value()), remap);
      else_yield = GetLastYieldStmt(if_stmt->else_body_.value());
    }

    for (size_t i = 0; i < if_stmt->return_vars_.size(); ++i) {
      VarPtr canonical_var;
      bool has_canonical = false;
      bool consistent = true;

      auto merge_yield_value = [&](const YieldStmtPtr& yield_stmt, const TpopVarRemap& branch_map) {
        if (!yield_stmt || i >= yield_stmt->value_.size()) {
          consistent = false;
          return;
        }
        VarPtr branch_canonical;
        if (!try_resolve_canonical_tpop(yield_stmt->value_[i], branch_map, &branch_canonical)) {
          consistent = false;
          return;
        }
        if (!has_canonical) {
          canonical_var = branch_canonical;
          has_canonical = true;
          return;
        }
        if (canonical_var.get() != branch_canonical.get()) {
          consistent = false;
        }
      };

      merge_yield_value(then_yield, then_map);
      if (else_map.has_value()) {
        merge_yield_value(else_yield, else_map.value());
      } else {
        consistent = false;
      }
      if (has_canonical && consistent) {
        remap[if_stmt->return_vars_[i].get()] = canonical_var;
      }
    }
  };

  auto propagate_loop_return_aliases_into = [&](const std::vector<IterArgPtr>& iter_args, const StmtPtr& body,
                                                const std::vector<VarPtr>& return_vars, TpopVarRemap& remap) {
    TpopVarRemap body_map = remap;
    for (const auto& iter_arg : iter_args) {
      propagate_alias_into(iter_arg, iter_arg ? iter_arg->initValue_ : nullptr, body_map);
    }
    body_map = build_body_alias_map(FlattenBody(body), body_map);
    auto yield_stmt = GetLastYieldStmt(body);
    if (!yield_stmt) {
      return;
    }
    for (size_t i = 0; i < iter_args.size() && i < yield_stmt->value_.size(); ++i) {
      propagate_alias_into(iter_args[i], yield_stmt->value_[i], body_map);
    }
    for (size_t i = 0; i < return_vars.size() && i < yield_stmt->value_.size(); ++i) {
      propagate_alias_into(return_vars[i], yield_stmt->value_[i], remap);
    }
  };

  propagate_stmt_aliases_into = [&](const StmtPtr& stmt, TpopVarRemap& remap) {
    if (auto assign = std::dynamic_pointer_cast<const AssignStmt>(stmt)) {
      propagate_alias_into(assign->var_, assign->value_, remap);
      return;
    }
    if (auto if_stmt = std::dynamic_pointer_cast<const IfStmt>(stmt)) {
      propagate_if_return_aliases_into(if_stmt, remap);
      return;
    }
    if (auto for_stmt = std::dynamic_pointer_cast<const ForStmt>(stmt)) {
      propagate_loop_return_aliases_into(for_stmt->iter_args_, for_stmt->body_, for_stmt->return_vars_,
                                         remap);
      return;
    }
    if (auto while_stmt = std::dynamic_pointer_cast<const WhileStmt>(stmt)) {
      propagate_loop_return_aliases_into(while_stmt->iter_args_, while_stmt->body_, while_stmt->return_vars_,
                                         remap);
    }
  };

  build_body_alias_map = [&](const std::vector<StmtPtr>& stmts, const TpopVarRemap& seed) -> TpopVarRemap {
    TpopVarRemap body_map = seed;
    for (const auto& stmt : stmts) {
      propagate_stmt_aliases_into(stmt, body_map);
    }
    return body_map;
  };

  for (size_t i = 0; i < normalized_inputs.size(); ++i) {
    VarPtr tpop_var;
    CallPtr tpop_call;
    if (IsTpopAssignStmt(normalized_inputs[i], &tpop_var, &tpop_call)) {
      const Var* canonical_var = CanonicalizeTpopRef(tpop_var.get(), local_tpop_var_remap);
      VarPtr chain_var = tpop_var;
      if (auto remap_it = local_tpop_var_remap.find(tpop_var.get());
          remap_it != local_tpop_var_remap.end() && remap_it->second) {
        chain_var = remap_it->second;
      }
      std::optional<int> pipe_id;
      if (tpop_call && tpop_call->HasKwarg("id")) {
        pipe_id = tpop_call->GetKwarg<int>("id", 0);
      }
      auto [it, inserted] = chains.emplace(canonical_var, TpopLifetime{i, no_tfree, chain_var, i, pipe_id});
      if (inserted) {
        tpop_order.push_back(canonical_var);
      } else {
        it->second.tpop_idx = i;
        it->second.tpop_var = chain_var;
        it->second.last_use_idx = (it->second.last_use_idx < i) ? i : it->second.last_use_idx;
        it->second.pipe_id = pipe_id;
      }
      continue;
    }

    VarPtr tfree_var;
    std::string tfree_op_name;
    if (IsTfreeStmt(normalized_inputs[i], &tfree_var, &tfree_op_name)) {
      const Var* canonical_var = CanonicalizeTpopRef(tfree_var.get(), local_tpop_var_remap);
      auto chain_it = chains.find(canonical_var);
      if (chain_it == chains.end() || tfree_op_name != expected_tfree) {
        continue;
      }
      if (tfree_var && chain_it->second.tpop_var && tfree_var.get() != chain_it->second.tpop_var.get()) {
        auto eval = std::dynamic_pointer_cast<const EvalStmt>(normalized_inputs[i]);
        auto call = eval ? std::dynamic_pointer_cast<const Call>(eval->expr_) : nullptr;
        INTERNAL_CHECK_SPAN(call && !call->args_.empty(), normalized_inputs[i]->span_)
            << "Internal error: expected tfree EvalStmt with a tile operand";
        auto new_call = MutableCopy(call);
        new_call->args_[0] = chain_it->second.tpop_var;
        auto new_eval = MutableCopy(eval);
        new_eval->expr_ = new_call;
        normalized_inputs[i] = new_eval;
      }
      chain_it->second.tfree_idx = i;
      continue;
    }

    const auto refs = CollectCallArgVarRefs(normalized_inputs[i]);
    for (const auto* ref : var_collectors::GetSortedVarRefs(refs)) {
      const Var* canonical_ref = CanonicalizeTpopRef(ref, local_tpop_var_remap);
      auto chain_it = chains.find(canonical_ref);
      if (chain_it == chains.end()) {
        continue;
      }
      chain_it->second.last_use_idx = i;
    }
    propagate_stmt_aliases_into(normalized_inputs[i], local_tpop_var_remap);
  }

  if (tpop_order.empty()) {
    return normalized_inputs;
  }

  std::vector<bool> remove_existing_tfree(normalized_inputs.size(), false);
  std::unordered_map<size_t, std::vector<StmtPtr>> deferred_tfrees;
  for (const auto* var : tpop_order) {
    auto& chain = chains.find(var)->second;
    if (chain.tfree_idx == no_tfree || chain.tfree_idx < chain.last_use_idx) {
      if (chain.tfree_idx != no_tfree) {
        remove_existing_tfree[chain.tfree_idx] = true;
      }
      deferred_tfrees[chain.last_use_idx].push_back(std::make_shared<EvalStmt>(
          CreateTfree(side, chain.tpop_var, normalized_inputs[chain.tpop_idx]->span_, chain.pipe_id),
          normalized_inputs[chain.tpop_idx]->span_));
    }
  }

  std::vector<StmtPtr> result;
  result.reserve(normalized_inputs.size() + deferred_tfrees.size());
  for (size_t i = 0; i < normalized_inputs.size(); ++i) {
    if (!remove_existing_tfree[i]) {
      result.push_back(normalized_inputs[i]);
    }
    if (auto deferred_it = deferred_tfrees.find(i); deferred_it != deferred_tfrees.end()) {
      result.insert(result.end(), deferred_it->second.begin(), deferred_it->second.end());
    }
  }
  return result;
}

}  // namespace tpop_tfree

namespace {

StmtPtr FinalizeNestedTpopTfrees(const StmtPtr& stmt, core_affinity::CoreSide side,
                                 const std::unordered_map<const Var*, VarPtr>& tpop_var_remap) {
  if (auto for_stmt = std::dynamic_pointer_cast<const ForStmt>(stmt)) {
    auto new_body = tpop_tfree::FinalizeTpopTfrees(FlattenBody(for_stmt->body_), side, tpop_var_remap);
    return loop_repair::RebuildForStmt(for_stmt, loop_repair::MakeBody(new_body, for_stmt->span_));
  }
  if (auto if_stmt = std::dynamic_pointer_cast<const IfStmt>(stmt)) {
    auto new_then = tpop_tfree::FinalizeTpopTfrees(FlattenBody(if_stmt->then_body_), side, tpop_var_remap);
    std::optional<std::vector<StmtPtr>> new_else;
    if (const auto& else_body = if_stmt->else_body_) {
      new_else = tpop_tfree::FinalizeTpopTfrees(FlattenBody(*else_body), side, tpop_var_remap);
    }
    return loop_repair::RebuildIfStmt(if_stmt, new_then, new_else);
  }
  if (auto while_stmt = std::dynamic_pointer_cast<const WhileStmt>(stmt)) {
    auto new_body = tpop_tfree::FinalizeTpopTfrees(FlattenBody(while_stmt->body_), side, tpop_var_remap);
    return loop_repair::RebuildWhileStmt(while_stmt, loop_repair::MakeBody(new_body, while_stmt->span_));
  }
  return stmt;
}

}  // namespace
}  // namespace ir
}  // namespace pypto
