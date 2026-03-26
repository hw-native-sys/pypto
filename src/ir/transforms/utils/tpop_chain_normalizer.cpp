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

#include "pypto/ir/transforms/utils/tpop_chain_normalizer.h"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/utils/core_affinity.h"
#include "pypto/ir/transforms/utils/loop_state_repair.h"
#include "pypto/ir/transforms/utils/scope_outline_utils.h"
#include "pypto/ir/transforms/utils/transform_utils.h"

namespace pypto {
namespace ir {
namespace {

const auto& FlattenBody = transform_utils::FlattenToStmts;

StmtPtr NormalizeNestedTpopChains(const StmtPtr& stmt, core_affinity::CoreSide side,
                                  const std::unordered_map<const Var*, VarPtr>& tpop_var_remap);

}  // namespace

namespace tpop_chain {

std::string GetTfreeOpName(core_affinity::CoreSide side) {
  return (side == core_affinity::CoreSide::AIC) ? "system.tfree_to_aiv" : "system.tfree_to_aic";
}

CallPtr CreateTfree(core_affinity::CoreSide side, const ExprPtr& tile, const Span& span) {
  return OpRegistry::GetInstance().Create(GetTfreeOpName(side), {tile}, {}, span);
}

bool IsTpopAssignStmt(const StmtPtr& stmt, VarPtr* result_var) {
  auto assign = std::dynamic_pointer_cast<const AssignStmt>(stmt);
  if (!assign) return false;
  auto call = std::dynamic_pointer_cast<const Call>(assign->value_);
  auto op = call ? std::dynamic_pointer_cast<const Op>(call->op_) : nullptr;
  if (!op || (op->name_ != "tile.tpop_from_aiv" && op->name_ != "tile.tpop_from_aic")) {
    return false;
  }
  if (result_var) *result_var = assign->var_;
  return true;
}

bool IsExpectedTpopOp(const std::string& op_name, FunctionType func_type) {
  if (func_type == FunctionType::AIC) return op_name == "tile.tpop_from_aiv";
  if (func_type == FunctionType::AIV) return op_name == "tile.tpop_from_aic";
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
  auto op = call ? std::dynamic_pointer_cast<const Op>(call->op_) : nullptr;
  if (!op || (op->name_ != "system.tfree_to_aiv" && op->name_ != "system.tfree_to_aic")) {
    return false;
  }
  if (op_name) *op_name = op->name_;
  if (tile_var) *tile_var = call && !call->args_.empty() ? AsVarLike(call->args_[0]) : nullptr;
  return true;
}

std::unordered_set<const Var*> CollectStmtVarRefs(const StmtPtr& stmt) {
  outline_utils::VarRefCollector refs;
  refs.VisitStmt(stmt);
  return refs.var_refs;
}

std::unordered_set<const Var*> CollectStmtDefinedVars(const StmtPtr& stmt) {
  std::unordered_set<const Var*> defs;
  if (auto assign = std::dynamic_pointer_cast<const AssignStmt>(stmt)) {
    defs.insert(assign->var_.get());
  } else if (auto for_stmt = std::dynamic_pointer_cast<const ForStmt>(stmt)) {
    for (const auto& ret : for_stmt->return_vars_) {
      defs.insert(ret.get());
    }
  } else if (auto if_stmt = std::dynamic_pointer_cast<const IfStmt>(stmt)) {
    for (const auto& ret : if_stmt->return_vars_) {
      defs.insert(ret.get());
    }
  } else if (auto while_stmt = std::dynamic_pointer_cast<const WhileStmt>(stmt)) {
    for (const auto& ret : while_stmt->return_vars_) {
      defs.insert(ret.get());
    }
  }
  return defs;
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
    outline_utils::VarRefCollector refs;
    refs.VisitExpr(arg);
    refs_set.insert(refs.var_refs.begin(), refs.var_refs.end());
  }
  return refs_set;
}

std::vector<const Var*> GetSortedVarRefs(const std::unordered_set<const Var*>& refs) {
  std::vector<const Var*> sorted_refs(refs.begin(), refs.end());
  std::sort(sorted_refs.begin(), sorted_refs.end(), [](const Var* lhs, const Var* rhs) {
    if (lhs == rhs) return false;
    if (lhs->name_hint_ != rhs->name_hint_) return lhs->name_hint_ < rhs->name_hint_;
    return lhs->UniqueId() < rhs->UniqueId();
  });
  return sorted_refs;
}

bool StmtReferencesVar(const StmtPtr& stmt, const Var* var) {
  if (!stmt || !var) return false;
  auto refs = CollectStmtVarRefs(stmt);
  return refs.count(var) > 0;
}

const Var* CanonicalizeTpopRef(const Var* var, const std::unordered_map<const Var*, VarPtr>& tpop_var_remap) {
  if (!var) return nullptr;
  auto it = tpop_var_remap.find(var);
  return (it != tpop_var_remap.end() && it->second) ? it->second.get() : var;
}

std::vector<StmtPtr> NormalizeTpopChains(const std::vector<StmtPtr>& stmts, core_affinity::CoreSide side,
                                         const std::unordered_map<const Var*, VarPtr>& tpop_var_remap) {
  std::vector<StmtPtr> normalized_inputs;
  normalized_inputs.reserve(stmts.size());
  for (const auto& stmt : stmts) {
    normalized_inputs.push_back(NormalizeNestedTpopChains(stmt, side, tpop_var_remap));
  }

  std::map<const Var*, TpopChain> chains;
  std::vector<const Var*> tpop_order;
  std::vector<bool> in_chain(normalized_inputs.size(), false);
  std::unordered_map<const Var*, size_t> def_indices;
  bool seen_first_tpop = false;

  for (size_t i = 0; i < normalized_inputs.size(); ++i) {
    auto assign = std::dynamic_pointer_cast<const AssignStmt>(normalized_inputs[i]);
    const auto stmt_defs = CollectStmtDefinedVars(normalized_inputs[i]);
    for (const auto* def : GetSortedVarRefs(stmt_defs)) {
      def_indices.try_emplace(def, i);
    }
    VarPtr tpop_var;
    if (assign && IsTpopAssignStmt(normalized_inputs[i], &tpop_var)) {
      seen_first_tpop = true;
      chains.emplace(tpop_var.get(), TpopChain{i, {}, std::numeric_limits<size_t>::max(), tpop_var, i});
      tpop_order.push_back(tpop_var.get());
      in_chain[i] = true;
      continue;
    }

    if (!seen_first_tpop) continue;

    VarPtr tfree_var;
    std::string tfree_op_name;
    const Var* tfree_key = nullptr;
    if (IsTfreeStmt(normalized_inputs[i], &tfree_var, &tfree_op_name) && tfree_var &&
        tfree_op_name == GetTfreeOpName(side)) {
      tfree_key = CanonicalizeTpopRef(tfree_var.get(), tpop_var_remap);
    }
    if (tfree_key && chains.count(tfree_key) > 0) {
      normalized_inputs[i] = std::make_shared<EvalStmt>(
          CreateTfree(side, chains.find(tfree_key)->second.tpop_var, normalized_inputs[i]->span_),
          normalized_inputs[i]->span_);
      chains.find(tfree_key)->second.tfree_idx = i;
      in_chain[i] = true;
      continue;
    }

    std::unordered_set<const Var*> refs;
    CallPtr call;
    if (assign) {
      call = std::dynamic_pointer_cast<const Call>(assign->value_);
    } else if (auto eval = std::dynamic_pointer_cast<const EvalStmt>(normalized_inputs[i])) {
      call = std::dynamic_pointer_cast<const Call>(eval->expr_);
    }
    if (call) {
      auto CollectExprRefs = [&](const ExprPtr& expr) {
        if (auto var_like = AsVarLike(expr)) {
          refs.insert(var_like.get());
          return;
        }
        outline_utils::VarRefCollector collector;
        collector.VisitExpr(expr);
        refs.insert(collector.var_refs.begin(), collector.var_refs.end());
      };
      for (const auto& arg : call->args_) {
        CollectExprRefs(arg);
      }
    } else {
      refs = CollectStmtVarRefs(normalized_inputs[i]);
    }
    const auto sorted_refs = GetSortedVarRefs(refs);
    for (const auto* ref : sorted_refs) {
      const Var* canonical_ref = CanonicalizeTpopRef(ref, tpop_var_remap);
      if (canonical_ref && chains.count(canonical_ref) > 0) {
        chains.find(canonical_ref)->second.last_use_idx =
            std::max(chains.find(canonical_ref)->second.last_use_idx, i);
      }
    }
    const Var* referenced_tpop = nullptr;
    bool has_multi_tpop_refs = false;
    for (const auto* ref : sorted_refs) {
      const Var* canonical_ref = CanonicalizeTpopRef(ref, tpop_var_remap);
      if (canonical_ref && chains.count(canonical_ref) > 0) {
        if (referenced_tpop && referenced_tpop != canonical_ref) {
          has_multi_tpop_refs = true;
          break;
        }
        referenced_tpop = canonical_ref;
      }
    }
    bool has_unsafe_dep = false;
    if (referenced_tpop && !has_multi_tpop_refs) {
      size_t tpop_idx = chains.find(referenced_tpop)->second.tpop_idx;
      for (const auto* ref : sorted_refs) {
        const Var* canonical_ref = CanonicalizeTpopRef(ref, tpop_var_remap);
        if (canonical_ref == referenced_tpop) continue;
        auto def_it = def_indices.find(ref);
        if (def_it != def_indices.end() && def_it->second > tpop_idx && def_it->second < i) {
          has_unsafe_dep = true;
          break;
        }
      }
    }
    if (referenced_tpop && !has_multi_tpop_refs && !has_unsafe_dep) {
      chains.find(referenced_tpop)->second.user_idxs.push_back(i);
      in_chain[i] = true;
    }
  }

  if (tpop_order.empty()) return normalized_inputs;

  size_t first_tpop = chains[tpop_order[0]].tpop_idx;
  std::vector<StmtPtr> result;
  result.reserve(normalized_inputs.size() + tpop_order.size());
  std::unordered_map<size_t, std::vector<StmtPtr>> deferred_tfrees;
  for (size_t i = 0; i < first_tpop; ++i) {
    result.push_back(normalized_inputs[i]);
  }

  for (const auto* var : tpop_order) {
    auto& chain = chains[var];
    result.push_back(normalized_inputs[chain.tpop_idx]);
    for (size_t user_idx : chain.user_idxs) {
      result.push_back(normalized_inputs[user_idx]);
    }
    size_t last_grouped_idx = chain.user_idxs.empty() ? chain.tpop_idx : chain.user_idxs.back();
    if (chain.last_use_idx <= last_grouped_idx) {
      if (chain.tfree_idx != std::numeric_limits<size_t>::max()) {
        result.push_back(normalized_inputs[chain.tfree_idx]);
        in_chain[chain.tfree_idx] = true;
      } else {
        result.push_back(std::make_shared<EvalStmt>(
            CreateTfree(side, chain.tpop_var, normalized_inputs[chain.tpop_idx]->span_),
            normalized_inputs[chain.tpop_idx]->span_));
      }
    } else if (chain.tfree_idx == std::numeric_limits<size_t>::max() ||
               chain.tfree_idx < chain.last_use_idx) {
      if (chain.tfree_idx != std::numeric_limits<size_t>::max()) {
        in_chain[chain.tfree_idx] = true;
      }
      deferred_tfrees[chain.last_use_idx].push_back(std::make_shared<EvalStmt>(
          CreateTfree(side, chain.tpop_var, normalized_inputs[chain.tpop_idx]->span_),
          normalized_inputs[chain.tpop_idx]->span_));
    } else if (chain.tfree_idx > chain.last_use_idx) {
      // Keep an existing tfree in its original position when it already follows
      // the true last use and moving it earlier would be unsafe.
      continue;
    } else {
      deferred_tfrees[chain.last_use_idx].push_back(std::make_shared<EvalStmt>(
          CreateTfree(side, chain.tpop_var, normalized_inputs[chain.tpop_idx]->span_),
          normalized_inputs[chain.tpop_idx]->span_));
      in_chain[chain.tfree_idx] = true;
    }
  }

  for (size_t i = first_tpop; i < normalized_inputs.size(); ++i) {
    if (!in_chain[i]) {
      result.push_back(normalized_inputs[i]);
    }
    auto deferred_it = deferred_tfrees.find(i);
    if (deferred_it != deferred_tfrees.end()) {
      result.insert(result.end(), deferred_it->second.begin(), deferred_it->second.end());
    }
  }
  return result;
}

}  // namespace tpop_chain

namespace {

StmtPtr NormalizeNestedTpopChains(const StmtPtr& stmt, core_affinity::CoreSide side,
                                  const std::unordered_map<const Var*, VarPtr>& tpop_var_remap) {
  if (auto for_stmt = std::dynamic_pointer_cast<const ForStmt>(stmt)) {
    auto new_body = tpop_chain::NormalizeTpopChains(FlattenBody(for_stmt->body_), side, tpop_var_remap);
    return loop_repair::RebuildForStmt(for_stmt, loop_repair::MakeBody(new_body, for_stmt->span_));
  }
  if (auto if_stmt = std::dynamic_pointer_cast<const IfStmt>(stmt)) {
    auto new_then = tpop_chain::NormalizeTpopChains(FlattenBody(if_stmt->then_body_), side, tpop_var_remap);
    std::optional<std::vector<StmtPtr>> new_else;
    if (if_stmt->else_body_.has_value()) {
      new_else =
          tpop_chain::NormalizeTpopChains(FlattenBody(if_stmt->else_body_.value()), side, tpop_var_remap);
    }
    return loop_repair::RebuildIfStmt(if_stmt, new_then, new_else);
  }
  if (auto while_stmt = std::dynamic_pointer_cast<const WhileStmt>(stmt)) {
    auto new_body = tpop_chain::NormalizeTpopChains(FlattenBody(while_stmt->body_), side, tpop_var_remap);
    return loop_repair::RebuildWhileStmt(while_stmt, loop_repair::MakeBody(new_body, while_stmt->span_));
  }
  return stmt;
}

}  // namespace
}  // namespace ir
}  // namespace pypto
