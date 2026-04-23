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
#include "pypto/ir/transforms/utils/var_collectors.h"

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
  return refs_set;
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

  for (size_t i = 0; i < normalized_inputs.size(); ++i) {
    VarPtr tpop_var;
    if (IsTpopAssignStmt(normalized_inputs[i], &tpop_var)) {
      chains.emplace(tpop_var.get(), TpopChain{i, {}, std::numeric_limits<size_t>::max(), tpop_var, i});
      tpop_order.push_back(tpop_var.get());
      continue;
    }

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
      continue;
    }

    auto refs = CollectCallArgVarRefs(normalized_inputs[i]);
    const auto sorted_refs = var_collectors::GetSortedVarRefs(refs);
    for (const auto* ref : sorted_refs) {
      const Var* canonical_ref = CanonicalizeTpopRef(ref, tpop_var_remap);
      if (canonical_ref && chains.count(canonical_ref) > 0) {
        chains.find(canonical_ref)->second.last_use_idx =
            std::max(chains.find(canonical_ref)->second.last_use_idx, i);
      }
    }
  }

  if (tpop_order.empty()) return normalized_inputs;

  std::vector<bool> skip_existing_tfree(normalized_inputs.size(), false);
  std::vector<std::vector<StmtPtr>> insert_after(normalized_inputs.size());
  for (const auto* var : tpop_order) {
    const auto& chain = chains[var];
    if (chain.tfree_idx == std::numeric_limits<size_t>::max()) {
      insert_after[chain.last_use_idx].push_back(std::make_shared<EvalStmt>(
          CreateTfree(side, chain.tpop_var, normalized_inputs[chain.tpop_idx]->span_),
          normalized_inputs[chain.tpop_idx]->span_));
      continue;
    }
    if (chain.tfree_idx < chain.last_use_idx) {
      skip_existing_tfree[chain.tfree_idx] = true;
      insert_after[chain.last_use_idx].push_back(std::make_shared<EvalStmt>(
          CreateTfree(side, chain.tpop_var, normalized_inputs[chain.tfree_idx]->span_),
          normalized_inputs[chain.tfree_idx]->span_));
    }
  }

  std::vector<StmtPtr> result;
  result.reserve(normalized_inputs.size() + tpop_order.size());
  for (size_t i = 0; i < normalized_inputs.size(); ++i) {
    if (!skip_existing_tfree[i]) {
      result.push_back(normalized_inputs[i]);
    }
    if (!insert_after[i].empty()) {
      result.insert(result.end(), insert_after[i].begin(), insert_after[i].end());
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
