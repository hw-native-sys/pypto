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

#include "pypto/codegen/pto/tpop_chain_reorder.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <unordered_set>
#include <vector>

#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/utils/scope_outline_utils.h"
#include "pypto/ir/transforms/utils/var_collectors.h"

namespace pypto {
namespace codegen {

using ir::As;

/// Collect variable references from a statement and classify against tpop chains.
struct StmtChainClassification {
  const ir::Var* tpop_ref = nullptr;
  bool is_multi_ref = false;
  bool has_unsafe_dep = false;
};

static std::unordered_set<const ir::Var*> CollectStmtVarRefs(const ir::StmtPtr& stmt) {
  std::unordered_set<const ir::Var*> refs;
  if (auto assign = As<ir::AssignStmt>(stmt)) {
    if (auto call = As<ir::Call>(assign->value_)) {
      for (const auto& arg : call->args_) {
        if (auto v = ir::AsVarLike(arg)) {
          refs.insert(v.get());
        } else {
          ir::outline_utils::VarDefUseCollector collector;
          collector.VisitExpr(arg);
          refs.insert(collector.var_uses.begin(), collector.var_uses.end());
        }
      }
    } else {
      ir::outline_utils::VarDefUseCollector collector;
      collector.VisitStmt(stmt);
      auto all_refs = collector.GetAllVarRefs();
      refs.insert(all_refs.begin(), all_refs.end());
    }
  } else {
    ir::outline_utils::VarDefUseCollector collector;
    collector.VisitStmt(stmt);
    auto all_refs = collector.GetAllVarRefs();
    refs.insert(all_refs.begin(), all_refs.end());
  }
  return refs;
}

static StmtChainClassification ClassifyStmtForTpopChain(
    const std::unordered_set<const ir::Var*>& refs,
    const std::map<const ir::Var*, TpopResultInfo>& tpop_result_vars,
    const std::set<const ir::Var*>& defined_since_first_tpop) {
  StmtChainClassification result;
  const auto sorted_refs = ir::var_collectors::GetSortedVarRefs(refs);
  for (const auto* var_ref : sorted_refs) {
    if (tpop_result_vars.count(var_ref) > 0) {
      if (result.tpop_ref && result.tpop_ref != var_ref) {
        result.is_multi_ref = true;
        break;
      }
      result.tpop_ref = var_ref;
    } else if (defined_since_first_tpop.count(var_ref) > 0) {
      result.has_unsafe_dep = true;
    }
  }
  return result;
}

std::vector<ir::StmtPtr> ReorderTpopChains(const std::vector<ir::StmtPtr>& stmts,
                                           const std::map<const ir::Var*, TpopResultInfo>& tpop_result_vars) {
  if (tpop_result_vars.empty()) return stmts;

  auto make_body = [](const std::vector<ir::StmtPtr>& body_stmts, const ir::Span& span) -> ir::StmtPtr {
    return std::make_shared<ir::SeqStmts>(body_stmts, span);
  };
  auto flatten_body = [](const ir::StmtPtr& body) -> std::vector<ir::StmtPtr> {
    if (auto seq = As<ir::SeqStmts>(body)) {
      return seq->stmts_;
    }
    return {body};
  };

  std::function<ir::StmtPtr(const ir::StmtPtr&)> reorder_nested =
      [&](const ir::StmtPtr& stmt) -> ir::StmtPtr {
    if (auto for_stmt = As<ir::ForStmt>(stmt)) {
      auto new_body = ReorderTpopChains(flatten_body(for_stmt->body_), tpop_result_vars);
      return std::make_shared<ir::ForStmt>(
          for_stmt->loop_var_, for_stmt->start_, for_stmt->stop_, for_stmt->step_, for_stmt->iter_args_,
          make_body(new_body, for_stmt->span_), for_stmt->return_vars_, for_stmt->span_, for_stmt->kind_,
          for_stmt->chunk_size_, for_stmt->chunk_policy_, for_stmt->loop_origin_);
    }
    if (auto if_stmt = As<ir::IfStmt>(stmt)) {
      auto new_then = ReorderTpopChains(flatten_body(if_stmt->then_body_), tpop_result_vars);
      std::optional<ir::StmtPtr> new_else;
      if (if_stmt->else_body_.has_value()) {
        auto else_stmts = ReorderTpopChains(flatten_body(if_stmt->else_body_.value()), tpop_result_vars);
        new_else = make_body(else_stmts, if_stmt->span_);
      }
      return std::make_shared<ir::IfStmt>(if_stmt->condition_, make_body(new_then, if_stmt->span_), new_else,
                                          if_stmt->return_vars_, if_stmt->span_);
    }
    if (auto while_stmt = As<ir::WhileStmt>(stmt)) {
      auto new_body = ReorderTpopChains(flatten_body(while_stmt->body_), tpop_result_vars);
      return std::make_shared<ir::WhileStmt>(while_stmt->condition_, while_stmt->iter_args_,
                                             make_body(new_body, while_stmt->span_), while_stmt->return_vars_,
                                             while_stmt->span_);
    }
    return stmt;
  };

  std::vector<ir::StmtPtr> normalized_inputs;
  normalized_inputs.reserve(stmts.size());
  for (const auto& stmt : stmts) {
    normalized_inputs.push_back(reorder_nested(stmt));
  }

  struct TpopChain {
    size_t tpop_idx;
    std::vector<size_t> user_idxs;
    size_t tfree_idx = SIZE_MAX;
    size_t last_use_idx = 0;
  };
  std::map<const ir::Var*, TpopChain> chains;
  std::vector<const ir::Var*> tpop_order;
  std::vector<bool> in_chain(normalized_inputs.size(), false);

  std::set<const ir::Var*> defined_since_first_tpop;
  bool seen_first_tpop = false;
  for (size_t i = 0; i < normalized_inputs.size(); ++i) {
    const auto stmt_defined_vars = ir::var_collectors::CollectStmtDefinedVars(normalized_inputs[i]);

    if (auto assign = As<ir::AssignStmt>(normalized_inputs[i])) {
      if (tpop_result_vars.count(assign->var_.get()) > 0) {
        seen_first_tpop = true;
        chains[assign->var_.get()] = {i, {}, SIZE_MAX, i};
        tpop_order.push_back(assign->var_.get());
        in_chain[i] = true;
        defined_since_first_tpop.insert(stmt_defined_vars.begin(), stmt_defined_vars.end());
        continue;
      }
    }

    if (!seen_first_tpop) {
      continue;
    }

    // Handle tfree statements specially
    if (auto eval = As<ir::EvalStmt>(normalized_inputs[i])) {
      if (auto call = As<ir::Call>(eval->expr_)) {
        if (call->op_ &&
            (call->op_->name_ == "system.tfree_to_aiv" || call->op_->name_ == "system.tfree_to_aic") &&
            !call->args_.empty()) {
          if (auto v = ir::AsVarLike(call->args_[0]); v && tpop_result_vars.count(v.get()) > 0) {
            if (chains.count(v.get()) > 0) {
              chains[v.get()].tfree_idx = i;
            }
            defined_since_first_tpop.insert(stmt_defined_vars.begin(), stmt_defined_vars.end());
            continue;
          }
        }
      }
    }

    auto refs = CollectStmtVarRefs(normalized_inputs[i]);
    auto classification = ClassifyStmtForTpopChain(refs, tpop_result_vars, defined_since_first_tpop);

    if (classification.tpop_ref) {
      if (chains.count(classification.tpop_ref) > 0) {
        chains[classification.tpop_ref].last_use_idx =
            std::max(chains[classification.tpop_ref].last_use_idx, i);
      }
    }

    if (classification.tpop_ref && !classification.is_multi_ref && !classification.has_unsafe_dep &&
        chains.count(classification.tpop_ref) > 0) {
      chains[classification.tpop_ref].user_idxs.push_back(i);
      in_chain[i] = true;
    }
    defined_since_first_tpop.insert(stmt_defined_vars.begin(), stmt_defined_vars.end());
  }

  if (tpop_order.empty()) return normalized_inputs;

  size_t first_tpop = chains[tpop_order[0]].tpop_idx;
  std::vector<ir::StmtPtr> result;
  result.reserve(normalized_inputs.size());

  for (size_t i = 0; i < first_tpop; ++i) {
    result.push_back(normalized_inputs[i]);
  }

  for (const auto* var : tpop_order) {
    auto& ch = chains[var];
    result.push_back(normalized_inputs[ch.tpop_idx]);
    for (size_t ui : ch.user_idxs) {
      result.push_back(normalized_inputs[ui]);
    }
    size_t last_grouped_idx = ch.user_idxs.empty() ? ch.tpop_idx : ch.user_idxs.back();
    if (ch.tfree_idx != SIZE_MAX && ch.last_use_idx <= last_grouped_idx) {
      result.push_back(normalized_inputs[ch.tfree_idx]);
      in_chain[ch.tfree_idx] = true;
    }
  }

  for (size_t i = first_tpop; i < normalized_inputs.size(); ++i) {
    if (!in_chain[i]) {
      result.push_back(normalized_inputs[i]);
    }
  }

  return result;
}

}  // namespace codegen
}  // namespace pypto
