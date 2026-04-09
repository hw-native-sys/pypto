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

#include "pypto/ir/transforms/utils/loop_state_repair.h"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/utils/dead_code_elimination.h"
#include "pypto/ir/transforms/utils/scope_outline_utils.h"
#include "pypto/ir/transforms/utils/transform_utils.h"

namespace pypto {
namespace ir {
namespace loop_repair {

const auto& FlattenBody = transform_utils::FlattenToStmts;

StmtPtr MakeBody(const std::vector<StmtPtr>& stmts, const Span& span) {
  return SeqStmts::Flatten(std::vector<StmtPtr>(stmts), span);
}

StmtPtr RebuildForStmt(const std::shared_ptr<const ForStmt>& f, const StmtPtr& new_body) {
  return std::make_shared<ForStmt>(f->loop_var_, f->start_, f->stop_, f->step_, f->iter_args_, new_body,
                                   f->return_vars_, f->span_, f->kind_, f->chunk_config_, f->attrs_);
}

StmtPtr RebuildForStmt(const std::shared_ptr<const ForStmt>& f, const std::vector<IterArgPtr>& iter_args,
                       const StmtPtr& new_body, const std::vector<VarPtr>& return_vars) {
  return std::make_shared<ForStmt>(f->loop_var_, f->start_, f->stop_, f->step_, iter_args, new_body,
                                   return_vars, f->span_, f->kind_, f->chunk_config_, f->attrs_);
}

StmtPtr RebuildWhileStmt(const std::shared_ptr<const WhileStmt>& w, const StmtPtr& new_body) {
  return std::make_shared<WhileStmt>(w->condition_, w->iter_args_, new_body, w->return_vars_, w->span_);
}

StmtPtr RebuildWhileStmt(const std::shared_ptr<const WhileStmt>& w, const std::vector<IterArgPtr>& iter_args,
                         const StmtPtr& new_body, const std::vector<VarPtr>& return_vars) {
  return std::make_shared<WhileStmt>(w->condition_, iter_args, new_body, return_vars, w->span_);
}

StmtPtr RebuildIfStmt(const std::shared_ptr<const IfStmt>& s, const std::vector<StmtPtr>& new_then,
                      const std::optional<std::vector<StmtPtr>>& new_else_stmts) {
  std::optional<StmtPtr> new_else;
  if (new_else_stmts.has_value()) {
    new_else = MakeBody(new_else_stmts.value(), s->span_);
  }
  return std::make_shared<IfStmt>(s->condition_, MakeBody(new_then, s->span_), new_else, s->return_vars_,
                                  s->span_);
}

StmtPtr RebuildLoop(const std::shared_ptr<const ForStmt>& for_stmt,
                    const std::shared_ptr<const WhileStmt>& while_stmt,
                    const std::vector<IterArgPtr>& iter_args, const StmtPtr& new_body,
                    const std::vector<VarPtr>& return_vars) {
  if (for_stmt) return RebuildForStmt(for_stmt, iter_args, new_body, return_vars);
  return RebuildWhileStmt(while_stmt, iter_args, new_body, return_vars);
}

// ============================================================================
// Internal helpers
// ============================================================================

namespace {

void CollectBodyRefsSkippingYield(const std::vector<StmtPtr>& stmts, std::unordered_set<const Var*>& refs) {
  for (const auto& stmt : stmts) {
    if (std::dynamic_pointer_cast<const YieldStmt>(stmt)) continue;
    outline_utils::VarDefUseCollector collector;
    collector.VisitStmt(stmt);
    auto all_refs = collector.GetAllVarRefs();
    refs.insert(all_refs.begin(), all_refs.end());
  }
}

StmtPtr FilterYieldStmt(const StmtPtr& stmt, const std::vector<size_t>& kept_indices) {
  return TransformLastStmt(stmt, [&](const StmtPtr& s) -> StmtPtr {
    auto yield_stmt = std::dynamic_pointer_cast<const YieldStmt>(s);
    if (!yield_stmt) return s;
    if (kept_indices.empty()) return nullptr;
    std::vector<ExprPtr> new_values;
    for (size_t idx : kept_indices) {
      INTERNAL_CHECK(idx < yield_stmt->value_.size())
          << "Internal error: yield index " << idx << " out of range " << yield_stmt->value_.size();
      new_values.push_back(yield_stmt->value_[idx]);
    }
    return std::make_shared<YieldStmt>(new_values, yield_stmt->span_);
  });
}

StmtPtr FixDanglingYieldStmt(const StmtPtr& stmt, const std::vector<IterArgPtr>& iter_args,
                             const std::unordered_set<const Var*>& defined_vars) {
  return TransformLastStmt(stmt, [&](const StmtPtr& s) -> StmtPtr {
    auto yield_stmt = std::dynamic_pointer_cast<const YieldStmt>(s);
    if (!yield_stmt) return s;

    std::vector<ExprPtr> new_values;
    for (size_t i = 0; i < yield_stmt->value_.size(); ++i) {
      outline_utils::VarDefUseCollector collector;
      collector.VisitExpr(yield_stmt->value_[i]);
      bool has_undefined = std::any_of(collector.var_uses.begin(), collector.var_uses.end(),
                                       [&](const Var* ref) { return !defined_vars.count(ref); });
      if (has_undefined && i < iter_args.size()) {
        new_values.push_back(iter_args[i]);
      } else {
        new_values.push_back(yield_stmt->value_[i]);
      }
    }
    return std::make_shared<YieldStmt>(new_values, yield_stmt->span_);
  });
}

std::vector<StmtPtr> FixDanglingLoopBodyYields(const std::vector<StmtPtr>& stmts,
                                               const std::vector<IterArgPtr>& iter_args,
                                               const std::unordered_set<const Var*>& defined_vars) {
  std::vector<StmtPtr> result;
  result.reserve(stmts.size());
  for (const auto& stmt : stmts) {
    result.push_back(FixDanglingYieldStmt(stmt, iter_args, defined_vars));
  }
  return result;
}

void PullDefinitionChain(const Var* var_ptr, const std::unordered_map<const Var*, StmtPtr>& def_map,
                         const std::unordered_set<const Var*>& already_defined,
                         std::unordered_set<const Var*>& pulled, std::vector<StmtPtr>& out) {
  if (pulled.count(var_ptr) || already_defined.count(var_ptr)) return;
  auto it = def_map.find(var_ptr);
  if (it == def_map.end()) return;

  pulled.insert(var_ptr);

  auto assign = std::dynamic_pointer_cast<const AssignStmt>(it->second);
  if (assign) {
    outline_utils::VarDefUseCollector collector;
    collector.VisitExpr(assign->value_);
    for (const Var* dep : collector.var_uses) {
      PullDefinitionChain(dep, def_map, already_defined, pulled, out);
    }
  }

  out.push_back(it->second);
}

}  // namespace

// ============================================================================
// Public functions
// ============================================================================

std::vector<StmtPtr> StripDeadIterArgs(const std::vector<StmtPtr>& stmts) {
  std::vector<std::unordered_set<const Var*>> suffix_refs(stmts.size());
  for (size_t i = stmts.size(); i-- > 0;) {
    if (i + 1 < stmts.size()) {
      suffix_refs[i] = suffix_refs[i + 1];
    }
    outline_utils::VarDefUseCollector collector;
    collector.VisitStmt(stmts[i]);
    auto all_refs = collector.GetAllVarRefs();
    suffix_refs[i].insert(all_refs.begin(), all_refs.end());
  }

  std::vector<StmtPtr> result;

  for (size_t idx = 0; idx < stmts.size(); ++idx) {
    auto for_stmt = std::dynamic_pointer_cast<const ForStmt>(stmts[idx]);
    auto while_stmt = std::dynamic_pointer_cast<const WhileStmt>(stmts[idx]);

    if (!for_stmt && !while_stmt) {
      if (auto if_stmt = std::dynamic_pointer_cast<const IfStmt>(stmts[idx])) {
        auto new_then = StripDeadIterArgs(FlattenBody(if_stmt->then_body_));
        auto new_else =
            ProcessElseBranch(if_stmt, [](const std::vector<StmtPtr>& es) { return StripDeadIterArgs(es); });
        result.push_back(RebuildIfStmt(if_stmt, new_then, new_else));
      } else {
        result.push_back(stmts[idx]);
      }
      continue;
    }

    const auto& iter_args = for_stmt ? for_stmt->iter_args_ : while_stmt->iter_args_;
    const auto& return_vars = for_stmt ? for_stmt->return_vars_ : while_stmt->return_vars_;
    const auto& body = for_stmt ? for_stmt->body_ : while_stmt->body_;
    const auto& span = for_stmt ? for_stmt->span_ : while_stmt->span_;

    auto processed_body = StripDeadIterArgs(FlattenBody(body));

    if (iter_args.empty()) {
      result.push_back(
          RebuildLoop(for_stmt, while_stmt, iter_args, MakeBody(processed_body, span), return_vars));
      continue;
    }

    std::unordered_set<const Var*> body_refs;
    CollectBodyRefsSkippingYield(processed_body, body_refs);

    static const std::unordered_set<const Var*> kEmptyRefs;
    const auto& after_refs = (idx + 1 < stmts.size()) ? suffix_refs[idx + 1] : kEmptyRefs;

    std::vector<size_t> kept_indices;
    for (size_t i = 0; i < iter_args.size(); ++i) {
      bool used_in_body = body_refs.count(iter_args[i].get()) > 0;
      bool return_var_used = i < return_vars.size() && after_refs.count(return_vars[i].get()) > 0;
      if (used_in_body || return_var_used) {
        kept_indices.push_back(i);
      }
    }

    std::vector<IterArgPtr> new_iter_args;
    std::vector<VarPtr> new_return_vars;
    for (size_t i : kept_indices) {
      new_iter_args.push_back(iter_args[i]);
      if (i < return_vars.size()) {
        new_return_vars.push_back(return_vars[i]);
      }
    }

    if (kept_indices.size() < iter_args.size() && !processed_body.empty()) {
      auto filtered_last = FilterYieldStmt(processed_body.back(), kept_indices);
      if (filtered_last) {
        processed_body.back() = filtered_last;
      } else {
        processed_body.pop_back();
      }
    }

    result.push_back(
        RebuildLoop(for_stmt, while_stmt, new_iter_args, MakeBody(processed_body, span), new_return_vars));
  }

  return result;
}

void BuildDefMap(const std::vector<StmtPtr>& stmts, std::unordered_map<const Var*, StmtPtr>& def_map) {
  for (const auto& stmt : stmts) {
    if (auto assign = std::dynamic_pointer_cast<const AssignStmt>(stmt)) {
      def_map[assign->var_.get()] = stmt;
    }
  }
}

std::vector<StmtPtr> FixupIterArgInitValues(const std::vector<StmtPtr>& stmts,
                                            const std::unordered_map<const Var*, StmtPtr>& original_def_map) {
  auto recurse = [&](const std::vector<StmtPtr>& s) { return FixupIterArgInitValues(s, original_def_map); };

  std::unordered_set<const Var*> defined_so_far;
  std::vector<StmtPtr> result;
  std::unordered_set<const Var*> pulled;

  for (const auto& stmt : stmts) {
    auto for_stmt = std::dynamic_pointer_cast<const ForStmt>(stmt);
    auto while_stmt = std::dynamic_pointer_cast<const WhileStmt>(stmt);

    const std::vector<IterArgPtr>* iter_args_ptr = nullptr;
    if (for_stmt) {
      iter_args_ptr = &for_stmt->iter_args_;
    } else if (while_stmt) {
      iter_args_ptr = &while_stmt->iter_args_;
    }
    if (iter_args_ptr && !iter_args_ptr->empty()) {
      std::vector<StmtPtr> missing_defs;
      for (const auto& iter_arg : *iter_args_ptr) {
        outline_utils::VarDefUseCollector collector;
        collector.VisitExpr(iter_arg->initValue_);
        for (const Var* ref : collector.var_uses) {
          if (!defined_so_far.count(ref) && !pulled.count(ref)) {
            PullDefinitionChain(ref, original_def_map, defined_so_far, pulled, missing_defs);
          }
        }
      }
      for (const auto& def : missing_defs) {
        if (auto assign = std::dynamic_pointer_cast<const AssignStmt>(def)) {
          defined_so_far.insert(assign->var_.get());
        }
      }
      result.insert(result.end(), missing_defs.begin(), missing_defs.end());
    }

    outline_utils::VarDefUseCollector stmt_defs;
    stmt_defs.VisitStmt(stmt);
    defined_so_far.insert(stmt_defs.var_defs.begin(), stmt_defs.var_defs.end());

    if (for_stmt) {
      auto new_body = recurse(FlattenBody(for_stmt->body_));
      result.push_back(RebuildForStmt(for_stmt, MakeBody(new_body, for_stmt->span_)));
    } else if (auto if_stmt = std::dynamic_pointer_cast<const IfStmt>(stmt)) {
      auto new_then = recurse(FlattenBody(if_stmt->then_body_));
      auto new_else = ProcessElseBranch(if_stmt, [&](const std::vector<StmtPtr>& es) { return recurse(es); });
      result.push_back(RebuildIfStmt(if_stmt, new_then, new_else));
    } else if (while_stmt) {
      auto new_body = recurse(FlattenBody(while_stmt->body_));
      result.push_back(RebuildWhileStmt(while_stmt, MakeBody(new_body, while_stmt->span_)));
    } else {
      result.push_back(stmt);
    }
  }

  return result;
}

std::vector<StmtPtr> FixupDanglingYieldValues(const std::vector<StmtPtr>& stmts) {
  std::unordered_set<const Var*> defined_so_far;

  std::vector<StmtPtr> result;
  for (const auto& stmt : stmts) {
    auto for_stmt = std::dynamic_pointer_cast<const ForStmt>(stmt);
    auto while_stmt = std::dynamic_pointer_cast<const WhileStmt>(stmt);

    if ((for_stmt && !for_stmt->iter_args_.empty()) || (while_stmt && !while_stmt->iter_args_.empty())) {
      const auto& iter_args = for_stmt ? for_stmt->iter_args_ : while_stmt->iter_args_;
      const auto& body = for_stmt ? for_stmt->body_ : while_stmt->body_;

      outline_utils::VarDefUseCollector body_def_collector;
      body_def_collector.VisitStmt(body);
      auto all_defined = defined_so_far;
      all_defined.insert(body_def_collector.var_defs.begin(), body_def_collector.var_defs.end());

      auto body_stmts = FixupDanglingYieldValues(FlattenBody(body));
      body_stmts = FixDanglingLoopBodyYields(body_stmts, iter_args, all_defined);

      const auto& span = for_stmt ? for_stmt->span_ : while_stmt->span_;
      if (for_stmt) {
        result.push_back(RebuildForStmt(for_stmt, MakeBody(body_stmts, span)));
      } else {
        result.push_back(RebuildWhileStmt(while_stmt, MakeBody(body_stmts, span)));
      }
    } else if (auto if_stmt = std::dynamic_pointer_cast<const IfStmt>(stmt)) {
      auto new_then = FixupDanglingYieldValues(FlattenBody(if_stmt->then_body_));
      auto new_else = ProcessElseBranch(
          if_stmt, [](const std::vector<StmtPtr>& es) { return FixupDanglingYieldValues(es); });
      result.push_back(RebuildIfStmt(if_stmt, new_then, new_else));
    } else {
      result.push_back(stmt);
    }

    outline_utils::VarDefUseCollector stmt_defs;
    stmt_defs.VisitStmt(stmt);
    defined_so_far.insert(stmt_defs.var_defs.begin(), stmt_defs.var_defs.end());
  }

  return result;
}

std::vector<StmtPtr> FinalizeSplitCoreBody(const std::vector<StmtPtr>& stmts,
                                           const std::unordered_map<const Var*, StmtPtr>& original_def_map) {
  auto repaired = StripDeadIterArgs(stmts);
  repaired = FixupIterArgInitValues(repaired, original_def_map);
  repaired = FixupDanglingYieldValues(repaired);
  repaired = dce::EliminateDeadCode(repaired);
  repaired = StripDeadIterArgs(repaired);
  return dce::EliminateDeadCode(repaired);
}

}  // namespace loop_repair
}  // namespace ir
}  // namespace pypto
