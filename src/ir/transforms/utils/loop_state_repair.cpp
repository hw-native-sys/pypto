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
#include "pypto/ir/transforms/utils/mutable_copy.h"
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
  auto new_for = MutableCopy(f);
  new_for->body_ = new_body;
  return new_for;
}

StmtPtr RebuildForStmt(const std::shared_ptr<const ForStmt>& f, const std::vector<IterArgPtr>& iter_args,
                       const StmtPtr& new_body, const std::vector<VarPtr>& return_vars) {
  auto new_for = MutableCopy(f);
  new_for->iter_args_ = iter_args;
  new_for->body_ = new_body;
  new_for->return_vars_ = return_vars;
  return new_for;
}

StmtPtr RebuildWhileStmt(const std::shared_ptr<const WhileStmt>& w, const StmtPtr& new_body) {
  auto new_while = MutableCopy(w);
  new_while->body_ = new_body;
  return new_while;
}

StmtPtr RebuildWhileStmt(const std::shared_ptr<const WhileStmt>& w, const std::vector<IterArgPtr>& iter_args,
                         const StmtPtr& new_body, const std::vector<VarPtr>& return_vars) {
  auto new_while = MutableCopy(w);
  new_while->iter_args_ = iter_args;
  new_while->body_ = new_body;
  new_while->return_vars_ = return_vars;
  return new_while;
}

StmtPtr RebuildIfStmt(const std::shared_ptr<const IfStmt>& s, const std::vector<StmtPtr>& new_then,
                      const std::optional<std::vector<StmtPtr>>& new_else_stmts) {
  std::optional<StmtPtr> new_else;
  if (new_else_stmts.has_value()) {
    new_else = MakeBody(new_else_stmts.value(), s->span_);
  }
  auto new_if = MutableCopy(s);
  new_if->then_body_ = MakeBody(new_then, s->span_);
  new_if->else_body_ = new_else;
  return new_if;
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
      INTERNAL_CHECK_SPAN(idx < yield_stmt->value_.size(), yield_stmt->span_)
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
    if (auto for_stmt = std::dynamic_pointer_cast<const ForStmt>(stmt)) {
      BuildDefMap(FlattenBody(for_stmt->body_), def_map);
    } else if (auto if_stmt = std::dynamic_pointer_cast<const IfStmt>(stmt)) {
      BuildDefMap(FlattenBody(if_stmt->then_body_), def_map);
      if (if_stmt->else_body_.has_value()) {
        BuildDefMap(FlattenBody(if_stmt->else_body_.value()), def_map);
      }
    } else if (auto while_stmt = std::dynamic_pointer_cast<const WhileStmt>(stmt)) {
      BuildDefMap(FlattenBody(while_stmt->body_), def_map);
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

namespace {

/// Return the trailing `YieldStmt` of `stmts`, or nullptr if absent.
std::shared_ptr<const YieldStmt> TrailingYield(const std::vector<StmtPtr>& stmts) {
  if (stmts.empty()) return nullptr;
  return std::dynamic_pointer_cast<const YieldStmt>(stmts.back());
}

/// True when every Var referenced by `expr` is in `defined`.
bool ExprRefsAllDefined(const ExprPtr& expr, const std::unordered_set<const Var*>& defined) {
  outline_utils::VarDefUseCollector collector;
  collector.VisitExpr(expr);
  for (const Var* ref : collector.var_uses) {
    if (!defined.count(ref)) return false;
  }
  return true;
}

/// Insert every Var that would be visible at the *outer* scope of the given
/// body into `out`. This is scope-aware: only the names that escape each
/// statement (its result, not its internal locals) are added. Specifically:
///   - `AssignStmt::var_` (top-level binding)
///   - `ForStmt::return_vars_` / `WhileStmt::return_vars_` (loop results)
///   - `IfStmt::return_vars_` (phi-style merge of branch yields)
/// `SeqStmts` are flattened. Vars defined inside nested loop/if bodies
/// (loop_vars, iter_args, branch-local AssignStmts) stay scoped to those
/// inner constructs and are *not* added — they would not be reachable by a
/// reference written in the outer body.
void CollectAllDefsInStmts(const std::vector<StmtPtr>& stmts, std::unordered_set<const Var*>& out) {
  for (const auto& s : stmts) {
    if (auto seq = std::dynamic_pointer_cast<const SeqStmts>(s)) {
      CollectAllDefsInStmts(seq->stmts_, out);
    } else if (auto assign = std::dynamic_pointer_cast<const AssignStmt>(s)) {
      out.insert(assign->var_.get());
    } else if (auto for_stmt = std::dynamic_pointer_cast<const ForStmt>(s)) {
      for (const auto& rv : for_stmt->return_vars_) out.insert(rv.get());
    } else if (auto while_stmt = std::dynamic_pointer_cast<const WhileStmt>(s)) {
      for (const auto& rv : while_stmt->return_vars_) out.insert(rv.get());
    } else if (auto if_stmt = std::dynamic_pointer_cast<const IfStmt>(s)) {
      for (const auto& rv : if_stmt->return_vars_) out.insert(rv.get());
    }
  }
}

/// Return the indices of the trailing yield's values (against the IfStmt's
/// return_vars_) whose expressions reference only vars in `branch_defs`. If
/// the branch has no trailing yield, every index is kept (the missing yield
/// can't make the value ill-defined here).
void NarrowKeptIndices(const std::shared_ptr<const YieldStmt>& yield,
                       const std::unordered_set<const Var*>& branch_defs, size_t num_return_vars,
                       std::vector<bool>& keep) {
  if (!yield) return;
  for (size_t i = 0; i < num_return_vars && i < yield->value_.size(); ++i) {
    if (keep[i] && !ExprRefsAllDefined(yield->value_[i], branch_defs)) {
      keep[i] = false;
    }
  }
}

}  // namespace

std::vector<StmtPtr> StripDanglingIfReturnVars(const std::vector<StmtPtr>& stmts,
                                               const std::unordered_set<const Var*>& extra_defined) {
  std::unordered_set<const Var*> outer_defined = extra_defined;
  std::vector<StmtPtr> result;
  result.reserve(stmts.size());

  for (const auto& stmt : stmts) {
    StmtPtr new_stmt = stmt;
    if (auto if_stmt = std::dynamic_pointer_cast<const IfStmt>(stmt)) {
      // Recurse into branches first so nested IfStmts are repaired bottom-up.
      // `outer_defined` carries vars defined by earlier siblings — pass it (not
      // just the original `extra_defined`) so nested branches see prior defs.
      auto new_then_stmts = StripDanglingIfReturnVars(FlattenBody(if_stmt->then_body_), outer_defined);
      std::optional<std::vector<StmtPtr>> new_else_stmts;
      if (if_stmt->else_body_.has_value()) {
        new_else_stmts = StripDanglingIfReturnVars(FlattenBody(*if_stmt->else_body_), outer_defined);
      }

      const size_t num_rv = if_stmt->return_vars_.size();
      std::vector<size_t> kept_indices;
      if (num_rv > 0) {
        // A branch's yield value is well-defined if it only references vars
        // visible before the IfStmt or defined within that branch (or in
        // `extra_defined`, e.g. vars that will be remapped by a later
        // DeepClone-with-tpop-remap step in ExpandMixedKernel).
        std::unordered_set<const Var*> then_defs = outer_defined;
        CollectAllDefsInStmts(new_then_stmts, then_defs);
        std::unordered_set<const Var*> else_defs = outer_defined;
        if (new_else_stmts.has_value()) {
          CollectAllDefsInStmts(*new_else_stmts, else_defs);
        }

        std::vector<bool> keep(num_rv, true);
        NarrowKeptIndices(TrailingYield(new_then_stmts), then_defs, num_rv, keep);
        if (new_else_stmts.has_value()) {
          NarrowKeptIndices(TrailingYield(*new_else_stmts), else_defs, num_rv, keep);
        }
        for (size_t i = 0; i < num_rv; ++i) {
          if (keep[i]) kept_indices.push_back(i);
        }
      }

      if (num_rv == 0 || kept_indices.size() == num_rv) {
        new_stmt = RebuildIfStmt(if_stmt, new_then_stmts, new_else_stmts);
      } else {
        // Drop the orphan indices from return_vars_ and from each branch's
        // trailing yield. FilterYieldStmt rewrites the tail YieldStmt
        // (returning nullptr when kept_indices is empty, in which case we
        // drop the trailing yield outright).
        std::vector<VarPtr> new_return_vars;
        new_return_vars.reserve(kept_indices.size());
        for (size_t idx : kept_indices) new_return_vars.push_back(if_stmt->return_vars_[idx]);

        auto filter_branch = [&](const std::vector<StmtPtr>& branch_stmts) {
          std::vector<StmtPtr> filtered = branch_stmts;
          if (TrailingYield(filtered)) {
            auto new_last = FilterYieldStmt(filtered.back(), kept_indices);
            if (new_last) {
              filtered.back() = new_last;
            } else {
              filtered.pop_back();
            }
          }
          return filtered;
        };

        auto filtered_then = filter_branch(new_then_stmts);
        std::optional<std::vector<StmtPtr>> filtered_else;
        if (new_else_stmts.has_value()) {
          filtered_else = filter_branch(*new_else_stmts);
        }

        auto new_if = MutableCopy(if_stmt);
        new_if->then_body_ = MakeBody(filtered_then, if_stmt->span_);
        new_if->else_body_ = filtered_else.has_value()
                                 ? std::optional<StmtPtr>(MakeBody(*filtered_else, if_stmt->span_))
                                 : std::nullopt;
        new_if->return_vars_ = new_return_vars;
        new_stmt = new_if;
      }
    } else if (auto for_stmt = std::dynamic_pointer_cast<const ForStmt>(stmt)) {
      // Seed the body's visible-defs set from `outer_defined` (vars defined
      // before the loop) plus the loop's own scope-defined vars (loop_var,
      // iter_args). CollectAllDefsInStmts on body statements wouldn't pick
      // these up on its own, so nested IfStmt yield checks need them passed
      // in via the extra_defined channel.
      auto body_extra = outer_defined;
      body_extra.insert(for_stmt->loop_var_.get());
      for (const auto& ia : for_stmt->iter_args_) body_extra.insert(ia.get());
      auto new_body = StripDanglingIfReturnVars(FlattenBody(for_stmt->body_), body_extra);
      new_stmt = RebuildForStmt(for_stmt, MakeBody(new_body, for_stmt->span_));
    } else if (auto while_stmt = std::dynamic_pointer_cast<const WhileStmt>(stmt)) {
      auto body_extra = outer_defined;
      for (const auto& ia : while_stmt->iter_args_) body_extra.insert(ia.get());
      auto new_body = StripDanglingIfReturnVars(FlattenBody(while_stmt->body_), body_extra);
      new_stmt = RebuildWhileStmt(while_stmt, MakeBody(new_body, while_stmt->span_));
    }

    result.push_back(new_stmt);
    outline_utils::VarDefUseCollector c;
    c.VisitStmt(new_stmt);
    outer_defined.insert(c.var_defs.begin(), c.var_defs.end());
  }

  return result;
}

std::vector<StmtPtr> FinalizeSplitCoreBody(const std::vector<StmtPtr>& stmts,
                                           const std::unordered_map<const Var*, StmtPtr>& original_def_map,
                                           const std::unordered_set<const Var*>& extra_defined) {
  // FixupDanglingYieldValues runs first so that dangling yield refs inside
  // IfStmts nested in a loop are rewritten to the loop's iter_arg when one is
  // available (issue #534 pattern). StripDanglingIfReturnVars then mops up the
  // genuinely orphan IfStmt return_vars that remain — those whose enclosing
  // loop has no matching iter_arg fallback (issue #1501 pattern: outer loop's
  // iter_args have themselves already been stripped by an earlier pass).
  auto repaired = StripDeadIterArgs(stmts);
  repaired = FixupIterArgInitValues(repaired, original_def_map);
  repaired = FixupDanglingYieldValues(repaired);
  repaired = StripDanglingIfReturnVars(repaired, extra_defined);
  repaired = dce::EliminateDeadCode(repaired);
  repaired = StripDeadIterArgs(repaired);
  return dce::EliminateDeadCode(repaired);
}

}  // namespace loop_repair
}  // namespace ir
}  // namespace pypto
