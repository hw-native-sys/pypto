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

#include "pypto/ir/transforms/utils/dead_code_elimination.h"

#include <any>
#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/utils/loop_state_repair.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/transforms/utils/scope_outline_utils.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {
namespace dce {

const auto& FlattenBody = transform_utils::FlattenToStmts;

std::string GetStmtOpName(const StmtPtr& stmt) {
  CallPtr call;
  if (auto assign = std::dynamic_pointer_cast<const AssignStmt>(stmt)) {
    call = std::dynamic_pointer_cast<const Call>(assign->value_);
  } else if (auto eval = std::dynamic_pointer_cast<const EvalStmt>(stmt)) {
    call = std::dynamic_pointer_cast<const Call>(eval->expr_);
  }
  if (call && call->op_) {
    if (auto op = std::dynamic_pointer_cast<const Op>(call->op_)) {
      return op->name_;
    }
  }
  return "";
}

bool IsSideEffectOp(const StmtPtr& stmt) {
  static const std::unordered_set<std::string> side_effect_ops = {"tile.tpush_to_aiv",
                                                                  "tile.tpush_to_aic",
                                                                  "tile.tpop_from_aic",
                                                                  "tile.tpop_from_aiv",
                                                                  "tile.store",
                                                                  "tile.assemble",
                                                                  "system.tfree_to_aic",
                                                                  "system.tfree_to_aiv",
                                                                  "system.reserve_buffer",
                                                                  "system.import_peer_buffer",
                                                                  "system.aic_initialize_pipe",
                                                                  "system.aiv_initialize_pipe"};
  // A Submit launches an asynchronous task — intrinsically side-effecting
  // regardless of the kernel's body. Short-circuit so a Submit assignment is
  // never classified as a removal candidate.
  ExprPtr value;
  if (auto assign = std::dynamic_pointer_cast<const AssignStmt>(stmt)) value = assign->value_;
  if (auto eval = std::dynamic_pointer_cast<const EvalStmt>(stmt)) value = eval->expr_;
  if (value && As<Submit>(value)) return true;
  return side_effect_ops.count(GetStmtOpName(stmt)) > 0;
}

void CollectAllAssignStmts(const std::vector<StmtPtr>& stmts,
                           std::vector<std::shared_ptr<const AssignStmt>>& assigns) {
  for (const auto& stmt : stmts) {
    if (auto assign = std::dynamic_pointer_cast<const AssignStmt>(stmt)) {
      assigns.push_back(assign);
    }
    if (auto for_stmt = std::dynamic_pointer_cast<const ForStmt>(stmt)) {
      CollectAllAssignStmts(FlattenBody(for_stmt->body_), assigns);
    } else if (auto if_stmt = std::dynamic_pointer_cast<const IfStmt>(stmt)) {
      CollectAllAssignStmts(FlattenBody(if_stmt->then_body_), assigns);
      if (if_stmt->else_body_.has_value()) {
        CollectAllAssignStmts(FlattenBody(if_stmt->else_body_.value()), assigns);
      }
    } else if (auto while_stmt = std::dynamic_pointer_cast<const WhileStmt>(stmt)) {
      CollectAllAssignStmts(FlattenBody(while_stmt->body_), assigns);
    } else if (auto scope_stmt = std::dynamic_pointer_cast<const ScopeStmt>(stmt)) {
      CollectAllAssignStmts(FlattenBody(scope_stmt->body_), assigns);
    }
  }
}

namespace {

using loop_repair::MakeBody;
using RemovablePredicate = std::function<bool(const StmtPtr&)>;

/// Collect live-root variables.
///
/// A statement is a "live root" when it is NOT classified as a removal
/// candidate by `is_removable`. Its own Var references (expressions and
/// direct fields, not nested-body refs) are added to the live set; the
/// nested body, if any, is recursed into separately so its own candidate
/// assignments remain eligible for removal.
void FindLiveRootsRecursiveImpl(const std::vector<StmtPtr>& stmts, const RemovablePredicate& is_removable,
                                std::unordered_set<const Var*>& live) {
  auto collect_expr_refs = [&](const ExprPtr& expr) {
    if (!expr) return;
    outline_utils::VarDefUseCollector collector;
    collector.VisitExpr(expr);
    live.insert(collector.var_uses.begin(), collector.var_uses.end());
  };
  auto collect_iter_arg_refs = [&](const auto& loop_stmt) {
    for (const auto& iter_arg : loop_stmt->iter_args_) {
      collect_expr_refs(iter_arg->initValue_);
    }
  };

  for (const auto& stmt : stmts) {
    // Live-root: non-candidate leaf statements contribute their refs. For
    // AssignStmt/EvalStmt/ReturnStmt/YieldStmt we use the full subtree
    // collector because they have no nested bodies — it is equivalent to
    // walking their direct Expr fields.
    bool is_leaf = std::dynamic_pointer_cast<const AssignStmt>(stmt) ||
                   std::dynamic_pointer_cast<const EvalStmt>(stmt) ||
                   std::dynamic_pointer_cast<const ReturnStmt>(stmt) ||
                   std::dynamic_pointer_cast<const YieldStmt>(stmt);
    if (is_leaf && !is_removable(stmt)) {
      outline_utils::VarDefUseCollector collector;
      collector.VisitStmt(stmt);
      auto all_refs = collector.GetAllVarRefs();
      live.insert(all_refs.begin(), all_refs.end());
    }

    // Control-flow headers: add direct-field refs (bounds, conditions,
    // iter-arg initializers) but defer body traversal to the recursive
    // call so nested candidate assignments remain eligible for removal.
    if (auto for_stmt = std::dynamic_pointer_cast<const ForStmt>(stmt)) {
      collect_expr_refs(for_stmt->start_);
      collect_expr_refs(for_stmt->stop_);
      collect_expr_refs(for_stmt->step_);
      if (for_stmt->chunk_config_.has_value()) collect_expr_refs(for_stmt->chunk_config_->size);
      collect_iter_arg_refs(for_stmt);
      FindLiveRootsRecursiveImpl(FlattenBody(for_stmt->body_), is_removable, live);
    } else if (auto if_stmt = std::dynamic_pointer_cast<const IfStmt>(stmt)) {
      collect_expr_refs(if_stmt->condition_);
      FindLiveRootsRecursiveImpl(FlattenBody(if_stmt->then_body_), is_removable, live);
      if (if_stmt->else_body_.has_value()) {
        FindLiveRootsRecursiveImpl(FlattenBody(if_stmt->else_body_.value()), is_removable, live);
      }
    } else if (auto while_stmt = std::dynamic_pointer_cast<const WhileStmt>(stmt)) {
      collect_expr_refs(while_stmt->condition_);
      collect_iter_arg_refs(while_stmt);
      FindLiveRootsRecursiveImpl(FlattenBody(while_stmt->body_), is_removable, live);
    } else if (auto scope_stmt = std::dynamic_pointer_cast<const ScopeStmt>(stmt)) {
      // ScopeStmt's own attrs_ can reference Vars defined in the enclosing
      // scope:
      //   - manual_dep_edges (vector<VarPtr>)        — pl.at(..., deps=[tid])
      //   - task_id_var      (VarPtr)                — pl.at(...) as tid
      //   - arg_direction_overrides_vars (vector<VarPtr>) — direction overrides
      // These are real uses; without adding them to ``live`` the only
      // assignment that feeds the attr (e.g. ``dep_tid = stage1_tid`` after
      // an inline-helper return) is mis-identified as dead and removed,
      // leaving a dangling attr Var reference (issue #1456).
      //
      // SpmdScopeStmt additionally has a ``core_num_`` Expr field (e.g.
      // ``with pl.spmd(n):``) that can reference scalar Vars; we collect
      // those refs too so a scalar assigned only for the block count is
      // not deleted.
      auto add_var = [&](const VarPtr& v) {
        if (v) live.insert(v.get());
      };
      for (const auto& [k, v] : scope_stmt->attrs_) {
        if (k == kAttrManualDepEdges || k == kAttrArgDirOverrideVars) {
          if (const auto* edges = std::any_cast<std::vector<VarPtr>>(&v)) {
            for (const auto& e : *edges) add_var(e);
          }
        } else if (k == kAttrTaskIdVar) {
          if (const auto* var = std::any_cast<VarPtr>(&v)) add_var(*var);
        }
      }
      if (auto spmd = std::dynamic_pointer_cast<const SpmdScopeStmt>(scope_stmt)) {
        collect_expr_refs(spmd->core_num_);
      }
      FindLiveRootsRecursiveImpl(FlattenBody(scope_stmt->body_), is_removable, live);
    }
  }
}

/// True when a filtered vector is pointer-identical to the original's
/// flattened form — lets callers avoid cloning a control-flow node whose
/// body did not change.
bool SameFlattenedBody(const std::vector<StmtPtr>& filtered, const std::vector<StmtPtr>& original) {
  if (filtered.size() != original.size()) return false;
  for (size_t i = 0; i < filtered.size(); ++i) {
    if (filtered[i].get() != original[i].get()) return false;
  }
  return true;
}

std::vector<StmtPtr> FilterDeadCodeImpl(const std::vector<StmtPtr>& stmts,
                                        const RemovablePredicate& is_removable,
                                        const std::unordered_set<const Var*>& live) {
  std::vector<StmtPtr> result;
  for (const auto& stmt : stmts) {
    if (auto assign = std::dynamic_pointer_cast<const AssignStmt>(stmt)) {
      if (is_removable(stmt) && !live.count(assign->var_.get())) continue;
      result.push_back(stmt);
    } else if (auto for_stmt = std::dynamic_pointer_cast<const ForStmt>(stmt)) {
      auto original = FlattenBody(for_stmt->body_);
      auto filtered = FilterDeadCodeImpl(original, is_removable, live);
      if (SameFlattenedBody(filtered, original)) {
        result.push_back(stmt);
        continue;
      }
      auto new_for = MutableCopy(for_stmt);
      new_for->body_ = MakeBody(filtered, for_stmt->span_);
      result.push_back(new_for);
    } else if (auto if_stmt = std::dynamic_pointer_cast<const IfStmt>(stmt)) {
      auto then_orig = FlattenBody(if_stmt->then_body_);
      auto filtered_then = FilterDeadCodeImpl(then_orig, is_removable, live);
      bool then_unchanged = SameFlattenedBody(filtered_then, then_orig);
      bool else_unchanged = true;
      std::optional<StmtPtr> filtered_else;
      if (if_stmt->else_body_.has_value()) {
        auto else_orig = FlattenBody(if_stmt->else_body_.value());
        auto fe = FilterDeadCodeImpl(else_orig, is_removable, live);
        else_unchanged = SameFlattenedBody(fe, else_orig);
        filtered_else = MakeBody(fe, if_stmt->span_);
      }
      if (then_unchanged && else_unchanged) {
        result.push_back(stmt);
        continue;
      }
      auto new_if = MutableCopy(if_stmt);
      new_if->then_body_ = MakeBody(filtered_then, if_stmt->span_);
      new_if->else_body_ = filtered_else;
      result.push_back(new_if);
    } else if (auto while_stmt = std::dynamic_pointer_cast<const WhileStmt>(stmt)) {
      auto original = FlattenBody(while_stmt->body_);
      auto filtered = FilterDeadCodeImpl(original, is_removable, live);
      if (SameFlattenedBody(filtered, original)) {
        result.push_back(stmt);
        continue;
      }
      auto new_while = MutableCopy(while_stmt);
      new_while->body_ = MakeBody(filtered, while_stmt->span_);
      result.push_back(new_while);
    } else if (auto scope_stmt = std::dynamic_pointer_cast<const ScopeStmt>(stmt)) {
      auto original = FlattenBody(scope_stmt->body_);
      auto filtered = FilterDeadCodeImpl(original, is_removable, live);
      if (SameFlattenedBody(filtered, original)) {
        result.push_back(stmt);
        continue;
      }
      auto new_body = MakeBody(filtered, scope_stmt->span_);
      // ScopeStmt is abstract; dispatch on the concrete subtype so
      // MutableCopy instantiates the correct class.
      StmtPtr new_scope;
      if (auto incore = std::dynamic_pointer_cast<const InCoreScopeStmt>(stmt)) {
        auto copy = MutableCopy(incore);
        copy->body_ = new_body;
        new_scope = copy;
      } else if (auto auto_incore = std::dynamic_pointer_cast<const AutoInCoreScopeStmt>(stmt)) {
        auto copy = MutableCopy(auto_incore);
        copy->body_ = new_body;
        new_scope = copy;
      } else if (auto cluster = std::dynamic_pointer_cast<const ClusterScopeStmt>(stmt)) {
        auto copy = MutableCopy(cluster);
        copy->body_ = new_body;
        new_scope = copy;
      } else if (auto hierarchy = std::dynamic_pointer_cast<const HierarchyScopeStmt>(stmt)) {
        auto copy = MutableCopy(hierarchy);
        copy->body_ = new_body;
        new_scope = copy;
      } else if (auto spmd = std::dynamic_pointer_cast<const SpmdScopeStmt>(stmt)) {
        auto copy = MutableCopy(spmd);
        copy->body_ = new_body;
        new_scope = copy;
      } else if (auto runtime = std::dynamic_pointer_cast<const RuntimeScopeStmt>(stmt)) {
        auto copy = MutableCopy(runtime);
        copy->body_ = new_body;
        new_scope = copy;
      } else {
        INTERNAL_CHECK(false) << "Unhandled ScopeStmt subtype in DCE: " << scope_stmt->TypeName();
      }
      result.push_back(new_scope);
    } else {
      result.push_back(stmt);
    }
  }
  return result;
}

std::vector<StmtPtr> EliminateDeadCodeCore(const std::vector<StmtPtr>& stmts,
                                           const RemovablePredicate& is_removable,
                                           const std::unordered_set<const Var*>& extra_live = {}) {
  std::unordered_set<const Var*> live;
  FindLiveRootsRecursiveImpl(stmts, is_removable, live);
  // Seed externally-referenced live roots (e.g. a scalar whose only consumer
  // is the ``core_num`` attr of a function dispatched elsewhere). The
  // fixed-point loop below then keeps each one's transitive RHS uses alive too.
  live.insert(extra_live.begin(), extra_live.end());

  std::vector<std::shared_ptr<const AssignStmt>> all_assigns;
  CollectAllAssignStmts(stmts, all_assigns);

  // Cache each assignment's RHS uses once so the fixed-point loop does not
  // re-walk the expression every iteration — the outer loop can iterate
  // O(chain length) times on long dependency chains.
  std::vector<std::unordered_set<const Var*>> assign_uses;
  assign_uses.reserve(all_assigns.size());
  for (const auto& assign : all_assigns) {
    outline_utils::VarDefUseCollector collector;
    collector.VisitExpr(assign->value_);
    assign_uses.emplace_back(std::move(collector.var_uses));
  }

  bool changed = true;
  while (changed) {
    changed = false;
    for (size_t i = all_assigns.size(); i-- > 0;) {
      if (!live.count(all_assigns[i]->var_.get())) continue;
      for (const Var* ref : assign_uses[i]) {
        if (live.insert(ref).second) changed = true;
      }
    }
  }

  return FilterDeadCodeImpl(stmts, is_removable, live);
}

/// Predicate for the default `EliminateDeadCode`: any AssignStmt that is not
/// a known side-effect op is a removal candidate.
bool IsRemovableForDefaultDce(const StmtPtr& stmt) {
  return std::dynamic_pointer_cast<const AssignStmt>(stmt) != nullptr && !IsSideEffectOp(stmt);
}

/// Walk an expression tree and report whether any Call or Submit appears.
/// Both are call-like and side-effecting from the DCE perspective: a Call
/// may invoke a kernel with arbitrary side effects, and a Submit launches
/// an asynchronous task. Either form must be preserved conservatively
/// because the IR has no purity annotations yet.
class CallLikeFinder : public IRVisitor {
 public:
  bool found = false;

 private:
  using IRVisitor::VisitExpr_;
  void VisitExpr_(const CallPtr& op) override {
    found = true;
    // Keep walking — nested args may also contain Calls, but early-exit
    // would require throwing. The overhead is bounded by the expression
    // size which is small.
    IRVisitor::VisitExpr_(op);
  }
  void VisitExpr_(const SubmitPtr& op) override {
    found = true;
    IRVisitor::VisitExpr_(op);
  }
};

bool ExprContainsCallLike(const ExprPtr& expr) {
  if (!expr) return false;
  CallLikeFinder finder;
  finder.VisitExpr(expr);
  return finder.found;
}

/// Predicate for `EliminateDeadScalarAssignments`: an AssignStmt with a
/// scalar-typed LHS whose RHS contains no `Call` anywhere. Any expression
/// containing a Call is conservatively preserved because the IR has no
/// purity annotations yet.
bool IsRemovableScalarAssign(const StmtPtr& stmt) {
  auto assign = std::dynamic_pointer_cast<const AssignStmt>(stmt);
  if (!assign) return false;
  if (!As<ScalarType>(assign->var_->GetType())) return false;
  if (ExprContainsCallLike(assign->value_)) return false;
  return true;
}

// ============================================================================
// EliminateDeadIfReturnVars — drop IfStmt phi return_vars with no Var* user.
// ============================================================================

/// Filter a branch's trailing YieldStmt in place, keeping only `kept_indices`.
/// When `kept_indices` is empty, the trailing YieldStmt is removed entirely.
/// No-op when the branch does not end in a YieldStmt.
void FilterTrailingYieldSlots(std::vector<StmtPtr>& branch, const std::vector<size_t>& kept_indices) {
  if (branch.empty()) return;
  auto yield = std::dynamic_pointer_cast<const YieldStmt>(branch.back());
  if (!yield) return;
  if (kept_indices.empty()) {
    branch.pop_back();
    return;
  }
  std::vector<ExprPtr> new_values;
  new_values.reserve(kept_indices.size());
  for (size_t idx : kept_indices) {
    INTERNAL_CHECK_SPAN(idx < yield->value_.size(), yield->span_)
        << "Internal error: yield index " << idx << " out of range " << yield->value_.size();
    new_values.push_back(yield->value_[idx]);
  }
  branch.back() = std::make_shared<YieldStmt>(std::move(new_values), yield->span_);
}

/// MutableCopy + replace body, dispatching on concrete ScopeStmt subtype.
/// Mirrors the dispatch in FilterDeadCodeImpl below.
StmtPtr RebuildScopeWithBody(const std::shared_ptr<const ScopeStmt>& scope_stmt, const StmtPtr& new_body) {
  if (auto incore = std::dynamic_pointer_cast<const InCoreScopeStmt>(scope_stmt)) {
    auto copy = MutableCopy(incore);
    copy->body_ = new_body;
    return copy;
  }
  if (auto auto_incore = std::dynamic_pointer_cast<const AutoInCoreScopeStmt>(scope_stmt)) {
    auto copy = MutableCopy(auto_incore);
    copy->body_ = new_body;
    return copy;
  }
  if (auto cluster = std::dynamic_pointer_cast<const ClusterScopeStmt>(scope_stmt)) {
    auto copy = MutableCopy(cluster);
    copy->body_ = new_body;
    return copy;
  }
  if (auto hierarchy = std::dynamic_pointer_cast<const HierarchyScopeStmt>(scope_stmt)) {
    auto copy = MutableCopy(hierarchy);
    copy->body_ = new_body;
    return copy;
  }
  if (auto spmd = std::dynamic_pointer_cast<const SpmdScopeStmt>(scope_stmt)) {
    auto copy = MutableCopy(spmd);
    copy->body_ = new_body;
    return copy;
  }
  if (auto runtime = std::dynamic_pointer_cast<const RuntimeScopeStmt>(scope_stmt)) {
    auto copy = MutableCopy(runtime);
    copy->body_ = new_body;
    return copy;
  }
  INTERNAL_CHECK_SPAN(false, scope_stmt->span_) << "Unhandled ScopeStmt subtype: " << scope_stmt->TypeName();
  return scope_stmt;
}

/// Walk `stmts` and rewrite each IfStmt by dropping return_vars_[i] whose Var*
/// is not in `live_uses`, plus the matching slot from each branch's trailing
/// YieldStmt. Recurses into nested branches, scope bodies, and loop bodies.
/// Sets `*changed` when any IfStmt was rewritten (used by the fixed-point
/// loop in EliminateDeadIfReturnVars).
std::vector<StmtPtr> RewriteDeadIfPhisOnce(const std::vector<StmtPtr>& stmts,
                                           const std::unordered_set<const Var*>& live_uses, bool* changed) {
  std::vector<StmtPtr> result;
  result.reserve(stmts.size());
  for (const auto& stmt : stmts) {
    if (auto if_stmt = std::dynamic_pointer_cast<const IfStmt>(stmt)) {
      // Recurse into branches first so a dead inner phi observed in this same
      // pass also gets cleared. The outer fixed-point still iterates until
      // convergence; the inner recursion just shortens the iteration count.
      auto then_stmts = FlattenBody(if_stmt->then_body_);
      auto new_then = RewriteDeadIfPhisOnce(then_stmts, live_uses, changed);
      std::optional<std::vector<StmtPtr>> else_stmts;
      std::optional<std::vector<StmtPtr>> new_else;
      if (if_stmt->else_body_.has_value()) {
        else_stmts = FlattenBody(if_stmt->else_body_.value());
        new_else = RewriteDeadIfPhisOnce(*else_stmts, live_uses, changed);
      }

      // Identify which return_vars are dead at this level.
      std::vector<size_t> kept_indices;
      std::vector<VarPtr> new_return_vars;
      kept_indices.reserve(if_stmt->return_vars_.size());
      new_return_vars.reserve(if_stmt->return_vars_.size());
      for (size_t i = 0; i < if_stmt->return_vars_.size(); ++i) {
        if (live_uses.count(if_stmt->return_vars_[i].get())) {
          kept_indices.push_back(i);
          new_return_vars.push_back(if_stmt->return_vars_[i]);
        }
      }
      const bool dropped_any = kept_indices.size() < if_stmt->return_vars_.size();

      const bool then_unchanged = SameFlattenedBody(new_then, then_stmts);
      const bool else_unchanged = !new_else.has_value() || SameFlattenedBody(*new_else, *else_stmts);

      if (!dropped_any && then_unchanged && else_unchanged) {
        result.push_back(stmt);
        continue;
      }

      if (dropped_any) {
        FilterTrailingYieldSlots(new_then, kept_indices);
        if (new_else.has_value()) FilterTrailingYieldSlots(*new_else, kept_indices);
        *changed = true;
      }

      auto new_if = MutableCopy(if_stmt);
      new_if->then_body_ = MakeBody(new_then, if_stmt->span_);
      if (new_else.has_value()) {
        new_if->else_body_ = MakeBody(*new_else, if_stmt->span_);
      }
      if (dropped_any) {
        new_if->return_vars_ = std::move(new_return_vars);
      }
      result.push_back(new_if);
    } else if (auto for_stmt = std::dynamic_pointer_cast<const ForStmt>(stmt)) {
      auto original = FlattenBody(for_stmt->body_);
      auto rewritten = RewriteDeadIfPhisOnce(original, live_uses, changed);
      if (SameFlattenedBody(rewritten, original)) {
        result.push_back(stmt);
        continue;
      }
      auto new_for = MutableCopy(for_stmt);
      new_for->body_ = MakeBody(rewritten, for_stmt->span_);
      result.push_back(new_for);
    } else if (auto while_stmt = std::dynamic_pointer_cast<const WhileStmt>(stmt)) {
      auto original = FlattenBody(while_stmt->body_);
      auto rewritten = RewriteDeadIfPhisOnce(original, live_uses, changed);
      if (SameFlattenedBody(rewritten, original)) {
        result.push_back(stmt);
        continue;
      }
      auto new_while = MutableCopy(while_stmt);
      new_while->body_ = MakeBody(rewritten, while_stmt->span_);
      result.push_back(new_while);
    } else if (auto scope_stmt = std::dynamic_pointer_cast<const ScopeStmt>(stmt)) {
      auto original = FlattenBody(scope_stmt->body_);
      auto rewritten = RewriteDeadIfPhisOnce(original, live_uses, changed);
      if (SameFlattenedBody(rewritten, original)) {
        result.push_back(stmt);
        continue;
      }
      result.push_back(RebuildScopeWithBody(scope_stmt, MakeBody(rewritten, scope_stmt->span_)));
    } else {
      result.push_back(stmt);
    }
  }
  return result;
}

}  // namespace

std::vector<StmtPtr> EliminateDeadCode(const std::vector<StmtPtr>& stmts) {
  return EliminateDeadCodeCore(stmts, IsRemovableForDefaultDce);
}

std::vector<StmtPtr> EliminateDeadScalarAssignments(const std::vector<StmtPtr>& stmts) {
  return EliminateDeadCodeCore(stmts, IsRemovableScalarAssign);
}

std::vector<StmtPtr> EliminateDeadIfReturnVars(const std::vector<StmtPtr>& stmts) {
  std::vector<StmtPtr> cur = stmts;
  while (true) {
    // Collect every Var* that appears as a use anywhere in `cur`. The base
    // visitor walks ScopeStmt attrs via VisitScopeAttrs and Submit::deps_
    // through Call/Submit traversal, so all known reference channels are
    // covered without extra plumbing.
    outline_utils::VarDefUseCollector collector;
    for (const auto& s : cur) collector.VisitStmt(s);

    bool changed = false;
    auto next = RewriteDeadIfPhisOnce(cur, collector.var_uses, &changed);
    if (!changed) return cur;
    cur = std::move(next);
  }
}

std::vector<StmtPtr> EliminateDeadScalarAssignments(const std::vector<StmtPtr>& stmts,
                                                    const std::unordered_set<const Var*>& protected_vars) {
  return EliminateDeadCodeCore(stmts, IsRemovableScalarAssign, protected_vars);
}

}  // namespace dce
}  // namespace ir
}  // namespace pypto
