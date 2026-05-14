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
      // ScopeStmt (InCore/AutoInCore/Cluster/Hierarchy/Spmd) has no Expr
      // fields that reference scalar Vars — only recurse into the body so
      // nested assignments remain eligible for removal.
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
                                           const RemovablePredicate& is_removable) {
  std::unordered_set<const Var*> live;
  FindLiveRootsRecursiveImpl(stmts, is_removable, live);

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

/// Walk an expression tree and report whether any Call appears — a direct
/// top-level Call OR a Call nested inside an arithmetic expression. Since
/// the IR has no purity annotations, any expression containing a Call must
/// be preserved conservatively.
class CallFinder : public IRVisitor {
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
};

bool ExprContainsCall(const ExprPtr& expr) {
  if (!expr) return false;
  CallFinder finder;
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
  if (ExprContainsCall(assign->value_)) return false;
  return true;
}

}  // namespace

std::vector<StmtPtr> EliminateDeadCode(const std::vector<StmtPtr>& stmts) {
  return EliminateDeadCodeCore(stmts, IsRemovableForDefaultDce);
}

std::vector<StmtPtr> EliminateDeadScalarAssignments(const std::vector<StmtPtr>& stmts) {
  return EliminateDeadCodeCore(stmts, IsRemovableScalarAssign);
}

}  // namespace dce
}  // namespace ir
}  // namespace pypto
