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

/*
 * The arithmetic simplification module takes reference from:
 * - Apache TVM (https://github.com/apache/tvm), Apache License 2.0
 * - MLC-Python (https://github.com/mlc-ai/mlc-python), Apache License 2.0
 */

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/arith/analyzer.h"
#include "pypto/ir/arith/ir_mutator_with_analyzer.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memref.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/structural_comparison.h"
#include "pypto/ir/transforms/utils/dead_code_elimination.h"
#include "pypto/ir/transforms/utils/deep_clone_utils.h"
#include "pypto/ir/transforms/utils/loop_state_repair.h"
#include "pypto/ir/transforms/utils/memref_utils.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/transforms/utils/simplify.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

namespace {

/// Collects Var pointers assigned more than once anywhere in the function.
/// Such Vars are unsafe to bind to a single value: a later assignment would
/// invalidate the value recorded at the first. A Var assigned exactly once is
/// always safe to bind — SimplifyMutator scopes every binding to the
/// loop / if-branch / while / spmd region the assignment lives in (see
/// UnbindScalarsSince), so a Var defined inside a branch or loop body never
/// leaks its value past that region even if the branch did not execute. In
/// SSA form every Var is single-assigned, so nothing is collected.
class MultiAssignCollector : public IRVisitor {
 public:
  explicit MultiAssignCollector(const std::vector<VarPtr>& params) {
    // Parameters already have a definition on entry. A body assignment to one
    // is a pre-SSA rebind, including when it occurs only in one branch.
    // Match AssignTypeSymmetry's counting so retyping cannot replace the
    // parameter identity used by a sibling branch.
    for (const auto& param : params) seen_.insert(param.get());
  }

  std::unordered_set<const Var*> multi_assigned;
  /// Vars with more than one definition site of any kind (parameter, AssignStmt,
  /// For/If/While binders). FoldProvableComparison never trusts such a loop
  /// variable to stay within its header range.
  std::unordered_set<const Var*> redefined;

  void VisitStmt_(const AssignStmtPtr& op) override {
    if (!seen_.insert(op->var_.get()).second) {
      multi_assigned.insert(op->var_.get());
    }
    Define(op->var_.get());
    IRVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const ForStmtPtr& op) override {
    Define(op->loop_var_.get());
    for (const auto& v : op->iter_args_) Define(v.get());
    for (const auto& v : op->return_vars_) Define(v.get());
    IRVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const IfStmtPtr& op) override {
    for (const auto& v : op->return_vars_) Define(v.get());
    IRVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const WhileStmtPtr& op) override {
    for (const auto& v : op->iter_args_) Define(v.get());
    for (const auto& v : op->return_vars_) Define(v.get());
    IRVisitor::VisitStmt_(op);
  }

  /// Also called for each function parameter before the body is visited.
  void Define(const Var* var) {
    if (!defined_.insert(var).second) redefined.insert(var);
  }

 private:
  std::unordered_set<const Var*> seen_;
  std::unordered_set<const Var*> defined_;
};

/// Answers, for each control-flow statement Fold A / Fold B may collapse,
/// whether any use of its return_vars would outlive the substitution the fold
/// records in place of a definition.
///
/// Both folds lift a kept branch / unrolled body by writing
/// `return_var -> yielded_value` into `var_remap_` instead of emitting an
/// AssignStmt, and ForStmt, WhileStmt and IfStmt each restore `var_remap_` to a
/// pre-body baseline on the way out so a body-internal remap cannot rewrite
/// siblings or post-loop code. A use that outlives that restore therefore keeps
/// the original Var -- which the fold left with no defining statement, i.e. a
/// dangling reference UseAfterDef reports.
///
/// The analysis is one pre-order walk over the function body that numbers the
/// restoring scopes. A scope owns the contiguous id range `[id, end)` of its
/// own subtree, so "every use of v sits inside scope S" reduces to two integer
/// comparisons. A monotonic `tick` orders uses against the fold site, so a use
/// *preceding* it inside the same scope (reachable pre-SSA via a loop-carried
/// read) counts as escaping too. One walk plus O(1) lookups per fold keeps the
/// pass within its O(N log N) budget.
///
/// Nothing escapes in SSA form: a value defined inside a region is never
/// referenced outside it, and every use is dominated by its definition. The
/// pipeline runs Simplify only after ConvertToSSA (positions 5 and 46), so the
/// materializing path this index unlocks is reachable only from callers that
/// run Simplify directly on pre-SSA IR.
class ReturnVarEscapeIndex : public IRVisitor {
 public:
  /// Index a function body. Called once per TransformSimplify run, before the
  /// mutator walks the same body, which keeps the pass linear in the IR size.
  ///
  /// Entries are keyed by raw Var / Stmt pointers, which stay valid because the
  /// Function owns the indexed body for the whole run. Nodes the mutator mints
  /// afterwards -- notably the bodies Fold B DeepClones -- are absent from the
  /// index and so are never confused with an indexed node: a fresh allocation
  /// cannot land on an address that is still live.
  void Index(const StmtPtr& body) {
    if (!body) return;
    EnterScope();
    VisitStmt(body);
    LeaveScope();
  }

  /// True when collapsing @p folded would strand a use of @p rv: a use outside
  /// the innermost var_remap_-restoring scope containing @p folded, or one that
  /// precedes @p folded within it.
  ///
  /// An unindexed statement answers false, keeping the substitution. That
  /// covers folds nested inside a Fold B clone, whose Var identities are minted
  /// after indexing; a clone's own Vars are unreachable from outside it, so the
  /// only case this leaves unhandled is a restore-scope *within* the clone
  /// standing between such a fold and a later use of its return var -- possible
  /// only pre-SSA, and no worse than the behaviour before this index existed.
  /// Re-indexing each clone would close it. It is left open because the gap is
  /// pre-SSA-only and the pipeline runs Simplify only after ConvertToSSA, so
  /// nothing but a direct pre-SSA caller can reach it.
  bool Escapes(const Stmt* folded, const Var* rv) const {
    auto site = sites_.find(folded);
    if (site == sites_.end()) return false;  // never indexed -> keep substituting
    auto use = uses_.find(rv);
    if (use == uses_.end()) return false;  // unused -> nothing to strand
    const size_t scope = site->second.scope;
    return use->second.min_scope < scope || use->second.max_scope >= scopes_[scope].end ||
           use->second.min_tick < site->second.tick;
  }

  void VisitVarLike_(const VarPtr& op) override {
    auto& use = uses_[op.get()];
    use.min_scope = std::min(use.min_scope, stack_.back());
    use.max_scope = std::max(use.max_scope, stack_.back());
    use.min_tick = std::min(use.min_tick, tick_);
    ++tick_;
    IRVisitor::VisitVarLike_(op);
  }

  // The three statements whose visitors restore var_remap_ around their bodies.
  // Their traversal order mirrors the IRVisitor base; it is spelled out here
  // only so the body is walked inside its own scope.

  void VisitStmt_(const ForStmtPtr& op) override {
    RecordSite(op.get());
    VisitExpr(op->loop_var_);
    VisitExpr(op->start_);
    VisitExpr(op->stop_);
    VisitExpr(op->step_);
    for (const auto& iter_arg : op->iter_args_) VisitExpr(iter_arg);
    EnterScope();
    VisitStmt(op->body_);
    LeaveScope();
    for (const auto& return_var : op->return_vars_) VisitExpr(return_var);
  }

  void VisitStmt_(const IfStmtPtr& op) override {
    RecordSite(op.get());
    VisitExpr(op->condition_);
    // Each branch is its own scope: the IfStmt visitor rebases var_remap_
    // between them, and keeps a branch's additions only when Fold A fires.
    // Treating a kept branch as restoring is conservative -- it materializes
    // where substitution would also have worked.
    EnterScope();
    VisitStmt(op->then_body_);
    LeaveScope();
    if (op->else_body_.has_value()) {
      EnterScope();
      VisitStmt(*op->else_body_);
      LeaveScope();
    }
    for (const auto& return_var : op->return_vars_) VisitExpr(return_var);
  }

  void VisitStmt_(const WhileStmtPtr& op) override {
    // No RecordSite: Simplify never folds a WhileStmt away, it only restores
    // var_remap_ around the body -- which is what makes this a scope.
    VisitExpr(op->condition_);
    for (const auto& iter_arg : op->iter_args_) VisitExpr(iter_arg);
    EnterScope();
    VisitStmt(op->body_);
    LeaveScope();
    for (const auto& return_var : op->return_vars_) VisitExpr(return_var);
  }

 private:
  /// A var_remap_-restoring region. Pre-order ids make a scope's subtree the
  /// half-open range [own id, end).
  struct Scope {
    size_t end = 0;
  };

  /// Where a foldable control-flow statement sits: its enclosing scope, and the
  /// tick it was reached at.
  struct Site {
    size_t scope = 0;
    size_t tick = 0;
  };

  /// The scope range and earliest tick over all uses of one Var.
  struct Use {
    size_t min_scope = std::numeric_limits<size_t>::max();
    size_t max_scope = 0;
    size_t min_tick = std::numeric_limits<size_t>::max();
  };

  void EnterScope() {
    stack_.push_back(scopes_.size());
    scopes_.push_back({});
  }

  void LeaveScope() {
    scopes_[stack_.back()].end = scopes_.size();
    stack_.pop_back();
  }

  void RecordSite(const Stmt* op) { sites_[op] = Site{stack_.back(), tick_}; }

  std::vector<Scope> scopes_;
  std::vector<size_t> stack_;
  std::unordered_map<const Stmt*, Site> sites_;
  std::unordered_map<const Var*, Use> uses_;
  size_t tick_ = 0;
};

/// Strip the trailing YieldStmt from @p body and return both the stripped
/// body and the yielded values that the caller should bind into its
/// var_remap_ as `return_vars[i] → yielded_values[i]`. Used by control-flow
/// folds (Fold A for IfStmt, Fold B for ForStmt) that lift a kept branch /
/// unrolled body into the parent scope.
///
/// Substituting return_vars via var_remap_ (rather than emitting a literal
/// `AssignStmt(return_var, yielded)`) avoids creating alias assignments that
/// the orchestration codegen's role-aware name disambiguation cannot lower
/// cleanly — e.g. `out__rv_v2 = out__co_l0_rv_v3` printing as
/// `auto out = out;` because both vars share the role-derived base name
/// `out`. The substituted form lets subsequent SimplifyMutator visits
/// (including the function's ReturnStmt) read the yielded value directly.
///
/// Follows the shared trailing-yield spine, preserving runtime/split wrappers.
///
/// Pre: @p body is a well-formed control-flow body with `return_vars`
/// non-empty — its (possibly nested) tail is a YieldStmt with
/// `value_.size() == return_vars.size()`.
struct StrippedYield {
  StmtPtr body;                         ///< @p body with the trailing YieldStmt removed.
  std::vector<ExprPtr> yielded_values;  ///< Values from the removed YieldStmt, one per return_var.
};

StrippedYield StripTrailingYield(const StmtPtr& body, size_t return_var_count) {
  auto yield_stmt = transform_utils::GetLastYieldStmt(body);
  INTERNAL_CHECK_SPAN(yield_stmt, body->span_)
      << "Internal error: control-flow body tail must be YieldStmt when "
         "return_vars is non-empty";
  INTERNAL_CHECK_SPAN(yield_stmt->value_.size() == return_var_count, yield_stmt->span_)
      << "Internal error: yielded value count " << yield_stmt->value_.size()
      << " does not match return_vars count " << return_var_count;
  auto stripped = transform_utils::RewriteLastYieldStmt(body, [](const YieldStmtPtr&) { return nullptr; });
  return {std::move(stripped), yield_stmt->value_};
}

/// Folding substitutes yielded expressions at their users. Calls must instead
/// remain at their evaluation site (including inside a MANUAL scope), so retain
/// the control flow until FlattenCallExpr has materialized their results.
bool HasEffectfulYield(const StmtPtr& body) {
  auto yield = body ? transform_utils::GetLastYieldStmt(body) : nullptr;
  return yield &&
         std::any_of(yield->value_.begin(), yield->value_.end(), transform_utils::ExprContainsCallLike);
}

/// Rebind storage expressions without folding values or revisiting a mapped
/// variable's type/initValue. A rebuilt IterArg is still a runtime value.
class StorageVarRemapper : public IRMutator {
 public:
  explicit StorageVarRemapper(const std::unordered_map<const Expr*, ExprPtr>& remap) : remap_(remap) {}

 protected:
  ExprPtr VisitExpr_(const VarPtr& op) override {
    auto it = remap_.find(op.get());
    return it != remap_.end() ? it->second : op;
  }

  ExprPtr VisitExpr_(const IterArgPtr& op) override {
    return VisitExpr_(std::static_pointer_cast<const Var>(op));
  }

 private:
  const std::unordered_map<const Expr*, ExprPtr>& remap_;
};

/// Preserve assignment type symmetry when a rewrite changes a previously
/// matching RHS type. Keep pre-existing mismatches visible to the verifier.
VarPtr MatchRewrittenValueType(const VarPtr& original_var, const TypePtr& original_value_type,
                               const ExprPtr& new_value, const VarPtr& new_var) {
  const auto& value_type = new_value->GetType();
  if (structural_equal(new_var->GetType(), value_type)) return new_var;
  if (!structural_equal(original_var->GetType(), original_value_type)) return new_var;
  return std::make_shared<Var>(new_var->name_hint_, value_type, new_var->span_);
}

/// Retained dead bodies still need enclosing binder substitutions and dependent
/// type rebuilds. Use the plain mutator so no arithmetic is evaluated, and treat
/// enclosing replacements as finalized leaves (especially rebuilt IterArgs).
class UnreachableBodyRemapper : public IRMutator {
 public:
  UnreachableBodyRemapper(const std::unordered_map<const Expr*, ExprPtr>& remap,
                          const std::unordered_set<const Var*>& multi_assigned)
      : enclosing_remap_(remap), multi_assigned_(multi_assigned) {}

 protected:
  ExprPtr VisitExpr_(const VarPtr& op) override {
    auto local = var_remap_.find(op.get());
    if (local != var_remap_.end()) return local->second;
    auto it = enclosing_remap_.find(op.get());
    if (it != enclosing_remap_.end()) return it->second;
    auto identity = definition_identities_.find(op.get());
    return identity != definition_identities_.end() ? identity->second : IRMutator::VisitExpr_(op);
  }

  ExprPtr VisitExpr_(const IterArgPtr& op) override {
    auto local = var_remap_.find(op.get());
    if (local != var_remap_.end()) return local->second;
    auto it = enclosing_remap_.find(op.get());
    if (it != enclosing_remap_.end()) return it->second;
    auto identity = definition_identities_.find(op.get());
    return identity != definition_identities_.end() ? identity->second : IRMutator::VisitExpr_(op);
  }

  ExprPtr VisitExpr_(const WindowBufferPtr& op) override {
    auto it = enclosing_remap_.find(op.get());
    return it != enclosing_remap_.end() ? it->second : IRMutator::VisitExpr_(op);
  }

  StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
    auto new_value = VisitExpr(op->value_);
    auto new_var = RewriteDefinition(op->var_);
    // An enclosing identity-view fold may change the RHS type without changing
    // any expression embedded in the LHS type. Propagate that type to the local
    // binder and its later uses, just as the evaluating Simplify visitor does.
    // Pre-SSA variables with other assignments must keep their shared type.
    if (multi_assigned_.count(op->var_.get()) == 0) {
      auto retyped = MatchRewrittenValueType(op->var_, op->value_->GetType(), new_value, new_var);
      new_var = std::move(retyped);
    }
    var_remap_[op->var_.get()] = new_var;
    // Rebuilt binder identities are canonical, not branch-local value facts.
    // Keep them for legal pre-SSA uses after a result-free control-flow scope.
    if (new_var != op->var_) definition_identities_[op->var_.get()] = new_var;
    if (new_value == op->value_ && new_var == op->var_) return op;
    auto result = MutableCopy(op);
    result->value_ = std::move(new_value);
    result->var_ = std::move(new_var);
    return result;
  }

  StmtPtr VisitStmt_(const IfStmtPtr& op) override {
    auto condition = VisitExpr(op->condition_);
    auto then_body = VisitScopedBody(op->then_body_);
    auto else_body = op->else_body_;
    if (else_body) else_body = VisitScopedBody(*else_body);
    auto return_vars = RewriteDefinitions(op->return_vars_);
    if (condition == op->condition_ && then_body == op->then_body_ && else_body == op->else_body_ &&
        return_vars == op->return_vars_) {
      return op;
    }
    auto result = MutableCopy(op);
    result->condition_ = std::move(condition);
    result->then_body_ = std::move(then_body);
    result->else_body_ = std::move(else_body);
    result->return_vars_ = std::move(return_vars);
    return result;
  }

  StmtPtr VisitStmt_(const ForStmtPtr& op) override {
    auto start = VisitExpr(op->start_);
    auto stop = VisitExpr(op->stop_);
    auto step = VisitExpr(op->step_);
    auto mark = definition_log_.size();
    auto loop_var = RewriteDefinition(op->loop_var_);
    auto iter_args = RewriteDefinitions(op->iter_args_);
    auto body_mark = definition_log_.size();
    auto body = VisitStmt(op->body_);
    LeaveLoopScope(mark, body_mark, op->return_vars_.empty());
    auto return_vars = RewriteDefinitions(op->return_vars_);
    if (start == op->start_ && stop == op->stop_ && step == op->step_ && loop_var == op->loop_var_ &&
        iter_args == op->iter_args_ && body == op->body_ && return_vars == op->return_vars_) {
      return op;
    }
    auto result = MutableCopy(op);
    result->start_ = std::move(start);
    result->stop_ = std::move(stop);
    result->step_ = std::move(step);
    result->loop_var_ = std::move(loop_var);
    result->iter_args_ = std::move(iter_args);
    result->body_ = std::move(body);
    result->return_vars_ = std::move(return_vars);
    return result;
  }

  StmtPtr VisitStmt_(const WhileStmtPtr& op) override {
    auto mark = definition_log_.size();
    auto iter_args = RewriteDefinitions(op->iter_args_);
    auto condition = VisitExpr(op->condition_);
    auto body_mark = definition_log_.size();
    auto body = VisitStmt(op->body_);
    LeaveLoopScope(mark, body_mark, op->return_vars_.empty());
    auto return_vars = RewriteDefinitions(op->return_vars_);
    if (iter_args == op->iter_args_ && condition == op->condition_ && body == op->body_ &&
        return_vars == op->return_vars_) {
      return op;
    }
    auto result = MutableCopy(op);
    result->iter_args_ = std::move(iter_args);
    result->condition_ = std::move(condition);
    result->body_ = std::move(body);
    result->return_vars_ = std::move(return_vars);
    return result;
  }

 private:
  template <typename Node>
  std::shared_ptr<const Node> RewriteDefinition(const std::shared_ptr<const Node>& var) {
    auto previous = var_remap_.find(var.get());
    ExprPtr old_mapping = previous != var_remap_.end() ? previous->second : nullptr;
    // Definitions shadow enclosing value aliases, but their dependent types
    // still follow the current scope's substitutions.
    auto identity = definition_identities_.find(var.get());
    ExprPtr definition;
    if (previous == var_remap_.end() && identity != definition_identities_.end()) {
      definition = identity->second;
    } else {
      definition = IRMutator::VisitExpr_(var);
    }
    auto rewritten = std::dynamic_pointer_cast<const Node>(definition);
    INTERNAL_CHECK_SPAN(rewritten, var->span_) << "Simplify: definition changed node kind";
    definition_log_.emplace_back(var.get(), std::move(old_mapping));
    var_remap_[var.get()] = rewritten;
    // Result binders can escape result-free parent scopes too. IterArgs keep
    // their scope-specific seeds and are deliberately excluded from this cache.
    if constexpr (std::is_same_v<Node, Var>) {
      if (rewritten != var) definition_identities_[var.get()] = rewritten;
    }
    return rewritten;
  }

  template <typename Ptr>
  std::vector<Ptr> RewriteDefinitions(const std::vector<Ptr>& vars) {
    std::vector<Ptr> result;
    result.reserve(vars.size());
    for (const auto& var : vars) result.push_back(RewriteDefinition(var));
    return result;
  }

  void RestoreDefinition(const Expr* key, const ExprPtr& previous) {
    if (previous) {
      var_remap_[key] = previous;
    } else {
      var_remap_.erase(key);
    }
  }

  void LeaveLoopScope(size_t mark, size_t body_mark, bool leaks_body) {
    if (!leaks_body) {
      RestoreDefinitions(mark);
      return;
    }
    // Match pre-SSA leak mode: body definitions escape a loop with no result
    // variables, but the induction variable and carries remain loop-local.
    for (size_t i = body_mark; i > mark; --i) {
      const auto& [key, previous] = definition_log_[i - 1];
      RestoreDefinition(key, previous);
    }
  }

  StmtPtr VisitScopedBody(const StmtPtr& body) {
    auto mark = definition_log_.size();
    auto result = VisitStmt(body);
    RestoreDefinitions(mark);
    return result;
  }

  void RestoreDefinitions(size_t mark) {
    // Undo only definitions introduced here. Each entry is pushed and popped
    // once, avoiding a full remap-table copy for every nested branch or loop.
    while (definition_log_.size() > mark) {
      const auto& [key, previous] = definition_log_.back();
      RestoreDefinition(key, previous);
      definition_log_.pop_back();
    }
  }

  // Scoped shadows take precedence over this identity cache, and enclosing
  // value aliases still win in sibling scopes where the definition is absent.
  std::unordered_map<const Expr*, ExprPtr> definition_identities_;
  std::vector<std::pair<const Expr*, ExprPtr>> definition_log_;
  const std::unordered_map<const Expr*, ExprPtr>& enclosing_remap_;
  const std::unordered_set<const Var*>& multi_assigned_;
};

class SimplifyMutator : public arith::IRMutatorWithAnalyzer {
 public:
  SimplifyMutator(arith::Analyzer* analyzer, std::unordered_set<const Var*> multi_assigned,
                  std::unordered_set<const Var*> redefined, ReturnVarEscapeIndex* escapes,
                  bool constants_only)
      : IRMutatorWithAnalyzer(analyzer),
        multi_assigned_(std::move(multi_assigned)),
        redefined_(std::move(redefined)),
        escapes_(escapes),
        constants_only_(constants_only) {}

  /// Reconcile all domain links once, after their sizes have been resolved.
  /// A single function-wide non-evaluating sweep avoids repeatedly walking
  /// nested domains and never simplifies arithmetic in retained dead bodies.
  StmtPtr RemapFinalizedWindows(const StmtPtr& body) {
    if (finalized_windows_.empty()) return body;
    UnreachableBodyRemapper remapper(finalized_windows_, multi_assigned_);
    return remapper.VisitStmt(body);
  }

  /// Fold scalar constant bindings at every Var leaf. Reached via the base
  /// IRMutator's qualified ExprFunctor::VisitExpr dispatch when walking Call
  /// args — `analyzer_->Simplify` at the SimplifyExpr level does not recurse
  /// into non-arithmetic Call nodes, so folding must happen at the leaf.
  ExprPtr VisitExpr_(const VarPtr& op) override {
    auto it = var_remap_.find(op.get());
    ExprPtr remapped = (it != var_remap_.end()) ? it->second : op;
    return SimplifyValue(remapped);
  }

  ExprPtr VisitExpr_(const IterArgPtr& op) override {
    // Loop headers rebuild IterArgs once, before visiting their uses. Treat a
    // use as a leaf like Var: the base mutator would revisit its type/init and
    // could mint an undeclared clone when folding a MemRef offset in a branch.
    return VisitExpr_(std::static_pointer_cast<const Var>(op));
  }

  ExprPtr VisitExpr_(const MemRefPtr& op) override { return RebindAllocatedMemRef(op); }

  /// Refresh the Call's result type_ so the in-memory IR matches what a
  /// fresh parse would produce (needed for roundtrip structural equality).
  ///
  /// Also drops identity ``tensor.view`` reinterprets per RFC #1300 §3.3:
  ///   - ``view(x, x.layout)`` → ``x`` (target layout matches source)
  ///
  /// Chain folding (``view(view(x, L1), L2)`` → ``view(x, L2)``)
  /// is intentionally not done at this layer: after SSA conversion the outer
  /// Call references its inner result via a Var binding, not inline, so a
  /// naive pointer inspection cannot see across the binding. A dedicated
  /// SSA-aware chain optimizer can be added if a real pipeline produces such
  /// chains.
  ExprPtr VisitExpr_(const CallPtr& op) override {
    auto base = IRMutator::VisitExpr_(op);
    auto call = std::dynamic_pointer_cast<const Call>(base);
    if (!constants_only_ && IsOp(call, "tensor.view")) {
      base = SimplifyTensorView(call);
    }
    auto new_type = SimplifyType(base->GetType());
    if (new_type.get() == base->GetType().get()) return base;
    call = std::dynamic_pointer_cast<const Call>(base);
    if (!call) return base;
    return std::make_shared<const Call>(call->op_, call->args_, call->kwargs_, call->attrs_, new_type,
                                        call->span_);
  }

  /// Fold arithmetic nodes (Add/Sub/Mul/Div/Min/Max/compare/bitwise/logical)
  /// after children are visited. Needed because Analyzer::Simplify at the
  /// statement-level top does not recurse into non-arithmetic containers
  /// (Call, MakeTuple), so an Add buried inside a shape arg would otherwise
  /// reach downstream with patterns like `K + 0` un-folded.
  ExprPtr VisitBinaryExpr_(const BinaryExprPtr& op) override {
    return FoldProvableComparison(SimplifyValue(IRMutator::VisitBinaryExpr_(op)));
  }

  /// Replace `loop_var OP constant` with a literal when the loop's static range
  /// decides it. IfStmt conditions already fold through Fold A, but a predicate
  /// passed as a call argument (e.g. the `tile.matmul_acc` init_cond that
  /// pipeline replication produces, `ko + 128 == 0`, which the rewriter turns
  /// into `ko == -128`) only got expression rewriting.
  ///
  /// Deliberately limited to the static-bound case, because the analyzer's bounds
  /// are unsound at the int64 edges and under narrow-dtype wrap, and a missed fold
  /// is always safe while a wrong literal is a miscompile:
  ///   - the operand is a loop variable in static_loop_ranges_ (INDEX, constant
  ///     INDEX bounds, no other definition), compared against an INDEX constant
  ///     (a narrower literal can hold an out-of-range value that wraps);
  ///   - the decision compares the header range [start, stop - 1], which holds
  ///     every value the variable takes, with the constant -- never the analyzer's
  ///     bound, which an enclosing `if` can tighten from a wrapping operand.
  ExprPtr FoldProvableComparison(const ExprPtr& expr) {
    if (constants_only_) return expr;
    auto cmp = As<BinaryExpr>(expr);
    // A comparison built through the bindings may carry a non-BOOL dtype; a
    // ConstBool would not type-check where such a value is used.
    if (!cmp || GetScalarDtype(expr) != DataType::BOOL) return expr;
    // Normalize to `var OP constant`; a constant on the left mirrors the order.
    VarPtr var = AsVarLike(cmp->left_);
    auto constant = As<ConstInt>(cmp->right_);
    bool mirrored = false;
    if (!var || !constant) {
      var = AsVarLike(cmp->right_);
      constant = As<ConstInt>(cmp->left_);
      mirrored = true;
    }
    if (!var || !constant || constant->dtype() != DataType::INDEX) return expr;
    auto range = static_loop_ranges_.find(var.get());
    if (range == static_loop_ranges_.end()) return expr;

    const int64_t lo = range->second.first;
    const int64_t hi = range->second.second;
    const int64_t c = constant->value_;

    // With the constant on the left, `c < v` is `v > c`: swap the ordering kinds.
    bool lt = As<Lt>(expr) != nullptr;
    bool le = As<Le>(expr) != nullptr;
    bool gt = As<Gt>(expr) != nullptr;
    bool ge = As<Ge>(expr) != nullptr;
    if (mirrored) {
      std::swap(lt, gt);
      std::swap(le, ge);
    }
    // `always`: every value in [lo, hi] satisfies `var OP c`; `never`: none does.
    // Neither means the range admits both outcomes, and the comparison stays.
    bool always = false;
    bool never = false;
    if (As<Eq>(expr)) {
      always = lo == c && hi == c;
      never = c < lo || c > hi;
    } else if (As<Ne>(expr)) {
      always = c < lo || c > hi;
      never = lo == c && hi == c;
    } else if (lt) {
      always = hi < c;
      never = lo >= c;
    } else if (le) {
      always = hi <= c;
      never = lo > c;
    } else if (gt) {
      always = lo > c;
      never = hi <= c;
    } else if (ge) {
      always = lo >= c;
      never = hi < c;
    }
    if (always) return std::make_shared<ConstBool>(true, expr->span_);
    if (never) return std::make_shared<ConstBool>(false, expr->span_);
    return expr;
  }

  ExprPtr VisitUnaryExpr_(const UnaryExprPtr& op) override {
    return SimplifyValue(IRMutator::VisitUnaryExpr_(op));
  }

  StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
    auto new_value = SimplifyExpr(op->value_);
    auto new_var = AdoptRewrittenValueType(op, new_value, MaybeRebuildVar(op->var_));
    auto new_type = new_var->GetType();

    // Register scalar assignments with the analyzer so downstream expressions
    // can be folded or proven. Two binding strengths:
    //
    //   * Full bind (BindScalar) — a proven constant is substituted into later
    //     uses in its region, including Call arguments and type metadata.
    //     Nested constants have the same value on every execution of their
    //     definition; the binding is scoped to that region, not to a loop's
    //     IterArg initial value or a branch's merged result.
    //   * Bound-only (BindScalarBound) — only a ConstIntBound is registered,
    //     no substitution. Applied to symbolic RHS (e.g. a loop-derived
    //     `ob_idx * 256 + 256`). Enough to prove dead branch guards like
    //     `if expr == 0` without inlining the
    //     scalar into every use site.
    //
    // Both kinds are logged in scalar_binding_log_ so the For/If/While/Spmd
    // visitors can unbind them on leaving the region where the assignment
    // lives, keeping the fold sound for pre-SSA callers.
    //
    // Skip a Var that MultiAssignCollector flagged as assigned more than once:
    // a later assignment would invalidate the value bound here.
    if (As<ScalarType>(new_type) && multi_assigned_.find(op->var_.get()) == multi_assigned_.end()) {
      if (IsConstScalar(new_value)) {
        BindScalar(new_var, new_value);
      } else {
        BindScalarBound(new_var, new_value);
      }
    }

    if (new_value.get() == op->value_.get() && new_var.get() == op->var_.get()) return op;
    auto result = MutableCopy(op);
    result->var_ = new_var;
    result->value_ = new_value;
    return result;
  }

  StmtPtr VisitStmt_(const ForStmtPtr& op) override {
    auto new_start = SimplifyExpr(op->start_);
    auto new_stop = SimplifyExpr(op->stop_);
    auto new_step = SimplifyExpr(op->step_);

    // Seed the whole header before rebuilding its types: storage metadata on
    // an earlier tensor carry may reference a later scalar carry.
    bool iter_args_changed = false;
    auto new_iter_args = RebuildIterArgs(op->iter_args_, &iter_args_changed);

    auto start_ci = As<ConstInt>(new_start);
    auto stop_ci = As<ConstInt>(new_stop);

    // Fold B: collapse a *pure* sequential ForStmt when the analyzer can
    // prove the trip count is 0 or 1. "Pure" means no attrs and Sequential
    // kind — these forms have no execution-model side effects (no
    // Parallel/Unroll/Pipeline scheduling) that a downstream pass might
    // depend on observing as a ForStmt.
    //
    // Trip-count proof:
    //   step >= 1 (forward iteration)
    //   AND (stop <= start            → 0 trips)
    //   OR  (start < stop             AND
    //        stop  <= start + step    → 1 trip)
    //
    // CanProve is used instead of literal-only ConstInt matching so closure-
    // captured trip counts like `for k in pl.range(C, C + 1)` collapse the
    // same way `pl.range(0, 1)` does.
    const bool is_pure_for = op->attrs_.empty() && op->kind_ == ForKind::Sequential;
    if (!constants_only_ && is_pure_for && analyzer_->CanProveGreaterEqual(new_step, 1)) {
      const Span& sp = op->span_;
      int trips = -1;
      if (analyzer_->CanProve(MakeLe(new_stop, new_start, sp))) {
        trips = 0;
      } else if (analyzer_->CanProve(MakeAnd(MakeLt(new_start, new_stop, sp),
                                             MakeLe(new_stop, MakeAdd(new_start, new_step, sp), sp), sp))) {
        trips = 1;
      }

      if (trips == 0) {
        // Loop body never executes: each return_var takes its iter_arg's init value.
        std::vector<StmtPtr> out;
        out.reserve(op->return_vars_.size());
        for (size_t i = 0; i < op->return_vars_.size(); ++i) {
          const auto& init = new_iter_args[i]->initValue_;
          auto rv = AdoptValueType(op->return_vars_[i], op->iter_args_[i]->initValue_->GetType(), init,
                                   MaybeRebuildVar(op->return_vars_[i]));
          out.push_back(std::make_shared<AssignStmt>(rv, init, sp));
        }
        return loop_repair::MakeBody(out, sp);
      }
      // The depth cap keeps a chain of nested single-trip loops from cloning
      // the remaining nest once per level (see kMaxNestedSingleTripFolds); past
      // it the loop falls through to the general path and stays a ForStmt.
      // Empty Require loops need no hardware evidence. Nonempty ones must
      // survive for qualification, even when ordinary folding could unfold them.
      if (trips == 1 && op->hardware_loop_policy_ == HardwareLoopPolicy::Auto &&
          fold_b_depth_ < kMaxNestedSingleTripFolds && !HasEffectfulYield(op->body_) &&
          std::none_of(
              new_iter_args.begin(), new_iter_args.end(),
              [](const IterArgPtr& arg) {
                // Substituting an effectful init could introduce a call into an
                // otherwise harmless yield (or duplicate it at other carry uses).
                return transform_utils::ExprContainsCallLike(arg->initValue_);
              })) {
        // Exactly one iteration: substitute loop_var → start and each
        // iter_arg → its init value via DeepClone (matches the substitution
        // pattern used by LoopUnrollMutator in unroll_loops_pass.cpp), then
        // re-visit so further folds happen, then rewrite the trailing
        // YieldStmt into AssignStmts on the return_vars.
        //
        // DeepClone is used (rather than a local var_remap_ override + analyzer
        // Bind) so the substitution penetrates Var references buried inside
        // type annotations — e.g. MemRef byte_offsets that mention the loop
        // variable. A var_remap_-only approach misses those uses because
        // SimplifyType / SimplifyExpr go through the analyzer at definition
        // sites only when the type is rebuilt; uses inside nested control-flow
        // bodies or tile-view shapes get printed verbatim.
        std::unordered_map<const Var*, ExprPtr> sub_map;
        sub_map.reserve(1 + op->iter_args_.size());
        sub_map.emplace(op->loop_var_.get(), new_start);
        for (size_t i = 0; i < op->iter_args_.size(); ++i) {
          sub_map.emplace(op->iter_args_[i].get(), new_iter_args[i]->initValue_);
        }
        // clone_def_vars=true gives the unrolled body fresh Var identities at
        // every DefField, matching LoopUnrollMutator. This keeps the lifted
        // copy structurally independent of the original (discarded) loop body
        // and lets the re-visit below bind the body's scalars on identities
        // distinct from anything in the surrounding scope.
        auto cloned = DeepClone(op->body_, sub_map, /*clone_def_vars=*/true);
        // The pre-pass indexes original Var identities. Carry its exclusion to
        // fresh definitions so cloning cannot make a reassigned scalar bindable.
        std::vector<const Var*> cloned_multi_assigned;
        std::vector<const Var*> cloned_redefined;
        for (const auto& [original, fresh] : cloned.var_map) {
          if (multi_assigned_.count(original) && multi_assigned_.insert(fresh.get()).second) {
            cloned_multi_assigned.push_back(fresh.get());
          }
          if (redefined_.count(original) && redefined_.insert(fresh.get()).second) {
            cloned_redefined.push_back(fresh.get());
          }
        }

        // Snapshot var_remap_ around the cloned-body visit. MaybeRebuildVar
        // inserts entries keyed by the cloned-body's defining-Var raw pointers
        // (the freshly-allocated clones); after this Fold returns, those clones
        // become unreachable as soon as the rebuilt AssignStmts replace them,
        // and the heap addresses can be recycled by a later make_shared<Var>
        // (e.g. another sibling Fold B that DeepClones a different body). A
        // recycled address would silently match the stale entry and substitute
        // the new Var with an unrelated value — producing AssignStmts whose
        // LHS Var has the wrong type for the RHS (observed on qwen3_decode's
        // q_proj/up_proj as a `pto.textract` whose dst aliases the matmul Acc
        // tile of an earlier iteration).
        //
        // The lift's own additions (return_vars[i] → yielded_values[i]) are
        // applied to the restored baseline below, so subsequent siblings that
        // reference this ForStmt's return_vars still substitute correctly.
        auto baseline_remap = var_remap_;

        // Re-visit so any algebraic patterns exposed by the substitution
        // (e.g. `0 + 64 → 64`) fold in this same Simplify run. The depth guard
        // is what keeps the clone above off the quadratic path — see
        // kMaxNestedSingleTripFolds.
        StmtPtr unrolled_body;
        {
          FoldBDepthGuard fold_guard(fold_b_depth_);
          unrolled_body = VisitStmt(cloned.cloned_body);
        }
        for (const auto* var : cloned_multi_assigned) multi_assigned_.erase(var);
        for (const auto* var : cloned_redefined) redefined_.erase(var);

        var_remap_ = std::move(baseline_remap);
        return LiftBodyToReturnVars(unrolled_body, op->return_vars_, op.get());
      }
    }

    bool bound = start_ci && stop_ci && stop_ci->value_ > start_ci->value_;
    // See FoldProvableComparison for why each condition is required.
    const bool static_loop_var =
        bound && start_ci->dtype() == DataType::INDEX && stop_ci->dtype() == DataType::INDEX &&
        GetScalarDtype(op->loop_var_) == DataType::INDEX && redefined_.count(op->loop_var_.get()) == 0 &&
        static_loop_ranges_
            .emplace(op->loop_var_.get(), std::make_pair(start_ci->value_, stop_ci->value_ - 1))
            .second;
    if (bound) {
      analyzer_->Bind(op->loop_var_, start_ci->value_, stop_ci->value_);
    }

    // Snapshot var_remap_ around the body visit. Nested folds inside the body
    // (Fold A on a nested IfStmt or Fold B on a nested single-trip ForStmt)
    // record remaps from outer Vars to body-internal values; leaking those to
    // siblings of this ForStmt or to the parent scope produces dangling
    // references. The RebuildIterArgs additions made above stay in
    // baseline_remap (they're valid in body and after).
    auto baseline_remap = var_remap_;

    // Scalars assigned inside the loop body are scoped to that body so their
    // values do not leak to siblings of this ForStmt or to code after the loop
    // (pre-SSA soundness).
    auto new_body = VisitScopedBody(op->body_);

    if (bound) {
      analyzer_->Unbind(op->loop_var_);
    }
    if (static_loop_var) static_loop_ranges_.erase(op->loop_var_.get());

    // Discard body-internal remaps — the For is being rebuilt, so the body's
    // SSA scope is preserved and outside code shouldn't see its private vars.
    var_remap_ = std::move(baseline_remap);

    // Rebuild return_vars after the body so folds discovered inside the body
    // are visible in return types.
    bool return_vars_changed = false;
    auto new_return_vars = RebuildVec(
        op->return_vars_, [this](const auto& v) { return MaybeRebuildVar(v); }, &return_vars_changed);

    bool changed = (new_start.get() != op->start_.get()) || (new_stop.get() != op->stop_.get()) ||
                   (new_step.get() != op->step_.get()) || (new_body.get() != op->body_.get()) ||
                   iter_args_changed || return_vars_changed;
    if (!changed) return op;

    auto result = MutableCopy(op);
    result->start_ = new_start;
    result->stop_ = new_stop;
    result->step_ = new_step;
    result->iter_args_ = std::move(new_iter_args);
    result->body_ = new_body;
    result->return_vars_ = std::move(new_return_vars);
    return result;
  }

  StmtPtr VisitStmt_(const SpmdScopeStmtPtr& op) override {
    // Fold the core_num expression whenever the value is compile-time-derivable.
    // Closure-captured Python ints arrive as ConstInt already, but closure
    // arithmetic (e.g. `core_num=MAX // TILE` where both operands are closure
    // ints) may still need one simplify pass after SSA conversion.
    auto new_core_num = SimplifyExpr(op->core_num_);
    // Before outlining, task dependencies and dispatch predicates live in
    // scope attrs. Rewrite them under the enclosing carry mappings just like
    // Submit operands, before visiting definitions local to the scope body.
    auto [new_attrs, attrs_changed] = MutateScopeAttrs(op->attrs_);
    auto new_body = VisitScopedBody(op->body_);
    if (new_core_num.get() == op->core_num_.get() && new_body.get() == op->body_.get() && !attrs_changed) {
      return op;
    }
    auto result = MutableCopy(op);
    result->core_num_ = new_core_num;
    result->body_ = new_body;
    if (attrs_changed) result->attrs_ = std::move(new_attrs);
    return result;
  }

  StmtPtr VisitStmt_(const CommDomainScopeStmtPtr& op) override {
    // Keep one identity while visiting the body: branch-local constraints must
    // not specialize a domain-wide allocation. Its size may depend on scalars
    // defined inside this body, so folding it before the body is too early.
    for (const auto& slot : op->slots_) var_remap_[slot.get()] = slot;
    auto new_body = VisitStmt(op->body_);
    std::vector<WindowBufferPtr> new_slots;
    new_slots.reserve(op->slots_.size());
    bool slots_changed = false;
    for (const auto& slot : op->slots_) {
      var_remap_.erase(slot.get());
      auto new_slot = As<WindowBuffer>(VisitExpr(slot));
      INTERNAL_CHECK_SPAN(new_slot, op->span_)
          << "Simplify: communication-domain slot mutated to non-WindowBuffer";
      var_remap_[slot.get()] = new_slot;
      var_remap_[new_slot.get()] = new_slot;
      if (new_slot != slot) {
        slots_changed = true;
        finalized_windows_[slot.get()] = new_slot;
        finalized_windows_[new_slot.get()] = new_slot;
      }
      new_slots.push_back(std::move(new_slot));
    }
    auto [new_attrs, attrs_changed] = MutateScopeAttrs(op->attrs_);
    if (new_body == op->body_ && !slots_changed && !attrs_changed) return op;
    auto result = MutableCopy(op);
    result->body_ = std::move(new_body);
    result->slots_ = std::move(new_slots);
    result->attrs_ = std::move(new_attrs);
    return result;
  }

  StmtPtr VisitStmt_(const IfStmtPtr& op) override {
    auto new_condition = SimplifyExpr(op->condition_);

    // Prove reachability before visiting either body: folding an unreachable
    // expression (e.g. x // 0 after inlining) can raise before Fold A drops it.
    // Query outside either branch's constraint context so the proof uses only
    // facts valid at entry. Preserve the constants-only mode's control flow.
    const bool always_true = !constants_only_ && analyzer_->CanProve(new_condition);
    const bool always_false =
        !constants_only_ && !always_true && analyzer_->CanProve(MakeNot(new_condition, new_condition->span_));

    // Snapshot var_remap_ so each branch's internal remap additions stay
    // scoped to that branch. Without this, a remap recorded inside the
    // then-branch (e.g. by Fold A on a nested IfStmt, or by Fold B on a
    // nested single-trip ForStmt) would still be live when the else-branch
    // is visited — substituting refs in else with values that name vars
    // defined inside the then-branch's lifted body, producing IR with
    // dangling references and type-mismatched assignments.
    //
    // Each branch is processed from the same baseline; its additions are
    // captured separately. When Fold A keeps a branch, that branch's snapshot
    // is adopted (the lifted body's defs become visible in the parent scope).
    // When the IfStmt is rebuilt, both snapshots are discarded — branch
    // bodies remain in their own scopes.
    auto baseline_remap = var_remap_;

    // Scalars assigned inside a branch are scoped to that branch: unbind them
    // after each branch visit so a then-branch scalar does not leak into the
    // else branch, and neither leaks past the IfStmt (pre-SSA soundness).
    auto scalar_mark = scalar_binding_log_.size();

    // Enter constraint scope for then branch (condition is known true).
    // Defer the unreachable body until we know whether the IfStmt survives.
    StmtPtr new_then = op->then_body_;
    if (!always_false) {
      auto ctx = analyzer_->GetConstraintContext(new_condition);
      new_then = VisitStmt(op->then_body_);
    }
    auto then_remap = std::move(var_remap_);
    var_remap_ = baseline_remap;
    UnbindScalarsSince(scalar_mark);

    // Enter constraint scope for else branch (condition is known false → Not(condition)).
    std::optional<StmtPtr> new_else = op->else_body_;
    if (!always_true && op->else_body_.has_value()) {
      auto ctx = analyzer_->GetConstraintContext(MakeNot(new_condition, new_condition->span_));
      new_else = VisitStmt(*op->else_body_);
    }
    auto else_remap = std::move(var_remap_);
    var_remap_ = baseline_remap;
    UnbindScalarsSince(scalar_mark);

    // Fold A: collapse the IfStmt when the analyzer can prove the polarity.
    // True  → keep then_body_; drop else.
    // False → keep else_body_ if present, else replace with empty body
    //         (which is only valid when return_vars_ is empty per IR contract).
    // LiftBodyToReturnVars strips the kept branch's trailing YieldStmt and
    // remaps or materializes its return_vars so later uses remain well-defined.
    const auto selected_body = always_true ? new_then : new_else.value_or(nullptr);
    if ((always_true || always_false) && !HasEffectfulYield(selected_body)) {
      StmtPtr kept;
      if (always_true) {
        kept = new_then;
        var_remap_ = std::move(then_remap);  // adopt kept branch's remaps
      } else if (new_else.has_value()) {
        kept = *new_else;
        var_remap_ = std::move(else_remap);
      } else {
        INTERNAL_CHECK_SPAN(op->return_vars_.empty(), op->span_)
            << "Internal error: IfStmt with no else branch must have empty return_vars_";
        return loop_repair::MakeBody({}, op->span_);
      }
      return LiftBodyToReturnVars(kept, op->return_vars_, op.get());
    }

    // An effectful yield can retain a proven IfStmt. Its dead body must still
    // refer to the enclosing rebuilt binders, including references in types,
    // but must not run through the analyzer or fold invalid arithmetic.
    if (always_false || (always_true && new_else.has_value())) {
      UnreachableBodyRemapper remapper(baseline_remap, multi_assigned_);
      if (always_false) {
        new_then = remapper.VisitStmt(op->then_body_);
      } else {
        new_else = remapper.VisitStmt(*op->else_body_);
      }
    }

    // Branch-local mappings were discarded, but the merged result's own type
    // must still follow enclosing binder rebuilds (for example an allocated
    // offset naming a rebuilt loop carry). Rebuild under the common baseline,
    // as ForStmt/WhileStmt do, so later leaf uses resolve to the same new binder.
    // These are new definitions even when pre-SSA input reuses an old Var.
    // Drop earlier value facts before rebuilding all result types, then bind
    // every later use to the retained IfStmt's result rather than its old alias.
    for (const auto& var : op->return_vars_) {
      var_remap_.erase(var.get());
      analyzer_->Unbind(var);
    }
    bool return_vars_changed = false;
    auto new_return_vars = RebuildVec(
        op->return_vars_,
        [this](const auto& var) {
          auto result = MaybeRebuildVar(var);
          var_remap_[var.get()] = result;
          return result;
        },
        &return_vars_changed);

    bool changed = (new_condition.get() != op->condition_.get()) ||
                   (new_then.get() != op->then_body_.get()) ||
                   (new_else.has_value() != op->else_body_.has_value()) ||
                   (new_else.has_value() && new_else->get() != op->else_body_->get()) || return_vars_changed;
    if (!changed) return op;
    auto result = MutableCopy(op);
    result->condition_ = new_condition;
    result->then_body_ = new_then;
    result->else_body_ = new_else;
    result->return_vars_ = std::move(new_return_vars);
    return result;
  }

  StmtPtr VisitStmt_(const WhileStmtPtr& op) override {
    // Rebuild iter_args before the condition and body, mirroring ForStmt.
    // An IterArg *use* is the same node as its declaration, so the base
    // IRMutator mints a fresh IterArg at the first use whose initValue_ the
    // analyzer rewrote (e.g. a top-level `i: Scalar[INDEX] = 0` that
    // VisitStmt_(AssignStmtPtr) full-bound). Rebuilding here seeds var_remap_
    // so the header and every body reference resolve to one node; without it
    // the header kept the stale IterArg while all uses pointed at an
    // undefined clone (UseAfterDef).
    bool iter_args_changed = false;
    auto new_iter_args = RebuildIterArgs(op->iter_args_, &iter_args_changed);

    auto new_condition = SimplifyExpr(op->condition_);

    // Snapshot var_remap_ around the body visit, as ForStmt does. VisitScopedBody
    // unbinds scalars but not remaps, and a nested fold inside the body (Fold A on
    // an IfStmt, Fold B on a single-trip ForStmt) records `outer_var -> body-local
    // value`. Leaking that past the loop rewrites post-loop uses of a leak-mode
    // body var into a value from one iteration's interior -- silently wrong, since
    // the var is still in scope so no verifier flags it. The RebuildIterArgs
    // additions above are captured in the baseline (they stay valid after the loop).
    auto baseline_remap = var_remap_;
    auto new_body = VisitScopedBody(op->body_);
    var_remap_ = std::move(baseline_remap);

    // Rebuild return_vars after the body so folds discovered inside it are
    // visible in the return types (same ordering rationale as ForStmt).
    bool return_vars_changed = false;
    auto new_return_vars = RebuildVec(
        op->return_vars_, [this](const auto& v) { return MaybeRebuildVar(v); }, &return_vars_changed);

    bool changed = (new_condition.get() != op->condition_.get()) || (new_body.get() != op->body_.get()) ||
                   iter_args_changed || return_vars_changed;
    if (!changed) return op;
    auto result = MutableCopy(op);
    result->iter_args_ = std::move(new_iter_args);
    result->condition_ = new_condition;
    result->body_ = new_body;
    result->return_vars_ = std::move(new_return_vars);
    return result;
  }

  StmtPtr VisitStmt_(const ReturnStmtPtr& op) override {
    std::vector<ExprPtr> new_values;
    bool changed = false;
    new_values.reserve(op->value_.size());
    for (const auto& val : op->value_) {
      auto new_val = SimplifyExpr(val);
      new_values.push_back(new_val);
      if (new_val.get() != val.get()) changed = true;
    }
    if (!changed) return op;
    auto result = MutableCopy(op);
    result->value_ = std::move(new_values);
    return result;
  }

  StmtPtr VisitStmt_(const YieldStmtPtr& op) override {
    std::vector<ExprPtr> new_values;
    bool changed = false;
    new_values.reserve(op->value_.size());
    for (const auto& val : op->value_) {
      auto new_val = SimplifyExpr(val);
      new_values.push_back(new_val);
      if (new_val.get() != val.get()) changed = true;
    }
    if (!changed) return op;
    auto result = MutableCopy(op);
    result->value_ = std::move(new_values);
    return result;
  }

  StmtPtr VisitStmt_(const EvalStmtPtr& op) override {
    auto new_expr = SimplifyExpr(op->expr_);
    if (new_expr.get() == op->expr_.get()) return op;
    auto result = MutableCopy(op);
    result->expr_ = new_expr;
    return result;
  }

 private:
  /// Identity elimination per RFC #1300 §3.3:
  /// ``view(x, layout=x.layout)`` → ``x``.
  ///
  /// Drops a ``tensor.view`` call when the requested target layout
  /// matches what the source already carries — the call is then a no-op
  /// metadata reinterpret and downstream consumers can use ``src`` directly.
  /// (When layouts differ, ``view`` performs the canonical-pair swap;
  /// such substantive reinterprets are preserved.)
  ///
  /// Chain folding (``view(view(x, L1), L2)`` → ``view(x, L2)``)
  /// is intentionally not implemented here. After SSA the outer Call's arg is
  /// a Var bound to the inner Call (not the inner Call inline), so naive
  /// pointer inspection cannot see across the binding. A dedicated SSA-aware
  /// chain optimizer can be added if real pipelines produce such chains.
  ExprPtr SimplifyTensorView(const std::shared_ptr<const Call>& call) {
    // Shape-only views (2 args: src + shape) are never identity folds --
    // they reinterpret shape by design. Only layout-only views (1 arg)
    // can be simplified when source and target layout are identical.
    if (call->args_.size() != 1) return call;
    auto src = call->args_[0];

    auto src_tensor = AsTensorTypeLike(src->GetType());
    auto out_tensor = AsTensorTypeLike(call->GetType());
    if (!src_tensor || !out_tensor) return call;

    // Bare TensorType is implicitly ND-packed.
    TensorLayout src_layout =
        src_tensor->tensor_view_.has_value() ? src_tensor->tensor_view_->layout : TensorLayout::ND;
    TensorLayout target_layout =
        out_tensor->tensor_view_.has_value() ? out_tensor->tensor_view_->layout : TensorLayout::ND;
    if (src_layout == target_layout) {
      return src;
    }
    return call;
  }

  /// Compose var-remap (via the base-class `var_remap_`) with analyzer-based
  /// constant folding — the Analyzer only knows about its own bindings and
  /// ignores our Var rebuilds, so remap must run first.
  ExprPtr SimplifyExpr(const ExprPtr& e) {
    if (!e) return e;
    return SimplifyValue(VisitExpr(e));
  }

  ExprPtr SimplifyValue(const ExprPtr& expr) {
    auto folded = analyzer_->Simplify(expr);
    return !constants_only_ || IsConstScalar(folded) ? folded : expr;
  }

  MemRefPtr RebindAllocatedMemRef(const MemRefPtr& memref) {
    if (memref->is_pinned_ || var_remap_.empty()) return memref;
    StorageVarRemapper remapper(var_remap_);
    auto new_base = As<Var>(remapper.VisitExpr(memref->base_));
    INTERNAL_CHECK_SPAN(new_base, memref->span_) << "Simplify: allocated MemRef base must remain a Var";
    auto new_offset = memref->byte_offset_ ? remapper.VisitExpr(memref->byte_offset_) : nullptr;
    auto new_index = memref->slot_index_;
    if (new_index.has_value() && *new_index) new_index = remapper.VisitExpr(*new_index);
    if (new_base == memref->base_ && new_offset == memref->byte_offset_ && new_index == memref->slot_index_) {
      return memref;
    }

    // Recompute under the current var_remap_: loop headers add mappings and
    // branch/loop exits restore them. Reuse a cached result only when its actual
    // address expressions still match, with strict free-variable identities.
    // This also lets Var and Call result types share one rebuilt MemRef even
    // though rebinding an arithmetic address creates fresh expression nodes.
    auto same_expr = [](const ExprPtr& lhs, const ExprPtr& rhs) {
      return lhs == rhs || (lhs && rhs && structural_equal(lhs, rhs));
    };
    auto cached = rebound_memrefs_.find(memref);
    if (cached != rebound_memrefs_.end()) {
      const auto& previous = cached->second;
      const bool same_index = new_index.has_value() == previous->slot_index_.has_value() &&
                              (!new_index.has_value() || same_expr(*new_index, *previous->slot_index_));
      if (new_base == previous->base_ && same_expr(new_offset, previous->byte_offset_) && same_index) {
        return previous;
      }
    }
    auto result = std::make_shared<MemRef>(memref->name_hint_, std::move(new_base), std::move(new_offset),
                                           memref->size_, memref->span_, memref->is_pinned_,
                                           memref->slot_count_, std::move(new_index));
    rebound_memrefs_[memref] = result;
    return result;
  }

  std::optional<MemRefPtr> SimplifyTypeMemRef(const std::optional<MemRefPtr>& memref, bool& changed) {
    if (!memref.has_value()) return memref;
    std::optional<MemRefPtr> rebound = RebindAllocatedMemRef(*memref);
    changed = changed || rebound != memref;
    return RemapPinnedMemRefExprs(
        rebound, [this](const ExprPtr& expr) { return SimplifyExpr(expr); }, changed);
  }

  std::vector<ExprPtr> SimplifyExprVec(const std::vector<ExprPtr>& vec, bool* changed) {
    return RebuildVec(vec, [this](const ExprPtr& e) { return SimplifyExpr(e); }, changed);
  }

  /// Map @p rebuild over @p vec; sets *changed if any element's identity
  /// differs from the input.
  template <typename Ptr, typename F>
  static std::vector<Ptr> RebuildVec(const std::vector<Ptr>& vec, F&& rebuild, bool* changed) {
    std::vector<Ptr> out;
    out.reserve(vec.size());
    for (const auto& x : vec) {
      auto nx = rebuild(x);
      if (nx.get() != x.get()) *changed = true;
      out.push_back(std::move(nx));
    }
    return out;
  }

  /// Rebuild a TensorType or TileType with its shape, stride, valid_shape,
  /// start_offset and declared slot index passed through `SimplifyExpr`.
  /// Resolved MemRefs retain their identity unless a binder is rebuilt; shared
  /// address rebinding then keeps their types coherent without folding runtime
  /// offsets. Their remaining symbolic address uses must stay live for DCE.
  ///
  /// Nothing is assumed about the variables inside them. Being a shape or a
  /// valid_shape makes the *field* an extent; it does not make every free
  /// variable in it one. `valid = max(-x, 0)` is a legal dynamic extent over a
  /// signed runtime scalar, and assuming `x >= 0` folds it to a constant `0`,
  /// silently shrinking the region.
  TypePtr SimplifyType(const TypePtr& type) {
    if (!type) return type;
    if (auto t = AsTensorTypeLike(type)) {
      bool changed = false;
      auto new_memref = SimplifyTypeMemRef(t->memref_, changed);
      auto new_shape = SimplifyExprVec(t->shape_, &changed);
      std::optional<TensorView> new_tv = t->tensor_view_;
      if (t->tensor_view_.has_value()) {
        const auto& tv = *t->tensor_view_;
        bool view_changed = false;
        auto new_stride = SimplifyExprVec(tv.stride, &view_changed);
        auto new_vs = SimplifyExprVec(tv.valid_shape, &view_changed);
        if (view_changed) {
          changed = true;
          new_tv = TensorView(std::move(new_stride), tv.layout, std::move(new_vs), tv.pad);
        }
      }
      // Remap the window_buffer_ back-reference in lockstep with the
      // CommDomainScopeStmt slots. Folding a WindowBuffer's size_ (e.g.
      // `world_size * 1 * 4` -> `world_size * 4` on a synthesized signal)
      // mints a FRESH WindowBuffer via VisitExpr_(WindowBufferPtr) and records
      // it in var_remap_; the scope slots pick that up. If this rebuild left
      // window_buffer_ pointing at the pre-fold object, the view var and its
      // scope slot would diverge and DistributedCodegen's ScopeForWindowBuffer
      // (pointer-identity scan) would fail with "not a slot of any open
      // CommDomainScopeStmt".
      std::optional<WindowBufferPtr> new_wb;
      if (auto distributed = As<DistributedTensorType>(type)) {
        new_wb = distributed->window_buffer_;
        if (distributed->window_buffer_.has_value()) {
          auto remapped = As<WindowBuffer>(VisitExpr(*distributed->window_buffer_));
          INTERNAL_CHECK(remapped) << "Simplify: DistributedTensorType window_buffer_ mutated to "
                                      "non-WindowBuffer";
          if (remapped.get() != distributed->window_buffer_->get()) changed = true;
          new_wb = std::move(remapped);
        }
      }
      if (!changed) return type;
      if (auto distributed = As<DistributedTensorType>(type)) {
        return std::make_shared<DistributedTensorType>(std::move(new_shape), t->dtype_, new_memref,
                                                       std::move(new_tv), std::move(new_wb));
      }
      return std::make_shared<TensorType>(std::move(new_shape), t->dtype_, new_memref, std::move(new_tv));
    }
    if (auto t = As<TileType>(type)) {
      bool changed = false;
      auto new_memref = SimplifyTypeMemRef(t->memref_, changed);
      auto new_shape = SimplifyExprVec(t->shape_, &changed);
      std::optional<TileView> new_tv = t->tile_view_;
      if (t->tile_view_.has_value()) {
        const auto& tv = *t->tile_view_;
        bool view_changed = false;
        auto new_vs = SimplifyExprVec(tv.valid_shape, &view_changed);
        auto new_stride = SimplifyExprVec(tv.stride, &view_changed);
        auto new_offset = tv.start_offset ? SimplifyExpr(tv.start_offset) : tv.start_offset;
        if (new_offset.get() != tv.start_offset.get()) view_changed = true;
        if (view_changed) {
          changed = true;
          TileView ntv = tv;
          ntv.valid_shape = std::move(new_vs);
          ntv.stride = std::move(new_stride);
          ntv.start_offset = std::move(new_offset);
          new_tv = std::move(ntv);
        }
      }
      if (!changed) return type;
      return std::make_shared<TileType>(std::move(new_shape), t->dtype_, new_memref, std::move(new_tv),
                                        t->memory_space_);
    }
    if (auto t = As<TupleType>(type)) {
      bool changed = false;
      std::vector<TypePtr> new_types;
      new_types.reserve(t->types_.size());
      for (const auto& inner : t->types_) {
        auto new_inner = SimplifyType(inner);
        if (new_inner.get() != inner.get()) changed = true;
        new_types.push_back(std::move(new_inner));
      }
      if (!changed) return type;
      return std::make_shared<TupleType>(std::move(new_types));
    }
    return type;
  }

  static bool IsConstScalar(const ExprPtr& e) {
    return e && (As<ConstInt>(e) || As<ConstFloat>(e) || As<ConstBool>(e));
  }

  /// RAII increment of fold_b_depth_ for the lifetime of the guard. Wrapped
  /// around the re-visit of a Fold B clone. See kMaxNestedSingleTripFolds.
  struct FoldBDepthGuard {
    int& depth;
    explicit FoldBDepthGuard(int& d) : depth(d) { ++depth; }
    ~FoldBDepthGuard() { --depth; }
    FoldBDepthGuard(const FoldBDepthGuard&) = delete;
    FoldBDepthGuard& operator=(const FoldBDepthGuard&) = delete;
  };

  /// Bind a scalar Var to a constant @p value (full substitution across all
  /// sub-analyzers) and log it so the enclosing scope can unbind it on exit.
  /// See VisitStmt_(AssignStmtPtr).
  void BindScalar(const VarPtr& var, const ExprPtr& value) {
    analyzer_->Bind(var, value);
    scalar_binding_log_.push_back(var);
  }

  /// Register only a ConstIntBound for a symbolic scalar Var (no value
  /// substitution) and log it for scoped unbinding. This lets the analyzer
  /// prove dead branch guards from the value's range without inlining the
  /// scalar into its use sites.
  ///
  /// The RHS range is recorded as-is. It is deliberately *not* intersected with
  /// whatever `const_int_bound(var)` answers for the still-unbound `var`: the
  /// value being produced is the strongest evidence available, and intersecting
  /// it with a weaker standing assumption can only delete reachable values.
  ///
  /// That mattered when an unbound `INDEX` Var defaulted to `[0, +inf)`:
  /// `window = pos - 1` was recorded as `[0, +inf)` instead of `[-1, +inf)`, so
  /// a user guard such as `if window >= 0` folded away as statically true and
  /// the bounds check the kernel depended on was silently dropped (issue
  /// #2500). That default is gone -- an unbound Var is now `[-inf, +inf]`
  /// whatever its dtype -- but the rule stands on its own, and still holds for
  /// any var the caller has bound before this point.
  void BindScalarBound(const VarPtr& var, const ExprPtr& value) {
    analyzer_->const_int_bound.Update(var, analyzer_->const_int_bound(value));
    scalar_binding_log_.push_back(var);
  }

  /// Unbind every scalar logged at or after @p mark and shrink the log back to
  /// @p mark. Called when leaving a structural region (loop / if-branch /
  /// while / spmd body) so a scalar defined inside it does not leak its value
  /// to siblings or to code after the region — required for pre-SSA soundness.
  void UnbindScalarsSince(size_t mark) {
    for (size_t i = scalar_binding_log_.size(); i > mark; --i) {
      analyzer_->Unbind(scalar_binding_log_[i - 1]);
    }
    scalar_binding_log_.resize(mark);
  }

  /// Visit a structural region's body, then unbind any scalars the body bound
  /// so their values do not leak to siblings
  /// or to code after the region (pre-SSA soundness). Used for loop / while /
  /// spmd bodies; IfStmt branches manage the mark manually because both
  /// branches share one mark and interleave var_remap_ snapshots.
  StmtPtr VisitScopedBody(const StmtPtr& body) {
    auto scalar_mark = scalar_binding_log_.size();
    auto new_body = VisitStmt(body);
    UnbindScalarsSince(scalar_mark);
    return new_body;
  }

  /// Keep ``var.type == value.type`` when simplification changed either side's
  /// type. An identity ``tensor.view`` fold (SimplifyTensorView) replaces a view
  /// with its source, whose type may spell the same layout differently — a bare
  /// ``Tensor[[8, 4]]`` versus the view's explicit
  /// ``TensorView(stride=[4, 1], ND)`` — whether the view is the whole RHS or
  /// nested inside it (e.g. a tuple element). The type change then reaches
  /// every later assignment that reads the retyped Var. Conversely,
  /// MaybeRebuildVar can fold the Var's own type (``[N + 0, 4]`` to ``[N, 4]``)
  /// while an alias RHS such as a function parameter keeps its unfolded type.
  /// Either way the Var takes the value's type; later uses follow through
  /// ``var_remap_``. Only an asymmetry this pass introduced is repaired — one
  /// already present in the input is left for the verifier to report. A
  /// multiply-assigned (pre-SSA) Var keeps its type: the other assignments
  /// still bind it.
  VarPtr AdoptRewrittenValueType(const AssignStmtPtr& op, const ExprPtr& new_value, const VarPtr& new_var) {
    if (multi_assigned_.count(op->var_.get()) > 0) return new_var;
    return AdoptValueType(op->var_, op->value_->GetType(), new_value, new_var);
  }

  /// Shared by AdoptRewrittenValueType and the assignments Simplify synthesizes
  /// itself (a folded loop's return vars): @p new_var, the rebuilt
  /// @p original_var, takes @p new_value's type when the two were equal before
  /// simplification (@p original_value_type) but are not any more.
  VarPtr AdoptValueType(const VarPtr& original_var, const TypePtr& original_value_type,
                        const ExprPtr& new_value, const VarPtr& new_var) {
    auto retyped = MatchRewrittenValueType(original_var, original_value_type, new_value, new_var);
    if (retyped != new_var) var_remap_[original_var.get()] = retyped;
    return retyped;
  }

  /// Rebuild a Var with a simplified type, recording the remap so downstream
  /// VarExpr references pick up the new identity. If the Var was already
  /// rebuilt earlier (e.g., at its defining AssignStmt during the body
  /// traversal), return that existing remap so ForStmt.return_vars_ stays
  /// identical to the body-side definition.
  VarPtr MaybeRebuildVar(const VarPtr& var) {
    if (!var) return var;
    if (auto existing = LookupVarRemap(var.get())) return existing;
    auto new_type = SimplifyType(var->GetType());
    if (new_type.get() == var->GetType().get()) return var;
    auto new_var = std::make_shared<Var>(var->name_hint_, new_type, var->span_);
    var_remap_[var.get()] = new_var;
    return new_var;
  }

  std::vector<IterArgPtr> RebuildIterArgs(const std::vector<IterArgPtr>& iter_args, bool* changed) {
    // Seeds belong to the enclosing scope, not the loop's carry scope. Fold
    // all of them before installing this header's binder remaps.
    std::vector<ExprPtr> new_inits;
    new_inits.reserve(iter_args.size());
    for (const auto& ia : iter_args) {
      auto new_init = SimplifyExpr(ia->initValue_);
      if (new_init != ia->initValue_) *changed = true;
      new_inits.push_back(std::move(new_init));
    }

    // A discovery pass preserves every original identity for a no-op header.
    // Once anything changes, clone the entire group: changing one binder can
    // require another carry's type to change, independently of header order.
    if (!*changed) {
      for (const auto& ia : iter_args) {
        if (SimplifyType(ia->GetType()) != ia->GetType()) {
          *changed = true;
          break;
        }
      }
    }
    if (!*changed) return iter_args;

    // Only these unpublished construction shells are mutable. A real IterArg
    // constructor gives each binder a fresh unique identity; MutableCopy is
    // unsuitable for identity-bearing nodes. Rewrite visitors treat mapped
    // carries as leaves; all types are finalized before visiting the body.
    class HeaderIterArg final : public IterArg {
     public:
      using IterArg::IterArg;
      void FinalizeType(TypePtr type) { type_ = std::move(type); }
    };
    std::vector<std::shared_ptr<HeaderIterArg>> fresh;
    std::vector<IterArgPtr> result;
    fresh.reserve(iter_args.size());
    result.reserve(iter_args.size());
    for (size_t i = 0; i < iter_args.size(); ++i) {
      const auto& ia = iter_args[i];
      auto new_ia =
          std::make_shared<HeaderIterArg>(ia->name_hint_, ia->GetType(), std::move(new_inits[i]), ia->span_);
      var_remap_[ia.get()] = new_ia;
      fresh.push_back(new_ia);
      result.push_back(std::move(new_ia));
    }
    // Every old -> fresh entry must exist before the first final type visit.
    // In particular, allocated byte offsets and retained slot indices must
    // refer to the declared fresh scalar even when its carry follows theirs.
    for (size_t i = 0; i < iter_args.size(); ++i) {
      fresh[i]->FinalizeType(SimplifyType(iter_args[i]->GetType()));
    }
    return result;
  }

  VarPtr LookupVarRemap(const Var* key) {
    auto it = var_remap_.find(key);
    if (it == var_remap_.end()) return nullptr;
    return std::dynamic_pointer_cast<const Var>(it->second);
  }

  /// Lift @p kept_body into the parent scope when a control-flow fold has
  /// chosen it as the surviving branch, @p folded being the statement being
  /// collapsed. Strips the trailing YieldStmt and records
  /// `return_vars[i] → yielded_value[i]` in `var_remap_` so any downstream
  /// uses (subsequent siblings, ReturnStmt) read the yielded value directly.
  /// Shared by Fold A (IfStmt) and the one-trip case of Fold B (ForStmt).
  ///
  /// Substituting via var_remap_ (rather than emitting `AssignStmt(rv, val)`)
  /// avoids alias assignments that the orchestration codegen lowers
  /// incorrectly when both sides derive from a role-tagged parameter name
  /// (e.g. both vars have base name "out", producing `auto out = out;`).
  ///
  /// The substitution only reaches uses visited while the entry is live: the
  /// enclosing ForStmt / WhileStmt / IfStmt rebases `var_remap_` on the way
  /// out. A return var whose uses outlive that restore is materialized as a
  /// real `AssignStmt` instead, since the fold is otherwise removing its only
  /// definition. That assignment stays *at the fold site*: the yielded value
  /// may name Vars local to the body, so it cannot be hoisted past the loop.
  /// ReturnVarEscapeIndex proves no return var escapes in SSA form, so this
  /// only fires for callers running Simplify on pre-SSA IR.
  StmtPtr LiftBodyToReturnVars(const StmtPtr& kept_body, const std::vector<VarPtr>& return_vars,
                               const Stmt* folded) {
    if (return_vars.empty()) return kept_body;
    auto stripped = StripTrailingYield(kept_body, return_vars.size());

    std::vector<StmtPtr> materialized;
    for (size_t i = 0; i < return_vars.size(); ++i) {
      const auto& value = stripped.yielded_values[i];
      if (escapes_->Escapes(folded, return_vars[i].get())) {
        auto rv = AdoptValueType(return_vars[i], value->GetType(), value, MaybeRebuildVar(return_vars[i]));
        materialized.push_back(std::make_shared<AssignStmt>(rv, value, value->span_));
        continue;
      }
      var_remap_[return_vars[i].get()] = value;
    }

    if (materialized.empty()) {
      return stripped.body ? stripped.body : loop_repair::MakeBody({}, kept_body->span_);
    }
    std::vector<StmtPtr> out;
    if (stripped.body) out = transform_utils::FlattenToStmts(stripped.body);
    out.insert(out.end(), materialized.begin(), materialized.end());
    return loop_repair::MakeBody(out, kept_body->span_);
  }

  std::unordered_set<const Var*> multi_assigned_;
  /// See MultiAssignCollector::redefined.
  std::unordered_set<const Var*> redefined_;
  /// Trusted enclosing loop variables mapped to their header range [start, stop - 1]
  /// (a step > 1 only skips values inside it). See FoldProvableComparison.
  std::unordered_map<const Var*, std::pair<int64_t, int64_t>> static_loop_ranges_;

  // Owning keys keep source MemRefs alive across type rewrites and temporary
  // control-flow clones; a recycled raw pointer must never hit this cache.
  std::unordered_map<MemRefPtr, MemRefPtr> rebound_memrefs_;

  /// Escape analysis for the return vars of foldable control-flow statements.
  /// Owned by TransformSimplify, valid for this mutator's lifetime.
  ReturnVarEscapeIndex* escapes_;

  /// InitMemRef shares constant propagation without changing the control-flow
  /// and carry structure used by the memory planner or unrelated symbolic math.
  bool constants_only_;

  /// Scalar Vars bound via BindScalar / BindScalarBound, in bind order. Used
  /// by UnbindScalarsSince to scope bindings to the region they were made in.
  std::vector<VarPtr> scalar_binding_log_;

  /// How many Fold B clones are on the stack above the statement being
  /// visited. 0 in the original function body; incremented for each enclosing
  /// re-visit of a Fold B clone. Gates further single-trip collapses, bounding
  /// the pass's cloning cost — see kMaxNestedSingleTripFolds.
  int fold_b_depth_ = 0;

  /// Old domain slots and finalized replacements retained until the final sweep.
  std::unordered_map<const Expr*, ExprPtr> finalized_windows_;

  /// Cap on Fold B's substitution depth: how many levels of *nested* single-trip
  /// loops one Simplify run collapses.
  ///
  /// Fold B lifts a one-trip body by DeepCloning it with `loop_var → start`
  /// substituted, then re-visits the clone so nested folds land in the same run.
  /// When the body is itself a single-trip loop, that re-visit clones again —
  /// so an unbounded chain clones the remaining nest once per level and the
  /// clone sizes run N, N-1, ..., 1. That is O(N^2), over the O(N log N) ceiling
  /// `.claude/rules/pass-complexity.md` sets.
  ///
  /// The cap restores a linear bound. Folds at one depth sit at disjoint
  /// positions, so they clone at most N nodes between them; across at most
  /// kMaxNestedSingleTripFolds depths the run clones O(N) nodes in total.
  ///
  /// Declining is always sound: a single-trip ForStmt is well-formed IR, and a
  /// later Simplify run collapses the next chunk of the chain. The value is far
  /// above any real nest of *provably* one-trip pure loops (a handful at most),
  /// so no kernel in the pipeline reaches it.
  static constexpr int kMaxNestedSingleTripFolds = 16;
};

FunctionPtr TransformSimplify(const FunctionPtr& func, bool constants_only) {
  MultiAssignCollector collector(func->params_);
  for (const auto& param : func->params_) collector.Define(param.get());
  collector.VisitStmt(func->body_);

  ReturnVarEscapeIndex escapes;
  if (!constants_only) escapes.Index(func->body_);

  auto analyzer = std::make_shared<arith::Analyzer>();
  SimplifyMutator mutator(analyzer.get(), std::move(collector.multi_assigned), std::move(collector.redefined),
                          &escapes, constants_only);
  auto new_body = mutator.RemapFinalizedWindows(mutator.VisitStmt(func->body_));

  // Final step: drop every dead yielded slot — IfStmt phi return_vars and
  // loop-carried iter_arg / return_var pairs — plus the matching yield slots,
  // with conservative scalar DCE on either side so both cascade directions
  // resolve in one Simplify invocation. The pre-prune scalar DCE removes dead
  // scalar bindings that were the slot's only consumer (so the now-dead slot
  // becomes visible to the prune step); the post-prune scalar DCE removes the
  // body scalar assigns orphaned by pruning (fixes #1603 — outlining was
  // capturing dead phi as a spurious Scalar[INDEX] return; the loop-carry half
  // is the same defect one node kind over, where SSA seeds a loop with a value
  // the body overwrites before reading and outlining then had to hand the
  // scalar back out of a device kernel).
  // Call-backed assignments are preserved because the IR has no purity
  // annotations yet — a Call may have observable side effects we cannot reason
  // about. A scalar consumed only by an SPMD ``core_num`` needs no special
  // shield: the spec rides the dispatch in this function and IRVisitor walks
  // the Call's Expr-valued attrs, so it is an ordinary use.
  auto flat = transform_utils::FlattenToStmts(new_body);
  auto pre_pruned = dce::EliminateDeadScalarAssignments(flat);
  auto slot_pruned = constants_only ? pre_pruned : dce::EliminateDeadYieldSlots(pre_pruned);
  auto pruned = dce::EliminateDeadScalarAssignments(slot_pruned);
  bool dce_changed = pruned.size() != flat.size() ||
                     !std::equal(pruned.begin(), pruned.end(), flat.begin(),
                                 [](const StmtPtr& a, const StmtPtr& b) { return a.get() == b.get(); });
  StmtPtr final_body = dce_changed ? loop_repair::MakeBody(pruned, new_body->span_) : new_body;

  if (final_body.get() == func->body_.get()) return func;
  auto result = MutableCopy(func);
  result->body_ = final_body;
  return result;
}

ProgramPtr TransformSimplifyProgram(const ProgramPtr& program) {
  if (!program) return program;

  auto new_functions = program->functions_;
  bool changed = false;
  for (auto& [gvar, func] : new_functions) {
    if (!func) continue;
    auto transformed = TransformSimplify(func, /*constants_only=*/false);
    if (transformed.get() != func.get()) {
      func = transformed;
      changed = true;
    }
  }
  if (!changed) return program;
  return std::make_shared<const Program>(std::move(new_functions), program->name_, program->span_);
}

}  // namespace

namespace transform_utils {

FunctionPtr SimplifyFunction(const FunctionPtr& func, bool constants_only) {
  return TransformSimplify(func, constants_only);
}

}  // namespace transform_utils

namespace pass {

Pass Simplify() { return CreateProgramPass(TransformSimplifyProgram, "Simplify", kSimplifyProperties); }

}  // namespace pass

}  // namespace ir
}  // namespace pypto
