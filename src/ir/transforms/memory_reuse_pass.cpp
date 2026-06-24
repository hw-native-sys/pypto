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

#include <algorithm>
#include <any>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/backend/common/backend_config.h"
#include "pypto/backend/common/backend_handler.h"
#include "pypto/core/any_cast.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memref.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_context.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/memref_collectors.h"
#include "pypto/ir/transforms/utils/memref_utils.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

/**
 * @brief Lifetime interval for a TileType variable (based on topological order)
 */
struct LifetimeInterval {
  VarPtr variable;           ///< The variable
  int def_point;             ///< Definition point (topological order)
  int last_use_point;        ///< Last use point (topological order)
  MemorySpace memory_space;  ///< Memory space
  uint64_t size;             ///< Size in bytes
};

namespace {

/**
 * @brief Result of lifetime computation
 */
struct LifetimeAnalysisResult {
  std::vector<LifetimeInterval> lifetimes;
  std::map<VarPtr, std::vector<VarPtr>> var_sharing_groups;
  // var -> per-(scf.if, slot) phi family ids (see LifetimeAnalyzer).
  std::map<const Var*, std::set<int>> phi_family_ids;
  // var -> its individual [def, last_use] interval (with phi/loop extension), for
  // the precise per-var pairwise interference check in the reuse packer.
  std::map<const Var*, std::pair<int, int>> var_liveness;
};

/**
 * @brief Collect all Var nodes referenced in an expression.
 */
class VarUseCollector : public IRVisitor {
 public:
  std::set<VarPtr> used_vars;

  void VisitExpr_(const VarPtr& var) override {
    used_vars.insert(var);
    IRVisitor::VisitExpr_(var);
  }
};

// ============================================================================
// Top-down target retargeting
//
// Walks nested control flow (ForStmt -> IfStmt -> branches) and, for each
// ForStmt's iter_arg / return_var chain, pushes the canonical target MemRef
// (= initValue's MemRef, established by InitMemRef) down through the yield
// graph. Producers along the chain are rewritten to land on the target:
//
//   - Unconstrained producers (plain matmul, move, etc.): retyped to write
//     directly to the target MemRef.
//   - Pinned producers (set_output_reuses_input, e.g. matmul_acc): we cannot
//     change their output MemRef, so we recurse onto the pinned-input var
//     and try to place that upstream.
//   - IfStmt return_vars: recurse into both branches' yield values with the
//     same target, and retype the return_var itself.
//
// Liveness check: a retype at AssignStmt S is only safe if target's base Ptr
// is not read between S and the enclosing ForStmt's yield.  The check walks
// S's full ancestor chain (innermost out), and at each enclosing SeqStmts
// scans the siblings that execute after the current walk node.  That covers
// reads inside the same branch, reads after a nested IfStmt in the parent
// body, and so on up to the enclosing ForStmt — where the retyped value is
// consumed.
//
// Complexity: the retargeter runs a single IR walk to build the def map
// (O(N)), then one TryRetargetVar per ForStmt iter_arg plus recursion into
// IfStmt branches.  The liveness check scans the tail of each enclosing
// SeqStmts up to the owning ForStmt; in the worst case of a deep linear
// chain this is O(N^2).  In practice body tails are short (matmul/
// accumulator loops have a handful of producers each), so the realised
// cost is well below that bound; we accept the super-linear worst case
// rather than threading a precomputed "bases read at-or-after" index that
// would complicate the pass for no measurable win on typical IR.
// ============================================================================

/// Describes where a TileType Var is defined.
struct VarDef {
  enum Kind { kAssign, kIfReturn, kForReturn, kIterArg, kUnknown };
  Kind kind = kUnknown;
  StmtPtr assign_stmt;    // AssignStmt (for kAssign)
  StmtPtr control_stmt;   // IfStmt/ForStmt (for kIfReturn/kForReturn)
  size_t return_idx = 0;  // index into return_vars_ (for kIfReturn/kForReturn)
  IterArgPtr iter_arg;    // for kIterArg
  // Full chain of enclosing stmts from outermost to innermost (does *not*
  // include the assign_stmt itself).  Populated for kAssign defs and used
  // by the liveness check to walk up through nested IfStmt branches to the
  // enclosing ForStmt's body.
  std::vector<StmtPtr> ancestors;
};

/// Walks the IR once to build the def map, recording every AssignStmt's full
/// enclosing-stmt chain so the liveness check can walk up past IfStmt /
/// ScopeStmt branches into the enclosing loop body.
class DefMapVisitor : public IRVisitor {
 public:
  std::map<VarPtr, VarDef> defs;

  void Run(const StmtPtr& body) { VisitStmt(body); }

 protected:
  // Generic: every stmt becomes an ancestor of its children.  We push on
  // enter and pop on exit so per-def ancestor snapshots are correct.
  void VisitStmt(const StmtPtr& stmt) override {
    if (!stmt) return;
    IRVisitor::VisitStmt(stmt);
  }

  void VisitStmt_(const SeqStmtsPtr& op) override {
    ancestor_stack_.push_back(op);
    for (const auto& s : op->stmts_) VisitStmt(s);
    ancestor_stack_.pop_back();
  }

  void VisitStmt_(const AssignStmtPtr& op) override {
    if (As<TileType>(op->var_->GetType())) {
      VarDef d;
      d.kind = VarDef::kAssign;
      d.assign_stmt = op;
      d.ancestors = ancestor_stack_;
      defs[op->var_] = d;
    }
    if (op->value_) VisitExpr(op->value_);
  }

  void VisitStmt_(const IfStmtPtr& op) override {
    for (size_t i = 0; i < op->return_vars_.size(); ++i) {
      const auto& rv = op->return_vars_[i];
      if (!As<TileType>(rv->GetType())) continue;
      VarDef d;
      d.kind = VarDef::kIfReturn;
      d.control_stmt = op;
      d.return_idx = i;
      defs[rv] = d;
    }
    ancestor_stack_.push_back(op);
    VisitStmt(op->then_body_);
    if (op->else_body_.has_value()) VisitStmt(op->else_body_.value());
    ancestor_stack_.pop_back();
  }

  void VisitStmt_(const ForStmtPtr& op) override {
    for (size_t i = 0; i < op->iter_args_.size(); ++i) {
      auto ia = op->iter_args_[i];
      if (!As<TileType>(ia->GetType())) continue;
      VarDef d;
      d.kind = VarDef::kIterArg;
      d.control_stmt = op;
      d.return_idx = i;
      d.iter_arg = ia;
      defs[std::static_pointer_cast<const Var>(ia)] = d;
    }
    for (size_t i = 0; i < op->return_vars_.size(); ++i) {
      const auto& rv = op->return_vars_[i];
      if (!As<TileType>(rv->GetType())) continue;
      VarDef d;
      d.kind = VarDef::kForReturn;
      d.control_stmt = op;
      d.return_idx = i;
      defs[rv] = d;
    }
    ancestor_stack_.push_back(op);
    VisitStmt(op->body_);
    ancestor_stack_.pop_back();
  }

  // Scope statements (InCore/AutoInCore/Cluster/Hierarchy/Spmd) must also
  // participate in the ancestor chain.  Without them, the liveness walk
  // would jump straight from a scope body's SeqStmts to the enclosing
  // loop body SeqStmts without finding its path-child, and reads after
  // the scope in the enclosing body would be missed.
  void VisitStmt_(const InCoreScopeStmtPtr& op) override { VisitScope(op, op->body_); }
  void VisitStmt_(const AutoInCoreScopeStmtPtr& op) override { VisitScope(op, op->body_); }
  void VisitStmt_(const ClusterScopeStmtPtr& op) override { VisitScope(op, op->body_); }
  void VisitStmt_(const HierarchyScopeStmtPtr& op) override { VisitScope(op, op->body_); }
  void VisitStmt_(const SpmdScopeStmtPtr& op) override { VisitScope(op, op->body_); }

 private:
  template <typename ScopeStmtPtrT>
  void VisitScope(const ScopeStmtPtrT& op, const StmtPtr& body) {
    ancestor_stack_.push_back(op);
    VisitStmt(body);
    ancestor_stack_.pop_back();
  }

  // Outermost-first stack of enclosing stmts during the walk.
  std::vector<StmtPtr> ancestor_stack_;
};

/// Visits a stmt subtree and collects the MemRef base Ptrs of every *read*
/// of a TileType Var.  Writes (the LHS of an AssignStmt) are intentionally
/// excluded; every other stmt/expression kind dispatches through the default
/// IRVisitor traversal, so new stmt types are covered automatically.
class SubtreeReadBaseCollector : public IRVisitor {
 public:
  std::set<const Var*> bases;

  void VisitExpr_(const VarPtr& var) override {
    if (auto tile = GetTileTypeWithMemRef(var->GetType())) {
      bases.insert(GetDefinedMemRef(tile)->base_.get());
    }
    IRVisitor::VisitExpr_(var);
  }

  void VisitStmt_(const AssignStmtPtr& op) override {
    // Skip op->var_ — the LHS is a definition, not a read.
    if (op->value_) VisitExpr(op->value_);
  }
};

inline bool SubtreeReadsBase(const StmtPtr& stmt, const Var* target_base) {
  if (!stmt) return false;
  SubtreeReadBaseCollector c;
  c.VisitStmt(stmt);
  return c.bases.count(target_base) > 0;
}

/// Plans top-down retypes. Produces (old Var -> new Type) map.
class TopDownRetargeter {
 public:
  /// Runs the analysis. Returns map: old VarPtr -> new Type (with target MemRef).
  std::map<VarPtr, TypePtr> Compute(const StmtPtr& func_body) {
    DefMapVisitor def_v;
    def_v.Run(func_body);
    defs_ = std::move(def_v.defs);
    VisitForStmts(func_body);
    return std::move(rewrites_);
  }

 private:
  std::map<VarPtr, VarDef> defs_;
  std::map<VarPtr, TypePtr> rewrites_;
  std::set<VarPtr> visiting_;  // cycle guard

  // Walk IR, calling Propagate for each ForStmt we encounter.
  void VisitForStmts(const StmtPtr& stmt) {
    if (!stmt) return;
    if (auto seq = As<SeqStmts>(stmt)) {
      for (const auto& s : seq->stmts_) VisitForStmts(s);
    } else if (auto for_stmt = As<ForStmt>(stmt)) {
      PropagateFromForStmt(for_stmt);
      VisitForStmts(for_stmt->body_);
    } else if (auto if_stmt = As<IfStmt>(stmt)) {
      VisitForStmts(if_stmt->then_body_);
      if (if_stmt->else_body_.has_value()) VisitForStmts(if_stmt->else_body_.value());
    } else if (auto scope = As<ScopeStmt>(stmt)) {
      VisitForStmts(scope->body_);
    }
  }

  void PropagateFromForStmt(const ForStmtPtr& for_stmt) {
    if (for_stmt->iter_args_.empty()) return;
    auto yield = FindYieldStmt(for_stmt->body_);
    if (!yield) return;

    for (size_t i = 0; i < for_stmt->iter_args_.size() && i < yield->value_.size(); ++i) {
      auto ia = for_stmt->iter_args_[i];
      // If a prior PropagateFromForStmt (an enclosing ForStmt) already retyped
      // this iter_arg, use the planned new TileType so we don't push a stale
      // target down the chain.
      auto ia_var_key = std::static_pointer_cast<const Var>(ia);
      auto rit = rewrites_.find(ia_var_key);
      TypePtr ia_type = (rit != rewrites_.end()) ? rit->second : ia->GetType();
      auto ia_tile = GetTileTypeWithMemRef(ia_type);
      if (!ia_tile) continue;
      auto target_memref = GetDefinedMemRef(ia_tile);
      auto target_memory = ia_tile->GetMemorySpace();

      auto yield_var = AsVarLike(yield->value_[i]);
      if (!yield_var) continue;

      TryRetargetVar(yield_var, target_memref, target_memory);
    }
  }

  /// Current (possibly-rewritten) MemRef base of `var`.
  const Var* CurrentBase(const VarPtr& var) {
    auto it = rewrites_.find(var);
    auto type = (it != rewrites_.end()) ? it->second : var->GetType();
    auto tile = GetTileTypeWithMemRef(type);
    if (!tile) return nullptr;
    return GetDefinedMemRef(tile)->base_.get();
  }

  /// Attempts to rewrite `var`'s MemRef to `target` by walking its producer chain.
  /// Returns true if var already has target MemRef or a rewrite was planned.
  bool TryRetargetVar(const VarPtr& var, const MemRefPtr& target, std::optional<MemorySpace> target_memory) {
    if (CurrentBase(var) == target->base_.get()) return true;  // already aligned
    if (!visiting_.insert(var).second) return false;           // cycle
    struct Guard {
      std::set<VarPtr>* s;
      VarPtr v;
      ~Guard() { s->erase(v); }
    } g{&visiting_, var};

    auto it = defs_.find(var);
    if (it == defs_.end()) return false;
    const auto& def = it->second;

    if (def.kind == VarDef::kAssign) {
      return RetargetAssign(var, def, target, target_memory);
    }
    if (def.kind == VarDef::kIfReturn) {
      return RetargetIfReturn(var, def, target, target_memory);
    }
    if (def.kind == VarDef::kForReturn) {
      return RetargetForReturn(var, def, target, target_memory);
    }
    if (def.kind == VarDef::kIterArg) {
      return RetargetIterArg(var, def, target, target_memory);
    }
    return false;
  }

  /// Retype a Var defined by an AssignStmt.
  bool RetargetAssign(const VarPtr& var, const VarDef& def, const MemRefPtr& target,
                      std::optional<MemorySpace> target_memory) {
    auto assign = As<AssignStmt>(def.assign_stmt);
    INTERNAL_CHECK_SPAN(assign, var->span_) << "Internal error: kAssign VarDef must carry an AssignStmt";
    auto call = As<Call>(assign->value_);
    if (!call) return false;
    const auto& reg = OpRegistry::GetInstance();
    if (!reg.IsRegistered(call->op_->name_)) return false;
    const auto& entry = reg.GetEntry(call->op_->name_);

    auto reuse_idx = entry.GetOutputReusesInputArg();
    if (reuse_idx.has_value()) {
      // Pinned output: can't change this stmt's LHS MemRef; recurse onto pinned input.
      if (*reuse_idx >= call->args_.size()) return false;
      auto input_var = AsVarLike(call->args_[*reuse_idx]);
      if (!input_var) return false;
      if (!TryRetargetVar(input_var, target, target_memory)) return false;
      // Also record that `var`'s MemRef should follow the pinned input to target.
      PlanRewrite(var, target, target_memory);
      return true;
    }

    // Decline retargeting for ops whose output memory is not fully captured
    // by the LHS type alone:
    //   1. View ops (set_output_memory_inherit_input) — output MemRef
    //      inherits the input's view byte_offset_ / size_, which rewriting
    //      with target's full MemRef would silently drop.
    //   2. Ops that encode output memory in a `target_memory` kwarg (e.g.
    //      tile.create / tile.move / tile.load) — retyping the LHS to a
    //      different memory_space would require rewriting the kwarg too.
    //      Same-memory_space retypes are safe: only the MemRef base changes,
    //      while the op's declared output memory stays consistent.
    //   3. Ops registered `not_inplace_safe()` (e.g. tile.mrgsort_format1)
    //      whose implementation requires src buffer != dst buffer.  If any
    //      input of the call already lives on `target_base`, retyping the
    //      output onto the same buffer creates an in-place execution that
    //      the op cannot handle and fails at runtime.
    if (IsOutputMemoryInheritInput(entry)) return false;
    if (HasKwarg(*call, "target_memory") && !TargetMemoryKwargMatches(*call, target_memory)) {
      return false;
    }
    if (!entry.IsInplaceSafe() && CallReadsBase(*call, target->base_.get())) return false;

    // Unconstrained: check liveness, then plan retype.
    if (!IsTargetDeadAtAssign(def, target->base_.get())) return false;
    PlanRewrite(var, target, target_memory);
    return true;
  }

  /// True if any argument of the call is a TileType Var whose MemRef base
  /// is `target_base`.  Used to detect would-be in-place execution before
  /// we retype the output onto the same buffer.
  static bool CallReadsBase(const Call& call, const Var* target_base) {
    SubtreeReadBaseCollector c;
    for (const auto& arg : call.args_) c.VisitExpr(arg);
    return c.bases.count(target_base) > 0;
  }

  /// True when the op is registered with set_output_memory_inherit_input.
  /// Delegates to the shared OpRegistryEntry predicate so passes that reason
  /// about pass-through ops (here and InferTileMemorySpace) agree on the set.
  static bool IsOutputMemoryInheritInput(const OpRegistryEntry& entry) {
    return entry.OutputMemoryInheritsInput();
  }

  static bool HasKwarg(const Call& call, const std::string& key) {
    for (const auto& [k, _] : call.kwargs_) {
      if (k == key) return true;
    }
    return false;
  }

  /// True when the call has a `target_memory` kwarg whose MemorySpace value
  /// equals `target_memory`. Used to allow retypes that change the MemRef
  /// base but keep the memory space the same — the kwarg stays consistent
  /// with the new LHS type without rewriting.  Caller must have verified
  /// `HasKwarg(call, "target_memory")` first; AnyCast errors propagate as
  /// pypto exceptions if the kwarg holds an unexpected type (which would be
  /// an internal IR consistency bug rather than a retype-decline signal).
  static bool TargetMemoryKwargMatches(const Call& call, std::optional<MemorySpace> target_memory) {
    if (!target_memory.has_value()) return false;
    for (const auto& [k, v] : call.kwargs_) {
      if (k != "target_memory") continue;
      return AnyCast<MemorySpace>(v, "kwarg key: target_memory") == *target_memory;
    }
    return false;
  }

  /// Retype an IfStmt return_var: recurse into both branches' yield values.
  bool RetargetIfReturn(const VarPtr& var, const VarDef& def, const MemRefPtr& target,
                        std::optional<MemorySpace> target_memory) {
    auto if_stmt = As<IfStmt>(def.control_stmt);
    if (!if_stmt) return false;
    size_t idx = def.return_idx;

    auto visit_branch = [&](const StmtPtr& body) -> bool {
      auto y = FindYieldStmt(body);
      if (!y || idx >= y->value_.size()) return false;
      auto yv = AsVarLike(y->value_[idx]);
      if (!yv) return false;
      return TryRetargetVar(yv, target, target_memory);
    };

    bool then_ok = visit_branch(if_stmt->then_body_);
    bool else_ok = if_stmt->else_body_.has_value() ? visit_branch(if_stmt->else_body_.value()) : true;
    if (!then_ok || !else_ok) return false;

    PlanRewrite(var, target, target_memory);
    return true;
  }

  /// Retype a ForStmt return_var: recurse into the loop's body-yield value.
  /// Pushing the target onto the body-yield-value also pushes it transitively
  /// onto the iter_arg (via the matmul_acc-style pinned chain) and onto the
  /// init.  After all recursive PlanRewrites, the iter_arg's MemRef will
  /// match the target, and YieldFixupMutator's PatchIterArgsAndReturnVars
  /// finalises iter_arg/return_var TileType updates.
  bool RetargetForReturn(const VarPtr& var, const VarDef& def, const MemRefPtr& target,
                         std::optional<MemorySpace> target_memory) {
    auto for_stmt = As<ForStmt>(def.control_stmt);
    if (!for_stmt) return false;
    size_t idx = def.return_idx;

    auto y = FindYieldStmt(for_stmt->body_);
    if (!y || idx >= y->value_.size()) return false;
    auto yv = AsVarLike(y->value_[idx]);
    if (!yv) return false;
    if (!TryRetargetVar(yv, target, target_memory)) return false;

    // Also retype the corresponding iter_arg's init so the next iteration's
    // loop-carried value reads from the same buffer the body just wrote.
    if (idx < for_stmt->iter_args_.size()) {
      auto ia = for_stmt->iter_args_[idx];
      auto init_var = AsVarLike(ia->initValue_);
      if (init_var && !TryRetargetVar(init_var, target, target_memory)) return false;
      // Plan rewrite for the IterArg itself (its TileType will be updated).
      PlanRewrite(std::static_pointer_cast<const Var>(ia), target, target_memory);
    }

    PlanRewrite(var, target, target_memory);
    return true;
  }

  /// Retype an IterArg: push the target onto its initValue. The iter_arg
  /// itself records a rewrite so RetypeApplier substitutes its TileType in
  /// body references.
  bool RetargetIterArg(const VarPtr& var, const VarDef& def, const MemRefPtr& target,
                       std::optional<MemorySpace> target_memory) {
    auto iter_arg = def.iter_arg;
    if (!iter_arg) return false;
    auto init_var = AsVarLike(iter_arg->initValue_);
    if (!init_var) return false;
    if (!TryRetargetVar(init_var, target, target_memory)) return false;
    PlanRewrite(var, target, target_memory);
    return true;
  }

  /// IterArg keys are stored under their `Var` base via static_pointer_cast;
  /// `rewrites_` keying compares shared_ptr control blocks, not static types,
  /// so `RetypeApplier`'s ForStmt handler can find the IterArg's planned
  /// retype using the same VarPtr-keyed lookup as ordinary AssignStmt LHS
  /// rewrites.  Note: when intermediate recursion partially populates
  /// `rewrites_` and a later step then declines, those entries remain — same
  /// tolerance as `RetargetIfReturn`'s branch-failure path.  This is safe
  /// because `RetypeApplier` only acts on retypes that landed at top-level
  /// callers (the for-stmt's iter_arg in `PropagateFromForStmt`), and the
  /// declined upper-level returns false so the partial chain isn't surfaced
  /// as the canonical retype.
  void PlanRewrite(const VarPtr& var, const MemRefPtr& target, std::optional<MemorySpace> target_memory) {
    auto new_type = CloneTypeWithMemRef(var->GetType(), target, target_memory);
    rewrites_[var] = new_type;
  }

  /// Is target's base Ptr unread between the AssignStmt and the end of its containing body?
  /// Also walks into nested control flow to check for reads there.
  /// Liveness check: walks the AssignStmt's full ancestor chain from innermost
  /// outward and, at every enclosing SeqStmts, scans the siblings that execute
  /// after the AssignStmt for reads of `target_base`.  Walking continues past
  /// IfStmt branches into the parent body so reads that appear after a nested
  /// IfStmt (but still within the enclosing ForStmt's body) are detected.  The
  /// walk stops at the first enclosing ForStmt — reads outside the loop body
  /// cannot observe the retyped value, which is consumed at the loop yield.
  bool IsTargetDeadAtAssign(const VarDef& def, const Var* target_base) {
    if (def.ancestors.empty()) return true;

    // `child_on_path` is the direct descendant of the current ancestor that
    // lies on the walk path toward the AssignStmt.  We update it as we step
    // outward so that, at each SeqStmts level, we can locate it in stmts_.
    StmtPtr child_on_path = def.assign_stmt;

    for (auto it = def.ancestors.rbegin(); it != def.ancestors.rend(); ++it) {
      const auto& anc = *it;

      if (auto seq = As<SeqStmts>(anc)) {
        auto pos = std::find(seq->stmts_.begin(), seq->stmts_.end(), child_on_path);
        if (pos != seq->stmts_.end()) {
          for (++pos; pos != seq->stmts_.end(); ++pos) {
            if (SubtreeReadsBase(*pos, target_base)) return false;
          }
        }
      }

      // Stop once we've scanned the body of the enclosing ForStmt: the
      // retyped value is consumed by that loop's yield, so anything outside
      // the loop cannot observe it.
      if (As<ForStmt>(anc)) return true;

      child_on_path = anc;
    }
    return true;
  }
};

/// Applies planned retypes to the IR.
class RetypeApplier : public IRMutator {
 public:
  explicit RetypeApplier(std::map<VarPtr, TypePtr> rewrites) : rewrites_(std::move(rewrites)) {}

  ExprPtr VisitExpr_(const VarPtr& op) override {
    auto it = var_substitution_.find(op);
    if (it != var_substitution_.end()) return it->second;
    return op;
  }

  ExprPtr VisitExpr_(const IterArgPtr& op) override {
    // Check var_remap_ first.  IRMutator::VisitStmt_(ForStmtPtr) registers
    // OLD → NEW for each iter_arg whose pointer changed when we visited the
    // For's iter_args_ list, then visits the body.  Without this lookup, every
    // body reference to OLD would build its own fresh IterArg (each
    // make_shared returns a distinct pointer), leaving the For's iter_args_
    // and body referencing two different IterArg objects with the same name —
    // a malformed-IR pattern that downstream substitution passes (e.g.,
    // Simplify's Fold B) cannot rewrite consistently.
    auto remap_it = var_remap_.find(op.get());
    if (remap_it != var_remap_.end()) {
      return remap_it->second;
    }

    INTERNAL_CHECK_SPAN(op->initValue_, op->span_) << "IterArg has null initValue";
    // Always recurse into initValue_ first so prior substitutions applied to
    // the AssignStmt that defines initValue (e.g., a tile.create whose LHS
    // we retyped) propagate into the IterArg's initValue field.
    auto new_init_value = VisitExpr(op->initValue_);
    INTERNAL_CHECK_SPAN(new_init_value, op->span_) << "IterArg initValue mutated to null";

    // Look up whether the IterArg itself is being retyped (its TileType
    // changed by a planned PlanRewrite). If so, return a new IterArg with the
    // retyped TileType and the (possibly substituted) initValue.
    auto var_key = std::static_pointer_cast<const Var>(op);
    auto it = var_substitution_.find(var_key);
    if (it != var_substitution_.end()) {
      auto sub_iter_arg = std::dynamic_pointer_cast<const IterArg>(it->second);
      if (sub_iter_arg) {
        return std::make_shared<IterArg>(sub_iter_arg->name_hint_, sub_iter_arg->GetType(), new_init_value,
                                         sub_iter_arg->span_);
      }
      return it->second;
    }
    if (new_init_value.get() != op->initValue_.get()) {
      return std::make_shared<IterArg>(op->name_hint_, op->GetType(), new_init_value, op->span_);
    }
    return op;
  }

  StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
    auto rit = rewrites_.find(op->var_);
    if (rit != rewrites_.end()) {
      auto new_var = std::make_shared<Var>(op->var_->name_hint_, rit->second, op->var_->span_);
      var_substitution_[op->var_] = new_var;
      return IRMutator::VisitStmt_(op);
    }
    if (auto followed = FollowRetargetedViewInput(op)) return followed;
    return IRMutator::VisitStmt_(op);
  }

  /// Re-anchor an inherit-input view (tile.transpose_view / reshape / slice /
  /// set_validshape / tensor.as_layout, ...) onto its input's reused buffer when
  /// the input was retargeted.  The planner never retargets a view's output
  /// directly (RetargetAssign declines OutputMemoryInheritsInput ops), so when its
  /// input (e.g. a tile.load) is retargeted onto a reused buffer, the view's
  /// declared MemRef is left pointing at the now-orphaned original buffer.  Codegen
  /// then emits the view's alloc_tile at the stale address while the load writes the
  /// reused one — the consumer reads an uninitialised buffer (qwen3 b_trans
  /// regression, issue #1776; the sub-region tile.slice case has the same shape).
  /// Re-derive the view's MemRef from its (retargeted) input, mirroring InitMemRef's
  /// ShareMemRefFrom.
  ///
  /// Returns the rewritten AssignStmt, or nullptr when `op` is not an inherit-input
  /// view whose input was retargeted (caller then falls back to the default visit).
  StmtPtr FollowRetargetedViewInput(const AssignStmtPtr& op) {
    auto orig_call = As<Call>(op->value_);
    if (!orig_call || orig_call->args_.empty()) return nullptr;
    const auto& reg = OpRegistry::GetInstance();
    if (!reg.IsRegistered(orig_call->op_->name_) ||
        !reg.GetEntry(orig_call->op_->name_).OutputMemoryInheritsInput()) {
      return nullptr;
    }
    auto in_var = AsVarLike(orig_call->args_[0]);
    if (!in_var) return nullptr;
    auto sub = var_substitution_.find(in_var);
    if (sub == var_substitution_.end()) return nullptr;  // input not retargeted
    auto out_mr_opt = GetTypeMemRef(op->var_->GetType());
    auto in_old_opt = GetTypeMemRef(in_var->GetType());
    auto in_new_opt = GetTypeMemRef(sub->second->GetType());
    if (!out_mr_opt || !in_old_opt || !in_new_opt) return nullptr;
    const auto& out_mr = *out_mr_opt;
    const auto& in_old = *in_old_opt;
    const auto& in_new = *in_new_opt;
    // Use the ORIGINAL (pre-mutation) types: a view inherited its input's buffer
    // iff its output base equalled the input base in the input IR.  This naturally
    // excludes the one inherit-input op that does NOT always inherit — a
    // data-permuting tile.transpose over a sub-region input gets a fresh buffer
    // (output base != input base from the start), so it is left untouched.  Fire
    // only when the view inherited AND the input actually moved to another buffer.
    if (out_mr->base_.get() != in_old->base_.get() || in_new->base_.get() == in_old->base_.get()) {
      return nullptr;
    }
    ExprPtr additional = ComputeViewByteOffset(orig_call, in_var->GetType());
    ExprPtr total_offset = AddByteOffsets(in_new->byte_offset_, additional);
    auto add0 = As<ConstInt>(additional);
    const bool pure_alias = add0 && add0->value_ == 0 && out_mr->size_ == in_new->size_;
    MemRefPtr new_mr =
        pure_alias ? in_new
                   : std::make_shared<MemRef>(in_new->base_, total_offset, out_mr->size_, out_mr->span_);
    // OutputMemoryInheritsInput also requires the view's memory space to follow its
    // input: if the retarget moved across memory spaces, carry the new space too.
    std::optional<MemorySpace> new_memory;
    if (auto in_new_tile = GetTileTypeWithMemRef(sub->second->GetType())) {
      new_memory = in_new_tile->GetMemorySpace();
    }
    auto new_type = CloneTypeWithMemRef(op->var_->GetType(), new_mr, new_memory);
    auto follow_var = std::make_shared<Var>(op->var_->name_hint_, new_type, op->var_->span_);
    var_substitution_[op->var_] = follow_var;
    auto recursed = As<AssignStmt>(IRMutator::VisitStmt_(op));
    INTERNAL_CHECK_SPAN(recursed, op->span_) << "Internal error: AssignStmt visit must yield an AssignStmt";
    return std::make_shared<AssignStmt>(follow_var, recursed->value_, recursed->span_);
  }

  StmtPtr VisitStmt_(const IfStmtPtr& op) override {
    // Pre-register substitutions for any retargeted return_var; the default
    // IRMutator IfStmt handler will then pick them up through VisitExpr_(Var).
    for (const auto& rv : op->return_vars_) {
      auto rit = rewrites_.find(rv);
      if (rit != rewrites_.end()) {
        var_substitution_[rv] = std::make_shared<Var>(rv->name_hint_, rit->second, rv->span_);
      }
    }
    return IRMutator::VisitStmt_(op);
  }

  StmtPtr VisitStmt_(const ForStmtPtr& op) override {
    // Pre-register the IterArg's TileType retype so body references to the
    // IterArg are substituted via VisitExpr_(IterArgPtr). The init value
    // recursion is handled inside VisitExpr_(IterArgPtr).
    for (const auto& ia : op->iter_args_) {
      auto var_key = std::static_pointer_cast<const Var>(ia);
      auto rit = rewrites_.find(var_key);
      if (rit != rewrites_.end()) {
        auto new_iter_arg = std::make_shared<IterArg>(ia->name_hint_, rit->second, ia->initValue_, ia->span_);
        var_substitution_[var_key] = new_iter_arg;
      }
    }
    for (const auto& rv : op->return_vars_) {
      auto rit = rewrites_.find(rv);
      if (rit != rewrites_.end()) {
        var_substitution_[rv] = std::make_shared<Var>(rv->name_hint_, rit->second, rv->span_);
      }
    }
    return IRMutator::VisitStmt_(op);
  }

 private:
  std::map<VarPtr, TypePtr> rewrites_;
  std::map<VarPtr, VarPtr> var_substitution_;
};

/**
 * @brief Full IR tree walker for lifetime analysis.
 *
 * Walks the entire IR tree (including nested control flow bodies) to
 * assign sequential order to all leaf statements and collect variable
 * definitions and uses.
 *
 * Two-phase design:
 *   Phase 1 (Analyze): Walk tree, assign orders, collect raw var uses, record loop boundaries.
 *   Phase 2 (ComputeEffectiveLastUse): For each var, extend lifetime to the end of any
 *           enclosing loop where the var is defined before the loop.
 */
class LifetimeAnalyzer : public IRVisitor {
 public:
  struct Result {
    std::vector<VarPtr> ordered_defs;        // TileType vars in definition order
    std::map<VarPtr, int> var_def_order;     // var -> def order
    std::map<VarPtr, int> var_last_use;      // var -> effective last use (with loop extension)
    std::map<VarPtr, StmtPtr> var_def_stmt;  // var -> defining AssignStmt (for op name extraction)
    // Phi families, keyed per (scf.if, return-slot): each family id groups one
    // phi return_var with its branch-local yield-sources (the same logical value
    // across mutually-exclusive paths), so they may always share a buffer.  A
    // var may belong to several families (e.g. a source yielded at >1 slot)
    // without those families merging, so we store a *set* of ids per var.
    std::map<const Var*, std::set<int>> phi_family_ids;
  };

  Result Analyze(const StmtPtr& func_body) {
    // Phase 1: Walk IR tree
    if (func_body) {
      VisitStmt(func_body);
    }

    // Phase 2: Apply loop-aware lifetime extension
    auto effective_last_use = ComputeEffectiveLastUse();

    return {std::move(ordered_defs_), std::move(var_def_order_), std::move(effective_last_use),
            std::move(var_def_stmt_), std::move(var_family_ids_)};
  }

 protected:
  // Container statements: recurse into children
  void VisitStmt_(const SeqStmtsPtr& op) override {
    for (const auto& stmt : op->stmts_) {
      VisitStmt(stmt);
    }
  }

  void VisitStmt_(const InCoreScopeStmtPtr& op) override { VisitStmt(op->body_); }
  void VisitStmt_(const AutoInCoreScopeStmtPtr& op) override { VisitStmt(op->body_); }
  void VisitStmt_(const ClusterScopeStmtPtr& op) override { VisitStmt(op->body_); }
  void VisitStmt_(const HierarchyScopeStmtPtr& op) override { VisitStmt(op->body_); }
  void VisitStmt_(const SpmdScopeStmtPtr& op) override { VisitStmt(op->body_); }

  // Leaf statements: assign order + collect defs/uses
  void VisitStmt_(const AssignStmtPtr& op) override {
    int order = current_order_++;

    // Record TileType variable definitions
    auto tile_type = As<TileType>(op->var_->GetType());
    if (tile_type) {
      ordered_defs_.push_back(op->var_);
      var_def_order_[op->var_] = order;
      var_def_stmt_[op->var_] = op;
    }

    // Collect variable uses from value expression (for ALL AssignStmt)
    CollectUsesFromExpr(op->value_, order);
  }

  void VisitStmt_(const EvalStmtPtr& op) override {
    int order = current_order_++;
    CollectUsesFromExpr(op->expr_, order);
  }

  void VisitStmt_(const YieldStmtPtr& op) override { CollectUsesFromValues(op->value_); }

  void VisitStmt_(const ReturnStmtPtr& op) override { CollectUsesFromValues(op->value_); }

  // Control flow: recurse into bodies with scope tracking
  void VisitStmt_(const IfStmtPtr& op) override {
    // Visit condition uses at the current order point (before branches)
    CollectUsesFromExpr(op->condition_, current_order_);

    // Visit then body first, else body second (sequential ordering), recording
    // each branch's order range so we can tell branch-local yield-sources apart.
    int then_start = current_order_;
    VisitStmt(op->then_body_);
    int then_end = current_order_ - 1;
    int else_start = -1;
    int else_end = -1;
    if (op->else_body_.has_value()) {
      else_start = current_order_;
      VisitStmt(*op->else_body_);
      else_end = current_order_ - 1;
    }

    // An scf.if return_var (phi) is NOT itself a tracked def -- it is materialized
    // by YieldFixup into a branch yield-source's buffer downstream, and tracking
    // it here would perturb that lowering.  But reads of the phi *after* the if
    // keep that buffer live; if we drop them (RecordRawUse early-returns on
    // untracked vars), the buffer's live range collapses at the yield and a later
    // temporary is packed onto the still-live phi value (merge_norm mi/li/oi,
    // #1821).  So we redirect the phi's reads onto its *branch-local* yield-
    // sources (the tiles actually holding the value), recorded in
    // `phi_local_sources_`.  A source defined OUTSIDE the branch (loaded before
    // the if and live elsewhere) is left alone -- it genuinely conflicts and
    // YieldFixup moves it into the phi buffer.
    //
    // Extending both branch sources to the phi's last read makes the two
    // mutually-exclusive sources overlap, so we also tag them with a fresh
    // per-slot family id; the reuse packer exempts intra-family pairs from the
    // interference check.  Ids are *per return slot* (not a transitive union): a
    // source shared across slots joins several families without merging them, so
    // two distinct phi results are never exempted.
    auto then_yield = transform_utils::GetLastYieldStmt(op->then_body_);
    auto else_yield =
        op->else_body_.has_value() ? transform_utils::GetLastYieldStmt(*op->else_body_) : nullptr;
    auto branch_local = [this](const VarPtr& v, int start, int end) {
      if (start < 0) return false;
      auto it = var_def_order_.find(v);
      return it != var_def_order_.end() && it->second >= start && it->second <= end;
    };
    for (size_t i = 0; i < op->return_vars_.size(); ++i) {
      const auto& rv = op->return_vars_[i];
      auto tile_type = As<TileType>(rv->GetType());
      if (!tile_type || !tile_type->memref_.has_value()) continue;
      int family_id = next_family_id_++;
      std::vector<VarPtr> local_sources;
      std::set<const Var*> seen_phi;
      // Flatten a nested-if phi source down to its terminal tracked tiles, and
      // tag THOSE with this (outer) slot's family id -- otherwise the terminals
      // only carry the inner slot's id and `overlap_blocks_sharing` (which sees
      // terminal vars, not the untracked phi wrapper) would reject safe nested
      // mutually-exclusive sharing.
      std::function<void(const VarPtr&, int, int)> add_source_var = [&](const VarPtr& src, int start,
                                                                        int end) {
        auto pit = phi_local_sources_.find(src.get());
        if (pit != phi_local_sources_.end()) {
          if (!seen_phi.insert(src.get()).second) return;
          for (const auto& nested : pit->second) add_source_var(nested, start, end);
          return;
        }
        if (!branch_local(src, start, end)) return;
        local_sources.push_back(src);
        var_family_ids_[src.get()].insert(family_id);
      };
      auto add_source = [&](const ExprPtr& val, int start, int end) {
        if (auto src = AsVarLike(val)) add_source_var(src, start, end);
      };
      if (then_yield && i < then_yield->value_.size())
        add_source(then_yield->value_[i], then_start, then_end);
      if (else_yield && i < else_yield->value_.size())
        add_source(else_yield->value_[i], else_start, else_end);
      if (!local_sources.empty()) {
        var_family_ids_[rv.get()].insert(family_id);
        phi_local_sources_[rv.get()] = std::move(local_sources);
      }
    }
  }

  void VisitStmt_(const ForStmtPtr& op) override {
    ProcessLoopBody(op->iter_args_, op->body_);
    RegisterReturnVars(op->iter_args_, op->return_vars_);
  }

  void VisitStmt_(const WhileStmtPtr& op) override {
    ProcessLoopBody(op->iter_args_, op->body_);
    RegisterReturnVars(op->iter_args_, op->return_vars_);
  }

 private:
  int current_order_ = 0;

  // Loop boundaries recorded during Phase 1
  struct LoopBoundary {
    int start_order;
    int end_order;
  };
  std::vector<LoopBoundary> loop_scopes_;

  // Variable tracking (Phase 1)
  std::vector<VarPtr> ordered_defs_;
  std::map<VarPtr, int> var_def_order_;
  std::map<VarPtr, int> var_raw_last_use_;  // Raw last use (without loop extension)
  std::map<VarPtr, StmtPtr> var_def_stmt_;

  // Maps ForStmt/WhileStmt return_vars to their corresponding initValue vars.
  // YieldFixup will place return_vars into initValue's MemRef buffer,
  // so any post-loop use of a return_var must extend the initValue's lifetime.
  std::map<VarPtr, VarPtr> return_var_to_init_var_;

  // Per-slot phi families: var -> set of family ids (see VisitStmt_(IfStmtPtr)).
  std::map<const Var*, std::set<int>> var_family_ids_;
  int next_family_id_ = 0;

  // scf.if phi return_var -> its branch-local yield-sources, so RecordRawUse can
  // redirect phi reads onto the tiles that actually hold the value.
  std::map<const Var*, std::vector<VarPtr>> phi_local_sources_;

  void CollectUsesFromExpr(const ExprPtr& expr, int use_order) {
    if (!expr) return;
    VarUseCollector collector;
    collector.VisitExpr(expr);
    for (const auto& var : collector.used_vars) {
      RecordRawUse(var, use_order);
    }
  }

  void CollectUsesFromValues(const std::vector<ExprPtr>& values) {
    int order = current_order_++;
    for (const auto& val : values) {
      CollectUsesFromExpr(val, order);
    }
  }

  void ProcessLoopBody(const std::vector<IterArgPtr>& iter_args, const StmtPtr& body) {
    for (const auto& iter_arg : iter_args) {
      if (iter_arg->initValue_) {
        CollectUsesFromExpr(iter_arg->initValue_, current_order_);
      }
    }
    int loop_start = current_order_;
    VisitStmt(body);
    int loop_end = current_order_ - 1;
    loop_scopes_.push_back({loop_start, loop_end});
  }

  void RegisterReturnVars(const std::vector<IterArgPtr>& iter_args, const std::vector<VarPtr>& return_vars) {
    size_t count = std::min(return_vars.size(), iter_args.size());
    for (size_t i = 0; i < count; ++i) {
      const auto& rv = return_vars[i];
      if (!As<TileType>(rv->GetType())) continue;

      // Map return_var -> initValue var so that post-loop uses of the
      // return_var extend the initValue's lifetime (not the return_var's).
      // We do NOT register return_vars in ordered_defs_ -- they must not
      // participate in sharing group computation, which would inflate
      // group lifetimes and block unrelated reuse opportunities.
      auto init_var = As<Var>(iter_args[i]->initValue_);
      if (init_var && var_def_order_.count(init_var)) {
        return_var_to_init_var_[rv] = init_var;
      }
    }
  }

  void RecordRawUse(const VarPtr& var, int use_order) {
    // If var is an scf.if phi return_var, redirect the use to its branch-local
    // yield-sources (the tiles holding the value), keeping their buffers live to
    // the phi's last read.  Sources may themselves be phis (chained scf.if), so
    // recurse.
    auto pit = phi_local_sources_.find(var.get());
    if (pit != phi_local_sources_.end()) {
      for (const auto& src : pit->second) RecordRawUse(src, use_order);
      return;
    }

    // If var is a loop return_var, redirect the use to its initValue var.
    // YieldFixup will alias the return_var to the initValue's MemRef,
    // so keeping the initValue live prevents premature buffer reuse.
    auto it = return_var_to_init_var_.find(var);
    const VarPtr& target = (it != return_var_to_init_var_.end()) ? it->second : var;

    if (!var_def_order_.count(target)) {
      return;
    }
    // operator[] default-inserts 0 for missing keys; use_order is always >= 0.
    var_raw_last_use_[target] = std::max(var_raw_last_use_[target], use_order);
  }

  /**
   * @brief Phase 2: Compute effective last use with loop-aware extension.
   *
   * For each variable, if it's used inside a loop body and defined before
   * that loop, extend its lifetime to the end of the loop. This handles
   * the case where the loop re-executes and the variable is needed again.
   */
  [[nodiscard]] std::map<VarPtr, int> ComputeEffectiveLastUse() const {
    std::map<VarPtr, int> effective;

    for (const auto& var : ordered_defs_) {
      int def_order = var_def_order_.at(var);
      int raw_last = var_raw_last_use_.count(var) ? var_raw_last_use_.at(var) : def_order;
      int extended = raw_last;

      // Check all loop scopes: if var is defined before the loop and
      // has any use inside the loop [start, end], extend to loop end.
      for (const auto& loop : loop_scopes_) {
        if (def_order < loop.start_order && raw_last >= loop.start_order) {
          // Variable is defined before loop and used inside it
          extended = std::max(extended, loop.end_order);
        }
      }

      effective[var] = extended;
    }

    return effective;
  }
};

/**
 * @brief Compute lifetime intervals by walking the full IR tree.
 *
 * Walks ALL statements
 * including those inside nested control flow (IfStmt/ForStmt/WhileStmt bodies).
 */
LifetimeAnalysisResult ComputeLifetimes(const StmtPtr& func_body) {
  std::vector<LifetimeInterval> lifetimes;

  // Step 1: Walk full IR tree to collect variable defs, uses, and ordering
  LifetimeAnalyzer analyzer;
  auto result = analyzer.Analyze(func_body);

  if (result.ordered_defs.empty()) {
    return {lifetimes, {}};
  }

  // Step 2: Build MemRef sharing groups (keyed by base_ Ptr identity)
  std::map<const Var*, std::vector<VarPtr>> memref_groups;
  for (const auto& var : result.ordered_defs) {
    auto tile_type = As<TileType>(var->GetType());
    if (tile_type && tile_type->memref_.has_value()) {
      const Var* base_ptr = tile_type->memref_.value()->base_.get();
      memref_groups[base_ptr].push_back(var);
    }
  }

  std::map<VarPtr, std::vector<VarPtr>> var_sharing_groups;
  for (const auto& [base_ptr, vars] : memref_groups) {
    if (vars.size() > 1) {
      for (const auto& var : vars) {
        var_sharing_groups[var] = vars;
      }
      LOG_DEBUG << "MemRef sharing group: " << vars.size() << " variables share same MemRef";
    }
  }

  // Step 3: Compute lifetime intervals (with MemRef sharing group merging)
  std::set<VarPtr> processed_vars;

  for (const auto& var : result.ordered_defs) {
    if (processed_vars.count(var)) {
      continue;
    }

    auto tile_type = As<TileType>(var->GetType());
    if (!tile_type || !tile_type->memref_.has_value()) {
      continue;
    }

    const auto& memref = tile_type->memref_.value();

    std::vector<VarPtr> sharing_group;
    if (var_sharing_groups.count(var)) {
      sharing_group = var_sharing_groups[var];
    } else {
      sharing_group = {var};
    }

    // Compute MERGED lifetime for all variables in the sharing group
    int min_def_point = INT_MAX;
    int max_last_use = INT_MIN;

    for (const auto& group_var : sharing_group) {
      int def_point = result.var_def_order.count(group_var) ? result.var_def_order.at(group_var) : 0;
      int last_use = result.var_last_use.count(group_var) ? result.var_last_use.at(group_var) : def_point;

      LOG_DEBUG << "Variable " << group_var->name_hint_ << " def=" << def_point << " last_use=" << last_use;

      min_def_point = std::min(min_def_point, def_point);
      max_last_use = std::max(max_last_use, last_use);
    }

    LifetimeInterval interval;
    interval.variable = sharing_group[0];
    interval.def_point = min_def_point;
    interval.last_use_point = max_last_use;
    auto representative_tile_type = As<TileType>(sharing_group[0]->GetType());
    INTERNAL_CHECK_SPAN(representative_tile_type != nullptr, sharing_group[0]->span_)
        << "Expected TileType for reuse interval";
    auto memory_space = representative_tile_type->GetMemorySpace();
    INTERNAL_CHECK_SPAN(memory_space.has_value(), sharing_group[0]->span_)
        << "TileType with MemRef must have memory_space for reuse analysis";
    interval.memory_space = *memory_space;
    interval.size = memref->size_;

    lifetimes.push_back(interval);

    for (const auto& group_var : sharing_group) {
      processed_vars.insert(group_var);
    }

    LOG_DEBUG << "Lifetime for sharing group (representative: " << sharing_group[0]->name_hint_
              << ", size: " << sharing_group.size() << "): [" << min_def_point << ", " << max_last_use << "]"
              << " space=" << static_cast<int>(interval.memory_space) << " size=" << interval.size;
  }

  // Per-var individual liveness (def + effective last use, including phi/loop
  // extension) for the packer's precise pairwise interference check.
  std::map<const Var*, std::pair<int, int>> var_liveness;
  for (const auto& [var, def_point] : result.var_def_order) {
    int last_use = result.var_last_use.count(var) ? result.var_last_use.at(var) : def_point;
    var_liveness[var.get()] = {def_point, last_use};
  }

  return {lifetimes, var_sharing_groups, std::move(result.phi_family_ids), std::move(var_liveness)};
}

// NOTE: The former tile-type reuse-compatibility gate (AreTileTypesCompatible)
// has been removed.  PTO codegen binds a per-var alloc_tile to each tile, so two
// tiles that share a physical MemRef can legally carry different shapes, dtypes,
// or TileView attributes (each alloc_tile aliases the same base with its own
// static signature).  The only genuine hazard was an op that reads an operand
// while writing its output in place onto that operand's buffer; that is now
// handled precisely by not_inplace_safe() and the per-operand
// forbid_output_alias() markers (see ForbidAliasCollector below), rather than by
// a coarse whole-tile shape/dtype match.

/**
 * @brief Check if two lifetimes overlap.
 *
 * Uses <= to allow "touching" lifetimes (last_use == def_point) to share
 * buffers: within a single statement, inputs are consumed before outputs
 * are produced.
 */
static bool LifetimesOverlap(const LifetimeInterval& a, const LifetimeInterval& b) {
  return !(a.last_use_point <= b.def_point || b.last_use_point <= a.def_point);
}

// ---------------------------------------------------------------------------
// Ascend910B split-AIV  load + tpop_from_aic  in-place hazard
// ---------------------------------------------------------------------------
//
// On Ascend910B AIV functions with a non-None SplitMode, an op that consumes
// BOTH a tile.load result (or a legal-view descendant of one) AND a
// tile.tpop_from_aic value must not write its output into the same physical
// buffer as the load result it reads.  Letting the writer's output reuse that
// buffer (an in-place touch) yields silently wrong results on this hardware.
//
// MemoryReuse owns every buffer-coalescing decision, so the cleanest fix is to
// never create the hazardous sharing in the first place.  This used to be
// undone after the fact by a dedicated LegalizePTOBufferReuse split pass; the
// guard below folds that responsibility into the reuse decision.
//
// `HazardInputs` is collected in a single forward IR walk:
//   - load_derived: vars produced by tile.load or by a legal view op chained
//     from a load (these alias the load's physical buffer).
//   - reads_tpop:   vars whose defining op consumes a tile.tpop_from_aic value.
// Membership is keyed on Var identity and checked against the reuse-decision
// representatives (a sharing group's earliest-defined member, which for a
// writer is the writer's own output and for a load is the load itself).

/// Op names whose output aliases the input MemRef and lowers to a PTO view
/// instruction — kept in sync with the PTO buffer-reuse view allowlist.
static bool IsLegalTileViewOp(const std::string& op_name) {
  return op_name == "tile.reshape" || op_name == "tile.extract" || op_name == "tile.slice" ||
         op_name == "tile.fillpad" || op_name == "tile.fillpad_inplace" || op_name == "tile.transpose_view" ||
         op_name == "tensor.slice";
}

struct HazardInputs {
  std::unordered_set<const Var*> load_derived;  ///< tile.load outputs + view descendants
  std::unordered_set<const Var*> reads_tpop;    ///< vars whose def consumes a tpop_from_aic value
};

class HazardInputCollector : public IRVisitor {
 public:
  void VisitStmt_(const AssignStmtPtr& op) override {
    if (GetTileTypeWithMemRef(op->var_->GetType())) {
      if (auto call = As<Call>(op->value_)) {
        const std::string op_name = call->op_ ? call->op_->name_ : std::string();
        std::vector<const Var*> input_vars;
        for (const auto& arg : call->args_) {
          if (auto v = As<Var>(arg)) input_vars.push_back(v.get());
        }
        // load_derived closure: defs precede uses in program order, so a view's
        // source is already classified by the time we reach the view.
        if (op_name == "tile.load") {
          inputs_.load_derived.insert(op->var_.get());
        } else if (IsLegalTileViewOp(op_name)) {
          for (const Var* in : input_vars) {
            if (inputs_.load_derived.count(in) != 0) {
              inputs_.load_derived.insert(op->var_.get());
              break;
            }
          }
        }
        if (op_name == "tile.tpop_from_aic") tpop_vars_.insert(op->var_.get());
        for (const Var* in : input_vars) {
          if (tpop_vars_.count(in) != 0) {
            inputs_.reads_tpop.insert(op->var_.get());
            break;
          }
        }
      }
    }
    IRVisitor::VisitStmt_(op);
  }

  HazardInputs Take() { return std::move(inputs_); }

 private:
  HazardInputs inputs_;
  std::unordered_set<const Var*> tpop_vars_;
};

// ---------------------------------------------------------------------------
// "Output must not alias this input" map
// ---------------------------------------------------------------------------
//
// An op may forbid its output from sharing a buffer with one or more of its
// input operands, because the hardware reads those inputs while writing the
// output (aliasing them corrupts results).  Two registry declarations feed the
// same per-output forbidden-input set, unified here:
//
//   * not_inplace_safe()        -> the op cannot run src == dst at all, so the
//                                  output must not alias ANY of its inputs
//                                  (e.g. tile.recip, tile.mrgsort_format1).
//   * forbid_output_alias(i)    -> the op is in-place-safe w.r.t. its value
//                                  operands but reads a *specific* operand
//                                  while writing dst (e.g. tile.sel's predicate
//                                  mask / tmp scratch), so dst must not alias
//                                  that one operand's buffer.
//
// Because every read input of `op->var_`'s defining call appears in `args_`,
// "forbid all inputs" is exactly "forbid every args_ entry" — so this single
// walk records, per output tile var, the input tile vars its op forbids the
// output from aliasing.  MemoryReuse then refuses to place the output on any of
// those inputs' buffers.  Distinct from the load+tpop hazard (backend-gated) —
// this is op-semantic and always collected.
// Maps each output tile Var to the input operand Vars its defining op forbids
// the output from sharing a buffer with.  Enforcement resolves each operand to
// the *physical buffer* it ends up on (following both reuse-map reassignment and
// VIEW inheritance) and blocks the output from landing there — see the use site.
using ForbidAliasMap = std::map<const Var*, std::vector<VarPtr>>;

// Resolve a tile Var to its MemRef base pointer (nullptr if it is not a tile or
// has no MemRef yet).  View ops share their source's base, so a view and its
// source resolve to the same pointer here.
static const Var* TileMemRefBase(const VarPtr& v) {
  auto t = As<TileType>(v->GetType());
  if (t && t->memref_.has_value() && t->memref_.value()->base_) return t->memref_.value()->base_.get();
  return nullptr;
}

class ForbidAliasCollector : public IRVisitor {
 public:
  // `sharing_groups` (from ComputeLifetimes) maps every var that shares a MemRef
  // to its group. IdentifyReuseOpportunities keys lifetimes on each group's
  // *representative* (`sharing_group[0]`), so the forbidden set must be keyed on
  // the representative of the defining op's output too — otherwise a forbidden
  // op whose output is a non-representative group member (e.g. its result is
  // also viewed/reshaped elsewhere) would never be looked up. Forbidden input
  // *values* need no such mapping: enforcement compares physical MemRef bases,
  // which every group member already shares.
  explicit ForbidAliasCollector(const std::map<VarPtr, std::vector<VarPtr>>& sharing_groups) {
    for (const auto& [var, group] : sharing_groups) {
      if (!group.empty()) member_to_rep_[var.get()] = group[0].get();
    }
  }

  void VisitStmt_(const AssignStmtPtr& op) override {
    if (auto call = As<Call>(op->value_); call && call->op_) {
      const auto& reg = OpRegistry::GetInstance();
      if (reg.IsRegistered(call->op_->name_)) {
        const auto& entry = reg.GetEntry(call->op_->name_);
        auto rep_it = member_to_rep_.find(op->var_.get());
        const Var* out_key = rep_it != member_to_rep_.end() ? rep_it->second : op->var_.get();
        auto forbid_arg = [&](size_t i) {
          if (i < call->args_.size()) {
            if (auto v = AsVarLike(call->args_[i])) forbidden_[out_key].push_back(v);
          }
        };
        if (!entry.IsInplaceSafe()) {
          // src != dst required: the output must not alias any input operand.
          for (size_t i = 0; i < call->args_.size(); ++i) forbid_arg(i);
        } else {
          for (size_t i : entry.ForbidOutputAliasArgs()) forbid_arg(i);
        }
        // A dtype-widening cast (output element wider than its input) cannot run
        // in place: element i is read at i*in_bytes but written at i*out_bytes,
        // so with out_bytes > in_bytes the write cursor outruns the read cursor
        // and clobbers input elements not yet converted -> corrupt results.
        // Narrowing / same-width casts are in-place-safe and keep the cross-dtype
        // reuse the removed gate enables, so forbid only the widening direction.
        if (call->op_->name_ == "tile.cast" && !call->args_.empty()) {
          auto out_t = As<TileType>(op->var_->GetType());
          auto in_t = As<TileType>(call->args_[0]->GetType());
          if (out_t && in_t && out_t->dtype_.GetBit() > in_t->dtype_.GetBit()) forbid_arg(0);
        }
        // tile.transpose is registered not_inplace_safe(), so its output is
        // already forbidden from aliasing any input above (pto.ttrans writes
        // dst directly from src on the scalar path — dst == src corrupts).
      }
    }
    IRVisitor::VisitStmt_(op);
  }

  ForbidAliasMap Take() { return std::move(forbidden_); }

 private:
  ForbidAliasMap forbidden_;
  std::map<const Var*, const Var*> member_to_rep_;  ///< sharing-group member -> representative
};

/// True only for Ascend910B AIV split-mode functions, which need the load +
/// tpop_from_aic in-place hazard guard.  All other backends / function kinds
/// reuse buffers freely.  Defensive against unit-test contexts that run
/// MemoryReuse without a configured backend.
bool NeedsLoadTpopHazardGuard(const FunctionPtr& func) {
  if (func->func_type_ != FunctionType::AIV) return false;
  if (!backend::BackendConfig::IsConfigured()) return false;
  const auto* ctx = PassContext::Current();
  if (ctx == nullptr) return false;
  if (!ctx->GetBackendHandler()->RequiresSplitLoadTpopWorkaround()) return false;
  const auto split_mode = func->GetSplitMode();
  return split_mode.has_value() && *split_mode != SplitMode::None;
}

/**
 * @brief Identify memory reuse opportunities from lifetime intervals.
 *
 * Global first-fit-decreasing packing: within each memory space, intervals are
 * placed **largest-first**, and every later interval joins the first existing
 * buffer all of whose members it can share with (see `can_share` below).  A
 * buffer's allocated size is fixed by its first (largest) member, so admitting a
 * smaller member afterwards is free — which makes first-fit cost-optimal for
 * this objective and, crucially, lets a *later, larger* interval host an
 * *earlier, smaller* one.  The former definition-order greedy had a
 * one-directional size gate (`source.size >= target.size`) so it could only let
 * a smaller interval reuse an earlier, larger buffer; two lifetime-disjoint L0
 * cube-input tiles whose small one was defined first were never coalesced.
 *
 * The representative (each buffer's first/largest member) is the MemRef every
 * other member is rebased onto downstream; its alloc dominates the whole
 * function (InitMemRef hoists every tile.alloc to the function-body head), so a
 * representative defined *after* some of its members is safe.  Because the
 * packer no longer processes intervals in program order, every pairwise gate
 * below (load+tpop hazard, no-alias) is checked in **both directions**.
 *
 * @param hazard  Ascend910B load + tpop_from_aic guard inputs.  Empty when the
 *                guard is inactive (non-910B / non-split-AIV), in which case the
 *                hazard check is a no-op and reuse behaviour is unchanged.
 * @param forbid_alias  Per-output forbidden-input map (ForbidAliasCollector):
 *                an op may forbid its output from sharing a buffer with specific
 *                input operands it reads while writing the output in place.
 *
 * Complexity: O(M log M) sort + O(M^2) worst-case pairwise checks over the M
 * tile intervals (M a subset of the N IR nodes) — same class as the prior
 * greedy implementation.
 */
std::map<VarPtr, VarPtr> IdentifyReuseOpportunities(
    const std::vector<LifetimeInterval>& lifetimes, const HazardInputs& hazard,
    const ForbidAliasMap& forbid_alias, const std::map<const Var*, std::set<int>>& phi_family_ids,
    const std::map<VarPtr, std::vector<VarPtr>>& sharing_groups,
    const std::map<const Var*, std::pair<int, int>>& var_liveness) {
  std::map<VarPtr, VarPtr> reuse_map;

  // Members of a sharing group (the vars that already physically share one base).
  // Returns a pointer (or nullptr) to avoid copying the vector in the O(M^2) loop.
  auto group_members = [&sharing_groups](const VarPtr& rep) -> const std::vector<VarPtr>* {
    auto it = sharing_groups.find(rep);
    return it != sharing_groups.end() ? &it->second : nullptr;
  };
  auto ids_intersect = [](const std::set<int>& x, const std::set<int>& y) {
    auto ix = x.begin();
    auto iy = y.begin();
    while (ix != x.end() && iy != y.end()) {
      if (*ix < *iy) {
        ++ix;
      } else if (*iy < *ix) {
        ++iy;
      } else {
        return true;
      }
    }
    return false;
  };
  // Two vars are in the same phi family (the same logical value across
  // mutually-exclusive scf.if paths for one return slot) -> they never conflict.
  auto same_phi_family_var = [&](const Var* x, const Var* y) {
    auto ix = phi_family_ids.find(x);
    auto iy = phi_family_ids.find(y);
    return ix != phi_family_ids.end() && iy != phi_family_ids.end() && ids_intersect(ix->second, iy->second);
  };
  // Precise per-var overlap (touching allowed: last_use == def is fine).
  auto var_overlap = [&](const Var* x, const Var* y) {
    auto ix = var_liveness.find(x);
    auto iy = var_liveness.find(y);
    if (ix == var_liveness.end() || iy == var_liveness.end()) return true;  // unknown -> conservative
    return !(ix->second.second <= iy->second.first || iy->second.second <= ix->second.first);
  };
  // Decide whether a group-interval overlap between two intervals actually
  // blocks sharing.  This is invoked ONLY when their merged intervals overlap.
  //
  // To keep behaviour identical to the prior coarse gate for everything that is
  // not phi-related, we block unless a phi family makes the overlap spurious:
  // some cross-group var pair must be phi-family-related (the same value across
  // mutually-exclusive scf.if paths), AND no *non*-family cross-group pair may
  // truly overlap.  The latter still catches a real conflict hidden behind a
  // branch-local alias of an outside-live buffer (e.g. `result = seed` aliasing
  // a `seed` that overlaps the sibling branch).
  auto overlap_blocks_sharing = [&](const LifetimeInterval& a, const LifetimeInterval& b) {
    const auto* ga = group_members(a.variable);
    const auto* gb = group_members(b.variable);
    const std::vector<VarPtr> sa = ga ? std::vector<VarPtr>{} : std::vector<VarPtr>{a.variable};
    const std::vector<VarPtr> sb = gb ? std::vector<VarPtr>{} : std::vector<VarPtr>{b.variable};
    bool family_pair = false;
    for (const auto& x : (ga ? *ga : sa)) {
      for (const auto& y : (gb ? *gb : sb)) {
        if (same_phi_family_var(x.get(), y.get())) {
          family_pair = true;
        } else if (var_overlap(x.get(), y.get())) {
          return true;  // genuine non-family conflict
        }
      }
    }
    return !family_pair;  // no phi relation -> block exactly as the coarse gate did
  };

  // Maps a tile's *original* MemRef base to the base its buffer is coalesced
  // onto by a committed reuse decision.  A reuse retargets the reused var (and
  // every VIEW that inherited its base) from its own base onto the
  // representative's base; resolve_base() chases that chain so the no-alias
  // guard sees a forbidden operand's *physical* buffer even when the operand is
  // a view whose own base is now stale.  Populated as decisions are committed.
  std::map<const Var*, const Var*> base_remap;
  std::function<const Var*(const Var*)> resolve_base = [&](const Var* b) -> const Var* {
    while (b != nullptr) {
      auto it = base_remap.find(b);
      if (it == base_remap.end() || it->second == b) break;
      b = it->second;
    }
    return b;
  };

  // The physical buffer base a tile var currently occupies: follow any committed
  // reuse onto its representative, then chase base coalescing / VIEW inheritance.
  auto physical_base = [&](const VarPtr& v) -> const Var* {
    VarPtr root = v;
    while (reuse_map.count(root)) root = reuse_map.at(root);
    return resolve_base(TileMemRefBase(root));
  };

  // Ascend910B split-AIV hazard (directional): `writer` consumes a
  // tile.tpop_from_aic value and is defined exactly where the load-derived
  // `input` is last read, so placing writer's output on input's buffer forms the
  // in-place load+tpop touch.  `hazard` is empty off-910B, so this is a no-op.
  auto hazard_blocks = [&hazard](const LifetimeInterval& writer, const LifetimeInterval& input) {
    return writer.def_point == input.last_use_point && hazard.reads_tpop.count(writer.variable.get()) != 0 &&
           hazard.load_derived.count(input.variable.get()) != 0;
  };

  // No-alias guard (directional): `writer`'s defining op forbids its output from
  // sharing a buffer with one or more input operands — either because the op is
  // not_inplace_safe (forbids ALL inputs, e.g. tile.recip) or because it marks a
  // specific operand via forbid_output_alias (e.g. tile.sel's mask/tmp, read by
  // the TSEL intrinsic while writing dst).  Both feed `forbid_alias`.  Block if
  // any forbidden operand resolves to the physical buffer `occupant` sits on.
  // Each operand's buffer is found by following its reuse chain to its owner and
  // taking the MemRef base — catching both an operand reassigned by an earlier
  // reuse decision AND one occupying the buffer via VIEW inheritance (a reshape
  // / slice / extract shares its source's base with no reuse-map entry).
  auto forbid_blocks = [&](const LifetimeInterval& writer, const LifetimeInterval& occupant) {
    auto fa_it = forbid_alias.find(writer.variable.get());
    if (fa_it == forbid_alias.end()) return false;
    const Var* occ_base = physical_base(occupant.variable);
    if (occ_base == nullptr) return false;
    for (const VarPtr& operand : fa_it->second) {
      if (physical_base(operand) == occ_base) return true;
    }
    return false;
  };

  // Can `cand` join a single physical buffer that already holds `member`?
  // Lifetimes must not overlap (touching is allowed: a buffer's reader is
  // consumed before the writer at the same statement produces its output), and
  // neither directional gate may block.  No tile-type / size check is needed:
  // PTO binds a per-var alloc_tile so differing shapes/dtypes legally alias one
  // base, and largest-first ordering guarantees the buffer is sized to its
  // representative (no member is ever larger than the buffer it joins).
  auto can_share = [&](const LifetimeInterval& cand, const LifetimeInterval& member) {
    // Group-interval overlap is a fast reject; when it fires, fall back to the
    // precise per-var check so mutually-exclusive / same-value phi-family tiles
    // may still share while a genuine conflict (incl. one hidden behind a
    // branch-local alias of an outside-live buffer) is still caught.
    if (LifetimesOverlap(cand, member) && overlap_blocks_sharing(cand, member)) return false;
    if (hazard_blocks(cand, member) || hazard_blocks(member, cand)) return false;
    if (forbid_blocks(cand, member) || forbid_blocks(member, cand)) return false;
    return true;
  };

  // Group interval indices by memory space — reuse only happens within a space.
  std::map<MemorySpace, std::vector<size_t>> by_space;
  for (size_t i = 0; i < lifetimes.size(); ++i) {
    by_space[lifetimes[i].memory_space].push_back(i);
  }

  for (auto& [space, indices] : by_space) {
    (void)space;
    // Largest-first; ties broken by definition order for determinism and so that
    // equal-size workloads reproduce the prior definition-order grouping.
    std::stable_sort(indices.begin(), indices.end(), [&lifetimes](size_t a, size_t b) {
      if (lifetimes[a].size != lifetimes[b].size) return lifetimes[a].size > lifetimes[b].size;
      return lifetimes[a].def_point < lifetimes[b].def_point;
    });

    // Each buffer is a list of interval indices sharing one MemRef; element 0 is
    // the representative (largest, earliest on ties) every member is rebased to.
    std::vector<std::vector<size_t>> buffers;
    for (size_t idx : indices) {
      const auto& cand = lifetimes[idx];
      bool placed = false;
      for (auto& buf : buffers) {
        bool fits = true;
        for (size_t member_idx : buf) {
          if (!can_share(cand, lifetimes[member_idx])) {
            fits = false;
            break;
          }
        }
        if (!fits) continue;
        const VarPtr& representative = lifetimes[buf.front()].variable;
        reuse_map[cand.variable] = representative;  // member -> representative
        // Record the base coalescing so resolve_base() can chase a view whose
        // owning tile is reused onto the representative's buffer (no-alias guard).
        if (const Var* cb = TileMemRefBase(cand.variable)) {
          if (const Var* rb = physical_base(representative)) {
            if (cb != rb) base_remap[cb] = rb;
          }
        }
        buf.push_back(idx);
        placed = true;
        LOG_DEBUG << "Variable " << cand.variable->name_hint_ << " reuses representative "
                  << representative->name_hint_ << " (lifetime [" << cand.def_point << ", "
                  << cand.last_use_point << "])";
        break;
      }
      if (!placed) buffers.push_back({idx});
    }
  }

  return reuse_map;
}

/**
 * @brief Apply MemRef sharing to the statement tree
 */
StmtPtr ApplyMemRefSharing(const StmtPtr& stmt, const std::map<VarPtr, VarPtr>& reuse_map,
                           const std::map<VarPtr, std::vector<VarPtr>>& var_sharing_groups) {
  // Custom IRMutator for MemRef sharing
  class MemRefSharingMutator : public IRMutator {
   public:
    explicit MemRefSharingMutator(const std::map<VarPtr, VarPtr>& reuse_map,
                                  const std::map<VarPtr, std::vector<VarPtr>>& sharing_groups)
        : reuse_map_(reuse_map), sharing_groups_(sharing_groups) {}

    StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
      // Check if this variable should reuse another's MemRef
      if (reuse_map_.count(op->var_)) {
        VarPtr source_var = reuse_map_.at(op->var_);

        // Get source's TileType and MemRef
        auto source_tile_type = As<TileType>(source_var->GetType());

        if (!source_tile_type || !source_tile_type->memref_.has_value()) {
          LOG_ERROR << "Source variable " << source_var->name_hint_ << " does not have MemRef";
          return IRMutator::VisitStmt_(op);
        }

        std::optional<MemRefPtr> source_memref = source_tile_type->memref_;

        // Get current variable's TileType
        auto curr_tile_type = As<TileType>(op->var_->GetType());

        if (!curr_tile_type) {
          LOG_ERROR << "Current variable " << op->var_->name_hint_ << " is not TileType";
          return IRMutator::VisitStmt_(op);
        }

        // Rebase a sharing-group member onto the reuse target, keeping its
        // offset relative to the representative and its own size — substituting
        // the target MemRef wholesale would collapse every member onto the
        // target's base offset (issue #1723: all per-head row slices read row 0).
        // Members coinciding with the representative reuse the target MemRef
        // object so plain reuse keeps MemRef identity.
        //
        // Const-offset only. A dynamic byte offset falls back to the target as-is;
        // this is safe for the same reason AllocateMemoryAddr falls back to the
        // bare base for dynamic offsets — such a view reaches codegen via
        // tile.slice → pto.subview, which re-derives its address from the slice
        // operands, not this MemRef's byte_offset (only const reshape-of-slice
        // chains, rebased below, depend on it).
        const MemRefPtr curr_memref = curr_tile_type->memref_.value_or(nullptr);
        auto rebase_memref = [&](const TileTypePtr& tile) -> std::optional<MemRefPtr> {
          if (!tile->memref_.has_value() || !curr_memref) return source_memref;
          const auto& old = tile->memref_.value();
          auto old_off = As<ConstInt>(old->byte_offset_);
          auto curr_off = As<ConstInt>(curr_memref->byte_offset_);
          if (!old_off || !curr_off) return source_memref;
          const int64_t rel = old_off->value_ - curr_off->value_;
          if (rel == 0 && old->size_ == (*source_memref)->size_) return source_memref;
          // The representative is the earliest-defined member (the whole-buffer
          // base in the alloc→subview flow), so members sit at non-negative
          // offsets within the target; a negative or out-of-range rel would
          // silently rebase onto a neighbouring allocation.
          INTERNAL_CHECK_SPAN(rel >= 0 && static_cast<uint64_t>(rel) + old->size_ <= (*source_memref)->size_,
                              old->span_)
              << "Internal error: sharing-group member offset " << old_off->value_
              << " rebased to rel=" << rel << " size=" << old->size_ << " falls outside reuse target (size "
              << (*source_memref)->size_ << ")";
          auto rel_expr = std::make_shared<ConstInt>(rel, DataType::INDEX, Span::unknown());
          return std::make_shared<MemRef>((*source_memref)->base_,
                                          AddByteOffsets((*source_memref)->byte_offset_, rel_expr),
                                          old->size_, old->span_);
        };

        // Create new TileType with shared MemRef
        auto new_tile_type = std::dynamic_pointer_cast<const TileType>(CloneTypeWithMemRefAndRemapExprs(
            curr_tile_type, source_memref, [this](const ExprPtr& expr) { return VisitExpr(expr); }));

        // Create new Var
        auto new_var = std::make_shared<const Var>(op->var_->name_hint_, new_tile_type, op->var_->span_);

        // Record the variable substitution mapping (old -> new)
        // This ensures that all subsequent references to the old variable will be replaced with the new one
        var_substitution_map_[op->var_] = new_var;

        // CRITICAL: If this variable shares MemRef with others (view operations),
        // we need to update ALL of them to use the new MemRef
        if (sharing_groups_.count(op->var_)) {
          const auto& sharing_group = sharing_groups_.at(op->var_);
          for (const auto& shared_var : sharing_group) {
            if (shared_var != op->var_) {
              // Create new Var for shared variable with same reused MemRef
              auto shared_tile_type = As<TileType>(shared_var->GetType());
              if (shared_tile_type) {
                auto new_shared_tile_type =
                    std::dynamic_pointer_cast<const TileType>(CloneTypeWithMemRefAndRemapExprs(
                        shared_tile_type, rebase_memref(shared_tile_type),
                        [this](const ExprPtr& expr) { return VisitExpr(expr); }));
                auto new_shared_var = std::make_shared<const Var>(shared_var->name_hint_,
                                                                  new_shared_tile_type, shared_var->span_);
                var_substitution_map_[shared_var] = new_shared_var;

                LOG_DEBUG << "Propagating reuse to sharing group member: " << shared_var->name_hint_;
              }
            }
          }
        }

        // Visit value expression (this will recursively apply substitutions)
        ExprPtr new_value = VisitExpr(op->value_);

        return std::make_shared<const AssignStmt>(new_var, new_value, op->span_);
      }

      return IRMutator::VisitStmt_(op);
    }

    // Override VisitExpr_ to replace variable references with their new versions
    ExprPtr VisitExpr_(const VarPtr& op) override {
      // Check if this variable has been replaced (i.e., it's the old version of a reused variable)
      if (var_substitution_map_.count(op)) {
        // Return the new version of the variable
        return var_substitution_map_.at(op);
      }
      // Otherwise, keep the variable as-is
      return op;
    }

   private:
    const std::map<VarPtr, VarPtr>& reuse_map_;
    const std::map<VarPtr, std::vector<VarPtr>>& sharing_groups_;  // var -> sharing group
    // Maps old variable objects to new variable objects (with reused MemRef)
    // This is needed because IR nodes are immutable, so we create new Var objects
    // and need to replace all references to the old ones
    std::map<VarPtr, VarPtr> var_substitution_map_;
  };

  MemRefSharingMutator mutator(reuse_map, var_sharing_groups);
  return mutator.VisitStmt(stmt);
}

/**
 * @brief Align loop-carried MemRefs to their initValue, top-down (fixes #1352).
 *
 * Greedy reuse (ApplyMemRefSharing) retypes an accumulator's *producer* and
 * *init* AssignStmt vars onto a reused buffer, but the loop-carried iter_arg /
 * return_var nodes never enter the lifetime/reuse maps (ComputeLifetimes
 * deliberately excludes them — see RegisterReturnVars), so they keep their
 * original buffer.  For *nested* pipelined accumulators this splits the chain:
 * an inner K-loop's initValue is the *outer* loop's iter_arg, and
 * YieldFixupMutator runs bottom-up — so it patches the inner carry against the
 * still-stale outer iter_arg and then inserts acc->acc tile.move ops to
 * reconcile the two buffers.  Ascend 910B has no hardware path between disjoint
 * accumulator addresses, so ptoas rejects those moves.
 *
 * This mutator restores the invariant codegen already assumes — "iter_arg /
 * return_var share initValue's buffer" — *before* YieldFixupMutator, and does
 * it top-down: it retypes each ForStmt's iter_args/return_vars to their
 * initValue's MemRef and seeds var_remap_ before recursing, so a nested loop
 * observes the corrected outer iter_arg as its init.  Once producers and
 * carries agree on one buffer, YieldFixupMutator inserts no spurious move.
 *
 * It only ever aligns a carry to its own init (a no-op when reuse left them
 * consistent), so a loop that genuinely yields a different buffer than its init
 * is untouched here and still gets its legitimate move from YieldFixupMutator.
 */
class AlignLoopCarriesToInitMutator : public IRMutator {
 public:
  StmtPtr VisitStmt_(const ForStmtPtr& op) override {
    if (op->iter_args_.empty()) return IRMutator::VisitStmt_(op);

    std::vector<IterArgPtr> new_iter_args = op->iter_args_;
    std::vector<VarPtr> new_return_vars = op->return_vars_;
    bool changed = false;

    for (size_t i = 0; i < op->iter_args_.size(); ++i) {
      // Resolve initValue through var_remap_ so an enclosing loop's alignment
      // (which rebuilt the init var this iter_arg carries) is observed here.
      auto init_expr = VisitExpr(op->iter_args_[i]->initValue_);
      bool init_changed = init_expr.get() != op->iter_args_[i]->initValue_.get();

      // MemRef alignment only applies when the init resolves to a tile-typed
      // var/iter_arg with a defined MemRef.  A non-tile or non-var carry (e.g.
      // a scalar, or a non-Var init expression) has no MemRef to align to, but
      // any remapped init_expr must still be propagated to the rebuilt IterArg
      // below — otherwise an enclosing loop's alignment is silently dropped.
      auto init_var = AsVarLike(init_expr);
      auto init_tile = init_var ? GetTileTypeWithMemRef(init_var->GetType()) : nullptr;
      MemRefPtr init_memref = nullptr;
      std::optional<MemorySpace> init_memory = std::nullopt;
      if (init_tile) {
        init_memref = GetDefinedMemRef(init_tile);
        init_memory = init_tile->GetMemorySpace();
      }

      // Re-type a carry's TileType onto init's MemRef, remapping any embedded
      // exprs (shape/strides) through var_remap_.  Only invoked when init_tile
      // is non-null, so init_memref is always defined at the call sites.
      auto retype_to_init = [&](const TileTypePtr& tile) {
        return CloneTypeWithMemRefAndRemapExprs(
            tile, init_memref, [this](const ExprPtr& e) { return VisitExpr(e); }, init_memory);
      };

      // Align the iter_arg's TileType to init's MemRef when both have a defined
      // MemRef, and always rebuild when the carried init reference changed so a
      // remapped init_expr is preserved regardless of the carry's type.
      auto ia_tile = As<TileType>(op->iter_args_[i]->GetType());
      bool ia_memref_differs = init_tile && ia_tile && ia_tile->memref_.has_value() &&
                               !MemRef::SameAllocation(GetDefinedMemRef(ia_tile), init_memref);
      if (ia_memref_differs || init_changed) {
        TypePtr new_ia_type = ia_memref_differs ? retype_to_init(ia_tile) : op->iter_args_[i]->GetType();
        new_iter_args[i] = std::make_shared<IterArg>(op->iter_args_[i]->name_hint_, new_ia_type, init_expr,
                                                     op->iter_args_[i]->span_);
        var_remap_[op->iter_args_[i].get()] = new_iter_args[i];
        changed = true;
      }

      // Align the matching return_var's TileType to init's MemRef (only when
      // init has a tile MemRef to align to).
      if (init_tile && i < new_return_vars.size()) {
        auto rv_tile = As<TileType>(op->return_vars_[i]->GetType());
        if (rv_tile && rv_tile->memref_.has_value() &&
            !MemRef::SameAllocation(GetDefinedMemRef(rv_tile), init_memref)) {
          new_return_vars[i] = std::make_shared<Var>(op->return_vars_[i]->name_hint_, retype_to_init(rv_tile),
                                                     op->return_vars_[i]->span_);
          var_remap_[op->return_vars_[i].get()] = new_return_vars[i];
          changed = true;
        }
      }
    }

    // Recurse into the body AFTER seeding var_remap_ so nested loops observe
    // the corrected outer iter_args as their init.
    auto new_body = VisitStmt(op->body_);

    // iter_arg references are confined to the loop body; drop them so a sibling
    // loop cannot pick up a stale remap.  return_var remaps must persist for
    // post-loop uses, so they are intentionally left in place.
    for (const auto& old_iter_arg : op->iter_args_) {
      var_remap_.erase(old_iter_arg.get());
    }

    if (!changed && new_body == op->body_) return op;
    auto new_for = MutableCopy(op);
    new_for->iter_args_ = std::move(new_iter_args);
    new_for->return_vars_ = std::move(new_return_vars);
    new_for->body_ = new_body;
    return new_for;
  }
};

/**
 * @brief Fix yield/return_var MemRef mismatch after MemoryReuse.
 *
 * Handles both ForStmt and IfStmt:
 * - ForStmt: MemoryReuse may assign different MemRefs to iter_arg and yield value.
 *   Since yield value becomes the next iteration's iter_arg, they must use the
 *   same buffer. When they differ, insert a tile.move before yield.
 * - IfStmt: MemoryReuse may change MemRefs of variables inside branches.
 *   Patch return_vars to match the yield value's MemRef.
 */
class YieldFixupMutator : public IRMutator {
 public:
  StmtPtr VisitStmt_(const ForStmtPtr& op) override {
    // First recurse into nested control flow
    auto result = IRMutator::VisitStmt_(op);
    auto for_stmt = As<ForStmt>(result);
    if (!for_stmt || for_stmt->iter_args_.empty()) return result;

    auto yield_stmt = FindYieldStmt(for_stmt->body_);
    if (!yield_stmt) return result;

    // Check each (iter_arg, yield_value) pair for MemRef mismatch.
    // Use initValue's MemRef as the target because codegen maps iter_arg to initValue's buffer,
    // and MemoryReuse may have updated initValue's MemRef without updating iter_arg's.
    std::vector<std::pair<size_t, VarPtr>> moves_to_insert;  // (index, new_moved_var)
    std::vector<StmtPtr> move_stmts;
    for (size_t i = 0; i < yield_stmt->value_.size() && i < for_stmt->iter_args_.size(); ++i) {
      auto yield_var = As<Var>(yield_stmt->value_[i]);
      if (!yield_var) continue;

      auto yield_tile = GetTileTypeWithMemRef(yield_var->GetType());
      if (!yield_tile) continue;
      auto yield_memref = GetDefinedMemRef(yield_tile);

      // Get the initValue's MemRef (the actual buffer used at runtime)
      auto init_var = As<Var>(for_stmt->iter_args_[i]->initValue_);
      if (!init_var) continue;
      auto init_tile = GetTileTypeWithMemRef(init_var->GetType());
      if (!init_tile) continue;
      auto init_memref = GetDefinedMemRef(init_tile);

      if (MemRef::SameAllocation(yield_memref, init_memref)) continue;

      // MemRef mismatch — create tile.move to copy yield value into initValue's buffer
      auto target_memory = init_tile->GetMemorySpace();
      auto [moved_var, move_stmt] = CreateTileMove(yield_var, init_memref, target_memory);

      move_stmts.emplace_back(std::move(move_stmt));
      moves_to_insert.emplace_back(i, moved_var);
    }

    if (moves_to_insert.empty()) {
      // Even without tile.move insertion, iter_arg and return_var may have stale MemRefs
      // (MemoryReuse updates initValue/yield but not iter_arg/return_var types).
      // Ensure all 4 loop-carry variables share the same MemRef (= initValue's).
      return PatchIterArgsAndReturnVars(for_stmt, yield_stmt);
    }

    // Build new yield values with moved vars substituted
    std::vector<ExprPtr> new_yield_values = yield_stmt->value_;
    for (const auto& [idx, moved_var] : moves_to_insert) {
      new_yield_values[idx] = moved_var;
    }
    auto new_yield = MutableCopy(yield_stmt);
    new_yield->value_ = std::move(new_yield_values);

    // Insert tile.move stmts before yield and replace yield in body
    auto new_body = InsertMovesAndReplaceYield(for_stmt->body_, new_yield, move_stmts);

    // Build intermediate ForStmt with new body, then patch iter_args/return_vars
    auto intermediate_for = MutableCopy(for_stmt);
    intermediate_for->body_ = new_body;

    return PatchIterArgsAndReturnVars(intermediate_for, new_yield);
  }

  StmtPtr VisitStmt_(const IfStmtPtr& op) override {
    // First recurse into nested control flow
    auto result = IRMutator::VisitStmt_(op);
    auto if_stmt = As<IfStmt>(result);
    if (!if_stmt || if_stmt->return_vars_.empty()) return result;

    // Find yield statements in each branch
    auto then_yield = FindYieldStmt(if_stmt->then_body_);
    auto else_yield = if_stmt->else_body_.has_value() ? FindYieldStmt(if_stmt->else_body_.value()) : nullptr;
    if (!then_yield && !else_yield) return result;

    // For each return_var position, check if the two branches yield tiles
    // with different MemRefs.  When they differ, insert tile.move in the
    // branch whose MemRef ≠ the canonical target (then-branch is canonical).
    bool body_changed = false;
    std::vector<std::pair<size_t, VarPtr>> else_moves;
    std::vector<StmtPtr> else_move_stmts;
    std::vector<VarPtr> new_return_vars = if_stmt->return_vars_;

    for (size_t i = 0; i < new_return_vars.size(); ++i) {
      VarPtr then_var =
          (then_yield && i < then_yield->value_.size()) ? As<Var>(then_yield->value_[i]) : nullptr;
      VarPtr else_var =
          (else_yield && i < else_yield->value_.size()) ? As<Var>(else_yield->value_[i]) : nullptr;

      auto then_tile = then_var ? GetTileTypeWithMemRef(then_var->GetType()) : nullptr;
      auto else_tile = else_var ? GetTileTypeWithMemRef(else_var->GetType()) : nullptr;
      if (!then_tile && !else_tile) continue;

      // Use then-branch MemRef as canonical target (fall back to else if then absent)
      auto target_tile = then_tile ? then_tile : else_tile;
      auto target_memref = GetDefinedMemRef(target_tile);
      auto target_memory = target_tile->GetMemorySpace();

      // Check else branch: if MemRef differs from target, insert tile.move
      if (then_tile && else_tile) {
        auto else_memref = GetDefinedMemRef(else_tile);
        if (!MemRef::SameAllocation(else_memref, target_memref)) {
          auto [moved_var, move_stmt] = CreateTileMove(else_var, target_memref, target_memory);
          else_move_stmts.emplace_back(std::move(move_stmt));
          else_moves.emplace_back(i, moved_var);
          body_changed = true;
        }
      }

      // Patch return_var to share target MemRef
      auto rv_tile = As<TileType>(new_return_vars[i]->GetType());
      if (rv_tile && rv_tile->memref_.has_value()) {
        auto rv_memref = GetDefinedMemRef(rv_tile);
        if (!MemRef::SameAllocation(rv_memref, target_memref)) {
          auto new_rv_type = CloneTypeWithMemRefAndRemapExprs(
              rv_tile, target_memref, [this](const ExprPtr& e) { return VisitExpr(e); }, target_memory);
          new_return_vars[i] =
              std::make_shared<Var>(new_return_vars[i]->name_hint_, new_rv_type, new_return_vars[i]->span_);
          var_remap_[if_stmt->return_vars_[i].get()] = new_return_vars[i];
          body_changed = true;
        }
      }
    }

    if (!body_changed) return result;

    // Rebuild else body with tile.move stmts inserted before yield
    auto new_else_body = if_stmt->else_body_;
    if (!else_moves.empty() && else_yield && if_stmt->else_body_.has_value()) {
      std::vector<ExprPtr> new_else_yield_values = else_yield->value_;
      for (const auto& [idx, moved_var] : else_moves) {
        new_else_yield_values[idx] = moved_var;
      }
      auto new_yield = MutableCopy(else_yield);
      new_yield->value_ = std::move(new_else_yield_values);
      new_else_body = InsertMovesAndReplaceYield(if_stmt->else_body_.value(), new_yield, else_move_stmts);
    }

    auto new_if = MutableCopy(if_stmt);
    new_if->else_body_ = new_else_body;
    new_if->return_vars_ = std::move(new_return_vars);
    return new_if;
  }

 private:
  // Create a tile.move operation that copies source into target_memref's buffer.
  // Returns (moved_var, move_assign_stmt).
  std::pair<VarPtr, StmtPtr> CreateTileMove(const VarPtr& source, const MemRefPtr& target_memref,
                                            std::optional<MemorySpace> target_memory) {
    INTERNAL_CHECK_SPAN(target_memory.has_value(), source->span_)
        << "Internal error: target TileType must have memory_space for tile.move";
    auto& op_reg = OpRegistry::GetInstance();
    std::vector<std::pair<std::string, std::any>> kwargs = {
        {"target_memory", std::any(target_memory.value())}};
    auto move_call = op_reg.Create("tile.move", {source}, kwargs, source->span_);
    auto moved_type = CloneTypeWithMemRefAndRemapExprs(
        source->GetType(), target_memref, [this](const ExprPtr& expr) { return VisitExpr(expr); },
        target_memory);
    auto moved_var = std::make_shared<Var>(source->name_hint_ + "_mv", moved_type, source->span_);
    auto move_stmt = std::make_shared<AssignStmt>(moved_var, move_call, source->span_);
    return {moved_var, move_stmt};
  }

  // Patch iter_args and return_vars to share initValue's MemRef.
  // Returns the original ForStmt if no patching is needed.
  StmtPtr PatchIterArgsAndReturnVars(const ForStmtPtr& for_stmt, const YieldStmtPtr& yield_stmt) {
    bool changed = false;
    std::vector<IterArgPtr> new_iter_args = for_stmt->iter_args_;
    std::vector<VarPtr> new_return_vars = for_stmt->return_vars_;

    for (size_t i = 0; i < for_stmt->iter_args_.size(); ++i) {
      auto init_var = As<Var>(for_stmt->iter_args_[i]->initValue_);
      if (!init_var) continue;
      auto init_tile = GetTileTypeWithMemRef(init_var->GetType());
      if (!init_tile) continue;
      auto init_memref = GetDefinedMemRef(init_tile);

      // Patch iter_arg if its MemRef differs from initValue's
      auto ia_tile = As<TileType>(for_stmt->iter_args_[i]->GetType());
      if (ia_tile && ia_tile->memref_.has_value()) {
        auto ia_memref = GetDefinedMemRef(ia_tile);
        if (!MemRef::SameAllocation(ia_memref, init_memref)) {
          auto new_ia_type = CloneTypeWithMemRefAndRemapExprs(
              ia_tile, init_memref, [this](const ExprPtr& expr) { return VisitExpr(expr); },
              init_tile->GetMemorySpace());
          new_iter_args[i] =
              std::make_shared<IterArg>(for_stmt->iter_args_[i]->name_hint_, new_ia_type,
                                        for_stmt->iter_args_[i]->initValue_, for_stmt->iter_args_[i]->span_);
          var_remap_[for_stmt->iter_args_[i].get()] = new_iter_args[i];
          changed = true;
        }
      }

      // Patch return_var to share yield value's MemRef (which should == initValue's after fixup)
      if (i >= new_return_vars.size() || !yield_stmt || i >= yield_stmt->value_.size()) continue;
      auto yield_var = As<Var>(yield_stmt->value_[i]);
      if (!yield_var) continue;
      auto yield_tile = GetTileTypeWithMemRef(yield_var->GetType());
      if (!yield_tile) continue;
      auto yield_memref = GetDefinedMemRef(yield_tile);
      auto rv_tile = As<TileType>(new_return_vars[i]->GetType());
      if (!rv_tile || !rv_tile->memref_.has_value()) continue;
      auto rv_memref = GetDefinedMemRef(rv_tile);
      if (!MemRef::SameAllocation(rv_memref, yield_memref)) {
        auto new_rv_type = CloneTypeWithMemRefAndRemapExprs(
            rv_tile, yield_memref, [this](const ExprPtr& expr) { return VisitExpr(expr); },
            yield_tile->GetMemorySpace());
        new_return_vars[i] =
            std::make_shared<Var>(new_return_vars[i]->name_hint_, new_rv_type, new_return_vars[i]->span_);
        // Register old→new so downstream references (e.g., ReturnStmt) are updated
        var_remap_[for_stmt->return_vars_[i].get()] = new_return_vars[i];
        changed = true;
      }
    }

    if (!changed) return for_stmt;

    auto patched_body = VisitStmt(for_stmt->body_);

    for (const auto& old_iter_arg : for_stmt->iter_args_) {
      var_remap_.erase(old_iter_arg.get());
    }

    auto new_for = MutableCopy(for_stmt);
    new_for->iter_args_ = new_iter_args;
    new_for->body_ = patched_body;
    new_for->return_vars_ = std::move(new_return_vars);
    return new_for;
  }

  // Replace YieldStmt in body and insert move AssignStmts before it.
  // Body structure is typically SeqStmts([...assigns..., YieldStmt]).
  // Move stmts go directly into the SeqStmts before the yield.
  static StmtPtr InsertMovesAndReplaceYield(const StmtPtr& body, const YieldStmtPtr& new_yield,
                                            const std::vector<StmtPtr>& move_stmts) {
    if (As<YieldStmt>(body)) {
      // Body is just a yield — wrap moves + yield in SeqStmts
      std::vector<StmtPtr> stmts;
      stmts.insert(stmts.end(), move_stmts.begin(), move_stmts.end());
      stmts.push_back(new_yield);
      return SeqStmts::Flatten(std::move(stmts), body->span_);
    }
    if (auto seq = As<SeqStmts>(body)) {
      std::vector<StmtPtr> new_children;
      for (const auto& child : seq->stmts_) {
        if (As<YieldStmt>(child)) {
          // Insert move stmts directly before the new yield
          new_children.insert(new_children.end(), move_stmts.begin(), move_stmts.end());
          new_children.push_back(new_yield);
        } else {
          new_children.push_back(child);
        }
      }
      return SeqStmts::Flatten(std::move(new_children), body->span_);
    }
    return body;
  }
};

// Check if a statement is a tile.alloc AssignStmt for an unused MemRef
bool IsUnusedAllocStmt(const StmtPtr& stmt, const std::set<const Var*>& used_bases) {
  auto assign = As<AssignStmt>(stmt);
  if (!assign) return false;
  auto call = As<Call>(assign->value_);
  if (!call) return false;
  if (call->op_->name_ != "tile.alloc" && call->op_->name_ != "tensor.alloc") return false;
  // Alloc LHS is a Ptr Var — check if any MemRef's base_ still references it
  return used_bases.find(assign->var_.get()) == used_bases.end();
}

// Remove unused alloc statements from a SeqStmts body
StmtPtr RemoveUnusedAllocStatements(const StmtPtr& body, const std::set<const Var*>& used_bases) {
  auto seq = As<SeqStmts>(body);
  if (!seq) return body;

  std::vector<StmtPtr> new_seq_stmts;
  bool changed = false;

  for (const auto& child : seq->stmts_) {
    if (IsUnusedAllocStmt(child, used_bases)) {
      changed = true;
      continue;
    }
    new_seq_stmts.push_back(child);
  }

  if (!changed) return body;
  return SeqStmts::Flatten(std::move(new_seq_stmts), body->span_);
}

/**
 * @brief Transform a function by identifying and applying memory reuse
 *
 * This transformation identifies memory reuse opportunities by walking the full
 * IR tree to compute variable lifetimes, then applying greedy MemRef sharing.
 * Variables that can share memory will point to the same MemRef object.
 * After sharing, redundant alloc operations are removed.
 */
FunctionPtr TransformMemoryReuse(const FunctionPtr& func) {
  INTERNAL_CHECK(func) << "MemoryReusePass cannot run on null function";

  // Orchestration functions submit tasks and never hold TileType variables,
  // so there is nothing for memory reuse to do — skip them silently.
  if (func->func_type_ == FunctionType::Orchestration) return func;

  // Step 0: Top-down retarget — propagate iter_arg/initValue MemRefs down the
  // yield chain so accumulator producers land directly in the canonical buffer.
  // This eliminates most accumulator-related move insertions downstream.
  StmtPtr new_body = func->body_;
  {
    TopDownRetargeter retargeter;
    auto rewrites = retargeter.Compute(new_body);
    if (!rewrites.empty()) {
      RetypeApplier applier(std::move(rewrites));
      new_body = applier.VisitStmt(new_body);
    }
  }

  // Step 1: Compute lifetimes by walking full IR tree
  auto analysis_result = ComputeLifetimes(new_body);

  if (analysis_result.lifetimes.empty()) {
    LOG_DEBUG << "No TileType variables found in function '" << func->name_ << "', skipping memory reuse";
    return func;
  }

  // Step 2: Identify reuse opportunities.  On Ascend910B split-AIV functions,
  // collect the load + tpop_from_aic hazard inputs so the reuse decision never
  // forms the hazardous in-place sharing (folds in the former
  // LegalizePTOBufferReuse responsibility).  Off-910B the inputs stay empty and
  // reuse behaviour is unchanged.
  HazardInputs hazard;
  if (NeedsLoadTpopHazardGuard(func)) {
    HazardInputCollector collector;
    collector.VisitStmt(new_body);
    hazard = collector.Take();
  }

  // Per-operand no-alias map (e.g. tile.sel's mask/tmp must not share the
  // output's buffer). Op-semantic, not backend-gated, so always collected.
  ForbidAliasCollector forbid_collector(analysis_result.var_sharing_groups);
  forbid_collector.VisitStmt(new_body);
  ForbidAliasMap forbid_alias = forbid_collector.Take();

  auto reuse_map = IdentifyReuseOpportunities(
      analysis_result.lifetimes, hazard, forbid_alias, analysis_result.phi_family_ids,
      analysis_result.var_sharing_groups, analysis_result.var_liveness);

  // Step 3: Apply MemRef sharing (skip if no reuse candidates)
  if (!reuse_map.empty()) {
    new_body = ApplyMemRefSharing(new_body, reuse_map, analysis_result.var_sharing_groups);

    // Step 3.5: Re-align loop-carried iter_arg/return_var MemRefs to their
    // (now-reused) initValue, top-down.  Reuse retypes producer/init AssignStmt
    // vars but leaves loop-carry nodes stale; for nested pipelined accumulators
    // that split chain would otherwise force YieldFixupMutator to emit invalid
    // acc->acc tile.move ops (#1352).
    AlignLoopCarriesToInitMutator align;
    new_body = align.VisitStmt(new_body);
  }

  // Step 4: Fix ForStmt/IfStmt yield/return_var MemRef mismatches
  YieldFixupMutator yield_fixup;
  new_body = yield_fixup.VisitStmt(new_body);

  // Step 5: Remove alloc statements for MemRefs no longer in use
  auto used_bases = memref_collectors::CollectUsedBasePtrs(new_body);
  new_body = RemoveUnusedAllocStatements(new_body, used_bases);

  auto result = std::make_shared<const Function>(func->name_, func->params_, func->param_directions_,
                                                 func->return_types_, new_body, func->span_, func->func_type_,
                                                 func->level_, func->role_, func->attrs_);
  return result;
}

}  // namespace

namespace pass {
Pass MemoryReuse() { return CreateFunctionPass(TransformMemoryReuse, "MemoryReuse", kMemoryReuseProperties); }
}  // namespace pass
}  // namespace ir
}  // namespace pypto
