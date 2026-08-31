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
#include <cctype>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/backend/common/backend.h"
#include "pypto/backend/common/backend_config.h"
#include "pypto/backend/common/backend_handler.h"
#include "pypto/core/any_cast.h"
#include "pypto/core/error.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/memref.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_context.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/structural_comparison.h"
#include "pypto/ir/transforms/utils/allocation_constraint_analysis.h"
#include "pypto/ir/transforms/utils/attrs.h"
#include "pypto/ir/transforms/utils/lifetime_analysis.h"
#include "pypto/ir/transforms/utils/memory_footprint.h"
#include "pypto/ir/transforms/utils/memref_collectors.h"
#include "pypto/ir/transforms/utils/memref_utils.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/transforms/utils/op_predicates.h"
#include "pypto/ir/transforms/utils/reserve_buffer_utils.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

namespace {

/**
 * @brief Collect all Var nodes referenced in an expression.
 */
class VarUseCollector : public IRVisitor {
 public:
  std::set<VarPtr> used_vars;

  void VisitVarLike_(const VarPtr& var) override {
    used_vars.insert(var);
    IRVisitor::VisitVarLike_(var);
  }
};

/// True when two MemRefs denote the same codegen-visible byte window.
/// Allocation identity alone is insufficient: one allocation can contain
/// multiple disjoint tile windows and pipeline slots.
bool SamePhysicalWindow(const MemRefPtr& lhs, const MemRefPtr& rhs) {
  if (!lhs || !rhs || !MemRef::SameAllocation(lhs, rhs) || lhs->size_ != rhs->size_ ||
      lhs->slot_count_ != rhs->slot_count_ || !structural_equal(lhs->byte_offset_, rhs->byte_offset_)) {
    return false;
  }

  const bool lhs_has_slot = lhs->slot_index_.has_value() && *lhs->slot_index_;
  const bool rhs_has_slot = rhs->slot_index_.has_value() && *rhs->slot_index_;
  if (lhs_has_slot != rhs_has_slot) return false;
  return !lhs_has_slot || structural_equal(*lhs->slot_index_, *rhs->slot_index_);
}

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
// Liveness check: one analysis walk numbers statements twice, once with each
// IfStmt's then arm first and once with its else arm first. A statement can
// execute after a definition iff it is later in both orderings; sibling arms
// are later in exactly one and are therefore excluded. Read/write events are
// indexed by overlapping physical-window component, so each query is O(log N)
// and the complete retargeting analysis remains O(N log N).
// ============================================================================

/// Describes where a TileType Var is defined.
struct VarDef {
  enum Kind { kAssign, kIfReturn, kForReturn, kIterArg, kUnknown };
  Kind kind = kUnknown;
  StmtPtr assign_stmt;   // AssignStmt (for kAssign)
  StmtPtr control_stmt;  // IfStmt/ForStmt (for kIfReturn/kForReturn)
  StmtPtr definition_stmt;
  ForStmtPtr nearest_for;
  size_t return_idx = 0;  // index into return_vars_ (for kIfReturn/kForReturn)
  IterArgPtr iter_arg;    // for kIterArg
  size_t order = 0;       // lexical statement order of the definition
};

/// Walks the IR once to build definitions, lexical statement intervals and
/// per-statement VarLike accesses. Definitions keep only their nearest loop;
/// no per-definition ancestor-chain copies are required.
class DefMapVisitor : public IRVisitor {
 public:
  std::map<VarPtr, VarDef> defs;
  std::map<const Stmt*, size_t> stmt_order;
  std::map<const Stmt*, size_t> stmt_end_order;
  std::map<const Stmt*, std::set<VarPtr>> reads;
  std::map<const Stmt*, std::set<VarPtr>> writes;
  StmtPtr root;

  void Run(const StmtPtr& body) {
    root = body;
    VisitStmt(body);
    for (auto& [var, def] : defs) {
      (void)var;
      if (def.kind == VarDef::kIfReturn || def.kind == VarDef::kForReturn) {
        def.order = stmt_end_order.at(def.control_stmt.get());
      }
    }
  }

 protected:
  void VisitStmt(const StmtPtr& stmt) override {
    if (!stmt) return;
    auto previous = current_stmt_;
    current_stmt_ = stmt;
    stmt_order.emplace(stmt.get(), next_order_++);
    IRVisitor::VisitStmt(stmt);
    stmt_end_order.emplace(stmt.get(), next_order_ - 1);
    current_stmt_ = previous;
  }

  void VisitStmt_(const SeqStmtsPtr& op) override {
    for (const auto& s : op->stmts_) VisitStmt(s);
  }

  void VisitStmt_(const AssignStmtPtr& op) override {
    if (As<TileType>(op->var_->GetType())) {
      VarDef d;
      d.kind = VarDef::kAssign;
      d.assign_stmt = op;
      d.definition_stmt = op;
      d.nearest_for = for_stack_.empty() ? nullptr : for_stack_.back();
      d.order = stmt_order.at(op.get());
      defs[op->var_] = d;
      writes[op.get()].insert(op->var_);
    }
    if (op->value_) VisitExpr(op->value_);
  }

  void VisitStmt_(const IfStmtPtr& op) override {
    VisitExpr(op->condition_);
    for (size_t i = 0; i < op->return_vars_.size(); ++i) {
      const auto& rv = op->return_vars_[i];
      if (!As<TileType>(rv->GetType())) continue;
      VarDef d;
      d.kind = VarDef::kIfReturn;
      d.control_stmt = op;
      d.definition_stmt = op;
      d.nearest_for = for_stack_.empty() ? nullptr : for_stack_.back();
      d.return_idx = i;
      d.order = stmt_order.at(op.get());
      defs[rv] = d;
    }
    VisitStmt(op->then_body_);
    if (op->else_body_.has_value()) VisitStmt(op->else_body_.value());
  }

  void VisitStmt_(const ForStmtPtr& op) override {
    VisitExpr(op->loop_var_);
    VisitExpr(op->start_);
    VisitExpr(op->stop_);
    VisitExpr(op->step_);
    for (const auto& iter_arg : op->iter_args_) VisitExpr(iter_arg);

    for (size_t i = 0; i < op->iter_args_.size(); ++i) {
      auto ia = op->iter_args_[i];
      if (!As<TileType>(ia->GetType())) continue;
      VarDef d;
      d.kind = VarDef::kIterArg;
      d.control_stmt = op;
      d.definition_stmt = op;
      d.nearest_for = op;
      d.return_idx = i;
      d.iter_arg = ia;
      d.order = stmt_order.at(op.get());
      defs[std::static_pointer_cast<const Var>(ia)] = d;
    }
    for (size_t i = 0; i < op->return_vars_.size(); ++i) {
      const auto& rv = op->return_vars_[i];
      if (!As<TileType>(rv->GetType())) continue;
      VarDef d;
      d.kind = VarDef::kForReturn;
      d.control_stmt = op;
      d.definition_stmt = op;
      d.nearest_for = op;
      d.return_idx = i;
      d.order = stmt_order.at(op.get());
      defs[rv] = d;
    }
    for_stack_.push_back(op);
    VisitStmt(op->body_);
    for_stack_.pop_back();
  }

  void VisitStmt_(const InCoreScopeStmtPtr& op) override { VisitStmt(op->body_); }
  void VisitStmt_(const ClusterScopeStmtPtr& op) override { VisitStmt(op->body_); }
  void VisitStmt_(const HierarchyScopeStmtPtr& op) override { VisitStmt(op->body_); }
  void VisitStmt_(const SpmdScopeStmtPtr& op) override { VisitStmt(op->body_); }

  void VisitVarLike_(const VarPtr& op) override {
    if (current_stmt_) reads[current_stmt_.get()].insert(op);
    IRVisitor::VisitVarLike_(op);
  }

 private:
  StmtPtr current_stmt_;
  std::vector<ForStmtPtr> for_stack_;
  size_t next_order_ = 0;
};

/// A second lexical numbering with every IfStmt's arms visited in reverse.
/// In structured control flow, statement A can execute before statement B iff
/// A precedes B in both this ordering and DefMapVisitor's then-first ordering.
class ReverseBranchOrderVisitor : public IRVisitor {
 public:
  std::map<const Stmt*, size_t> stmt_order;
  std::map<const Stmt*, size_t> stmt_end_order;

  void Run(const StmtPtr& body) { VisitStmt(body); }

 protected:
  void VisitStmt(const StmtPtr& stmt) override {
    if (!stmt) return;
    stmt_order.emplace(stmt.get(), next_order_++);
    IRVisitor::VisitStmt(stmt);
    stmt_end_order.emplace(stmt.get(), next_order_ - 1);
  }

  void VisitStmt_(const IfStmtPtr& op) override {
    VisitExpr(op->condition_);
    if (op->else_body_.has_value()) VisitStmt(*op->else_body_);
    VisitStmt(op->then_body_);
    for (const auto& return_var : op->return_vars_) VisitExpr(return_var);
  }

 private:
  size_t next_order_ = 0;
};

/// Visits a stmt subtree and collects the MemRef base Ptrs of every *read*
/// of a TileType Var.  Writes (the LHS of an AssignStmt) are intentionally
/// excluded; every other stmt/expression kind dispatches through the default
/// IRVisitor traversal, so new stmt types are covered automatically.
class ExprReadBaseCollector : public IRVisitor {
 public:
  std::set<const Var*> bases;

  void VisitVarLike_(const VarPtr& var) override {
    if (auto tile = GetTileTypeWithMemRef(var->GetType())) {
      bases.insert(GetDefinedMemRef(tile)->base_.get());
    }
    IRVisitor::VisitVarLike_(var);
  }

  void VisitStmt_(const AssignStmtPtr& op) override {
    // Skip op->var_ — the LHS is a definition, not a read.
    if (op->value_) VisitExpr(op->value_);
  }
};

/// Range-maximum index over events (primary lexical order, reverse-branch
/// lexical order). A query succeeds when an event lies in the primary interval
/// and is also after the lower bound in the reverse ordering.
class ReachableEventIndex {
 public:
  void Build(std::vector<std::pair<size_t, size_t>> events) {
    if (events.empty()) return;
    std::sort(events.begin(), events.end());
    events.erase(std::unique(events.begin(), events.end()), events.end());
    size_ = 1;
    while (size_ < events.size()) size_ *= 2;
    primary_.reserve(events.size());
    max_reverse_.assign(size_ * 2, 0);
    for (size_t i = 0; i < events.size(); ++i) {
      primary_.push_back(events[i].first);
      max_reverse_[size_ + i] = events[i].second + 1;
    }
    for (size_t i = size_; i-- > 1;) {
      max_reverse_[i] = std::max(max_reverse_[i * 2], max_reverse_[i * 2 + 1]);
    }
  }

  [[nodiscard]] bool HasAfter(size_t primary_begin, size_t primary_end, size_t reverse_lower) const {
    if (primary_.empty() || primary_begin > primary_end) return false;
    size_t left = static_cast<size_t>(std::lower_bound(primary_.begin(), primary_.end(), primary_begin) -
                                      primary_.begin());
    size_t right = static_cast<size_t>(std::upper_bound(primary_.begin(), primary_.end(), primary_end) -
                                       primary_.begin());
    left += size_;
    right += size_;
    size_t maximum = 0;
    while (left < right) {
      if (left & 1U) maximum = std::max(maximum, max_reverse_[left++]);
      if (right & 1U) maximum = std::max(maximum, max_reverse_[--right]);
      left /= 2;
      right /= 2;
    }
    return maximum > reverse_lower + 1;
  }

 private:
  std::vector<size_t> primary_;
  std::vector<size_t> max_reverse_;
  size_t size_ = 0;
};

/// Range-minimum index over read events and their logical definition versions.
/// Aliases can share a physical allocation while naming different values over
/// time (notably an IterArg's initializer and the corresponding ForStmt
/// return). A post-If read is an independent observation of the old value only
/// when the handle being read was defined before that IfStmt.
///
/// These queries begin after the complete IfStmt subtree, where primary lexical
/// order alone proves reachability (both arms precede the suffix in either
/// branch ordering). The index therefore stores the minimum handle-definition
/// order per primary interval. Build is O(N log N), each query is O(log N), and
/// the complete pass remains O(N log N).
class VersionedReachableEventIndex {
 public:
  struct Event {
    size_t primary;
    size_t definition;
  };

  void Build(std::vector<Event> events) {
    if (events.empty()) return;
    std::sort(events.begin(), events.end(), [](const Event& lhs, const Event& rhs) {
      return std::tie(lhs.primary, lhs.definition) < std::tie(rhs.primary, rhs.definition);
    });

    primary_.reserve(events.size());
    std::vector<size_t> leaf_minimum;
    leaf_minimum.reserve(events.size());
    for (size_t i = 0; i < events.size();) {
      primary_.push_back(events[i].primary);
      size_t minimum = events[i].definition;
      size_t j = i + 1;
      while (j < events.size() && events[j].primary == events[i].primary) {
        minimum = std::min(minimum, events[j].definition);
        ++j;
      }
      leaf_minimum.push_back(minimum);
      i = j;
    }

    size_ = 1;
    while (size_ < primary_.size()) size_ *= 2;
    minimum_definition_.assign(size_ * 2, std::numeric_limits<size_t>::max());
    for (size_t i = 0; i < leaf_minimum.size(); ++i) {
      minimum_definition_[size_ + i] = leaf_minimum[i];
    }
    for (size_t i = size_; i-- > 1;) {
      minimum_definition_[i] = std::min(minimum_definition_[i * 2], minimum_definition_[i * 2 + 1]);
    }
  }

  [[nodiscard]] bool HasAfterFromHandleDefinedBefore(size_t primary_begin, size_t primary_end,
                                                     size_t definition_upper) const {
    if (primary_.empty() || primary_begin > primary_end) return false;
    size_t left = static_cast<size_t>(std::lower_bound(primary_.begin(), primary_.end(), primary_begin) -
                                      primary_.begin());
    size_t right = static_cast<size_t>(std::upper_bound(primary_.begin(), primary_.end(), primary_end) -
                                       primary_.begin());
    left += size_;
    right += size_;
    size_t minimum = std::numeric_limits<size_t>::max();
    while (left < right) {
      if (left & 1U) minimum = std::min(minimum, minimum_definition_[left++]);
      if (right & 1U) minimum = std::min(minimum, minimum_definition_[--right]);
      left /= 2;
      right /= 2;
    }
    return minimum < definition_upper;
  }

 private:
  std::vector<size_t> primary_;
  std::vector<size_t> minimum_definition_;
  size_t size_ = 0;
};

bool IsA5Target() {
  if (!backend::BackendConfig::IsConfigured()) return false;
  const auto* ctx = PassContext::Current();
  return ctx != nullptr && ctx->GetBackendHandler()->GetPtoTargetArch() == "a5";
}

bool IsA5Prelu(const CallPtr& call) { return IsOp(call, "tile.prelu") && IsA5Target(); }

/// Plans top-down retypes. Produces (old Var -> new Type) map.
class TopDownRetargeter {
 public:
  /// Runs the analysis. Returns map: old VarPtr -> new Type (with target MemRef).
  std::map<VarPtr, TypePtr> Compute(const StmtPtr& func_body) {
    BuildAnalysis(func_body);
    VisitForStmts(func_body);
    return std::move(rewrites_);
  }

  /// Coalesce peeled loop-carried accumulator if-phis.
  ///
  /// DIRECTION OF TRAVEL (long horizon — do not act on this without a migration).
  /// The shape this repairs is the *hand-peeled* split-K idiom:
  ///
  ///     if k == 0: acc = tile.matmul(a, b)          // fresh Acc buffer
  ///     else:      acc = tile.matmul_acc(acc, a, b) // in-place, another buffer
  ///
  /// It gives one logical value two producers on two L0C buffers, which the
  /// machine cannot represent — there is no Acc->Acc copy, so this repair exists
  /// only to rewrite buffer identity after the fact.  The predicated spelling
  /// says the same thing on one buffer and needs no repair:
  ///
  ///     acc = tile.matmul_acc(acc, a, b, init_cond=(k == 0))
  ///
  /// `AutoTileMatmulL0` already *generates* the predicated form (and head-peels
  /// the bias K-loop, which has no `init_cond` operand), so no compiler-generated
  /// IR reaches this function any more — every remaining caller is a peel in the
  /// author's own source.  We keep repairing those: existing kernels written with
  /// the peel must keep compiling, and pypto-lib alone has 95 such sites.
  ///
  /// Retiring this repair is therefore a *long-horizon* intent, not a near-term
  /// plan. It would need the source-level migration to land first. Until then,
  /// prefer `init_cond` in new code and leave existing peels alone.
  ///
  /// LowerPipelineLoops peels a stage=2 K-loop into an epilogue IfStmt whose live
  /// branch is an in-place accumulator (matmul_acc, output aliasing input on the
  /// accumulator buffer) and whose dead `if k==0` branch is a fresh matmul seed on a
  /// *different* buffer.  Left alone, YieldFixupMutator reconciles the two by copying
  /// the accumulator onto the seed's buffer via an Acc->Acc tile.move — both a second
  /// co-live L0C buffer (overflow) and an op TMovOp::verify rejects on every target
  /// (there is no legal Acc->Acc tmov pair).  We instead retarget the seed producer
  /// onto the accumulator buffer so both branches share it and no move is emitted,
  /// matching mad_acc's shared-%dst in-place semantics.
  ///
  /// The seed retype bypasses the *global* dead-at-assign check (the accumulator
  /// buffer is legitimately live at the post-if phi consumer, which the global
  /// check would treat as a conflict), but only after `TryCoalesceAccIfPhi`
  /// verifies the two preconditions branch exclusivity actually needs: (a) the
  /// seed producer is lexically inside the branch, and (b) a branch-scoped
  /// liveness scan (`IsTargetDeadAtAssign(..., stop_at=if)`) finds no same-branch
  /// tail read of the accumulator buffer. For both a direct `matmul_acc` and a
  /// branch-local loop seeded outside the IfStmt, an alias-aware post-if scan
  /// proves that no external handle to the reused input remains observable
  /// independently of the phi. When any condition fails, that phi is left to
  /// YieldFixup instead of being coalesced.
  /// Returns the rewrite map (apply via RetypeApplier).  A needed-but-declined
  /// retarget (after the preconditions hold) is a hard error — no legal Acc->Acc
  /// move exists to fall back to.
  std::map<VarPtr, TypePtr> CoalesceAccumulatorIfPhis(const StmtPtr& func_body) {
    BuildAnalysis(func_body);
    VisitIfPhisForAccumulator(func_body);
    return std::move(rewrites_);
  }

 private:
  struct RewriteJournalEntry {
    VarPtr var;
    std::optional<TypePtr> previous;
  };

  class RewriteTransaction {
   public:
    explicit RewriteTransaction(TopDownRetargeter& owner)
        : owner_(owner), checkpoint_(owner.rewrite_journal_.size()) {}
    RewriteTransaction(const RewriteTransaction&) = delete;
    RewriteTransaction& operator=(const RewriteTransaction&) = delete;
    ~RewriteTransaction() {
      if (!committed_) owner_.RollbackRewrites(checkpoint_);
    }

    void Commit() { committed_ = true; }

   private:
    TopDownRetargeter& owner_;
    size_t checkpoint_;
    bool committed_ = false;
  };

  std::map<VarPtr, VarDef> defs_;
  std::map<const Stmt*, size_t> stmt_order_;
  std::map<const Stmt*, size_t> stmt_end_order_;
  std::map<const Stmt*, size_t> reverse_stmt_order_;
  std::map<const Stmt*, size_t> reverse_stmt_end_order_;
  StmtPtr analysis_root_;
  std::map<VarPtr, VarPtr> alias_source_;
  std::map<VarPtr, VarPtr> alias_root_cache_;
  std::map<VarPtr, VarPtr> inplace_source_;
  std::map<VarPtr, VarPtr> inplace_root_cache_;
  std::map<VarPtr, VarPtr> accumulator_input_cache_;
  std::set<VarPtr> non_accumulator_producers_;
  std::map<const MemRef*, size_t> memref_component_;
  std::map<size_t, ReachableEventIndex> physical_access_index_;
  std::map<const Var*, VersionedReachableEventIndex> alias_read_index_;
  std::map<VarPtr, TypePtr> rewrites_;
  std::vector<RewriteJournalEntry> rewrite_journal_;
  std::set<VarPtr> visiting_;  // cycle guard

  void BuildAnalysis(const StmtPtr& func_body) {
    DefMapVisitor def_v;
    def_v.Run(func_body);
    defs_ = std::move(def_v.defs);
    stmt_order_ = std::move(def_v.stmt_order);
    stmt_end_order_ = std::move(def_v.stmt_end_order);
    analysis_root_ = def_v.root;

    ReverseBranchOrderVisitor reverse_v;
    reverse_v.Run(func_body);
    reverse_stmt_order_ = std::move(reverse_v.stmt_order);
    reverse_stmt_end_order_ = std::move(reverse_v.stmt_end_order);
    BuildAliasFamilies();
    BuildAccessIndexes(def_v.reads, def_v.writes);
  }

  static size_t FindSet(std::vector<size_t>& parent, size_t value) {
    while (parent[value] != value) {
      parent[value] = parent[parent[value]];
      value = parent[value];
    }
    return value;
  }

  static void UnionSets(std::vector<size_t>& parent, size_t lhs, size_t rhs) {
    lhs = FindSet(parent, lhs);
    rhs = FindSet(parent, rhs);
    if (lhs != rhs) parent[rhs] = lhs;
  }

  void BuildAccessIndexes(const std::map<const Stmt*, std::set<VarPtr>>& reads,
                          const std::map<const Stmt*, std::set<VarPtr>>& writes) {
    memref_component_.clear();
    physical_access_index_.clear();
    alias_read_index_.clear();

    std::map<const MemRef*, MemRefPtr> unique_memrefs;
    auto register_var = [&](const VarPtr& var) {
      if (auto tile = GetTileTypeWithMemRef(var->GetType())) {
        auto memref = GetDefinedMemRef(tile);
        unique_memrefs.emplace(memref.get(), memref);
      }
    };
    for (const auto& [var, def] : defs_) {
      (void)def;
      register_var(var);
    }
    for (const auto& [stmt, vars] : reads) {
      (void)stmt;
      for (const auto& var : vars) register_var(var);
    }
    for (const auto& [stmt, vars] : writes) {
      (void)stmt;
      for (const auto& var : vars) register_var(var);
    }

    std::vector<MemRefPtr> memrefs;
    memrefs.reserve(unique_memrefs.size());
    for (const auto& [ptr, memref] : unique_memrefs) {
      (void)ptr;
      memrefs.push_back(memref);
    }
    std::vector<size_t> parent(memrefs.size());
    for (size_t i = 0; i < parent.size(); ++i) parent[i] = i;

    std::map<const Var*, std::vector<size_t>> by_base;
    for (size_t i = 0; i < memrefs.size(); ++i) by_base[memrefs[i]->base_.get()].push_back(i);
    for (const auto& [base, members] : by_base) {
      (void)base;
      struct StaticInterval {
        int64_t begin;
        __int128 end;
        size_t index;
      };
      std::vector<StaticInterval> intervals;
      std::vector<size_t> dynamic;
      for (size_t index : members) {
        if (auto offset = As<ConstInt>(memrefs[index]->byte_offset_)) {
          intervals.push_back(StaticInterval{
              offset->value_, static_cast<__int128>(offset->value_) + memrefs[index]->size_, index});
        } else {
          dynamic.push_back(index);
        }
      }
      std::sort(intervals.begin(), intervals.end(), [](const auto& lhs, const auto& rhs) {
        return std::tie(lhs.begin, lhs.end) < std::tie(rhs.begin, rhs.end);
      });
      if (!intervals.empty()) {
        size_t representative = intervals.front().index;
        __int128 component_end = intervals.front().end;
        for (size_t i = 1; i < intervals.size(); ++i) {
          if (static_cast<__int128>(intervals[i].begin) < component_end) {
            UnionSets(parent, representative, intervals[i].index);
            component_end = std::max(component_end, intervals[i].end);
          } else {
            representative = intervals[i].index;
            component_end = intervals[i].end;
          }
        }
      }
      // Symbolic offsets may overlap every window in their allocation.
      if (!dynamic.empty() && !members.empty()) {
        for (size_t index : members) UnionSets(parent, dynamic.front(), index);
      }
    }

    std::map<size_t, size_t> dense_component;
    for (size_t i = 0; i < memrefs.size(); ++i) {
      const size_t root = FindSet(parent, i);
      auto [it, inserted] = dense_component.emplace(root, dense_component.size());
      (void)inserted;
      memref_component_[memrefs[i].get()] = it->second;
    }

    std::map<size_t, std::vector<std::pair<size_t, size_t>>> physical_events;
    std::map<const Var*, std::vector<VersionedReachableEventIndex::Event>> alias_events;
    auto record = [&](const Stmt* stmt, const VarPtr& var, bool is_read) {
      auto primary = stmt_order_.find(stmt);
      auto reverse = reverse_stmt_order_.find(stmt);
      if (primary == stmt_order_.end() || reverse == reverse_stmt_order_.end()) return;
      if (auto tile = GetTileTypeWithMemRef(var->GetType())) {
        auto component = memref_component_.find(GetDefinedMemRef(tile).get());
        if (component != memref_component_.end()) {
          physical_events[component->second].emplace_back(primary->second, reverse->second);
        }
      }
      if (is_read) {
        auto def = defs_.find(var);
        // Function inputs and other externally defined handles predate every
        // statement in the analysed body.
        const size_t definition = def == defs_.end() ? 0 : def->second.order;
        alias_events[ResolveAliasRoot(var).get()].push_back(
            VersionedReachableEventIndex::Event{primary->second, definition});
      }
    };
    for (const auto& [stmt, vars] : reads) {
      for (const auto& var : vars) record(stmt, var, true);
    }
    for (const auto& [stmt, vars] : writes) {
      for (const auto& var : vars) record(stmt, var, false);
    }
    for (auto& [component, events] : physical_events) {
      physical_access_index_[component].Build(std::move(events));
    }
    for (auto& [root, events] : alias_events) alias_read_index_[root].Build(std::move(events));
  }

  [[nodiscard]] bool IsInside(const StmtPtr& inner, const StmtPtr& outer) const {
    if (!inner || !outer) return false;
    auto inner_order = stmt_order_.find(inner.get());
    auto outer_begin = stmt_order_.find(outer.get());
    auto outer_end = stmt_end_order_.find(outer.get());
    return inner_order != stmt_order_.end() && outer_begin != stmt_order_.end() &&
           outer_end != stmt_end_order_.end() && outer_begin->second <= inner_order->second &&
           inner_order->second <= outer_end->second;
  }

  void BuildAliasFamilies() {
    alias_source_.clear();
    alias_root_cache_.clear();
    inplace_source_.clear();
    inplace_root_cache_.clear();
    accumulator_input_cache_.clear();
    non_accumulator_producers_.clear();

    const auto& registry = OpRegistry::GetInstance();
    for (const auto& [var, def] : defs_) {
      if (def.kind == VarDef::kIterArg) {
        if (def.iter_arg) {
          if (auto init = AsVarLike(def.iter_arg->initValue_)) alias_source_[var] = init;
        }
        continue;
      }
      if (def.kind == VarDef::kForReturn) {
        auto loop = As<ForStmt>(def.control_stmt);
        if (loop && def.return_idx < loop->iter_args_.size()) {
          alias_source_[var] = std::static_pointer_cast<const Var>(loop->iter_args_[def.return_idx]);
        }
        continue;
      }
      if (def.kind != VarDef::kAssign) continue;
      auto assign = As<AssignStmt>(def.assign_stmt);
      if (!assign) continue;

      if (auto source = AsVarLike(assign->value_)) {
        // A bare SSA assignment is semantic identity even when lowering has
        // temporarily left stale MemRef metadata on one side.
        alias_source_[var] = source;
        inplace_source_[var] = source;
        continue;
      }

      auto call = As<Call>(assign->value_);
      if (!call || !call->op_ || !registry.IsRegistered(call->op_->name_)) continue;
      if (!op_predicates::OutputInheritsSourceBuffer(call->op_->name_)) continue;
      const auto& entry = registry.GetEntry(call->op_->name_);
      if (auto reuse_idx = entry.GetOutputReusesInputArg();
          reuse_idx.has_value() && *reuse_idx < call->args_.size()) {
        if (auto source = AsVarLike(call->args_[*reuse_idx])) {
          alias_source_[var] = source;
          inplace_source_[var] = source;
        }
        continue;
      }
      if (!call->args_.empty()) {
        if (auto source = AsVarLike(call->args_[0])) alias_source_[var] = source;
      }
    }
  }

  VarPtr ResolveAliasRoot(const VarPtr& var) {
    auto cached = alias_root_cache_.find(var);
    if (cached != alias_root_cache_.end()) return cached->second;

    std::vector<VarPtr> path;
    std::set<const Var*> seen;
    VarPtr current = var;
    while (seen.insert(current.get()).second) {
      auto source = alias_source_.find(current);
      if (source == alias_source_.end()) break;
      path.push_back(current);
      current = source->second;
      auto root_cached = alias_root_cache_.find(current);
      if (root_cached != alias_root_cache_.end()) {
        current = root_cached->second;
        break;
      }
    }
    for (const auto& member : path) alias_root_cache_[member] = current;
    alias_root_cache_[var] = current;
    return current;
  }

  VarPtr ResolveInplaceRoot(const VarPtr& var) {
    auto cached = inplace_root_cache_.find(var);
    if (cached != inplace_root_cache_.end()) return cached->second;

    std::vector<VarPtr> path;
    std::set<const Var*> seen;
    VarPtr current = var;
    while (seen.insert(current.get()).second) {
      auto source = inplace_source_.find(current);
      if (source == inplace_source_.end()) break;
      path.push_back(current);
      current = source->second;
      auto root_cached = inplace_root_cache_.find(current);
      if (root_cached != inplace_root_cache_.end()) {
        current = root_cached->second;
        break;
      }
    }
    for (const auto& member : path) inplace_root_cache_[member] = current;
    inplace_root_cache_[var] = current;
    return current;
  }

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

  // Walk the IR; coalesce every accumulator if-phi we encounter.
  void VisitIfPhisForAccumulator(const StmtPtr& stmt) {
    if (!stmt) return;
    if (auto seq = As<SeqStmts>(stmt)) {
      for (const auto& s : seq->stmts_) VisitIfPhisForAccumulator(s);
    } else if (auto for_stmt = As<ForStmt>(stmt)) {
      VisitIfPhisForAccumulator(for_stmt->body_);
    } else if (auto if_stmt = As<IfStmt>(stmt)) {
      // Coalesce nested accumulator phis before their enclosing K-window phi.
      // Otherwise a later child rewrite can overwrite the outer target and
      // reintroduce a divergent L0C allocation.
      VisitIfPhisForAccumulator(if_stmt->then_body_);
      if (if_stmt->else_body_.has_value()) VisitIfPhisForAccumulator(if_stmt->else_body_.value());
      TryCoalesceAccIfPhi(if_stmt);
    } else if (auto scope = As<ScopeStmt>(stmt)) {
      VisitIfPhisForAccumulator(scope->body_);
    }
  }

  struct AccumulatorTarget {
    MemRefPtr memref;
    std::optional<MemorySpace> memory_space;
    // Canonical reused input whose pre-existing aliases must remain observable.
    // This is populated for both direct in-place accumulator producers and
    // branch-local accumulator loops.
    VarPtr reused_input;
  };

  // Resolve the canonical allocation of an in-place accumulator producer. A
  // bare SSA alias may still carry a stale type after a nested child rewrite,
  // so return the underlying producer's target rather than a boolean that would
  // make the caller accidentally select the alias's stale MemRef.
  std::optional<AccumulatorTarget> FindInplaceAccumulatorProducer(const VarPtr& var) {
    auto in_var = FindAccumulatorInput(var);
    if (!in_var) return std::nullopt;
    auto in_tile = CurrentTileType(in_var);
    if (!in_tile) return std::nullopt;
    // The registered reuse input is authoritative. Lowering may have left the
    // Call result typed on an older allocation; TryRetargetVar below repairs the
    // result onto this canonical input allocation before the phi is rewritten.
    return AccumulatorTarget{GetDefinedMemRef(in_tile), in_tile->GetMemorySpace(), in_var};
  }

  VarPtr FindAccumulatorInput(const VarPtr& var) {
    auto cached = accumulator_input_cache_.find(var);
    if (cached != accumulator_input_cache_.end()) return cached->second;
    if (non_accumulator_producers_.count(var) != 0) return nullptr;

    std::vector<VarPtr> path;
    std::set<const Var*> seen;
    VarPtr current = var;
    VarPtr result;
    while (seen.insert(current.get()).second) {
      auto known = accumulator_input_cache_.find(current);
      if (known != accumulator_input_cache_.end()) {
        result = known->second;
        break;
      }
      if (non_accumulator_producers_.count(current) != 0) break;
      path.push_back(current);

      auto def = defs_.find(current);
      if (def == defs_.end() || def->second.kind != VarDef::kAssign) break;
      auto assign = As<AssignStmt>(def->second.assign_stmt);
      if (!assign) break;
      if (auto alias = AsVarLike(assign->value_)) {
        current = alias;
        continue;
      }
      auto call = As<Call>(assign->value_);
      if (!call || !call->op_) break;
      const auto& registry = OpRegistry::GetInstance();
      if (!registry.IsRegistered(call->op_->name_)) break;
      auto reuse_idx = registry.GetEntry(call->op_->name_).GetOutputReusesInputArg();
      if (!reuse_idx.has_value() || *reuse_idx >= call->args_.size()) break;
      result = AsVarLike(call->args_[*reuse_idx]);
      break;
    }
    for (const auto& member : path) {
      if (result) {
        accumulator_input_cache_[member] = result;
      } else {
        non_accumulator_producers_.insert(member);
      }
    }
    return result;
  }

  /// True when ``var`` is an alias/in-place producer chain whose reused input
  /// ultimately reaches ``root``. This proves accumulator semantics without
  /// relying on the MemRefs already being coalesced: MemoryReuse may have just
  /// packed one member of the family onto another physical window, which is the
  /// mismatch this analysis exists to repair.
  bool IsInplaceChainRootedAt(const VarPtr& var, const VarPtr& root) {
    return ResolveInplaceRoot(var).get() == root.get();
  }

  /// Resolve a branch-local accumulator loop seeded by a value defined outside
  /// ``branch_scope``. Both locality halves are required: the loop itself must
  /// be inside the branch and its seed must be outside it.
  std::optional<AccumulatorTarget> FindExternallySeededAccumulatorLoop(const VarPtr& var,
                                                                       const IfStmtPtr& branch_scope) {
    auto it = defs_.find(var);
    if (it == defs_.end() || it->second.kind != VarDef::kForReturn) return std::nullopt;
    auto loop = As<ForStmt>(it->second.control_stmt);
    if (!loop || it->second.return_idx >= loop->iter_args_.size()) return std::nullopt;

    if (!IsInside(loop, branch_scope)) return std::nullopt;

    auto init = AsVarLike(loop->iter_args_[it->second.return_idx]->initValue_);
    auto init_tile = init ? CurrentTileType(init) : nullptr;
    if (!init || !CurrentTileType(var) || !init_tile) return std::nullopt;

    auto init_def = defs_.find(init);
    if (init_def != defs_.end() && IsInside(init_def->second.definition_stmt, branch_scope)) {
      return std::nullopt;
    }

    auto body_yield = FindYieldStmt(loop->body_);
    if (!body_yield || it->second.return_idx >= body_yield->value_.size()) return std::nullopt;
    auto yielded = AsVarLike(body_yield->value_[it->second.return_idx]);
    auto iter_arg = std::static_pointer_cast<const Var>(loop->iter_args_[it->second.return_idx]);
    if (!yielded || !IsInplaceChainRootedAt(yielded, iter_arg)) return std::nullopt;
    return AccumulatorTarget{GetDefinedMemRef(init_tile), init_tile->GetMemorySpace(), init};
  }

  // For an IfStmt whose branches yield an in-place accumulator on one side and a
  // fresh seed on the other (a different L0C buffer), retarget the seed onto the
  // accumulator buffer.  Scoped to Acc — the ISA case with no legal reconciling
  // move.  A declined retarget is a hard error (see CoalesceAccumulatorIfPhis).
  void TryCoalesceAccIfPhi(const IfStmtPtr& if_stmt) {
    if (!if_stmt->else_body_.has_value() || if_stmt->return_vars_.empty()) return;
    auto then_yield = FindYieldStmt(if_stmt->then_body_);
    auto else_yield = FindYieldStmt(if_stmt->else_body_.value());
    if (!then_yield || !else_yield) return;

    for (size_t i = 0; i < if_stmt->return_vars_.size(); ++i) {
      if (i >= then_yield->value_.size() || i >= else_yield->value_.size()) continue;
      auto then_var = AsVarLike(then_yield->value_[i]);
      auto else_var = AsVarLike(else_yield->value_[i]);
      if (!then_var || !else_var) continue;

      auto then_target = FindInplaceAccumulatorProducer(then_var);
      if (!then_target) then_target = FindExternallySeededAccumulatorLoop(then_var, if_stmt);
      auto else_target = FindInplaceAccumulatorProducer(else_var);
      if (!else_target) else_target = FindExternallySeededAccumulatorLoop(else_var, if_stmt);
      if (then_target.has_value() == else_target.has_value()) continue;  // need exactly one accumulator

      const bool then_acc = then_target.has_value();
      const VarPtr& acc_var = then_acc ? then_var : else_var;
      const VarPtr& seed_var = then_acc ? else_var : then_var;
      const AccumulatorTarget& target = then_acc ? *then_target : *else_target;

      auto seed_tile = CurrentTileType(seed_var);
      if (!seed_tile) continue;
      if (target.memory_space != MemorySpace::Acc) continue;  // Acc-only (no legal move)

      const bool seed_already_shared = SamePhysicalWindow(target.memref, GetDefinedMemRef(seed_tile));

      auto seed_def = defs_.find(seed_var);
      if (!seed_already_shared && seed_def == defs_.end()) continue;

      // The `check_liveness=false` bypass below is only sound when branch
      // exclusivity actually applies, which requires BOTH:
      //  (a) the seed producer is lexically *inside* this IfStmt's branch — a
      //      pre-if value yielded through the branch runs unconditionally and
      //      would clobber the accumulator the sibling in-place branch reads; and
      //  (b) the accumulator buffer is dead *within the branch* after the seed
      //      (exclusivity covers only cross-branch and post-if reads, not a
      //      same-branch tail read between the seed producer and the yield).
      // When either fails, leave the phi untouched here. YieldFixup will fail
      // loudly rather than emit an unsupported Acc->Acc move.
      if (!seed_already_shared) {
        if (!IsInside(seed_def->second.definition_stmt, if_stmt)) continue;
        if (!IsTargetDeadAfterDefinition(seed_def->second, target.memref,
                                         /*stop_at=*/if_stmt.get())) {
          continue;
        }
      }

      // A branch-local loop can reuse a seed defined outside the IfStmt only
      // when that external value is not observed independently after the phi.
      // Otherwise retargeting the sibling seed would clobber the external value
      // on the path where the loop does not execute.
      if (target.reused_input &&
          HasIndependentExternalUseAfterIf(if_stmt, if_stmt->return_vars_[i], target)) {
        continue;
      }

      // Coalescing one phi is atomic. Retargeting the accumulator side may
      // recursively plan several rewrites before the seed side is inspected;
      // a later decline must not leak that partial family into the pass-wide
      // rewrite map.
      RewriteTransaction transaction(*this);

      // Align any alias/ForReturn wrapper on the accumulator side with the
      // canonical producer target before using it as the phi's allocation.
      INTERNAL_CHECK_SPAN(TryRetargetVar(acc_var, target.memref, target.memory_space), acc_var->span_)
          << "Internal error: cannot align the accumulator branch with its canonical L0C allocation";

      // Now safe: (a)+(b) plus exclusivity cover every read of the target allocation, so we
      // bypass the global liveness (which would false-decline on the legitimate
      // post-if phi consumer). A remaining decline is a genuine "cannot coalesce
      // this Acc phi" — fail loud, since no legal Acc->Acc move exists.
      bool ok = seed_already_shared;
      if (!seed_already_shared) {
        if (seed_def->second.kind == VarDef::kAssign) {
          auto seed_assign = As<AssignStmt>(seed_def->second.assign_stmt);
          if (!seed_assign) continue;
          if (As<Call>(seed_assign->value_)) {
            ok = RetargetAssign(seed_var, seed_def->second, target.memref, target.memory_space,
                                /*check_liveness=*/false);
          } else if (AsVarLike(seed_assign->value_)) {
            ok = TryRetargetVar(seed_var, target.memref, target.memory_space);
          } else {
            continue;
          }
        } else if (seed_def->second.kind == VarDef::kForReturn) {
          ok = TryRetargetVar(seed_var, target.memref, target.memory_space);
        } else {
          continue;
        }
      }
      INTERNAL_CHECK_SPAN(ok, seed_var->span_)
          << "Internal error: cannot coalesce L0C accumulator across a peeled if-phi — seed producer '"
          << seed_var->name_hint_
          << "' refused retarget onto the accumulator buffer, which would force an illegal "
             "Acc->Acc tile.move.";

      // The phi itself is part of the semantic accumulator family. Pinning it
      // here is required by the PTOAS planner, which skips legacy MemoryReuse
      // and otherwise declares a separate head handle for the merged value.
      PlanRewrite(if_stmt->return_vars_[i], target.memref, target.memory_space);
      transaction.Commit();
    }
  }

  TileTypePtr CurrentTileType(const VarPtr& var) {
    auto it = rewrites_.find(var);
    auto type = (it != rewrites_.end()) ? it->second : var->GetType();
    return GetTileTypeWithMemRef(type);
  }

  MemRefPtr CurrentMemRef(const VarPtr& var) {
    auto tile = CurrentTileType(var);
    return tile ? GetDefinedMemRef(tile) : nullptr;
  }

  /// Current (possibly-rewritten) MemRef base of `var`.
  const Var* CurrentBase(const VarPtr& var) {
    auto tile = CurrentTileType(var);
    if (!tile) return nullptr;
    return GetDefinedMemRef(tile)->base_.get();
  }

  /// Attempts to rewrite `var`'s MemRef to `target` by walking its producer chain.
  /// Returns true if var already has target MemRef or a rewrite was planned.
  bool TryRetargetVar(const VarPtr& var, const MemRefPtr& target, std::optional<MemorySpace> target_memory) {
    if (SamePhysicalWindow(CurrentMemRef(var), target)) return true;  // already aligned
    if (!visiting_.insert(var).second) return false;                  // cycle
    struct Guard {
      std::set<VarPtr>* s;
      VarPtr v;
      ~Guard() { s->erase(v); }
    } g{&visiting_, var};

    RewriteTransaction transaction(*this);

    auto it = defs_.find(var);
    if (it == defs_.end()) return false;
    const auto& def = it->second;

    bool retargeted = false;
    if (def.kind == VarDef::kAssign) {
      retargeted = RetargetAssign(var, def, target, target_memory);
    } else if (def.kind == VarDef::kIfReturn) {
      retargeted = RetargetIfReturn(var, def, target, target_memory);
    } else if (def.kind == VarDef::kForReturn) {
      retargeted = RetargetForReturn(var, def, target, target_memory);
    } else if (def.kind == VarDef::kIterArg) {
      retargeted = RetargetIterArg(var, def, target, target_memory);
    }
    if (retargeted) transaction.Commit();
    return retargeted;
  }

  /// Retype a Var defined by an AssignStmt.
  ///
  /// `check_liveness` gates the general dead-at-assign check (IsTargetDeadAtAssign).
  /// It is true for the normal loop-carry retarget path.  It is set false only by
  /// CoalesceAccumulatorIfPhis: coalescing an IfStmt phi's two branch yields onto one
  /// buffer is always safe (the phi is redefined by exactly one branch at runtime, so
  /// the branches are mutually exclusive and the target's downstream liveness cannot be
  /// violated by a branch-local producer).  The op-legality checks below still apply.
  bool RetargetAssign(const VarPtr& var, const VarDef& def, const MemRefPtr& target,
                      std::optional<MemorySpace> target_memory, bool check_liveness = true) {
    auto assign = As<AssignStmt>(def.assign_stmt);
    INTERNAL_CHECK_SPAN(assign, var->span_) << "Internal error: kAssign VarDef must carry an AssignStmt";
    auto call = As<Call>(assign->value_);
    if (!call) {
      auto alias = AsVarLike(assign->value_);
      if (!alias) return false;
      auto alias_tile = GetTileTypeWithMemRef(alias->GetType());
      auto var_tile = GetTileTypeWithMemRef(var->GetType());
      if (!alias_tile || !var_tile ||
          !SamePhysicalWindow(GetDefinedMemRef(alias_tile), GetDefinedMemRef(var_tile))) {
        return false;
      }
      if (!TryRetargetVar(alias, target, target_memory)) return false;
      PlanRewrite(var, target, target_memory);
      return true;
    }
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
    //   4. Ops with a registered *fixed* output memory space (e.g. every
    //      `tile.add`-style elementwise op is `set_output_memory(Vec)`) when
    //      the target sits in a different space.  The op physically cannot
    //      write there, so retyping the LHS would only mint IR that claims a
    //      Vec-only op produces DDR.  Declining leaves the carry yielding a
    //      different buffer than its init, which YieldFixupMutator reconciles
    //      with a real move (see AlignLoopCarriesToInitMutator's contract).
    //      Same-space targets are unaffected: only the MemRef base moves.
    if (IsOutputMemoryInheritInput(entry)) {
      // Most view ops can carry a sub-region offset, so retargeting their LHS
      // directly would lose view-relative addressing. tile.set_validshape is
      // the narrow exception: it is a zero-offset, shape-preserving alias of
      // its first argument. Retarget that storage producer first, then record
      // the view result on the same target. This lets padded loop-carried Acc
      // initializers coalesce with their accumulator instead of leaving a
      // divergent Acc carry for YieldFixup to reject.
      if (!IsOp(call, "tile.set_validshape") || call->args_.empty()) return false;
      auto input_var = AsVarLike(call->args_[0]);
      if (!input_var) return false;
      auto input_tile = GetTileTypeWithMemRef(input_var->GetType());
      auto output_tile = GetTileTypeWithMemRef(var->GetType());
      if (!input_tile || !output_tile ||
          !SamePhysicalWindow(GetDefinedMemRef(input_tile), GetDefinedMemRef(output_tile))) {
        return false;
      }
      if (!TryRetargetVar(input_var, target, target_memory)) return false;
      PlanRewrite(var, target, target_memory);
      return true;
    }
    if (HasKwarg(*call, "target_memory") && !TargetMemoryKwargMatches(*call, target_memory)) {
      return false;
    }
    if (FixedOutputMemoryConflicts(entry, *call, target_memory)) return false;
    if (!entry.IsInplaceSafe()) {
      const size_t read_arg_count = IsA5Prelu(call) ? 2 : call->args_.size();
      if (CallReadsBase(*call, target->base_.get(), read_arg_count)) return false;
    }

    // Unconstrained: check liveness, then plan retype.  (Skipped for if-phi
    // branch coalescing, where branch exclusivity is a stronger guarantee.)
    if (check_liveness && !IsTargetDeadAtAssign(def, target)) return false;
    PlanRewrite(var, target, target_memory);
    return true;
  }

  /// True if any argument of the call is a TileType Var whose MemRef base
  /// is `target_base`.  Used to detect would-be in-place execution before
  /// we retype the output onto the same buffer.
  static bool CallReadsBase(const Call& call, const Var* target_base, size_t arg_count) {
    ExprReadBaseCollector c;
    for (size_t i = 0; i < std::min(arg_count, call.args_.size()); ++i) {
      c.VisitExpr(call.args_[i]);
    }
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

  /// True when the op resolves to a concrete output memory space that differs
  /// from `target_memory` — i.e. retyping the LHS would claim the op writes
  /// somewhere it physically cannot.  Ops that defer the choice (the resolver
  /// returns nullopt: inherit-input and the retargetable `target_memory`-kwarg
  /// ops) are not constrained here and are handled by their own guards above.
  static bool FixedOutputMemoryConflicts(const OpRegistryEntry& entry, const Call& call,
                                         std::optional<MemorySpace> target_memory) {
    if (!target_memory.has_value()) return false;
    const auto& spec = entry.GetMemorySpec();
    if (!spec.has_value() || !spec->deduce_output_memory) return false;
    // Resolve against the call's own kwargs so a kwarg-driven op reports the
    // space it actually produces rather than its registered default.
    auto fixed = spec->deduce_output_memory(call.kwargs_);
    return fixed.has_value() && *fixed != *target_memory;
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
  /// rewrites. Every recursive retarget attempt is transactional: if any arm
  /// declines, RollbackRewrites restores all entries written since that
  /// attempt's checkpoint, including successful nested attempts.
  void PlanRewrite(const VarPtr& var, const MemRefPtr& target, std::optional<MemorySpace> target_memory) {
    auto new_type = CloneTypeWithMemRef(var->GetType(), target, target_memory);
    auto previous = rewrites_.find(var);
    rewrite_journal_.push_back(RewriteJournalEntry{
        var, previous == rewrites_.end() ? std::nullopt : std::optional<TypePtr>(previous->second)});
    rewrites_[var] = new_type;
  }

  void RollbackRewrites(size_t checkpoint) {
    INTERNAL_CHECK(checkpoint <= rewrite_journal_.size()) << "Invalid rewrite transaction checkpoint";
    while (rewrite_journal_.size() > checkpoint) {
      auto entry = std::move(rewrite_journal_.back());
      rewrite_journal_.pop_back();
      if (entry.previous.has_value()) {
        rewrites_[entry.var] = *entry.previous;
      } else {
        rewrites_.erase(entry.var);
      }
    }
  }

  /// True when a branch-local accumulator loop's external seed is read after
  /// the enclosing IfStmt through a path other than the IfStmt return value.
  /// Such a read observes the pre-loop value on the sibling path, so placing
  /// that sibling's producer in the seed's window would be a clobber.
  bool HasIndependentExternalUseAfterIf(const IfStmtPtr& if_stmt, const VarPtr& if_return,
                                        const AccumulatorTarget& target) {
    if (!target.reused_input) return false;
    auto return_def = defs_.find(if_return);
    if (return_def == defs_.end() || return_def->second.kind != VarDef::kIfReturn) return true;
    auto primary_end = stmt_end_order_.find(if_stmt.get());
    auto root_end = stmt_end_order_.find(analysis_root_.get());
    if (primary_end == stmt_end_order_.end() || root_end == stmt_end_order_.end()) {
      return true;
    }
    auto reads = alias_read_index_.find(ResolveAliasRoot(target.reused_input).get());
    if (reads == alias_read_index_.end()) return false;
    auto if_begin = stmt_order_.find(if_stmt.get());
    if (if_begin == stmt_order_.end()) return true;
    return reads->second.HasAfterFromHandleDefinedBefore(primary_end->second + 1, root_end->second,
                                                         if_begin->second);
  }

  /// Is target's physical window untouched after the definition and before the
  /// nearest loop (or caller-supplied branch boundary) completes? The dual
  /// lexical ordering excludes mutually exclusive sibling arms without walking
  /// ancestor chains or rescanning subtrees.
  bool IsTargetDeadAfterDefinition(const VarDef& def, const MemRefPtr& target,
                                   const Stmt* stop_at = nullptr) {
    if (!target || !def.definition_stmt) return false;
    auto component = memref_component_.find(target.get());
    if (component == memref_component_.end()) return false;
    auto accesses = physical_access_index_.find(component->second);
    if (accesses == physical_access_index_.end()) return true;

    const Stmt* boundary = stop_at;
    if (!boundary && def.nearest_for) boundary = def.nearest_for.get();
    if (!boundary && analysis_root_) boundary = analysis_root_.get();
    auto boundary_end = stmt_end_order_.find(boundary);
    auto reverse_definition = reverse_stmt_order_.find(def.definition_stmt.get());
    if (boundary_end == stmt_end_order_.end() || reverse_definition == reverse_stmt_order_.end()) {
      return false;
    }
    if (def.kind == VarDef::kIfReturn || def.kind == VarDef::kForReturn) {
      auto reverse_end = reverse_stmt_end_order_.find(def.definition_stmt.get());
      if (reverse_end == reverse_stmt_end_order_.end()) return false;
      return !accesses->second.HasAfter(def.order + 1, boundary_end->second, reverse_end->second);
    }
    return !accesses->second.HasAfter(def.order + 1, boundary_end->second, reverse_definition->second);
  }

  bool IsTargetDeadAtAssign(const VarDef& def, const MemRefPtr& target, const Stmt* stop_at = nullptr) {
    INTERNAL_CHECK(def.kind == VarDef::kAssign)
        << "Internal error: IsTargetDeadAtAssign requires an assignment definition";
    return IsTargetDeadAfterDefinition(def, target, stop_at);
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
  /// set_validshape / tensor.view, ...) onto its input's reused buffer when
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

  Result Analyze(const StmtPtr& func_body, const std::vector<VarPtr>& entry_vars = {}) {
    // Function parameters have no defining AssignStmt. Seed Tile parameters at
    // the entry point so function-wide allocation planning includes their
    // MemRefs and body uses extend their lifetimes normally.
    for (const VarPtr& var : entry_vars) {
      const auto tile_type = As<TileType>(var->GetType());
      if (!tile_type || !tile_type->memref_.has_value()) continue;
      ordered_defs_.push_back(var);
      var_def_order_[var] = current_order_;
    }

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
      if (then_yield && i < then_yield->value_.size()) {
        add_source(then_yield->value_[i], then_start, then_end);
      }
      if (else_yield && i < else_yield->value_.size()) {
        add_source(else_yield->value_[i], else_start, else_end);
      }
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

  /// Resolve a loop carry's init to the variable that actually owns the buffer.
  ///
  /// A nested loop seeds its carry from the *enclosing* loop's IterArg, and an
  /// IterArg is an alias of its own init rather than a definition in its own
  /// right -- like return_vars, iter-args are deliberately absent from
  /// `ordered_defs_` / `var_def_order_`, so they carry no lifetime of their own.
  /// Walking the carry chain to the first non-IterArg therefore lands on the
  /// AssignStmt-defined tile whose buffer the whole loop nest shares, which is
  /// the variable a use of any return_var in that nest has to keep alive.
  ///
  /// Returns nullptr when the chain does not end at a tracked carrier; the
  /// caller then records no mapping, exactly as before.
  [[nodiscard]] VarPtr ResolveCarryInitCarrier(const ExprPtr& init_value) const {
    auto var = AsVarLike(init_value);
    // An init is always defined before its own loop, so the chain is acyclic in
    // well-formed IR; the visited set bounds it anyway rather than hanging.
    std::set<const Var*> seen;
    while (var) {
      auto iter_arg = As<IterArg>(var);
      if (!iter_arg) return var;
      if (!seen.insert(var.get()).second) return nullptr;
      var = AsVarLike(iter_arg->initValue_);
    }
    return nullptr;
  }

  void RegisterReturnVars(const std::vector<IterArgPtr>& iter_args, const std::vector<VarPtr>& return_vars) {
    size_t count = std::min(return_vars.size(), iter_args.size());
    for (size_t i = 0; i < count; ++i) {
      const auto& rv = return_vars[i];
      if (!As<TileType>(rv->GetType())) continue;

      // Map return_var -> the init's buffer carrier so that post-loop uses of the
      // return_var extend that carrier's lifetime (not the return_var's).
      // We do NOT register return_vars in ordered_defs_ -- they must not
      // participate in sharing group computation, which would inflate
      // group lifetimes and block unrelated reuse opportunities.
      auto init_var = ResolveCarryInitCarrier(iter_args[i]->initValue_);
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
LifetimeAnalysisResult AnalyzeAllocationLifetimesImpl(const StmtPtr& func_body,
                                                      const std::vector<VarPtr>& entry_vars = {}) {
  std::vector<LifetimeInterval> lifetimes;

  // Step 1: Walk full IR tree to collect variable defs, uses, and ordering
  LifetimeAnalyzer analyzer;
  auto result = analyzer.Analyze(func_body, entry_vars);

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

  // Pipeline-stage membership per interval representative (parsed once from the
  // defining Call's pipeline_membership attr, set by LowerPipelineLoops), plus
  // the subset whose defining op is a load (vs compute).
  std::map<const Var*, std::vector<std::pair<int32_t, int32_t>>> pipeline_membership;
  std::set<const Var*> pipeline_load_tiles;

  for (const auto& var : result.ordered_defs) {
    if (processed_vars.count(var)) {
      continue;
    }

    auto tile_type = As<TileType>(var->GetType());
    if (!tile_type || !tile_type->memref_.has_value()) {
      continue;
    }

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
    interval.size = GetDefinedMemRef(representative_tile_type)->size_;

    lifetimes.push_back(interval);

    // Record pipeline-stage membership for this interval's representative so the
    // reuse packer can forbid cross-stage coalescing. Every sharing-group member
    // is a tile of one pipeline clone and carries the same membership; take the
    // first non-empty one (views with no Call def simply contribute nothing).
    for (const auto& group_var : sharing_group) {
      auto dit = result.var_def_stmt.find(group_var);
      if (dit == result.var_def_stmt.end()) continue;
      auto assign = As<AssignStmt>(dit->second);
      if (!assign) continue;
      auto call = As<Call>(assign->value_);
      if (!call) continue;
      auto packed = call->GetAttr<std::string>(kPipelineMembershipAttr, std::string());
      if (!packed.empty()) {
        // Parse the membership string once here; the packer compares pre-parsed
        // vectors so it never re-parses in its O(N²) pairwise loop.
        pipeline_membership[interval.variable.get()] = ParsePipelineMembership(packed);
        if (IsOp(call, "tile.load") || IsOp(call, "tile.read")) {
          pipeline_load_tiles.insert(interval.variable.get());
        }
        break;
      }
    }

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

  return {lifetimes,
          var_sharing_groups,
          std::move(result.phi_family_ids),
          std::move(var_liveness),
          std::move(pipeline_membership),
          std::move(pipeline_load_tiles)};
}

// NOTE: The former tile-type reuse-compatibility gate (AreTileTypesCompatible)
// has been removed. PTO codegen binds a per-var alloc_tile to each tile, so two
// tiles that share a physical MemRef can legally carry different shapes, dtypes,
// or TileView attributes. In-place read/write hazards are handled precisely by
// not_inplace_safe() and per-operand forbid_output_alias() markers (see
// ForbidAliasCollector below).

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
// `HazardInputs` is collected by walking the function body (see
// `HazardInputCollector::Run` for why that walk runs twice):
//   - load_derived: vars produced by tile.load or by a legal view op chained
//     from a load (these alias the load's physical buffer).
//   - reads_tpop:   vars whose defining op consumes a tile.tpop_from_aic value.
// Both sets are keyed on Var identity and checked against the reuse-decision
// representatives (a sharing group's earliest-defined member, which for a
// writer is the writer's own output and for a load is the load itself).
//
// Reading an *operand* takes one extra step, because a value can reach the op
// through a loop carry.  An IterArg is never an AssignStmt def, so it is never a
// member of either set; but the carry chain it belongs to (init value, IterArg,
// yield value) has already been fused onto one MemRef base by
// MaterializeSemanticAliases, so an IterArg operand is classified by that base
// instead — see `HazardInputCollector::IsTainted`.  A carry also breaks program
// order, since the producer that taints the base may stand after the use it
// taints; `HazardInputCollector::Run` covers that with a second traversal.

// Resolve a tile Var to its MemRef base pointer (nullptr if it is not a tile or
// has no MemRef yet).  View ops share their source's base, so a view and its
// source resolve to the same pointer here.
static const Var* TileMemRefBase(const VarPtr& v) {
  auto t = As<TileType>(v->GetType());
  if (t && t->memref_.has_value() && t->memref_.value()->base_) return t->memref_.value()->base_.get();
  return nullptr;
}

/// Op names whose output aliases the input MemRef and lowers to a PTO view
/// instruction — kept in sync with the PTO buffer-reuse view allowlist.
static bool IsLegalTileViewOp(const OpPtr& op) {
  return IsOp(op, "tile.reshape") || IsOp(op, "tile.extract") || IsOp(op, "tile.slice") ||
         IsOp(op, "tile.fillpad") || IsOp(op, "tile.fillpad_inplace") || IsOp(op, "tile.transpose_view") ||
         IsOp(op, "tile.reinterpret_view") || IsOp(op, "tensor.slice");
}

using HazardInputs = AllocationHazardInputs;

class HazardInputCollector : public IRVisitor {
 public:
  /// Classify `body` in TWO traversals, then hand the result to `Take()`.
  ///
  /// One forward walk is not enough, because a value can reach a use through a
  /// loop back edge — from a producer that stands *after* that use in program
  /// order:
  ///
  /// ```
  /// for (carry,) in range(..., init=[non_tpop]):
  ///   w = tile.add(l, carry)        // from iteration 1 on, `carry` is a tpop value
  ///   p = tile.tpop_from_aic()      // ... but the producer is only seen here
  ///   yield p                       // p must-aliases carry's base
  /// ```
  ///
  /// A single walk classifies `w` before `p` taints the carry's base, so `w`
  /// escapes `reads_tpop` and its output may be coalesced onto `l`'s buffer.
  ///
  /// Two traversals suffice, and a third could never add anything: the base
  /// sets are complete after the first pass.  `tpop_bases_` only ever takes a
  /// `tile.tpop_from_aic` def, which is order-independent; `load_derived_bases_`
  /// additionally takes view outputs, and a view shares its source's base, so
  /// marking one inserts a base that is already present.  Only the *Var* sets
  /// grow in the second pass, and those propagate in program order within it.
  /// Both traversals are O(N), so the collector stays linear.
  void Run(const StmtPtr& body) {
    VisitStmt(body);
    VisitStmt(body);
  }

  void VisitStmt_(const AssignStmtPtr& op) override {
    if (GetTileTypeWithMemRef(op->var_->GetType())) {
      if (auto call = As<Call>(op->value_)) {
        // AsVarLike, not As<Var>: an operand may be an enclosing loop's IterArg
        // — the loop-carry shape of `tile.add(x, c_iter)` / `tile.matmul_acc(
        // c_iter, ...)`.  IterArg has its own ObjectKind, so As<Var> returns
        // null there and would drop the operand, and with it the load / tpop
        // taint the carry brings into the body.
        std::vector<VarPtr> input_vars;
        for (const auto& arg : call->args_) {
          if (auto v = AsVarLike(arg)) input_vars.push_back(v);
        }
        // load_derived closure: within one traversal defs precede uses, so a
        // view's source is already classified by the time we reach the view;
        // a source that only becomes tainted across a loop back edge is picked
        // up by the second traversal (see `Run`).
        if (IsOp(call, "tile.load")) {
          MarkLoadDerived(op->var_);
        } else if (IsLegalTileViewOp(call->op_)) {
          for (const VarPtr& in : input_vars) {
            if (IsLoadDerived(in)) {
              MarkLoadDerived(op->var_);
              break;
            }
          }
        }
        if (IsOp(call, "tile.tpop_from_aic")) MarkTpop(op->var_);
        for (const VarPtr& in : input_vars) {
          if (IsTpop(in)) {
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
  // Taint a var and the physical buffer it occupies.  The buffer mirror exists
  // for loop carries: an IterArg is never itself an AssignStmt def, so it can
  // never be a member of the Var-keyed sets no matter how it is spelled at the
  // use site.  MaterializeSemanticAliases (which runs immediately before this
  // pass) has already fused each carry — init value, IterArg, yield value —
  // onto ONE MemRef base, so a carry's taint is exactly the taint of the base it
  // must-aliases.
  void MarkLoadDerived(const VarPtr& v) {
    inputs_.load_derived.insert(v.get());
    if (const Var* base = TileMemRefBase(v)) load_derived_bases_.insert(base);
  }

  void MarkTpop(const VarPtr& v) {
    tpop_vars_.insert(v.get());
    if (const Var* base = TileMemRefBase(v)) tpop_bases_.insert(base);
  }

  [[nodiscard]] bool IsLoadDerived(const VarPtr& v) const {
    return IsTainted(inputs_.load_derived, load_derived_bases_, v);
  }

  [[nodiscard]] bool IsTpop(const VarPtr& v) const { return IsTainted(tpop_vars_, tpop_bases_, v); }

  // Var identity stays the exact test for an ordinary operand — two tiles that
  // merely share a base (a view and its source) are still classified by the view
  // allowlist above, not by the buffer.  Only a carry, which identity provably
  // cannot answer for, falls back to its must-aliased base.
  [[nodiscard]] static bool IsTainted(const std::unordered_set<const Var*>& vars,
                                      const std::unordered_set<const Var*>& bases, const VarPtr& v) {
    if (vars.count(v.get()) != 0) return true;
    if (!As<IterArg>(v)) return false;
    const Var* base = TileMemRefBase(v);
    return base != nullptr && bases.count(base) != 0;
  }

  HazardInputs inputs_;
  std::unordered_set<const Var*> tpop_vars_;
  std::unordered_set<const Var*> load_derived_bases_;  ///< MemRef bases holding a load_derived value
  std::unordered_set<const Var*> tpop_bases_;          ///< MemRef bases holding a tpop_from_aic value
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
using ForbidAliasMap = AllocationForbidAliasMap;

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
    // Multi-result calls are assigned to a tuple temporary and unpacked by
    // subsequent TupleGetItem assignments. Carry the call's semantic
    // no-alias inputs onto every physical tile result; the tuple temporary
    // itself has no allocation interval to constrain.
    if (const auto get_item = As<TupleGetItemExpr>(op->value_)) {
      if (const VarPtr tuple = AsVarLike(get_item->tuple_)) {
        const auto pending = tuple_forbidden_.find(tuple.get());
        if (pending != tuple_forbidden_.end()) RecordForOutput(op->var_, pending->second);
      }
    }

    if (auto call = As<Call>(op->value_); call && call->op_) {
      const auto& reg = OpRegistry::GetInstance();
      if (reg.IsRegistered(call->op_->name_)) {
        const auto& entry = reg.GetEntry(call->op_->name_);
        std::vector<VarPtr> forbidden_inputs;
        auto forbid_arg = [&](size_t i) {
          if (i < call->args_.size()) {
            if (auto v = AsVarLike(call->args_[i])) forbidden_inputs.push_back(v);
          }
        };
        if (!entry.IsInplaceSafe()) {
          // Non-in-place ops forbid output aliasing active inputs. A5 TPRELU
          // retains tmp (arg 2) in the ABI but does not read it, so only src
          // and slope remain active.
          const size_t forbidden_arg_count = IsA5Prelu(call) ? 2 : call->args_.size();
          for (size_t i = 0; i < forbidden_arg_count; ++i) forbid_arg(i);
        } else {
          for (size_t i : entry.ForbidOutputAliasArgs()) forbid_arg(i);
        }
        // A dtype-widening cast (output element wider than its input) cannot run
        // in place: element i is read at i*in_bytes but written at i*out_bytes,
        // so with out_bytes > in_bytes the write cursor outruns the read cursor
        // and clobbers input elements not yet converted -> corrupt results.
        // Narrowing / same-width casts are in-place-safe and keep the cross-dtype
        // reuse the removed gate enables, so forbid only the widening direction.
        if (IsOp(call, "tile.cast") && !call->args_.empty()) {
          auto out_t = As<TileType>(op->var_->GetType());
          auto in_t = As<TileType>(call->args_[0]->GetType());
          if (out_t && in_t && out_t->dtype_.GetBit() > in_t->dtype_.GetBit()) forbid_arg(0);
        }
        RecordForOutput(op->var_, forbidden_inputs);
        // tile.transpose is registered not_inplace_safe(), so its output is
        // already forbidden from aliasing any input above (pto.ttrans writes
        // dst directly from src on the scalar path — dst == src corrupts).
      }
    }
    IRVisitor::VisitStmt_(op);
  }

  ForbidAliasMap Take() { return std::move(forbidden_); }

 private:
  void RecordForOutput(const VarPtr& output, const std::vector<VarPtr>& forbidden_inputs) {
    if (!output || forbidden_inputs.empty()) return;
    if (As<TupleType>(output->GetType())) {
      tuple_forbidden_[output.get()] = forbidden_inputs;
      return;
    }
    if (!As<TileType>(output->GetType())) return;
    const auto rep_it = member_to_rep_.find(output.get());
    const Var* out_key = rep_it != member_to_rep_.end() ? rep_it->second : output.get();
    auto& recorded = forbidden_[out_key];
    recorded.insert(recorded.end(), forbidden_inputs.begin(), forbidden_inputs.end());
  }

  ForbidAliasMap forbidden_;
  std::map<const Var*, std::vector<VarPtr>> tuple_forbidden_;
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
  // A split AIV function needs the guard. The split may be signalled either as a
  // function-level mode (single-mode / standalone kernels carry ``split``) or
  // ONLY as the mode-agnostic ``split_aiv`` marker — multi-mode explicit
  // pl.split_aiv regions have their per-region modes lowered + erased by
  // LowerAutoVectorSplit, so no single function-level mode survives. Key on the
  // marker too so the guard still fires for them.
  if (func->HasAttr(kAttrSplitAiv) && func->GetAttr<bool>(kAttrSplitAiv, false)) return true;
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
 * tile intervals (M a subset of the N IR nodes). The capacity-gated shed loop
 * re-runs the O(M^2) FFD pack once per shed step, so the worst case is O(S·M^2)
 * where S = Σ(F_g − 1) is the total double-buffering depth shed across a space's
 * pipeline groups. S is bounded by (#pipeline groups × depth) — a small constant
 * on real kernels (groups few, depth 2–4) — so this stays the same O(M^2) class as
 * the prior greedy in practice; the FFD base was already O(M^2) pre-#1475.
 */
// Cross-group shed objective (see docs/en/dev/passes/34-memory_reuse.md, pipeline-stage guard): when a
// space overflows at every group's max-affordable depth,
// the packer lowers one pipeline group's double-buffering depth by a residue. The MaxRelief heuristic
// selects **which** group loses a level — lower score sheds first; ties always break by lowest group id
// (deterministic). MaxRelief is used unconditionally: the pluggable objective (and its ArrivalOrder
// alternative) was retired because it never diverged from MaxRelief on any real kernel — the multi-group
// forced-shed it needs to differ does not occur in practice.
struct ShedCandidate {
  int32_t group;         ///< pipeline group id (the tie-break key)
  uint64_t bytes_freed;  ///< align(slot_g): bytes reclaimed by dropping one residue buffer of this group
};

/// MaxRelief: shed the group that frees the most bytes per level ⇒ fewest levels lost to fit the space.
inline double ScoreMaxRelief(const ShedCandidate& c) { return -static_cast<double>(c.bytes_freed); }

/// One tile independently bound to a declared allocation, with its lifetime.
struct BoundMember {
  VarPtr var;
  int def_point;
  int last_use_point;
};

/// Group the tiles a user *independently bound* to each pinned buffer.
///
/// A tile whose defining op inherits its source's buffer (a view, an in-place
/// op, or a bare SSA alias) lands on the same base without the author binding it
/// there — it is the same data, so overlapping with its source is expected and
/// legal. Only independently produced tiles compete for the buffer's bytes, and
/// only those are checked for lifetime overlap.
class IndependentlyBoundCollector : public IRVisitor {
 public:
  IndependentlyBoundCollector(const std::set<const Var*>& pinned_bases,
                              const std::map<const Var*, std::pair<int, int>>& var_liveness)
      : pinned_bases_(pinned_bases), var_liveness_(var_liveness) {}

  /// One slot of a declared allocation: its base, and the byte offset that tells
  /// slots of the same allocation apart. `InitMemRef` has already consumed the
  /// slot index into that offset, so a null offset (a dynamic index) collapses to
  /// "some slot" — see the note on `by_slot`.
  struct SlotKey {
    const Var* base;
    int64_t byte_offset;

    bool operator<(const SlotKey& other) const {
      if (base != other.base) return base < other.base;
      return byte_offset < other.byte_offset;
    }
  };

  /// Slot -> the tiles bound to it, with their lifetimes.
  ///
  /// Keyed per *slot*, not per allocation: two tiles on different slots of one
  /// declared allocation are supposed to be live together — that is what a
  /// ping-pong is — and only tiles landing on the *same* slot can corrupt each
  /// other. Tiles whose offset is a runtime expression are skipped entirely,
  /// since which slot they occupy is not a static fact.
  std::map<SlotKey, std::vector<BoundMember>> by_slot;

  void VisitStmt_(const AssignStmtPtr& op) override {
    if (op && op->var_) Record(op);
    IRVisitor::VisitStmt_(op);
  }

 private:
  void Record(const AssignStmtPtr& op) {
    auto memref = GetTypeMemRef(op->var_->GetType());
    if (!memref.has_value() || !(*memref)->base_) return;
    const Var* base = (*memref)->base_.get();
    if (pinned_bases_.count(base) == 0) return;
    if (AsVarLike(op->value_)) return;  // bare SSA alias — same data
    if (auto call = As<Call>(op->value_);
        call && call->op_ && op_predicates::OutputInheritsSourceBuffer(call->op_->name_)) {
      return;
    }
    auto liveness = var_liveness_.find(op->var_.get());
    if (liveness == var_liveness_.end()) return;
    // A runtime slot index leaves no static slot to attribute the tile to, so
    // there is nothing to compare it against. The author owns the rotation in
    // that case; isolation from other allocations still holds.
    auto const_offset = As<ConstInt>((*memref)->byte_offset_);
    if (!const_offset) return;
    by_slot[{base, const_offset->value_}].push_back(
        {op->var_, liveness->second.first, liveness->second.second});
  }

  const std::set<const Var*>& pinned_bases_;
  const std::map<const Var*, std::pair<int, int>>& var_liveness_;
};

/// Reject a declared allocation whose independently bound tiles are co-live.
///
/// Binding two co-live tiles to one buffer is not a reuse decision, it is data
/// corruption: the later write lands on bytes the earlier tile still needs. The
/// packer refuses such a pairing among its own candidates; a hand-written
/// binding bypasses that path, so it is re-checked here. Touching is allowed —
/// one tile's last read may be the statement that produces the next.
///
/// Sorted sweep rather than a pairwise scan: a *valid* program is exactly one
/// where every pair is disjoint, so a pairwise check never exits early and would
/// be quadratic in the member count — which grows with the IR once a binding
/// inside a loop body is unrolled.
void ValidateDeclaredAllocs(const StmtPtr& body, const std::set<const Var*>& pinned_bases,
                            const std::map<const Var*, std::pair<int, int>>& var_liveness) {
  if (pinned_bases.empty()) return;
  IndependentlyBoundCollector collector(pinned_bases, var_liveness);
  collector.VisitStmt(body);

  for (auto& [slot, members] : collector.by_slot) {
    const Var* base = slot.base;
    std::sort(members.begin(), members.end(),
              [](const BoundMember& a, const BoundMember& b) { return a.def_point < b.def_point; });
    for (size_t i = 1; i < members.size(); ++i) {
      const BoundMember& prev = members[i - 1];
      const BoundMember& cur = members[i];
      CHECK_SPAN(prev.last_use_point <= cur.def_point, cur.var->span_)
          << "Tiles '" << prev.var->name_hint_ << "' and '" << cur.var->name_hint_
          << "' are both bound to the same slot of the declared allocation '" << base->name_hint_
          << "' but are live at the same time (" << prev.var->name_hint_ << ": [" << prev.def_point << ", "
          << prev.last_use_point << "], " << cur.var->name_hint_ << ": [" << cur.def_point << ", "
          << cur.last_use_point
          << "]). Sharing one slot would overwrite data that is still needed — put them on different "
             "slots, declare separate allocations, or bind only tiles whose lifetimes do not overlap.";
    }
  }
}

/// Collect the full byte extent of allocations declared by the author.
///
/// InitMemRef hoists every alloc to the function body's top level. When an
/// allocation exists, prepending it forms a SeqStmts; a function with no
/// allocation may legitimately retain a singleton body such as ReturnStmt.
/// A multi-slot declaration is larger than any one member MemRef, so DSA-RP
/// must take the extent from the alloc statement rather than reconstruct it
/// from members.
std::map<const Var*, uint64_t> CollectPinnedAllocSizes(const StmtPtr& body, const Span& span,
                                                       const std::string& consumer) {
  INTERNAL_CHECK_SPAN(body, span) << consumer << " expected a non-null function body";

  std::map<const Var*, uint64_t> pinned_allocations;
  auto collect = [&](const StmtPtr& stmt) {
    auto base = GetPinnedAllocBase(stmt);
    if (!base) return;
    auto assign = As<AssignStmt>(stmt);
    auto call = assign ? As<Call>(assign->value_) : nullptr;
    auto size = call && call->args_.size() >= 2 ? As<ConstInt>(call->args_[1]) : nullptr;
    INTERNAL_CHECK_SPAN(size && size->value_ > 0, stmt->span_)
        << consumer << " expected declared allocation '" << base->name_hint_
        << "' to have a positive constant byte extent after InitMemRef";
    pinned_allocations.emplace(base.get(), static_cast<uint64_t>(size->value_));
  };

  if (auto top_level = As<SeqStmts>(body)) {
    for (const auto& stmt : top_level->stmts_) collect(stmt);
  } else {
    collect(body);
  }
  return pinned_allocations;
}

std::map<VarPtr, VarPtr> IdentifyReuseOpportunities(
    const std::vector<LifetimeInterval>& lifetimes, const HazardInputs& hazard,
    const ForbidAliasMap& forbid_alias, const std::map<const Var*, std::set<int>>& phi_family_ids,
    const std::map<VarPtr, std::vector<VarPtr>>& sharing_groups,
    const std::map<const Var*, std::pair<int, int>>& var_liveness,
    const std::map<const Var*, std::vector<std::pair<int32_t, int32_t>>>& pipeline_membership,
    const std::set<const Var*>& pipeline_load_tiles,
    const std::map<MemorySpace, uint64_t>& reserved_end_by_space, const std::set<const Var*>& pinned_bases,
    const FunctionPtr& func, std::vector<Diagnostic>* out_hints) {
  std::map<VarPtr, VarPtr> reuse_map;

  // A declared allocation (`pl.Tile[..., pl.MemRef("name"), ...]`) has exactly the membership
  // the kernel author wrote. The packer neither adds tiles to it nor moves its
  // tiles into someone else's buffer — that is the whole point of naming it:
  // coalescing tiles whose lifetimes merely happen not to overlap introduces a
  // false dependency that serializes them.
  //
  // Resolved once per interval rather than inside `can_share`: that predicate is
  // the innermost step of the O(M^2) pack, which itself re-runs per shed step, so
  // a type cast + set lookup there would be paid M^2 times — and by every kernel,
  // including the vast majority that declare no allocation at all. Same reason
  // `pipeline_membership` is pre-parsed (see LifetimeAnalysisResult).
  std::vector<bool> is_pinned(lifetimes.size(), false);
  if (!pinned_bases.empty()) {
    for (size_t i = 0; i < lifetimes.size(); ++i) {
      if (!lifetimes[i].variable) continue;
      auto memref = GetTypeMemRef(lifetimes[i].variable->GetType());
      is_pinned[i] = memref.has_value() && (*memref)->base_ && pinned_bases.count((*memref)->base_.get()) > 0;
    }
  }

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

  // Pipeline ping-pong guard (symmetric): two tiles that share a pipeline group
  // with a *different* stage index belong to replicated clones the scheduler
  // overlaps, so collapsing them onto one buffer injects a false write-after-read
  // that serializes the stages — even though their program-order lifetimes are
  // disjoint (that disjointness is exactly what pipelining hides). Non-pipelined
  // tiles carry no membership and never trigger this. See ``kPipelineMembershipAttr``.
  //
  // Default (capacity-gated, #1475): protect concurrent clones in EVERY space —
  // including the L0 matmul spaces (Left/Right/Acc/Bias) and regardless of whether
  // a tile is a load or a `tile.move` result — up to the max-affordable
  // double-buffering depth `F_g` (see below). This is what fixes the L0b operand
  // collapse: the binding matmul operands are L0 `tile.move` results, so the legacy
  // predicate below skipped them and merged 8 → 1.
  //
  // Legacy predicate (the `!gated` fallback, used only when a space's capacity is
  // unknown or when `force_legacy` re-packs after an unfittable shed): the L0
  // spaces are exempt entirely and only *load* buffers get per-stage privacy.
  // Forbidding *all* cross-stage reuse without a capacity check is infeasible — F
  // full copies of every intermediate overflow the budget (e.g. stage=4 RMSNorm
  // needs 4x67KB > 188KB UB), and 2x64KB Right > 64KB L0b — so the ungated path
  // blocks iff the two tiles are same-group / different-stage AND at least one is a
  // load. The capacity gate replaces this coarse heuristic with an exact per-space
  // fit; the predicate is retained verbatim only as the never-worse-than-legacy floor.
  auto is_l0_space = [](MemorySpace s) {
    return s == MemorySpace::Left || s == MemorySpace::Right || s == MemorySpace::Acc ||
           s == MemorySpace::Bias || s == MemorySpace::LeftScale || s == MemorySpace::RightScale;
  };
  // Capacity-gated (#1475): keep software-pipelined operands in separate buffers so the pipeline
  // stages double-buffer instead of serializing on a shared buffer. #1900's `pipeline_membership` tags
  // give each operand its (group, stage); `stage` is the clone index 0..F-1 of a `pl.pipeline(stage=F)`
  // group whose clones run concurrently and must occupy distinct buffers. Each group `g` gets an
  // affordable residue count `F_g = min(D_g, ⌊C_s / slot_g⌋)`; clone stage k lands in residue
  // `ordinal(k) mod F_g`, so clones < F_g apart never share (exact double-buffering when F_g == D_g,
  // maximal spread when capacity forces F_g < D_g). This adjacency guarantee is structural and
  // model-free — it is `mod k`, kept. `F_g` is *mutable*: the FFD driver's shed loop lowers one group's
  // depth (re-packing, exact SpaceFootprint fit) when a space overflows, and `F_g == 1` collapses that
  // group to the legacy merge. Unknown capacity (`cap == 0`) ⇒ `F_g = 1` (conservative merge — never a
  // separation we cannot verify fits). The whole-space fit is now the allocator's realized footprint
  // (Primitive A, in the driver below), not a `Σ` estimate — so co-resident tiles and alignment are exact.
  std::map<std::pair<MemorySpace, int32_t>, uint64_t> group_slot;  // (space, group) -> max tile
  std::map<std::pair<MemorySpace, int32_t>, int32_t> group_depth;  // (space, group) -> F_g (mutable)
  std::map<std::pair<MemorySpace, int32_t>, std::map<int32_t, int32_t>>
      group_stage_ordinal;                                                   // stage->ordinal
  std::map<std::pair<MemorySpace, int32_t>, int32_t> group_requested_depth;  // (space, group) -> D_g (stages)
  std::set<MemorySpace> force_legacy_spaces;  // spaces whose gated packing overflowed → legacy fallback fired
  // Spaces with a *known* (non-zero) capacity — only these are capacity-gated. A space whose bound is
  // unknown (`cap == 0`, incl. no backend configured) falls through to the legacy predicate: gating it to
  // F_g == 1 would merge *everything*, dropping even the legacy non-L0 load-only separation (#1900) and
  // thus separating strictly LESS than legacy. Legacy-for-unknown makes "never worse than legacy" hold
  // for separation too, not just for overflow.
  std::set<MemorySpace> capacity_known_spaces;
  {
    const backend::Backend* be = backend::BackendConfig::IsConfigured() ? backend::GetBackend() : nullptr;
    std::map<std::pair<MemorySpace, int32_t>, std::set<int32_t>> group_stages;
    for (const auto& iv : lifetimes) {
      auto it = pipeline_membership.find(iv.variable.get());
      if (it == pipeline_membership.end()) continue;
      for (const auto& [g, st] : it->second) {
        auto key = std::make_pair(iv.memory_space, g);
        group_slot[key] = std::max(group_slot[key], iv.size);
        group_stages[key].insert(st);
      }
    }
    for (const auto& [key, slot] : group_slot) {
      // Optimistic per-slot depth using the *raw* space size (not cap − reserved_start): this is only the
      // initial F_g. The shed loop below re-checks with the exact reserved-aware SpaceFootprint and lowers
      // F_g further if the reserved region or co-resident tiles don't actually leave room.
      const uint64_t cap = be ? be->GetMemSize(key.first) : 0;
      const int32_t depth = static_cast<int32_t>(group_stages[key].size());
      int32_t f = 1;  // cap == 0 (unknown capacity) ⇒ this space is NOT gated (legacy predicate below)
      if (cap != 0 && slot != 0) {
        f = static_cast<int32_t>(
            std::max<uint64_t>(1, std::min<uint64_t>(static_cast<uint64_t>(depth), cap / slot)));
        capacity_known_spaces.insert(key.first);
      }
      group_depth[key] = f;
      group_requested_depth[key] = depth;
      int32_t ordinal = 0;  // stages may be sparse; compare dense ordinals mod F, not raw stage mod F
      for (const int32_t st : group_stages[key]) group_stage_ordinal[key][st] = ordinal++;
    }
  }
  // `force_legacy` (set by the driver's fallback below) makes the guard behave exactly like the
  // legacy predicate for a space whose gated packing overflowed — so the fallback is legacy by construction.
  bool force_legacy = false;
  auto pipeline_blocks = [&pipeline_membership, &pipeline_load_tiles, &is_l0_space, &group_depth,
                          &group_stage_ordinal, &force_legacy,
                          &capacity_known_spaces](const LifetimeInterval& a, const LifetimeInterval& b) {
    // The binding matmul operands live in L0 and are `tile.move` results, so the legacy guard's
    // `is_l0_space` exemption AND its load-only restriction both skip them — the per-stage operands
    // merge (8 → 1) and the cube matmuls serialize. The capacity-gated path (always on) protects
    // pipeline operands in every space and regardless of load/move, up to the max-affordable
    // double-buffering depth. A space whose capacity is unknown (`cap == 0`) is NOT gated — it uses the
    // legacy predicate, so gated packing is never worse than legacy there. (a and b share a memory
    // space — reuse only happens within a space.)
    const bool gated = !force_legacy && capacity_known_spaces.count(a.memory_space) != 0;
    if (!gated && is_l0_space(a.memory_space)) return false;  // legacy: L0 exempt
    auto ia = pipeline_membership.find(a.variable.get());
    if (ia == pipeline_membership.end()) return false;
    auto ib = pipeline_membership.find(b.variable.get());
    if (ib == pipeline_membership.end()) return false;
    if (gated) {
      // Keep separate iff a shared group's dense stage ordinals fall in different residues mod F_g (the
      // current, possibly-shed, affordable residue count). F_g == 1 ⇒ same residue ⇒ mergeable (legacy).
      // The whole-space fit is decided by the driver's exact SpaceFootprint, not here.
      for (const auto& [ga, sa] : ia->second) {
        for (const auto& [gb, sb] : ib->second) {
          if (ga != gb) continue;
          const auto key = std::make_pair(a.memory_space, ga);
          auto dit = group_depth.find(key);
          const int32_t k = (dit != group_depth.end() && dit->second > 0) ? dit->second : 1;
          int32_t ra = sa;
          int32_t rb = sb;
          auto omit = group_stage_ordinal.find(key);
          if (omit != group_stage_ordinal.end()) {
            auto oa = omit->second.find(sa);
            if (oa != omit->second.end()) ra = oa->second;
            auto ob = omit->second.find(sb);
            if (ob != omit->second.end()) rb = ob->second;
          }
          if (((ra % k) + k) % k != ((rb % k) + k) % k) return true;  // different buffer → separate
        }
      }
      return false;
    }
    if (!PipelineMembershipsConflict(ia->second, ib->second)) return false;  // not cross-stage
    // Legacy: block only if at least one side is a load buffer.
    return pipeline_load_tiles.count(a.variable.get()) != 0 ||
           pipeline_load_tiles.count(b.variable.get()) != 0;
  };

  // Can `cand` join a single physical buffer that already holds `member`?
  // Lifetimes must not overlap (touching is allowed: a buffer's reader is
  // consumed before the writer at the same statement produces its output), and
  // neither directional gate may block. No tile-type / size check is needed:
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
    if (pipeline_blocks(cand, member)) return false;  // symmetric — one call suffices
    return true;
  };

  // Group interval indices by memory space — reuse only happens within a space.
  std::map<MemorySpace, std::vector<size_t>> by_space;
  for (size_t i = 0; i < lifetimes.size(); ++i) {
    by_space[lifetimes[i].memory_space].push_back(i);
  }

  // Allocator policy for the exact per-space footprint (Primitive A) — only when a backend is configured.
  const backend::Backend* pack_be = backend::BackendConfig::IsConfigured() ? backend::GetBackend() : nullptr;
  auto alloc_policy = pack_be ? pack_be->CreateMemoryAllocatorPolicy() : nullptr;

  for (auto& [space_binding, indices_binding] : by_space) {
    // Bind the structured bindings to regular local references: the lambdas below capture by `[&]`,
    // and capturing a structured binding name directly is only a C++20 extension (project is C++17).
    auto& space = space_binding;
    auto& indices = indices_binding;
    // Largest-first; ties broken by definition order for determinism and so that
    // equal-size workloads reproduce the prior definition-order grouping.
    std::sort(indices.begin(), indices.end(), [&lifetimes](size_t a, size_t b) {
      if (lifetimes[a].size != lifetimes[b].size) return lifetimes[a].size > lifetimes[b].size;
      if (lifetimes[a].def_point != lifetimes[b].def_point) {
        return lifetimes[a].def_point < lifetimes[b].def_point;
      }
      return a < b;
    });

    // First-fit-decreasing pack with the current per-group residue counts (pipeline_blocks reads the
    // mutable F_g). Each buffer is a list of interval indices, element 0 the representative (largest,
    // earliest on ties). Pure — it does NOT touch reuse_map, since the shed loop below may re-pack.
    auto pack = [&]() {
      std::vector<std::vector<size_t>> buffers;
      // Parallel to `buffers`: a declared allocation's membership is fixed by the author,
      // so it opens its own slot and never takes another tile. Handling it here
      // rather than as a `can_share` gate keeps the check out of the O(M^2) inner
      // loop, and skips the whole buffer scan for a pinned candidate.
      std::vector<char> buffer_pinned;
      for (size_t idx : indices) {
        if (is_pinned[idx]) {
          buffers.push_back({idx});
          buffer_pinned.push_back(1);
          continue;
        }
        const auto& cand = lifetimes[idx];
        bool placed = false;
        for (size_t b = 0; b < buffers.size(); ++b) {
          if (buffer_pinned[b] != 0) continue;
          bool fits = true;
          for (size_t member_idx : buffers[b]) {
            if (!can_share(cand, lifetimes[member_idx])) {
              fits = false;
              break;
            }
          }
          if (!fits) continue;
          buffers[b].push_back(idx);
          placed = true;
          break;
        }
        if (!placed) {
          buffers.push_back({idx});
          buffer_pinned.push_back(0);
        }
      }
      return buffers;
    };

    std::vector<std::vector<size_t>> buffers = pack();

    // Graceful cross-group depth shed: while the space's exact allocator footprint (SpaceFootprint,
    // Primitive A) overflows, lower the largest-slot group's depth by one residue and re-pack. Re-running
    // the FFD (not merging in place) avoids in-place non-monotonicity, but the FFD is still not monotone
    // under the *relaxed* (F_g==1) predicate, so exhausting the shed does NOT guarantee a fit: the loop
    // ends in a from-scratch legacy re-pack + diagnostic (see the `!shed_group` branch below), which
    // guarantees no fit regression vs legacy. The shed objective is MaxRelief (largest slot first, tie
    // by lowest group id), applied unconditionally.
    if (alloc_policy) {
      const uint64_t cap = pack_be->GetMemSize(space);
      auto rit = reserved_end_by_space.find(space);
      const uint64_t reserved_start = rit != reserved_end_by_space.end() ? rit->second : 0;
      auto footprint = [&](const std::vector<std::vector<size_t>>& bufs) {
        SpaceFootprint fp(space, *alloc_policy, reserved_start);
        for (const auto& buf : bufs) {
          uint64_t slot = 0;
          for (size_t idx : buf) slot = std::max(slot, lifetimes[idx].size);
          (void)fp.OpenBuffer(slot);
        }
        return fp.HighWater();
      };
      // Bound the shed re-packs so the loop stays within the pass-complexity budget
      // (.claude/rules/pass-complexity.md). A legitimate shed converges in Σ_g(F_g−1) steps over the few
      // co-live groups, but the pipeline-group count can grow with generated IR, so cap the full re-packs
      // at a constant. The cap only trips in a pathological many-group overflow, where the legacy fallback
      // is the correct safe outcome anyway — keeping the shed at O(kMaxShedRepacks · M²), the same O(M²)
      // class as the base FFD packer, instead of O(groups · M²).
      constexpr int kMaxShedRepacks = 256;
      int shed_repacks = 0;
      while (cap != 0 && footprint(buffers) > cap) {
        // Pick the group to shed by MaxRelief; ties break by lowest group id (deterministic). Once the
        // re-pack budget is spent, stop picking (shed_group stays empty) and take the legacy fallback.
        const bool within_budget = (++shed_repacks <= kMaxShedRepacks);
        std::optional<int32_t> shed_group;
        double best_score = 0.0;
        if (within_budget) {
          for (const auto& [key, f] : group_depth) {
            if (key.first != space || f <= 1) continue;
            const uint64_t slot = group_slot.count(key) ? group_slot.at(key) : 0;
            const double s =
                ScoreMaxRelief(ShedCandidate{key.second, alloc_policy->AlignAddress(slot, space)});
            if (!shed_group || s < best_score || (s == best_score && key.second < *shed_group)) {
              best_score = s;
              shed_group = key.second;
            }
          }
        }
        if (!shed_group) {
          // No group left to shed — either every group is at depth 1 (shed exhausted) or the re-pack
          // budget was spent (pathological many-group overflow). FFD is NOT monotone under the relaxed
          // (F_g==1) predicate, so this packing may be worse than legacy; re-pack in legacy mode
          // (`force_legacy`, byte-identical to the legacy predicate) — no fit regression vs legacy, and if
          // legacy also overflows it is a genuine overflow legacy would hit too (AllocateMemoryAddr surfaces
          // it).
          force_legacy = true;
          buffers = pack();
          force_legacy = false;
          force_legacy_spaces.insert(space);
          // Fold into the diagnostic channel (Warning) so all capacity-degradation signals go through one
          // place (perf hint for a partial reduction below; this Warning for a space that fits at no depth).
          if (out_hints != nullptr) {
            const bool still_overflows = footprint(buffers) > cap;
            // A single physical buffer larger than the entire space is not a
            // reuse or pipeline-depth failure.  Legacy packing cannot repair
            // it, and the final allocator (or an earlier operation-specific
            // check such as AutoTileMatmulL0's explicit-L0 operand diagnostic)
            // reports the impossible allocation.  Do not obscure that error
            // with a fallback warning that suggests reuse or stage count is
            // responsible.  Keep warning for aggregate pressure: every buffer
            // fits individually, but their required co-residency does not.
            const bool has_intrinsically_oversized_buffer =
                std::any_of(buffers.begin(), buffers.end(), [&](const std::vector<size_t>& buf) {
                  uint64_t slot = 0;
                  for (size_t idx : buf) slot = std::max(slot, lifetimes[idx].size);
                  return slot > cap;
                });
            const std::string why =
                within_budget ? "at any double-buffering depth" : "within the shed-repack budget";
            if (!has_intrinsically_oversized_buffer) {
              out_hints->emplace_back(
                  DiagnosticSeverity::Warning, "MemoryReuse", 0,
                  "capacity-gated reuse could not fit memory space " + MemorySpaceToString(space) + " " +
                      why + "; fell back to the legacy packing" +
                      (still_overflows ? " (which also overflows — reduce tile size or stage count)" : "") +
                      ".",
                  func ? func->span_ : Span::unknown());
            }
          }
          break;
        }
        group_depth[std::make_pair(space, *shed_group)] -= 1;
        buffers = pack();
      }
    }

    // Commit the final packing: members [1..] reuse the representative [0]'s MemRef. The base coalescing
    // lets resolve_base() chase a view whose owning tile is reused onto the representative's buffer.
    for (const auto& buf : buffers) {
      const VarPtr& representative = lifetimes[buf.front()].variable;
      for (size_t m = 1; m < buf.size(); ++m) {
        const auto& cand = lifetimes[buf[m]];
        reuse_map[cand.variable] = representative;
        if (const Var* cb = TileMemRefBase(cand.variable)) {
          if (const Var* rb = physical_base(representative)) {
            if (cb != rb) base_remap[cb] = rb;
          }
        }
      }
    }
  }

  // Loud diagnostic (perf hint): a pipeline group whose achieved depth F_g fell below the requested D_g
  // means the capacity gate could not honor the programmer's `pl.pipeline(stage=D)` — stages k and k+F_g
  // share a buffer and re-serialize (the false WAR the pipeline meant to avoid). Surface it (correct, just
  // slower) with the concrete fix, rather than silently degrading. Spaces that hit the legacy fallback
  // already emitted a Warning above, so skip them here to avoid a double signal.
  if (out_hints != nullptr) {
    for (const auto& [key, achieved] : group_depth) {
      if (capacity_known_spaces.count(key.first) == 0) continue;  // unknown capacity ⇒ not gated
      if (force_legacy_spaces.count(key.first) != 0) continue;    // already warned at the space level
      auto rit = group_requested_depth.find(key);
      const int32_t requested = rit != group_requested_depth.end() ? rit->second : achieved;
      if (achieved >= requested) continue;  // fully double-buffered as requested — nothing to report
      const uint64_t slot = group_slot.count(key) != 0 ? group_slot.at(key) : 0;
      const uint64_t cap = pack_be != nullptr ? pack_be->GetMemSize(key.first) : 0;
      // Effective (free) capacity is the total minus the reserved region — the same reserved_start the exact
      // SpaceFootprint fit begins at. Co-resident non-pipeline tiles and other pipeline groups also consume
      // the space, but they are not a single subtractable constant, so the byte threshold is only exact when
      // this operand's own footprint is the binding constraint. Distinguish the two shed causes so the fix
      // stays honest: (a) slot-bound — `slot*requested` overflows even an otherwise-empty free region, so
      // shrink to `free_cap/requested`; (b) space-pressure — it would fit alone, so the fix is to relieve the
      // co-residents, not shrink this tile (which already satisfies the per-slot bound).
      const uint64_t reserved =
          reserved_end_by_space.count(key.first) != 0 ? reserved_end_by_space.at(key.first) : 0;
      const uint64_t free_cap = cap > reserved ? cap - reserved : 0;
      // Use the *aligned* slot — the same per-buffer increment SpaceFootprint bumps by — so the slot-bound
      // vs space-pressure classification and the byte threshold match the real fit (raw `slot` under-counts
      // when the tile isn't an alignment multiple).
      const uint64_t aligned_slot =
          alloc_policy != nullptr ? alloc_policy->AlignAddress(slot, key.first) : slot;
      const uint64_t need = aligned_slot * static_cast<uint64_t>(requested);
      const bool slot_bound = free_cap == 0 || need > free_cap;
      // Source-agnostic wording: `pipeline_membership` is stamped both by an explicit `pl.pipeline(stage=)`
      // and by compiler-synthesized pipelines (e.g. #1900's cross-core skew clones), so blame "software
      // pipelining", not the user's `stage=`, and offer `pl.pipeline(stage=)` only as an example.
      std::ostringstream msg;
      msg << "software pipelining requested depth " << requested << " for pipeline group " << key.second
          << " in " << MemorySpaceToString(key.first) << ", but only " << achieved << " of " << requested
          << " buffers fit (" << aligned_slot << " B per stage, " << free_cap << " B free";
      if (reserved != 0) msg << " after " << reserved << " B reserved";
      msg << ") — stages " << achieved << " apart share storage and serialize. ";
      if (slot_bound) {
        msg << "This operand alone needs " << need << " B for depth " << requested
            << "; shrink the per-stage tile to <= "
            << (requested > 0 ? free_cap / static_cast<uint64_t>(requested) : free_cap)
            << " B, or reduce the pipeline depth (e.g. `pl.pipeline(stage=)`) to " << achieved << ".";
      } else {
        msg << "The operand would fit depth " << requested
            << " on its own, but co-resident buffers / other pipeline groups over-subscribe the space; "
            << "relieve the co-residents (smaller or fewer co-live tiles) or reduce the pipeline depth to "
            << achieved << ".";
      }
      out_hints->emplace_back(DiagnosticSeverity::PerfHint, "MemoryReuse", 0, "PH-MR-001", msg.str(),
                              func ? func->span_ : Span::unknown());
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
        // Symbolic offsets retain the exact relative-address expression:
        // target_offset + (member_offset - representative_offset). Collapsing a
        // dynamic member onto the target MemRef would silently select the
        // representative's window.
        const MemRefPtr curr_memref = curr_tile_type->memref_.value_or(nullptr);
        auto rebase_memref = [&](const TileTypePtr& tile) -> std::optional<MemRefPtr> {
          if (!tile->memref_.has_value() || !curr_memref) return source_memref;
          const auto& old = tile->memref_.value();
          auto old_off = As<ConstInt>(old->byte_offset_);
          auto curr_off = As<ConstInt>(curr_memref->byte_offset_);
          if (!old_off || !curr_off) {
            ExprPtr relative;
            if (structural_equal(old->byte_offset_, curr_memref->byte_offset_)) {
              relative = MakeZeroByteOffset();
            } else if (curr_off && curr_off->value_ == 0) {
              relative = old->byte_offset_;
            } else {
              relative = MakeSub(old->byte_offset_, curr_memref->byte_offset_, old->span_);
            }
            auto rebased_offset = AddByteOffsets((*source_memref)->byte_offset_, relative);
            return std::make_shared<MemRef>((*source_memref)->base_, rebased_offset, old->size_, old->span_,
                                            (*source_memref)->is_pinned_, (*source_memref)->slot_count_,
                                            (*source_memref)->slot_index_);
          }
          const __int128 wide_rel = static_cast<__int128>(old_off->value_) - curr_off->value_;
          INTERNAL_CHECK_SPAN(wide_rel >= 0 && wide_rel <= std::numeric_limits<int64_t>::max() &&
                                  wide_rel + old->size_ <= (*source_memref)->size_,
                              old->span_)
              << "Internal error: sharing-group member offset " << old_off->value_
              << " cannot be represented inside reuse target (size " << (*source_memref)->size_ << ")";
          const int64_t rel = static_cast<int64_t>(wide_rel);
          if (rel == 0 && old->size_ == (*source_memref)->size_) return source_memref;
          auto rel_expr = std::make_shared<ConstInt>(rel, DataType::INDEX, Span::unknown());
          return std::make_shared<MemRef>((*source_memref)->base_,
                                          AddByteOffsets((*source_memref)->byte_offset_, rel_expr),
                                          old->size_, old->span_, (*source_memref)->is_pinned_,
                                          (*source_memref)->slot_count_, (*source_memref)->slot_index_);
        };

        // Rebase the representative through the same path as every member of
        // its semantic sharing group.  A smaller tile may reuse a larger
        // allocation; substituting source_memref wholesale here would give the
        // representative the larger window while its aliases retain their
        // original window, splitting one loop-carried accumulator family.
        auto new_tile_type = std::dynamic_pointer_cast<const TileType>(
            CloneTypeWithMemRefAndRemapExprs(curr_tile_type, rebase_memref(curr_tile_type),
                                             [this](const ExprPtr& expr) { return VisitExpr(expr); }));

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
      // Addresses, not allocations: an iter_arg left on a sibling slot of one
      // multi-slot allocation names different bytes and still has to be moved to
      // init's. Anything not provably init's address is realigned -- this only
      // retypes a node, so realigning an unprovable pair costs nothing while
      // leaving it keeps a possibly-wrong slot in the metadata.
      bool ia_memref_differs =
          init_tile && ia_tile && ia_tile->memref_.has_value() &&
          CompareBaseAddress(GetDefinedMemRef(ia_tile), init_memref) != AddressRelation::kSame;
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
            CompareBaseAddress(GetDefinedMemRef(rv_tile), init_memref) != AddressRelation::kSame) {
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
 *   same buffer. When they differ, insert a tile.move before yield, except that
 *   divergent Acc buffers are rejected because Acc->Acc moves are unsupported.
 * - IfStmt: MemoryReuse may change MemRefs of variables inside branches.
 *   Patch return_vars to match the yield value's MemRef, using the same Acc guard.
 */
class YieldFixupMutator : public IRMutator {
 public:
  YieldFixupMutator() = default;

  /// `fixup_if_stmts=false` reconciles only ForStmt carries. Used under
  /// memory_planner=PtoAS, where PTO codegen already re-points a branch-local
  /// producer at the phi handle (#1956/#1985) and copies in anything it declines
  /// to re-point — a copy-free path that an IR-level `tile.move` would displace
  /// with an extra buffer plus a `pto.tmov`. Loop carries have no such codegen
  /// path, so they still need the move.
  explicit YieldFixupMutator(bool fixup_if_stmts) : fixup_if_stmts_(fixup_if_stmts) {}

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
    // One index over the carry buffers serves both the overlap rejection below
    // and the conflict graph in OrderCarryCopies.
    const std::vector<MemRefPtr> carry_ranges = CollectCarryRanges(for_stmt);
    const CarryRangeIndex carry_index(carry_ranges);
    RejectOverlappingCarryBuffers(for_stmt, carry_index);

    std::vector<CarryCopy> copies;
    for (size_t i = 0; i < yield_stmt->value_.size() && i < for_stmt->iter_args_.size(); ++i) {
      // AsVarLike, not As<Var>: a nested loop both yields and seeds carries with
      // the enclosing loop's IterArg, whose own ObjectKind As<Var> does not match.
      // Skipping those left the nested carry with no writeback at all.
      auto yield_var = AsVarLike(yield_stmt->value_[i]);
      if (!yield_var) continue;

      // Yielding this position's own iter_arg leaves the carry untouched, so
      // there is nothing to write back. Short-circuit before comparing buffers:
      // it is the one case where the two sides are the same storage by
      // construction, whatever their (possibly still stale) types say.
      if (yield_var.get() == static_cast<const Var*>(for_stmt->iter_args_[i].get())) continue;

      auto yield_tile = GetTileTypeWithMemRef(yield_var->GetType());
      if (!yield_tile) continue;
      auto yield_memref = GetDefinedMemRef(yield_tile);

      // Get the initValue's MemRef (the actual buffer used at runtime)
      auto init_var = AsVarLike(for_stmt->iter_args_[i]->initValue_);
      if (!init_var) continue;
      auto init_tile = GetTileTypeWithMemRef(init_var->GetType());
      if (!init_tile) continue;
      auto init_memref = GetDefinedMemRef(init_tile);

      // Compare addresses, not allocations: two slots of one pl.MemRef(slots=N)
      // share a base, so "same allocation" would skip the copy that moves the
      // value from one slot to the other and silently carry the wrong slot.
      const AddressRelation relation = CompareBaseAddress(yield_memref, init_memref);
      if (relation == AddressRelation::kSame) continue;
      // Two symbolic offsets into one allocation may or may not name the same
      // bytes. Copying is wrong if they coincide (a tile.move onto itself, which
      // Acc cannot lower at all) and skipping is wrong if they do not, so say so
      // rather than pick one and hope.
      CHECK_SPAN(relation == AddressRelation::kDifferent, for_stmt->span_)
          << "Loop-carried value '" << for_stmt->iter_args_[i]->name_hint_
          << "' is yielded at a slot of its own allocation that cannot be shown equal to, or "
             "distinct from, the slot it is carried in, so whether the carry needs a copy is "
             "undecidable here. Subscript the declared allocation with the same expression at both "
             "ends, or give the carry its own allocation.";

      // MemRef mismatch — create tile.move to copy yield value into initValue's buffer
      auto target_memory = init_tile->GetMemorySpace();
      auto [moved_var, move_stmt] = CreateTileMove(yield_var, init_memref, target_memory);

      copies.push_back(CarryCopy{i, yield_var, std::move(moved_var), std::move(move_stmt), yield_memref,
                                 init_memref, target_memory});
    }

    if (copies.empty()) {
      // Even without tile.move insertion, iter_arg and return_var may have stale MemRefs
      // (MemoryReuse updates initValue/yield but not iter_arg/return_var types).
      // Ensure all 4 loop-carry variables share the same MemRef (= initValue's).
      return PatchIterArgsAndReturnVars(for_stmt, yield_stmt);
    }

    // `pl.yield_` rebinds every carry simultaneously, so the copies that realize
    // it must not observe each other's writes. Order them first.
    std::vector<StmtPtr> spill_stmts;
    const std::vector<size_t> order = OrderCarryCopies(&copies, carry_index, &spill_stmts);

    // Build new yield values with moved vars substituted
    std::vector<ExprPtr> new_yield_values = yield_stmt->value_;
    for (const auto& copy : copies) {
      new_yield_values[copy.index] = copy.moved_var;
    }
    auto new_yield = MutableCopy(yield_stmt);
    new_yield->value_ = std::move(new_yield_values);

    // A spill reads a carry buffer before any copy writes one, so every spill
    // precedes every copy.
    std::vector<StmtPtr> move_stmts = std::move(spill_stmts);
    move_stmts.reserve(move_stmts.size() + order.size());
    for (size_t idx : order) move_stmts.push_back(copies[idx].move_stmt);

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
    if (!fixup_if_stmts_) return result;
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
      // AsVarLike, not As<Var>: a branch may yield the enclosing loop's IterArg
      // unchanged (`if cond: acc = f(acc)` leaves `yield acc_iter` on the other
      // arm).  IterArg has its own ObjectKind, so As<Var> returns null there and
      // the arm would silently skip its reconciling move while the phi's
      // return_var is still patched onto the sibling arm's buffer -- leaving the
      // phi buffer unwritten on that path and feeding garbage into the carry.
      VarPtr then_var =
          (then_yield && i < then_yield->value_.size()) ? AsVarLike(then_yield->value_[i]) : nullptr;
      VarPtr else_var =
          (else_yield && i < else_yield->value_.size()) ? AsVarLike(else_yield->value_[i]) : nullptr;

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
        // Addresses, not allocations: an arm sitting on a sibling slot of the same
        // allocation still has to be copied into the phi's slot.
        const AddressRelation arm = CompareBaseAddress(else_memref, target_memref);
        CHECK_SPAN(arm != AddressRelation::kUnknown, if_stmt->span_)
            << "Branch result '" << if_stmt->return_vars_[i]->name_hint_
            << "' is produced at a slot of its own allocation that cannot be shown equal to, or "
               "distinct from, the slot the other branch produces, so whether this arm needs a copy "
               "is undecidable here. Subscript the declared allocation with the same expression in "
               "both branches, or give the branches separate allocations.";
        if (arm == AddressRelation::kDifferent) {
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
        // Metadata only, so realign anything not provably already the target.
        if (CompareBaseAddress(rv_memref, target_memref) != AddressRelation::kSame) {
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

 public:
  /// Allocations this fixup had to create (cycle spills), for the caller to
  /// hoist onto the function-body head. Empty for every loop whose carries do
  /// not form a copy cycle, which is the overwhelmingly common case.
  std::vector<StmtPtr> TakePendingAllocs() { return std::move(pending_allocs_); }

 private:
  bool fixup_if_stmts_ = true;

  /// One reconciling copy for one loop carry: `dst_memref <- source`.
  struct CarryCopy {
    size_t index;                           ///< iter_arg / yield position
    VarPtr source;                          ///< what the copy reads
    VarPtr moved_var;                       ///< the copy's LHS, replacing the yield value
    StmtPtr move_stmt;                      ///< the `tile.move` itself
    MemRefPtr src_memref;                   ///< byte range the copy reads
    MemRefPtr dst_memref;                   ///< the carry buffer the copy writes
    std::optional<MemorySpace> dst_memory;  ///< memory space of `dst_memref`
  };

  std::vector<StmtPtr> pending_allocs_;
  uint64_t spill_counter_ = 0;

  /// Byte ranges grouped by allocation, so overlap questions are index lookups
  /// rather than all-pairs scans.
  ///
  /// Within an allocation the constant-offset ranges are sorted by start, which
  /// makes both users linear-after-sort: the pairwise-overlap check only has to
  /// compare neighbours (sorted by start, if any two ranges overlap then some
  /// adjacent pair does), and a query binary-searches to its first candidate and
  /// stops at the first range past its end. A range whose offset is not constant
  /// has no position on that axis, so it is kept aside and treated as touching
  /// everything in its allocation — the same conservative answer
  /// `MemRef::MayAlias` gives.
  class CarryRangeIndex {
   public:
    /// `ranges[i]` is the byte range owned by carry `i`; a null entry is skipped.
    explicit CarryRangeIndex(const std::vector<MemRefPtr>& ranges) {
      for (size_t i = 0; i < ranges.size(); ++i) {
        if (!ranges[i]) continue;
        const Var* base = ranges[i]->base_.get();
        auto offset = As<ConstInt>(ranges[i]->byte_offset_);
        if (!offset) {
          unpositioned_[base].push_back(i);
          continue;
        }
        const int64_t begin = offset->value_;
        positioned_[base].push_back(Entry{begin, RangeEnd(begin, ranges[i]->size_, ranges[i]->span_), i});
      }
      for (auto& [base, entries] : positioned_) {
        static_cast<void>(base);
        std::sort(entries.begin(), entries.end(),
                  [](const Entry& a, const Entry& b) { return a.begin < b.begin; });
      }
    }

    /// Call `fn(i)` for every indexed range that may overlap `query`.
    template <typename Fn>
    void ForEachOverlapping(const MemRefPtr& query, const Fn& fn) const {
      const Var* base = query->base_.get();
      if (auto loose = unpositioned_.find(base); loose != unpositioned_.end()) {
        for (size_t i : loose->second) fn(i);
      }
      auto bucket = positioned_.find(base);
      if (bucket == positioned_.end()) return;
      const auto& entries = bucket->second;

      auto offset = As<ConstInt>(query->byte_offset_);
      if (!offset) {  // unknown position — conservatively touches the whole allocation
        for (const Entry& entry : entries) fn(entry.index);
        return;
      }
      const int64_t begin = offset->value_;
      const int64_t end = RangeEnd(begin, query->size_, query->span_);
      // Entries are sorted by start, so the first candidate is the first one
      // starting at or after `begin`, less the single earlier one that may still
      // cover `begin`.
      auto it = std::lower_bound(entries.begin(), entries.end(), begin,
                                 [](const Entry& entry, int64_t value) { return entry.begin < value; });
      if (it != entries.begin() && std::prev(it)->end > begin) --it;
      for (; it != entries.end() && it->begin < end; ++it) {
        if (it->end > begin) fn(it->index);
      }
    }

    /// Every pair of indexed ranges that overlaps, as (lower index, higher index).
    [[nodiscard]] std::vector<std::pair<size_t, size_t>> OverlappingPairs() const {
      std::vector<std::pair<size_t, size_t>> pairs;
      for (const auto& [base, entries] : positioned_) {
        for (size_t n = 1; n < entries.size(); ++n) {
          if (entries[n].begin < entries[n - 1].end) {
            pairs.emplace_back(std::minmax(entries[n - 1].index, entries[n].index));
          }
        }
        // An unpositioned range conservatively covers its whole allocation, so it
        // conflicts with every other range there. One witness is enough to report.
        auto loose = unpositioned_.find(base);
        if (loose != unpositioned_.end() && !entries.empty()) {
          pairs.emplace_back(std::minmax(loose->second.front(), entries.front().index));
        }
      }
      for (const auto& [base, loose] : unpositioned_) {
        static_cast<void>(base);
        if (loose.size() > 1) pairs.emplace_back(std::minmax(loose[0], loose[1]));
      }
      return pairs;
    }

   private:
    struct Entry {
      int64_t begin;
      int64_t end;
      size_t index;
    };

    /// One past the last byte of `[begin, begin + size)`, as a signed offset.
    ///
    /// `MemRef::size_` is unsigned and the interval arithmetic below is signed,
    /// so a size past `INT64_MAX` or a sum that wraps would silently corrupt both
    /// the sort order and every overlap answer derived from it. No on-chip buffer
    /// comes anywhere near that, so a violation is a corrupt MemRef rather than
    /// something to accommodate.
    static int64_t RangeEnd(int64_t begin, uint64_t size, const Span& span) {
      constexpr auto kMax = std::numeric_limits<int64_t>::max();
      INTERNAL_CHECK_SPAN(size <= static_cast<uint64_t>(kMax), span)
          << "Internal error: MemRef size " << size << " does not fit a signed byte offset";
      const auto signed_size = static_cast<int64_t>(size);
      INTERNAL_CHECK_SPAN(begin <= kMax - signed_size, span)
          << "Internal error: MemRef byte range [" << begin << ", +" << size << ") overflows";
      return begin + signed_size;
    }
    std::map<const Var*, std::vector<Entry>> positioned_;
    std::map<const Var*, std::vector<size_t>> unpositioned_;
  };

  /// The byte range each carry owns, indexed by iter_arg position (null when the
  /// carry is not a tile with a MemRef).
  static std::vector<MemRefPtr> CollectCarryRanges(const ForStmtPtr& for_stmt) {
    std::vector<MemRefPtr> ranges(for_stmt->iter_args_.size());
    for (size_t i = 0; i < for_stmt->iter_args_.size(); ++i) {
      // AsVarLike: a nested loop seeds its carries from the enclosing loop's
      // IterArg, which As<Var> would skip -- and skipping it is exactly how two
      // nested carries seeded from one outer carry would slip past the check below.
      auto init_var = AsVarLike(for_stmt->iter_args_[i]->initValue_);
      if (!init_var) continue;
      auto init_tile = GetTileTypeWithMemRef(init_var->GetType());
      if (!init_tile) continue;
      ranges[i] = GetDefinedMemRef(init_tile);
    }
    return ranges;
  }

  /// Reject a loop whose carries do not each own their buffer.
  ///
  /// Two carries whose buffers overlap cannot both survive an iteration: whatever
  /// order the writebacks run in, the second one destroys the first. The usual
  /// source is seeding two `pl.range` carries from one tile (`lag1 = one;
  /// lag2 = one`), which makes both carries that tile's buffer. That is a real
  /// limitation of the carry lowering rather than a compiler invariant, so it is
  /// reported to the author -- and reported here, where the carries are still
  /// named, rather than downstream as a degenerate `tile.move` onto itself.
  ///
  /// Byte ranges, not allocations: disjoint slices of one declared multi-slot
  /// buffer are independent carries and stay legal.
  static void RejectOverlappingCarryBuffers(const ForStmtPtr& for_stmt, const CarryRangeIndex& index) {
    for (const auto& [lhs, rhs] : index.OverlappingPairs()) {
      CHECK_SPAN(false, for_stmt->span_)
          << "Loop-carried values '" << for_stmt->iter_args_[lhs]->name_hint_ << "' and '"
          << for_stmt->iter_args_[rhs]->name_hint_
          << "' share the same on-chip buffer, so one iteration cannot preserve both. Give each "
             "loop-carried tile its own initial value -- build them with separate ops instead of "
             "seeding both from the same tile.";
    }
  }

  /// Order the carry copies so none overwrites a buffer another still reads.
  ///
  /// `pl.yield_` rebinds every carry at once, but the copies realizing it run in
  /// sequence, so copy A must precede copy B whenever A reads the buffer B
  /// writes. Emitting them in iter_arg order instead collapses a shift register:
  /// for `lag2 = lag1; lag1 = v`, writing lag1's buffer first makes lag2 read the
  /// new lag1 and the loop silently carries `lag2 == lag1` ([#2481]).
  ///
  /// Returns indices into `*copies` in emission order. A copy cycle — the swap
  /// `a, b = b, a`, where each carry's value lives in the other's buffer — has no
  /// valid order, so its members are spilled into scratch buffers first (appended
  /// to `*spills`, which the caller emits ahead of every copy) and their copies
  /// read the spills instead. A spill destination is a fresh allocation nothing
  /// else writes, so spilling removes every outgoing edge of the spilled copy.
  ///
  /// The graph is built once and consumed incrementally: a spill drops only the
  /// spilled copy's own outgoing edges, so the traversal resumes where it stalled
  /// instead of rebuilding anything. Victims come from the residual graph's
  /// strongly connected components, so a node merely downstream of a cycle is
  /// never spilled, and one victim set is enough to leave the residual acyclic —
  /// the traversal stalls at most once.
  ///
  /// Cost is O(k log k + E) in one loop's carry count k: `carry_index` answers each
  /// source with a binary search plus one step per range it actually overlaps, so
  /// k carries on disjoint slots of one allocation cost k lookups rather than k^2
  /// pairwise tests, and every edge is relaxed at most once.
  std::vector<size_t> OrderCarryCopies(std::vector<CarryCopy>* copies, const CarryRangeIndex& carry_index,
                                       std::vector<StmtPtr>* spills) {
    const size_t count = copies->size();

    // A copy writes its carry's byte range, so `carry_index` — built over exactly
    // those ranges — already indexes every destination. Map the carry positions it
    // reports back onto copy positions; a carry with no copy has nothing to order.
    std::map<size_t, size_t> copy_of_carry;
    for (size_t j = 0; j < count; ++j) copy_of_carry.emplace((*copies)[j].index, j);

    // A copy reads its source's byte range and writes its destination's, so two
    // copies conflict exactly when one's source overlaps the other's destination.
    std::vector<std::vector<size_t>> successors(count);
    std::vector<size_t> in_degree(count, 0);
    for (size_t i = 0; i < count; ++i) {
      carry_index.ForEachOverlapping((*copies)[i].src_memref, [&](size_t carry) {
        auto hit = copy_of_carry.find(carry);
        if (hit == copy_of_carry.end() || hit->second == i) return;
        successors[i].push_back(hit->second);  // i reads what hit->second overwrites
        ++in_degree[hit->second];
      });
    }

    std::vector<size_t> order;
    order.reserve(count);
    std::vector<bool> emitted(count, false);
    std::vector<size_t> ready;
    for (size_t i = 0; i < count; ++i) {
      if (in_degree[i] == 0) ready.push_back(i);
    }

    auto release = [&](size_t node) {
      for (size_t next : successors[node]) {
        if (--in_degree[next] == 0 && !emitted[next]) ready.push_back(next);
      }
      successors[node].clear();
    };

    while (order.size() < count) {
      if (!ready.empty()) {
        const size_t node = ready.back();
        ready.pop_back();
        if (emitted[node]) continue;
        // Kahn: every predecessor is already emitted, and an edge means "must run
        // before", so appending here builds the emission order directly.
        emitted[node] = true;
        order.push_back(node);
        release(node);
        continue;
      }

      // Nothing is runnable, so every remaining copy sits on or behind a cycle.
      // Take victims from the strongly connected components rather than from the
      // first node that still has an outgoing edge: such a node need not be on a
      // cycle at all, and spilling one that merely sits downstream of one costs a
      // scratch buffer without unblocking anything. One victim per component, all
      // of them in this round, so k/2 independent two-cycles cost a single pass.
      const std::vector<size_t> victims = CycleVictims(successors, emitted);
      INTERNAL_CHECK_SPAN(!victims.empty(), (*copies)[0].source->span_)
          << "Internal error: loop-carry copy ordering stalled with no cycle to break";
      // CycleVictims returns a set that leaves the residual acyclic, so Kahn
      // drains the rest without stalling again.
      for (size_t victim : victims) {
        SpillCarrySource(&(*copies)[victim], spills);
        release(victim);
        if (in_degree[victim] == 0 && !emitted[victim]) ready.push_back(victim);
      }
    }
    return order;
  }

  /// The strongly connected components of `nodes`, using only edges between
  /// nodes still marked active.
  ///
  /// Tarjan, iteratively so a deep chain cannot overflow the stack. Restricting
  /// to a node subset is what lets the caller re-decompose one component without
  /// walking the whole graph again.
  static std::vector<std::vector<size_t>> StronglyConnectedComponents(
      const std::vector<std::vector<size_t>>& successors, const std::vector<char>& active,
      const std::vector<size_t>& nodes) {
    constexpr size_t kUnvisited = std::numeric_limits<size_t>::max();
    const size_t count = successors.size();
    std::vector<size_t> index(count, kUnvisited);
    std::vector<size_t> lowlink(count, 0);
    std::vector<char> in_subset(count, 0);
    std::vector<char> on_stack(count, 0);
    for (size_t node : nodes) in_subset[node] = 1;

    std::vector<size_t> component_stack;
    std::vector<std::vector<size_t>> components;
    size_t next_index = 0;

    for (size_t root : nodes) {
      if (!active[root] || index[root] != kUnvisited) continue;
      index[root] = lowlink[root] = next_index++;
      component_stack.push_back(root);
      on_stack[root] = 1;
      std::vector<std::pair<size_t, size_t>> frames{{root, size_t{0}}};

      while (!frames.empty()) {
        // Re-read the frame each step: descending can reallocate the vector.
        const size_t node = frames.back().first;
        if (frames.back().second < successors[node].size()) {
          const size_t next = successors[node][frames.back().second++];
          if (!in_subset[next] || !active[next]) continue;
          if (index[next] == kUnvisited) {
            index[next] = lowlink[next] = next_index++;
            component_stack.push_back(next);
            on_stack[next] = 1;
            frames.emplace_back(next, size_t{0});
          } else if (on_stack[next]) {
            lowlink[node] = std::min(lowlink[node], index[next]);
          }
          continue;
        }

        if (lowlink[node] == index[node]) {
          std::vector<size_t> component;
          while (true) {
            const size_t member = component_stack.back();
            component_stack.pop_back();
            on_stack[member] = 0;
            component.push_back(member);
            if (member == node) break;
          }
          components.push_back(std::move(component));
        }

        frames.pop_back();
        if (!frames.empty()) {
          const size_t parent = frames.back().first;
          lowlink[parent] = std::min(lowlink[parent], lowlink[node]);
        }
      }
    }
    return components;
  }

  /// A set of copies whose spilling leaves the residual graph acyclic.
  ///
  /// Every cycle lies inside one strongly connected component, so it is enough
  /// to make each component acyclic. Spilling a node drops all of its outgoing
  /// edges, which breaks every cycle through it -- but a component holding
  /// several cycles that do not all pass through one node stays cyclic after a
  /// single spill, so what is left of that component is decomposed again. The
  /// re-decomposition is confined to the component's own nodes: the caller gets
  /// one acyclic residual from one call and never re-walks the whole graph.
  ///
  /// Victims are always on a cycle at the moment they are chosen, so no scratch
  /// buffer is spent on a copy that merely sits downstream of one. The set is
  /// not guaranteed minimal -- a minimum feedback vertex set is NP-hard, and the
  /// lowest-numbered member is picked instead.
  ///
  /// Cost is O(k + E) for the first decomposition plus, per component that needs
  /// more than one spill, another pass over that component alone.
  static std::vector<size_t> CycleVictims(const std::vector<std::vector<size_t>>& successors,
                                          const std::vector<bool>& emitted) {
    const size_t count = successors.size();
    std::vector<char> active(count, 0);
    std::vector<size_t> all_nodes;
    for (size_t i = 0; i < count; ++i) {
      if (emitted[i]) continue;
      active[i] = 1;
      all_nodes.push_back(i);
    }

    std::vector<size_t> victims;
    std::vector<std::vector<size_t>> pending{std::move(all_nodes)};
    while (!pending.empty()) {
      const std::vector<size_t> subset = std::move(pending.back());
      pending.pop_back();
      for (auto& component : StronglyConnectedComponents(successors, active, subset)) {
        const bool has_self_edge = component.size() == 1 &&
                                   std::find(successors[component[0]].begin(), successors[component[0]].end(),
                                             component[0]) != successors[component[0]].end();
        if (component.size() < 2 && !has_self_edge) continue;

        const size_t victim = *std::min_element(component.begin(), component.end());
        victims.push_back(victim);
        active[victim] = 0;  // its outgoing edges are gone once it reads a spill

        // The rest of this component may still hold a cycle the victim was not on.
        std::vector<size_t> remainder;
        for (size_t member : component) {
          if (member != victim) remainder.push_back(member);
        }
        if (remainder.size() > 1) pending.push_back(std::move(remainder));
      }
    }
    return victims;
  }

  /// Redirect one carry copy through a fresh scratch buffer, so the copy no
  /// longer reads a buffer its siblings overwrite. Emits the scratch allocation
  /// into `pending_allocs_` for the caller to hoist.
  void SpillCarrySource(CarryCopy* copy, std::vector<StmtPtr>* spills) {
    auto src_tile = GetTileTypeWithMemRef(copy->source->GetType());
    INTERNAL_CHECK_SPAN(src_tile, copy->source->span_)
        << "Internal error: a loop-carry copy source must be a TileType with a MemRef";
    auto src_memref = GetDefinedMemRef(src_tile);
    auto src_memory = src_tile->GetMemorySpace();
    INTERNAL_CHECK_SPAN(src_memory.has_value(), copy->source->span_)
        << "Internal error: a loop-carry copy source must carry a memory space to spill";

    // `mem_<space>_carry_spill_N` cannot collide with InitMemRef's
    // `mem_<space>_<N>`, which never has a non-numeric segment before its
    // counter, and the counter is per-function so two spills cannot collide
    // either. The name is cosmetic, but it has to stay unique for the
    // print -> parse roundtrip to rebind the same allocation.
    std::string space_str = MemorySpaceToString(*src_memory);
    std::transform(space_str.begin(), space_str.end(), space_str.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    const std::string base_name = "mem_" + space_str + "_carry_spill_" + std::to_string(spill_counter_++);
    auto scratch_base = std::make_shared<Var>(base_name, GetPtrType(), copy->source->span_);
    auto scratch_memref = std::make_shared<MemRef>(scratch_base, static_cast<int64_t>(0), src_memref->size_,
                                                   copy->source->span_);
    pending_allocs_.push_back(CreateAllocStatement(scratch_memref, *src_memory));

    auto [spilled_var, spill_stmt] = CreateTileMove(copy->source, scratch_memref, src_memory);
    spills->push_back(std::move(spill_stmt));

    auto [moved_var, move_stmt] = CreateTileMove(spilled_var, copy->dst_memref, copy->dst_memory);
    copy->source = spilled_var;
    copy->moved_var = std::move(moved_var);
    copy->move_stmt = std::move(move_stmt);
    copy->src_memref = scratch_memref;
  }

  // Create a tile.move operation that copies source into target_memref's buffer.
  // Returns (moved_var, move_assign_stmt).
  std::pair<VarPtr, StmtPtr> CreateTileMove(const VarPtr& source, const MemRefPtr& target_memref,
                                            std::optional<MemorySpace> target_memory) {
    INTERNAL_CHECK_SPAN(target_memory.has_value(), source->span_)
        << "Internal error: target TileType must have memory_space for tile.move";
    auto source_tile = GetTileTypeWithMemRef(source->GetType());
    INTERNAL_CHECK_SPAN(source_tile, source->span_)
        << "Internal error: YieldFixup tile.move source must be a TileType with MemRef";
    INTERNAL_CHECK_SPAN(
        !(source_tile->GetMemorySpace() == MemorySpace::Acc && target_memory.value() == MemorySpace::Acc),
        source->span_)
        << "Internal error: MemoryReuse cannot reconcile divergent L0C accumulator buffers with "
           "tile.move; accumulator control-flow values must be coalesced before YieldFixup.";
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
      auto init_var = AsVarLike(for_stmt->iter_args_[i]->initValue_);
      if (!init_var) continue;
      auto init_tile = GetTileTypeWithMemRef(init_var->GetType());
      if (!init_tile) continue;
      auto init_memref = GetDefinedMemRef(init_tile);

      // Patch iter_arg if its MemRef differs from initValue's. Addresses, not
      // allocations: an iter_arg left on a sibling slot of one multi-slot
      // allocation is a different buffer and still has to be re-pointed. These
      // sites only retype a node, they never move data, so anything not provably
      // init's address is repointed: an unprovable pair costs nothing to realign
      // and would otherwise leave a possibly-wrong slot in the metadata.
      auto ia_tile = As<TileType>(for_stmt->iter_args_[i]->GetType());
      if (ia_tile && ia_tile->memref_.has_value()) {
        auto ia_memref = GetDefinedMemRef(ia_tile);
        if (CompareBaseAddress(ia_memref, init_memref) != AddressRelation::kSame) {
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
      auto yield_var = AsVarLike(yield_stmt->value_[i]);
      if (!yield_var) continue;
      auto yield_tile = GetTileTypeWithMemRef(yield_var->GetType());
      if (!yield_tile) continue;
      auto yield_memref = GetDefinedMemRef(yield_tile);
      auto rv_tile = As<TileType>(new_return_vars[i]->GetType());
      if (!rv_tile || !rv_tile->memref_.has_value()) continue;
      auto rv_memref = GetDefinedMemRef(rv_tile);
      if (CompareBaseAddress(rv_memref, yield_memref) != AddressRelation::kSame) {
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
  if (!IsOp(call, "tile.alloc") && !IsOp(call, "tensor.alloc")) return false;
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

/// Strip the transient ``pipeline_membership`` attr from every Call once
/// MemoryReuse has consumed it (in ComputeLifetimes). The attr exists only to
/// carry pipeline-stage identity from LowerPipelineLoops to here; leaving it on
/// the IR would ride downstream into later passes and codegen. Stripping at the
/// end of MemoryReuse keeps the post-reuse IR clean.
class StripPipelineMembershipMutator : public IRMutator {
 public:
  ExprPtr VisitExpr_(const CallPtr& op) override {
    auto visited = IRMutator::VisitExpr_(op);
    auto call = As<Call>(visited);
    if (!call || !call->HasAttr(kPipelineMembershipAttr)) return visited;
    auto new_attrs = StripAttr(call->attrs_, kPipelineMembershipAttr);
    return std::make_shared<Call>(call->op_, call->args_, call->kwargs_, std::move(new_attrs),
                                  call->GetType(), call->span_);
  }
};

/// Reconcile bare-Var SSA identity copies whose LHS/RHS buffers diverged.
///
/// An AssignStmt whose value is a bare Var (`x = y`, not a Call) is a pure SSA
/// rename — `x` must alias `y`'s buffer.  CoalesceAccumulatorIfPhis retargets an
/// if-phi's producers + return_var onto the accumulator buffer, which can leave a
/// downstream copy of the (now-moved) return_var still typed on the old buffer:
///   c: Tile[..., mem_acc_17] = c_phi_2   // c_phi_2 was retargeted to mem_acc_5
/// Codegen would then store from the stale buffer.  This single forward pass
/// retypes every such copy's LHS to its RHS's MemRef and substitutes the LHS's
/// downstream uses, so the whole rename chain follows.  A no-op when no identity
/// copy has a buffer mismatch (the common case).
class NormalizeIdentityCopyBuffersMutator : public IRMutator {
 public:
  ExprPtr VisitExpr_(const VarPtr& op) override {
    auto it = subst_.find(op);
    return it != subst_.end() ? it->second : op;
  }

  StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
    // Only bare-Var identity copies are pure renames; Call producers are not.
    auto src_var = AsVarLike(op->value_);
    if (src_var) {
      auto new_src = AsVarLike(VisitExpr(op->value_));  // follow prior substitutions
      auto lhs_tile = new_src ? GetTileTypeWithMemRef(op->var_->GetType()) : nullptr;
      auto rhs_tile = new_src ? GetTileTypeWithMemRef(new_src->GetType()) : nullptr;
      if (lhs_tile && rhs_tile &&
          !SamePhysicalWindow(GetDefinedMemRef(lhs_tile), GetDefinedMemRef(rhs_tile))) {
        auto new_lhs = std::make_shared<Var>(op->var_->name_hint_, new_src->GetType(), op->var_->span_);
        subst_[op->var_] = new_lhs;
        return std::make_shared<AssignStmt>(new_lhs, new_src, op->span_);
      }
    }
    // An in-place accumulator Call (GetOutputReusesInputArg, e.g. tile.matmul_acc)
    // is not a bare rename, but its output MemRef must alias its reused input's.
    // Retargeting and pipeline/control-flow lowering can expose a stale output
    // type; normalize it even when the input itself was not substituted during
    // this traversal (the non-divisor K-peel matmul_acc tail is one example).
    if (auto reanchored = ReanchorInplaceOutput(op)) return reanchored;
    return IRMutator::VisitStmt_(op);
  }

 private:
  /// Re-anchor an in-place op's output onto its reused input's buffer, preserving
  /// the output's tile metadata (shape/dtype/space) and swapping only the MemRef.
  /// This also repairs a producer/output mismatch introduced by control-flow or
  /// pipeline lowering even when the input itself was not substituted in this
  /// traversal. Returns nullptr when the allocation already agrees.
  StmtPtr ReanchorInplaceOutput(const AssignStmtPtr& op) {
    auto call = As<Call>(op->value_);
    if (!call) return nullptr;
    const auto& reg = OpRegistry::GetInstance();
    if (!reg.IsRegistered(call->op_->name_)) return nullptr;
    auto reuse_idx = reg.GetEntry(call->op_->name_).GetOutputReusesInputArg();
    if (!reuse_idx.has_value() || *reuse_idx >= call->args_.size()) return nullptr;
    auto in_var = AsVarLike(call->args_[*reuse_idx]);
    if (!in_var) return nullptr;
    auto new_in = AsVarLike(VisitExpr(in_var));  // follow prior subst_ renames
    if (!new_in) return nullptr;
    auto lhs_tile = GetTileTypeWithMemRef(op->var_->GetType());
    auto in_new_tile = GetTileTypeWithMemRef(new_in->GetType());
    if (!lhs_tile || !in_new_tile ||
        SamePhysicalWindow(GetDefinedMemRef(lhs_tile), GetDefinedMemRef(in_new_tile))) {
      return nullptr;
    }
    auto new_type = CloneTypeWithMemRef(op->var_->GetType(), GetDefinedMemRef(in_new_tile),
                                        in_new_tile->GetMemorySpace());
    auto new_lhs = std::make_shared<Var>(op->var_->name_hint_, new_type, op->var_->span_);
    subst_[op->var_] = new_lhs;
    auto recursed = As<AssignStmt>(IRMutator::VisitStmt_(op));
    INTERNAL_CHECK_SPAN(recursed, op->span_) << "Internal error: AssignStmt visit must yield an AssignStmt";
    return std::make_shared<AssignStmt>(new_lhs, recursed->value_, recursed->span_);
  }

  std::map<VarPtr, ExprPtr> subst_;
};

/**
 * @brief Transform a function by identifying and applying memory reuse
 *
 * This transformation identifies memory reuse opportunities by walking the full
 * IR tree to compute variable lifetimes, then applying greedy MemRef sharing.
 * Variables that can share memory will point to the same MemRef object.
 * After sharing, redundant alloc operations are removed.
 */
// Semantic must-alias materialization — the "Step 0" formerly at the head of
// MemoryReuse. Propagates each ForStmt iter_arg/initValue's canonical MemRef
// down the yield/producer chain so accumulator producers (and other loop-carry
// / in-place chains) write directly into the carried buffer. This is a
// *semantics-required* aliasing (the loop accumulator must live in one buffer),
// as opposed to the opportunistic lifetime coalescing in MemoryReuse.
//
// Split into its own pass so it can run without the (skippable) lifetime-reuse
// phase: when ptoas owns lifetime reuse (memory_planner=PTOAS), this still runs
// while MemoryReuse is skipped, so codegen can emit a shared tile_buf handle for
// the must-alias buffers and ptoas PlanMemory does the reuse.
FunctionPtr TransformMaterializeSemanticAliases(const FunctionPtr& func) {
  INTERNAL_CHECK(func) << "MaterializeSemanticAliases cannot run on null function";

  // Orchestration bodies submit tasks and never hold TileType variables; a
  // Graph body is orchestration too, so it is skipped for the same reason.
  if (IsOrchestrationLike(func->func_type_)) return func;

  StmtPtr new_body = func->body_;
  TopDownRetargeter retargeter;
  auto rewrites = retargeter.Compute(new_body);
  if (!rewrites.empty()) {
    RetypeApplier applier(std::move(rewrites));
    new_body = applier.VisitStmt(new_body);
  }

  // Peeled accumulator phis are semantic must-alias boundaries. Coalesce them
  // before lifetime analysis so nested K-loop branches enter every memory
  // planner as one allocation family. The legacy planner repeats this repair
  // after packing because packing may introduce new carry mismatches.
  {
    TopDownRetargeter acc_coalescer;
    auto acc_rewrites = acc_coalescer.CoalesceAccumulatorIfPhis(new_body);
    if (!acc_rewrites.empty()) {
      RetypeApplier applier(std::move(acc_rewrites));
      new_body = applier.VisitStmt(new_body);
    }
  }
  // Accumulator-phi retargeting can expose bare SSA aliases or in-place outputs
  // whose types still name the pre-retarget allocation. Those are semantic
  // aliases too, so normalize them before any planner observes lifetimes or
  // emits handles. Planner-specific repetitions remain below because later
  // YieldFixup/reuse steps can create fresh mismatches.
  new_body = NormalizeIdentityCopyBuffersMutator().VisitStmt(new_body);

  // Under memory_planner=PtoAS or DsaRP the whole MemoryReuse pass is skipped.
  // DsaRP must therefore run the remaining correctness normalizations from
  // MemoryReuse Steps 4 through 4.5 in the same order. They are not optimizations: when a
  // peeled accumulator if-phi or
  // loop yields a value living in a different buffer than its iter_arg/return_var,
  // it inserts the `tile.move` that writes the result back into the carry. Without
  // it the carry is never updated and the loop silently becomes a no-op — the
  // `[N, 1]` col-vector carry of an online softmax is the shape that hits this,
  // because its branch producer runs on a `[1, N]` view in its own buffer.
  //
  // Run it here so both alternative planners reconcile carries. Under the
  // legacy PyPTO planner it stays where it is: Step 4 must run *after* reuse
  // decisions, which can themselves create fresh mismatches.
  //
  // PTOAS needs only the ForStmt YieldFixup half: addr-less codegen already
  // re-points a branch-local producer at the if-phi handle. DSA-RP instead
  // materializes both IfStmt and ForStmt fixups and repairs bare-Var identity
  // copies before lifetime analysis and placement.
  const auto* ctx = PassContext::Current();
  if (ctx != nullptr && ctx->GetMemoryPlanner() == MemoryPlanner::DsaRP) {
    // Identity-copy normalization brackets YieldFixup on both sides; see the
    // matching step in TransformMemoryReuse for why each side is needed.
    new_body = NormalizeIdentityCopyBuffersMutator().VisitStmt(new_body);

    YieldFixupMutator yield_fixup(/*fixup_if_stmts=*/true);
    new_body = yield_fixup.VisitStmt(new_body);
    new_body = InsertAllocsIntoBody(new_body, yield_fixup.TakePendingAllocs());
    new_body = NormalizeIdentityCopyBuffersMutator().VisitStmt(new_body);
  } else if (ctx != nullptr && ctx->GetMemoryPlanner() == MemoryPlanner::PtoAS) {
    YieldFixupMutator yield_fixup(/*fixup_if_stmts=*/false);
    new_body = yield_fixup.VisitStmt(new_body);
    new_body = InsertAllocsIntoBody(new_body, yield_fixup.TakePendingAllocs());
  }

  if (new_body == func->body_) return func;

  return std::make_shared<const Function>(func->name_, func->params_, func->param_directions_,
                                          func->return_types_, new_body, func->span_, func->func_type_,
                                          func->level_, func->role_, func->attrs_);
}

FunctionPtr TransformMemoryReuse(const FunctionPtr& func) {
  INTERNAL_CHECK(func) << "MemoryReusePass cannot run on null function";

  // Orchestration bodies submit tasks and never hold TileType variables, so
  // there is nothing for memory reuse to do — skip them silently. A Graph body
  // is orchestration too.
  if (IsOrchestrationLike(func->func_type_)) return func;

  // Step 0 (semantic must-alias retarget) now runs in the preceding
  // MaterializeSemanticAliases pass, so the body here is already retargeted.
  StmtPtr new_body = func->body_;

  // Step 1: Compute lifetimes by walking full IR tree
  auto analysis_result = AnalyzeAllocationLifetimes(new_body);

  if (analysis_result.lifetimes.empty()) {
    LOG_DEBUG << "No TileType variables found in function '" << func->name_ << "', skipping memory reuse";
    return func;
  }

  // Step 2: Identify reuse opportunities.  On Ascend910B split-AIV functions,
  // collect the load + tpop_from_aic hazard inputs so the reuse decision never
  // forms the hazardous in-place sharing (folds in the former
  // LegalizePTOBufferReuse responsibility).  Off-910B the inputs stay empty and
  // reuse behaviour is unchanged.
  const AllocationConstraintAnalysis constraint_analysis =
      AnalyzeAllocationConstraints(func, analysis_result, "MemoryReuse");
  const HazardInputs& hazard = constraint_analysis.target_hazard_inputs;
  const ForbidAliasMap& forbid_alias = constraint_analysis.forbid_alias;

  // Per-space reserved end (the SpaceFootprint reserved_start for the exact fit check). Only meaningful
  // with a configured backend; empty otherwise ⇒ reserved_start defaults to 0.
  std::map<MemorySpace, uint64_t> reserved_end_by_space;
  if (backend::BackendConfig::IsConfigured()) {
    auto policy = backend::GetBackend()->CreateMemoryAllocatorPolicy();
    if (policy) {
      // Shared with AllocateMemoryAddr — the reserved start is parity-by-construction, not comment-synced.
      const auto resolution = ResolveReserveBufferBases(func, *policy);
      for (const auto& [space, end] : resolution.reserved_end_by_space) reserved_end_by_space[space] = end;
    }
  }
  // Bases the author declared via a one-argument `pl.MemRef(...)`, materialized by
  // InitMemRef as a pinned alloc. Their membership is off-limits to the packer.
  // InitMemRef hoists every alloc to the body's top-level SeqStmts, so this scans
  // that list in place rather than flattening a copy of the whole body. Missing
  // that shape must not fail open: an empty `pinned_bases` silently disables both
  // the packer isolation and the co-liveness check below, handing the author back
  // exactly the coalescing the binding was written to prevent.
  const std::set<const Var*>& pinned_bases = constraint_analysis.declared_allocation_bases;

  std::vector<Diagnostic> hints;
  auto reuse_map = IdentifyReuseOpportunities(
      analysis_result.lifetimes, hazard, forbid_alias, analysis_result.phi_family_ids,
      analysis_result.var_sharing_groups, analysis_result.var_liveness, analysis_result.pipeline_membership,
      analysis_result.pipeline_load_tiles, reserved_end_by_space, pinned_bases, func, &hints);
  // Surface capacity-forced pipeline-depth reductions (perf hints) and legacy-fallback overflows
  // (warnings) through the unified diagnostic channel → perf_hints.log / stderr.
  if (!hints.empty()) EmitDiagnostics(hints, "MemoryReuse");

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

  // Step 3.75: Coalesce peeled loop-carried accumulator if-phis so YieldFixupMutator
  // does not reconcile them with an illegal Acc->Acc tile.move.  See
  // TopDownRetargeter::CoalesceAccumulatorIfPhis.  Must run after all ForStmt-carry
  // coalescing (Steps 0/3/3.5) so the accumulator branch is on its final buffer, and
  // before YieldFixupMutator (Step 4) so it observes one buffer per phi.  A no-op when
  // no accumulator if-phi exists (e.g. non-pipelined kernels).
  {
    TopDownRetargeter acc_coalescer;
    auto acc_rewrites = acc_coalescer.CoalesceAccumulatorIfPhis(new_body);
    if (!acc_rewrites.empty()) {
      RetypeApplier applier(std::move(acc_rewrites));
      new_body = applier.VisitStmt(new_body);
    }
  }

  // Step 3.9 / 4.5: Reconcile bare-Var SSA identity copies whose buffers diverged
  // from the value they rename.  No-op when no mismatch exists.
  //
  // This runs on both sides of YieldFixup because both sides create such a
  // divergence, from opposite directions:
  //
  //  - before, because Step 3.75's accumulator-if-phi retarget leaves a rename
  //    stranded on the pre-coalesce buffer, and because a carry rename such as
  //    `lag2 = lag1` only lands on lag1's buffer here. YieldFixup has to see the
  //    settled buffer to order the carry writebacks against each other: against
  //    the pre-normalization buffers every rename still sits on a buffer of its
  //    own, so no conflict is visible and a shift register silently collapses
  //    ([#2481]).
  //  - after, because YieldFixup's own IfStmt fixup repoints a phi return_var
  //    onto the canonical branch buffer, which strands that phi's downstream
  //    `c = c_phi` rename in exactly the same way.
  //
  // The mutator is idempotent, so the second run is a no-op whenever the first
  // already settled everything.
  new_body = NormalizeIdentityCopyBuffersMutator().VisitStmt(new_body);

  // Step 4: Fix ForStmt/IfStmt yield/return_var MemRef mismatches
  YieldFixupMutator yield_fixup;
  new_body = yield_fixup.VisitStmt(new_body);
  // A carry-copy cycle needs a scratch buffer; its alloc belongs on the body head
  // like every other allocation, and Step 5 below keeps it because it is in use.
  new_body = InsertAllocsIntoBody(new_body, yield_fixup.TakePendingAllocs());

  // Step 4.5: the second half of the bracket described at Step 3.9.
  new_body = NormalizeIdentityCopyBuffersMutator().VisitStmt(new_body);

  // Step 5: Remove alloc statements for MemRefs no longer in use
  auto used_bases = memref_collectors::CollectUsedBasePtrs(new_body);
  new_body = RemoveUnusedAllocStatements(new_body, used_bases);

  // Step 6: Strip the now-consumed pipeline_membership attr so it does not ride
  // downstream into later passes / codegen. It was only needed to carry stage
  // identity from LowerPipelineLoops to the reuse decision above.
  new_body = StripPipelineMembershipMutator().VisitStmt(new_body);

  auto result = std::make_shared<const Function>(func->name_, func->params_, func->param_directions_,
                                                 func->return_types_, new_body, func->span_, func->func_type_,
                                                 func->level_, func->role_, func->attrs_);
  return result;
}

}  // namespace

AllocationConstraintAnalysis AnalyzeAllocationConstraints(const FunctionPtr& func,
                                                          const LifetimeAnalysisResult& lifetimes,
                                                          const char* consumer) {
  AllocationConstraintAnalysis result;
  result.declared_allocation_sizes = CollectPinnedAllocSizes(func->body_, func->span_, consumer);
  for (const auto& [base, size] : result.declared_allocation_sizes) {
    static_cast<void>(size);
    result.declared_allocation_bases.insert(base);
  }
  ValidateDeclaredAllocs(func->body_, result.declared_allocation_bases, lifetimes.var_liveness);

  result.needs_load_tpop_hazard_guard = NeedsLoadTpopHazardGuard(func);
  if (result.needs_load_tpop_hazard_guard) {
    HazardInputCollector collector;
    collector.Run(func->body_);
    result.target_hazard_inputs = collector.Take();
  }

  ForbidAliasCollector forbid_collector(lifetimes.var_sharing_groups);
  forbid_collector.VisitStmt(func->body_);
  result.forbid_alias = forbid_collector.Take();
  return result;
}

LifetimeAnalysisResult AnalyzeAllocationLifetimes(const StmtPtr& func_body) {
  return AnalyzeAllocationLifetimesImpl(func_body);
}

LifetimeAnalysisResult AnalyzeAllocationLifetimes(const FunctionPtr& func) {
  INTERNAL_CHECK(func != nullptr) << "Cannot analyze allocation lifetimes for a null function";
  return AnalyzeAllocationLifetimesImpl(func->body_, func->params_);
}

namespace pass {
Pass MaterializeSemanticAliases() {
  return CreateFunctionPass(TransformMaterializeSemanticAliases, "MaterializeSemanticAliases",
                            kMaterializeSemanticAliasesProperties);
}
Pass MemoryReuse() { return CreateFunctionPass(TransformMemoryReuse, "MemoryReuse", kMemoryReuseProperties); }
}  // namespace pass
}  // namespace ir
}  // namespace pypto
