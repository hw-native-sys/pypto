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

#include "pypto/ir/transforms/utils/reg_memory_analysis.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <map>
#include <set>
#include <utility>
#include <vector>

#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/utils/op_predicates.h"
#include "pypto/ir/type.h"

namespace pypto::ir {
namespace {
// Half-open [begin, end) byte intervals relative to one MemRef base. Add()
// coalesces adjacent intervals as well as overlaps, so each range is maximal.
using Intervals = std::map<int64_t, int64_t>;

// Restrict state tracking to allocations touched by Reg memory operations.
// Views and store-result versions meet at their shared MemRef base identity.
class RegisterRoots : public IRVisitor {
 public:
  std::set<const Var*> roots;
  // One dynamic access makes the allocation user-managed. Excluding the
  // whole alias group avoids rejecting a static read after a dynamic write.
  std::set<const Var*> dynamic_roots;

 protected:
  void VisitExpr_(const CallPtr& call) override {
    if (IsOp(call, "reg.load") || IsOp(call, "reg.store")) {
      auto effect = GetRegMemoryEffect(call);
      auto tile = As<TileType>(effect.target->GetType());
      if (effect.root) {
        roots.insert(effect.root.get());
        auto is_static = [](const std::vector<ExprPtr>& shape) {
          return std::all_of(shape.begin(), shape.end(),
                             [](const ExprPtr& dim) { return As<ConstInt>(dim) != nullptr; });
        };
        if (!effect.begin || !effect.end || !is_static(tile->shape_) ||
            (tile->tile_view_ && !is_static(tile->tile_view_->valid_shape))) {
          dynamic_roots.insert(effect.root.get());
        }
      }
    }
    IRVisitor::VisitExpr_(call);
  }
};

// Disjoint initialized byte intervals, with an undo journal for conditional
// execution. Each insertion/deletion costs O(log N); the journal visits only
// changed intervals instead of copying the whole function's state per branch.
// An exhaustive conditional retains only facts proved on both paths; a
// possibly-empty loop establishes no new definite initialization.
class InitializationChecker : public IRVisitor {
 public:
  explicit InitializationChecker(const std::set<const Var*>& roots) : roots_(roots) {}
  // Credit an input parameter or data-producing Tile result's valid region.
  // Unknown geometry contributes no proof; this never changes Tile metadata.
  void Initialize(const ExprPtr& expr) {
    auto tile = As<TileType>(expr->GetType());
    if (!tile || !tile->memref_.has_value()) return;
    const auto memref = tile->memref_.value();
    if (!roots_.count(memref->base_.get())) return;
    auto offset = As<ConstInt>(memref->byte_offset_);
    if (!offset || offset->value_ < 0) return;
    const int64_t bytes = static_cast<int64_t>(tile->dtype_.GetByte());
    if (bytes <= 0) return;
    int64_t begin = offset->value_, count = 1;
    // A missing start_offset is the canonical zero origin, not an unknown address.
    if (tile->tile_view_ && tile->tile_view_->start_offset) {
      auto start = As<ConstInt>(tile->tile_view_->start_offset);
      if (!start || start->value_ < 0 ||
          start->value_ > (std::numeric_limits<int64_t>::max() - begin) / bytes) {
        return;
      }
      begin += start->value_ * bytes;
    }
    const auto& valid_shape = tile->tile_view_ ? tile->tile_view_->valid_shape : tile->shape_;
    if (!valid_shape.empty() && valid_shape.size() != tile->shape_.size()) return;
    // Validate every dimension first: a zero outer valid extent means no bytes
    // are initialized, even if a partial inner extent would stop prefix growth.
    for (size_t i = 0; i < tile->shape_.size(); ++i) {
      auto shape = As<ConstInt>(tile->shape_[i]);
      auto valid = valid_shape.empty() ? shape : As<ConstInt>(valid_shape[i]);
      if (!shape || !valid || valid->value_ <= 0 || shape->value_ < valid->value_) return;
    }
    // Credit the contiguous initialized prefix. A partial innermost dimension
    // still initializes its first row (e.g. [1,128] valid [1,64] initializes 64
    // elements). Stop extending at the first physical gap, rather than losing
    // that prefix or claiming the padding between rows as initialized.
    // TODO(reg-vmi-generalize): represent rectangular strided initialization
    // symbolically so later valid rows can be proved without enumerating rows.
    for (size_t i = tile->shape_.size(); i > 0; --i) {
      auto shape = As<ConstInt>(tile->shape_[i - 1]);
      auto valid = tile->tile_view_ && !tile->tile_view_->valid_shape.empty()
                       ? As<ConstInt>(tile->tile_view_->valid_shape[i - 1])
                       : shape;
      if (!shape || !valid || valid->value_ < 0 || shape->value_ < valid->value_) return;
      if (valid->value_ && count > std::numeric_limits<int64_t>::max() / valid->value_) return;
      count *= valid->value_;
      if (valid->value_ != shape->value_) break;
    }
    if (count > (std::numeric_limits<int64_t>::max() - begin) / bytes) return;
    Add(memref->base_.get(), begin, begin + count * bytes);
  }

 protected:
  void VisitStmt_(const AssignStmtPtr& stmt) override {
    // Check RHS reads before crediting the assignment's produced output.
    IRVisitor::VisitStmt_(stmt);
    auto call = As<Call>(stmt->value_);
    if (!call) return;  // Plain aliases inherit the same root's state.
    if (IsOp(call, "reg.full_mask")) {
      masks_[stmt->var_.get()] = As<RegType>(stmt->var_->GetType())->shape_;
    } else if (IsOp(call, "reg.active_mask")) {
      masks_[stmt->var_.get()] = As<ConstInt>(call->args_[0])->value_;
    }
    if (IsOp(call, "tile.create") || IsOp(call, "reg.store") ||
        op_predicates::IsBufferAliasingViewOp(call->op_->name_)) {
      return;
    }
    // An actual Tile-producing operation initializes its valid output region.
    // A bare create/allocation and a view are not data-producing operations.
    Initialize(stmt->var_);
  }
  void VisitExpr_(const CallPtr& call) override {
    if (IsOp(call, "reg.load") || IsOp(call, "reg.store")) {
      auto effect = GetRegMemoryEffect(call);
      if (!effect.root || !roots_.count(effect.root.get())) {
        IRVisitor::VisitExpr_(call);
        return;
      }
      INTERNAL_CHECK_SPAN(effect.begin && effect.end, call->span_)
          << "Static register analysis requires known endpoints";
      auto tile = As<TileType>(effect.target->GetType());
      const int64_t bytes = static_cast<int64_t>(tile->dtype_.GetByte());
      INTERNAL_CHECK_SPAN(bytes > 0, call->span_) << "Register element must occupy at least one byte";
      CHECK_SPAN(*effect.end <= std::numeric_limits<int64_t>::max() / bytes, call->span_)
          << "Register byte interval overflows";
      int64_t begin = *effect.begin * bytes, end = *effect.end * bytes;
      if (effect.access == ArgEffect::Read) {
        CHECK_SPAN(Contains(effect.root.get(), begin, end), call->span_)
            << "reg.load reads an uninitialized interval";
      } else {
        if (effect.mask) {
          auto mask = AsVarLike(effect.mask);
          auto it = masks_.find(mask.get());
          // An unknown mask never initializes its false lanes. Known mask
          // generators prove only a prefix, not the entire register width.
          end = it == masks_.end() ? begin : begin + it->second * bytes;
        }
        Add(effect.root.get(), begin, end);
      }
    }
    IRVisitor::VisitExpr_(call);
  }
  void VisitStmt_(const IfStmtPtr& stmt) override {
    VisitExpr(stmt->condition_);
    if (auto condition = As<ConstBool>(stmt->condition_)) {
      if (condition->value_) {
        VisitStmt(stmt->then_body_);
      } else if (stmt->else_body_) {
        VisitStmt(*stmt->else_body_);
      }
      return;
    }
    const auto checkpoint = journal_.size();
    VisitStmt(stmt->then_body_);
    // Snapshot only final intervals changed by this arm, not the entire state.
    // Merged-away insertions need no entry: their surviving interval covers
    // them. Incoming facts already survive Undo without being intersected.
    std::map<const Var*, Intervals> then_changes;
    if (stmt->else_body_) {
      for (size_t i = checkpoint; i < journal_.size(); ++i) {
        const auto& change = journal_[i];
        if (!change.inserted) continue;
        const auto& state = initialized_.at(change.root);
        auto it = state.find(change.begin);
        if (it != state.end()) then_changes[change.root].emplace(*it);
      }
    }
    Undo(checkpoint);
    if (stmt->else_body_) {
      VisitStmt(*stmt->else_body_);
      // Ordered disjoint intervals allow a linear overlap walk per changed
      // range, with logarithmic entry lookup. Include partial overlaps, not
      // just identical store boundaries, and leave else-only bytes unproved.
      std::vector<Change> common;
      for (const auto& [root, ranges] : then_changes) {
        auto state = initialized_.find(root);
        if (state == initialized_.end()) continue;
        for (const auto& [begin, end] : ranges) {
          auto it = state->second.lower_bound(begin);
          if (it != state->second.begin() && std::prev(it)->second > begin) --it;
          for (; it != state->second.end() && it->first < end; ++it) {
            auto overlap_begin = std::max(begin, it->first);
            auto overlap_end = std::min(end, it->second);
            if (overlap_begin < overlap_end) common.push_back({root, overlap_begin, overlap_end, true});
          }
        }
      }
      Undo(checkpoint);
      for (const auto& change : common) Add(change.root, change.begin, change.end);
    }
  }
  void VisitStmt_(const ForStmtPtr& stmt) override {
    auto start = As<ConstInt>(stmt->start_), stop = As<ConstInt>(stmt->stop_),
         step = As<ConstInt>(stmt->step_);
    if (start && stop && step &&
        ((step->value_ > 0 && start->value_ >= stop->value_) ||
         (step->value_ < 0 && start->value_ <= stop->value_))) {
      return;
    }
    auto checkpoint = journal_.size();
    // Static nonempty loops execute the body at least once; one traversal is
    // sufficient for the currently supported static access intervals. Unknown
    // trip counts may be zero, so their writes cannot establish outgoing facts.
    IRVisitor::VisitStmt_(stmt);
    if (!(start && stop && step && step->value_ != 0)) Undo(checkpoint);
  }
  void VisitStmt_(const WhileStmtPtr& stmt) override {
    // Validate reads inside the body, but assume no guaranteed iteration.
    auto checkpoint = journal_.size();
    IRVisitor::VisitStmt_(stmt);
    Undo(checkpoint);
  }

 private:
  struct Change {
    const Var* root;
    int64_t begin, end;
    // Undo erases inserted intervals and restores deleted intervals.
    bool inserted;
  };
  // Borrowed from the collector, which outlives this function-local checker.
  const std::set<const Var*>& roots_;
  // Definite initialization, shared by all Tile aliases of a MemRef base.
  std::map<const Var*, Intervals> initialized_;
  // Known true-prefix lane counts for direct full_mask/active_mask results.
  // Not rolled back: SSA identity and scope/dominance validation prevent a
  // branch-local definition from being used outside its legal scope. Carries
  // and mask expressions are not propagated and remain unknown here.
  std::map<const Var*, int64_t> masks_;
  // Reversible interval edits; checkpoints avoid copying the entire state.
  std::vector<Change> journal_;

  // Maximal disjoint intervals make the predecessor of begin the only range
  // that could contain the entire requested access.
  bool Contains(const Var* root, int64_t begin, int64_t end) const {
    auto state = initialized_.find(root);
    if (state == initialized_.end()) return false;
    auto it = state->second.upper_bound(begin);
    return it != state->second.begin() && std::prev(it)->second >= end;
  }
  // Union a proven write/output region into the state, recording inverse edits.
  // Empty writes (including unknown/empty masks) establish no new facts.
  void Add(const Var* root, int64_t begin, int64_t end) {
    if (begin >= end) return;
    auto& intervals = initialized_[root];
    auto it = intervals.lower_bound(begin);
    if (it != intervals.begin() && std::prev(it)->second >= begin) --it;
    while (it != intervals.end() && it->first <= end) {
      begin = std::min(begin, it->first);
      end = std::max(end, it->second);
      journal_.push_back({root, it->first, it->second, false});
      it = intervals.erase(it);
    }
    intervals.emplace(begin, end);
    journal_.push_back({root, begin, end, true});
  }
  // Reverse edits in LIFO order, including merges that removed old intervals.
  void Undo(size_t checkpoint) {
    while (journal_.size() > checkpoint) {
      auto change = journal_.back();
      journal_.pop_back();
      auto& intervals = initialized_[change.root];
      if (change.inserted) {
        intervals.erase(change.begin);
      } else {
        intervals.emplace(change.begin, change.end);
      }
    }
  }
};
}  // namespace

void VerifyRegMemoryInitialization(const FunctionPtr& function) {
  // Runtime-bound declarations have no body to analyze.
  if (!function || !function->body_) return;
  RegisterRoots collector;
  collector.VisitStmt(function->body_);
  for (auto root : collector.dynamic_roots) collector.roots.erase(root);
  if (collector.roots.empty()) return;
  InitializationChecker checker(collector.roots);
  // In/InOut contents come from the caller. Out parameters supply storage only
  // and must be written before a Reg load can consume their contents.
  for (size_t i = 0; i < function->params_.size(); ++i) {
    if (function->param_directions_[i] != ParamDirection::Out) checker.Initialize(function->params_[i]);
  }
  checker.VisitStmt(function->body_);
}
}  // namespace pypto::ir
