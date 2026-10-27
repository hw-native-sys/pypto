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
using Intervals = std::map<int64_t, int64_t>;

class RegisterRoots : public IRVisitor {
 public:
  bool present = false;
  std::set<const Var*> roots;

 protected:
  void VisitExpr_(const CallPtr& call) override {
    if (IsOp(call, "reg.load") || IsOp(call, "reg.store")) {
      present = true;
      auto tile = As<TileType>(call->args_[0]->GetType());
      if (tile && tile->memref_) roots.insert((*tile->memref_)->base_.get());
    }
    IRVisitor::VisitExpr_(call);
  }
};

// Disjoint initialized byte intervals, with an undo journal for conditional
// execution. Each insertion/deletion costs O(log N); the journal visits only
// changed intervals instead of copying the whole function's state per branch.
// A conditional/possibly-empty loop establishes no new definite initialization.
class InitializationChecker : public IRVisitor {
 public:
  explicit InitializationChecker(const std::set<const Var*>& roots) : roots_(roots) {}
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
      // TODO(reg-vmi-generalize): PTOAS can lower dynamic and strided VMI
      // accesses, but P0 proves only one statically bounded contiguous byte
      // interval. Replace this guard with symbolic range/interval reasoning
      // when RegOp carries those access modes.
      auto effect = GetRegMemoryEffect(call);
      CHECK_SPAN(effect.root && effect.begin && effect.end, call->span_)
          << "Cannot prove register offset bounds and initialization";
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
    Undo(checkpoint);
    if (stmt->else_body_) {
      VisitStmt(*stmt->else_body_);
      Undo(checkpoint);
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
    IRVisitor::VisitStmt_(stmt);
    if (!(start && stop && step && step->value_ != 0)) Undo(checkpoint);
  }
  void VisitStmt_(const WhileStmtPtr& stmt) override {
    auto checkpoint = journal_.size();
    IRVisitor::VisitStmt_(stmt);
    Undo(checkpoint);
  }

 private:
  struct Change {
    const Var* root;
    int64_t begin, end;
    bool inserted;
  };
  const std::set<const Var*>& roots_;
  std::map<const Var*, Intervals> initialized_;
  std::map<const Var*, int64_t> masks_;
  std::vector<Change> journal_;

  bool Contains(const Var* root, int64_t begin, int64_t end) const {
    auto state = initialized_.find(root);
    if (state == initialized_.end()) return false;
    auto it = state->second.upper_bound(begin);
    return it != state->second.begin() && std::prev(it)->second >= end;
  }
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
  if (!collector.present) return;
  InitializationChecker checker(collector.roots);
  for (size_t i = 0; i < function->params_.size(); ++i) {
    if (function->param_directions_[i] != ParamDirection::Out) checker.Initialize(function->params_[i]);
  }
  checker.VisitStmt(function->body_);
}
}  // namespace pypto::ir
