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

#ifndef PYPTO_IR_TRANSFORMS_UTILS_BUFFER_VIEW_ANALYSIS_H_
#define PYPTO_IR_TRANSFORMS_UTILS_BUFFER_VIEW_ANALYSIS_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <unordered_map>

#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memref.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/structural_comparison.h"
#include "pypto/ir/type.h"

namespace pypto::ir {

/// Lexical SSA provenance for Vec windows. Inherited pitch is not recoverable
/// from a slice's TileType: its physical shape describes only the window.
/// Index each definition once; references never recursively visit initializers.
class BufferViewAnalysis : public IRVisitor {
 public:
  struct View {
    bool window = false;
    std::optional<int64_t> row_bytes;
    CallPtr slice;
    VarPtr root;
    std::optional<std::array<int64_t, 2>> offsets = std::array<int64_t, 2>{0, 0};
  };

  explicit BufferViewAnalysis(const StmtPtr& body) { VisitStmt(body); }

  View Get(const ExprPtr& value) const {
    const auto var = AsVarLike(value);
    const auto found = views_.find(var.get());
    if (found != views_.end()) return found->second;
    View view;
    view.root = var;
    view.row_bytes = PhysicalRowBytes(As<TileType>(value->GetType()));
    return view;
  }

  bool IsWindow(const ExprPtr& value) const { return Get(value).window; }

  bool HasDenseRowPitch(const ExprPtr& value) const {
    const auto tile = As<TileType>(value->GetType());
    if (!tile || tile->shape_.size() != 2) return false;
    const auto view = Get(value);
    // Native relabels count the inherited physical pitch even for one row.
    return view.row_bytes && view.row_bytes == PhysicalRowBytes(tile);
  }

 protected:
  void VisitVarLike_(const VarPtr&) override {}
  void VisitExpr_(const IterArgPtr&) override {}

  void VisitStmt_(const ForStmtPtr& loop) override { VisitLoop(loop); }
  void VisitStmt_(const WhileStmtPtr& loop) override { VisitLoop(loop); }

  void VisitStmt_(const AssignStmtPtr& assign) override {
    if (!As<TileType>(assign->var_->GetType())) return;
    auto view = Get(assign->var_);
    if (AsVarLike(assign->value_)) {
      view = Get(assign->value_);
    } else if (const auto call = As<Call>(assign->value_); call && !call->args_.empty()) {
      if (IsOp(call, "tile.slice")) {
        view = Get(call->args_[0]);
        view.window = true;
        view.slice = call;
        const auto offsets = As<MakeTuple>(call->args_[2]);
        if (!offsets || offsets->elements_.size() != 2) view.offsets = std::nullopt;
        if (view.offsets) {
          for (size_t i = 0; i < 2; ++i) {
            const auto offset = As<ConstInt>(offsets->elements_[i]);
            if (!offset || offset->value_ < 0 ||
                offset->value_ > std::numeric_limits<int64_t>::max() - (*view.offsets)[i]) {
              view.offsets = std::nullopt;
              break;
            }
            (*view.offsets)[i] += offset->value_;
          }
        }
      } else if (IsOp(call, "tile.reshape") || IsOp(call, "tile.reinterpret_view")) {
        view.window = IsWindow(call->args_[0]);
        if (view.window) view.root = nullptr;
        if (!HasDenseRowPitch(call->args_[0])) view.row_bytes = std::nullopt;
      } else if (IsOp(call, "tile.set_validshape")) {
        view = Get(call->args_[0]);
      }
    }
    views_.emplace(assign->var_.get(), view);
  }

 private:
  void BindStorage(const VarPtr& binding, const ExprPtr& source) {
    const auto target = As<TileType>(binding->GetType());
    const auto input = source ? As<TileType>(source->GetType()) : nullptr;
    if (!target || !input) return;
    auto view = Get(source);
    bool same_storage = !target->memref_ && !input->memref_;
    if (target->memref_ && input->memref_ && *target->memref_ && *input->memref_) {
      same_storage = MemRef::SameAllocation(*target->memref_, *input->memref_);
    }
    if (!same_storage || !structural_equal(target, input)) {
      if (view.window) {
        // Before carry reconciliation, a different binding does not prove
        // which physical root or pitch survives. Keep selection conservative
        // without giving a transfer stale source coordinates.
        view.root = nullptr;
        view.row_bytes = std::nullopt;
        view.offsets = std::nullopt;
      } else {
        view = Get(binding);
      }
    }
    views_.insert_or_assign(binding.get(), view);
  }

  template <typename LoopPtr>
  void VisitLoop(const LoopPtr& loop) {
    // Carries denote their entry storage even for zero iterations. Cache each
    // binding before the body; nested initializers then use a bounded lookup.
    for (const auto& argument : loop->iter_args_) BindStorage(argument, argument->initValue_);
    VisitStmt(loop->body_);
    for (size_t i = 0; i < loop->return_vars_.size() && i < loop->iter_args_.size(); ++i) {
      // A yielded producer supplies data, not the carry's physical row pitch.
      BindStorage(loop->return_vars_[i], loop->iter_args_[i]);
    }
  }

  static std::optional<int64_t> PhysicalRowBytes(const TileTypePtr& tile) {
    if (!tile || tile->shape_.size() != 2) return std::nullopt;
    const auto cols = As<ConstInt>(tile->shape_[1]);
    const auto bytes = tile->dtype_.GetBit() / 8;
    if (!cols || cols->value_ <= 0 || bytes <= 0 ||
        cols->value_ > std::numeric_limits<int64_t>::max() / bytes) {
      return std::nullopt;
    }
    return cols->value_ * bytes;
  }

  std::unordered_map<const Var*, View> views_;
};

}  // namespace pypto::ir
#endif  // PYPTO_IR_TRANSFORMS_UTILS_BUFFER_VIEW_ANALYSIS_H_
