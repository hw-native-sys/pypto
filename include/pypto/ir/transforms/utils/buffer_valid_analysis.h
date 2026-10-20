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

#ifndef PYPTO_IR_TRANSFORMS_UTILS_BUFFER_VALID_ANALYSIS_H_
#define PYPTO_IR_TRANSFORMS_UTILS_BUFFER_VALID_ANALYSIS_H_

#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/tile_view_semantics.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/utils/buffer_view_analysis.h"
#include "pypto/ir/transforms/utils/memref_utils.h"
#include "pypto/ir/type.h"

namespace pypto::ir {

// Zero extents also need dynamic allocation metadata, and must not become a
// native subview on the empty path. Physical capacity always remains static.
inline bool NeedsRuntimeBufferValid(const TypePtr& type) {
  const auto tile = As<TileType>(type);
  if (!tile) return false;
  const auto view = tile_view_semantics::GetEffectiveTileView(*tile);
  for (const auto& extent : view.valid_shape) {
    const auto constant = As<ConstInt>(extent);
    if (!constant || constant->value_ == 0) return true;
  }
  return false;
}

// A forward SSA scan proves that no alias of an old metadata version is used
// after set_validshape. Window aliases retain their parent's version. This is
// intentionally limited to straight-line dynamic state: no branch joins or
// fixed-point iteration, and no recursive traversal of alias initializers.
class BufferValidAnalysis : public IRVisitor {
 public:
  explicit BufferValidAnalysis(const StmtPtr& body) : views_(body) { VisitStmt(body); }
  bool supported() const { return valid_ && !(dynamic_ && restricted_); }

  void VisitExpr(const ExprPtr& expr) override {
    if (!expr) return;
    dynamic_ |= NeedsRuntimeBufferValid(expr->GetType());
    IRVisitor::VisitExpr(expr);
  }

 protected:
  void VisitVarLike_(const VarPtr& var) override {
    const auto found = versions_.find(var.get());
    if (found != versions_.end()) {
      const auto current = current_.find(found->second.root);
      valid_ &= current != current_.end() && current->second == found->second.version;
    }
  }
  void VisitExpr_(const IterArgPtr& var) override { VisitVarLike_(var); }

  void VisitStmt_(const AssignStmtPtr& assign) override {
    VisitExpr(assign->value_);
    if (!As<TileType>(assign->var_->GetType())) return;
    dynamic_ |= NeedsRuntimeBufferValid(assign->var_->GetType());
    Version version{assign->var_.get(), assign->var_.get()};
    ExprPtr alias = AsVarLike(assign->value_);
    const auto call = As<Call>(assign->value_);
    const bool setter = call && IsOp(call, "tile.set_validshape");
    if (call && !call->args_.empty() &&
        (setter || IsOp(call, "tile.slice") || IsOp(call, "tile.reshape") ||
         IsOp(call, "tile.reinterpret_view"))) {
      alias = call->args_[0];
    }
    if (const auto var = AsVarLike(alias)) {
      const auto found = versions_.find(var.get());
      if (found != versions_.end()) version = found->second;
    }
    if (setter) {
      valid_ &= !views_.IsWindow(call->args_[0]);
      version.version = assign->var_.get();
    }
    current_[version.root] = version.version;
    versions_.emplace(assign->var_.get(), version);
  }

  void VisitExpr_(const CallPtr& call) override {
    if (IsOp(call, "tile.reshape") || IsOp(call, "tile.reinterpret_view") || IsOp(call, "tile.assemble") ||
        IsOp(call, "tile.move") || IsOp(call, "tile.extract") || IsOp(call, "tile.transpose_view") ||
        IsOp(call, "tile.matmul") || IsOp(call, "tile.matmul_acc")) {
      restricted_ = true;
    }
    if (IsOp(call, "tile.slice") && !call->args_.empty() &&
        NeedsRuntimeBufferValid(call->args_[0]->GetType()) && views_.IsWindow(call->args_[0])) {
      // Deferred dynamic windows are reconstructed only at guarded consumers.
      // Nested dynamic-valid windows need an independent dominance/empty recipe.
      valid_ = false;
    }
    IRVisitor::VisitExpr_(call);
  }
  void VisitExpr_(const SubmitPtr&) override { valid_ = false; }
  void VisitStmt_(const IfStmtPtr& stmt) override {
    restricted_ = true;
    IRVisitor::VisitStmt_(stmt);
  }
  void VisitStmt_(const ForStmtPtr& stmt) override {
    restricted_ = true;
    IRVisitor::VisitStmt_(stmt);
  }
  void VisitStmt_(const WhileStmtPtr& stmt) override {
    restricted_ = true;
    IRVisitor::VisitStmt_(stmt);
  }

 private:
  struct Version {
    const Var* root;
    const Var* version;
  };
  BufferViewAnalysis views_;
  std::unordered_map<const Var*, Version> versions_;
  std::unordered_map<const Var*, const Var*> current_;
  bool valid_ = true;
  bool dynamic_ = false;
  bool restricted_ = false;
};

// PYPTO merges symbolic allocations as well as assigning addresses. A dynamic
// descriptor can only join full-capacity members with the same physical type.
// DSA/PTOAS retain the symbolic allocations and may independently reuse their
// addresses; that does not merge descriptor metadata and needs no restriction.
class BufferRootCompatibility : public IRVisitor {
 public:
  using Key = std::tuple<std::string, int64_t, int64_t, uint64_t>;
  struct Root {
    bool dynamic = false;
    bool initialized = false;
    std::optional<Key> key;
  };
  explicit BufferRootCompatibility(const StmtPtr& body) : views_(body) { VisitStmt(body); }
  Root Get(const VarPtr& value) const {
    const auto memory = GetTypeMemRef(value->GetType()).value_or(nullptr);
    const auto found = memory ? roots_.find(memory->base_.get()) : roots_.end();
    return found == roots_.end() ? Root{} : found->second;
  }
  static bool Compatible(const Root& a, const Root& b) {
    return !(a.dynamic || b.dynamic) || (a.key && a.key == b.key);
  }
  static Root Merge(const Root& a, const Root& b) {
    return {a.dynamic || b.dynamic, true, a.key == b.key ? a.key : std::nullopt};
  }

 protected:
  void VisitVarLike_(const VarPtr& value) override {
    const auto tile = As<TileType>(value->GetType());
    if (!tile || views_.IsWindow(value) || !seen_.insert(tile.get()).second) return;
    const auto memory = tile->memref_.value_or(nullptr);
    if (!memory) return;
    auto& root = roots_[memory->base_.get()];
    root.dynamic |= NeedsRuntimeBufferValid(tile);
    std::optional<Key> key;
    if (tile->shape_.size() == 2) {
      const auto rows = As<ConstInt>(tile->shape_[0]);
      const auto cols = As<ConstInt>(tile->shape_[1]);
      const auto origin = As<ConstInt>(memory->byte_offset_);
      if (rows && cols && origin && origin->value_ == 0) {
        key = Key{tile->dtype_.ToString(), rows->value_, cols->value_, memory->size_};
      }
    }
    if (!root.initialized) {
      root.key = key;
    } else if (root.key != key) {
      root.key.reset();
    }
    root.initialized = true;
  }
  void VisitExpr_(const IterArgPtr& value) override { VisitVarLike_(value); }

 private:
  BufferViewAnalysis views_;
  std::unordered_set<const TileType*> seen_;
  std::unordered_map<const Var*, Root> roots_;
};

}  // namespace pypto::ir
#endif  // PYPTO_IR_TRANSFORMS_UTILS_BUFFER_VALID_ANALYSIS_H_
