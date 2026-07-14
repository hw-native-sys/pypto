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

#include <cstddef>
#include <string>

#include "internal.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {
namespace {

bool IsNdTile(const TileTypePtr& tile_type) { return tile_type && tile_type->shape_.size() > 2; }

class PreconditionAnalysis : public IRVisitor {
 public:
  void VisitStmt_(const AssignStmtPtr& op) override {
    if (!op) return;
    if (auto call = As<Call>(op->value_)) {
      CheckCall(call);
    }
    IRVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const EvalStmtPtr& op) override {
    if (!op) return;
    if (auto call = As<Call>(op->expr_)) {
      CheckCall(call);
    }
    IRVisitor::VisitStmt_(op);
  }

 private:
  static void CheckStaticShape(const TileTypePtr& tile_type, const std::string& op_name) {
    if (!IsNdTile(tile_type)) return;
    for (size_t i = 0; i < tile_type->shape_.size(); ++i) {
      CHECK(As<ConstInt>(tile_type->shape_[i]))
          << "FlattenTileNdTo2D: tile op '" << op_name << "' has a dynamic (non-constant) "
          << "dimension " << i << " in its >2D tile shape, which cannot be flattened to 2D. "
          << "Hardware tiles map to fixed-size on-chip buffers, so every tile dimension must "
          << "be a compile-time constant; a pl.dynamic dimension has no static bound and cannot "
          << "back a tile dimension directly. Fix by either: (1) iterating/tiling the dynamic "
          << "dimension with pl.range/pl.parallel so each per-iteration tile slice is static, or "
          << "(2) reshaping the tensor to 2D before the InCore (pl.at) scope so the dynamic extent "
          << "lands on the pl.parallel loop bound instead of inside the tile shape.";
    }
  }

  void CheckCall(const CallPtr& call) {
    if (!call || !call->op_ || As<GlobalVar>(call->op_)) return;

    const auto& name = call->op_->name_;
    if (name.substr(0, 5) != "tile.") return;

    for (const auto& arg : call->args_) {
      CheckStaticShape(As<TileType>(arg->GetType()), name);
    }
    CheckStaticShape(As<TileType>(call->GetType()), name);

    if (IsOp(call, "tile.read") || IsOp(call, "tile.write") || IsOp(call, "tile.slice")) {
      if (!call->args_.empty()) {
        auto input_tile = As<TileType>(call->args_[0]->GetType());
        CHECK(!IsNdTile(input_tile)) << "FlattenTileNdTo2D: " << name << " is not supported on >2D tiles";
      }
    }

    if (IsOp(call, "tile.sum") || IsOp(call, "tile.max") || IsOp(call, "tile.min")) {
      if (!call->args_.empty()) {
        auto input_tile = As<TileType>(call->args_[0]->GetType());
        if (IsNdTile(input_tile)) {
          int axis = call->GetKwarg<int>("axis", -1);
          int last_axis = static_cast<int>(input_tile->shape_.size()) - 1;
          CHECK(axis == last_axis) << "FlattenTileNdTo2D: tile reduce op '" << name
                                   << "' must reduce along the last axis "
                                   << "(axis=" << last_axis << "), but got axis=" << axis;
          bool keepdim = call->GetKwarg<bool>("keepdim", false);
          CHECK(keepdim) << "FlattenTileNdTo2D: tile reduce op '" << name
                         << "' on >2D tile must use keepdim=True to maintain 2D output shape";
        }
      }
    }
  }
};

}  // namespace

namespace flatten_tile_nd_to_2d {

void Analyze(const FunctionPtr& func) {
  PreconditionAnalysis analysis;
  analysis.VisitStmt(func->body_);
}

}  // namespace flatten_tile_nd_to_2d
}  // namespace ir
}  // namespace pypto
