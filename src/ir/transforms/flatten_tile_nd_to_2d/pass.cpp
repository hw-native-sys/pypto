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
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/transforms/utils/narrow_loop_carry.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/type.h"
#include "src/ir/transforms/flatten_tile_nd_to_2d/internal.h"

namespace pypto {
namespace ir {
namespace pass {
namespace {

/// A helper borrows the caller's flattened Tile, so its parameter rank and
/// scalar element coordinates must change together. GM Tensor coordinates
/// remain logical ND coordinates and are deliberately left untouched.
class FlattenSimtTiles : public IRMutator {
 public:
  explicit FlattenSimtTiles(const FunctionPtr& helper) {
    for (const auto& parameter : helper->params_) {
      auto tile = As<TileType>(parameter->GetType());
      if (!tile || tile->shape_.size() <= 2) continue;
      CHECK_SPAN(!tile->tile_view_, parameter->span_)
          << "Flattening ND SIMT Tile parameters currently requires an implicit packed Tile layout";
      int64_t rows = 1;
      for (size_t i = 0; i + 1 < tile->shape_.size(); ++i) {
        auto extent = As<ConstInt>(tile->shape_[i]);
        CHECK_SPAN(
            extent && extent->value_ > 0 && extent->value_ <= std::numeric_limits<int64_t>::max() / rows,
            parameter->span_)
            << "ND SIMT Tile helper parameters require positive static leading dimensions";
        rows *= extent->value_;
      }
      std::vector<ExprPtr> shape{std::make_shared<ConstInt>(rows, DataType::INDEX, parameter->span_),
                                 tile->shape_.back()};
      auto type = std::make_shared<TileType>(std::move(shape), tile->dtype_, std::nullopt, std::nullopt,
                                             tile->memory_space_);
      var_remap_[parameter.get()] = std::make_shared<Var>(parameter->name_hint_, type, parameter->span_);
    }
  }

 protected:
  ExprPtr VisitExpr_(const CallPtr& call) override {
    if ((!IsOp(call, "simt.load") && !IsOp(call, "simt.store")) || call->args_.empty()) {
      return IRMutator::VisitExpr_(call);
    }
    auto tile = As<TileType>(call->args_[0]->GetType());
    if (!tile || tile->shape_.size() <= 2) return IRMutator::VisitExpr_(call);
    std::vector<ExprPtr> arguments;
    for (const auto& argument : call->args_) arguments.push_back(VisitExpr(argument));
    auto indices = As<MakeTuple>(arguments[1]);
    INTERNAL_CHECK_SPAN(indices && indices->elements_.size() == tile->shape_.size(), call->span_)
        << "Internal error: SIMT Tile indices must match the pre-flatten rank";
    auto to_index = [&](const ExprPtr& value) -> ExprPtr {
      if (As<ScalarType>(value->GetType())->dtype_ == DataType::INDEX) return value;
      return std::make_shared<Cast>(value, DataType::INDEX, call->span_);
    };
    auto row = to_index(indices->elements_[0]);
    for (size_t i = 1; i + 1 < tile->shape_.size(); ++i) {
      row = std::make_shared<Add>(
          std::make_shared<Mul>(row, to_index(tile->shape_[i]), DataType::INDEX, call->span_),
          to_index(indices->elements_[i]), DataType::INDEX, call->span_);
    }
    arguments[1] =
        std::make_shared<MakeTuple>(std::vector<ExprPtr>{row, indices->elements_.back()}, call->span_);
    auto result = OpRegistry::GetInstance().Create(call->op_->name_, arguments, call->kwargs_, call->span_);
    return transform_utils::PreserveCallAttrs(call, result);
  }
};

FunctionPtr TransformFunction(const FunctionPtr& func) {
  if (func->func_type_ == FunctionType::SIMT) return FlattenSimtTiles(func).VisitFunction(func);
  if (!IsInCoreType(func->func_type_)) {
    return func;
  }

  flatten_tile_nd_to_2d::Analyze(func);
  auto flattened = flatten_tile_nd_to_2d::Rewrite(func);
  // Unrolling a `tile.batch_matmul` into 2D matmuls is what first narrows an
  // accumulator whose left operand carries a runtime valid row count, so the loop
  // carry that accumulator flows through has to be re-declared at that extent
  // before this pass returns -- otherwise it publishes a carry its own TypeCheck
  // and AccCompactValid verifiers reject (issue #2470).
  return narrow_loop_carry::NarrowAccCarries(flattened);
}

}  // namespace

Pass FlattenTileNdTo2D() {
  return CreateFunctionPass(TransformFunction, "FlattenTileNdTo2D", kFlattenTileNdTo2DProperties);
}

}  // namespace pass
}  // namespace ir
}  // namespace pypto
