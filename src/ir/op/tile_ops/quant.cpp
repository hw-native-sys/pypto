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

#include <any>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/tile_view_semantics.h"
#include "pypto/ir/type.h"

namespace pypto::ir {
namespace {

TileTypePtr CheckQuantInput(const ExprPtr& value, const char* role) {
  auto tile = As<TileType>(value->GetType());
  CHECK_SPAN(tile && tile->shape_.size() == 2 && tile->dtype_ == DataType::FP32, value->span_)
      << "tile.quant requires a rank-2 FP32 " << role << " tile";
  return tile;
}

void CheckQuantParameter(const TileTypePtr& tile, const TileView& view, const ExprPtr& valid_rows,
                         const char* role, const Span& span) {
  // TQUANT broadcasts the coefficient at the flat address base + row. A
  // boxed tile or a padded column stride would feed it unrelated coefficients.
  auto physical_cols = As<ConstInt>(tile->shape_[1]);
  auto parameter_cols = As<ConstInt>(view.valid_shape[1]);
  CHECK_SPAN(physical_cols && physical_cols->value_ == 1 && parameter_cols && parameter_cols->value_ == 1 &&
                 view.slayout == TileLayout::none_box,
             span)
      << "tile.quant requires " << role << " to be an unboxed [rows, 1] column tile";
  CHECK_SPAN(AreExprsEqual(view.valid_shape[0], valid_rows), span)
      << "tile.quant requires " << role << " to have the same valid row count as the source";
}

TypePtr DeduceQuantType(const std::vector<ExprPtr>& args) {
  CHECK(args.size() == 2 || args.size() == 3) << "tile.quant requires src, scale, and an optional offset";
  auto src = CheckQuantInput(args[0], "source");
  const auto src_view = tile_view_semantics::GetEffectiveTileView(*src);
  CHECK_SPAN(src_view.blayout == TileLayout::row_major && src_view.slayout == TileLayout::none_box,
             args[0]->span_)
      << "tile.quant requires a row-major unboxed source tile";
  const auto physical_cols = As<ConstInt>(src->shape_[1]);
  constexpr int kDestinationRowAlignment = 32;
  CHECK_SPAN(
      physical_cols && physical_cols->value_ > 0 && physical_cols->value_ % kDestinationRowAlignment == 0,
      args[0]->span_)
      << "tile.quant requires a positive physical column count divisible by 32 for the 8-bit result";

  auto scale = CheckQuantInput(args[1], "scale");
  const auto scale_view = tile_view_semantics::GetEffectiveTileView(*scale);
  CheckQuantParameter(scale, scale_view, src_view.valid_shape[0], "scale", args[1]->span_);
  const bool has_offset = args.size() == 3;
  if (has_offset) {
    auto offset = CheckQuantInput(args[2], "offset");
    const auto offset_view = tile_view_semantics::GetEffectiveTileView(*offset);
    CheckQuantParameter(offset, offset_view, src_view.valid_shape[0], "offset", args[2]->span_);
    // The asymmetric PTO-ISA overload uses the same TileDataPara type for
    // the scale value and the offset pointer.
    CHECK_SPAN(tile_view_semantics::ShapeExprListsEquivalent(scale->shape_, offset->shape_) &&
                   scale_view.blayout == offset_view.blayout && scale_view.fractal == offset_view.fractal &&
                   scale_view.pad == offset_view.pad && scale_view.compact == offset_view.compact,
               args[2]->span_)
        << "tile.quant requires offset and scale to have matching physical shapes and layouts";
  }

  TileView result_view;
  result_view.valid_shape = src_view.valid_shape;
  return std::make_shared<TileType>(src->shape_, has_offset ? DataType::UINT8 : DataType::INT8, std::nullopt,
                                    result_view);
}

}  // namespace

REGISTER_OP("tile.quant")
    .set_op_category("TileOp")
    .set_description("A5 row-wise FP32 to 8-bit quantization with optional asymmetric offset")
    .add_argument("src", "FP32 data tile")
    .add_argument("scale", "FP32 per-row scale column tile")
    .add_argument("offset", "Optional FP32 per-row offset column tile")
    .set_input_memory(0, MemorySpace::Vec)
    .set_input_memory(1, MemorySpace::Vec)
    .set_input_memory(2, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .no_arg_writes()
    .not_inplace_safe()
    .functional_execution_memory_access()
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>&) {
      return DeduceQuantType(args);
    });

}  // namespace pypto::ir
