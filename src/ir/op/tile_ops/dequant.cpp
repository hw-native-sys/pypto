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
#include <cstdint>
#include <limits>
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

TileTypePtr CheckParameter(const ExprPtr& value, const ExprPtr& valid_rows, const char* role) {
  auto tile = As<TileType>(value->GetType());
  CHECK_SPAN(tile && tile->shape_.size() == 2 && tile->dtype_ == DataType::FP32, value->span_)
      << "tile.dequant requires a rank-2 FP32 " << role << " tile";

  const auto view = tile_view_semantics::GetEffectiveTileView(*tile);
  const auto valid_cols = As<ConstInt>(view.valid_shape[1]);
  const auto physical_cols = As<ConstInt>(tile->shape_[1]);
  CHECK_SPAN(view.slayout == TileLayout::none_box && valid_cols && valid_cols->value_ == 1 &&
                 (view.blayout == TileLayout::row_major ||
                  (view.blayout == TileLayout::col_major && physical_cols && physical_cols->value_ == 1)),
             value->span_)
      << "tile.dequant requires " << role << " to be an unboxed column tile with one valid column";
  CHECK_SPAN(AreExprsEqual(view.valid_shape[0], valid_rows), value->span_)
      << "tile.dequant requires " << role << " to have the same valid row count as the source";
  return tile;
}

TypePtr DeduceDequantType(const std::vector<ExprPtr>& args) {
  CHECK(args.size() == 3) << "tile.dequant requires src, scale, and offset";
  const auto& source_span = args[0]->span_;
  auto src = As<TileType>(args[0]->GetType());
  CHECK_SPAN(
      src && src->shape_.size() == 2 && (src->dtype_ == DataType::INT8 || src->dtype_ == DataType::INT16),
      source_span)
      << "tile.dequant requires a rank-2 signed INT8 or INT16 source tile";
  const auto src_view = tile_view_semantics::GetEffectiveTileView(*src);
  CHECK_SPAN(src_view.blayout == TileLayout::row_major && src_view.slayout == TileLayout::none_box,
             source_span)
      << "tile.dequant requires a row-major unboxed source tile";

  auto scale = CheckParameter(args[1], src_view.valid_shape[0], "scale");
  auto offset = CheckParameter(args[2], src_view.valid_shape[0], "offset");
  const auto scale_view = tile_view_semantics::GetEffectiveTileView(*scale);
  const auto offset_view = tile_view_semantics::GetEffectiveTileView(*offset);
  // Both parameters are instances of the same TileDataPara in the TDEQUANT ABI.
  CHECK_SPAN(tile_view_semantics::ShapeExprListsEquivalent(scale->shape_, offset->shape_) &&
                 scale_view.blayout == offset_view.blayout && scale_view.fractal == offset_view.fractal &&
                 scale_view.pad == offset_view.pad && scale_view.compact == offset_view.compact,
             args[2]->span_)
      << "tile.dequant requires scale and offset to have matching physical shapes and layouts";

  auto result_shape = src->shape_;
  const auto cols = As<ConstInt>(result_shape[1]);
  constexpr int64_t kFp32VectorLanes = 64;
  constexpr int64_t kVectorBlockBytes = 32;
  CHECK_SPAN(cols && cols->value_ > 0 &&
                 cols->value_ <= std::numeric_limits<int32_t>::max() - (kFp32VectorLanes - 1) &&
                 cols->value_ % (kVectorBlockBytes / src->dtype_.GetByte()) == 0,
             source_span)
      << "tile.dequant requires a static positive source width with 32-byte row alignment";

  const auto valid_cols = As<ConstInt>(src_view.valid_shape[1]);
  const auto max_valid_cols = valid_cols ? valid_cols->value_ : cols->value_;
  // The pinned A5 TDEQUANT uses the first vector predicate for every repeat.
  // A short last repeat after a full vector therefore needs enough output
  // row storage for a full-width store.
  if (max_valid_cols > kFp32VectorLanes) {
    const auto store_cols = (max_valid_cols + kFp32VectorLanes - 1) / kFp32VectorLanes * kFp32VectorLanes;
    if (store_cols > cols->value_) {
      result_shape[1] = std::make_shared<ConstInt>(store_cols, DataType::INDEX, source_span);
    }
  }
  TileView result_view;
  result_view.valid_shape = src_view.valid_shape;
  return std::make_shared<TileType>(result_shape, DataType::FP32, std::nullopt, result_view);
}

}  // namespace

REGISTER_OP("tile.dequant")
    .set_op_category("TileOp")
    .set_description("A5 row-wise INT8/INT16 to FP32 dequantization")
    .add_argument("src", "Signed INT8 or INT16 data tile")
    .add_argument("scale", "FP32 per-row scale column tile")
    .add_argument("offset", "FP32 per-row offset column tile")
    .set_input_memory(0, MemorySpace::Vec)
    .set_input_memory(1, MemorySpace::Vec)
    .set_input_memory(2, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .no_arg_writes()
    .not_inplace_safe()
    .functional_execution_memory_access()
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>&) {
      return DeduceDequantType(args);
    });

}  // namespace pypto::ir
