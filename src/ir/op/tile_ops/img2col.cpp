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
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/tile_view_semantics.h"
#include "pypto/ir/type.h"
#include "pypto/ir/type_inference.h"
#include "src/ir/op/img2col_utils.h"

namespace pypto::ir {
namespace {

TypePtr DeduceImg2colType(const std::vector<ExprPtr>& args,
                          const std::vector<std::pair<std::string, std::any>>& kwargs) {
  CHECK(args.size() == 4) << "tile.img2col requires src, pos_m, pos_k, shape";
  const auto& span = args[0]->span_;
  auto src = As<TileType>(args[0]->GetType());
  CHECK_SPAN(src && src->shape_.size() == 2, span) << "tile.img2col requires a 2D source tile";
  CHECK_SPAN(src->memory_space_ == MemorySpace::Mat, span) << "tile.img2col requires a Mat source";
  const auto view = tile_view_semantics::GetEffectiveTileView(*src);
  CHECK_SPAN(
      view.blayout == TileLayout::col_major && view.slayout == TileLayout::row_major && view.fractal == 512,
      span)
      << "tile.img2col requires a canonical NZ source layout";
  CHECK_SPAN(view.stride.empty(), span) << "tile.img2col does not support a strided source view";
  CHECK_SPAN(tile_view_semantics::ShapeExprListsEquivalent(GetValidShape(src), src->shape_), span)
      << "tile.img2col requires a fully valid source tile";

  auto shape = GetImg2colShape(*src, args, kwargs, "tile.img2col");
  TileView result_view;
  result_view.blayout = TileLayout::row_major;
  result_view.slayout = TileLayout::row_major;
  return std::make_shared<TileType>(shape, src->dtype_, std::nullopt, result_view, MemorySpace::Left);
}

REGISTER_OP("tile.img2col")
    .set_op_category("TileOp")
    .set_description("Unfold an L1 feature map into an L0A tile using TIMG2COL")
    .add_argument("src", "Full NZ feature map [H*W, C] in Mat memory")
    .add_argument("pos_m", "Starting flattened output spatial position")
    .add_argument("pos_k", "Starting C1, KH, KW, C0 position")
    .add_argument("shape", "Static destination shape [M, K]")
    .set_attr<int>("fmap_h")
    .set_attr<int>("fmap_w")
    .set_attr<int>("kernel_h")
    .set_attr<int>("kernel_w")
    .set_attr<int>("stride_h")
    .set_attr<int>("stride_w")
    .set_attr<int>("dilation_h")
    .set_attr<int>("dilation_w")
    .set_attr<int>("pad_top")
    .set_attr<int>("pad_bottom")
    .set_attr<int>("pad_left")
    .set_attr<int>("pad_right")
    .set_input_memory(0, MemorySpace::Mat)
    .set_output_memory(MemorySpace::Left)
    .not_inplace_safe()
    .f_deduce_type(DeduceImg2colType);

}  // namespace
}  // namespace pypto::ir
