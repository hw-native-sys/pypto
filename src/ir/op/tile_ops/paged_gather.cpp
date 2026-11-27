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

/**
 * @file paged_gather.cpp
 * @brief tile.gather_row — load one GM row directly into a sub-region of an
 *        on-chip (L1/Mat or UB/Vec) tile.
 *
 * This is the per-row primitive of the paged-gather lowering. It lowers to
 * ``pto.subview`` (of the destination accumulator) + ``pto.partition_view`` (of
 * the GM source) + ``pto.tload`` writing GM -> the subview directly, with **no
 * pto.tmov**. Filling an L1 (Mat) tile on a2a3 is only possible via GM->Mat
 * ``pto.tload`` (MAT->MAT tmov is unsupported), so the per-row gather must write
 * straight into the accumulator sub-region — mirroring CANN's TGatherInL1.
 *
 * Destination-Preserve-Source (DPS): the op writes into its first argument in
 * place (``set_output_reuses_input(0)``), so a loop-carried accumulator is
 * updated row by row without copying the whole tile.
 *
 * ``shapes`` vs the optional ``valid_shape``: ptoas types ``pto.subview``'s
 * ``sizes`` as a static ``I64ArrayAttr``, so the window shape can never be
 * dynamic. ``valid_shape`` carries the *runtime* transfer extent instead — it
 * feeds the subview's ``Optional<Index> valid_row/valid_col`` operands and the
 * GM ``pto.partition_view`` sizes, both of which accept dynamic SSA values. A
 * dynamic row count therefore costs nothing in allocation or box alignment: the
 * tile stays statically ``shapes``-sized and only the copy length varies.
 */

#include <any>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/tile_view_semantics.h"
#include "pypto/ir/transforms/utils/tensor_view_semantics.h"
#include "pypto/ir/type.h"
#include "pypto/ir/type_inference.h"

namespace pypto {
namespace ir {

static void CheckMxScaleGather(const std::vector<ExprPtr>& args, const TensorTypePtr& src,
                               const TensorView& src_view, const TileTypePtr& dst, bool transpose) {
  const auto& span = args[1]->span_;
  CHECK_SPAN(!transpose && src->shape_.size() == 2 && src->dtype_ == DataType::FP8E8M0, span)
      << "MX_A_ND gather requires a 2D FP8E8M0 source and transpose=False";
  CHECK_SPAN(dst->shape_.size() == 2 && dst->GetMemorySpace() == MemorySpace::Mat, args[0]->span_)
      << "MX_A_ND gather requires a 2D Mat destination";
  const auto view = tile_view_semantics::GetEffectiveTileView(*dst);
  CHECK_SPAN(view.blayout == TileLayout::row_major && view.slayout == TileLayout::row_major &&
                 view.fractal == tile_view_semantics::kMXScaleFractal,
             args[0]->span_)
      << "MX_A_ND gather requires an A/ZZ (row/row/32) destination";
  CHECK_SPAN(tile_view_semantics::ShapeExprListsEquivalent(GetEffectiveTensorValidShape(*src), src->shape_) &&
                 tile_view_semantics::ShapeExprListsEquivalent(view.valid_shape, dst->shape_),
             span)
      << "MX_A_ND gather requires fully valid source and destination tiles";
  const auto& stride = src_view.stride;
  const auto packed =
      tensor_view_semantics::BuildLogicalStridesFromLayout(src->shape_, TensorLayout::MX_A_ND);
  CHECK_SPAN(stride.empty() || tile_view_semantics::ShapeExprListsEquivalent(stride, packed), span)
      << "MX_A_ND gather requires a packed row-major source";

  const auto dst_offset = As<MakeTuple>(args[2]);
  const auto src_offset = As<MakeTuple>(args[3]);
  const auto shape = As<MakeTuple>(args[4]);
  const auto valid = args.size() == 6 ? As<MakeTuple>(args[5]) : shape;
  CHECK_SPAN(dst_offset && src_offset && shape && valid && dst_offset->elements_.size() == 2 &&
                 src_offset->elements_.size() == 2 && shape->elements_.size() == 2 &&
                 valid->elements_.size() == 2,
             span)
      << "MX_A_ND gather requires 2D offsets, shapes and valid_shape";

  const auto source_groups = As<ConstInt>(src->shape_[1]);
  const auto target_groups = As<ConstInt>(dst->shape_[1]);
  const auto groups = As<ConstInt>(shape->elements_[1]);
  const auto valid_groups = As<ConstInt>(valid->elements_[1]);
  const auto src_group_offset = As<ConstInt>(src_offset->elements_[1]);
  const auto dst_group_offset = As<ConstInt>(dst_offset->elements_[1]);
  CHECK_SPAN(source_groups && target_groups && groups && valid_groups && src_group_offset &&
                 dst_group_offset && groups->value_ > 0 &&
                 groups->value_ % tile_view_semantics::kMXSFractalCols == 0 &&
                 valid_groups->value_ == groups->value_ && source_groups->value_ == groups->value_ &&
                 target_groups->value_ == groups->value_ && src_group_offset->value_ == 0 &&
                 dst_group_offset->value_ == 0,
             span)
      << "MX_A_ND gather requires a full span of paired source and destination groups at group offset zero";

  constexpr int64_t kBoxRows = tile_view_semantics::kMXSFractalRows;
  const auto rows = As<ConstInt>(shape->elements_[0]);
  const auto valid_rows = As<ConstInt>(valid->elements_[0]);
  const auto target_rows = As<ConstInt>(dst->shape_[0]);
  const auto target_start = As<ConstInt>(dst_offset->elements_[0]);
  CHECK_SPAN(rows && valid_rows && target_rows && target_start && rows->value_ > 0 &&
                 rows->value_ % kBoxRows == 0 && valid_rows->value_ > 0 &&
                 valid_rows->value_ % kBoxRows == 0 && target_rows->value_ % kBoxRows == 0 &&
                 target_start->value_ >= 0 && target_start->value_ % kBoxRows == 0,
             span)
      << "MX_A_ND gather requires static destination offsets and complete 16x2 scale boxes";
  CHECK_SPAN(target_start->value_ <= target_rows->value_ &&
                 rows->value_ <= target_rows->value_ - target_start->value_,
             args[0]->span_)
      << "MX_A_ND gather window is outside the destination";
  const auto source_start_type = As<ScalarType>(src_offset->elements_[0]->GetType());
  CHECK_SPAN(source_start_type &&
                 (source_start_type->dtype_.IsInt() || source_start_type->dtype_ == DataType::INDEX),
             span)
      << "MX_A_ND gather source row offset must be an integer scalar";
  if (const auto source_start = As<ConstInt>(src_offset->elements_[0])) {
    const auto source_rows = As<ConstInt>(src->shape_[0]);
    CHECK_SPAN(source_start->value_ >= 0 &&
                   (!source_rows || (source_start->value_ <= source_rows->value_ &&
                                     valid_rows->value_ <= source_rows->value_ - source_start->value_)),
               span)
        << "MX_A_ND gather window is outside the source";
  }
}

static TypePtr DeduceTileGatherRowType(const std::vector<ExprPtr>& args,
                                       const std::vector<std::pair<std::string, std::any>>& kwargs,
                                       const std::string& op_name) {
  CHECK(args.size() == 5 || args.size() == 6)
      << "The operator " << op_name
      << " requires 5-6 arguments (dst, src, dst_offset, src_offset, shapes[, valid_shape]), but got "
      << args.size();

  auto dst_type = As<TileType>(args[0]->GetType());
  CHECK(dst_type) << "The operator " << op_name << " requires dst to be a TileType, but got "
                  << args[0]->GetType()->TypeName();

  const auto src_type = AsTensorTypeLike(args[1]->GetType());
  CHECK(src_type) << "The operator " << op_name << " requires src to be a TensorType (GM), but got "
                  << args[1]->GetType()->TypeName();
  const auto& tensor_view = src_type->tensor_view_;
  const auto* src_view = tensor_view.has_value() ? &*tensor_view : nullptr;
  const auto layout = src_view ? src_view->layout : TensorLayout::ND;
  const bool mx_scale = layout == TensorLayout::MX_A_ND;
  CHECK(mx_scale || As<TensorType>(args[1]->GetType()))
      << "The operator " << op_name << " requires MX_A_ND for a distributed source tensor";
  CHECK_SPAN(!IsMxTensorLayout(layout) || mx_scale, args[1]->span_)
      << "The operator " << op_name << " does not support MX-layout source tensors other than MX_A_ND";
  const bool transpose = GetKwargOr<bool>(kwargs, "transpose", false);
  if (!mx_scale) ValidateDmaInnerStride(src_type, args[1]->span_, op_name, transpose);
  CHECK(dst_type->dtype_ == src_type->dtype_)
      << "The operator " << op_name << " requires dst and src to share dtype, but got "
      << dst_type->dtype_.ToString() << " and " << src_type->dtype_.ToString();

  CheckGatherRowOperands(args, kwargs, op_name);
  if (mx_scale) CheckMxScaleGather(args, src_type, *src_view, dst_type, transpose);

  // DPS: the result is the destination accumulator, written in place. The
  // optional valid_shape narrows only *this* transfer — the accumulator keeps
  // its full extent (its other rows hold previously gathered data), so it must
  // NOT propagate into the result type.
  return dst_type;
}

REGISTER_OP("tile.gather_row")
    .set_op_category("TileOp")
    .set_description(
        "Load a GM window directly into a sub-region of an on-chip tile (Mat/L1 or Vec/UB) via "
        "pto.subview + pto.tload (GM->on-chip, no pto.tmov). DPS: writes into dst in place. Used as "
        "the per-row primitive of the paged-gather lowering (CANN TGatherInL1 equivalent).")
    .add_argument("dst", "Destination accumulator tile (Mat or Vec)")
    .add_argument("src", "Source tensor in GM (TensorType)")
    .add_argument("dst_offset", "Destination [row, col] offset (TupleType of ScalarType)")
    .add_argument("src_offset", "Source [row, col] offset within the GM tensor (TupleType of ScalarType)")
    .add_argument("shapes", "GM window shape [r, c] (TupleType of ScalarType)")
    .add_argument("valid_shape",
                  "Optional transfer extent [r, c] within shapes (TupleType of ScalarType). "
                  "Ordinary layouts may use runtime Scalar[INDEX] values; defaults to shapes.")
    .set_attr<bool>("transpose")
    .set_output_reuses_input(0)
    // DPS: loads a GM window into a sub-region of `dst`; the rest of the
    // accumulator tile passes through, so the prior content is read.
    .set_arg_effect(0, ArgEffect::ReadWrite)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileGatherRowType(args, kwargs, "tile.gather_row");
    });

}  // namespace ir
}  // namespace pypto
