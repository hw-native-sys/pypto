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

#ifndef PYPTO_IR_TRANSFORMS_UTILS_PACKED_FP4_CAST_H_
#define PYPTO_IR_TRANSFORMS_UTILS_PACKED_FP4_CAST_H_

#include <initializer_list>

#include "pypto/core/dtype.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/tile_view_semantics.h"
#include "pypto/ir/type.h"

namespace pypto::ir {

// Native A5 FP4 TCVT has a working contiguous DN path only when both descriptors
// carry full static validity. Its dynamic-valid path can fault even for dense
// DN data. Layout inference must preserve this narrow case for codegen to emit
// static-valid aliases; all other casts retain the backend's ND requirement.
inline bool IsDensePackedFp4DnCast(const CallPtr& call) {
  if (!call || !IsOp(call, "tile.cast") || call->args_.empty()) return false;
  const auto source = As<TileType>(call->args_[0]->GetType());
  const auto result = As<TileType>(call->GetType());
  if (!source || !result || (!source->dtype_.IsPackedFp4() && !result->dtype_.IsPackedFp4())) {
    return false;
  }
  for (const auto& tile : {source, result}) {
    if (tile->shape_.size() != 2 || (!tile->dtype_.IsPackedFp4() && tile->dtype_ != DataType::BF16)) {
      return false;
    }
    const auto rows = As<ConstInt>(tile->shape_[0]);
    const auto cols = As<ConstInt>(tile->shape_[1]);
    if (!rows || !cols || rows->value_ <= 0 || cols->value_ <= 0) return false;
    const auto view = tile_view_semantics::GetEffectiveTileView(*tile);
    if (view.blayout != TileLayout::col_major || view.slayout != TileLayout::none_box ||
        !tile_view_semantics::ShapeExprListsEquivalent(view.valid_shape, tile->shape_)) {
      return false;
    }
    if (!view.stride.empty()) {
      if (view.stride.size() != 2) return false;
      const auto row_stride = As<ConstInt>(view.stride[0]);
      const auto col_stride = As<ConstInt>(view.stride[1]);
      if (!row_stride || !col_stride || row_stride->value_ != 1 || col_stride->value_ != rows->value_) {
        return false;
      }
    }
    if (view.start_offset) {
      const auto offset = As<ConstInt>(view.start_offset);
      if (!offset || offset->value_ != 0) return false;
    }
  }
  return true;
}

}  // namespace pypto::ir

#endif  // PYPTO_IR_TRANSFORMS_UTILS_PACKED_FP4_CAST_H_
