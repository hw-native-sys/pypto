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

#ifndef SRC_IR_OP_IMG2COL_UTILS_H_
#define SRC_IR_OP_IMG2COL_UTILS_H_

#include <any>
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/type.h"
#include "pypto/ir/type_inference.h"

namespace pypto::ir {

// Both levels describe the same TIMG2COL window and share its geometry limits.
inline std::vector<ExprPtr> GetImg2colShape(const ShapedType& src, const std::vector<ExprPtr>& args,
                                            const std::vector<std::pair<std::string, std::any>>& kwargs,
                                            const std::string& op_name) {
  const auto& span = args[0]->span_;
  const auto dtype = src.dtype_;
  CHECK_SPAN(dtype == DataType::FP16 || dtype == DataType::BF16 || dtype == DataType::FP32 ||
                 dtype == DataType::INT8,
             span)
      << op_name << " supports FP16, BF16, FP32 and INT8";
  const int64_t c0 = dtype == DataType::INT8 ? 32 : (dtype == DataType::FP32 ? 8 : 16);
  auto rows = As<ConstInt>(src.shape_[0]);
  auto channels = As<ConstInt>(src.shape_[1]);
  CHECK_SPAN(rows && channels, span) << op_name << " requires a static source shape";
  CHECK_SPAN(rows->value_ > 0 && rows->value_ % 16 == 0 && channels->value_ > 0 &&
                 channels->value_ <= 65535 && channels->value_ % c0 == 0,
             span)
      << op_name << " requires H*W divisible by 16 and channels divisible by C0=32/sizeof(dtype)";

  auto attr = [&](const char* name, int fallback, int64_t lower, int64_t upper) {
    const int64_t value = GetKwargOr<int>(kwargs, name, fallback);
    CHECK_SPAN(value >= lower && value <= upper, span)
        << op_name << " " << name << " must be in [" << lower << ", " << upper << "]";
    return value;
  };
  const int64_t h = attr("fmap_h", 0, 1, 65535);
  const int64_t w = attr("fmap_w", 0, 1, 65535);
  const int64_t kh = attr("kernel_h", 0, 1, 511);
  const int64_t kw = attr("kernel_w", 0, 1, 511);
  const int64_t sh = attr("stride_h", 1, 1, 255);
  const int64_t sw = attr("stride_w", 1, 1, 255);
  const int64_t dh = attr("dilation_h", 1, 1, 255);
  const int64_t dw = attr("dilation_w", 1, 1, 255);
  const int64_t pt = attr("pad_top", 0, 0, 255);
  const int64_t pb = attr("pad_bottom", 0, 0, 255);
  const int64_t pl = attr("pad_left", 0, 0, 255);
  const int64_t pr = attr("pad_right", 0, 0, 255);
  CHECK_SPAN(h * w == rows->value_, span) << op_name << " source rows must equal H*W";
  const int64_t padded_h = h + pt + pb - dh * (kh - 1) - 1;
  const int64_t padded_w = w + pl + pr - dw * (kw - 1) - 1;
  CHECK_SPAN(padded_h >= 0 && padded_w >= 0, span) << op_name << " kernel exceeds padded image";
  const std::array<int64_t, 2> bounds = {(padded_h / sh + 1) * (padded_w / sw + 1),
                                         channels->value_ * kh * kw};
  auto shape = As<MakeTuple>(args[3]);
  CHECK_SPAN(shape && shape->elements_.size() == 2, span) << op_name << " shape must be a static pair";
  for (size_t axis = 0; axis < 2; ++axis) {
    auto dim = As<ConstInt>(shape->elements_[axis]);
    const int64_t alignment = axis == 0 ? 16 : c0;
    CHECK_SPAN(dim && dim->value_ > 0 && dim->value_ <= 65535 && dim->value_ % alignment == 0, span)
        << op_name << " shape must be positive, uint16-sized and aligned to (16, C0)";
    CHECK_SPAN(dim->value_ <= bounds[axis], span) << op_name << " shape exceeds unfolded image";
    auto index_type = As<ScalarType>(args[axis + 1]->GetType());
    CHECK_SPAN(index_type && index_type->dtype_.IsIndexLike(), span)
        << op_name << " positions must be index-like scalars";
    if (auto pos = As<ConstInt>(args[axis + 1])) {
      CHECK_SPAN(pos->value_ >= 0 && pos->value_ <= 65535 && pos->value_ + dim->value_ <= bounds[axis], span)
          << op_name << " position is outside the unfolded image";
      CHECK_SPAN(axis == 0 || pos->value_ % c0 == 0, span) << op_name << " pos_k must be C0-aligned";
    }
  }
  return shape->elements_;
}

}  // namespace pypto::ir

#endif  // SRC_IR_OP_IMG2COL_UTILS_H_
