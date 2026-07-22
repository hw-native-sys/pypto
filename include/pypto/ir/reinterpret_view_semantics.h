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
 * @file reinterpret_view_semantics.h
 * @brief Shared byte-preserving shape inference for tensor/tile reinterpret views
 */

#ifndef PYPTO_IR_REINTERPRET_VIEW_SEMANTICS_H_
#define PYPTO_IR_REINTERPRET_VIEW_SEMANTICS_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/tile_view_semantics.h"
#include "pypto/ir/type.h"

namespace pypto::ir::reinterpret_view_semantics {

/** Result of resolving a byte-preserving reinterpret view. */
struct ReinterpretViewPlan {
  std::vector<ExprPtr> shape;
  std::vector<ExprPtr> valid_shape;
};

/** Return whether dtype is byte-addressable by the current PTO backends. */
inline bool IsSupportedDType(DataType dtype) {
  return dtype == DataType::INT8 || dtype == DataType::INT16 || dtype == DataType::INT32 ||
         dtype == DataType::INT64 || dtype == DataType::UINT8 || dtype == DataType::UINT16 ||
         dtype == DataType::UINT32 || dtype == DataType::UINT64 || dtype == DataType::FP16 ||
         dtype == DataType::FP32 || dtype == DataType::BF16;
}

/** Preserve only dtype-independent padding across a reinterpret view. */
inline PadValue NormalizePad(PadValue source_pad) {
  // zero has the same bit pattern for every supported dtype, while max/min are
  // defined by the source element type and are invalid after reinterpretation.
  return source_pad == PadValue::zero ? PadValue::zero : PadValue::null;
}

/** Extract and validate a shape tuple operand. */
inline std::vector<ExprPtr> ExtractShape(const ExprPtr& shape_arg, const std::string& op_name) {
  auto tuple_type = As<TupleType>(shape_arg->GetType());
  CHECK_SPAN(tuple_type, shape_arg->span_)
      << op_name << " shape must be a TupleType, but got " << shape_arg->GetType()->TypeName();
  CHECK_SPAN(!tuple_type->types_.empty(), shape_arg->span_) << op_name << " shape must have rank >= 1";

  for (size_t i = 0; i < tuple_type->types_.size(); ++i) {
    auto scalar_type = As<ScalarType>(tuple_type->types_[i]);
    CHECK_SPAN(scalar_type, shape_arg->span_)
        << op_name << " shape element " << i << " must be a scalar, but got "
        << tuple_type->types_[i]->TypeName();
    CHECK_SPAN(scalar_type->dtype_.IsIndexLike(), shape_arg->span_)
        << op_name << " shape element " << i << " must have dtype INT64, UINT64, or INDEX, but got "
        << scalar_type->dtype_.ToString();
  }

  if (auto tuple = As<MakeTuple>(shape_arg)) {
    return tuple->elements_;
  }
  std::vector<ExprPtr> shape;
  shape.reserve(tuple_type->types_.size());
  for (size_t i = 0; i < tuple_type->types_.size(); ++i) {
    shape.emplace_back(std::make_shared<TupleGetItemExpr>(shape_arg, static_cast<int>(i), shape_arg->span_));
  }
  return shape;
}

namespace detail {

inline void ValidatePhysicalShape(const std::vector<ExprPtr>& shape, const std::string& shape_name,
                                  const std::string& op_name, const Span& span) {
  CHECK_SPAN(!shape.empty(), span) << op_name << " " << shape_name << " must have rank >= 1";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (auto dim = As<ConstInt>(shape[i])) {
      CHECK_SPAN(dim->value_ > 0, span)
          << op_name << " " << shape_name << " dimension " << i << " must be positive, got " << dim->value_;
    }
  }
}

inline void ValidateValidShape(const std::vector<ExprPtr>& valid_shape,
                               const std::vector<ExprPtr>& physical_shape, const std::string& op_name,
                               const Span& span) {
  if (valid_shape.empty()) return;
  CHECK_SPAN(valid_shape.size() == physical_shape.size(), span)
      << op_name << " source valid_shape rank " << valid_shape.size() << " does not match source rank "
      << physical_shape.size();
  for (size_t i = 0; i < valid_shape.size(); ++i) {
    auto valid = As<ConstInt>(valid_shape[i]);
    if (!valid) continue;
    CHECK_SPAN(valid->value_ >= 0, span)
        << op_name << " source valid_shape dimension " << i << " must be non-negative, got " << valid->value_;
    if (auto physical = As<ConstInt>(physical_shape[i])) {
      CHECK_SPAN(valid->value_ <= physical->value_, span)
          << op_name << " source valid_shape dimension " << i << " (" << valid->value_
          << ") exceeds physical dimension " << physical->value_;
    }
  }
}

inline std::optional<int64_t> StaticElementCount(const std::vector<ExprPtr>& shape,
                                                 const std::string& shape_name, const std::string& op_name,
                                                 const Span& span) {
  int64_t count = 1;
  for (size_t i = 0; i < shape.size(); ++i) {
    auto dim = As<ConstInt>(shape[i]);
    if (!dim) return std::nullopt;
    CHECK_SPAN(dim->value_ > 0, span)
        << op_name << " " << shape_name << " dimension " << i << " must be positive, got " << dim->value_;
    int64_t next = 0;
    CHECK_SPAN(!__builtin_mul_overflow(count, dim->value_, &next), span)
        << op_name << " " << shape_name << " element count overflows int64";
    count = next;
  }
  return count;
}

inline ExprPtr ScaleExtent(const ExprPtr& extent, size_t src_bytes, size_t dst_bytes,
                           const std::string& extent_name, size_t axis, const std::string& op_name,
                           const Span& span) {
  INTERNAL_CHECK_SPAN(src_bytes != 0 && dst_bytes != 0, span)
      << "Internal error: " << op_name << " dtype byte widths must be nonzero, got source=" << src_bytes
      << " and target=" << dst_bytes;
  if (src_bytes == dst_bytes) return extent;

  if (src_bytes > dst_bytes) {
    CHECK_SPAN(src_bytes % dst_bytes == 0, span)
        << op_name << " cannot represent dtype byte ratio " << src_bytes << ":" << dst_bytes;
    const int64_t factor = static_cast<int64_t>(src_bytes / dst_bytes);
    if (auto value = As<ConstInt>(extent)) {
      int64_t scaled = 0;
      CHECK_SPAN(!__builtin_mul_overflow(value->value_, factor, &scaled), span)
          << op_name << " " << extent_name << " dimension " << axis << " overflows int64";
      return std::make_shared<ConstInt>(scaled, value->dtype(), extent->span_);
    }
    auto factor_expr = std::make_shared<ConstInt>(factor, DataType::INDEX, span);
    return std::make_shared<Mul>(extent, factor_expr, DataType::INDEX, span);
  }

  CHECK_SPAN(dst_bytes % src_bytes == 0, span)
      << op_name << " cannot represent dtype byte ratio " << src_bytes << ":" << dst_bytes;
  const int64_t divisor = static_cast<int64_t>(dst_bytes / src_bytes);
  auto value = As<ConstInt>(extent);
  CHECK_SPAN(value, span) << op_name << " cannot prove that dynamic " << extent_name << " dimension " << axis
                          << " is divisible by " << divisor
                          << " for the wider target dtype; provide a static divisible extent";
  CHECK_SPAN(value->value_ % divisor == 0, span)
      << op_name << " " << extent_name << " dimension " << axis << " (" << value->value_
      << ") is not divisible by " << divisor << " for the wider target dtype";
  return std::make_shared<ConstInt>(value->value_ / divisor, value->dtype(), extent->span_);
}

inline std::optional<ExprPtr> TryScaleExtent(const ExprPtr& extent, size_t src_bytes, size_t dst_bytes,
                                             const Span& span) {
  if (src_bytes == 0 || dst_bytes == 0) return std::nullopt;
  if (src_bytes == dst_bytes) return extent;
  if (src_bytes > dst_bytes) {
    if (src_bytes % dst_bytes != 0) return std::nullopt;
    const int64_t factor = static_cast<int64_t>(src_bytes / dst_bytes);
    if (auto value = As<ConstInt>(extent)) {
      int64_t scaled = 0;
      if (__builtin_mul_overflow(value->value_, factor, &scaled)) return std::nullopt;
      return std::make_shared<ConstInt>(scaled, value->dtype(), extent->span_);
    }
    auto factor_expr = std::make_shared<ConstInt>(factor, DataType::INDEX, span);
    return std::make_shared<Mul>(extent, factor_expr, DataType::INDEX, span);
  }

  if (dst_bytes % src_bytes != 0) return std::nullopt;
  const int64_t divisor = static_cast<int64_t>(dst_bytes / src_bytes);
  if (auto value = As<ConstInt>(extent)) {
    if (value->value_ % divisor != 0) return std::nullopt;
    return std::make_shared<ConstInt>(value->value_ / divisor, value->dtype(), extent->span_);
  }

  // Recognize the common inverse-roundtrip form (extent * divisor) / divisor.
  if (auto mul = As<Mul>(extent)) {
    if (auto rhs = As<ConstInt>(mul->right_); rhs && rhs->value_ == divisor) {
      return mul->left_;
    }
    if (auto lhs = As<ConstInt>(mul->left_); lhs && lhs->value_ == divisor) {
      return mul->right_;
    }
  }
  return std::nullopt;
}

inline std::vector<ExprPtr> BuildAutoShape(const std::vector<ExprPtr>& source, size_t contiguous_axis,
                                           size_t src_bytes, size_t dst_bytes, const std::string& shape_name,
                                           const std::string& op_name, const Span& span) {
  CHECK_SPAN(!source.empty(), span) << op_name << " requires rank >= 1";
  CHECK_SPAN(contiguous_axis < source.size(), span)
      << op_name << " contiguous axis " << contiguous_axis << " is out of range for rank " << source.size();
  std::vector<ExprPtr> result = source;
  result[contiguous_axis] =
      ScaleExtent(source[contiguous_axis], src_bytes, dst_bytes, shape_name, contiguous_axis, op_name, span);
  return result;
}

inline std::optional<std::vector<ExprPtr>> TryBuildAutoShape(const std::vector<ExprPtr>& source,
                                                             size_t contiguous_axis, size_t src_bytes,
                                                             size_t dst_bytes, const Span& span) {
  if (source.empty() || contiguous_axis >= source.size()) return std::nullopt;
  auto scaled = TryScaleExtent(source[contiguous_axis], src_bytes, dst_bytes, span);
  if (!scaled.has_value()) return std::nullopt;
  std::vector<ExprPtr> result = source;
  result[contiguous_axis] = *scaled;
  return result;
}

inline void CheckExplicitByteSize(const std::vector<ExprPtr>& source_shape,
                                  const std::vector<ExprPtr>& target_shape, size_t src_bytes,
                                  size_t dst_bytes, const std::string& op_name, const Span& span) {
  auto source_count = StaticElementCount(source_shape, "source shape", op_name, span);
  auto target_count = StaticElementCount(target_shape, "target shape", op_name, span);
  CHECK_SPAN(source_count.has_value() && target_count.has_value(), span)
      << op_name
      << " cannot prove byte-size equality for an explicit shape with dynamic dimensions; omit shape to "
         "use layout-aware auto detection, or provide fully static shapes";

  int64_t source_bytes = 0;
  int64_t target_bytes = 0;
  CHECK_SPAN(!__builtin_mul_overflow(*source_count, static_cast<int64_t>(src_bytes), &source_bytes), span)
      << op_name << " source byte size overflows int64";
  CHECK_SPAN(!__builtin_mul_overflow(*target_count, static_cast<int64_t>(dst_bytes), &target_bytes), span)
      << op_name << " target byte size overflows int64";
  CHECK_SPAN(source_bytes == target_bytes, span)
      << op_name << " requires equal source and target byte sizes, but source has " << source_bytes
      << " bytes and target has " << target_bytes << " bytes";
}

}  // namespace detail

/**
 * Resolve an exact-byte reinterpretation, including layout-aware auto shape and valid shape.
 *
 * An explicit arbitrary shape is accepted only for a fully-valid source. A partially-valid
 * rectangle can only be represented by the automatically derived shape, whose physically
 * contiguous axis is scaled by the dtype byte ratio.
 */
inline ReinterpretViewPlan Resolve(const std::vector<ExprPtr>& source_shape,
                                   const std::vector<ExprPtr>& source_valid_shape, DataType source_dtype,
                                   DataType target_dtype, size_t contiguous_axis,
                                   const std::optional<std::vector<ExprPtr>>& requested_shape,
                                   const std::string& op_name, const Span& span) {
  CHECK_SPAN(IsSupportedDType(source_dtype), span)
      << op_name << " does not support source dtype " << source_dtype.ToString()
      << "; supported dtypes are byte-addressable int/uint8/16/32/64, FP16, BF16, and FP32";
  CHECK_SPAN(IsSupportedDType(target_dtype), span)
      << op_name << " does not support target dtype " << target_dtype.ToString()
      << "; supported dtypes are byte-addressable int/uint8/16/32/64, FP16, BF16, and FP32";
  CHECK_SPAN(source_dtype != target_dtype, span)
      << op_name << " requires source and target dtypes to differ; use reshape/view for shape-only changes";
  detail::ValidatePhysicalShape(source_shape, "source shape", op_name, span);
  detail::ValidateValidShape(source_valid_shape, source_shape, op_name, span);

  const size_t src_bytes = source_dtype.GetByte();
  const size_t dst_bytes = target_dtype.GetByte();
  std::optional<std::vector<ExprPtr>> auto_shape =
      detail::TryBuildAutoShape(source_shape, contiguous_axis, src_bytes, dst_bytes, span);

  std::vector<ExprPtr> target_shape;
  if (requested_shape.has_value()) {
    target_shape = *requested_shape;
    detail::ValidatePhysicalShape(target_shape, "target shape", op_name, span);
    const bool matches_auto =
        auto_shape.has_value() && tile_view_semantics::ShapeExprListsEquivalent(target_shape, *auto_shape);
    if (!matches_auto) {
      detail::CheckExplicitByteSize(source_shape, target_shape, src_bytes, dst_bytes, op_name, span);
    }
  } else {
    // Use the diagnostic-producing path when automatic widening is not provable.
    target_shape = auto_shape.has_value() ? *auto_shape
                                          : detail::BuildAutoShape(source_shape, contiguous_axis, src_bytes,
                                                                   dst_bytes, "shape", op_name, span);
  }

  const bool has_explicit_valid = !source_valid_shape.empty();
  const bool is_partial =
      has_explicit_valid && !tile_view_semantics::ShapeExprListsEquivalent(source_valid_shape, source_shape);
  const std::vector<ExprPtr>& effective_valid = has_explicit_valid ? source_valid_shape : source_shape;

  if (requested_shape.has_value() && is_partial) {
    if (!auto_shape.has_value()) {
      auto_shape =
          detail::BuildAutoShape(source_shape, contiguous_axis, src_bytes, dst_bytes, "shape", op_name, span);
    }
    CHECK_SPAN(tile_view_semantics::ShapeExprListsEquivalent(target_shape, *auto_shape), span)
        << op_name
        << " cannot preserve a partial valid_shape through an arbitrary explicit reshape; omit shape or use "
           "the layout-aware auto-detected shape";
  }

  std::vector<ExprPtr> target_valid =
      is_partial ? detail::BuildAutoShape(effective_valid, contiguous_axis, src_bytes, dst_bytes,
                                          "valid_shape", op_name, span)
                 : target_shape;
  return ReinterpretViewPlan{std::move(target_shape), std::move(target_valid)};
}

}  // namespace pypto::ir::reinterpret_view_semantics

#endif  // PYPTO_IR_REINTERPRET_VIEW_SEMANTICS_H_
