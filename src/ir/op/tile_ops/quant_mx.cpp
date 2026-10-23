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
 * @file quant_mx.cpp
 * @brief MX quantization tile ops
 *
 * ``group_axis`` matches PTOAS ``grpAxis``: axis1 is the A-side [M,K] path;
 * axis0 is the B-side [N,K] path (LowerCompositeOps transposes to [K,N] first).
 * GM TensorLayout.MX_* remains a tensor/view annotation, not a tile-op kwarg.
 *
 * LowerCompositeOps expands tile.tquant_mx into value-returning
 * tile.tquant_mx_raw + tile.tmov_x2zz (gather_compare-style SSA), then
 * reinterpret / transpose_view to the public scale and FP8 data types.
 * FP4E2M1X2 data is already returned in its native packed type. Internal ops run
 * before InferTileMemorySpace; scratch tiles are created with an explicit
 * MemorySpace::Vec annotation so the later memory planner still gets addresses.
 */

#include <any>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/any_cast.h"
#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/tile_view_semantics.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

namespace {

using tile_view_semantics::kMXGroupSize;
using tile_view_semantics::kMXScaleFractal;
using tile_view_semantics::kMXSFractalCols;
using tile_view_semantics::kMXSFractalRows;

struct MxQuantTypeInfo {
  DataType public_dst_dtype;
  DataType raw_dst_dtype;
};

MxQuantTypeInfo ResolveMxQuantType(DataType dtype, const std::string& op_name, const Span& span) {
  if (dtype == DataType::FP4E2M1X2) return {dtype, dtype};
  CHECK_SPAN(dtype == DataType::FP8E4M3FN, span)
      << "The operator " << op_name << " requires dtype FP8E4M3FN or FP4E2M1X2, but got " << dtype.ToString();
  return {DataType::FP8E4M3FN, DataType::INT8};
}

void CheckFp4QuantSource(const TileTypePtr& src, DataType dtype, int group_axis, const std::string& op_name,
                         const Span& span) {
  if (dtype != DataType::FP4E2M1X2) return;
  CHECK_SPAN(src->dtype_ == DataType::FP16 || src->dtype_ == DataType::BF16, span)
      << "The operator " << op_name << " requires FP16 or BF16 source for MXFP4_E2M1";
  // Axis0 receives a transposed source in the public API. Its contiguous
  // dimension is N, so its packed row stride must contain 32 whole bytes.
  auto packed_extent = As<ConstInt>(src->shape_[group_axis == 0 ? 0 : 1]);
  CHECK_SPAN(packed_extent && packed_extent->value_ > 0 && packed_extent->value_ % 2 == 0, span)
      << "The operator " << op_name << " requires an even packed FP4 extent";
  if (group_axis == 0) {
    CHECK_SPAN(packed_extent->value_ % 64 == 0, span)
        << "The operator " << op_name << " requires N divisible by 64 for MXFP4 carrier row alignment";
  }
}

int64_t GetStaticElementCount(const TileTypePtr& type, const std::string& operand, const std::string& op_name,
                              const Span& span) {
  const TileView view = tile_view_semantics::GetEffectiveTileView(*type);
  int64_t count = 1;
  // PTOAS TQUANT/X2ZZ need static positive valid extents (see EmitStaticValidTileView).
  for (const auto& dim_expr : view.valid_shape) {
    auto dim = As<ConstInt>(dim_expr);
    CHECK_SPAN(dim && dim->value_ > 0, span)
        << "The operator " << op_name << " requires " << operand << " to have a static positive valid_shape";
    CHECK_SPAN(count <= std::numeric_limits<int64_t>::max() / dim->value_, span)
        << "The operator " << op_name << " " << operand << " valid element count overflows int64";
    count *= dim->value_;
  }
  return count;
}

int64_t GetStaticPhysicalElementCount(const TileTypePtr& type, const std::string& operand,
                                      const std::string& op_name, const Span& span) {
  int64_t count = 1;
  for (const auto& dim_expr : type->shape_) {
    auto dim = As<ConstInt>(dim_expr);
    CHECK_SPAN(dim && dim->value_ > 0, span) << "The operator " << op_name << " requires " << operand
                                             << " to have a static positive physical shape";
    CHECK_SPAN(count <= std::numeric_limits<int64_t>::max() / dim->value_, span)
        << "The operator " << op_name << " " << operand << " physical element count overflows int64";
    count *= dim->value_;
  }
  return count;
}

int ParseGroupAxis(const std::vector<std::pair<std::string, std::any>>& kwargs, const std::string& op_name,
                   const Span& span, bool required) {
  std::optional<int> group_axis_kw;
  for (const auto& [key, value] : kwargs) {
    if (key == "group_axis") {
      group_axis_kw = AnyCast<int>(value, "kwarg key: group_axis");
      CHECK_SPAN(*group_axis_kw == 0 || *group_axis_kw == 1, span)
          << "The operator " << op_name << " group_axis must be 0 or 1, but got " << *group_axis_kw;
    }
  }
  CHECK_SPAN(!required || group_axis_kw.has_value(), span)
      << "The operator " << op_name << " requires group_axis";
  return group_axis_kw.value_or(1);
}

DataType ParseDtype(const std::vector<std::pair<std::string, std::any>>& kwargs) {
  DataType dtype = DataType::FP8E4M3FN;
  for (const auto& [key, value] : kwargs) {
    if (key == "dtype") dtype = AnyCast<DataType>(value, "kwarg key: dtype");
  }
  return dtype;
}

void CheckQuantAlgorithm(const std::vector<std::pair<std::string, std::any>>& kwargs, DataType src_dtype,
                         int group_axis, const std::string& op_name, const Span& span) {
  std::string algorithm = "ocp";
  for (const auto& [key, value] : kwargs) {
    if (key == "quant_scale_alg") algorithm = AnyCast<std::string>(value, "kwarg key: quant_scale_alg");
  }
  CHECK_SPAN(algorithm == "ocp" || algorithm == "nv", span)
      << "The operator " << op_name << " requires quant_scale_alg 'ocp' or 'nv', but got " << algorithm;
  CHECK_SPAN(algorithm != "nv" || group_axis != 1 || src_dtype == DataType::FP32, span)
      << "The operator " << op_name << " does not support axis1 FP16/BF16 NV quantization in PTOAS";
}

void CheckSrcBasics(const TileTypePtr& src_type, const std::string& op_name, const Span& span,
                    bool require_full_valid = true) {
  CHECK_SPAN(src_type, span) << "The operator " << op_name << " requires src to be a TileType";
  CHECK_SPAN(src_type->shape_.size() == 2, span) << "The operator " << op_name << " requires 2D src tile";
  const TileView src_view = tile_view_semantics::GetEffectiveTileView(*src_type);
  CHECK_SPAN(!require_full_valid ||
                 tile_view_semantics::ShapeExprListsEquivalent(src_view.valid_shape, src_type->shape_),
             span)
      << "The operator " << op_name
      << " does not support a partial src valid_shape; valid_shape must match the physical shape";
  const bool src_supported = src_type->dtype_ == DataType::FP16 || src_type->dtype_ == DataType::BF16 ||
                             src_type->dtype_ == DataType::FP32;
  CHECK_SPAN(src_supported, span) << "The operator " << op_name
                                  << " requires src dtype in {FP16, FP32, BF16}, but got "
                                  << src_type->dtype_.ToString();
}

// Public tile.tquant_mx: one src -> TupleType{FP8 or packed FP4 quant, FP8E8M0 scale}.
TypePtr DeducePublicTQuantMxType(const std::vector<ExprPtr>& args,
                                 const std::vector<std::pair<std::string, std::any>>& kwargs) {
  constexpr const char* kOpName = "tile.tquant_mx";
  const Span span = args.empty() ? Span::unknown() : args[0]->span_;
  CHECK_SPAN(args.size() == 1, span) << "The operator " << kOpName << " requires exactly 1 argument";
  auto src_type = As<TileType>(args[0]->GetType());
  CheckSrcBasics(src_type, kOpName, span);

  const DataType dtype = ParseDtype(kwargs);
  const int group_axis = ParseGroupAxis(kwargs, kOpName, span, /*required=*/true);
  const MxQuantTypeInfo type_info = ResolveMxQuantType(dtype, kOpName, span);
  CheckFp4QuantSource(src_type, dtype, group_axis, kOpName, span);
  CheckQuantAlgorithm(kwargs, src_type->dtype_, group_axis, kOpName, span);

  ExprPtr dim0 = src_type->shape_[0];
  ExprPtr dim1 = src_type->shape_[1];
  auto dim0_const = As<ConstInt>(dim0);
  auto dim1_const = As<ConstInt>(dim1);
  CHECK_SPAN(dim0_const && dim1_const, span)
      << "The operator " << kOpName << " requires static " << (group_axis == 0 ? "N and K" : "M and K");
  CHECK_SPAN(dim0_const->value_ > 0 && dim1_const->value_ > 0, span)
      << "The operator " << kOpName << " requires positive dimensions";
  CHECK_SPAN(dim1_const->value_ % 64 == 0, span)
      << "The operator " << kOpName << " requires K divisible by 64, but got " << dim1_const->value_;
  if (group_axis == 1) {
    CHECK_SPAN(dim0_const->value_ % kMXSFractalRows == 0, span)
        << "The operator " << kOpName << " requires M divisible by " << kMXSFractalRows << ", but got "
        << dim0_const->value_;
  } else {
    CHECK_SPAN(dim0_const->value_ % kMXGroupSize == 0, span)
        << "The operator " << kOpName << " with group_axis=0 requires N divisible by " << kMXGroupSize
        << " for A5 Vec row-byte alignment, but got " << dim0_const->value_;
  }

  std::vector<ExprPtr> dst_shape = src_type->shape_;
  if (group_axis == 0) dst_shape = {dim1, dim0};  // src [N,K] → quant [K,N]
  if (dtype == DataType::FP4E2M1X2) {
    const int64_t packed_cols = (group_axis == 0 ? dim0_const->value_ : dim1_const->value_) / 2;
    dst_shape[1] = std::make_shared<ConstInt>(packed_cols, DataType::INDEX, span);
  }
  TileView dst_view;
  dst_view.valid_shape = dst_shape;
  dst_view.blayout = TileLayout::row_major;
  dst_view.slayout = TileLayout::none_box;
  auto dst_type = std::make_shared<TileType>(dst_shape, type_info.public_dst_dtype, std::nullopt, dst_view);

  TileView scale_view;
  std::vector<ExprPtr> scale_shape;
  if (group_axis == 1) {
    scale_shape = {dim0, std::make_shared<ConstInt>(dim1_const->value_ / kMXGroupSize, DataType::INDEX,
                                                    Span::unknown())};
  } else {
    scale_shape = {
        std::make_shared<ConstInt>(dim1_const->value_ / kMXGroupSize, DataType::INDEX, Span::unknown()),
        dim0};
  }
  scale_view.valid_shape = scale_shape;
  const TileLayout scale_layout = group_axis == 0 ? TileLayout::col_major : TileLayout::row_major;
  scale_view.blayout = scale_layout;
  scale_view.slayout = scale_layout;
  scale_view.fractal = kMXScaleFractal;
  auto scale_type = std::make_shared<TileType>(scale_shape, DataType::FP8E8M0, std::nullopt, scale_view);
  return std::make_shared<TupleType>(std::vector<TypePtr>{dst_type, scale_type});
}

// Raw tile.tquant_mx_raw: (src, max_ws, scaling_ws) -> TupleType{INT8 or packed FP4, UINT8 exp}.
// max/scaling are write-only workspace inputs (gather_compare tmp style).
TypePtr DeduceRawTQuantMxType(const std::vector<ExprPtr>& args,
                              const std::vector<std::pair<std::string, std::any>>& kwargs) {
  constexpr const char* kOpName = "tile.tquant_mx_raw";
  const Span span = args.empty() ? Span::unknown() : args[0]->span_;
  CHECK_SPAN(args.size() == 3, span) << "The operator " << kOpName
                                     << " requires src, max, and scaling workspaces";
  auto src_type = As<TileType>(args[0]->GetType());
  CheckSrcBasics(src_type, kOpName, span, /*require_full_valid=*/false);

  const DataType dtype = ParseDtype(kwargs);
  const int group_axis = ParseGroupAxis(kwargs, kOpName, span, /*required=*/true);
  const MxQuantTypeInfo type_info = ResolveMxQuantType(dtype, kOpName, span);
  CheckQuantAlgorithm(kwargs, src_type->dtype_, group_axis, kOpName, span);
  bool interleave = false;
  for (const auto& [key, value] : kwargs) {
    if (key == "interleave") interleave = AnyCast<bool>(value, "kwarg key: interleave");
  }
  CHECK_SPAN(!interleave || (group_axis == 0 && src_type->dtype_ != DataType::FP32), span)
      << "The operator " << kOpName << " interleave requires axis0 FP16/BF16 source";

  if (dtype == DataType::FP4E2M1X2) {
    CHECK_SPAN(src_type->dtype_ == DataType::FP16 || src_type->dtype_ == DataType::BF16, span)
        << "The operator " << kOpName << " requires FP16 or BF16 source for MXFP4_E2M1";
  }

  auto dim0_const = As<ConstInt>(src_type->shape_[0]);
  auto dim1_const = As<ConstInt>(src_type->shape_[1]);
  CHECK_SPAN(dim0_const && dim1_const, span) << "The operator " << kOpName << " requires static M and K";
  CHECK_SPAN(dim0_const->value_ > 0 && dim1_const->value_ > 0, span)
      << "The operator " << kOpName << " requires positive dimensions";
  const auto src_view = tile_view_semantics::GetEffectiveTileView(*src_type);
  CHECK_SPAN(src_view.valid_shape.size() == 2, span)
      << "The operator " << kOpName << " requires a 2D valid shape";
  auto valid_rows = As<ConstInt>(src_view.valid_shape[0]);
  auto valid_cols = As<ConstInt>(src_view.valid_shape[1]);
  CHECK_SPAN(valid_rows && valid_cols && valid_rows->value_ > 0 && valid_cols->value_ > 0 &&
                 valid_rows->value_ <= dim0_const->value_ && valid_cols->value_ <= dim1_const->value_,
             span)
      << "The operator " << kOpName << " requires static positive valid dimensions within physical shape";
  CHECK_SPAN(!interleave || (dim0_const->value_ % 64 == 0 && valid_rows->value_ % 64 == 0), span)
      << "The operator " << kOpName << " interleave requires physical and valid rows divisible by 64";
  if (dtype == DataType::FP4E2M1X2) {
    CHECK_SPAN(dim1_const->value_ % 2 == 0 && valid_cols->value_ % 2 == 0, span)
        << "The operator " << kOpName << " requires even physical and valid columns for packed FP4";
    CHECK_SPAN(dim1_const->value_ % 64 == 0, span)
        << "The operator " << kOpName
        << " requires physical columns divisible by 64 for MXFP4 carrier row alignment";
  }
  if (group_axis == 1) {
    CHECK_SPAN(valid_cols->value_ == dim1_const->value_, span)
        << "The operator " << kOpName << " axis1 flat exponents require physical columns equal valid columns";
    CHECK_SPAN(valid_cols->value_ % kMXGroupSize == 0, span)
        << "The operator " << kOpName << " requires K divisible by " << kMXGroupSize
        << " for group_axis=1, but got " << valid_cols->value_;
  } else {
    CHECK_SPAN(valid_rows->value_ % kMXGroupSize == 0, span)
        << "The operator " << kOpName << " requires dim0 divisible by " << kMXGroupSize
        << " for group_axis=0, but got " << valid_rows->value_;
  }

  auto max_type = As<TileType>(args[1]->GetType());
  auto scaling_type = As<TileType>(args[2]->GetType());
  CHECK_SPAN(max_type && max_type->dtype_ == src_type->dtype_, span)
      << "The operator " << kOpName << " requires max scratch dtype to match src ("
      << src_type->dtype_.ToString() << "), but got "
      << (max_type ? max_type->dtype_.ToString() : std::string("<non-tile>"));
  CHECK_SPAN(scaling_type && scaling_type->dtype_ == src_type->dtype_, span)
      << "The operator " << kOpName << " requires scaling scratch dtype to match src ("
      << src_type->dtype_.ToString() << "), but got "
      << (scaling_type ? scaling_type->dtype_.ToString() : std::string("<non-tile>"));

  const int64_t grouped_rows = group_axis == 0 ? valid_rows->value_ / kMXGroupSize : valid_rows->value_;
  const int64_t grouped_cols = group_axis == 0 ? valid_cols->value_ : valid_cols->value_ / kMXGroupSize;
  CHECK_SPAN(grouped_rows <= std::numeric_limits<int64_t>::max() / grouped_cols, span)
      << "The operator " << kOpName << " scale-group count overflows int64";
  const int64_t groups = grouped_rows * grouped_cols;
  CHECK_SPAN(GetStaticElementCount(max_type, "max scratch", kOpName, span) == groups, span)
      << "The operator " << kOpName << " requires max scratch valid element count " << groups;
  CHECK_SPAN(GetStaticElementCount(scaling_type, "scaling scratch", kOpName, span) == groups, span)
      << "The operator " << kOpName << " requires scaling scratch valid element count " << groups;

  TileView dst_view;
  std::vector<ExprPtr> dst_shape = src_type->shape_;
  if (dtype == DataType::FP4E2M1X2) {
    dst_shape[1] = std::make_shared<ConstInt>(dim1_const->value_ / 2, DataType::INDEX, span);
  }
  dst_view.valid_shape = src_view.valid_shape;
  if (dtype == DataType::FP4E2M1X2) {
    dst_view.valid_shape[1] = std::make_shared<ConstInt>(valid_cols->value_ / 2, DataType::INDEX, span);
  }
  dst_view.blayout = TileLayout::row_major;
  dst_view.slayout = TileLayout::none_box;
  auto raw_dst_type = std::make_shared<TileType>(dst_shape, type_info.raw_dst_dtype, std::nullopt, dst_view);

  const int64_t aux_rows = group_axis == 0 ? grouped_rows : 1;
  const int64_t aux_cols = group_axis == 0 ? grouped_cols : groups;
  auto make_dim = [](int64_t v) { return std::make_shared<ConstInt>(v, DataType::INDEX, Span::unknown()); };
  std::vector<ExprPtr> exp_shape = {make_dim(aux_rows), make_dim(aux_cols)};
  TileView exp_view;
  exp_view.valid_shape = exp_shape;
  if (group_axis == 0) {
    exp_shape[1] = src_type->shape_[1];
  } else {
    CHECK_SPAN(aux_cols <= std::numeric_limits<int64_t>::max() - 31, span)
        << "The operator " << kOpName << " padded exponent columns overflow int64";
    exp_shape[1] = make_dim((aux_cols + 31) / 32 * 32);
  }
  if (interleave) {
    CHECK_SPAN(dim1_const->value_ <= (std::numeric_limits<int64_t>::max() - 31) / 2, span)
        << "The operator " << kOpName << " interleaved exponent columns overflow int64";
    exp_shape = {make_dim(dim0_const->value_ / 64), make_dim((dim1_const->value_ * 2 + 31) / 32 * 32)};
    exp_view.valid_shape = {make_dim(valid_rows->value_ / 64), make_dim(valid_cols->value_ * 2)};
  }
  exp_view.blayout = TileLayout::row_major;
  exp_view.slayout = TileLayout::none_box;
  exp_view.fractal = kMXScaleFractal;
  auto raw_exp_type = std::make_shared<TileType>(exp_shape, DataType::UINT8, std::nullopt, exp_view);
  return std::make_shared<TupleType>(std::vector<TypePtr>{raw_dst_type, raw_exp_type});
}

// Public + internal tile.tmov_x2zz: (src, tmp) -> UINT8 ZZ TileType.
//
// Axis1: TQUANT emits a legacy-flat exp `[1, M*G]`; ZZ dst is `[M, G]` with
// align16 row padding. Callers (LowerCompositeOps) pass `dst_rows`/`dst_cols`.
// Axis1 tmp contract (bytes): 64 + ceil(dst_rows/16) * dst_cols, then typically
// 32-byte-align the physical allocation.
// Axis0 (TMovDnTo2Zz): DN `[M̂,N]` -> ZZ `[N,M̂]`; ISA still requires a Vec tmp
// operand — PyPTO uses a minimal 32-byte-aligned scratch (one Vec pad unit).
TypePtr DeduceTileTMovX2ZzType(const std::vector<ExprPtr>& args,
                               const std::vector<std::pair<std::string, std::any>>& kwargs) {
  constexpr const char* kOpName = "tile.tmov_x2zz";
  const Span span = args.empty() ? Span::unknown() : args[0]->span_;
  CHECK_SPAN(args.size() == 2, span) << "The operator " << kOpName << " requires src and tmp";
  auto src = As<TileType>(args[0]->GetType());
  auto tmp = As<TileType>(args[1]->GetType());
  CHECK_SPAN(src && tmp, span) << "The operator " << kOpName << " requires TileType operands";
  CHECK_SPAN(src->dtype_ == DataType::UINT8 && tmp->dtype_ == DataType::UINT8, span)
      << "The operator " << kOpName << " requires raw UINT8 src/tmp";
  CHECK_SPAN(src->shape_.size() == 2 && tmp->shape_.size() == 2, span)
      << "The operator " << kOpName << " requires rank-2 tiles";

  const int group_axis = ParseGroupAxis(kwargs, kOpName, span, /*required=*/true);
  const TileView src_view = tile_view_semantics::GetEffectiveTileView(*src);
  const TileView tmp_view = tile_view_semantics::GetEffectiveTileView(*tmp);
  CHECK_SPAN(src_view.blayout == TileLayout::row_major && src_view.slayout == TileLayout::none_box, span)
      << "The operator " << kOpName << " requires src row_major/none_box";
  CHECK_SPAN(tmp_view.blayout == TileLayout::row_major && tmp_view.slayout == TileLayout::none_box, span)
      << "The operator " << kOpName << " requires tmp row_major/none_box";

  auto src_rows = As<ConstInt>(src_view.valid_shape[0]);
  auto src_cols = As<ConstInt>(src_view.valid_shape[1]);
  CHECK_SPAN(src_rows && src_cols, span) << "The operator " << kOpName << " requires static src valid shapes";

  std::optional<int64_t> dst_rows_kw;
  std::optional<int64_t> dst_cols_kw;
  for (const auto& [key, value] : kwargs) {
    if (key == "dst_rows") dst_rows_kw = AnyCast<int>(value, "kwarg key: dst_rows");
    if (key == "dst_cols") dst_cols_kw = AnyCast<int>(value, "kwarg key: dst_cols");
  }

  constexpr int64_t kVecByteAlign = 32;  // Vec pad unit; not kMXGroupSize.
  int64_t zz_rows = 0;
  int64_t zz_cols = 0;
  int64_t zz_physical_rows = 0;
  int64_t zz_physical_cols = 0;
  int64_t tmp_bytes = kVecByteAlign;
  if (group_axis == 1) {
    CHECK_SPAN(dst_rows_kw.has_value() && dst_cols_kw.has_value(), span)
        << "The operator " << kOpName
        << " with group_axis=1 requires dst_rows and dst_cols kwargs "
           "(ZZ [M,G] cannot be recovered from legacy-flat TQUANT exp [1,M*G] alone)";
    zz_rows = *dst_rows_kw;
    zz_cols = *dst_cols_kw;
    CHECK_SPAN(zz_rows > 0 && zz_cols > 0, span)
        << "The operator " << kOpName << " requires positive dst_rows/dst_cols";
    CHECK_SPAN(zz_cols % 2 == 0, span)
        << "The operator " << kOpName << " requires axis1 grouped exponent columns to be even";
    CHECK_SPAN(zz_rows <= std::numeric_limits<int64_t>::max() / zz_cols, span)
        << "The operator " << kOpName << " dst element count overflows int64";
    const int64_t groups = zz_rows * zz_cols;
    // Accept either legacy-flat [1, groups] or already-shaped [M, G] sources.
    const bool flat_src = src_rows->value_ == 1 && src_cols->value_ == groups;
    const bool shaped_src = src_rows->value_ == zz_rows && src_cols->value_ == zz_cols;
    CHECK_SPAN(flat_src || shaped_src, span)
        << "The operator " << kOpName << " axis1 src valid shape must be [1, " << groups << "] or ["
        << zz_rows << ", " << zz_cols << "], but got [" << src_rows->value_ << ", " << src_cols->value_
        << "]";
    CHECK_SPAN(zz_rows <= std::numeric_limits<int64_t>::max() - 15, span)
        << "The operator " << kOpName << " padded row count overflows int64";
    const int64_t row_blocks = (zz_rows + 15) / 16;
    CHECK_SPAN(row_blocks <= std::numeric_limits<int64_t>::max() / 16, span)
        << "The operator " << kOpName << " padded row count overflows int64";
    zz_physical_rows = row_blocks * 16;
    zz_physical_cols = zz_cols;
    const int64_t padded_elements = zz_physical_rows * zz_physical_cols;
    CHECK_SPAN(GetStaticPhysicalElementCount(src, "src", kOpName, span) >= padded_elements, span)
        << "The operator " << kOpName << " requires src capacity for align16(dst rows) * dst cols";
    CHECK_SPAN(row_blocks <= (std::numeric_limits<int64_t>::max() - 64) / zz_cols, span)
        << "The operator " << kOpName << " tmp capacity overflows int64";
    tmp_bytes = 64 + row_blocks * zz_cols;
  } else {
    // TMovDnTo2Zz: DN [M̂,N] -> ZZ [N,M̂].
    CHECK_SPAN(src_rows->value_ >= 2 && src_rows->value_ % 2 == 0, span)
        << "The operator " << kOpName << " requires axis0 src rows to be an even count >= 2";
    CHECK_SPAN(src_cols->value_ % kMXSFractalRows == 0, span)
        << "The operator " << kOpName << " requires axis0 src columns divisible by " << kMXSFractalRows;
    auto src_physical_cols = As<ConstInt>(src->shape_[1]);
    CHECK_SPAN(src_physical_cols && src_physical_cols->value_ == src_cols->value_, span)
        << "The operator " << kOpName << " requires axis0 source physical stride to be tight";
    zz_rows = src_cols->value_;
    zz_cols = src_rows->value_;
    zz_physical_rows = zz_rows;
    zz_physical_cols = zz_cols;
    tmp_bytes = kVecByteAlign;  // minimal ISA-required Vec tmp pad
  }
  CHECK_SPAN(GetStaticPhysicalElementCount(tmp, "tmp", kOpName, span) >= tmp_bytes, span)
      << "The operator " << kOpName << " requires tmp capacity of at least " << tmp_bytes << " bytes (axis"
      << group_axis << " X-to-ZZ workspace contract)";

  auto make_dim = [](int64_t v) { return std::make_shared<ConstInt>(v, DataType::INDEX, Span::unknown()); };
  std::vector<ExprPtr> physical = {make_dim(zz_physical_rows), make_dim(zz_physical_cols)};
  TileView dst_view;
  dst_view.valid_shape = {make_dim(zz_rows), make_dim(zz_cols)};
  dst_view.blayout = TileLayout::row_major;
  dst_view.slayout = TileLayout::row_major;
  dst_view.fractal = kMXScaleFractal;
  return std::make_shared<TileType>(physical, DataType::UINT8, std::nullopt, dst_view);
}

}  // namespace

REGISTER_OP("tile.tquant_mx")
    .set_op_category("TileOp")
    .set_description(
        "MX block-32 dynamic quantization: TupleType{quantized data, e8m0_scale FP8E8M0}. "
        "LowerCompositeOps rewrites this into tile.tquant_mx_raw + tile.tmov_x2zz (value-returning), "
        "then reinterpret/transpose_view to the public dtypes. dtype is FP8E4M3FN or FP4E2M1X2. group_axis "
        "is "
        "PTOAS grpAxis (1=A-side [M,K], 0=B-side [N,K] with transpose).")
    .add_argument("src", "Source tile (FP16/FP32/BF16, 2D)")
    .set_attr<DataType>("dtype")
    .set_attr<int>("group_axis")
    .set_attr<std::string>("quant_scale_alg")
    .set_input_memory(0, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    // Two results (dst, scale), carried by the deduced TupleType; the single
    // argument is a pure read.
    .set_output_arity(2)
    .no_arg_writes()
    .not_inplace_safe()
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeducePublicTQuantMxType(args, kwargs);
    });

REGISTER_OP("tile.tquant_mx_raw")
    .set_op_category("TileOp")
    .set_description(
        "Internal value-returning MX TQUANT: ins(src, max_ws, scaling_ws) outs TupleType{INT8 or "
        "FP4E2M1X2 dst, UINT8 exp}. max/scaling are write-only workspace inputs. Lowers to pto.tquant.mx.")
    .add_argument("src", "Source tile (FP16/FP32/BF16, 2D)")
    .add_argument("max", "Per-group max workspace matching src dtype (write-only)")
    .add_argument("scaling", "Per-group scaling workspace matching src dtype (write-only)")
    .set_attr<DataType>("dtype")
    .set_attr<int>("group_axis")
    .set_attr<std::string>("quant_scale_alg")
    .set_attr<bool>("interleave")
    .set_input_memory(0, MemorySpace::Vec)
    .set_input_memory(1, MemorySpace::Vec)
    .set_input_memory(2, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    // Two results (raw dst, raw exp), carried by the deduced TupleType. max and
    // scaling are hardware-written scratch that LowerCompositeOps synthesizes --
    // written, but carrying no result the caller reads.
    .set_output_arity(2)
    .set_arg_effect(0, ArgEffect::Read)
    .set_arg_effect(1, ArgEffect::Write)
    .set_arg_effect(2, ArgEffect::Write)
    .set_workspace_arg(1)
    .set_workspace_arg(2)
    .not_inplace_safe()
    .functional_execution_memory_access()
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceRawTQuantMxType(args, kwargs);
    });

REGISTER_OP("tile.tmov_x2zz")
    .set_op_category("TileOp")
    .set_description(
        "Exponent X-to-ZZ layout conversion: ins(src, tmp) outs(ZZ UINT8 tile). tmp is a write-only "
        "workspace. Axis1 requires capacity 64+ceil(rows/16)*cols bytes; axis0 requires a minimal "
        "32-byte-aligned Vec pad. Codegen emits the non-scaling third-operand form of pto.tmov. "
        "A5-only.")
    .add_argument("src", "Canonical exponent source (raw UINT8)")
    .add_argument("tmp", "X-to-ZZ temporary tile (raw UINT8, write-only)")
    .set_attr<int>("group_axis")
    .set_attr<int>("dst_rows")
    .set_attr<int>("dst_cols")
    .set_input_memory(0, MemorySpace::Vec)
    .set_input_memory(1, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .set_arg_effect(1, ArgEffect::Write)
    .not_inplace_safe()
    .functional_execution_memory_access()
    .set_lane_invariant_arg(1)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileTMovX2ZzType(args, kwargs);
    });

}  // namespace ir
}  // namespace pypto
