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

#include <algorithm>
#include <any>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "pypto/backend/common/backend.h"
#include "pypto/backend/common/backend_config.h"
#include "pypto/backend/common/backend_handler.h"
#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/cast_saturation.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/transforms/pass_context.h"
#include "pypto/ir/transforms/utils/tile_conversion_utils.h"
#include "pypto/ir/type.h"
#include "src/ir/transforms/lower_composite/lower_composite_builder.h"
#include "src/ir/transforms/lower_composite/lower_composite_rules.h"

namespace pypto {
namespace ir {
namespace lower_composite {

namespace {

ExprPtr MxDimension(int64_t value, const Span& span) {
  return std::make_shared<ConstInt>(value, DataType::INDEX, span);
}

ExprPtr MxShape(int64_t rows, int64_t cols, const Span& span) {
  return std::make_shared<MakeTuple>(std::vector<ExprPtr>{MxDimension(rows, span), MxDimension(cols, span)},
                                     span);
}

ExprPtr PackMxExponent(const ExprPtr& exponent, int64_t group_rows, int64_t group_cols, int pack_axis,
                       bool transpose_scale, LoweringBuilder& b, const Span& span) {
  auto& reg = OpRegistry::GetInstance();
  // Axis1 X-to-ZZ needs 64 + ceil(rows/16)*cols scratch bytes. The axis0
  // TMovDnTo2Zz form needs a Vec scratch operand of at least one 32-byte block.
  constexpr int64_t kVecByteAlign = 32;
  int64_t tmp_bytes = kVecByteAlign;
  if (pack_axis == 1) {
    INTERNAL_CHECK_SPAN(group_rows <= std::numeric_limits<int64_t>::max() - 15, span)
        << "Internal error: tile.tquant_mx padded exponent rows overflow int64";
    const int64_t row_blocks = (group_rows + 15) / 16;
    INTERNAL_CHECK_SPAN(row_blocks <= (std::numeric_limits<int64_t>::max() - 64) / group_cols, span)
        << "Internal error: tile.tquant_mx exponent temporary size overflows int64";
    tmp_bytes = 64 + row_blocks * group_cols;
  }
  INTERNAL_CHECK_SPAN(tmp_bytes <= std::numeric_limits<int64_t>::max() - (kVecByteAlign - 1), span)
      << "Internal error: tile.tquant_mx aligned exponent temporary size overflows int64";
  const int64_t tmp_physical_bytes = (tmp_bytes + kVecByteAlign - 1) / kVecByteAlign * kVecByteAlign;
  auto tmp = b.Bind("tq_x2zz_tmp",
                    reg.Create("tile.create", {MxShape(1, tmp_physical_bytes, span)},
                               {{"dtype", DataType::UINT8}, {"target_memory", MemorySpace::Vec}}, span),
                    span);

  std::vector<std::pair<std::string, std::any>> x2zz_kwargs = {{"group_axis", pack_axis}};
  if (pack_axis == 1) {
    INTERNAL_CHECK_SPAN(
        group_rows <= std::numeric_limits<int>::max() && group_cols <= std::numeric_limits<int>::max(), span)
        << "Internal error: tile.tquant_mx exponent tile shape exceeds X-to-ZZ limits";
    x2zz_kwargs.emplace_back("dst_rows", static_cast<int>(group_rows));
    x2zz_kwargs.emplace_back("dst_cols", static_cast<int>(group_cols));
  }
  auto zz = b.Bind("tq_exp_zz", reg.Create("tile.tmov_x2zz", {exponent, tmp}, x2zz_kwargs, span), span);
  auto scale = b.Bind("tq_scale",
                      reg.Create("tile.reinterpret_view", {zz}, {{"dtype", DataType::FP8E8M0}}, span), span);
  if (transpose_scale) {
    // ZZ [N, K/32] and NN [K/32, N] use the same bytes.
    return b.Bind("tq_scale_nn", reg.Create("tile.transpose_view", {scale}, {}, span), span);
  }
  return scale;
}

ExprPtr LowerE5M2Quantization(const ExprPtr& src, int group_axis, LoweringBuilder& b, const Span& span) {
  // This recipe needs the native E5M2 narrowing, signed ordering
  // comparisons and the MX exponent-packing operator. Query backend policy
  // through the active pass context, independent of its architecture name.
  const auto* handler = tile_conversion_utils::ActiveBackendHandler();
  const auto* context = PassContext::Current();
  if (handler != nullptr) {
    const auto* target = context != nullptr ? context->GetBackend() : backend::BackendConfig::GetBackend();
    const auto& edges = handler->GetTcvtAdjacency().edges;
    const auto narrowing = std::make_pair(DataType::FP32, DataType::FP8E5M2);
    CHECK_SPAN(std::find(edges.begin(), edges.end(), narrowing) != edges.end() &&
                   handler->SupportsOrderingCompareDataType(DataType::INT32) &&
                   target->GetOpInfo("tile.tmov_x2zz") != nullptr,
               span)
        << "tile.tquant_mx with FP8E5M2 requires native E5M2 narrowing and MX exponent packing";
  }
  const auto src_type = As<TileType>(src->GetType());
  const auto rows = As<ConstInt>(src_type->shape_[0])->value_;
  const auto cols = As<ConstInt>(src_type->shape_[1])->value_;
  constexpr int64_t kGroupSize = 32;
  const int64_t group_cols = cols / kGroupSize;
  INTERNAL_CHECK_SPAN(rows <= std::numeric_limits<int64_t>::max() / group_cols, span)
      << "Internal error: tile.tquant_mx scale-group count overflows int64";
  const int64_t groups = rows * group_cols;
  auto& reg = OpRegistry::GetInstance();
  auto bind = [&](const std::string& name, const std::string& op, const std::vector<ExprPtr>& args,
                  const std::vector<std::pair<std::string, std::any>>& kwargs = {}) {
    return b.Bind(name, reg.Create(op, args, kwargs, span), span);
  };
  auto i32 = [&](int64_t value) { return std::make_shared<ConstInt>(value, DataType::INT32, span); };
  auto cast = [&](const std::string& name, const ExprPtr& input, DataType dtype) {
    constexpr int kRoundNearestEven = 1;
    std::vector<std::pair<std::string, std::any>> kwargs = {{"target_type", dtype},
                                                            {"mode", kRoundNearestEven}};
    if (dtype == DataType::FP8E5M2) {
      kwargs.emplace_back("saturation_mode", static_cast<int>(SaturationMode::kOn));
    }
    return bind(name, "tile.cast", {input}, kwargs);
  };
  auto alias = [&](const std::string& name, const ExprPtr& input, DataType dtype) {
    return bind(name, "tile.reinterpret_view", {input}, {{"dtype", dtype}});
  };
  auto select = [&](const std::string& name, const ExprPtr& mask, const ExprPtr& on_true,
                    const ExprPtr& on_false) {
    const std::vector<ExprPtr> args = {mask, on_true, on_false};
    auto call = As<Call>(reg.Create("tile.select", args, {}, span));
    return b.Bind(name, LowerTileSelectRule(call, args, b), span);
  };

  // The input uses [M,K] or [N,K], so consecutive groups of 32 are rows in
  // either orientation. Reduce unsigned FP magnitudes via signed INT32:
  // clearing the sign bit makes finite values monotonic and puts every NaN
  // above infinity, without depending on the hardware float max NaN mode.
  const auto grouped_shape = MxShape(groups, kGroupSize, span);
  const auto flat_shape = MxShape(1, groups, span);
  const auto source_f32 =
      src_type->dtype_ == DataType::FP32
          ? src
          : bind("tq_f32", "tile.cast", {src}, {{"target_type", DataType::FP32}, {"mode", 0}});
  const auto grouped = bind("tq_groups", "tile.reshape", {source_f32, grouped_shape});
  const auto bits = alias("tq_bits", grouped, DataType::INT32);
  const auto magnitude = bind("tq_abs_bits", "tile.ands", {bits, i32(0x7FFFFFFF)});
  const auto scratch = bind("tq_amax_tmp", "tile.create", {grouped_shape},
                            {{"dtype", DataType::INT32}, {"target_memory", MemorySpace::Vec}});
  const auto maximum = bind("tq_amax", "tile.row_max", {magnitude, scratch});
  // Row reductions produce contiguous column vectors. Flatten to a single
  // row for the elementwise exponent arithmetic and scale packing.
  const auto maximum_row = bind("tq_amax_row", "tile.reshape", {maximum, flat_shape});

  // OCP shared exponent is floor(log2(amax)) - Emax, with Emax = 15 for
  // E5M2. FP32 and E8M0 have the same exponent bias. Clamp zero and tiny
  // groups to the smallest E8M0 scale, whose reciprocal is 2^127.
  constexpr int kExponentShift = 23;
  constexpr int kE5M2MaxExponent = 15;
  constexpr int kE8M0NaN = 255;
  constexpr int kReciprocalExponentBias = 254;
  constexpr int kCompareGreater = 4;
  const auto ieee_exp = bind("tq_f32_exp", "tile.shrs", {maximum_row, i32(kExponentShift)});
  const auto shared = bind("tq_shared", "tile.subs", {ieee_exp, i32(kE5M2MaxExponent)});
  const auto clamped = bind("tq_clamped", "tile.maximums", {shared, i32(0)});
  const auto nan_mask =
      bind("tq_nan", "tile.cmps", {maximum_row, i32(0x7F800000)}, {{"cmp_type", kCompareGreater}});
  const auto scale_codes = select("tq_codes", nan_mask, i32(kE8M0NaN), clamped);
  const auto exp_bytes = cast("tq_exp", scale_codes, DataType::UINT8);

  const auto bias = bind("tq_recip_bias", "tile.full", {flat_shape, i32(kReciprocalExponentBias)},
                         {{"dtype", DataType::INT32}});
  const auto recip_exp = bind("tq_recip_exp", "tile.sub", {bias, clamped});
  const auto recip_bits = bind("tq_recip_bits", "tile.shls", {recip_exp, i32(kExponentShift)});
  const auto recip_nan = select("tq_recip_or_nan", nan_mask, i32(0x7FC00000), recip_bits);
  const auto recip_f32 = alias("tq_recip_f32", recip_nan, DataType::FP32);
  const auto reciprocal = bind("tq_recip", "tile.reshape", {recip_f32, MxShape(groups, 1, span)});
  const auto scaled = bind("tq_scaled", "tile.row_expand_mul", {grouped, reciprocal});

  // The A5 E5M2 TCVT can round a finite overflow to infinity even with
  // satmode=ON. Clamp the normalized values explicitly to the finite E5M2
  // range. Ordered comparisons leave NaNs from the group scale intact.
  constexpr int kCompareLess = 2;
  constexpr double kE5M2Max = 57344.0;
  const auto min_value = std::make_shared<ConstFloat>(-kE5M2Max, DataType::FP32, span);
  const auto max_value = std::make_shared<ConstFloat>(kE5M2Max, DataType::FP32, span);
  const auto above = bind("tq_above_max", "tile.cmps", {scaled, max_value}, {{"cmp_type", kCompareGreater}});
  const auto below = bind("tq_below_min", "tile.cmps", {scaled, min_value}, {{"cmp_type", kCompareLess}});
  const auto capped = select("tq_capped", above, max_value, scaled);
  const auto normalized = select("tq_normalized", below, min_value, capped);
  auto oriented = bind("tq_scaled_mk", "tile.reshape", {normalized, MxShape(rows, cols, span)});
  if (group_axis == 0) {
    // Orient the normalized values before narrowing. Plain TTRANS uses
    // FP32 here, so it needs no reinterpretation of the E5M2 payload.
    oriented = bind("tq_scaled_kn", "tile.transpose", {oriented, MxDimension(0, span), MxDimension(1, span)});
  }
  const auto payload = cast("tq_quant", oriented, DataType::FP8E5M2);
  const auto scale = PackMxExponent(exp_bytes, rows, group_cols, 1, group_axis == 0, b, span);
  return std::make_shared<MakeTuple>(std::vector<ExprPtr>{payload, scale}, span);
}

}  // namespace

// E4M3FN and FP4 lower to grouped TQUANT. E5M2 uses the same scale layouts
// but computes its shared exponent separately before a native E5M2 cast.
ExprPtr LowerTileTQuantMxRule(const CallPtr& call, const std::vector<ExprPtr>& args, LoweringBuilder& b) {
  const auto& span = call->span_;
  auto& reg = OpRegistry::GetInstance();
  auto src = args[0];
  const int group_axis = call->GetKwarg<int>("group_axis");
  INTERNAL_CHECK_SPAN(group_axis == 0 || group_axis == 1, span)
      << "Internal error: tile.tquant_mx group_axis must be 0 or 1";
  const DataType dtype = call->GetKwarg<DataType>("dtype", DataType::FP8E4M3FN);
  if (dtype == DataType::FP8E5M2) return LowerE5M2Quantization(src, group_axis, b, span);
  const bool packed_b = group_axis == 0;

  if (packed_b) {
    auto axis0 = std::make_shared<ConstInt>(0, DataType::INDEX, span);
    auto axis1 = std::make_shared<ConstInt>(1, DataType::INDEX, span);
    src = b.Bind("tq_src_kn", reg.Create("tile.transpose", {src, axis0, axis1}, {}, span), span);
  }

  auto src_tile = As<TileType>(src->GetType());
  INTERNAL_CHECK_SPAN(src_tile && src_tile->shape_.size() == 2, span)
      << "Internal error: tile.tquant_mx lowering requires a 2D source tile";
  auto rows_const = As<ConstInt>(src_tile->shape_[0]);
  auto cols_const = As<ConstInt>(src_tile->shape_[1]);
  // PTOAS special requirement: TQUANT/X2ZZ need static physical+valid extents so
  // EmitStaticValidTileView can emit a concrete treshape result type (not v_*=?).
  INTERNAL_CHECK_SPAN(rows_const && cols_const, span)
      << "Internal error: tile.tquant_mx lowering requires static source shapes";
  const int64_t rows = rows_const->value_;
  const int64_t cols = cols_const->value_;
  // Positive dims / group-axis divisibility already enforced by DeduceTileTQuantMxType;
  // re-check before `/ 32` so a broken invariant cannot silently truncate.
  INTERNAL_CHECK_SPAN(rows > 0 && cols > 0, span)
      << "Internal error: tile.tquant_mx lowering requires positive source dimensions";
  INTERNAL_CHECK_SPAN((group_axis == 1 && cols % 32 == 0) || (group_axis == 0 && rows % 32 == 0), span)
      << "Internal error: tile.tquant_mx source is not divisible by its group axis";
  const int64_t group_rows = group_axis == 0 ? rows / 32 : rows;
  const int64_t group_cols = group_axis == 0 ? cols : cols / 32;
  INTERNAL_CHECK_SPAN(group_rows <= std::numeric_limits<int64_t>::max() / group_cols, span)
      << "Internal error: tile.tquant_mx scale-group count overflows int64";
  const int64_t groups = group_rows * group_cols;

  auto make_dim = [&](int64_t value) { return std::make_shared<ConstInt>(value, DataType::INDEX, span); };
  auto make_shape = [&](int64_t dim0, int64_t dim1) {
    return std::make_shared<MakeTuple>(std::vector<ExprPtr>{make_dim(dim0), make_dim(dim1)}, span);
  };
  auto bind_typed_create = [&](const std::string& name, int64_t physical_rows, int64_t physical_cols,
                               int64_t valid_rows, int64_t valid_cols, DataType dtype, TileLayout slayout,
                               int64_t fractal = 512, TileLayout blayout = TileLayout::row_major) {
    auto shape = make_shape(physical_rows, physical_cols);
    std::vector<std::pair<std::string, std::any>> create_kwargs = {{"dtype", dtype},
                                                                   {"target_memory", MemorySpace::Vec}};
    auto created = As<Call>(reg.Create("tile.create", {shape}, create_kwargs, span));
    INTERNAL_CHECK_SPAN(created, span) << "Internal error: tile.create did not produce a Call";
    TileView view;
    view.valid_shape = {make_dim(valid_rows), make_dim(valid_cols)};
    view.blayout = blayout;
    view.slayout = slayout;
    view.fractal = fractal;
    auto type =
        std::make_shared<TileType>(std::vector<ExprPtr>{make_dim(physical_rows), make_dim(physical_cols)},
                                   dtype, std::nullopt, view, MemorySpace::Vec);
    auto typed_create =
        std::make_shared<Call>(created->op_, created->args_, created->kwargs_, created->attrs_, type, span);
    return b.Bind(name, typed_create, span);
  };

  // Axis1 uses the legacy-flat exponent branch so FP32, FP16, and BF16 all
  // remain supported by the pinned PTO-ISA. Axis0 uses canonical [M/32,N].
  const int64_t aux_rows = group_axis == 0 ? group_rows : 1;
  const int64_t aux_cols = group_axis == 0 ? group_cols : groups;
  auto max_tile = bind_typed_create("tq_max", aux_rows, aux_cols, aux_rows, aux_cols, src_tile->dtype_,
                                    TileLayout::none_box);

  int64_t scaling_physical_cols = aux_cols;
  if (group_axis == 1) {
    // PTOAS grouped-TQUANT scratch sizing and FP32 unroll thresholds.
    constexpr int64_t kFp32ScaleAlignment = 64;
    constexpr int64_t kFp16ScaleAlignment = 128;
    constexpr int64_t kFp32UnrollMinElements = 1024;
    constexpr int64_t kFp32UnrollMultiple = 256;
    const int64_t align = src_tile->dtype_ == DataType::FP32 ? kFp32ScaleAlignment : kFp16ScaleAlignment;
    int64_t scale_elements = groups;
    INTERNAL_CHECK_SPAN(rows <= std::numeric_limits<int64_t>::max() / cols, span)
        << "Internal error: tile.tquant_mx source element count overflows int64";
    const int64_t source_elements = rows * cols;
    const bool unroll = src_tile->dtype_ == DataType::FP32 && source_elements > kFp32UnrollMinElements &&
                        source_elements % kFp32UnrollMultiple == 0;
    INTERNAL_CHECK_SPAN(!unroll || scale_elements <= std::numeric_limits<int64_t>::max() / 2, span)
        << "Internal error: tile.tquant_mx unrolled scale scratch size overflows int64";
    if (unroll) scale_elements *= 2;
    INTERNAL_CHECK_SPAN(scale_elements <= std::numeric_limits<int64_t>::max() - (align - 1), span)
        << "Internal error: tile.tquant_mx aligned scale scratch size overflows int64";
    scaling_physical_cols = (scale_elements + align - 1) / align * align;
  }
  auto scaling_tile = bind_typed_create("tq_scaling", aux_rows, scaling_physical_cols, aux_rows, aux_cols,
                                        src_tile->dtype_, TileLayout::none_box);

  auto public_types = As<TupleType>(call->GetType());
  INTERNAL_CHECK_SPAN(public_types && public_types->types_.size() == 2, span)
      << "Internal error: tile.tquant_mx must return exactly two tile types";
  auto public_dst_type = As<TileType>(public_types->types_[0]);
  INTERNAL_CHECK_SPAN(public_dst_type && (public_dst_type->dtype_ == DataType::FP8E4M3FN ||
                                          public_dst_type->dtype_ == DataType::FP4E2M1X2),
                      span)
      << "Internal error: tile.tquant_mx public destination has an unsupported MX dtype";

  // Value-returning TQUANT (gather_compare-style): Bind the TupleType result,
  // then project dst/exp so InitMemRef and PTO output preparation see real
  // TupleGetItem consumers. max/scaling remain write-only workspace inputs.
  const std::string algorithm = call->GetKwarg<std::string>("quant_scale_alg", "ocp");
  std::vector<std::pair<std::string, std::any>> quant_kwargs = {{"dtype", dtype}, {"group_axis", group_axis}};
  if (algorithm != "ocp") quant_kwargs.emplace_back("quant_scale_alg", algorithm);
  auto raw_tuple = b.Bind(
      "tq_raw", reg.Create("tile.tquant_mx_raw", {src, max_tile, scaling_tile}, quant_kwargs, span), span);
  auto raw_dst = b.Bind("tq_dst", std::make_shared<TupleGetItemExpr>(raw_tuple, 0, span), span);
  auto raw_exp = b.Bind("tq_exp", std::make_shared<TupleGetItemExpr>(raw_tuple, 1, span), span);

  // MXFP8 uses PTOAS's raw INT8 destination and exposes a zero-copy FP8 alias.
  // MXFP4 already has the native packed destination type and carrier shape;
  // no byte reinterpretation or second packing step is needed.
  ExprPtr dst_tile = raw_dst;
  if (dtype == DataType::FP8E4M3FN) {
    dst_tile =
        b.Bind("tq_quant", reg.Create("tile.reinterpret_view", {raw_dst}, {{"dtype", dtype}}, span), span);
  }

  auto exp_tile = PackMxExponent(raw_exp, group_rows, group_cols, group_axis, packed_b, b, span);
  return std::make_shared<MakeTuple>(std::vector<ExprPtr>{dst_tile, exp_tile}, span);
}

}  // namespace lower_composite
}  // namespace ir
}  // namespace pypto
