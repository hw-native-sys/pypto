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
#include <optional>
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
#include "pypto/ir/span.h"
#include "pypto/ir/type.h"
#include "src/ir/transforms/lower_composite/lower_composite_builder.h"
#include "src/ir/transforms/lower_composite/lower_composite_rules.h"

namespace pypto {
namespace ir {
namespace lower_composite {

// ============================================================================
// ``tile.tquant_mx`` lowering — grouped TQUANT + exponent X-to-ZZ.
//
// Public group_axis=1 is the A-side [M,K] path. group_axis=0 is the B-side
// [N,K] path: transpose to [K,N] first, then PTOAS axis0. Axis0 X-to-ZZ follows
// pto-isa TMovDnTo2Zz (pin be5ccb76): DN [M̂,N] -> ZZ [N,M̂] row/row, then a
// zero-copy tile.transpose_view yields the public [M̂,N] col/col scale. Same-
// InCore mix with matmul_mx is not supported yet; stage through GM between AIV
// and AIC (follow-up).
ExprPtr LowerTileTQuantMxRule(const CallPtr& call, const std::vector<ExprPtr>& args, LoweringBuilder& b) {
  const auto& span = call->span_;
  auto& reg = OpRegistry::GetInstance();
  auto src = args[0];
  const int group_axis = call->GetKwarg<int>("group_axis");
  INTERNAL_CHECK_SPAN(group_axis == 0 || group_axis == 1, span)
      << "Internal error: tile.tquant_mx group_axis must be 0 or 1";
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
  // then project dst/exp so InitMemRef + ResolveTupleResultElements see real
  // TupleGetItem consumers. max/scaling remain write-only workspace inputs.
  DataType dtype = call->GetKwarg<DataType>("dtype", DataType::FP8E4M3FN);
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

  // Axis1 X-to-ZZ tmp: 64 + ceil(rows/16)*cols bytes. Axis0 (TMovDnTo2Zz): ISA
  // still requires a Vec tmp operand; use one 32-byte Vec pad unit.
  constexpr int64_t kVecByteAlign = 32;
  int64_t tmp_bytes = kVecByteAlign;
  if (group_axis == 1) {
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
  auto x2zz_tmp = bind_typed_create("tq_x2zz_tmp", 1, tmp_physical_bytes, 1, tmp_physical_bytes,
                                    DataType::UINT8, TileLayout::none_box);
  // Value-returning X-to-ZZ: InitMemRef allocates ZZ dst from the deduced type.
  // Axis1 needs dst_rows/dst_cols because TQUANT exp is legacy-flat [1,M*G].
  std::vector<std::pair<std::string, std::any>> x2zz_kwargs = {{"group_axis", group_axis}};
  if (group_axis == 1) {
    x2zz_kwargs.emplace_back("dst_rows", static_cast<int>(group_rows));
    x2zz_kwargs.emplace_back("dst_cols", static_cast<int>(group_cols));
  }
  auto zz_exp =
      b.Bind("tq_exp_zz", reg.Create("tile.tmov_x2zz", {raw_exp, x2zz_tmp}, x2zz_kwargs, span), span);
  ExprPtr exp_tile = b.Bind(
      "tq_scale", reg.Create("tile.reinterpret_view", {zz_exp}, {{"dtype", DataType::FP8E8M0}}, span), span);
  if (group_axis == 0) {
    // ZZ [N,M̂] row/row <-> public MX_B [M̂,N] col/col over the same bytes.
    exp_tile = b.Bind("tq_scale_nn", reg.Create("tile.transpose_view", {exp_tile}, {}, span), span);
  }
  return std::make_shared<MakeTuple>(std::vector<ExprPtr>{dst_tile, exp_tile}, span);
}

}  // namespace lower_composite
}  // namespace ir
}  // namespace pypto
