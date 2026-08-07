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
 * @file quant.cpp
 * @brief Quantization tile ops: tquant_mx and tdequant
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
#include "pypto/ir/type_inference.h"

namespace pypto {
namespace ir {

namespace {

void ValidateMxQuantMode(const std::string& mode, const std::string& op_name) {
  if (mode == "mxfp8_e4m3" || mode == "mxfp8") {
    return;
  }
  CHECK(false) << "The operator " << op_name << " got an unknown mode '" << mode
               << "'; expected one of {mxfp8_e4m3, mxfp8}";
}

int64_t GetStaticElementCount(const TileTypePtr& type, const std::string& operand,
                              const std::string& op_name) {
  const TileView view = tile_view_semantics::GetEffectiveTileView(*type);
  int64_t count = 1;
  for (const auto& dim_expr : view.valid_shape) {
    auto dim = As<ConstInt>(dim_expr);
    CHECK(dim && dim->value_ > 0) << "The operator " << op_name << " requires " << operand
                                  << " to have a static positive valid_shape";
    CHECK(count <= std::numeric_limits<int64_t>::max() / dim->value_)
        << "The operator " << op_name << " " << operand << " valid element count overflows int64";
    count *= dim->value_;
  }
  return count;
}

TypePtr DeduceTileTQuantMxType(const std::vector<ExprPtr>& args,
                               const std::vector<std::pair<std::string, std::any>>& kwargs,
                               const std::string& op_name) {
  // ``tile.tquant_mx`` (1 arg, DSL form) and ``tile.tquant_mx_dps`` (src, max,
  // scaling, dst, exp) share this deduction. The lower_composite pass
  // materializes every PTOAS output as an IR tile so the memory planner can
  // assign non-overlapping addresses. ``mode`` selects the ptoas quant_type.
  const bool is_dps = op_name == "tile.tquant_mx_dps";
  INTERNAL_CHECK((!is_dps && args.size() == 1) || (is_dps && args.size() == 5))
      << "Internal error: " << op_name << " got invalid argument count " << args.size();
  auto src_type = As<TileType>(args[0]->GetType());
  CHECK(src_type) << "The operator " << op_name << " requires src to be a TileType, but got "
                  << args[0]->GetType()->TypeName();
  CHECK(src_type->shape_.size() == 2) << "The operator " << op_name << " requires 2D src tile";
  CHECK(src_type->dtype_ == DataType::FP16 || src_type->dtype_ == DataType::FP32 ||
        src_type->dtype_ == DataType::BF16)
      << "The operator " << op_name << " requires src dtype in {FP16, FP32, BF16}, but got "
      << src_type->dtype_.ToString();
  const TileView src_view = tile_view_semantics::GetEffectiveTileView(*src_type);
  CHECK(tile_view_semantics::ShapeExprListsEquivalent(src_view.valid_shape, src_type->shape_))
      << "The operator " << op_name
      << " does not support a partial src valid_shape; valid_shape must match the physical shape";

  std::string mode = "mxfp8_e4m3";
  std::optional<TensorLayout> pack_layout;
  for (const auto& [key, value] : kwargs) {
    if (key == "mode") {
      mode = AnyCast<std::string>(value, "kwarg key: mode");
    } else if (key == "layout") {
      pack_layout = AnyCast<TensorLayout>(value, "kwarg key: layout");
      CHECK(*pack_layout == TensorLayout::MX_A_ZZ || *pack_layout == TensorLayout::MX_B_NN)
          << "The operator " << op_name
          << " layout must be MX_A_ZZ or MX_B_NN (ND/None are not allowed), got "
          << TensorLayoutToString(*pack_layout);
    }
  }

  TileTypePtr max_type;
  TileTypePtr scaling_type;
  TileTypePtr raw_dst_type;
  TileTypePtr raw_exp_type;
  if (is_dps) {
    // ptoas TQuantMxOp requires max/scaling element type to match src.
    max_type = As<TileType>(args[1]->GetType());
    scaling_type = As<TileType>(args[2]->GetType());
    CHECK(max_type && max_type->dtype_ == src_type->dtype_)
        << "The operator " << op_name << " requires max scratch dtype to match src ("
        << src_type->dtype_.ToString() << "), but got "
        << (max_type ? max_type->dtype_.ToString() : std::string("<non-tile>"));
    CHECK(scaling_type && scaling_type->dtype_ == src_type->dtype_)
        << "The operator " << op_name << " requires scaling scratch dtype to match src ("
        << src_type->dtype_.ToString() << "), but got "
        << (scaling_type ? scaling_type->dtype_.ToString() : std::string("<non-tile>"));
    raw_dst_type = As<TileType>(args[3]->GetType());
    raw_exp_type = As<TileType>(args[4]->GetType());
    CHECK(raw_dst_type && raw_dst_type->dtype_ == DataType::INT8)
        << "The operator " << op_name << " requires dst dtype INT8";
    CHECK(raw_exp_type && raw_exp_type->dtype_ == DataType::UINT8)
        << "The operator " << op_name << " requires exp dtype UINT8";
    CHECK(tile_view_semantics::ShapeExprListsEquivalent(raw_dst_type->shape_, src_type->shape_))
        << "The operator " << op_name << " requires dst shape to match src";
  }

  // Validate the mode and expose its semantic result types at the public IR
  // boundary. LowerCompositeOps supplies the raw INT8/UINT8 destinations that
  // PTOAS requires and aliases them as these FP8 types.
  ValidateMxQuantMode(mode, op_name);

  ExprPtr m_dim = src_type->shape_[0];
  ExprPtr k_dim = src_type->shape_[1];
  auto m_const = As<ConstInt>(m_dim);
  auto k_const = As<ConstInt>(k_dim);
  CHECK(m_const && k_const) << "The operator " << op_name << " requires static M and K dimensions";
  CHECK(m_const->value_ > 0) << "The operator " << op_name << " requires positive M, but got "
                             << m_const->value_;
  CHECK(m_const->value_ % 16 == 0) << "The operator " << op_name << " requires M divisible by 16, but got "
                                   << m_const->value_;
  CHECK(k_const->value_ > 0 && k_const->value_ % 32 == 0)
      << "The operator " << op_name << " requires K divisible by 32, but got " << k_const->value_;
  if (pack_layout.has_value()) {
    CHECK(k_const->value_ % 64 == 0) << "The operator " << op_name
                                     << " with layout=" << TensorLayoutToString(*pack_layout)
                                     << " requires K divisible by 64, but got " << k_const->value_;
  }
  constexpr int64_t kMaxTileElements = 59461;
  CHECK(m_const->value_ <= kMaxTileElements / k_const->value_)
      << "The operator " << op_name << " requires M*K <= " << kMaxTileElements << ", but got "
      << m_const->value_ << "*" << k_const->value_;

  DataType dst_dtype = DataType::FP8E4M3FN;
  // Public packed forms: MX_A_ZZ keeps [M,K]; MX_B_NN returns transposed [K,N]
  // (ExpandMxPackedQuant inserts the INT8 transpose). Flat (no layout) keeps src shape.
  std::vector<ExprPtr> dst_shape = src_type->shape_;
  if (pack_layout == TensorLayout::MX_B_NN) {
    dst_shape = {k_dim, m_dim};  // src is [N,K] → quant [K,N]
  }
  TileView dst_view;
  dst_view.valid_shape = dst_shape;
  InheritTileViewLayout(dst_view, src_type);
  auto dst_type = std::make_shared<TileType>(dst_shape, dst_dtype, std::nullopt, dst_view);

  // E8M0 scale as flat [1, groups].
  // For MX_A_ZZ: groups = M*K/32; for MX_B_NN (src [N,K]): groups = N*K/32.
  const int64_t k_groups = k_const->value_ / 32;
  CHECK(m_const->value_ <= std::numeric_limits<int64_t>::max() / k_groups)
      << "The operator " << op_name << " scale-group count overflows int64";
  auto groups_dim = std::make_shared<ConstInt>(m_const->value_ * k_groups, DataType::INDEX, Span::unknown());
  auto one = std::make_shared<ConstInt>(1, DataType::INDEX, Span::unknown());

  if (is_dps) {
    const int64_t groups = m_const->value_ * k_groups;
    CHECK(GetStaticElementCount(max_type, "max scratch", op_name) == groups)
        << "The operator " << op_name << " requires max scratch valid element count " << groups;
    CHECK(GetStaticElementCount(scaling_type, "scaling scratch", op_name) == groups)
        << "The operator " << op_name << " requires scaling scratch valid element count " << groups;
    CHECK(GetStaticElementCount(raw_exp_type, "exp destination", op_name) == groups)
        << "The operator " << op_name << " requires exp destination valid element count " << groups;
    const TileView dst_view = tile_view_semantics::GetEffectiveTileView(*raw_dst_type);
    CHECK(tile_view_semantics::ShapeExprListsEquivalent(dst_view.valid_shape, src_type->shape_))
        << "The operator " << op_name << " requires dst valid_shape to match src";
    return GetUnknownType();
  }

  TileView scale_view;
  scale_view.valid_shape = {one, groups_dim};
  scale_view.blayout = TileLayout::row_major;
  scale_view.slayout = TileLayout::none_box;
  scale_view.fractal = tile_view_semantics::kMXScaleFractal;
  auto scale_type = std::make_shared<TileType>(std::vector<ExprPtr>{one, groups_dim}, DataType::FP8E8M0,
                                               std::nullopt, scale_view);

  std::vector<TypePtr> elements{dst_type, scale_type};
  return std::make_shared<TupleType>(std::move(elements));
}

TypePtr DeduceTileTDequantType(const std::vector<ExprPtr>& args,
                               const std::vector<std::pair<std::string, std::any>>& kwargs,
                               const std::string& op_name) {
  CHECK(args.size() == 3) << "The operator " << op_name
                          << " requires exactly 3 arguments (src, scale, offset), but got " << args.size();
  auto src_type = As<TileType>(args[0]->GetType());
  auto scale_type = As<TileType>(args[1]->GetType());
  auto offset_type = As<TileType>(args[2]->GetType());
  CHECK(src_type) << "The operator " << op_name << " requires src to be a TileType";
  CHECK(scale_type) << "The operator " << op_name << " requires scale to be a TileType";
  CHECK(offset_type) << "The operator " << op_name << " requires offset to be a TileType";
  CHECK(src_type->dtype_ == DataType::INT8 || src_type->dtype_ == DataType::INT16)
      << "The operator " << op_name << " requires src dtype INT8 or INT16, but got "
      << src_type->dtype_.ToString();
  CHECK(scale_type->dtype_ == DataType::FP32)
      << "The operator " << op_name << " requires scale dtype FP32, but got "
      << scale_type->dtype_.ToString();
  CHECK(offset_type->dtype_ == DataType::FP32)
      << "The operator " << op_name << " requires offset dtype FP32, but got "
      << offset_type->dtype_.ToString();
  CHECK(src_type->shape_.size() == 2) << "The operator " << op_name << " requires 2D src";
  const TileView src_view = tile_view_semantics::GetEffectiveTileView(*src_type);
  auto check_per_row_param = [&](const TileTypePtr& type, const char* name) {
    const TileView param_view = tile_view_semantics::GetEffectiveTileView(*type);
    CHECK(param_view.valid_shape.size() == 2)
        << "The operator " << op_name << " requires " << name << " to be a 2D [rows, 1] tile";
    CHECK(AreExprsEqual(param_view.valid_shape[0], src_view.valid_shape[0]))
        << "The operator " << op_name << " requires " << name << " row count to match src row count";
    auto columns = As<ConstInt>(param_view.valid_shape[1]);
    CHECK(columns && columns->value_ == 1)
        << "The operator " << op_name << " requires " << name << " shape [rows, 1]";
  };
  check_per_row_param(scale_type, "scale");
  check_per_row_param(offset_type, "offset");
  CHECK(tile_view_semantics::ShapeExprListsEquivalent(scale_type->shape_, offset_type->shape_))
      << "The operator " << op_name << " requires scale and offset to have the same physical shape";

  TileView tile_view = tile_view_semantics::GetImplicitTileView(src_type->shape_, MemorySpace::Vec);
  tile_view.valid_shape = src_view.valid_shape;
  tile_view.blayout = TileLayout::row_major;
  return std::make_shared<TileType>(src_type->shape_, DataType::FP32, std::nullopt, tile_view);
}

}  // namespace

REGISTER_OP("tile.tquant_mx")
    .set_op_category("TileOp")
    .set_description(
        "MX block-32 dynamic quantization: TupleType{quantized (FP8E4M3FN), e8m0_scale "
        "(FP8E8M0)}. The lower_composite pass rewrites the flat one-source form into "
        "tile.tquant_mx_dps, materializing source-dtype scratch (max, scaling) as IR-level tiles so the "
        "memory planner can address them; codegen then emits pto.tquant.mx. mode selects "
        "mxfp8_e4m3 (alias: mxfp8). The FP8 value is stored as its byte representation (mirrors "
        "pto-isa's int8_t FP8 / uint8_t E8M0 tiles); the tstore byte-copies into the FP8 outputs.")
    .add_argument("src", "Source tile (FP16/FP32/BF16, 2D)")
    .set_attr<std::string>("mode")
    .set_attr<TensorLayout>("layout")
    .set_input_memory(0, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .not_inplace_safe()
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileTQuantMxType(args, kwargs, "tile.tquant_mx");
    });

// Internal DPS form produced by the tile.tquant_mx lowering rule. Carries the two
// source-dtype scratch tiles plus both public outputs as explicit operands so
// the memory planner assigns non-overlapping addresses (codegen-internal scratch cannot get an address at
// --pto-level=level3). It is emitted as an EvalStmt and produces UnknownType;
// codegen lowers it to the ptoas pto.tquant.mx instruction.
REGISTER_OP("tile.tquant_mx_dps")
    .set_op_category("TileOp")
    .set_description(
        "Internal DPS form of tile.tquant_mx with explicit source-dtype scratch operands. Lowers to "
        "pto.tquant.mx: TQuant(dst, src, exp, max, scaling). dst=raw i8; exp=raw ui8; "
        "max/scaling match src dtype.")
    .add_argument("src", "Source tile (FP16/FP32/BF16, 2D)")
    .add_argument("max", "Per-group max scratch tile matching src dtype (write-only)")
    .add_argument("scaling", "Per-group scaling scratch tile matching src dtype (write-only)")
    .add_argument("dst", "Quantized destination tile (raw INT8 bytes, write-only)")
    .add_argument("exp", "E8M0 exponent destination tile (raw UINT8 bytes, write-only)")
    .set_attr<std::string>("mode")
    .set_input_memory(0, MemorySpace::Vec)
    .set_input_memory(1, MemorySpace::Vec)
    .set_input_memory(2, MemorySpace::Vec)
    .set_input_memory(3, MemorySpace::Vec)
    .set_input_memory(4, MemorySpace::Vec)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileTQuantMxType(args, kwargs, "tile.tquant_mx_dps");
    });

REGISTER_OP("tile.tdequant")
    .set_op_category("TileOp")
    .set_description("Dequantize integer tile with per-row scale/offset: dst = (src - offset) * scale")
    .add_argument("src", "Quantized source tile (INT8/INT16, 2D)")
    .add_argument("scale", "Per-row scale tile")
    .add_argument("offset", "Per-row offset tile")
    .set_input_memory(0, MemorySpace::Vec)
    .set_input_memory(1, MemorySpace::Vec)
    .set_input_memory(2, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .not_inplace_safe()
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileTDequantType(args, kwargs, "tile.tdequant");
    });

}  // namespace ir
}  // namespace pypto
