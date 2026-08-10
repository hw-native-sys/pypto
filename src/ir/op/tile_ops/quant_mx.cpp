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

struct MxQuantTypeInfo {
  DataType public_dst_dtype;
  DataType dps_dst_dtype;
  bool is_fp4;
};

MxQuantTypeInfo ResolveMxQuantType(DataType dtype, const std::string& op_name) {
  if (dtype == DataType::FP8E4M3FN) {
    return {DataType::FP8E4M3FN, DataType::INT8, false};
  }
  if (dtype == DataType::FP4) {
    return {DataType::FP4, DataType::FP4, true};
  }
  CHECK(false) << "The operator " << op_name << " requires dtype FP8E4M3FN or FP4, but got "
               << dtype.ToString();
  return {DataType::FP8E4M3FN, DataType::INT8, false};
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

int64_t GetStaticPhysicalElementCount(const TileTypePtr& type, const std::string& operand,
                                      const std::string& op_name) {
  int64_t count = 1;
  for (const auto& dim_expr : type->shape_) {
    auto dim = As<ConstInt>(dim_expr);
    CHECK(dim && dim->value_ > 0) << "The operator " << op_name << " requires " << operand
                                  << " to have a static positive physical shape";
    CHECK(count <= std::numeric_limits<int64_t>::max() / dim->value_)
        << "The operator " << op_name << " " << operand << " physical element count overflows int64";
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
  // assign non-overlapping addresses. ``dtype`` selects the PTOAS quant type.
  const bool is_dps = op_name == "tile.tquant_mx_dps";
  INTERNAL_CHECK((!is_dps && args.size() == 1) || (is_dps && args.size() == 5))
      << "Internal error: " << op_name << " got invalid argument count " << args.size();
  auto src_type = As<TileType>(args[0]->GetType());
  CHECK(src_type) << "The operator " << op_name << " requires src to be a TileType, but got "
                  << args[0]->GetType()->TypeName();
  CHECK(src_type->shape_.size() == 2) << "The operator " << op_name << " requires 2D src tile";
  const TileView src_view = tile_view_semantics::GetEffectiveTileView(*src_type);
  CHECK(tile_view_semantics::ShapeExprListsEquivalent(src_view.valid_shape, src_type->shape_))
      << "The operator " << op_name
      << " does not support a partial src valid_shape; valid_shape must match the physical shape";

  DataType dtype = DataType::FP8E4M3FN;
  std::optional<TensorLayout> pack_layout;
  int group_axis = 1;
  for (const auto& [key, value] : kwargs) {
    if (key == "dtype") {
      dtype = AnyCast<DataType>(value, "kwarg key: dtype");
    } else if (key == "layout") {
      pack_layout = AnyCast<TensorLayout>(value, "kwarg key: layout");
      CHECK(*pack_layout == TensorLayout::MX_A_ZZ || *pack_layout == TensorLayout::MX_B_NN)
          << "The operator " << op_name
          << " layout must be MX_A_ZZ or MX_B_NN (ND/None are not allowed), got "
          << TensorLayoutToString(*pack_layout);
    } else if (key == "group_axis") {
      group_axis = AnyCast<int>(value, "kwarg key: group_axis");
      CHECK(group_axis == 0 || group_axis == 1)
          << "The operator " << op_name << " group_axis must be 0 or 1, but got " << group_axis;
    }
  }
  CHECK(is_dps || pack_layout.has_value()) << "The operator " << op_name << " requires layout";
  const MxQuantTypeInfo type_info = ResolveMxQuantType(dtype, op_name);
  const bool src_supported = src_type->dtype_ == DataType::FP16 || src_type->dtype_ == DataType::BF16 ||
                             (!type_info.is_fp4 && src_type->dtype_ == DataType::FP32);
  CHECK(src_supported) << "The operator " << op_name << " with dtype=" << dtype.ToString()
                       << " requires src dtype in "
                       << (type_info.is_fp4 ? "{FP16, BF16}" : "{FP16, FP32, BF16}") << ", but got "
                       << src_type->dtype_.ToString();

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
    CHECK(raw_dst_type && raw_dst_type->dtype_ == type_info.dps_dst_dtype)
        << "The operator " << op_name << " with dtype=" << dtype.ToString() << " requires dst dtype "
        << type_info.dps_dst_dtype.ToString();
    CHECK(raw_exp_type && raw_exp_type->dtype_ == DataType::UINT8)
        << "The operator " << op_name << " requires exp dtype UINT8";
    CHECK(tile_view_semantics::ShapeExprListsEquivalent(raw_dst_type->shape_, src_type->shape_))
        << "The operator " << op_name << " requires dst shape to match src";
  }

  ExprPtr dim0 = src_type->shape_[0];
  ExprPtr dim1 = src_type->shape_[1];
  auto dim0_const = As<ConstInt>(dim0);
  auto dim1_const = As<ConstInt>(dim1);
  CHECK(dim0_const && dim1_const) << "The operator " << op_name << " requires static "
                                  << (pack_layout == TensorLayout::MX_B_NN ? "N and K" : "M and K");
  CHECK(dim0_const->value_ > 0 && dim1_const->value_ > 0)
      << "The operator " << op_name << " requires positive dimensions";
  if (!is_dps || group_axis == 1) {
    CHECK(dim0_const->value_ % 16 == 0)
        << "The operator " << op_name << " requires " << (pack_layout == TensorLayout::MX_B_NN ? "N" : "M")
        << " divisible by 16, but got " << dim0_const->value_;
    CHECK(dim1_const->value_ % 32 == 0)
        << "The operator " << op_name << " requires K divisible by 32 for group_axis=1, but got "
        << dim1_const->value_;
  } else {
    CHECK(dim0_const->value_ % 32 == 0)
        << "The operator " << op_name << " requires dim0 divisible by 32 for group_axis=0, but got "
        << dim0_const->value_;
  }
  if (pack_layout.has_value()) {
    CHECK(dim1_const->value_ % 64 == 0)
        << "The operator " << op_name << " with layout=" << TensorLayoutToString(*pack_layout)
        << " requires K divisible by 64, but got " << dim1_const->value_;
    if (*pack_layout == TensorLayout::MX_B_NN) {
      CHECK(dim0_const->value_ % 32 == 0)
          << "The operator " << op_name
          << " with layout=MX_B_NN requires N divisible by 32 for A5 Vec row-byte alignment, but got "
          << dim0_const->value_;
      if (type_info.is_fp4 && src_type->dtype_ == DataType::FP16) {
        CHECK(dim0_const->value_ % 64 == 0)
            << "The operator " << op_name
            << " with layout=MX_B_NN and FP16 MXFP4 requires N divisible by 64 so the packed "
               "destination stride is 32-byte aligned, but got "
            << dim0_const->value_;
      }
    }
  }
  if (is_dps && type_info.is_fp4 && src_type->dtype_ == DataType::FP16 && group_axis == 0) {
    CHECK(dim1_const->value_ % 64 == 0)
        << "The operator " << op_name
        << " with FP16 MXFP4 group_axis=0 requires dim1 divisible by 64 so the packed destination "
           "stride is 32-byte aligned, but got "
        << dim1_const->value_;
  }
  DataType dst_dtype = type_info.public_dst_dtype;
  // MX_A_ZZ keeps [M,K]; MX_B_NN returns transposed [K,N]
  // (LowerCompositeOps inserts the transpose before axis0 quantization).
  std::vector<ExprPtr> dst_shape = src_type->shape_;
  if (pack_layout == TensorLayout::MX_B_NN) {
    dst_shape = {dim1, dim0};  // src is [N,K] → quant [K,N]
  }
  TileView dst_view;
  dst_view.valid_shape = dst_shape;
  dst_view.blayout = TileLayout::row_major;
  dst_view.slayout = TileLayout::none_box;
  auto dst_type = std::make_shared<TileType>(dst_shape, dst_dtype, std::nullopt, dst_view);

  const int64_t grouped_rows = group_axis == 0 ? dim0_const->value_ / 32 : dim0_const->value_;
  const int64_t grouped_cols = group_axis == 0 ? dim1_const->value_ : dim1_const->value_ / 32;
  CHECK(grouped_rows <= std::numeric_limits<int64_t>::max() / grouped_cols)
      << "The operator " << op_name << " scale-group count overflows int64";
  const int64_t groups = grouped_rows * grouped_cols;

  if (is_dps) {
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
  std::vector<ExprPtr> scale_shape;
  if (pack_layout == TensorLayout::MX_A_ZZ) {
    scale_shape = {dim0,
                   std::make_shared<ConstInt>(dim1_const->value_ / 32, DataType::INDEX, Span::unknown())};
  } else if (pack_layout == TensorLayout::MX_B_NN) {
    scale_shape = {std::make_shared<ConstInt>(dim1_const->value_ / 32, DataType::INDEX, Span::unknown()),
                   dim0};
  }
  scale_view.valid_shape = scale_shape;
  const TileLayout scale_layout =
      pack_layout == TensorLayout::MX_B_NN ? TileLayout::col_major : TileLayout::row_major;
  scale_view.blayout = scale_layout;
  scale_view.slayout = scale_layout;
  scale_view.fractal = tile_view_semantics::kMXScaleFractal;
  auto scale_type = std::make_shared<TileType>(scale_shape, DataType::FP8E8M0, std::nullopt, scale_view);

  std::vector<TypePtr> elements{dst_type, scale_type};
  return std::make_shared<TupleType>(std::move(elements));
}

TypePtr DeduceTileTMovX2ZzType(const std::vector<ExprPtr>& args,
                               const std::vector<std::pair<std::string, std::any>>& kwargs) {
  constexpr const char* kOpName = "tile.tmov_x2zz_dps";
  CHECK(args.size() == 3) << "The operator " << kOpName << " requires src, tmp, and dst";
  auto src = As<TileType>(args[0]->GetType());
  auto tmp = As<TileType>(args[1]->GetType());
  auto dst = As<TileType>(args[2]->GetType());
  CHECK(src && tmp && dst) << "The operator " << kOpName << " requires TileType operands";
  CHECK(src->dtype_ == DataType::UINT8 && tmp->dtype_ == DataType::UINT8 && dst->dtype_ == DataType::UINT8)
      << "The operator " << kOpName << " requires raw UINT8 src/tmp/dst";
  CHECK(src->shape_.size() == 2 && tmp->shape_.size() == 2 && dst->shape_.size() == 2)
      << "The operator " << kOpName << " requires rank-2 tiles";

  int group_axis = 1;
  for (const auto& [key, value] : kwargs) {
    if (key == "group_axis") group_axis = AnyCast<int>(value, "kwarg key: group_axis");
  }
  CHECK(group_axis == 0 || group_axis == 1)
      << "The operator " << kOpName << " group_axis must be 0 or 1, but got " << group_axis;

  const TileView src_view = tile_view_semantics::GetEffectiveTileView(*src);
  const TileView dst_view = tile_view_semantics::GetEffectiveTileView(*dst);
  const TileView tmp_view = tile_view_semantics::GetEffectiveTileView(*tmp);
  CHECK(src_view.blayout == TileLayout::row_major && src_view.slayout == TileLayout::none_box)
      << "The operator " << kOpName << " requires src row_major/none_box";
  CHECK(dst_view.blayout == TileLayout::row_major && dst_view.slayout == TileLayout::row_major)
      << "The operator " << kOpName << " requires dst row_major/row_major (ZZ)";
  CHECK(tmp_view.blayout == TileLayout::row_major && tmp_view.slayout == TileLayout::none_box)
      << "The operator " << kOpName << " requires tmp row_major/none_box";
  CHECK(GetStaticElementCount(src, "src", kOpName) == GetStaticElementCount(dst, "dst", kOpName))
      << "The operator " << kOpName << " requires src and dst to hold the same exponent count";

  auto src_rows = As<ConstInt>(src_view.valid_shape[0]);
  auto src_cols = As<ConstInt>(src_view.valid_shape[1]);
  auto dst_rows = As<ConstInt>(dst_view.valid_shape[0]);
  auto dst_cols = As<ConstInt>(dst_view.valid_shape[1]);
  CHECK(src_rows && src_cols && dst_rows && dst_cols)
      << "The operator " << kOpName << " requires static valid shapes";
  if (group_axis == 1) {
    CHECK(dst_cols->value_ % 2 == 0) << "The operator " << kOpName
                                     << " requires axis1 grouped exponent columns to be even";
    CHECK(dst_rows->value_ <= std::numeric_limits<int64_t>::max() - 15)
        << "The operator " << kOpName << " padded row count overflows int64";
    const int64_t row_blocks = (dst_rows->value_ + 15) / 16;
    CHECK(row_blocks <= std::numeric_limits<int64_t>::max() / 16)
        << "The operator " << kOpName << " padded row count overflows int64";
    const int64_t padded_rows = row_blocks * 16;
    CHECK(padded_rows <= std::numeric_limits<int64_t>::max() / dst_cols->value_)
        << "The operator " << kOpName << " padded capacity overflows int64";
    const int64_t padded_elements = padded_rows * dst_cols->value_;
    CHECK(GetStaticPhysicalElementCount(src, "src", kOpName) >= padded_elements &&
          GetStaticPhysicalElementCount(dst, "dst", kOpName) >= padded_elements)
        << "The operator " << kOpName << " requires src/dst capacity for align16(dst rows) * dst cols";
    CHECK(row_blocks <= (std::numeric_limits<int64_t>::max() - 64) / dst_cols->value_)
        << "The operator " << kOpName << " tmp capacity overflows int64";
    const int64_t tmp_bytes = 64 + row_blocks * dst_cols->value_;
    CHECK(GetStaticPhysicalElementCount(tmp, "tmp", kOpName) >= tmp_bytes)
        << "The operator " << kOpName << " requires tmp capacity of at least " << tmp_bytes << " bytes";
  } else {
    CHECK(src_rows->value_ >= 2 && src_rows->value_ % 2 == 0)
        << "The operator " << kOpName << " requires axis0 src rows to be an even count >= 2";
    CHECK(src_cols->value_ % 16 == 0)
        << "The operator " << kOpName << " requires axis0 src columns divisible by 16";
    auto src_physical_cols = As<ConstInt>(src->shape_[1]);
    CHECK(src_physical_cols && src_physical_cols->value_ == src_cols->value_)
        << "The operator " << kOpName << " requires axis0 source physical stride to be tight";
  }
  return GetUnknownType();
}

}  // namespace

REGISTER_OP("tile.tquant_mx")
    .set_op_category("TileOp")
    .set_description(
        "MX block-32 dynamic quantization: TupleType{quantized (FP8E4M3FN or packed FP4 E2M1), e8m0_scale "
        "(FP8E8M0)}. The lower_composite pass rewrites the one-source form into "
        "tile.tquant_mx_dps, materializing source-dtype scratch (max, scaling) as IR-level tiles so the "
        "memory planner can address them; codegen then emits pto.tquant.mx. dtype selects "
        "FP8E4M3FN or FP4. MXFP8 uses a raw INT8 "
        "destination alias; MXFP4 writes a native packed FP4 destination.")
    .add_argument("src", "Source tile (FP16/FP32/BF16 for MXFP8; FP16/BF16 for MXFP4, 2D)")
    .set_attr<DataType>("dtype")
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
        "pto.tquant.mx: TQuant(dst, src, exp, max, scaling). dst=raw i8 for MXFP8 or native packed FP4 "
        "for MXFP4; exp=raw ui8; "
        "max/scaling match src dtype.")
    .add_argument("src", "Source tile (FP16/FP32/BF16 for MXFP8; FP16/BF16 for MXFP4, 2D)")
    .add_argument("max", "Per-group max scratch tile matching src dtype (write-only)")
    .add_argument("scaling", "Per-group scaling scratch tile matching src dtype (write-only)")
    .add_argument("dst", "Quantized destination tile (raw INT8 for MXFP8 or packed FP4, write-only)")
    .add_argument("exp", "E8M0 exponent destination tile (raw UINT8 bytes, write-only)")
    .set_attr<DataType>("dtype")
    .set_attr<int>("group_axis")
    .set_input_memory(0, MemorySpace::Vec)
    .set_input_memory(1, MemorySpace::Vec)
    .set_input_memory(2, MemorySpace::Vec)
    .set_input_memory(3, MemorySpace::Vec)
    .set_input_memory(4, MemorySpace::Vec)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileTQuantMxType(args, kwargs, "tile.tquant_mx_dps");
    });

REGISTER_OP("tile.tmov_x2zz_dps")
    .set_op_category("TileOp")
    .set_description(
        "Internal side-effecting exponent X-to-ZZ layout conversion. src/tmp/dst are explicit raw "
        "UINT8 Vec tiles; codegen emits the non-scaling third-operand form of pto.tmov.")
    .add_argument("src", "Canonical exponent source (raw UINT8)")
    .add_argument("tmp", "Explicit X-to-ZZ temporary tile (raw UINT8)")
    .add_argument("dst", "ZZ exponent destination (raw UINT8)")
    .set_attr<int>("group_axis")
    .set_input_memory(0, MemorySpace::Vec)
    .set_input_memory(1, MemorySpace::Vec)
    .set_input_memory(2, MemorySpace::Vec)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileTMovX2ZzType(args, kwargs);
    });

}  // namespace ir
}  // namespace pypto
