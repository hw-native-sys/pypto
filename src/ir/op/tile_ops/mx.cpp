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
 * @file mx.cpp
 * @brief MX (block-scale) tile ops: tquant, tdequant, tget_scale_addr
 */

#include <any>
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
#include "pypto/ir/type.h"
#include "pypto/ir/type_inference.h"

namespace pypto {
namespace ir {

namespace {

DataType GetMxQuantOutDtype(const std::string& mode, const std::string& op_name) {
  if (mode == "mxfp8_e4m3" || mode == "mxfp8") return DataType::FP8E4M3FN;
  if (mode == "mxfp8_e5m2") return DataType::FP8E5M2;
  if (mode == "mxfp4" || mode == "mxfp4_e2m1") return DataType::FP4;
  CHECK(false) << "The operator " << op_name << " got an unknown mode '" << mode
               << "'; expected one of {mxfp8_e4m3, mxfp8, mxfp8_e5m2, mxfp4, mxfp4_e2m1}";
  return DataType::FP8E4M3FN;  // unreachable
}

TypePtr DeduceTileTQuantType(const std::vector<ExprPtr>& args,
                             const std::vector<std::pair<std::string, std::any>>& kwargs,
                             const std::string& op_name) {
  CHECK(args.size() == 1) << "The operator " << op_name << " requires exactly 1 argument (src), but got "
                          << args.size();
  auto src_type = As<TileType>(args[0]->GetType());
  CHECK(src_type) << "The operator " << op_name << " requires src to be a TileType, but got "
                  << args[0]->GetType()->TypeName();
  CHECK(src_type->shape_.size() == 2) << "The operator " << op_name << " requires 2D src tile";
  CHECK(src_type->dtype_ == DataType::FP16 || src_type->dtype_ == DataType::FP32 ||
        src_type->dtype_ == DataType::BF16)
      << "The operator " << op_name << " requires src dtype in {FP16, FP32, BF16}, but got "
      << src_type->dtype_.ToString();

  std::string mode = "mxfp8_e4m3";
  for (const auto& [key, value] : kwargs) {
    if (key == "mode") {
      mode = AnyCast<std::string>(value, "kwarg key: mode");
    }
  }
  DataType out_dtype = GetMxQuantOutDtype(mode, op_name);

  ExprPtr m_dim = src_type->shape_[0];
  ExprPtr k_dim = src_type->shape_[1];
  auto k_const = As<ConstInt>(k_dim);
  ExprPtr scale_k;
  if (k_const) {
    CHECK(k_const->value_ > 0 && k_const->value_ % 32 == 0)
        << "The operator " << op_name << " requires K divisible by 32, but got " << k_const->value_;
    scale_k = std::make_shared<ConstInt>(k_const->value_ / 32, DataType::INDEX, Span::unknown());
  } else {
    // Dynamic K: scale cols unknown at this point; use a placeholder shape [M, ?]
    // by reusing K (will be refined by later shape inference if needed).
    scale_k = k_dim;
  }

  TileView dst_view;
  dst_view.valid_shape = src_type->shape_;
  InheritTileViewLayout(dst_view, src_type);
  auto dst_type = std::make_shared<TileType>(src_type->shape_, out_dtype, std::nullopt, dst_view);

  TileView scale_view;
  scale_view.valid_shape = {m_dim, scale_k};
  scale_view.blayout = TileLayout::row_major;
  scale_view.slayout = TileLayout::row_major;
  scale_view.fractal = 32;
  auto scale_type = std::make_shared<TileType>(std::vector<ExprPtr>{m_dim, scale_k}, DataType::FP8E8M0,
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
  CHECK(scale_type->dtype_.IsFloat()) << "The operator " << op_name << " requires floating scale dtype";
  CHECK(offset_type->dtype_.IsFloat()) << "The operator " << op_name << " requires floating offset dtype";
  CHECK(src_type->shape_.size() == 2) << "The operator " << op_name << " requires 2D src";

  TileView tile_view;
  tile_view.valid_shape = src_type->shape_;
  InheritTileViewLayout(tile_view, src_type);
  return std::make_shared<TileType>(src_type->shape_, DataType::FP32, std::nullopt, tile_view);
}

TypePtr DeduceTileTGetScaleAddrType(const std::vector<ExprPtr>& args,
                                    const std::vector<std::pair<std::string, std::any>>& kwargs,
                                    const std::string& op_name) {
  CHECK(args.size() == 2) << "The operator " << op_name
                          << " requires exactly 2 arguments (dst_scale, src), but got " << args.size();
  auto dst_type = As<TileType>(args[0]->GetType());
  auto src_type = As<TileType>(args[1]->GetType());
  CHECK(dst_type) << "The operator " << op_name << " requires dst_scale to be a TileType";
  CHECK(src_type) << "The operator " << op_name << " requires src to be a TileType";
  CHECK(dst_type->dtype_ == DataType::FP8E8M0)
      << "The operator " << op_name << " requires dst_scale dtype FP8E8M0, but got "
      << dst_type->dtype_.ToString();
  // Address-binding op: result reuses dst_scale tile type (same shape/dtype).
  return std::make_shared<TileType>(dst_type->shape_, dst_type->dtype_, /*memref=*/std::nullopt,
                                    dst_type->tile_view_, dst_type->memory_space_);
}

}  // namespace

REGISTER_OP("tile.tquant")
    .set_op_category("TileOp")
    .set_description(
        "MX block-32 dynamic quantization: returns TupleType{quantized, e8m0_scale}. "
        "Maps to pto.tquant.mx. mode attr selects mxfp8_e4m3/mxfp8_e5m2/mxfp4.")
    .add_argument("src", "Source tile (FP16/FP32/BF16, 2D)")
    .set_attr<std::string>("mode")
    .set_input_memory(0, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .not_inplace_safe()
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileTQuantType(args, kwargs, "tile.tquant");
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
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileTDequantType(args, kwargs, "tile.tdequant");
    });

REGISTER_OP("tile.tget_scale_addr")
    .set_op_category("TileOp")
    .set_description(
        "Bind MX scale-tile address from a Left/Right data tile (A5): "
        "dst_addr = src_addr >> SHIFT_MX_ADDR. Maps to pto.tget_scale_addr.")
    .add_argument("dst_scale", "Destination scale tile (FP8E8M0, LeftScale/RightScale)")
    .add_argument("src", "Source Left/Right data tile whose address is scaled")
    .set_input_memory(0, {MemorySpace::LeftScale, MemorySpace::RightScale, MemorySpace::Mat})
    .set_input_memory(1, {MemorySpace::Left, MemorySpace::Right, MemorySpace::Mat})
    .set_output_memory(MemorySpace::LeftScale)  // overridden by reuse of input 0 when present
    .set_output_reuses_input(0)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileTGetScaleAddrType(args, kwargs, "tile.tget_scale_addr");
    });

}  // namespace ir
}  // namespace pypto
