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
  // ``tile.tquant`` (1 arg, DSL form) and ``tile.tquant_dps`` (3 args: src, max,
  // scaling, internal form) share this deduction. The lower_composite pass
  // rewrites the 1-arg DSL form into the 3-arg DPS form, materializing the FP32
  // scratch (max, scaling) as IR-level tiles so the memory planner assigns them
  // addresses (required at --pto-level=level3 — codegen-internal scratch cannot
  // get an address there). ``mode`` still selects the ptoas ``quant_type``.
  INTERNAL_CHECK(args.size() == 1 || args.size() == 3)
      << "Internal error: " << op_name << " requires 1 argument (src) or 3 (src, max, scaling), but got "
      << args.size();
  auto src_type = As<TileType>(args[0]->GetType());
  CHECK(src_type) << "The operator " << op_name << " requires src to be a TileType, but got "
                  << args[0]->GetType()->TypeName();
  CHECK(src_type->shape_.size() == 2) << "The operator " << op_name << " requires 2D src tile";
  CHECK(src_type->dtype_ == DataType::FP16 || src_type->dtype_ == DataType::FP32 ||
        src_type->dtype_ == DataType::BF16)
      << "The operator " << op_name << " requires src dtype in {FP16, FP32, BF16}, but got "
      << src_type->dtype_.ToString();

  if (args.size() == 3) {
    // ptoas TQuantMxOp requires max/scaling element type to match src.
    auto max_type = As<TileType>(args[1]->GetType());
    auto scaling_type = As<TileType>(args[2]->GetType());
    CHECK(max_type && max_type->dtype_ == src_type->dtype_)
        << "The operator " << op_name << " requires max scratch dtype to match src ("
        << src_type->dtype_.ToString() << "), but got "
        << (max_type ? max_type->dtype_.ToString() : std::string("<non-tile>"));
    CHECK(scaling_type && scaling_type->dtype_ == src_type->dtype_)
        << "The operator " << op_name << " requires scaling scratch dtype to match src ("
        << src_type->dtype_.ToString() << "), but got "
        << (scaling_type ? scaling_type->dtype_.ToString() : std::string("<non-tile>"));
  }

  std::string mode = "mxfp8_e4m3";
  for (const auto& [key, value] : kwargs) {
    if (key == "mode") {
      mode = AnyCast<std::string>(value, "kwarg key: mode");
    }
  }
  // Validate the mode (selects ptoas quant_type); the result dtypes are raw bytes.
  GetMxQuantOutDtype(mode, op_name);

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

  // dst element type: ptoas TQuantMxOp::verify() is mode-dependent — MXFP8 dst
  // must be i8/ui8 (FP8 stored as its raw byte, mirrors pto-isa's int8_t FP8
  // tile), while MXFP4_E2M1 dst must be the packed !pto.f4E2M1x2 type. The tstore
  // byte-copies the result into the FP8/FP4 output tensor.
  //
  // MXFP4 packs two logical elements into one storage unit (pto-isa
  // float4_e2m1x2_t). IR shape_/valid_shape for the dst are therefore storage
  // units [M, K/2]. Scale groups and V2C buffers still size from the PHYSICAL
  // padded extents (M_phys*K_phys/32) so alloc_tile rows stay 32-byte aligned;
  // partial validity is applied later via tile.set_validshape, not by shrinking
  // the physical buffer.
  bool is_mxfp4 = (mode == "mxfp4" || mode == "mxfp4_e2m1");
  DataType dst_dtype = is_mxfp4 ? DataType::FP4 : DataType::INT8;
  auto pack_fp4_k = [&](const ExprPtr& logical_k) -> ExprPtr {
    auto c = As<ConstInt>(logical_k);
    if (c) {
      CHECK(c->value_ > 0 && (c->value_ % 2) == 0)
          << "The operator " << op_name << " requires even K for MXFP4 2-elements-per-byte packing, but got "
          << c->value_;
      return std::make_shared<ConstInt>(c->value_ / 2, DataType::INDEX, Span::unknown());
    }
    return MakeFloorDiv(logical_k, std::make_shared<ConstInt>(2, DataType::INDEX, Span::unknown()));
  };
  std::vector<ExprPtr> dst_shape = src_type->shape_;
  if (is_mxfp4) {
    dst_shape = {m_dim, pack_fp4_k(k_dim)};
  }
  TileView dst_view;
  // Keep physical (storage) valid_shape so Canonicalize can drop TileView when
  // valid≡shape — sticking a narrowed valid from a padded source conflicts with
  // later set_validshape / cross-region reassignment of the same SSA.
  dst_view.valid_shape = is_mxfp4 ? dst_shape : src_type->shape_;
  InheritTileViewLayout(dst_view, src_type);
  auto dst_type = std::make_shared<TileType>(dst_shape, dst_dtype, std::nullopt, dst_view);

  // scale (e8m0 exp): raw uint8 bytes as flat [1, groups] (groups = M*K/32).
  // Flat is already 32-byte-row-aligned (one row of groups bytes), so no pad is
  // needed. Downstream move(..., target_shape=[M, K/32]) reshapes as Mat ui8
  // then aliases !pto.f8E8M0 for ScaleLeft (pto.treshape rejects f8E8M0).
  auto m_const = As<ConstInt>(m_dim);
  ExprPtr groups_dim;
  if (m_const && k_const) {
    int64_t groups = m_const->value_ * (k_const->value_ / 32);
    groups_dim = std::make_shared<ConstInt>(groups, DataType::INDEX, Span::unknown());
  } else {
    // Dynamic M/K: keep a conservative placeholder; static shapes are required
    // for the target_shape reshape path used by v4-pro.
    groups_dim = scale_k;
  }
  auto one = std::make_shared<ConstInt>(1, DataType::INDEX, Span::unknown());
  TileView scale_view;
  scale_view.valid_shape = {one, groups_dim};
  scale_view.blayout = TileLayout::row_major;
  scale_view.slayout = TileLayout::none_box;
  scale_view.fractal = 32;
  auto scale_type = std::make_shared<TileType>(std::vector<ExprPtr>{one, groups_dim}, DataType::UINT8,
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
  CHECK(dst_type->dtype_ == DataType::FP8E8M0 || dst_type->dtype_ == DataType::UINT8)
      << "The operator " << op_name
      << " requires dst_scale dtype FP8E8M0 (or raw UINT8 from host prequant), but got "
      << dst_type->dtype_.ToString();
  // Address-binding op: result reuses dst_scale tile type (same shape/dtype).
  return std::make_shared<TileType>(dst_type->shape_, dst_type->dtype_, /*memref=*/std::nullopt,
                                    dst_type->tile_view_, dst_type->memory_space_);
}

}  // namespace

REGISTER_OP("tile.tquant")
    .set_op_category("TileOp")
    .set_description(
        "MX block-32 dynamic quantization: TupleType{quantized (raw int8 bytes), e8m0_scale "
        "(raw uint8 bytes)}. The lower_composite pass rewrites this 1-arg DSL form into "
        "tile.tquant_dps, materializing the FP32 scratch (max, scaling) as IR-level tiles so the "
        "memory planner can address them; codegen then emits pto.tquant.mx. mode selects "
        "mxfp8_e4m3/mxfp8_e5m2/mxfp4. The FP8 value is stored as its byte representation (mirrors "
        "pto-isa's int8_t FP8 / uint8_t E8M0 tiles); the tstore byte-copies into the FP8 outputs.")
    .add_argument("src", "Source tile (FP16/FP32/BF16, 2D)")
    .set_attr<std::string>("mode")
    .set_input_memory(0, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .not_inplace_safe()
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileTQuantType(args, kwargs, "tile.tquant");
    });

// Internal DPS form produced by the tile.tquant lowering rule. Carries the two
// FP32 scratch tiles (max, scaling) as explicit operands so the memory planner
// assigns them addresses (codegen-internal scratch cannot get an address at
// --pto-level=level3). Has NO composite-lowering rule (idempotency); codegen
// lowers it to the ptoas pto.tquant.mx instruction.
REGISTER_OP("tile.tquant_dps")
    .set_op_category("TileOp")
    .set_description(
        "Internal DPS form of tile.tquant with explicit FP32 scratch operands. Lowers to "
        "pto.tquant.mx: TQuant(dst, src, exp, max, scaling). dst=raw i8 (or f4E2M1x2 "
        "for MXFP4); exp=raw ui8; max/scaling match src dtype.")
    .add_argument("src", "Source tile (FP16/FP32/BF16, 2D)")
    .add_argument("max", "Per-group FP32 max scratch tile (write-only, hardware-consumed)")
    .add_argument("scaling", "Per-group FP32 scaling scratch tile (write-only, hardware-consumed)")
    .set_attr<std::string>("mode")
    .set_input_memory(0, MemorySpace::Vec)
    .set_input_memory(1, MemorySpace::Vec)
    .set_input_memory(2, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .not_inplace_safe()
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileTQuantType(args, kwargs, "tile.tquant_dps");
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
