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
 * @file unary.cpp
 * @brief Unary tile operations (Neg, Exp, Recip, Sqrt, Rsqrt, Cast)
 *
 * This file implements unary operations for tile-level programming.
 * Unary operations take a TileType and return a TileType with the same shape.
 */

#include <algorithm>
#include <any>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/any_cast.h"
#include "pypto/core/dtype.h"
#include "pypto/core/error.h"
#include "pypto/core/logging.h"
#include "pypto/ir/cast_saturation.h"
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

TypePtr DeduceTileUnaryType(const std::vector<ExprPtr>& args,
                            const std::vector<std::pair<std::string, std::any>>& kwargs,
                            const std::string& op_name) {
  CHECK(args.size() == 1) << "The operator " << op_name << " requires exactly 1 argument, but got "
                          << args.size();

  // Argument must be TileType
  auto tile_type = As<TileType>(args[0]->GetType());
  CHECK(tile_type) << "The operator " << op_name << " requires argument to be a TileType, but got "
                   << args[0]->GetType()->TypeName();

  // Unary operations preserve shape, dtype, and the source tile's valid_shape (issue #1370).
  TileView tile_view;
  tile_view.valid_shape = GetValidShape(tile_type);
  InheritTileViewLayout(tile_view, tile_type);
  return std::make_shared<TileType>(tile_type->shape_, tile_type->dtype_, std::nullopt, tile_view);
}

// tile.rsqrt accepts 1 arg (basic mode) or 2 args (high-precision mode with tmp workspace).
// The tmp tile must match the input in shape and dtype.
TypePtr DeduceTileRsqrtType(const std::vector<ExprPtr>& args,
                            const std::vector<std::pair<std::string, std::any>>& kwargs,
                            const std::string& op_name) {
  CHECK(args.size() == 1 || args.size() == 2)
      << "The operator " << op_name << " requires 1 or 2 arguments, but got " << args.size();

  auto tile_type = As<TileType>(args[0]->GetType());
  CHECK(tile_type) << "The operator " << op_name << " requires first argument to be a TileType, but got "
                   << args[0]->GetType()->TypeName();

  if (args.size() == 2) {
    auto tmp_type = As<TileType>(args[1]->GetType());
    CHECK(tmp_type) << "The operator " << op_name << " requires tmp argument to be a TileType, but got "
                    << args[1]->GetType()->TypeName();
    CHECK(tmp_type->dtype_ == tile_type->dtype_)
        << op_name << ": tmp tile dtype (" << tmp_type->dtype_.ToString() << ") must match input dtype ("
        << tile_type->dtype_.ToString() << ")";
    CHECK(tmp_type->shape_.size() == tile_type->shape_.size())
        << op_name << ": tmp tile rank (" << tmp_type->shape_.size() << ") must match input rank ("
        << tile_type->shape_.size() << ")";
    for (size_t i = 0; i < tile_type->shape_.size(); ++i) {
      CHECK(DimensionsEqual(tmp_type->shape_[i], tile_type->shape_[i]))
          << op_name << ": tmp tile shape mismatch at dimension " << i << " (tmp: " << tmp_type->shape_[i]
          << ", input: " << tile_type->shape_[i] << ")";
    }
  }

  TileView tile_view;
  tile_view.valid_shape = GetValidShape(tile_type);
  InheritTileViewLayout(tile_view, tile_type);
  return std::make_shared<TileType>(tile_type->shape_, tile_type->dtype_, std::nullopt, tile_view);
}

// Shared FP32-only deducer for transcendental tile ops (tile.sin, tile.cos).
// These ops are intentionally FP32-only to avoid silent precision loss; callers
// must explicitly cast non-FP32 inputs via tile.cast (or pl.cast at the DSL layer).
TypePtr DeduceTileFP32OnlyType(const std::vector<ExprPtr>& args,
                               const std::vector<std::pair<std::string, std::any>>& kwargs,
                               const std::string& op_name) {
  // Reuse the standard unary deducer for shape/layout handling and basic validation.
  TypePtr base_type = DeduceTileUnaryType(args, kwargs, op_name);
  auto tile_type = As<TileType>(base_type);

  // FP32-only: do NOT auto-promote. Reject non-FP32 inputs with an actionable error.
  CHECK(tile_type->dtype_ == DataType::FP32)
      << op_name << " is FP32-only, but got input with dtype " << tile_type->dtype_.ToString()
      << ". Cast the input to FP32 explicitly via pl.cast(tile, pl.FP32) before applying " << op_name << ".";

  return base_type;
}

TypePtr DeduceTileCastType(const std::vector<ExprPtr>& args,
                           const std::vector<std::pair<std::string, std::any>>& kwargs,
                           const std::string& op_name) {
  CHECK(args.size() == 1 || args.size() == 2)
      << "The operator " << op_name << " requires 1 or 2 arguments (tile[, tmp]), but got " << args.size();

  // Argument must be TileType
  auto tile_type = As<TileType>(args[0]->GetType());
  CHECK(tile_type) << "The operator " << op_name << " requires argument to be a TileType, but got "
                   << args[0]->GetType()->TypeName();

  if (args.size() == 2) {
    auto tmp_type = As<TileType>(args[1]->GetType());
    CHECK(tmp_type) << "The operator " << op_name
                    << " requires optional second argument 'tmp' to be a TileType, but got "
                    << args[1]->GetType()->TypeName();
  }

  ValidateCastSaturationModeKwarg(kwargs, op_name);

  // Read target_type from kwargs
  bool found_target_type = false;
  DataType target_dtype;
  for (const auto& [key, value] : kwargs) {
    if (key == "target_type") {
      // Handle both DataType and int for backward compatibility
      if (value.type() == typeid(DataType)) {
        target_dtype = AnyCast<DataType>(value, "kwarg key: target_type");
      } else if (value.type() == typeid(int)) {
        target_dtype = static_cast<DataType>(AnyCast<int>(value, "kwarg key: target_type"));
      } else {
        throw TypeError("target_type must be a DataType or int, but got " + std::string(value.type().name()));
      }
      found_target_type = true;
      break;
    }
  }
  CHECK(found_target_type) << "tile.cast requires 'target_type' kwarg";

  // `mode` is a declared attr that codegen reads unconditionally
  // (MakeModalCodegenPTO -> pto.tcvt {rmode = ...}). A missing kwarg silently
  // reads back as 0 == round_mode NONE instead of the DSL default ROUND, so
  // require it here rather than let a mode-less cast reach the backend.
  const bool found_mode =
      std::any_of(kwargs.begin(), kwargs.end(), [](const auto& kv) { return kv.first == "mode"; });
  CHECK(found_mode) << op_name
                    << " requires a 'mode' kwarg (round mode: none(0), rint(1), round(2), "
                       "floor(3), ceil(4), trunc(5), odd(6)). Pass mode=\"round\" (2) to match "
                       "the pl.cast / tile_ops.cast default, or mode=\"none\" (0) when the "
                       "conversion cannot round (e.g. int -> int).";

  // Reject same-dtype cast: the hardware pto.tcvt instruction is for
  // cross-dtype conversion, and a same-dtype invocation can corrupt values
  // rather than acting as an identity copy. Detecting this at construction
  // time keeps malformed casts out of every downstream pass and codegen.
  CHECK(tile_type->dtype_ != target_dtype)
      << op_name << ": target_type " << target_dtype.ToString()
      << " equals input dtype; same-dtype cast is not a valid operation. "
      << "Remove the cast or use a different target_type.";

  // Packed FP4 ↔ a wider type is a 2:1 last-axis conversion
  // (bf16[M,K] ↔ f4x2[M,K/2]): shape,
  // valid_shape, and ND stride all follow that change.
  const Span& span = args[0]->span_;
  ValidatePackedFp4CastMode(target_dtype, GetRequiredKwarg<int>(kwargs, "mode", "cast"), span);
  const auto src_view = tile_view_semantics::GetEffectiveTileView(*tile_type);
  const bool packed_cast = tile_type->dtype_.IsPackedFp4() || target_dtype.IsPackedFp4();
  const bool packed_dn_cast = packed_cast && src_view.blayout == TileLayout::col_major;
  CHECK_SPAN(!packed_cast || src_view.slayout == TileLayout::none_box, span)
      << "cast involving packed FP4 requires none_box storage; native FP4 TCVT does not convert boxed "
         "layouts";
  if (packed_dn_cast) {
    // The native DN path is correct only when static-valid descriptors select
    // the contiguous conversion. Dynamic-valid or partial DN tiles misaddress
    // the row-strided implementation; codegen supplies static-valid aliases.
    const bool static_2d =
        tile_type->shape_.size() == 2 &&
        std::all_of(tile_type->shape_.begin(), tile_type->shape_.end(), [](const ExprPtr& extent) {
          auto value = As<ConstInt>(extent);
          return value && value->value_ > 0;
        });
    CHECK_SPAN(
        static_2d && tile_view_semantics::ShapeExprListsEquivalent(src_view.valid_shape, tile_type->shape_),
        span)
        << "packed FP4 DN cast requires a full static 2D valid shape; partial or padded DN casts are "
           "unsupported";
    CHECK_SPAN((tile_type->dtype_.IsPackedFp4() || tile_type->dtype_ == DataType::BF16) &&
                   (target_dtype.IsPackedFp4() || target_dtype == DataType::BF16),
               span)
        << "packed FP4 DN cast supports only BF16 and packed FP4 endpoints; use ND storage for wider cast "
           "chains";
    auto one = std::make_shared<ConstInt>(1, DataType::INDEX, span);
    const std::vector<ExprPtr> expected_stride = {one, tile_type->shape_[0]};
    CHECK_SPAN(src_view.stride.empty() ||
                   tile_view_semantics::ShapeExprListsEquivalent(src_view.stride, expected_stride),
               span)
        << "packed FP4 DN cast requires canonical contiguous column-major strides";
    auto offset = As<ConstInt>(src_view.start_offset);
    CHECK_SPAN(!src_view.start_offset || (offset && offset->value_ == 0), span)
        << "packed FP4 DN cast requires zero tile start_offset";
  }
  auto adjust_shape = [&](std::vector<ExprPtr> dims) {
    if (packed_dn_cast) std::swap(dims[0], dims[1]);
    dims = AdjustPackedFp4CastLastDim(std::move(dims), tile_type->dtype_, target_dtype, span);
    if (packed_dn_cast) std::swap(dims[0], dims[1]);
    return dims;
  };
  auto shape = adjust_shape(tile_type->shape_);
  TileView tile_view;
  tile_view.valid_shape = adjust_shape(GetValidShape(tile_type));
  tile_view.stride =
      AdjustPackedFp4CastStrides(src_view.stride, shape, tile_type->dtype_, target_dtype, span);
  if (packed_dn_cast) {
    tile_view.stride = {std::make_shared<ConstInt>(1, DataType::INDEX, span), shape[0]};
  }
  InheritTileViewLayout(tile_view, tile_type);
  // Dense DN casts execute in Vec; recording that placement now keeps memory
  // inference from canonicalizing the BF16 intermediate back to row-major.
  const std::optional<MemorySpace> memory =
      packed_dn_cast ? std::make_optional(MemorySpace::Vec) : std::nullopt;
  return std::make_shared<TileType>(std::move(shape), target_dtype, std::nullopt, tile_view, memory);
}

// ============================================================================
// Op Registration
// ============================================================================

REGISTER_OP("tile.neg")
    .set_op_category("TileOp")
    .functional_execution_memory_access()
    .set_description("Negation of a tile (element-wise)")
    .add_argument("tile", "Input tile (TileType)")
    .set_input_memory(0, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileUnaryType(args, kwargs, "tile.neg");
    });

REGISTER_OP("tile.exp")
    .set_op_category("TileOp")
    .functional_execution_memory_access()
    .set_description("Exponential function of a tile (element-wise)")
    .add_argument("tile", "Input tile (TileType)")
    .set_input_memory(0, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileUnaryType(args, kwargs, "tile.exp");
    });

REGISTER_OP("tile.sin")
    .set_op_category("TileOp")
    .functional_execution_memory_access()
    .set_description("Element-wise sine of a tile (radians). FP32 only.")
    .add_argument("tile", "Input tile (TileType, FP32)")
    .set_input_memory(0, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileFP32OnlyType(args, kwargs, "tile.sin");
    });

REGISTER_OP("tile.cos")
    .set_op_category("TileOp")
    .functional_execution_memory_access()
    .set_description("Element-wise cosine of a tile (radians). FP32 only.")
    .add_argument("tile", "Input tile (TileType, FP32)")
    .set_input_memory(0, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileFP32OnlyType(args, kwargs, "tile.cos");
    });

REGISTER_OP("tile.recip")
    .set_op_category("TileOp")
    .functional_execution_memory_access()
    .set_description("Reciprocal (1/x) of a tile (element-wise)")
    .add_argument("tile", "Input tile (TileType)")
    .set_attr<bool>("high_precision")
    .set_input_memory(0, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .not_inplace_safe()
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      auto result_type = DeduceTileUnaryType(args, kwargs, "tile.recip");
      auto tile_type = As<TileType>(args[0]->GetType());
      CHECK(!GetKwargOr<bool>(kwargs, "high_precision", false) || tile_type->dtype_ == DataType::FP16 ||
            tile_type->dtype_ == DataType::FP32)
          << "The operator tile.recip supports high_precision only for FP16 or FP32 because the PTOAS "
             "high-precision template does not implement other dtypes";
      return result_type;
    });

REGISTER_OP("tile.sqrt")
    .set_op_category("TileOp")
    .functional_execution_memory_access()
    .set_description("Square root of a tile (element-wise)")
    .add_argument("tile", "Input tile (TileType)")
    .set_input_memory(0, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileUnaryType(args, kwargs, "tile.sqrt");
    });

REGISTER_OP("tile.rsqrt")
    .set_op_category("TileOp")
    .set_description(
        "Reciprocal square root (1/sqrt(x)) of a tile (element-wise). "
        "Passing an optional second tmp tile activates the high-precision PTO path.")
    .add_argument("tile", "Input tile (TileType)")
    .add_argument("tmp", "Optional scratch tile for high-precision path (TileType)")
    // A2/A3 initializes tmp before reading it; incoming contents and validity
    // are not source data. Its physical shape and dtype remain deducer checks.
    .set_arg_effect(1, ArgEffect::Write)
    .set_workspace_arg(1)
    .set_input_memory(0, MemorySpace::Vec)
    .set_input_memory(1, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    // The output must not share a buffer with the input or scratch during the
    // high-precision computation.
    .not_inplace_safe()
    // Workspace is not a lane-invariant capacity declaration: tmp must still
    // match the input's rank and every physical dimension after splitting.
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileRsqrtType(args, kwargs, "tile.rsqrt");
    });

REGISTER_OP("tile.cast")
    .set_op_category("TileOp")
    .functional_execution_memory_access()
    .set_description("Cast tile to target data type (element-wise)")
    .add_argument("tile", "Input tile (TileType)")
    .add_argument("tmp", "Optional A2/A3 scratch tile for non-saturating narrowing pto.tcvt")
    .set_attr<DataType>("target_type")
    .set_attr<int>("mode")  // Round Mode: None(0), RINT(1), ROUND(2), FLOOR(3), CEIL(4), TRUNC(5), ODD(6)
    // Optional destination saturation: OFF(0) / ON(1). Absent means "keep the
    // backend default" — see include/pypto/ir/cast_saturation.h.
    .set_attr<int>("saturation_mode")
    .set_input_memory(0, MemorySpace::Vec)
    .set_input_memory(1, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .forbid_output_alias(1)
    .set_lane_invariant_arg(1)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileCastType(args, kwargs, "tile.cast");
    });

REGISTER_OP("tile.log")
    .set_op_category("TileOp")
    .functional_execution_memory_access()
    .set_description("Natural logarithm of a tile (element-wise)")
    .add_argument("tile", "Input tile (TileType)")
    .set_attr<bool>("high_precision")
    .set_input_memory(0, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      auto result_type = DeduceTileUnaryType(args, kwargs, "tile.log");
      auto tile_type = As<TileType>(args[0]->GetType());
      CHECK(tile_type->dtype_ == DataType::FP16 || tile_type->dtype_ == DataType::FP32)
          << "tile.log requires an FP16 or FP32 tile operand, but got " << tile_type->dtype_.ToString();
      return result_type;
    });

REGISTER_OP("tile.abs")
    .set_op_category("TileOp")
    .functional_execution_memory_access()
    .set_description("Absolute value of a tile (element-wise)")
    .add_argument("tile", "Input tile (TileType)")
    .set_input_memory(0, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileUnaryType(args, kwargs, "tile.abs");
    });

REGISTER_OP("tile.relu")
    .set_op_category("TileOp")
    .functional_execution_memory_access()
    .set_description("ReLU activation function of a tile (element-wise)")
    .add_argument("tile", "Input tile (TileType)")
    .set_input_memory(0, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileUnaryType(args, kwargs, "tile.relu");
    });

REGISTER_OP("tile.not")
    .set_op_category("TileOp")
    .functional_execution_memory_access()
    .set_description("Element-wise bitwise NOT of a tile")
    .add_argument("tile", "Input tile (TileType) with int16 or uint16 dtype")
    .set_input_memory(0, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      const std::string op_name = "tile.not";
      CHECK(args.size() == 1) << "The operator " << op_name << " requires exactly 1 argument, but got "
                              << args.size();
      auto tile_type = As<TileType>(args[0]->GetType());
      CHECK(tile_type) << "The operator " << op_name << " requires argument to be a TileType, but got "
                       << args[0]->GetType()->TypeName();
      CHECK(tile_type->dtype_ == DataType::INT16 || tile_type->dtype_ == DataType::UINT16)
          << "The operator " << op_name << " requires int16 or uint16 tile dtype, but got "
          << tile_type->dtype_.ToString();
      TileView tile_view;
      tile_view.valid_shape = GetValidShape(tile_type);
      InheritTileViewLayout(tile_view, tile_type);
      return std::make_shared<TileType>(tile_type->shape_, tile_type->dtype_, std::nullopt, tile_view);
    });

}  // namespace ir
}  // namespace pypto
