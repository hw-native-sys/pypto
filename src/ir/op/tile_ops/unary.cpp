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
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/op_registry.h"
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

  // Unary operations preserve shape and data type
  TileView tile_view;
  tile_view.valid_shape = tile_type->shape_;
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
  tile_view.valid_shape = tile_type->shape_;
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
  CHECK(args.size() == 1) << "The operator " << op_name << " requires exactly 1 argument, but got "
                          << args.size();

  // Argument must be TileType
  auto tile_type = As<TileType>(args[0]->GetType());
  CHECK(tile_type) << "The operator " << op_name << " requires argument to be a TileType, but got "
                   << args[0]->GetType()->TypeName();

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

  // Cast operation preserves shape but changes data type
  TileView tile_view;
  tile_view.valid_shape = tile_type->shape_;
  InheritTileViewLayout(tile_view, tile_type);
  return std::make_shared<TileType>(tile_type->shape_, target_dtype, std::nullopt, tile_view);
}

// ============================================================================
// Op Registration
// ============================================================================

REGISTER_OP("tile.neg")
    .set_op_category("TileOp")
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
    .set_description("Reciprocal (1/x) of a tile (element-wise)")
    .add_argument("tile", "Input tile (TileType)")
    .set_input_memory(0, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .not_inplace_safe()
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileUnaryType(args, kwargs, "tile.recip");
    });

REGISTER_OP("tile.sqrt")
    .set_op_category("TileOp")
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
    .set_input_memory(0, MemorySpace::Vec)
    .set_input_memory(1, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileRsqrtType(args, kwargs, "tile.rsqrt");
    });

REGISTER_OP("tile.cast")
    .set_op_category("TileOp")
    .set_description("Cast tile to target data type (element-wise)")
    .add_argument("tile", "Input tile (TileType)")
    .set_attr<DataType>("target_type")
    .set_attr<int>("mode")  // Round Mode: None(0), RINT(1), ROUND(2), FLOOR(3), CEIL(4), TRUNC(5), ODD(6)
    .set_input_memory(0, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileCastType(args, kwargs, "tile.cast");
    });

REGISTER_OP("tile.log")
    .set_op_category("TileOp")
    .set_description("Natural logarithm of a tile (element-wise)")
    .add_argument("tile", "Input tile (TileType)")
    .set_input_memory(0, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileUnaryType(args, kwargs, "tile.log");
    });

REGISTER_OP("tile.abs")
    .set_op_category("TileOp")
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
      tile_view.valid_shape = tile_type->shape_;
      InheritTileViewLayout(tile_view, tile_type);
      return std::make_shared<TileType>(tile_type->shape_, tile_type->dtype_, std::nullopt, tile_view);
    });

}  // namespace ir
}  // namespace pypto
