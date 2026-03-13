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
  return std::make_shared<TileType>(tile_type->shape_, tile_type->dtype_, std::nullopt, tile_view);
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

REGISTER_OP("tile.recip")
    .set_op_category("TileOp")
    .set_description("Reciprocal (1/x) of a tile (element-wise)")
    .add_argument("tile", "Input tile (TileType)")
    .set_input_memory(0, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
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
    .set_description("Reciprocal square root (1/sqrt(x)) of a tile (element-wise)")
    .add_argument("tile", "Input tile (TileType)")
    .set_input_memory(0, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileUnaryType(args, kwargs, "tile.rsqrt");
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
      return std::make_shared<TileType>(tile_type->shape_, tile_type->dtype_, std::nullopt, tile_view);
    });

}  // namespace ir
}  // namespace pypto
