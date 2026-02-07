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
 * @brief Unary block operations (Neg, Exp, Recip, Sqrt, Rsqrt, Cast)
 *
 * This file implements unary operations for block-level programming.
 * Unary operations take a TileType and return a TileType with the same shape.
 */

#include <algorithm>
#include <any>
#include <memory>
#include <string>
#include <vector>

#include "pypto/codegen/cce/cce_codegen.h"
#include "pypto/core/any_cast.h"
#include "pypto/core/common.h"
#include "pypto/core/logging.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/type.h"
#include "pypto/ir/type_inference.h"

namespace pypto {
namespace ir {

TypePtr DeduceBlockUnaryType(const std::vector<ExprPtr>& args,
                             const std::vector<std::pair<std::string, std::any>>& kwargs,
                             const std::string& op_name) {
  CHECK(args.size() == 1) << "The operator " << op_name << " requires exactly 1 argument, but got "
                          << args.size();

  // Argument must be TileType
  auto tile_type = As<TileType>(args[0]->GetType());
  CHECK(tile_type) << "The operator " << op_name << " requires argument to be a TileType, but got "
                   << args[0]->GetType()->TypeName();

  // Unary operations preserve shape and data type
  return std::make_shared<TileType>(tile_type->shape_, tile_type->dtype_);
}

TypePtr DeduceBlockCastType(const std::vector<ExprPtr>& args,
                            const std::vector<std::pair<std::string, std::any>>& kwargs,
                            const std::string& op_name) {
  CHECK(args.size() == 1) << "The operator " << op_name << " requires exactly 1 argument, but got "
                          << args.size();

  // Argument must be TileType
  auto tile_type = As<TileType>(args[0]->GetType());
  CHECK(tile_type) << "The operator " << op_name << " requires argument to be a TileType, but got "
                   << args[0]->GetType()->TypeName();

  // Extract target_dtype from kwargs
  auto it = std::find_if(kwargs.begin(), kwargs.end(),
                         [](const auto& pair) { return pair.first == "target_dtype"; });
  CHECK(it != kwargs.end()) << "The operator " << op_name << " requires 'target_dtype' kwarg";

  DataType target_dtype = static_cast<DataType>(std::any_cast<int>(it->second));

  // Cast operation preserves shape but changes data type
  return std::make_shared<TileType>(tile_type->shape_, target_dtype);
}

// ============================================================================
// Op Registration
// ============================================================================

REGISTER_OP("block.neg")
    .set_op_category("BlockOp")
    .set_description("Negation of a tile (element-wise)")
    .add_argument("tile", "Input tile (TileType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceBlockUnaryType(args, kwargs, "block.neg");
    });

REGISTER_OP("block.exp")
    .set_op_category("BlockOp")
    .set_description("Exponential function of a tile (element-wise)")
    .add_argument("tile", "Input tile (TileType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceBlockUnaryType(args, kwargs, "block.exp");
    });

REGISTER_OP("block.recip")
    .set_op_category("BlockOp")
    .set_description("Reciprocal (1/x) of a tile (element-wise)")
    .add_argument("tile", "Input tile (TileType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceBlockUnaryType(args, kwargs, "block.recip");
    });

REGISTER_OP("block.sqrt")
    .set_op_category("BlockOp")
    .set_description("Square root of a tile (element-wise)")
    .add_argument("tile", "Input tile (TileType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceBlockUnaryType(args, kwargs, "block.sqrt");
    });

REGISTER_OP("block.rsqrt")
    .set_op_category("BlockOp")
    .set_description("Reciprocal square root (1/sqrt(x)) of a tile (element-wise)")
    .add_argument("tile", "Input tile (TileType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceBlockUnaryType(args, kwargs, "block.rsqrt");
    });

REGISTER_OP("block.cast")
    .set_op_category("BlockOp")
    .set_description("Cast tile to target data type (element-wise)")
    .set_pipe(PipeType::V)
    .add_argument("tile", "Input tile (TileType)")
    .set_attr<int>("target_dtype")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceBlockCastType(args, kwargs, "block.cast");
    });

REGISTER_OP("block.log")
    .set_op_category("BlockOp")
    .set_description("Natural logarithm of a tile (element-wise)")
    .set_pipe(PipeType::V)
    .add_argument("tile", "Input tile (TileType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceBlockUnaryType(args, kwargs, "block.log");
    });

REGISTER_OP("block.abs")
    .set_op_category("BlockOp")
    .set_description("Absolute value of a tile (element-wise)")
    .set_pipe(PipeType::V)
    .add_argument("tile", "Input tile (TileType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceBlockUnaryType(args, kwargs, "block.abs");
    });

REGISTER_OP("block.relu")
    .set_op_category("BlockOp")
    .set_description("ReLU activation function of a tile (element-wise)")
    .set_pipe(PipeType::V)
    .add_argument("tile", "Input tile (TileType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceBlockUnaryType(args, kwargs, "block.relu");
    });

}  // namespace ir
}  // namespace pypto
