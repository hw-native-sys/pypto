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
 * @file memory.cpp
 * @brief Memory block operations (get_block_idx, ub_copy_in, ub_copy_out)
 *
 * This file implements memory operations for block-level programming.
 * These operations handle data movement between tensors and unified buffers (tiles).
 */

#include <memory>
#include <string>
#include <vector>

#include "pypto/core/common.h"
#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/core.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/type.h"
#include "pypto/ir/type_inference.h"

namespace pypto {
namespace ir {

TypePtr DeduceBlockGetBlockIdxType(const std::vector<ExprPtr>& args,
                                   const std::vector<std::pair<std::string, std::any>>& kwargs,
                                   const std::string& op_name) {
  CHECK(args.size() == 0) << "The operator " << op_name << " requires no arguments, but got " << args.size();

  // get_block_idx returns INT32 scalar
  return std::make_shared<ScalarType>(DataType::INT32);
}

TypePtr DeduceBlockUbCopyInType(const std::vector<ExprPtr>& args,
                                const std::vector<std::pair<std::string, std::any>>& kwargs,
                                const std::string& op_name) {
  // ub_copy_in signature: (tensor, row_offset, col_offset, height, width)
  // We need at least the tensor argument
  CHECK(args.size() >= 1) << "The operator " << op_name << " requires at least 1 argument, but got "
                          << args.size();

  // First argument must be TensorType
  auto tensor_type = std::dynamic_pointer_cast<const TensorType>(args[0]->GetType());
  CHECK(tensor_type) << "The operator " << op_name << " requires first argument to be a TensorType, but got "
                     << args[0]->GetType()->TypeName();

  // If we have shape arguments (height, width), use them to determine tile shape
  // Otherwise, we need to infer from context or use dynamic dimensions
  std::vector<ExprPtr> tile_shape;

  if (args.size() >= 5) {
    // We have height and width arguments (args[3] and args[4])
    // These should be scalar expressions that we can use as dimensions
    // For now, we'll use them directly as shape dimensions
    tile_shape.push_back(args[3]);
    tile_shape.push_back(args[4]);
  } else {
    // Use dynamic dimensions if shape is not provided
    // Create ConstInt expressions for dynamic dimensions
    auto dynamic_dim_height =
        std::make_shared<ConstInt>(static_cast<int>(kDynamicDim), DataType::INT32, Span::unknown());
    auto dynamic_dim_width =
        std::make_shared<ConstInt>(static_cast<int>(kDynamicDim), DataType::INT32, Span::unknown());
    tile_shape.push_back(dynamic_dim_height);
    tile_shape.push_back(dynamic_dim_width);
  }

  // Return TileType with same dtype as tensor
  return std::make_shared<TileType>(tile_shape, tensor_type->dtype_);
}

TypePtr DeduceBlockUbCopyOutType(const std::vector<ExprPtr>& args,
                                 const std::vector<std::pair<std::string, std::any>>& kwargs,
                                 const std::string& op_name) {
  // ub_copy_out signature: (tile, row_offset, col_offset, height, width, output_tensor)
  // We need at least the tile and output_tensor arguments
  CHECK(args.size() >= 2) << "The operator " << op_name << " requires at least 2 arguments, but got "
                          << args.size();

  // First argument must be TileType
  auto tile_type = std::dynamic_pointer_cast<const TileType>(args[0]->GetType());
  CHECK(tile_type) << "The operator " << op_name << " requires first argument to be a TileType, but got "
                   << args[0]->GetType()->TypeName();

  // Last argument should be the output tensor
  auto output_tensor_type = std::dynamic_pointer_cast<const TensorType>(args.back()->GetType());
  CHECK(output_tensor_type) << "The operator " << op_name
                            << " requires last argument to be a TensorType, but got "
                            << args.back()->GetType()->TypeName();

  // ub_copy_out returns the output tensor (same type)
  return output_tensor_type;
}

// ============================================================================
// Registration Function for Block Memory Operations
// ============================================================================

REGISTER_OP("block.get_block_idx")
    .set_op_category("BlockOp")
    .set_description("Get the current block index")
    .no_argument()
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceBlockGetBlockIdxType(args, kwargs, "block.get_block_idx");
    });

REGISTER_OP("block.ub_copy_in")
    .set_op_category("BlockOp")
    .set_description("Copy data from tensor to unified buffer (tile)")
    .add_argument("tensor", "Source tensor (TensorType)")
    .add_argument("row_offset", "Row offset (scalar)")
    .add_argument("col_offset", "Column offset (scalar)")
    .add_argument("height", "Tile height (scalar)")
    .add_argument("width", "Tile width (scalar)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceBlockUbCopyInType(args, kwargs, "block.ub_copy_in");
    });

REGISTER_OP("block.ub_copy_out")
    .set_op_category("BlockOp")
    .set_description("Copy data from unified buffer (tile) to tensor")
    .add_argument("tile", "Source tile (TileType)")
    .add_argument("row_offset", "Row offset (scalar)")
    .add_argument("col_offset", "Column offset (scalar)")
    .add_argument("height", "Output height (scalar)")
    .add_argument("width", "Output width (scalar)")
    .add_argument("output_tensor", "Output tensor (TensorType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceBlockUbCopyOutType(args, kwargs, "block.ub_copy_out");
    });

}  // namespace ir
}  // namespace pypto
