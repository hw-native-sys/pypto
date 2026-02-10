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
 * @brief Memory block operations (get_block_idx, load, store)
 *
 * This file implements memory operations for block-level programming.
 * These operations handle data movement between tensors and unified buffers (tiles).
 */

#include <any>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "pypto/core/common.h"
#include "pypto/core/dtype.h"
#include "pypto/core/error.h"
#include "pypto/core/logging.h"
#include "pypto/ir/core.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/type.h"
#include "pypto/ir/type_inference.h"

namespace pypto {
namespace ir {

// Helper to get kwargs value with default (uses vector to preserve order)
template <typename T>
T GetKwarg(const std::vector<std::pair<std::string, std::any>>& kwargs, const std::string& key,
           const std::optional<T>& default_value = std::nullopt) {
  for (const auto& [k, v] : kwargs) {
    if (k == key) {
      return AnyCast<T>(v, "kwarg key: " + key);
    }
  }
  if (default_value) {
    return *default_value;
  }
  throw ValueError("Missing kwarg: " + key);
}

TypePtr DeduceBlockGetBlockIdxType(const std::vector<ExprPtr>& args,
                                   const std::vector<std::pair<std::string, std::any>>& kwargs,
                                   const std::string& op_name) {
  CHECK(args.size() == 0) << "The operator " << op_name << " requires no arguments, but got " << args.size();

  // get_block_idx returns UINT64 scalar
  return std::make_shared<ScalarType>(DataType::UINT64);
}

TypePtr DeduceBlockLoadType(const std::vector<ExprPtr>& args,
                            const std::vector<std::pair<std::string, std::any>>& kwargs,
                            const std::string& op_name) {
  // load signature: (tensor, offsets_tuple, shapes_tuple)
  CHECK(args.size() == 3) << "The operator " << op_name
                          << " requires 3 arguments (tensor, offsets, shapes), but got " << args.size();

  // First argument must be TensorType
  auto tensor_type = As<TensorType>(args[0]->GetType());
  CHECK(tensor_type) << "The operator " << op_name << " requires first argument to be a TensorType, but got "
                     << args[0]->GetType()->TypeName();

  // Second argument must be TupleType (offsets)
  auto offsets_tuple = As<MakeTuple>(args[1]);
  CHECK(offsets_tuple) << "The operator " << op_name
                       << " requires second argument to be a tuple (offsets), but got "
                       << args[1]->GetType()->TypeName();

  // Third argument must be TupleType (shapes)
  auto shapes_tuple = As<MakeTuple>(args[2]);
  CHECK(shapes_tuple) << "The operator " << op_name
                      << " requires third argument to be a tuple (shapes), but got "
                      << args[2]->GetType()->TypeName();

  // Verify offsets and shapes have same number of dimensions
  CHECK(offsets_tuple->elements_.size() == shapes_tuple->elements_.size())
      << "The operator " << op_name
      << " requires offsets and shapes to have same number of dimensions, but got "
      << offsets_tuple->elements_.size() << " offsets and " << shapes_tuple->elements_.size() << " shapes";

  CHECK(shapes_tuple->elements_.size() > 0)
      << "The operator " << op_name << " requires at least one dimension, but got empty shapes tuple";

  // Build tile shape from shapes tuple
  std::vector<ExprPtr> tile_shape;
  for (const auto& shape_expr : shapes_tuple->elements_) {
    tile_shape.push_back(shape_expr);
  }

  // Return TileType with same dtype as tensor
  return std::make_shared<TileType>(tile_shape, tensor_type->dtype_);
}

TypePtr DeduceBlockStoreType(const std::vector<ExprPtr>& args,
                             const std::vector<std::pair<std::string, std::any>>& kwargs,
                             const std::string& op_name) {
  // store signature: (tile, offsets_tuple, shapes_tuple, output_tensor)
  CHECK(args.size() == 4) << "The operator " << op_name
                          << " requires 4 arguments (tile, offsets, shapes, output_tensor), but got "
                          << args.size();

  // First argument must be TileType
  auto tile_type = As<TileType>(args[0]->GetType());
  CHECK(tile_type) << "The operator " << op_name << " requires first argument to be a TileType, but got "
                   << args[0]->GetType()->TypeName();

  // Second argument must be TupleType (offsets)
  auto offsets_tuple = As<MakeTuple>(args[1]);
  CHECK(offsets_tuple) << "The operator " << op_name
                       << " requires second argument to be a tuple (offsets), but got "
                       << args[1]->GetType()->TypeName();

  // Third argument must be TupleType (shapes)
  auto shapes_tuple = As<MakeTuple>(args[2]);
  CHECK(shapes_tuple) << "The operator " << op_name
                      << " requires third argument to be a tuple (shapes), but got "
                      << args[2]->GetType()->TypeName();

  // Verify offsets and shapes have same number of dimensions
  CHECK(offsets_tuple->elements_.size() == shapes_tuple->elements_.size())
      << "The operator " << op_name
      << " requires offsets and shapes to have same number of dimensions, but got "
      << offsets_tuple->elements_.size() << " offsets and " << shapes_tuple->elements_.size() << " shapes";

  CHECK(shapes_tuple->elements_.size() > 0)
      << "The operator " << op_name << " requires at least one dimension, but got empty shapes tuple";

  // Fourth argument must be the output tensor
  auto output_tensor_type = As<TensorType>(args[3]->GetType());
  CHECK(output_tensor_type) << "The operator " << op_name
                            << " requires fourth argument to be a TensorType, but got "
                            << args[3]->GetType()->TypeName();

  // store returns the output tensor (same type)
  return output_tensor_type;
}

TypePtr DeduceBlockMoveType(const std::vector<ExprPtr>& args,
                            const std::vector<std::pair<std::string, std::any>>& kwargs,
                            const std::string& op_name) {
  // Validate args: expect exactly 1 argument (tile)
  CHECK(args.size() == 1) << "The operator " << op_name << " requires 1 argument, but got " << args.size();

  // Validate first argument is TileType
  auto tile_type = As<TileType>(args[0]->GetType());
  CHECK(tile_type) << "The operator " << op_name << " requires first argument to be a TileType, but got "
                   << args[0]->GetType()->TypeName();

  // Extract transpose attribute (default: false)
  bool transpose = GetKwarg<bool>(kwargs, "transpose", false);

  // Determine output shape based on transpose flag
  const auto& input_shape = tile_type->shape_;
  std::vector<ExprPtr> output_shape;

  if (transpose && input_shape.size() == 2) {
    // Transpose: swap dimensions [H, W] -> [W, H]
    output_shape = {input_shape[1], input_shape[0]};
  } else {
    // No transpose: keep original shape
    output_shape = input_shape;
  }

  // Return TileType with computed shape and same dtype (no explicit MemRef)
  return std::make_shared<TileType>(output_shape, tile_type->dtype_);
}

TypePtr DeduceBlockAllocType(const std::vector<ExprPtr>& args,
                             const std::vector<std::pair<std::string, std::any>>& kwargs,
                             const std::string& op_name) {
  // alloc signature: (memory_space, addr, size, id)
  // Takes MemRef fields as arguments and returns MemRefType
  CHECK(args.size() == 4) << "The operator " << op_name << " requires exactly 4 arguments, but got "
                          << args.size();

  // Return MemRefType
  return GetMemRefType();
}

TypePtr DeduceBlockZerosType(const std::vector<ExprPtr>& args,
                             const std::vector<std::pair<std::string, std::any>>& kwargs,
                             const std::string& op_name) {
  // zeros signature: (dim1, dim2, ..., dimN) with dtype kwarg
  // Creates a zero-initialized tile with arbitrary dimensions
  CHECK(args.size() >= 1) << "The operator " << op_name << " requires at least 1 dimension argument, but got "
                          << args.size();

  // Extract dtype from kwargs (required)
  int dtype_code = GetKwarg<int>(kwargs, "dtype");
  DataType dtype = static_cast<DataType>(dtype_code);

  // All arguments are shape dimensions
  std::vector<ExprPtr> shape;
  for (size_t i = 0; i < args.size(); ++i) {
    // Verify each dimension is a scalar type
    auto scalar_type = As<ScalarType>(args[i]->GetType());
    CHECK(scalar_type) << "The operator " << op_name << " requires dimension " << i
                       << " to be a scalar, but got " << args[i]->GetType()->TypeName();
    shape.push_back(args[i]);
  }

  // Return TileType with specified shape and dtype
  return std::make_shared<TileType>(shape, dtype);
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

REGISTER_OP("block.load")
    .set_op_category("BlockOp")
    .set_description("Copy data from tensor to unified buffer (tile)")
    .add_argument("tensor", "Source tensor (TensorType)")
    .add_argument("offsets", "Offsets in each dimension (TupleType of ScalarType)")
    .add_argument("shapes", "Shape of tile in each dimension (TupleType of ScalarType)")
    .set_attr<int>("target_memory")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceBlockLoadType(args, kwargs, "block.load");
    });

REGISTER_OP("block.store")
    .set_op_category("BlockOp")
    .set_description("Copy data from unified buffer (tile) to tensor")
    .add_argument("tile", "Source tile (TileType)")
    .add_argument("offsets", "Offsets in each dimension (TupleType of ScalarType)")
    .add_argument("shapes", "Shape of tile in each dimension (TupleType of ScalarType)")
    .add_argument("output_tensor", "Output tensor (TensorType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceBlockStoreType(args, kwargs, "block.store");
    });

REGISTER_OP("block.l0c_store")
    .set_op_category("BlockOp")
    .set_description("Copy data from L0C tile to GM tensor")
    .add_argument("tile", "Source tile (TileType)")
    .add_argument("offsets", "Offsets in each dimension (TupleType of ScalarType)")
    .add_argument("shapes", "Shape of tile in each dimension (TupleType of ScalarType)")
    .add_argument("output_tensor", "Output tensor (TensorType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceBlockStoreType(args, kwargs, "block.l0c_store");
    });

REGISTER_OP("block.move")
    .set_op_category("BlockOp")
    .set_description("Move tile to memory levels (UB/L1/L0A/L0B) with optional transpose")
    .add_argument("tile", "Input tile (TileType)")
    .set_attr<bool>("transpose")
    .set_attr<int>("target_memory")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceBlockMoveType(args, kwargs, "block.move");
    });

REGISTER_OP("block.alloc")
    .set_op_category("BlockOp")
    .set_description("Allocate memory for a MemRef object")
    .add_argument("memory_space", "Memory space (int enum value)")
    .add_argument("addr", "Starting address expression")
    .add_argument("size", "Size in bytes (scalar)")
    .add_argument("id", "MemRef ID (scalar)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceBlockAllocType(args, kwargs, "block.alloc");
    });

REGISTER_OP("block.zeros")
    .set_op_category("BlockOp")
    .set_description("Create a zero-initialized tile with specified shape and dtype")
    .set_pipe(PipeType::V)
    .add_argument("dims", "Shape dimensions (one or more scalars)")
    .set_attr<int>("dtype")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceBlockZerosType(args, kwargs, "block.zeros");
    });

}  // namespace ir
}  // namespace pypto
