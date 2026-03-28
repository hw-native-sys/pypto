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
 * @brief Memory tile operations (get_block_idx, load, store)
 *
 * This file implements memory operations for tile-level programming.
 * These operations handle data movement between tensors and unified buffers (tiles).
 */

#include <any>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/core/error.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/type.h"

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

TypePtr DeduceTileGetBlockIdxType(const std::vector<ExprPtr>& args,
                                  const std::vector<std::pair<std::string, std::any>>& kwargs,
                                  const std::string& op_name) {
  CHECK(args.size() == 0) << "The operator " << op_name << " requires no arguments, but got " << args.size();

  // get_block_idx returns UINT64 scalar
  return std::make_shared<ScalarType>(DataType::UINT64);
}

TypePtr DeduceTileLoadType(const std::vector<ExprPtr>& args,
                           const std::vector<std::pair<std::string, std::any>>& kwargs,
                           const std::string& op_name) {
  // load signature: (tensor, offsets_tuple, shapes_tuple, valid_shapes_tuple)
  CHECK(args.size() == 4) << "The operator " << op_name
                          << " requires 4 arguments (tensor, offsets, shapes, valid_shapes), but got "
                          << args.size();

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

  // Fourth argument must be TupleType (valid_shapes)
  auto valid_shapes_tuple = As<MakeTuple>(args[3]);
  CHECK(valid_shapes_tuple) << "The operator " << op_name
                            << " requires fourth argument to be a tuple (valid shapes), but got "
                            << args[3]->GetType()->TypeName();

  // Verify offsets, shapes and valid_shapes have same number of dimensions
  CHECK(offsets_tuple->elements_.size() == shapes_tuple->elements_.size())
      << "The operator " << op_name
      << " requires offsets and shapes to have same number of dimensions, but got "
      << offsets_tuple->elements_.size() << " offsets and " << shapes_tuple->elements_.size() << " shapes";
  CHECK(valid_shapes_tuple->elements_.size() == shapes_tuple->elements_.size())
      << "The operator " << op_name
      << " requires valid_shapes and shapes to have same number of dimensions, but got "
      << valid_shapes_tuple->elements_.size() << " valid_shapes and " << shapes_tuple->elements_.size()
      << " shapes";
  CHECK(shapes_tuple->elements_.size() > 0)
      << "The operator " << op_name << " requires at least one dimension, but got empty shapes tuple";

  auto target_memory = GetKwarg<MemorySpace>(kwargs, "target_memory");
  bool transpose = GetKwarg<bool>(kwargs, "transpose", false);

  // Transpose is only supported when loading to L1 (Mat)
  CHECK(!transpose || target_memory == MemorySpace::Mat)
      << "The operator " << op_name
      << " only supports transpose=true when target_memory is Mat (L1), but got "
      << static_cast<int>(target_memory);

  CHECK(!transpose || shapes_tuple->elements_.size() == 2)
      << "The operator " << op_name << " only supports transpose=true for 2D loads, but got "
      << shapes_tuple->elements_.size() << "D";

  // Nz/Zn for transpose false/true
  TileView tile_view;
  if (target_memory == MemorySpace::Mat) {
    tile_view.blayout = TileLayout::col_major;
    tile_view.slayout = TileLayout::row_major;
    if (transpose) {
      std::swap(tile_view.blayout, tile_view.slayout);
    }
  } else if (auto last_dim = As<ConstInt>(shapes_tuple->elements_.back());
             last_dim && last_dim->value_ == 1) {
    tile_view.blayout = TileLayout::col_major;
  }

  // Build tile shape from shapes tuple
  std::vector<ExprPtr> tile_shape;
  for (const auto& shape_expr : shapes_tuple->elements_) {
    tile_shape.push_back(shape_expr);
  }

  tile_view.valid_shape = valid_shapes_tuple->elements_;

  // Return TileType with same dtype as tensor and TileView containing valid_shape
  return std::make_shared<TileType>(tile_shape, tensor_type->dtype_, std::nullopt, tile_view);
}

TypePtr DeduceTileStoreType(const std::vector<ExprPtr>& args,
                            const std::vector<std::pair<std::string, std::any>>& kwargs,
                            const std::string& op_name) {
  // store signature: (tile, offsets_tuple, output_tensor[, shapes_tuple])
  // shapes_tuple is an optional 4th argument injected by FlattenTileNdTo2D
  // for ND tensors to carry the original partition shape for codegen.
  CHECK(args.size() == 3 || args.size() == 4)
      << "The operator " << op_name
      << " requires 3 or 4 arguments (tile, offsets, output_tensor[, shapes]), but got " << args.size();

  // First argument must be TileType
  auto tile_type = As<TileType>(args[0]->GetType());
  CHECK(tile_type) << "The operator " << op_name << " requires first argument to be a TileType, but got "
                   << args[0]->GetType()->TypeName();

  // Second argument must be TupleType (offsets)
  auto offsets_tuple = As<MakeTuple>(args[1]);
  CHECK(offsets_tuple) << "The operator " << op_name
                       << " requires second argument to be a tuple (offsets), but got "
                       << args[1]->GetType()->TypeName();

  // Third argument must be the output tensor
  auto output_tensor_type = As<TensorType>(args[2]->GetType());
  CHECK(output_tensor_type) << "The operator " << op_name
                            << " requires third argument to be a TensorType, but got "
                            << args[2]->GetType()->TypeName();

  // Optional fourth argument (when 4 args total) must be a shapes tuple
  if (args.size() == 4) {
    auto shapes_tuple = As<MakeTuple>(args[3]);
    CHECK(shapes_tuple) << "The operator " << op_name
                        << " requires optional 4th argument to be a shapes tuple (MakeTuple)";
    CHECK(!shapes_tuple->elements_.empty())
        << "The operator " << op_name << " requires non-empty shapes tuple when provided";
    CHECK(shapes_tuple->elements_.size() == offsets_tuple->elements_.size())
        << "The operator " << op_name
        << " requires shapes and offsets to have the same number of dimensions, but got "
        << shapes_tuple->elements_.size() << " shapes and " << offsets_tuple->elements_.size() << " offsets";
  }

  // store returns the output tensor (same type)
  return output_tensor_type;
}

TypePtr DeduceTileMoveType(const std::vector<ExprPtr>& args,
                           const std::vector<std::pair<std::string, std::any>>& kwargs,
                           const std::string& op_name) {
  // Validate args: expect exactly 1 argument (tile)
  CHECK(args.size() == 1) << "The operator " << op_name << " requires 1 argument, but got " << args.size();

  // Validate first argument is TileType
  auto tile_type = As<TileType>(args[0]->GetType());
  CHECK(tile_type) << "The operator " << op_name << " requires first argument to be a TileType, but got "
                   << args[0]->GetType()->TypeName();

  // Extract MemorySpace
  MemorySpace space = GetKwarg<MemorySpace>(kwargs, "target_memory");

  const auto& input_shape = tile_type->shape_;

  TileView tile_view;

  // Default: retain source tile's layout
  if (tile_type->tile_view_) {
    tile_view.blayout = tile_type->tile_view_->blayout;
    tile_view.slayout = tile_type->tile_view_->slayout;
  }

  // Hardcoded layout for Left/Right (hardware requirements)
  if (space == MemorySpace::Left) {
    tile_view.blayout = TileLayout::col_major;  // L0A requires ColMajor block layout for TMATMUL
    tile_view.slayout = TileLayout::row_major;
  } else if (space == MemorySpace::Right) {
    tile_view.blayout = TileLayout::row_major;
    tile_view.slayout = TileLayout::col_major;
  }

  // Explicit kwargs override everything
  tile_view.blayout = GetKwarg<TileLayout>(kwargs, "blayout", tile_view.blayout);
  tile_view.slayout = GetKwarg<TileLayout>(kwargs, "slayout", tile_view.slayout);

  // Keep original shape
  std::vector<ExprPtr> output_shape = input_shape;

  // Preserve input valid_shape (may be narrower than shape_)
  auto input_valid_shape = (tile_type->tile_view_ && !tile_type->tile_view_->valid_shape.empty())
                               ? tile_type->tile_view_->valid_shape
                               : input_shape;
  tile_view.valid_shape = input_valid_shape;

  // Preserve pad value from input tile
  if (tile_type->tile_view_ && tile_type->tile_view_->pad != PadValue::null) {
    tile_view.pad = tile_type->tile_view_->pad;
  }

  // Return TileType with computed shape and same dtype (no explicit MemRef)
  return std::make_shared<TileType>(output_shape, tile_type->dtype_, std::nullopt, tile_view);
}

TypePtr DeduceTileAllocType(const std::vector<ExprPtr>& args,
                            const std::vector<std::pair<std::string, std::any>>& kwargs,
                            const std::string& op_name) {
  // alloc signature: (memory_space, addr, size, id)
  // Takes MemRef fields as arguments and returns MemRefType
  CHECK(args.size() == 4) << "The operator " << op_name << " requires exactly 4 arguments, but got "
                          << args.size();

  // Return MemRefType
  return GetMemRefType();
}

TypePtr DeduceTileCreateTileType(const std::vector<ExprPtr>& args,
                                 const std::vector<std::pair<std::string, std::any>>& kwargs,
                                 const std::string& op_name) {
  // create_tile signature: (shape)
  // TileType requires static compile-time constant shapes
  CHECK(args.size() == 1) << "The operator " << op_name << " requires exactly 1 argument, but got "
                          << args.size();

  // Extract dtype attribute
  DataType dtype = GetKwarg<DataType>(kwargs, "dtype");

  // First argument must be MakeTuple with static ConstInt elements
  auto make_tuple = As<MakeTuple>(args[0]);
  CHECK(make_tuple)
      << "The operator " << op_name
      << " requires first argument to be a MakeTuple expression with static shape values, but got "
      << args[0]->TypeName();

  // Validate all elements are ConstInt (static compile-time constants)
  std::vector<ExprPtr> tile_shape;
  tile_shape.reserve(make_tuple->elements_.size());

  for (size_t i = 0; i < make_tuple->elements_.size(); ++i) {
    auto const_int = As<ConstInt>(make_tuple->elements_[i]);
    CHECK(const_int) << "The operator " << op_name << " shape element " << i
                     << " must be a compile-time constant (ConstInt), but got "
                     << make_tuple->elements_[i]->TypeName();
    CHECK(const_int->value_ > 0) << "The operator " << op_name << " shape element " << i
                                 << " must be positive, got " << const_int->value_;
    tile_shape.push_back(make_tuple->elements_[i]);
  }

  CHECK(!tile_shape.empty()) << "The operator " << op_name << " requires non-empty shape";

  // Return TileType with the static shape and dtype
  TileView tile_view;
  if (tile_shape.size() == 2) {
    auto rows_const = As<ConstInt>(tile_shape[0]);
    auto cols_const = As<ConstInt>(tile_shape[1]);
    if (rows_const && cols_const && rows_const->value_ > 1 && cols_const->value_ == 1) {
      tile_view.blayout = TileLayout::col_major;
    }
  }
  tile_view.valid_shape = tile_shape;
  return std::make_shared<TileType>(tile_shape, dtype, std::nullopt, tile_view);
}

TypePtr DeduceTileFullType(const std::vector<ExprPtr>& args,
                           const std::vector<std::pair<std::string, std::any>>& kwargs,
                           const std::string& op_name) {
  // tile.full signature: (shape, value)
  CHECK(args.size() == 2) << "The operator " << op_name << " requires exactly 2 arguments, but got "
                          << args.size();

  // Extract dtype attribute
  DataType dtype = GetKwarg<DataType>(kwargs, "dtype");

  // First argument must be MakeTuple with static ConstInt elements
  auto make_tuple = As<MakeTuple>(args[0]);
  CHECK(make_tuple)
      << "The operator " << op_name
      << " requires first argument to be a MakeTuple expression with static shape values, but got "
      << args[0]->TypeName();

  // Validate all elements are ConstInt (static compile-time constants)
  std::vector<ExprPtr> tile_shape;
  tile_shape.reserve(make_tuple->elements_.size());

  for (size_t i = 0; i < make_tuple->elements_.size(); ++i) {
    auto const_int = As<ConstInt>(make_tuple->elements_[i]);
    CHECK(const_int) << "The operator " << op_name << " shape element " << i
                     << " must be a compile-time constant (ConstInt), but got "
                     << make_tuple->elements_[i]->TypeName();
    CHECK(const_int->value_ > 0) << "The operator " << op_name << " shape element " << i
                                 << " must be positive, got " << const_int->value_;
    tile_shape.push_back(make_tuple->elements_[i]);
  }

  CHECK(!tile_shape.empty()) << "The operator " << op_name << " requires non-empty shape";

  // Second argument must be ConstInt or ConstFloat
  CHECK(As<ConstInt>(args[1]) || As<ConstFloat>(args[1]))
      << "The operator " << op_name
      << " requires second argument to be a constant value (ConstInt or ConstFloat), but got "
      << args[1]->TypeName();

  // Return TileType with the static shape and dtype
  TileView tile_view;
  tile_view.valid_shape = tile_shape;
  return std::make_shared<TileType>(tile_shape, dtype, std::nullopt, tile_view);
}

TypePtr DeduceTileReadType(const std::vector<ExprPtr>& args,
                           const std::vector<std::pair<std::string, std::any>>& kwargs,
                           const std::string& op_name) {
  // tile.read: Read a scalar value from a tile at given indices
  // Args: (tile, indices_tuple)
  // Returns: ScalarType with tile's element dtype
  CHECK(args.size() == 2) << "tile.read requires exactly 2 arguments (tile, indices), but got "
                          << args.size();

  // First argument must be TileType
  auto tile_type = As<TileType>(args[0]->GetType());
  CHECK(tile_type) << "tile.read requires first argument to be a TileType, but got "
                   << args[0]->GetType()->TypeName();

  // Second argument must be TupleType (indices)
  auto indices_type = As<TupleType>(args[1]->GetType());
  CHECK(indices_type) << "tile.read requires indices to be TupleType, but got "
                      << args[1]->GetType()->TypeName();

  // Validate indices count matches tile rank
  CHECK(indices_type->types_.size() == tile_type->shape_.size())
      << "tile.read indices count (" << indices_type->types_.size() << ") must match tile rank ("
      << tile_type->shape_.size() << ")";

  // Validate all index elements are ScalarType with integer dtype
  for (size_t i = 0; i < indices_type->types_.size(); ++i) {
    auto scalar_type = As<ScalarType>(indices_type->types_[i]);
    CHECK(scalar_type) << "tile.read index element " << i << " must be ScalarType, but got "
                       << indices_type->types_[i]->TypeName();
    CHECK(scalar_type->dtype_.IsInt())
        << "tile.read index element " << i << " must have integer dtype, but got "
        << scalar_type->dtype_.ToString();
  }

  return std::make_shared<ScalarType>(tile_type->dtype_);
}

TypePtr DeduceTileWriteType(const std::vector<ExprPtr>& args,
                            const std::vector<std::pair<std::string, std::any>>& kwargs,
                            const std::string& op_name) {
  // tile.write: Write a scalar value into a tile at given indices
  // Args: (tile, indices_tuple, value)
  // Returns: TileType (the destination tile, for chaining)
  CHECK(args.size() == 3) << "tile.write requires exactly 3 arguments (tile, indices, value), but got "
                          << args.size();

  auto tile_type = As<TileType>(args[0]->GetType());
  CHECK(tile_type) << "tile.write requires first argument to be a TileType, but got "
                   << args[0]->GetType()->TypeName();

  auto indices_type = As<TupleType>(args[1]->GetType());
  CHECK(indices_type) << "tile.write requires indices to be TupleType, but got "
                      << args[1]->GetType()->TypeName();

  CHECK(indices_type->types_.size() == tile_type->shape_.size())
      << "tile.write indices count (" << indices_type->types_.size() << ") must match tile rank ("
      << tile_type->shape_.size() << ")";

  for (size_t i = 0; i < indices_type->types_.size(); ++i) {
    auto scalar_type = As<ScalarType>(indices_type->types_[i]);
    CHECK(scalar_type) << "tile.write index element " << i << " must be ScalarType, but got "
                       << indices_type->types_[i]->TypeName();
    CHECK(scalar_type->dtype_.IsInt())
        << "tile.write index element " << i << " must have integer dtype, but got "
        << scalar_type->dtype_.ToString();
  }

  auto value_type = As<ScalarType>(args[2]->GetType());
  CHECK(value_type) << "tile.write requires third argument (value) to be a ScalarType, but got "
                    << args[2]->GetType()->TypeName();

  CHECK(value_type->dtype_ == tile_type->dtype_)
      << "tile.write requires value dtype to match tile dtype, but got value dtype "
      << value_type->dtype_.ToString() << " and tile dtype " << tile_type->dtype_.ToString();

  return args[0]->GetType();
}

REGISTER_OP("tile.write")
    .set_op_category("TileOp")
    .set_description("Write a scalar value into a tile at given indices")
    .add_argument("tile", "Destination tile (TileType)")
    .add_argument("indices", "Index dimensions (TupleType of ScalarType)")
    .add_argument("value", "Scalar value to write (ScalarType)")
    .set_input_memory(0, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileWriteType(args, kwargs, "tile.write");
    });

// ============================================================================
// Registration Function for Block Memory Operations
// ============================================================================

REGISTER_OP("tile.get_block_idx")
    .set_op_category("TileOp")
    .set_description("Get the current block index")
    .no_argument()
    .no_memory_spec()
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileGetBlockIdxType(args, kwargs, "tile.get_block_idx");
    });

REGISTER_OP("tile.read")
    .set_op_category("TileOp")
    .set_description("Read a scalar value from a tile at given indices")
    .add_argument("tile", "Input tile (TileType)")
    .add_argument("indices", "Index dimensions (TupleType of ScalarType)")
    .set_input_memory(0, MemorySpace::Vec)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileReadType(args, kwargs, "tile.read");
    });

REGISTER_OP("tile.create")
    .set_op_category("TileOp")
    .set_description("Create a tile")
    .add_argument("shape", "Shape dimensions (TupleType of ScalarType(INT64))")
    .set_attr<DataType>("dtype")
    .set_attr<MemorySpace>("target_memory")
    .set_output_memory_from_kwarg("target_memory", MemorySpace::Vec)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileCreateTileType(args, kwargs, "tile.create");
    });

REGISTER_OP("tile.load")
    .set_op_category("TileOp")
    .set_description("Copy data from tensor to unified buffer (tile)")
    .add_argument("tensor", "Source tensor (TensorType)")
    .add_argument("offsets", "Offsets in each dimension (TupleType of ScalarType)")
    .add_argument("shapes", "Shape of tile in each dimension (TupleType of ScalarType)")
    .add_argument("valid_shapes", "Valid shape of tile in each dimension (TupleType of ScalarType). ")
    .set_attr<MemorySpace>("target_memory")
    .set_attr<bool>("transpose")
    .set_output_memory_from_kwarg("target_memory", MemorySpace::Vec)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileLoadType(args, kwargs, "tile.load");
    });

REGISTER_OP("tile.store")
    .set_op_category("TileOp")
    .set_description("Copy data from unified buffer (tile) to tensor")
    .add_argument("tile", "Source tile (TileType)")
    .add_argument("offsets", "Offsets in each dimension (TupleType of ScalarType)")
    .add_argument("output_tensor", "Output tensor (TensorType)")
    .add_argument("shapes",
                  "Optional ND partition shape (TupleType). "
                  "Injected by FlattenTileNdTo2D for ND tensors.")
    .set_input_memory(0, {MemorySpace::Vec, MemorySpace::Acc})
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileStoreType(args, kwargs, "tile.store");
    });

REGISTER_OP("tile.move")
    .set_op_category("TileOp")
    .set_description("Move tile between memory levels (Vec/Mat/Left/Right)")
    .add_argument("tile", "Input tile (TileType)")
    .set_attr<MemorySpace>("target_memory")
    .set_attr<TileLayout>("blayout")
    .set_attr<TileLayout>("slayout")
    .set_output_memory_from_kwarg("target_memory", MemorySpace::Vec)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileMoveType(args, kwargs, "tile.move");
    });

REGISTER_OP("tile.alloc")
    .set_op_category("TileOp")
    .set_description("Allocate memory for a MemRef object")
    .add_argument("memory_space", "Memory space (int enum value)")
    .add_argument("addr", "Starting address expression")
    .add_argument("size", "Size in bytes (scalar)")
    .add_argument("id", "MemRef ID (scalar)")
    .no_memory_spec()
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileAllocType(args, kwargs, "tile.alloc");
    });

REGISTER_OP("tile.full")
    .set_op_category("TileOp")
    .set_description("Create a tile of specified shape and filling value in UB")
    .add_argument("shape", "Shape dimensions (TupleType of ScalarType(INT64))")
    .add_argument("value", "Filling value (ConstInt or ConstFloat)")
    .set_attr<DataType>("dtype")
    .set_output_memory(MemorySpace::Vec)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileFullType(args, kwargs, "tile.full");
    });

}  // namespace ir
}  // namespace pypto
