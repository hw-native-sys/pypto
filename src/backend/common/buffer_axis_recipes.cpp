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

#include "pypto/backend/common/buffer_axis_recipes.h"

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

#include "pypto/backend/common/buffer_type_support.h"
#include "pypto/backend/common/pto_native_ops.h"
#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/type.h"

namespace pypto::backend {

const std::vector<BufferAxisRecipe>& GetBufferAxisRecipes() {
  static const std::vector<BufferAxisRecipe> recipes = {
      {"tile.row_max", "buffer.row_max", GetPtoNativeOpSpec("tile.row_max").native_op, true},
      {"tile.row_sum", "buffer.row_sum", GetPtoNativeOpSpec("tile.row_sum").native_op, true},
      {"tile.row_min", "buffer.row_min", GetPtoNativeOpSpec("tile.row_min").native_op, true},
      {"tile.row_prod", "buffer.row_prod", GetPtoNativeOpSpec("tile.row_prod").native_op, true},
      {"tile.row_expand_add", "buffer.row_expand_add", GetPtoNativeOpSpec("tile.row_expand_add").native_op,
       false},
      {"tile.row_expand_sub", "buffer.row_expand_sub", GetPtoNativeOpSpec("tile.row_expand_sub").native_op,
       false},
      {"tile.row_expand_mul", "buffer.row_expand_mul", GetPtoNativeOpSpec("tile.row_expand_mul").native_op,
       false},
      {"tile.row_expand_div", "buffer.row_expand_div", GetPtoNativeOpSpec("tile.row_expand_div").native_op,
       false},
      {"tile.row_expand_max", "buffer.row_expand_max", GetPtoNativeOpSpec("tile.row_expand_max").native_op,
       false},
      {"tile.row_expand_min", "buffer.row_expand_min", GetPtoNativeOpSpec("tile.row_expand_min").native_op,
       false},
      {"tile.col_sum", "buffer.col_sum", GetPtoNativeOpSpec("tile.col_sum").native_op, true, 0},
      {"tile.col_max", "buffer.col_max", GetPtoNativeOpSpec("tile.col_max").native_op, true, 0},
      {"tile.col_min", "buffer.col_min", GetPtoNativeOpSpec("tile.col_min").native_op, true, 0},
      {"tile.col_prod", "buffer.col_prod", GetPtoNativeOpSpec("tile.col_prod").native_op, true, 0},
      {"tile.col_expand_add", "buffer.col_expand_add", GetPtoNativeOpSpec("tile.col_expand_add").native_op,
       false, 0},
      {"tile.col_expand_sub", "buffer.col_expand_sub", GetPtoNativeOpSpec("tile.col_expand_sub").native_op,
       false, 0},
      {"tile.col_expand_mul", "buffer.col_expand_mul", GetPtoNativeOpSpec("tile.col_expand_mul").native_op,
       false, 0},
      {"tile.col_expand_div", "buffer.col_expand_div", GetPtoNativeOpSpec("tile.col_expand_div").native_op,
       false, 0},
      {"tile.col_expand_max", "buffer.col_expand_max", GetPtoNativeOpSpec("tile.col_expand_max").native_op,
       false, 0},
      {"tile.col_expand_min", "buffer.col_expand_min", GetPtoNativeOpSpec("tile.col_expand_min").native_op,
       false, 0},
  };
  return recipes;
}

const BufferAxisRecipe* FindLogicalBufferAxisRecipe(const std::string& name) {
  static const auto index = [] {
    std::unordered_map<std::string, const BufferAxisRecipe*> result;
    for (const auto& recipe : GetBufferAxisRecipes()) result.emplace(recipe.logical_op, &recipe);
    return result;
  }();
  const auto found = index.find(name);
  return found == index.end() ? nullptr : found->second;
}

const BufferAxisRecipe* FindBufferAxisRecipe(const std::string& name) {
  static const auto index = [] {
    std::unordered_map<std::string, const BufferAxisRecipe*> result;
    for (const auto& recipe : GetBufferAxisRecipes()) result.emplace(recipe.buffer_op, &recipe);
    return result;
  }();
  const auto found = index.find(name);
  return found == index.end() ? nullptr : found->second;
}

namespace {

// Return a contract diagnostic without throwing: the same predicate must serve
// whole-kernel selection before planning and validation of explicit Buffer IR.
const char* OperandError(const BufferAxisRecipe& recipe, const std::vector<ir::BufferTypePtr>& types) {
  for (size_t i = 0; i < types.size(); ++i) {
    const auto& type = types[i];
    const bool column = recipe.axis == 1 && i == (recipe.reduction ? 2 : 1);
    if (column) {
      if (!IsColumnVectorBuffer(type)) return "requires a static aligned FP32 column vector";
    } else if (!type || type->shape_.size() != 2 || type->dtype_ != DataType::FP32 ||
               type->memory_space_ != ir::MemorySpace::Vec || type->blayout_ != ir::TileLayout::row_major ||
               type->slayout_ != ir::TileLayout::none_box || type->fractal_ != 512 ||
               type->pad_ != ir::PadValue::null || type->compact_ != ir::CompactMode::null ||
               type->shape_[1] % 8 != 0) {
      return "requires aligned rank-2 row-major Vec FP32 operands";
    }
    if (type->valid_shape_[0] <= 0 || type->valid_shape_[1] <= 0) {
      return "requires nonempty static valid extents";
    }
  }
  const auto& source = types[0];
  if (recipe.axis == 0) {
    const auto& row = types[1];
    if (row->shape_[0] != 1 || row->valid_shape_[0] != 1 || row->shape_[1] != source->shape_[1] ||
        row->valid_shape_[1] != source->valid_shape_[1]) {
      return "requires a single row preserving physical and valid columns";
    }
    if (!recipe.reduction &&
        (types[2]->shape_ != source->shape_ || types[2]->valid_shape_ != source->valid_shape_)) {
      return "requires the broadcast destination to preserve source physical and valid shape";
    }
    return nullptr;
  }
  const auto& column = types[recipe.reduction ? 2 : 1];
  if (column->shape_[0] != source->shape_[0] || column->valid_shape_[0] != source->valid_shape_[0]) {
    return "requires the column vector to preserve physical and valid rows";
  }
  if (recipe.reduction) {
    const auto& workspace = types[1];
    if (workspace->shape_[0] < source->shape_[0] || workspace->shape_[1] < source->shape_[1] ||
        workspace->valid_shape_ != workspace->shape_) {
      return "requires full-valid scratch covering the source physical shape";
    }
  } else if (types[2]->shape_ != source->shape_ || types[2]->valid_shape_ != source->valid_shape_) {
    return "requires the broadcast destination to preserve source physical and valid shape";
  }
  return nullptr;
}

}  // namespace

bool SupportsLogicalBufferAxis(const BufferAxisRecipe& recipe, const ir::CallPtr& call) {
  // Binary-tree col_sum and optional-scratch/packed row_expand_add have
  // separate storage contracts and remain on Tile.
  if (call->args_.size() != recipe.InputCount() || !call->kwargs_.empty()) return false;
  std::vector<ir::BufferTypePtr> types;
  for (const auto& arg : call->args_) types.push_back(StaticLogicalBufferDescriptor(arg->GetType()));
  types.push_back(StaticLogicalBufferDescriptor(call->GetType()));
  return OperandError(recipe, types) == nullptr;
}

void ValidateBufferAxisOperands(const BufferAxisRecipe& recipe, const std::vector<ir::ExprPtr>& args) {
  CHECK(args.size() == recipe.InputCount() + 1)
      << recipe.buffer_op
      << (recipe.InputCount() == 2 ? " requires two inputs and an explicit destination"
                                   : " requires one input and an explicit destination");
  std::vector<ir::BufferTypePtr> types;
  for (const auto& arg : args) {
    CHECK(arg) << recipe.buffer_op << " operands must not be null";
    types.push_back(ir::As<ir::BufferType>(arg->GetType()));
  }
  const auto* error = OperandError(recipe, types);
  CHECK(!error) << recipe.buffer_op << " " << error;
}

}  // namespace pypto::backend
