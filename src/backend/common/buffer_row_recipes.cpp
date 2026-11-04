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

#include "pypto/backend/common/buffer_row_recipes.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "pypto/backend/common/buffer_type_support.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/tile_view_semantics.h"
#include "pypto/ir/type.h"

namespace pypto::backend {

const std::vector<BufferRowRecipe>& GetBufferRowRecipes() {
  static const std::vector<BufferRowRecipe> recipes = {
      {"tile.row_max", "buffer.row_max", "pto.trowmax", true},
      {"tile.row_sum", "buffer.row_sum", "pto.trowsum", true},
      {"tile.row_expand_sub", "buffer.row_expand_sub", "pto.trowexpandsub", false},
      {"tile.row_expand_div", "buffer.row_expand_div", "pto.trowexpanddiv", false},
  };
  return recipes;
}

const BufferRowRecipe* FindLogicalBufferRowRecipe(const std::string& name) {
  static const auto index = [] {
    std::unordered_map<std::string, const BufferRowRecipe*> result;
    for (const auto& recipe : GetBufferRowRecipes()) result.emplace(recipe.logical_op, &recipe);
    return result;
  }();
  const auto found = index.find(name);
  return found == index.end() ? nullptr : found->second;
}

const BufferRowRecipe* FindBufferRowRecipe(const std::string& name) {
  static const auto index = [] {
    std::unordered_map<std::string, const BufferRowRecipe*> result;
    for (const auto& recipe : GetBufferRowRecipes()) result.emplace(recipe.buffer_op, &recipe);
    return result;
  }();
  const auto found = index.find(name);
  return found == index.end() ? nullptr : found->second;
}

namespace {

// Return a contract diagnostic without throwing: the same predicate must serve
// whole-kernel selection before planning and validation of explicit Buffer IR.
const char* OperandError(const BufferRowRecipe& recipe, const std::vector<ir::BufferTypePtr>& types) {
  for (size_t i = 0; i < types.size(); ++i) {
    const auto& type = types[i];
    const bool column = i == (recipe.reduction ? 2 : 1);
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

ir::BufferTypePtr LogicalDescriptor(const ir::TypePtr& type) {
  const auto tile = ir::As<ir::TileType>(type);
  if (!tile || tile->shape_.size() != 2) return nullptr;
  const auto view = ir::tile_view_semantics::GetEffectiveTileView(*tile);
  if (!IsDenseVecBufferTile(tile, view) && !IsColumnVectorBufferTile(tile, view)) return nullptr;
  const auto offset = ir::As<ir::ConstInt>(view.start_offset);
  if (view.start_offset && (!offset || offset->value_ != 0)) return nullptr;
  std::vector<int64_t> shape;
  std::vector<int64_t> valid;
  const auto& extents = view.valid_shape.empty() ? tile->shape_ : view.valid_shape;
  if (extents.size() != 2) return nullptr;
  for (size_t i = 0; i < 2; ++i) {
    const auto dimension = ir::As<ir::ConstInt>(tile->shape_[i]);
    const auto extent = ir::As<ir::ConstInt>(extents[i]);
    if (!dimension || dimension->value_ <= 0 || !extent || extent->value_ <= 0 ||
        extent->value_ > dimension->value_) {
      return nullptr;
    }
    shape.push_back(dimension->value_);
    valid.push_back(extent->value_);
  }
  if (!view.stride.empty()) {
    const std::vector<int64_t> expected = view.blayout == ir::TileLayout::col_major
                                              ? std::vector<int64_t>{1, shape[0]}
                                              : std::vector<int64_t>{shape[1], 1};
    if (view.stride.size() != 2) return nullptr;
    for (size_t i = 0; i < 2; ++i) {
      const auto stride = ir::As<ir::ConstInt>(view.stride[i]);
      if (!stride || stride->value_ != expected[i]) return nullptr;
    }
  }
  return std::make_shared<ir::BufferType>(shape, tile->dtype_, ir::MemorySpace::Vec, valid, view.blayout);
}

}  // namespace

bool SupportsLogicalBufferRow(const BufferRowRecipe& recipe, const ir::CallPtr& call) {
  if (call->args_.size() != 2 || !call->kwargs_.empty()) return false;
  std::vector<ir::BufferTypePtr> types;
  for (const auto& arg : call->args_) types.push_back(LogicalDescriptor(arg->GetType()));
  types.push_back(LogicalDescriptor(call->GetType()));
  return OperandError(recipe, types) == nullptr;
}

void ValidateBufferRowOperands(const BufferRowRecipe& recipe, const std::vector<ir::ExprPtr>& args) {
  CHECK(args.size() == 3) << recipe.buffer_op << " requires two inputs and an explicit destination";
  std::vector<ir::BufferTypePtr> types;
  for (const auto& arg : args) {
    CHECK(arg) << recipe.buffer_op << " operands must not be null";
    types.push_back(ir::As<ir::BufferType>(arg->GetType()));
  }
  const auto* error = OperandError(recipe, types);
  CHECK(!error) << recipe.buffer_op << " " << error;
}

}  // namespace pypto::backend
