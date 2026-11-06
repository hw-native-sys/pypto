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

#include "pypto/backend/common/buffer_indexed_recipes.h"

#include <any>
#include <cstddef>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pypto/backend/common/buffer_type_support.h"
#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/type.h"

namespace pypto::backend {

const std::vector<BufferIndexedRecipe>& GetBufferIndexedRecipes() {
  static const std::vector<BufferIndexedRecipe> recipes = {
      {"tile.gather", "buffer.gather", "pto.tgather", false, 3},
      {"tile.scatter", "buffer.scatter", "pto.tscatter", true, 2},
  };
  return recipes;
}

const BufferIndexedRecipe* FindLogicalBufferIndexedRecipe(const std::string& name) {
  static const auto index = [] {
    std::unordered_map<std::string, const BufferIndexedRecipe*> result;
    for (const auto& recipe : GetBufferIndexedRecipes()) result.emplace(recipe.logical_op, &recipe);
    return result;
  }();
  const auto found = index.find(name);
  return found == index.end() ? nullptr : found->second;
}

const BufferIndexedRecipe* FindBufferIndexedRecipe(const std::string& name) {
  static const auto index = [] {
    std::unordered_map<std::string, const BufferIndexedRecipe*> result;
    for (const auto& recipe : GetBufferIndexedRecipes()) result.emplace(recipe.buffer_op, &recipe);
    return result;
  }();
  const auto found = index.find(name);
  return found == index.end() ? nullptr : found->second;
}

namespace {
bool Dense(const ir::BufferTypePtr& type) {
  if (!type ||
      (type->dtype_ != DataType::FP16 && type->dtype_ != DataType::FP32 && type->dtype_ != DataType::INT32)) {
    return false;
  }
  return IsAlignedDenseVecBuffer(type);
}

bool Geometry(const ir::BufferTypePtr& a, const ir::BufferTypePtr& b) {
  return a->shape_ == b->shape_ && a->valid_shape_ == b->valid_shape_;
}

const char* OperandError(const BufferIndexedRecipe& recipe, const std::vector<ir::BufferTypePtr>& types) {
  for (const auto& type : types) {
    if (!Dense(type)) return "requires aligned dense rank-2 Vec buffers with positive static valid extents";
  }
  const auto& source = types[0];
  const auto& indices = types[1];
  const auto& destination = types.back();
  if (indices->dtype_ != DataType::INT32 || destination->dtype_ != source->dtype_) {
    return "requires INT32 indices and matching source/destination dtypes";
  }
  if (recipe.scatter) {
    // TSCATTER zeros the physical destination, including padding. Require a
    // full-valid destination so its write effect covers every byte it changes.
    if (source->dtype_ != DataType::FP32 || !Geometry(source, indices) ||
        destination->valid_shape_ != destination->shape_) {
      return "requires FP32 values, matching source/index extents and a full-valid destination";
    }
  } else {
    // Logical gather currently deduces a full-valid result from the index
    // capacity. Do not read undefined index tails or table padding by opening
    // partial descriptors before a separate valid-region contract exists.
    for (const auto& type : types) {
      if (type->valid_shape_ != type->shape_) return "requires full-valid gather operands";
    }
    if (!Geometry(indices, destination) || !Geometry(indices, types[2]) ||
        types[2]->dtype_ != DataType::INT32) {
      return "requires destination and INT32 scratch shapes matching the indices";
    }
  }
  return nullptr;
}
}  // namespace

bool SupportsLogicalBufferIndexed(const BufferIndexedRecipe& recipe, const ir::CallPtr& call) {
  if (call->args_.size() != 3 || !call->kwargs_.empty()) return false;
  std::vector<ir::BufferTypePtr> types;
  for (size_t i = recipe.scatter ? 1 : 0; i < call->args_.size(); ++i) {
    types.push_back(StaticLogicalBufferDescriptor(call->args_[i]->GetType()));
  }
  types.push_back(StaticLogicalBufferDescriptor(call->GetType()));
  if (OperandError(recipe, types)) return false;
  if (recipe.scatter) {
    const auto base = StaticLogicalBufferDescriptor(call->args_[0]->GetType());
    return base && Geometry(base, types.back()) && base->dtype_ == types.back()->dtype_;
  }
  return true;
}

void ValidateBufferIndexedOperands(const BufferIndexedRecipe& recipe, const std::vector<ir::ExprPtr>& args,
                                   const std::vector<std::pair<std::string, std::any>>& kwargs) {
  CHECK(args.size() == recipe.input_count + 1)
      << recipe.buffer_op << " requires explicit inputs and destination";
  CHECK(kwargs.empty()) << recipe.buffer_op << " does not accept attributes";
  std::vector<ir::BufferTypePtr> types;
  for (const auto& arg : args) {
    CHECK(arg) << recipe.buffer_op << " operands must not be null";
    types.push_back(ir::As<ir::BufferType>(arg->GetType()));
  }
  const auto* error = OperandError(recipe, types);
  CHECK(!error) << recipe.buffer_op << " " << error;
}

std::vector<std::pair<size_t, size_t>> BufferIndexedDisjointPairs(const BufferIndexedRecipe& recipe) {
  // TGATHER can overwrite its destination while still reading the table or
  // indices; its FP16 path also writes scratch. TSCATTER clears dst before
  // reading src/indices. Read/read aliases remain legal.
  if (recipe.scatter) return {{0, 2}, {1, 2}};
  return {{0, 3}, {1, 3}, {2, 3}, {0, 2}, {1, 2}};
}

}  // namespace pypto::backend
