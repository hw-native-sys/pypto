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
#include <cstdint>
#include <limits>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pypto/backend/common/buffer_type_support.h"
#include "pypto/backend/common/gather_compare_scratch.h"
#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/type.h"

namespace pypto::backend {

const std::vector<BufferIndexedRecipe>& GetBufferIndexedRecipes() {
  static const std::vector<BufferIndexedRecipe> recipes = {
      {"tile.gather", "buffer.gather", "pto.tgather", false, 3},
      {"tile.scatter", "buffer.scatter", "pto.tscatter", true, 2},
      {"tile.gather_compare", "buffer.gather_compare", "pto.tgather", false, 4, true},
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

const char* CompareError(const std::vector<ir::BufferTypePtr>& types, const ir::ScalarTypePtr& threshold,
                         int mode, int offset) {
  for (const auto& type : types) {
    if (!IsAlignedDenseVecBuffer(type)) return "requires aligned dense static Vec buffers";
  }
  const auto& source = types[0];
  const auto& scratch = types[1];
  const auto& indices = types[2];
  const auto& counts = types[3];
  if ((source->dtype_ != DataType::FP16 && source->dtype_ != DataType::FP32 &&
       source->dtype_ != DataType::INT32) ||
      !threshold || threshold->dtype_ != source->dtype_ || (mode != 0 && mode != 4) ||
      (source->dtype_ == DataType::INT32 && mode != 0)) {
    return "requires FP16/FP32 EQ or GT, or INT32 EQ, with a matching scalar threshold";
  }
  if (source->valid_shape_ != source->shape_ || source->shape_[1] % (256 / source->dtype_.GetByte()) != 0) {
    return "requires full-valid source rows aligned to a native 256-byte comparison repeat";
  }
  const auto plan = PlanGatherCompareScratch(source->shape_[0], source->shape_[1], source->dtype_.GetByte());
  if (!plan || scratch->dtype_ != DataType::UINT8 || scratch->shape_[0] != 1 ||
      scratch->shape_ != scratch->valid_shape_ ||
      scratch->shape_[0] > std::numeric_limits<int64_t>::max() / scratch->shape_[1] ||
      scratch->shape_[0] * scratch->shape_[1] < plan->bytes ||
      (scratch->shape_[0] * scratch->shape_[1]) % plan->mask_cols != 0) {
    return "requires UINT8 workspace covering mask, indices and thresholds";
  }
  if (indices->dtype_ != DataType::INT32 || indices->shape_[0] != source->shape_[0] ||
      indices->shape_[1] < source->shape_[1] || indices->valid_shape_ != indices->shape_ ||
      counts->dtype_ != DataType::INT32 || counts->shape_[0] != 1 ||
      counts->shape_[1] != ((source->shape_[0] + 7) / 8) * 8 ||
      counts->valid_shape_ != std::vector<int64_t>{1, source->shape_[0]}) {
    return "requires ordered INT32 index and count destinations with complete index capacity";
  }
  if (offset < 0 || source->shape_[0] > (std::numeric_limits<int32_t>::max() - offset) / source->shape_[1]) {
    return "requires nonnegative offset and indices representable as INT32";
  }
  return nullptr;
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
  if (recipe.compare) {
    const auto results = ir::As<ir::TupleType>(call->GetType());
    if (call->args_.size() != 3 || !results || results->types_.size() != 2) return false;
    const std::vector<ir::BufferTypePtr> types = {StaticLogicalBufferDescriptor(call->args_[0]->GetType()),
                                                  StaticLogicalBufferDescriptor(call->args_[2]->GetType()),
                                                  StaticLogicalBufferDescriptor(results->types_[0]),
                                                  StaticLogicalBufferDescriptor(results->types_[1])};
    return !CompareError(types, ir::As<ir::ScalarType>(call->args_[1]->GetType()),
                         call->GetKwarg<int>("cmp_mode", -1), call->GetKwarg<int>("offset", 0));
  }
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
  if (recipe.compare) {
    CHECK(args.size() == 6) << recipe.buffer_op
                            << " requires source, threshold, mask, workspace and two destinations";
    for (const auto& arg : args) CHECK(arg) << recipe.buffer_op << " operands must not be null";
    const std::vector<ir::BufferTypePtr> types = {
        ir::As<ir::BufferType>(args[0]->GetType()), ir::As<ir::BufferType>(args[3]->GetType()),
        ir::As<ir::BufferType>(args[4]->GetType()), ir::As<ir::BufferType>(args[5]->GetType())};
    const auto* error =
        CompareError(types, ir::As<ir::ScalarType>(args[1]->GetType()),
                     ir::GetIntKwarg(kwargs, "cmp_mode", -1), ir::GetIntKwarg(kwargs, "offset", 0));
    CHECK(!error) << recipe.buffer_op << " " << error;
    const auto plan =
        PlanGatherCompareScratch(types[0]->shape_[0], types[0]->shape_[1], types[0]->dtype_.GetByte());
    INTERNAL_CHECK(plan.has_value()) << "Internal error: validated gather scratch plan is unavailable";
    const auto mask = ir::As<ir::BufferType>(args[2]->GetType());
    CHECK(mask && IsAlignedDenseVecBuffer(mask) && mask->dtype_ == DataType::UINT8 &&
          mask->shape_ == std::vector<int64_t>({plan->rows, plan->mask_cols}) &&
          mask->valid_shape_ == mask->shape_)
        << recipe.buffer_op << " requires the canonical mask descriptor";
    return;
  }
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
  if (recipe.compare) return {{0, 3}, {0, 4}, {0, 5}, {3, 4}, {3, 5}, {4, 5}};
  if (recipe.scatter) return {{0, 2}, {1, 2}};
  return {{0, 3}, {1, 3}, {2, 3}, {0, 2}, {1, 2}};
}

}  // namespace pypto::backend
