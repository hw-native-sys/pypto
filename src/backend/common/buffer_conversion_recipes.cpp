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

#include "pypto/backend/common/buffer_conversion_recipes.h"

#include <any>
#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pypto/backend/common/buffer_type_support.h"
#include "pypto/core/any_cast.h"
#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/type.h"

namespace pypto::backend {

const std::vector<BufferConversionRecipe>& GetBufferConversionRecipes() {
  using K = BufferConversionKind;
  static const std::vector<BufferConversionRecipe> recipes = {
      {"tile.cast", "buffer.cast", "pto.tcvt", K::Cast, 1},
      {"tile.cmp", "buffer.cmp", "pto.tcmp", K::Compare, 2},
      {"tile.cmps", "buffer.cmps", "pto.tcmps", K::CompareScalar, 2, 1},
      {"tile.sel", "buffer.sel", "pto.tsel", K::Select, 4, -1, 3},
      {"tile.sels", "buffer.sels", "pto.tsels", K::SelectScalar, 4, 3, 2},
  };
  return recipes;
}

const BufferConversionRecipe* FindLogicalBufferConversionRecipe(const std::string& name) {
  static const auto index = [] {
    std::unordered_map<std::string, const BufferConversionRecipe*> result;
    for (const auto& recipe : GetBufferConversionRecipes()) result.emplace(recipe.logical_op, &recipe);
    return result;
  }();
  const auto found = index.find(name);
  return found == index.end() ? nullptr : found->second;
}

const BufferConversionRecipe* FindBufferConversionRecipe(const std::string& name) {
  static const auto index = [] {
    std::unordered_map<std::string, const BufferConversionRecipe*> result;
    for (const auto& recipe : GetBufferConversionRecipes()) result.emplace(recipe.buffer_op, &recipe);
    return result;
  }();
  const auto found = index.find(name);
  return found == index.end() ? nullptr : found->second;
}

namespace {
using K = BufferConversionKind;

bool Geometry(const ir::BufferTypePtr& a, const ir::BufferTypePtr& b) {
  return a->shape_ == b->shape_ && a->valid_shape_ == b->valid_shape_;
}

// A predicate byte encodes eight logical columns, least-significant bit first.
// Physical predicate rows are padded to 32 bytes, independently of valid bytes.
bool Predicate(const ir::BufferTypePtr& mask, const ir::BufferTypePtr& values) {
  const auto bytes = (values->shape_[1] + 7) / 8;
  return mask->dtype_ == DataType::UINT8 &&
         mask->shape_ == std::vector<int64_t>{values->shape_[0], ((bytes + 31) / 32) * 32} &&
         mask->valid_shape_ ==
             std::vector<int64_t>{values->valid_shape_[0], (values->valid_shape_[1] + 7) / 8};
}

const char* OperandError(const BufferConversionRecipe& recipe, const std::vector<ir::BufferTypePtr>& types) {
  for (size_t i = 0; i < types.size(); ++i) {
    if (static_cast<int>(i) != recipe.scalar_index && !IsAlignedDenseVecBuffer(types[i])) {
      return "requires aligned dense rank-2 Vec buffers with positive static valid extents";
    }
  }
  const auto& dst = types.back();
  if (recipe.kind == K::Cast) {
    const auto& src = types[0];
    const bool pair = (src->dtype_ == DataType::FP16 && dst->dtype_ == DataType::FP32) ||
                      (src->dtype_ == DataType::FP32 && dst->dtype_ == DataType::FP16);
    return pair && Geometry(src, dst)
               ? nullptr
               : "requires FP16/FP32 conversion with matching physical and valid shapes";
  }
  if (recipe.kind == K::Compare || recipe.kind == K::CompareScalar) {
    const auto& src = types[0];
    if (src->dtype_ != DataType::FP32 || !Predicate(dst, src)) {
      return "requires FP32 inputs and a matching packed UINT8 predicate destination";
    }
    if (recipe.kind == K::Compare && (types[1]->dtype_ != src->dtype_ || !Geometry(src, types[1]))) {
      return "requires matching comparison input descriptors";
    }
    return nullptr;
  }
  const auto& src = types[1];
  const auto& scratch = types[recipe.scratch_index];
  if (src->dtype_ != DataType::FP32 || dst->dtype_ != DataType::FP32 || !Geometry(src, dst) ||
      !Predicate(types[0], src)) {
    return "requires matching FP32 selection values and a packed UINT8 predicate";
  }
  const bool scalar = recipe.kind == K::SelectScalar;
  if (!scalar && (types[2]->dtype_ != DataType::FP32 || !Geometry(src, types[2]))) {
    return "requires matching selection value descriptors";
  }
  // Match the existing A2/A3 scratch recipes; no hidden allocation is allowed.
  const auto dtype = scalar ? DataType::FP32 : DataType::UINT32;
  const std::vector<int64_t> shape{1, scalar ? src->shape_[1] : 16};
  if (scratch->dtype_ != dtype || scratch->shape_ != shape || scratch->valid_shape_ != shape) {
    return "requires full-valid A2/A3 scratch: UINT32 [1,16] for sel or FP32 [1,C] for sels";
  }
  return nullptr;
}

bool Attributes(const BufferConversionRecipe& recipe,
                const std::vector<std::pair<std::string, std::any>>& kwargs, bool logical) {
  int mode = 0;
  int saturation = 0;
  int comparison = -1;
  for (const auto& [key, value] : kwargs) {
    if (recipe.kind == K::Cast && key == "target_type" && logical) continue;
    if (value.type() != typeid(int)) return false;
    const int integer = AnyCast<int>(value, "Buffer conversion attribute");
    if (recipe.kind == K::Cast && key == "mode") {
      mode = integer;
    } else if (recipe.kind == K::Cast && key == "saturation_mode") {
      saturation = integer;
    } else if ((recipe.kind == K::Compare || recipe.kind == K::CompareScalar) && key == "cmp_type") {
      comparison = integer;
    } else {
      return false;
    }
  }
  // This rollout uses IEEE half/single conversion. Other rounding/saturation
  // modes and scratch-backed integer narrowing keep their existing Tile path.
  if (recipe.kind == K::Cast) return (mode == 0 || mode == 1) && saturation == 0;
  if (recipe.kind == K::Compare || recipe.kind == K::CompareScalar) return comparison >= 0 && comparison < 6;
  return true;
}

bool Scalar(const ir::ExprPtr& arg, bool logical) {
  const auto scalar = ir::As<ir::ScalarType>(arg->GetType());
  if (!scalar) return false;
  const auto dtype = scalar->dtype_;
  return dtype == DataType::FP32 || (logical && (dtype == DataType::FP16 || dtype == DataType::BF16 ||
                                                 dtype.IsSignedInt() || dtype == DataType::INDEX));
}
}  // namespace

bool SupportsLogicalBufferConversion(const BufferConversionRecipe& recipe, const ir::CallPtr& call) {
  if (call->args_.size() != recipe.input_count || !Attributes(recipe, call->kwargs_, true)) return false;
  std::vector<ir::BufferTypePtr> types;
  for (size_t i = 0; i < call->args_.size(); ++i) {
    if (static_cast<int>(i) == recipe.scalar_index) {
      if (!Scalar(call->args_[i], true)) return false;
      types.push_back(nullptr);
    } else {
      types.push_back(StaticLogicalBufferDescriptor(call->args_[i]->GetType()));
    }
  }
  types.push_back(StaticLogicalBufferDescriptor(call->GetType()));
  return OperandError(recipe, types) == nullptr;
}

void ValidateBufferConversionOperands(const BufferConversionRecipe& recipe,
                                      const std::vector<ir::ExprPtr>& args,
                                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
  CHECK(args.size() == recipe.input_count + 1)
      << recipe.buffer_op << " requires explicit inputs and destination";
  CHECK(Attributes(recipe, kwargs, false)) << recipe.buffer_op << " has unsupported conversion attributes";
  std::vector<ir::BufferTypePtr> types;
  for (size_t i = 0; i < args.size(); ++i) {
    CHECK(args[i]) << recipe.buffer_op << " operands must not be null";
    if (static_cast<int>(i) == recipe.scalar_index) {
      CHECK(Scalar(args[i], false)) << recipe.buffer_op << " requires an FP32 scalar";
      types.push_back(nullptr);
    } else {
      types.push_back(ir::As<ir::BufferType>(args[i]->GetType()));
    }
  }
  const auto* error = OperandError(recipe, types);
  CHECK(!error) << recipe.buffer_op << " " << error;
}

std::vector<BufferConversionAliasPair> BufferConversionAliasPairs(const BufferConversionRecipe& recipe) {
  const size_t dst = recipe.input_count;
  std::vector<BufferConversionAliasPair> pairs;
  for (size_t i = 0; i < dst; ++i) {
    if (static_cast<int>(i) == recipe.scalar_index) continue;
    const bool value =
        (recipe.kind == K::Select && (i == 1 || i == 2)) || (recipe.kind == K::SelectScalar && i == 1);
    pairs.push_back({i, dst, value});
    if (recipe.scratch_index >= 0 && static_cast<int>(i) != recipe.scratch_index) {
      pairs.push_back({i, static_cast<size_t>(recipe.scratch_index), false});
    }
  }
  return pairs;
}

}  // namespace pypto::backend
