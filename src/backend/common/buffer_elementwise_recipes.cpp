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

#include "pypto/backend/common/buffer_elementwise_recipes.h"

#include <algorithm>
#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/transforms/structural_comparison.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace backend {

const std::vector<BufferElementwiseRecipe>& GetBufferElementwiseRecipes() {
  using Kind = BufferElementwiseOperandKind;
  static const std::vector<BufferElementwiseRecipe> recipes = {
      {"tile.add",
       "buffer.add",
       "pto.tadd",
       {{0}, {1}},
       BufferPrecisionKind::None,
       BufferElementwiseTypePolicy::MatchingVec},
      {"tile.mul",
       "buffer.mul",
       "pto.tmul",
       {{0}, {1}},
       BufferPrecisionKind::None,
       BufferElementwiseTypePolicy::MatchingVec},
      {"tile.sub", "buffer.sub", "pto.tsub", {{0}, {1}}},
      {"tile.div", "buffer.div", "pto.tdiv", {{0}, {1}}, BufferPrecisionKind::Div},
      {"tile.maximum", "buffer.maximum", "pto.tmax", {{0}, {1}}},
      {"tile.minimum", "buffer.minimum", "pto.tmin", {{0}, {1}}},
      {"tile.abs", "buffer.abs", "pto.tabs", {{0}}},
      {"tile.exp", "buffer.exp", "pto.texp", {{0}}},
      {"tile.sqrt", "buffer.sqrt", "pto.tsqrt", {{0}}},
      {"tile.neg", "buffer.neg", "pto.tneg", {{0}}},
      {"tile.relu", "buffer.relu", "pto.trelu", {{0}}},
      {"tile.log", "buffer.log", "pto.tlog", {{0}}, BufferPrecisionKind::Log},
      {"tile.recip",
       "buffer.recip",
       "pto.trecip",
       {{0}},
       BufferPrecisionKind::Recip,
       BufferElementwiseTypePolicy::DenseFP32,
       BufferDestinationAliasPolicy::Disjoint},
      {"tile.adds", "buffer.adds", "pto.tadds", {{0}, {1, Kind::ElementScalar}}},
      {"tile.subs", "buffer.subs", "pto.tsubs", {{0}, {1, Kind::ElementScalar}}},
      {"tile.muls", "buffer.muls", "pto.tmuls", {{0}, {1, Kind::ElementScalar}}},
      {"tile.divs", "buffer.divs", "pto.tdivs", {{0}, {1, Kind::ElementScalar}}},
      {"tile.maximums", "buffer.maximums", "pto.tmaxs", {{0}, {1, Kind::ElementScalar}}},
      {"tile.minimums", "buffer.minimums", "pto.tmins", {{0}, {1, Kind::ElementScalar}}},
      {"tile.lrelu", "buffer.lrelu", "pto.tlrelu", {{0}, {1, Kind::ElementScalar}}},
      {"tile.full", "buffer.full", "pto.texpands", {{1, Kind::ElementScalar}}},
  };
  return recipes;
}

const BufferElementwiseRecipe* FindLogicalBufferElementwiseRecipe(const std::string& name) {
  static const auto index = [] {
    std::unordered_map<std::string, const BufferElementwiseRecipe*> result;
    for (const auto& recipe : GetBufferElementwiseRecipes()) result.emplace(recipe.logical_op, &recipe);
    return result;
  }();
  auto found = index.find(name);
  return found == index.end() ? nullptr : found->second;
}

const BufferElementwiseRecipe* FindBufferElementwiseRecipe(const std::string& name) {
  static const auto index = [] {
    std::unordered_map<std::string, const BufferElementwiseRecipe*> result;
    for (const auto& recipe : GetBufferElementwiseRecipes()) result.emplace(recipe.buffer_op, &recipe);
    return result;
  }();
  auto found = index.find(name);
  return found == index.end() ? nullptr : found->second;
}

std::vector<std::string> GetBufferElementwiseRecipeNames() {
  std::vector<std::string> names;
  for (const auto& recipe : GetBufferElementwiseRecipes()) names.emplace_back(recipe.logical_op);
  std::sort(names.begin(), names.end());
  return names;
}

void ValidateBufferElementwiseOperands(const BufferElementwiseRecipe& recipe,
                                       const std::vector<ir::ExprPtr>& args) {
  CHECK(args.size() == recipe.inputs.size() + 1)
      << recipe.buffer_op << " requires " << recipe.inputs.size() + 1 << " explicit operands, got "
      << args.size();
  CHECK(args.back()) << recipe.buffer_op << " destination must not be null";
  const auto descriptor = ir::As<ir::BufferType>(args.back()->GetType());
  CHECK(descriptor) << recipe.buffer_op << " destination must have BufferType";
  for (size_t i = 0; i < args.size(); ++i) {
    CHECK(args[i]) << recipe.buffer_op << " argument " << i << " must not be null";
    if (i < recipe.inputs.size() && recipe.inputs[i].kind == BufferElementwiseOperandKind::ElementScalar) {
      const auto scalar = ir::As<ir::ScalarType>(args[i]->GetType());
      CHECK(scalar && scalar->dtype_ == descriptor->dtype_)
          << recipe.buffer_op << " argument " << i << " requires a scalar matching the destination dtype";
      continue;
    }
    auto buffer = ir::As<ir::BufferType>(args[i]->GetType());
    CHECK(buffer) << recipe.buffer_op << " argument " << i << " must have BufferType";
    CHECK(buffer->memory_space_ == ir::MemorySpace::Vec)
        << recipe.buffer_op << " argument " << i << " must be in Vec memory";
    CHECK(ir::structural_equal(descriptor, buffer))
        << recipe.buffer_op
        << " requires identical physical descriptors, including valid shape, layout and padding";
  }
  if (recipe.types == BufferElementwiseTypePolicy::MatchingVec) {
    // These are the arithmetic types validated on both current targets. BF16
    // transfer support does not establish a portable native add/mul contract.
    CHECK(descriptor->dtype_ == DataType::FP16 || descriptor->dtype_ == DataType::FP32 ||
          descriptor->dtype_ == DataType::INT32)
        << recipe.buffer_op << " currently requires FP16/FP32/INT32 arithmetic operands";
  }
  if (recipe.types == BufferElementwiseTypePolicy::DenseFP32) {
    CHECK(descriptor->dtype_ == DataType::FP32) << recipe.buffer_op << " currently requires FP32 operands";
    CHECK((descriptor->shape_.size() == 1 || descriptor->shape_.size() == 2) &&
          descriptor->blayout_ == ir::TileLayout::row_major &&
          descriptor->slayout_ == ir::TileLayout::none_box && descriptor->fractal_ == 512 &&
          descriptor->pad_ == ir::PadValue::null && descriptor->compact_ == ir::CompactMode::null)
        << recipe.buffer_op
        << " currently requires a static dense rank-1/rank-2 row-major layout; valid extents may be runtime";
  }
  if (recipe.destination_alias == BufferDestinationAliasPolicy::Disjoint) {
    for (size_t i = 0; i < recipe.inputs.size(); ++i) {
      CHECK(args[i] != args.back()) << recipe.buffer_op << " requires a distinct destination";
    }
  }
}

const char* BufferPrecisionAttributeName(BufferPrecisionKind precision) {
  switch (precision) {
    case BufferPrecisionKind::Div:
      return "div_precision";
    case BufferPrecisionKind::Log:
      return "log_precision";
    case BufferPrecisionKind::Recip:
      return "recip_precision";
    case BufferPrecisionKind::None:
      return nullptr;
  }
  INTERNAL_UNREACHABLE << "Internal error: invalid Buffer precision recipe";
}

}  // namespace backend
}  // namespace pypto
